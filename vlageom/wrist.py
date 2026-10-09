"""Wrist tile: sweep the wrist roll joint through its range and pool the sweeps of many houses into one
principal-component basis.

Sweep (GPU + EGL, `sweep_episode`): roll out one episode with the policy, take the first query after
query 0 at which the hand is open (executed gripper command below 127), and set the wrist roll joint
(arm joint index 6, the Franka FR3 joint 7) to `n_steps` evenly spaced angles over the joint's limits
in the simulator, plus the episode's own angle at that query. All other joints keep their values.
For each angle the robot-mounted cameras are moved to the new hand pose, both cameras are rendered,
and the policy is queried once with the same flow noise as in the rollout; the action expert's
per-token output is read at the final flow step. The policy reads two frames per camera, the
current one and the one 8 steps earlier: the earlier frames are the rollout's own, the current
frames and the joint state show the swept pose.

Pooled basis (CPU, `pooled_pca`): token 0 of action-expert layer 9 from every house's sweep, sorted by
joint angle, minus that house's own sweep mean; one SVD over all houses. The highlighted trace is one
house's sweep, centered the same way and projected into the same basis.

numpy only at import; simulator code is imported inside the sweep functions.
"""
from __future__ import annotations

import csv
import glob
import logging
import re
from pathlib import Path

import numpy as np

from vlageom import paths, util

log = logging.getLogger(__name__)

TILE = "wrist"
CONFIG = "FrankaPickDroidDataGenConfig"     # MolmoSpaces config the episodes are sampled from
ARM_JOINTS = 7
ROLL_JOINT = 6                              # arm joint index of the wrist roll joint (FR3 joint 7)
LAYERS = [0, 4, 9, 16, 24, 32]              # action-expert layers stored per sweep row
MAX_STEPS = 600                             # rollout length limit
N_STEPS = 38                                # evenly spaced angles over the joint range (+ the episode's own)
SCAN_EPS = 0.15                             # rad, joint perturbation of the roll-joint scan

# Pooled basis.
PANEL_LAYER = 9                             # action-expert layer of the pooled PCA
PANEL_TOKEN = 0                             # action token of the pooled PCA
MIN_PX = 20.0                               # minimum mean exo-camera pixel value of a sweep
MIN_ROWS = 10                               # minimum sweep rows
MIN_SPAN_DEG = 300.0                        # minimum swept angle range (deg)
# LAPACK's blocked SVD splits its work by the number of BLAS threads, so the last bits of the result
# depend on that number. The paper's panel was computed with 8 OpenBLAS threads; `pooled_pca` runs
# the SVD with the same number.
BLAS_THREADS = 8


# ---- shipped episode list --------------------------------------------------------------------- #
def read_episodes(path: str | Path | None = None) -> list[dict]:
    """Rows of lists/wrist_episodes.csv as dicts of ints: catalog_index, episode_index, house, seed."""
    path = Path(path) if path is not None else paths.lists_dir() / "wrist_episodes.csv"
    with open(path, newline="") as f:
        return [{k: int(v) for k, v in r.items()} for r in csv.DictReader(f)]


def sweep_dir(create: bool = False) -> Path:
    """`data_root()/wrist/sweeps`: one npz per episode."""
    return paths.tile_dir(TILE, "sweeps", create=create)


def sweep_path(catalog_index: int, root: str | Path | None = None) -> Path:
    """Sweep file of one episode: `<root>/ep<catalog_index>.npz` (root default `sweep_dir()`)."""
    return Path(root if root is not None else sweep_dir()) / f"ep{int(catalog_index)}.npz"


def default_sweep_glob() -> str:
    return str(sweep_dir() / "ep*.npz")


# ---- sweep (simulator) ------------------------------------------------------------------------ #
def identify_roll_joint(eng, snap, q0, R0, ee0, eps: float = SCAN_EPS):
    """The arm joint whose +eps change rotates the hand most per unit of EE translation (the roll
    joint turns the hand about its own approach axis and barely moves the EE point).

    Returns (joint index, unit rotation axis of that change in the robot base frame)."""
    from vlageom.sim.kinematics import set_arm_qpos
    from vlageom.sim.so3 import so3_axis, so3_geodesic

    best = (-1, -1.0, None)
    log.info("joint scan (rotation deg / EE drift mm for +%.2f rad):", eps)
    for j in range(ARM_JOINTS):
        eng._restore(snap)
        q = q0.copy()
        q[j] += eps
        set_arm_qpos(eng, q)
        m = eng.read_ee_full()
        if m is None:
            continue
        R = np.asarray(m[:3, :3], np.float64)
        ee = np.asarray(m[:3, 3], np.float64)
        rot = so3_geodesic(R0, R)
        drift = np.linalg.norm(ee - ee0)
        ratio = rot / (drift + 1e-6)
        axis = so3_axis(R @ R0.T)
        log.info("  joint %d: rot %5.1f deg  drift %5.1f mm  ratio %.0f", j, np.degrees(rot),
                 drift * 1000, ratio)
        if ratio > best[1]:
            best = (j, ratio, axis)
    return best[0], best[2]


def unwrap_roll(rows: list[dict], q0_jr: float) -> float:
    """Make each row's `roll_deg` continuous along the sweep and zero-anchored at the row nearest the
    episode's own angle; keep the value in [-180, 180] as `roll_deg_wrapped`. `rows` are in
    ascending joint-angle order. Returns max |min(|roll|, 360 - |roll|) - geodesic angle| (deg),
    a consistency check that is ~0."""
    rw = np.radians([r["roll_deg"] for r in rows])
    ru = np.unwrap(rw)
    ib = int(np.argmin(np.abs(np.array([r["jval"] for r in rows], np.float64) - float(q0_jr))))
    ru = ru - ru[ib] + rw[ib]
    for r, w, u in zip(rows, rw, ru):
        r["roll_deg_wrapped"] = float(np.degrees(w))
        r["roll_deg"] = float(np.degrees(u))
    fold = np.array([min(abs(r["roll_deg"]), 360.0 - abs(r["roll_deg"])) for r in rows])
    return float(np.max(np.abs(fold - np.array([r["geo_deg"] for r in rows]))))


def sweep_episode(eng, episode_index: int, house: int, seed: int, *, max_steps: int = MAX_STEPS,
                  n_steps: int = N_STEPS, layers=LAYERS) -> dict:
    """Roll out one episode and sweep its wrist roll joint (see the module docstring).

    `eng` is a loaded `vlageom.sim.BranchEngine`. Returns the arrays written by `save_sweep`."""
    from vlageom.sim.encode import encode_obs
    from vlageom.sim.kinematics import arm_joint_limits, pick_poses, set_arm_qpos
    from vlageom.sim.render import exo_frame, fresh_obs, wrist_cam_name, wrist_frame_real
    from vlageom.sim.so3 import signed_angle_about, so3_geodesic

    eng.sample(house, seed)
    base = eng.run_baseline(max_steps)
    snaps, qsteps = eng._baseline_snaps, base["query_steps"]
    poses = pick_poses(base, 1)                     # first open-hand query after query 0
    if not poses:
        raise RuntimeError(f"ep{episode_index} h{house}: no query with the hand open")
    c = poses[0]
    snap = snaps[int(qsteps[c])]
    flow_seed = eng.scene_seed + 1000 * int(snap["chunk"])   # the rollout's flow seed at this query
    q0 = np.asarray(base["qpos_arm"][c], np.float32)[:ARM_JOINTS]
    eng._restore(snap)
    set_arm_qpos(eng, q0)
    m0 = eng.read_ee_full()
    R0 = np.asarray(m0[:3, :3], np.float64)
    ee0 = np.asarray(m0[:3, 3], np.float64)
    log.info("ep%d h%d pose c%d (%d queries, success %s)", episode_index, house, c,
             base["n_queries"], base["success"])

    jr, roll_axis = identify_roll_joint(eng, snap, q0, R0, ee0)
    lo, hi = (float(x) for x in arm_joint_limits(eng)[jr])
    log.info("roll joint %d: limits %.4f..%.4f rad (%.1f deg), episode angle %.4f rad", jr, lo, hi,
             np.degrees(hi - lo), q0[jr])

    wname = wrist_cam_name(eng)
    vals = np.linspace(lo, hi, n_steps)
    vals = np.sort(np.concatenate([vals, [q0[jr]]]))          # include the episode's own angle
    rows = []
    for v in vals:
        eng._restore(snap)
        q = q0.copy()
        q[jr] = v
        set_arm_qpos(eng, q)
        m = eng.read_ee_full()
        if m is None:
            continue
        R = np.asarray(m[:3, :3], np.float64)
        ee = np.asarray(m[:3, 3], np.float64)
        obs = fresh_obs(eng)                       # cameras follow the swept hand, real renders
        act = encode_obs(eng, obs, layers, flow_seed, per_token=True)
        if act is None:
            continue
        rows.append(dict(jval=float(v), jrel_deg=float(np.degrees(v - q0[jr])),
                         roll_deg=float(np.degrees(signed_angle_about(R0, R, roll_axis))),
                         geo_deg=float(np.degrees(so3_geodesic(R0, R))),
                         drift_mm=float(np.linalg.norm(ee - ee0) * 1000),
                         frame=exo_frame(obs), wframe=wrist_frame_real(eng, wname),
                         act=np.stack([act[L] for L in layers])))
    if not rows:
        raise RuntimeError(f"ep{episode_index} h{house}: no sweep row produced a model query")
    err = unwrap_roll(rows, q0[jr])
    roll = [r["roll_deg"] for r in rows]
    log.info("swept %d angles; hand roll %.1f..%.1f deg (fold check %.3f deg); max EE drift %.3f mm",
             len(rows), min(roll), max(roll), err, max(r["drift_mm"] for r in rows))
    return dict(act=np.stack([r["act"] for r in rows]).astype(np.float32),
                frames=np.array([r["frame"] for r in rows], dtype=object),
                frames_wrist=np.array([r["wframe"] for r in rows], dtype=object),
                jval=np.array([r["jval"] for r in rows], np.float32),
                jrel_deg=np.array([r["jrel_deg"] for r in rows], np.float32),
                roll_deg=np.array([r["roll_deg"] for r in rows], np.float32),
                roll_deg_wrapped=np.array([r["roll_deg_wrapped"] for r in rows], np.float32),
                geo_deg=np.array([r["geo_deg"] for r in rows], np.float32),
                drift_mm=np.array([r["drift_mm"] for r in rows], np.float32),
                layers=np.array(list(layers)), roll_joint=jr, q0=q0,
                meta=f"ep{episode_index} h{house} c{c}",
                sweep_range_rad=np.array([lo, hi], np.float64))


def save_sweep(path: str | Path, sweep: dict) -> Path:
    """Write a `sweep_episode` result to `path` (npz, written to a temporary name, then renamed)."""
    import os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **sweep)
    os.replace(tmp, path)
    return path


# Keys of a sweep file, with what they hold (row = one swept angle, ascending):
SWEEP_KEYS = {
    "act": "(rows, len(layers), 16, 768) float32 action-expert output per action token, final flow step",
    "frames": "(rows,) object: exo-camera image (uint8 HxWx3) of the observation the policy read",
    "frames_wrist": "(rows,) object: wrist-camera render (uint8 HxWx3) at the swept pose",
    "jval": "(rows,) float32 roll joint angle (rad)",
    "jrel_deg": "(rows,) float32 roll joint angle minus the episode's own angle (deg)",
    "roll_deg": "(rows,) float32 hand rotation about the roll axis (deg), continuous, 0 at the episode's angle",
    "roll_deg_wrapped": "(rows,) float32 the same, in [-180, 180]",
    "geo_deg": "(rows,) float32 rotation angle between the swept and the episode's hand orientation (deg)",
    "drift_mm": "(rows,) float32 EE position change (mm)",
    "layers": "(len(layers),) action-expert layers of `act`",
    "roll_joint": "() arm joint index that was swept",
    "q0": "(7,) float32 arm joint angles at the sweep query",
    "meta": "() 'ep<episode_index> h<house> c<query index>'",
    "sweep_range_rad": "(2,) float64 joint limits swept (rad)",
}


# ---- pooled basis (CPU) ----------------------------------------------------------------------- #
def house_of(meta) -> int:
    """House index from a sweep file's `meta` string ('ep.. h<house> c..'), or -1."""
    m = re.search(r"h(\d+)", str(meta))
    return int(m.group(1)) if m else -1


def sweep_files(pattern: str) -> list[str]:
    """Sweep files matching a glob pattern, in sorted path order (the order the houses are stacked)."""
    return sorted(glob.glob(pattern))


def input_identity(files: list[str], trace: str, center: str = "house",
                   layer: int = PANEL_LAYER) -> str:
    """Record of the pooled inputs: path, size and modification time (ns) of every sweep file and of
    the trace file, then the centering mode and layer. Stored as the panel's `src`."""
    def ident(f):
        try:
            st = Path(f).stat()
            return f"{f}|{st.st_size}|{st.st_mtime_ns}"
        except OSError:
            return f"{f}|missing"
    return ("pooled:" + "|".join(ident(f) for f in list(files) + [trace])
            + f"|{center}|L{layer}")


def find_trace(files: list[str], house: int) -> str:
    """The file among `files` whose sweep is from `house`."""
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            if house_of(z["meta"]) == int(house):
                return f
    raise SystemExit(f"no sweep file for house {house} among {len(files)} files")


def _tok(z, layer, token):
    layers = [int(x) for x in z["layers"]]
    return np.asarray(z["act"], np.float64)[:, layers.index(layer), token, :]


def pooled_pca(files: list[str], trace: str, *, center: str = "house", layer: int = PANEL_LAYER,
               token: int = PANEL_TOKEN, min_px: float = MIN_PX, min_rows: int = MIN_ROWS,
               min_span: float = MIN_SPAN_DEG, blas_threads: int = BLAS_THREADS) -> dict:
    """Pool the sweeps of `files` into one PCA (see the module docstring).

    Each file contributes token `token` of layer `layer` per row, sorted by joint angle; with
    center="house" its own mean over the sweep is subtracted ("none" pools raw activations). A file
    is left out when it cannot be read, its mean exo-camera pixel value is below `min_px`, it has
    fewer than `min_rows` rows, or its angle range is below `min_span` degrees. The SVD and the
    projections run with `blas_threads` BLAS threads (see `BLAS_THREADS`).

    Returns the panel arrays:
      Z3 (n, 3) / Z2 (n, 2)  trace sweep in PC1-3 / PC1-2 (float64), rows sorted by angle
      roll (n,)              trace joint angle (deg)
      evr (3,)               explained-variance ratio of PC1-3
      nat3 (N, 3) float32    every pooled point in PC1-3
      nat_roll (N,) float32  joint angle of every pooled point (deg)
      house (N,)             house of every pooled point
      n_houses               number of pooled files
      files (n_houses,)      pooled file paths
      src                    `input_identity` of the inputs"""
    acts, ang, hid, kept, dropped = [], [], [], [], []
    for f in files:
        try:
            z = np.load(f, allow_pickle=True)
            A = _tok(z, layer, token)
            q = np.degrees(np.asarray(z["jval"], float))
            px = float(np.mean([np.asarray(x, np.uint8).mean() for x in z["frames"]]))
        except Exception as e:                                   # noqa: BLE001
            dropped.append((f, float("nan"), 0, 0.0, f"unreadable: {type(e).__name__}"))
            continue
        if px < min_px or len(q) < min_rows or np.ptp(q) < min_span:
            dropped.append((f, px, len(q), float(np.ptp(q)), "below a minimum"))
            continue
        o = np.argsort(q)
        A, q = A[o], q[o]
        if center == "house":
            A = A - A.mean(0)
        acts.append(A)
        ang.append(q)
        hid.append(np.full(len(q), house_of(z["meta"])))
        kept.append(f)
    for f, px, n, sp, why in dropped:
        log.warning("left out %s: %s (mean px %.1f, rows %d, span %.0f deg)", Path(f).name, why,
                    px, n, sp)
    if not acts:
        raise SystemExit("every sweep file was left out")
    X = np.vstack(acts)
    mu = X.mean(0)
    Z = X - mu
    zt = np.load(trace, allow_pickle=True)
    At = _tok(zt, layer, token)
    qt = np.degrees(np.asarray(zt["jval"], float))
    o = np.argsort(qt)
    At, qt = At[o], qt[o]
    if center == "house":
        At = At - At.mean(0)
    with util.blas_threads(blas_threads):
        _, S, Vh = np.linalg.svd(Z, full_matrices=False)
        nat3 = Z @ Vh.T[:, :3]
        Z3 = (At - mu) @ Vh.T[:, :3]
    evr = S**2 / (S**2).sum()
    log.info("%d houses x %d angles = %d points, center=%s, evr %s, trace %s", len(kept), len(qt),
             len(X), center, np.round(evr[:3], 4), Path(trace).name)
    return dict(Z2=Z3[:, :2], Z3=Z3, roll=qt, evr=evr[:3], nat3=nat3.astype(np.float32),
                nat_roll=np.concatenate(ang).astype(np.float32), house=np.concatenate(hid),
                n_houses=len(kept), files=np.array(kept),
                src=input_identity(files, trace, center, layer))


# ---- numbers derived from the panel ----------------------------------------------------------- #
def sweep_range_deg(roll) -> float:
    """Swept joint-angle range (deg) of the trace."""
    return float(np.ptp(np.asarray(roll, np.float64)))


# The trace counts as a closed loop when the distance between its two ends is at most this fraction
# of its length (`end_gap_ratio`), and as an open arc otherwise.
CLOSED_LOOP_MAX_GAP = 0.10


def end_gap_ratio(Z) -> float:
    """Distance between the first and last point of a path divided by the path length."""
    Z = np.asarray(Z, np.float64)
    return float(np.linalg.norm(Z[-1] - Z[0]) / np.linalg.norm(np.diff(Z, axis=0), axis=1).sum())


def trace_shape(Z) -> str:
    """'closed loop' if `end_gap_ratio(Z)` <= CLOSED_LOOP_MAX_GAP, else 'open arc'."""
    return "closed loop" if end_gap_ratio(Z) <= CLOSED_LOOP_MAX_GAP else "open arc"


def sweep_note(roll, Z) -> str:
    """The tile's note, e.g. '346° sweep · open arc': the range of the trace angles `roll` and the
    shape of the trace `Z` (its PC1-3 coordinates, see `trace_shape`)."""
    return f"{sweep_range_deg(roll):.0f}° sweep · {trace_shape(Z)}"
