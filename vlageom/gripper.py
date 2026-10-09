"""Gripper tile: gripper edits fitted near the grasp, and the panel arrays drawn from them.

An edit is a vector delta (768,) added to the output of action-expert block 32 on the last of the
ten flow steps. It is fitted at one model query of a policy rollout so that the decoded gripper
command flips (open -> close before the grasp, close -> open after it) while the decoded arm
actions keep the arm where it is (see `vlageom.steer.delta.optimize_delta`). The queries are
picked around the stable-grasp onset by `vlageom.steer.landmarks.find_landmark_windows`.

Per edit, one file `ep{episode_index}_h{house}_{landmark}_q{query}_aw300_sw3000.npz` (`EDIT_KEYS`)
holds the edit, the fit metrics, the layer-32 activation without the edit (`base_act`, the mean
over the 16 action tokens) and with it (`steered_act`), and the fit settings.

The panel (`panel_arrays`) is a centered SVD of all edits of the listed episodes; the activation
cloud and the edits are projected on the top three edit directions.
"""
from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

import numpy as np

from vlageom import paths

log = logging.getLogger(__name__)

# ---- settings of the paper's gripper catalog ---------------------------------------------------- #
CONFIG = "FrankaPickDroidDataGenConfig"   # MolmoSpaces task config of the rollouts
MAX_STEPS = 600                           # rollout length cap (env steps)
LAYER = 32                                # action-expert block whose output is edited
LANDMARKS = ("before_grasp", "after_grasp")
WINDOW = dict(w_pre=3, w_post=3, buffer=3, span=6)       # find_landmark_windows arguments
FIT = dict(iters=1000, lr=0.1, lam=1e-4, grip_weight=1.0)  # optimize_delta arguments
ARM_WEIGHT = 300.0                        # weight of the arm-hold term
SMOOTH_WEIGHT = 3000.0                    # weight of the arm smoothness term
ARM_MODE = "freeze"                       # arm target = the joint angles at the edited query

EDIT_RE = re.compile(r"ep(\d+)_h(\d+)_(before_grasp|after_grasp)_q(\d+)_aw300_sw3000\.npz")

# Keys of an edit file, in the order they are written.
EDIT_KEYS = (
    "layer", "delta", "flip", "grip_before", "grip_after", "arm_leakage", "arm_vs_natural",
    "arm_jerk", "nat_jerk", "arm_span", "norm", "grip_exec", "arm_leak_exec", "arm_jerk_exec",
    "arm_span_exec", "base_act", "steered_act", "target", "branch_query", "grasp_onset_query",
    "n_queries", "house", "seed", "episode_index", "arm_mode", "arm_weight", "smooth_weight",
    "grip_weight", "fit_iters", "fit_lr", "fit_lam", "arm_target", "qpos_arm",
)

# Keys of panels/gripper.npz.
PANEL_KEYS = ("px", "py", "pz", "ddx", "ddy", "ddz", "target", "aper", "fc", "fo", "scale", "evr")


# ---- episode list ------------------------------------------------------------------------------- #
def read_episodes(path=None) -> list[tuple[int, int, int, int]]:
    """[(episode_index, house, seed, new_sampler)] from `lists/gripper_episodes.csv` (or `path`),
    in file order (sorted by house, then episode_index). See `sample_episode` for new_sampler."""
    path = Path(path) if path is not None else paths.lists_dir() / "gripper_episodes.csv"
    with open(path, newline="") as f:
        return [(int(r["episode_index"]), int(r["house"]), int(r["seed"]), int(r["new_sampler"]))
                for r in csv.DictReader(f)]


def edit_name(episode_index: int, house: int, landmark: str, query: int) -> str:
    return (f"ep{episode_index}_h{house}_{landmark}_q{query}"
            f"_aw{ARM_WEIGHT:g}_sw{SMOOTH_WEIGHT:g}.npz")


# ---- harvest ------------------------------------------------------------------------------------ #
def sample_episode(eng, house: int, seed: int, new_sampler: int) -> None:
    """Sample one listed episode; its scene is set by (house, seed, new_sampler).

    `eng.sample` seeds python `random`, numpy and torch with `seed`, then the MolmoSpaces task
    sampler loads the house and picks the object and the robot placement from those RNGs.
      new_sampler=1  the MolmoSpaces environment and task sampler are created for this episode
                     (built again from the config if the process has sampled before).
                     A new task sampler draws a number from the seeded numpy RNG and reseeds all
                     three RNGs with it before it samples (`SimEnv.ensure_task_sampler`).
      new_sampler=0  the episode is sampled by an existing task sampler (created first if the
                     process has none), from the RNG state set by `seed`; the house is loaded from
                     file."""
    if new_sampler:
        if eng.sim_env.task_sampler is not None:
            eng.new_env()
        eng.sample(house, seed)
    else:
        eng.sim_env.ensure_task_sampler()
        eng.sample(house, seed, reload_house=True)


def fit_episode(eng, base: dict, episode_index: int, house: int, seed: int, edits_dir,
                *, capture_path=None, overwrite: bool = False, fit: bool = True) -> int:
    """Fit and write the edits of one episode; returns the number of files written.

    `base` is a `BranchEngine.run_baseline` result (or a `load_capture` result, which has the same
    per-query signals). Episodes without a stable grasp write nothing. With `capture_path`, the
    per-query model inputs of a grasping episode are also saved there (`BranchEngine.save_capture`)
    so the edits can be fitted again without a rollout. fit=False saves only those inputs."""
    from vlageom.steer.landmarks import find_landmark_windows, find_landmarks

    lms = find_landmarks(base)
    windows = find_landmark_windows(base, **WINDOW)
    present = [n for n in LANDMARKS if n in lms]
    log.info("ep%d house%d seed%d: %d queries, onset=%s, windows=%s",
             episode_index, house, seed, base["n_queries"], lms.get("_grasp_onset_query"),
             {n: (windows.get(n) or [lms[n]["chunk"]]) for n in present})
    if not present:
        log.warning("ep%d house%d seed%d: no stable grasp; no edits", episode_index, house, seed)
        return 0
    if capture_path is not None:
        eng.save_capture(capture_path, base)
    if not fit:
        return 0

    edits_dir = Path(edits_dir)
    edits_dir.mkdir(parents=True, exist_ok=True)
    qpos_arm_by_q = base.get("qpos_arm", [])
    n_written = 0
    for name in present:
        spec = lms[name]
        target = float(spec["target"])
        for q in windows.get(name, [spec["chunk"]]):
            outpath = edits_dir / edit_name(episode_index, house, name, q)
            if outpath.exists() and not overwrite:
                continue
            qarm = qpos_arm_by_q[q] if 0 <= q < len(qpos_arm_by_q) else None
            arm_tgt = eng.freeze_arm_target(qarm)
            arm_mode = ARM_MODE
            if arm_tgt is None:
                log.warning("ep%d %s q%d: no arm target (joint angles or normalizer missing); "
                            "the arm term holds the unedited arm actions", episode_index, name, q)
                arm_mode = "natural(fallback)"
            fits = eng.fit_deltas(q, target, [LAYER], arm_target=arm_tgt, arm_weight=ARM_WEIGHT,
                                  smooth_weight=SMOOTH_WEIGHT, **FIT)
            fit = fits[LAYER]["fit"]
            flip = (fit.grip_after > 0) if target > 0 else (fit.grip_after < 0)
            np.savez(
                outpath,
                layer=np.int64(LAYER),
                delta=fit.delta.astype(np.float32),
                flip=np.bool_(flip),
                grip_before=np.float32(fit.grip_before), grip_after=np.float32(fit.grip_after),
                arm_leakage=np.float32(fit.arm_leakage),
                arm_vs_natural=np.float32(fit.arm_vs_natural),
                arm_jerk=np.float32(fit.arm_jerk), nat_jerk=np.float32(fit.nat_jerk),
                arm_span=np.float32(fit.arm_span), norm=np.float32(fit.norm),
                grip_exec=np.float32(fit.grip_exec), arm_leak_exec=np.float32(fit.arm_leak_exec),
                arm_jerk_exec=np.float32(fit.arm_jerk_exec),
                arm_span_exec=np.float32(fit.arm_span_exec),
                base_act=fits[LAYER]["base_act"].astype(np.float32),
                steered_act=fits[LAYER]["steered_act"].astype(np.float32),
                target=np.float32(target), branch_query=np.int64(q),
                grasp_onset_query=np.int64(lms.get("_grasp_onset_query", -1)),
                n_queries=np.int64(base["n_queries"]),
                house=np.int64(house), seed=np.int64(seed), episode_index=np.int64(episode_index),
                arm_mode=np.str_(arm_mode),
                arm_weight=np.float32(ARM_WEIGHT), smooth_weight=np.float32(SMOOTH_WEIGHT),
                grip_weight=np.float32(FIT["grip_weight"]),
                fit_iters=np.int64(FIT["iters"]), fit_lr=np.float32(FIT["lr"]),
                fit_lam=np.float32(FIT["lam"]),
                arm_target=(np.asarray(arm_tgt, np.float32) if arm_tgt is not None
                            else np.zeros(0, np.float32)),
                qpos_arm=(np.asarray(qarm, np.float32) if qarm is not None
                          else np.zeros(0, np.float32)),
            )
            n_written += 1
    log.info("ep%d house%d seed%d: wrote %d edit(s)", episode_index, house, seed, n_written)
    return n_written


# ---- panel -------------------------------------------------------------------------------------- #
def edit_files(edits_dir, episodes) -> list[Path]:
    """Edit files in `edits_dir` that belong to `episodes` [(episode_index, house, ...)], sorted by
    file name."""
    keep = {(int(e[0]), int(e[1])) for e in episodes}
    out = []
    for f in sorted(Path(edits_dir).glob("*.npz")):
        m = EDIT_RE.fullmatch(f.name)
        if m and (int(m.group(1)), int(m.group(2))) in keep:
            out.append(f)
    return out


def load_edits(files) -> dict:
    """Stack the edit files: H = base_act (n, 768), D = delta (n, 768), aper = grip_before (n,),
    target (n,) as float32, plus per-row episode_index and house (int64)."""
    H, D, aper, tgt, ep, house = [], [], [], [], [], []
    for f in files:
        m = EDIT_RE.fullmatch(Path(f).name)
        d = np.load(f, allow_pickle=True)
        H.append(np.asarray(d["base_act"], np.float32))
        D.append(np.asarray(d["delta"], np.float32))
        aper.append(float(d["grip_before"]))
        tgt.append(float(d["target"]))
        ep.append(int(m.group(1)))
        house.append(int(m.group(2)))
    return dict(H=np.stack(H), D=np.stack(D), aper=np.asarray(aper, np.float32),
                target=np.asarray(tgt, np.float32), episode_index=np.asarray(ep, np.int64),
                house=np.asarray(house, np.int64))


def pc_signs(px, py, pz, ddx, ddy, ddz, aper, target) -> np.ndarray:
    """Orientation (+1 / -1) of edit PC1-3. SVD signs are arbitrary; these rules set them:
      PC1: closing edits (target +1) have a larger mean PC1 component than opening edits;
      PC2: the cloud's PC2 coordinate correlates positively with the decoded gripper command;
      PC3: the mean edit has a positive PC3 component."""
    px, py, ddx, ddz = (np.asarray(a, np.float64) for a in (px, py, ddx, ddz))
    close, open_ = target > 0, target < 0
    s1 = ddx[close].mean() - ddx[open_].mean()
    s2 = np.corrcoef(py, np.asarray(aper, np.float64))[0, 1]
    s3 = ddz.mean()
    return np.array([1.0 if s >= 0 else -1.0 for s in (s1, s2, s3)], np.float32)


def panel_arrays(H, D, aper, target) -> dict:
    """The gripper panel: centered SVD of the edits D (n, 768); the activations H (n, 768),
    centered on their mean, and the edits (uncentered) projected on edit PC1-3.

    Returns px, py, pz (cloud), ddx, ddy, ddz (edits), target, aper, evr (explained-variance ratio
    of PC1-3, float64), and a 2-D binned field per edit direction over PC1-PC2 (fc for closing
    edits, fo for opening edits; rows [x, y, dx, dy] of the cells with at least 3 edits on a 16 x 16
    grid over the cloud) with the arrow scale `scale` (0.9 cell width / median arrow length)."""
    _, S, Vh = np.linalg.svd(D - D.mean(0), full_matrices=False)
    V = Vh.T.astype(np.float32)
    evr = S**2 / (S**2).sum()

    c = H.mean(0)                          # cloud centered on the activation mean
    proj = [(H - c) @ V[:, k] for k in range(3)] + [D @ V[:, k] for k in range(3)]
    signs = pc_signs(*proj, aper, target)
    V[:, :3] *= signs[None, :]

    e0, e1, e2 = V[:, 0], V[:, 1], V[:, 2]
    px, py, pz = (H - c) @ e0, (H - c) @ e1, (H - c) @ e2
    dx, dy, dz = D @ e0, D @ e1, D @ e2    # edits projected uncentered

    def binned(sel, nbins=16, min_count=3):
        xe = np.linspace(px.min(), px.max(), nbins + 1)   # one grid over the whole cloud
        ye = np.linspace(py.min(), py.max(), nbins + 1)
        ix = np.clip(np.digitize(px, xe) - 1, 0, nbins - 1)
        iy = np.clip(np.digitize(py, ye) - 1, 0, nbins - 1)
        rows = []
        for i in range(nbins):
            for j in range(nbins):
                m = sel & (ix == i) & (iy == j)
                if int(m.sum()) >= min_count:
                    rows.append([px[m].mean(), py[m].mean(), dx[m].mean(), dy[m].mean()])
        return np.asarray(rows), float(xe[1] - xe[0])

    fc, cellw = binned(target > 0)
    fo, _ = binned(target < 0)
    allmag = np.concatenate([np.hypot(fc[:, 2], fc[:, 3]), np.hypot(fo[:, 2], fo[:, 3])])
    scale = (0.9 * cellw) / (np.median(allmag) + 1e-9)

    return dict(px=px, py=py, pz=pz, ddx=dx, ddy=dy, ddz=dz, target=target, aper=aper,
                fc=fc, fo=fo, scale=np.float64(scale), evr=evr[:3].astype(np.float64))
