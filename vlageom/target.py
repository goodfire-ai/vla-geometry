"""Target object position: the object-teleport store, the out-of-house probe score and the 3-D
read-out picture drawn in the target tile of Figure 1.

Object-teleport store
---------------------
`pipelines/target/10_teleport_harvest.py` writes one file per house and `20_pack_store.py` packs
them into memory-mappable arrays. Layout under a store root (default `data_root()/target/teleport`):

  houses/h{house}.npz      harvest output, one file per house (rows = placement x query)
  frames/h{house}.npz      96 px exo and wrist thumbnails of the first query of every placement
  prep/X_{variant}.npy     (N, 13, 2560) float16, VLM activations pooled over one token group
                           (`VARIANTS_LLM`) at the 13 layers in meta.json "layers_llm"
  prep/X_ae.npy            (N, 11, 768) float16, action expert at the final flow step, mean over
                           the 16 action tokens, at the 11 layers in meta.json "layers_ae"
  prep/X_ae_tok.npy        (N, 11, 16, 768) float16, the same per action token
  prep/X_chunk.npy         (N, 16, 8) float32, the predicted action chunk
  prep/labels.npz          per-row labels (house, config_id, chunk_q, obj_base, ee, rel_ee, dist, ...)
  prep/meta.json           {n, layers_llm, layers_ae, n_tok, houses, n_houses, gate_pass_frac}

Rows are ordered by house file name (lexical order of "h{house}.npz"), then as written by the
harvest. `chunk_q` is the policy query index (0 = the first query, before the robot moves);
`obj_base` is the realized object position in the robot base frame (x forward, y left, z up, m).

The distance tile reads the same store.

Target tile
-----------
`build_cloud` fits a 3-output ridge read-out of `obj_base` from the VLM's last-token activation at
layer 21 on all first-query rows except one held-out house, and returns the activation cloud in the
orthonormal 3-D basis spanned by the read-out weights. `target_panel` adds the object path of the
held-out house (pipelines/target/30_square_path_harvest.py) and writes the arrays the figure draws.
`probe_scores` computes the printed R^2: out-of-house ridge probes of x, y and z with the shipped
fold map, penalty chosen on held-out training houses.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from vlageom import paths
from vlageom.util import file_md5

# VLM token groups stored per row (same order as vlageom.sim.encode.VARIANTS).
VARIANTS_LLM = ["seq_all", "img_all", "img_exo", "img_wrist", "img_prev", "hdr_txt",
                "txt_tail", "txt_instr", "phrase_pickup", "phrase_pickup_end", "last_tok"]

# Approach-convergence label `g_conv` in labels.npz: a placement passes when the object-to-EE
# distance shrinks by at least GATE_CONV_M between the first and the last captured query, or is
# below GATE_NEAR_M at the last query.
GATE_CONV_M = 0.05
GATE_NEAR_M = 0.15

# Fig 1 target tile settings.
CELL = ("last_tok", 21)          # VLM final token, layer 21
HELDOUT_HOUSE = 44               # house of the object path; left out of the basis fit
CLOUD_MAX = 4000                 # cloud points drawn
VOXEL = 0.06                     # voxel size of the density thinning (normalized units)
SPLINE_SMOOTH = 0.02             # splprep smoothing of the object path
WAYPOINT_TOL = 1e-3              # m; consecutive waypoints closer than this are merged
PROBE_TARGETS = ("fwd", "lat", "up")
# BLAS threads of the paper's probe and panel runs. The ridge and RidgeCV fits split their matrix
# products across threads, so the thread count changes results at the level of floating-point
# rounding; with 24 threads the paper's arrays are reproduced bit for bit.
BLAS_THREADS = 24


def teleport_root() -> Path:
    """Default object-teleport store root: data_root()/target/teleport."""
    return paths.tile_dir("target", "teleport")


def square_path_root() -> Path:
    """Default root of the object-path harvest: data_root()/target/shape_square."""
    return paths.tile_dir("target", "shape_square")


def probe_json_default() -> Path:
    return paths.tile_dir("target", "probe", f"{CELL[0]}_L{CELL[1]}.json")


# ---------------------------------------------------------------------------------------------- #
# store access
# ---------------------------------------------------------------------------------------------- #
class TeleportStore:
    """Read access to a packed object-teleport store (`<root>/prep`)."""

    def __init__(self, root=None):
        self.root = Path(root) if root is not None else teleport_root()
        self.houses_dir = self.root / "houses"
        self.frames_dir = self.root / "frames"
        self.prep = self.root / "prep"

    def meta(self) -> dict:
        return json.loads((self.prep / "meta.json").read_text())

    def labels(self) -> dict:
        return dict(np.load(self.prep / "labels.npz", allow_pickle=False))

    def layer_index(self, stream: str, layer: int, meta: dict | None = None) -> int:
        """Index of `layer` in the store's layer list of `stream` (a VLM variant, 'ae' or 'ae_tok')."""
        meta = self.meta() if meta is None else meta
        key = "layers_ae" if stream in ("ae", "ae_tok") else "layers_llm"
        return int(list(meta[key]).index(int(layer)))

    def load_cell(self, variant: str, layer: int, token: int | None = None):
        """(keep_idx, X float32 (n_keep, d), labels filtered to the kept rows) for one stream and
        layer. Rows with a non-finite activation are dropped (img_prev is NaN on first queries).
        variant: one of VARIANTS_LLM, 'ae' or 'ae_tok' (token 0..15 required for 'ae_tok')."""
        li = self.layer_index(variant, layer)
        if variant == "ae_tok":
            if token is None:
                raise ValueError("ae_tok needs a token index")
            X = np.load(self.prep / "X_ae_tok.npy", mmap_mode="r")[:, li, int(token), :]
        elif variant == "ae":
            X = np.load(self.prep / "X_ae.npy", mmap_mode="r")[:, li, :]
        else:
            X = np.load(self.prep / f"X_{variant}.npy", mmap_mode="r")[:, li, :]
        X = np.ascontiguousarray(X, dtype=np.float32)
        keep = np.isfinite(X).all(1)
        lab = self.labels()
        return np.nonzero(keep)[0], X[keep], {k: v[keep] for k, v in lab.items()}

    def provenance(self, variant: str = CELL[0]) -> dict:
        """{store, labels_md5, X_<variant>: [bytes, mtime]} of the packed store."""
        xs = (self.prep / f"X_{variant}.npy").stat()
        return {"store": str(self.prep.resolve()), "labels_md5": file_md5(self.prep / "labels.npz"),
                f"X_{variant}": [xs.st_size, xs.st_mtime]}


# ---------------------------------------------------------------------------------------------- #
# packing (20_pack_store.py)
# ---------------------------------------------------------------------------------------------- #
LAB_VEC = ["obj_base", "ee", "rel_ee", "u", "cmd_xy"]
LAB_SCA = ["dist", "fwd", "bearing_deg", "settle_disp"]
LAB_INT = ["config_id", "chunk_q", "step", "lean_contacts", "in_env"]
PACK_STREAMS = VARIANTS_LLM + ["ae", "ae_tok", "chunk"]


def pack_one(job):
    """Copy one house file into the memmaps at rows row0..row0+n; return (row0, n, labels)."""
    f, row0, streams, mm_paths = job
    z = np.load(f, allow_pickle=False)
    n = len(z["dist"])
    for v in streams:
        src = {"ae": "ae", "ae_tok": "ae_tok", "chunk": "chunk"}.get(v, f"X_{v}")
        mm = np.lib.format.open_memmap(mm_paths[v], mode="r+")
        mm[row0:row0 + n] = z[src].astype(mm.dtype)
        del mm
    lab = {k: np.asarray(z[k]) for k in LAB_VEC + LAB_SCA + LAB_INT}
    lab["house"] = np.full(n, int(z["house"]), np.int32)
    obj = str(z["obj_name"]) if "obj_name" in z.files else ""
    lab["obj_type"] = np.full(n, obj.split("_")[0].lower() if obj else "", dtype="U24")
    lab["bench"] = np.full(n, str(z["bench"]), dtype="U8")
    g = np.zeros(n, np.float32)
    cid, cq, dist = lab["config_id"], lab["chunk_q"], lab["dist"]
    for c in np.unique(cid):
        m = cid == c
        d0 = dist[m][np.argmin(cq[m])]
        d3 = dist[m][np.argmax(cq[m])]
        ok = (d0 - d3 >= GATE_CONV_M) or (d3 < GATE_NEAR_M)
        g[m] = 1.0 if ok else 0.0
    lab["g_conv"] = g
    return row0, n, lab


# ---------------------------------------------------------------------------------------------- #
# out-of-house probe score (40_probe.py)
# ---------------------------------------------------------------------------------------------- #
def probe_scores(store: TeleportStore, folds, variant: str = CELL[0], layer: int = CELL[1]) -> dict:
    """Out-of-house ridge probes of the object position (base-frame x, y, z) from one activation
    cell, on the first-query rows.

    Penalty per target and fold from {1e0..1e6} on held-out training houses
    (`vlageom.probes.cv_ridge_select`); scores are pooled out-of-fold R^2 over all rows. `folds` is
    a `vlageom.probes.FoldMap`. Returns a dict with per-target r2, per-fold r2, MAE (m), chosen
    penalties, and `note_r2` = mean R^2 over x, y, z."""
    from vlageom.probes import cv_ridge_select, r2, r2_finite
    from vlageom.probes import mae as mae_

    _keep, X, lab = store.load_cell(variant, layer)
    mask = lab["chunk_q"] == 0
    Xm = X[mask]
    labm = {k: v[mask] for k, v in lab.items()}
    houses = labm["house"]
    ob = labm["obj_base"]
    targets = {"fwd": ob[:, 0], "lat": ob[:, 1], "up": ob[:, 2]}
    oof, chosen = cv_ridge_select(Xm, targets, houses, folds)
    splits = folds.splits(houses)
    out = {}
    for k in targets:
        y, p = np.asarray(targets[k], np.float64), oof[k]
        fr2 = []
        for _tr, te in splits:
            okt = np.isfinite(p[te])
            if okt.sum() >= 8:
                fr2.append(r2(y[te][okt], p[te][okt]))
        out[k] = {"r2": r2_finite(targets[k], p), "r2_folds": [float(v) for v in fr2],
                  "r2_fold_std": float(np.std(fr2)) if fr2 else float("nan"),
                  "mae_m": mae_(y, p), "alpha_per_fold": chosen[k],
                  "alpha_med": float(np.median(chosen[k]))}
    note = float(np.mean([out[k]["r2"] for k in PROBE_TARGETS]))
    return {"cell": f"{variant}@L{layer}", "rows": "first query (chunk_q == 0)",
            "n_rows": int(mask.sum()), "n_houses": int(len(np.unique(houses))),
            "fold_map": folds.name, "fold_map_sha1": folds.sha1(),
            "targets": out, "note_r2": note, "note_printed": f"{note:.2f}",
            "store": store.provenance(variant)}


def r2_note(r2) -> str:
    """The Figure 1 legend text for the probe score, e.g. "$R^2$ = 0.77"."""
    return f"$R^2$ = {float(r2):.2f}"


# ---------------------------------------------------------------------------------------------- #
# read-out basis and cloud (50_panel.py)
# ---------------------------------------------------------------------------------------------- #
def voxel_keep(Pn, vox, seed):
    """Indices that keep about one random point per voxel of size `vox` (density thinning)."""
    k = np.floor(np.asarray(Pn, float) / vox).astype(np.int64)
    k -= k.min(0)
    dims = k.max(0) + 1
    h = k[:, 0] + dims[0] * (k[:, 1] + dims[1] * k[:, 2])
    perm = np.random.default_rng(seed).permutation(len(Pn))
    _, first = np.unique(h[perm], return_index=True)
    return np.sort(perm[first])


def build_cloud(store: TeleportStore, variant: str = CELL[0], layer: int = CELL[1],
                heldout_house: int = HELDOUT_HOUSE, cloud_max: int = CLOUD_MAX,
                voxel: float = VOXEL) -> dict:
    """Rank-3 ridge read-out of obj_base and the first-query activation cloud in its basis.

    Fit: StandardScaler + RidgeCV (efficient leave-one-out over ALPHAS_GCV) from the activations
    of all first-query rows except `heldout_house` to obj_base (x, y, z). Basis B = the first 3
    columns of the QR factor of the weight matrix (orthonormal, so distances are not distorted);
    decode matrix A = W B (B^T B)^-1. Cloud = every first-query row projected on B, centred on the
    cloud mean and divided by the 95th-percentile radius. Colours: the per-axis empirical CDF of
    obj_base (x -> red, y -> green, z -> blue). The drawn cloud is thinned to about one point per
    voxel, then capped at `cloud_max` points (both with seed 0).

    Returns {"bg", "bg_rgb", "proj" (activations -> normalized 3-D), "color" (obj_base -> RGB),
    "decode" (A, intercept, centre, scale), "n_houses", "fitstats"}."""
    from numpy.linalg import inv, qr

    from vlageom.probes import ALPHAS_GCV, ridgecv_fit

    _keep, X, lab = store.load_cell(variant, layer)
    m = lab["chunk_q"] == 0
    X = X[m].astype(np.float64)
    ob = lab["obj_base"][m].astype(np.float64)
    houses = lab["house"][m]

    mfit = houses != heldout_house
    sc, rg = ridgecv_fit(X[mfit], ob[mfit], alphas=ALPHAS_GCV)
    W, c = rg.coef_, rg.intercept_
    B = qr(W.T)[0][:, :3]
    A = W @ B @ inv(B.T @ B)

    def proj_raw(Xin):
        return sc.transform(np.atleast_2d(np.asarray(Xin, float))) @ B

    s_cloud = proj_raw(X)
    center = s_cloud.mean(0)
    scale = float(np.percentile(np.linalg.norm(s_cloud - center, axis=1), 95)) or 1.0

    def norm(P):
        return (np.asarray(P, float) - center) / scale

    pred_te = rg.predict(sc.transform(X[~mfit])) if (~mfit).any() else None
    fitstats = {
        "alpha": float(rg.alpha_), "n_fit": int(mfit.sum()), "n_heldout": int((~mfit).sum()),
        "r2_insample": [round(float(v), 3) for v in
                        1 - ((rg.predict(sc.transform(X[mfit])) - ob[mfit]) ** 2).sum(0)
                        / ((ob[mfit] - ob[mfit].mean(0)) ** 2).sum(0)],
        "heldout_house_mae_cm": [round(float(v) * 100, 1)
                                 for v in np.abs(pred_te - ob[~mfit]).mean(0)]
        if pred_te is not None else None}

    srt = [np.sort(ob[:, k]) for k in range(3)]
    yk = np.linspace(0, 1, len(ob))

    def colorfn(E):
        E = np.atleast_2d(np.asarray(E, float))
        return np.clip(np.stack([np.interp(E[:, k], srt[k], yk) for k in range(3)], 1), 0, 1)

    bg, bg_ob = s_cloud, ob
    bi = voxel_keep(norm(bg), voxel, 0)
    bg, bg_ob = bg[bi], bg_ob[bi]
    if len(bg) > cloud_max:
        bi = np.random.default_rng(0).permutation(len(bg))[:cloud_max]
        bg, bg_ob = bg[bi], bg_ob[bi]

    return {"bg": norm(bg), "bg_rgb": colorfn(bg_ob),
            "proj": lambda acts: norm(proj_raw(acts)),
            "color": colorfn, "decode": (A, np.asarray(c, float), center, scale),
            "n_houses": int(len(np.unique(houses))), "fitstats": fitstats}


def smooth_curve(sub, closed, smooth, n=600):
    """Cubic smoothing spline through the points `sub` (k, 3), periodic when `closed`; returns
    (curve (n, 3), spline parameter of each input point, the n sample parameters)."""
    from scipy.interpolate import splev, splprep

    x, y, z = sub[:, 0], sub[:, 1], sub[:, 2]
    tck, u = splprep([x, y, z], s=smooth, per=1 if closed else 0, k=3)
    uu = np.linspace(0.0, 1.0, n)
    return np.array(splev(uu, tck)).T, np.asarray(u), uu


def reframe_U(A_dec, scale_dec):
    """Rotation that aligns the 3-D basis axes with the decoded x (forward), y (left), z (up)
    directions: QR of the decoded axis directions, each column signed to point along its axis."""
    dirs = np.linalg.inv(float(scale_dec) * A_dec.T)
    U, _ = np.linalg.qr(dirs.T)
    for k in range(3):
        if np.dot(U[:, k], dirs[k]) < 0:
            U[:, k] = -U[:, k]
    return U


def merge_close_waypoints(ob, closed, tol=WAYPOINT_TOL):
    """Indices of the waypoints kept after merging consecutive realized positions closer than
    `tol` (and, for a closed path, a last point that coincides with the first)."""
    keep = [0]
    for i in range(1, len(ob)):
        if np.linalg.norm(ob[i] - ob[keep[-1]]) > tol:
            keep.append(i)
    if closed and len(keep) > 1 and np.linalg.norm(ob[keep[-1]] - ob[keep[0]]) <= tol:
        keep.pop()
    return keep


def load_square_path(path_dir, variant: str = CELL[0], layer: int = CELL[1]):
    """(activations float64 (k, d), obj_base (k, 3), closed) of an object-path harvest directory
    (acts.npz + manifest.json from 30_square_path_harvest.py)."""
    path_dir = Path(path_dir)
    man = json.loads((path_dir / "manifest.json").read_text())
    closed = bool(man.get("closed", True))
    az = np.load(path_dir / "acts.npz")
    layers = [int(x) for x in az["layers_llm"]]
    acts = az[f"X_{variant}"][:, layers.index(int(layer)), :].astype(np.float64)
    ob = np.array([p["obj_base"] for p in man["points"]], float)
    return acts, ob, closed


def target_panel(store: TeleportStore, path_dir, *, note: dict | None = None,
                 log=print) -> dict:
    """The target tile arrays of Figure 1.

    bg (4000, 3) / rgb (4000, 3): cloud points and colours; sub (k, 3) / sub_rgb (k, 3): the
    held-out house's object-path waypoints and colours; curve (600, 3) / curve_rgb (600, 3): the
    smoothed path; origin (3,): the projection of the zero activation vector. All coordinates are
    in the normalized read-out basis, rotated by `reframe_U`. src: provenance string (store,
    labels md5, path directory). note_r2 / note_r2_xyz: the printed R^2 (from `probe_scores`), when
    `note` is given."""
    d = build_cloud(store)
    log(f"[target] basis fit: {json.dumps(d['fitstats'])} houses {d['n_houses']}")
    bg = np.asarray(d["bg"], float)
    bgc = np.clip(np.asarray(d["bg_rgb"], float), 0, 1)
    proj, colorfn = d["proj"], d["color"]
    A_dec, _c_dec, _center_dec, scale_dec = d["decode"]

    acts, ob, closed = load_square_path(path_dir)
    keep = merge_close_waypoints(ob, closed)
    ob, acts = ob[keep], acts[keep]
    log(f"[target] object path: {len(keep)} waypoints kept, closed={closed}")

    sub = np.asarray(proj(acts))
    rgb = np.clip(np.asarray(colorfn(ob)), 0, 1)
    curve, upar, uu = smooth_curve(sub, closed, SPLINE_SMOOTH)
    rgb_curve = np.clip(np.stack([np.interp(uu, upar, rgb[:, c]) for c in range(3)], 1), 0, 1)

    U = reframe_U(A_dec, scale_dec)
    src = dict(store.provenance(CELL[0]), trace=os.path.abspath(str(path_dir)))
    out = dict(bg=bg @ U, rgb=bgc, sub=sub @ U, sub_rgb=rgb, curve=curve @ U,
               curve_rgb=rgb_curve,
               origin=(np.asarray(proj(np.zeros((1, acts.shape[1])))) @ U)[0],
               src=np.array(json.dumps(src, sort_keys=True)))
    if note is not None:
        out["note_r2"] = np.float64(note["note_r2"])
        out["note_r2_xyz"] = np.array([note["targets"][k]["r2"] for k in PROBE_TARGETS], np.float64)
    return out
