"""End-effector (EE) position and target object position (Figure 2).

Both variables are read from the same rows of one store: in every house the arm is moved to 10
poses by inverse kinematics and the pickup object to 5 placements, and every (placement, pose)
cell is captured twice by the policy, right after the move (query q0) and after the policy has
executed one action chunk of 8 steps (query q1).

Crossed store (`pipelines/position/10_crossed_harvest.py`, `20_pack_store.py`)
-------------------------------------------------------------------------------
Layout under a store root (default `data_root()/position/crossed`):

  houses/h{house}.npz     harvest output, one file per house (rows = cell x query, at most 100)
  prep/X_{variant}.npy    (N, 36, 2560) float16, VLM activations pooled over one token group
                          (`VARIANTS_LLM`) at all 36 layers; stage 20 packs the token groups it is
                          given (by default the two that Figure 2 reads, hdr_txt and txt_instr)
  prep/X_ae.npy, X_ae_tok.npy, X_chunk.npy
                          action expert (token mean / per token, 36 layers) and the predicted
                          action chunk, when packed
  prep/labels.npz         per-row labels (house, config_id, pose_id, chunk_q, ee, obj_base, ...)
  prep/meta.json          {n, layers_llm, layers_ae, n_tok, variants_llm, houses, n_houses,
                          rows_per_grain, ee_std_cm, obj_std_cm}
  prep/excluded_houses.txt
                          houses left out of the packed store (see `exo_unchanging`)

Rows are ordered by house file name (lexical order of "h{house}.npz"), then as written by the
harvest. `chunk_q` is the query (0 = right after the move, 1 = after one executed chunk), `ee` the
EE position and `obj_base` the object position, both in the robot base frame (x forward, y left,
z up, metres), read from the simulator at the query.

Figure 2 top row
----------------
  a-c  `probe_cell`: out-of-house ridge read-out of (x, y, z) per variable, its rank 1/2/3 R^2,
       the singular-value fractions of its predictions, per-axis and 3-D error (q1 rows)
  d    `procrustes_layer`: world displacement -> activation displacement maps inside the rank-3
       read-out subspace, per layer (q0 rows)
  e    `curvature`: turning angle of straight world paths inside the subspace vs the distance
       between hand and object (paths from the line harvests, subspace fit on q0 rows)

Figure 2 trace grids
--------------------
`grid_chart` fits the rank-3 read-out subspace without the shape house (house 44) and
`grid_panel` projects the five hand paths (EE grid) or object paths (target grid) of that house
into it; `figures/fig2.py` draws them.

Everything here imports numpy only at module level; scikit-learn, the simulator and torch are
imported inside the functions that need them.
"""
from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path

import numpy as np

from vlageom import paths
from vlageom.util import file_md5, save_npz

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------------------------- #
# constants
# ---------------------------------------------------------------------------------------------- #
# Gripper width line of Figure 2a (cm): the largest finger opening of the Robotiq 2F-85, MolmoSpaces
# franka_droid inter_finger_dist_range (0, 0.087) m.
GRIPPER_CM = 8.7

VARIANTS_LLM = ["seq_all", "img_all", "img_exo", "img_wrist", "img_prev", "hdr_txt",
                "txt_tail", "txt_instr", "phrase_pickup", "phrase_pickup_end", "last_tok"]
OPTIONAL_VARIANTS = {"img_prev", "phrase_pickup", "phrase_pickup_end"}
N_LAYERS = 36                                   # VLM and action-expert blocks
TARGETS = {"ee": "ee", "obj": "obj_base"}       # variable -> label key
VARIABLE_NAMES = {"ee": "EE position", "obj": "target object position"}
# The read-out cell of each variable: (VLM token group, layer).
CELLS = {"ee": ("hdr_txt", 20), "obj": ("txt_instr", 21)}
PACK_DEFAULT_STREAMS = sorted({s for s, _ in CELLS.values()})
GRAINS = {"q0": 0, "q1": 1}
FIG_GRAIN = "q1"               # rows of panels a-c
PROC_GRAIN = "q0"              # rows of panel d
CHART_GRAIN = "q0"             # rows the curvature and trace-grid subspaces are fit on
SHAPE_HOUSE = 44               # house of the trace grids; left out of their subspace fits
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5]   # RidgeCV penalty grid of every fit here
ALPHA_LABEL = 1e-2             # ridge penalty of the label-to-label baseline
N_SPLITS = 5
FOLD_MAP = "position"          # lists/folds/position.csv
# BLAS threads of the paper's probe, Procrustes and subspace fits. Matrix products split across
# threads, so the thread count changes results at the level of floating-point rounding; with 8
# threads the paper's arrays are reproduced bit for bit.
BLAS_THREADS = 8

# Curvature (panel e)
SEG_M = 0.07                   # segment length of the local turning angle (m)
STRIDE_MIN = 2
MIN_PTS = 12                   # paths with fewer waypoints are not used
N_BINS = 6                     # quantile bins of the hand-object distance
N_RAND = 8                     # random 3-D projections of the reference curve
RAND_SEED = {"ee": 52, "obj": 37}
OBJECT_LINE_KINDS = ("line_fwd", "line_lat", "line_ee")

# Trace grids
EE_GRID = [("fig8", "figure-8"), ("line_x", "x line"), ("line_y", "y line"), ("circle", "circle"),
           ("square", "square")]
OBJ_GRID = [("infinity", "figure-8"), ("line_fwd", "x line"), ("line_lat", "y line"),
            ("ellipse", "circle"), ("square", "square")]
WAYPOINT_TOL = 1e-3            # m; consecutive waypoints closer than this are merged
EE_COLOR_DECIMALS = 3          # the EE colours are rounded to 3 decimals


# ---------------------------------------------------------------------------------------------- #
# locations and lists
# ---------------------------------------------------------------------------------------------- #
def crossed_root() -> Path:
    """Default crossed-store root: data_root()/position/crossed."""
    return paths.tile_dir("position", "crossed")


def hand_lines_root() -> Path:
    """Default root of the straight hand lines: data_root()/position/hand_lines."""
    return paths.tile_dir("position", "hand_lines")


def object_lines_root() -> Path:
    """Default root of the straight object lines: data_root()/position/object_lines."""
    return paths.tile_dir("position", "object_lines")


def shapes_root() -> Path:
    """Default root of the trace-grid paths: data_root()/position/shapes."""
    return paths.tile_dir("position", "shapes")


def probe_dir() -> Path:
    return paths.tile_dir("position", "probe")


def procrustes_dir() -> Path:
    return paths.tile_dir("position", "procrustes")


def curvature_json_default() -> Path:
    return paths.tile_dir("position", "curvature", f"curvature_{CHART_GRAIN}.json")


def read_house_list(path) -> list:
    """[(house, bench, bench_idx), ...] from a CSV with columns house,bench,bench_idx."""
    with open(path, newline="") as f:
        return [(int(r["house"]), r["bench"], int(r["bench_idx"])) for r in csv.DictReader(f)]


def crossed_house_list_path() -> Path:
    return paths.lists_dir() / "position" / "crossed_houses.csv"


def line_house_list_path() -> Path:
    return paths.lists_dir() / "position" / "line_houses.csv"


def ee_color_reference_path() -> Path:
    return paths.lists_dir() / "position" / "ee_color_reference.csv"


# ---------------------------------------------------------------------------------------------- #
# packed store
# ---------------------------------------------------------------------------------------------- #
class PositionStore:
    """Read access to a packed crossed store (`<root>/prep`)."""

    def __init__(self, root=None):
        """`root`: the store root (holding prep/) or the prep directory itself (default
        `crossed_root()`)."""
        self.root = Path(root) if root is not None else crossed_root()
        self.prep = self.root if (self.root / "meta.json").exists() else self.root / "prep"

    def meta(self) -> dict:
        return json.loads((self.prep / "meta.json").read_text())

    def labels(self) -> dict:
        return dict(np.load(self.prep / "labels.npz", allow_pickle=False))

    def layer_index(self, stream: str, layer: int, meta: dict | None = None) -> int:
        meta = self.meta() if meta is None else meta
        key = "layers_ae" if stream in ("ae", "ae_tok") else "layers_llm"
        return int(list(meta[key]).index(int(layer)))

    def stream_path(self, stream: str) -> Path:
        p = self.prep / f"X_{stream}.npy"
        if not p.exists():
            raise SystemExit(f"{p} does not exist: pack this token group with "
                             f"pipelines/position/20_pack_store.py --streams {stream}")
        return p

    def layer_array(self, stream: str, layer: int, meta: dict | None = None) -> np.ndarray:
        """(N, d) float32 activations of every row at one layer (no rows dropped)."""
        li = self.layer_index(stream, layer, meta)
        X = np.load(self.stream_path(stream), mmap_mode="r")
        return np.asarray(X[:, li, :], dtype=np.float32)

    def load_cell(self, stream: str, layer: int, token: int | None = None):
        """(keep_idx, X float32 (n_keep, d), labels of the kept rows) for one stream and layer;
        rows with a non-finite activation are dropped (img_prev is undefined at q0)."""
        li = self.layer_index(stream, layer)
        if stream == "ae_tok":
            if token is None:
                raise ValueError("ae_tok needs a token index")
            X = np.load(self.stream_path("ae_tok"), mmap_mode="r")[:, li, int(token), :]
        else:
            X = np.load(self.stream_path(stream), mmap_mode="r")[:, li, :]
        X = np.ascontiguousarray(X, dtype=np.float32)
        keep = np.isfinite(X).all(1)
        lab = self.labels()
        return np.nonzero(keep)[0], X[keep], {k: v[keep] for k, v in lab.items()}

    def provenance(self) -> dict:
        return {"store": str(self.prep.resolve()), "labels_md5": file_md5(self.prep / "labels.npz")}


def row_masks(lab) -> dict:
    """{"q0", "q1", "all"} boolean row masks."""
    q = lab["chunk_q"]
    return {"q0": q == 0, "q1": q == 1, "all": np.ones(len(q), bool)}


LAB_VEC = ["obj_base", "ee", "rel_ee", "u", "cmd_xy", "ee_target", "ee_tele", "qpos_set", "qpos_q",
           "obj_world"]
LAB_SCA = ["dist", "fwd", "bearing_deg", "settle_disp", "ik_err", "exo_px"]
LAB_INT = ["config_id", "pose_id", "chunk_q", "step", "lean_contacts", "in_env"]
LAB_MAT = ["ee44"]
ALL_STREAMS = VARIANTS_LLM + ["ae", "ae_tok", "chunk"]


def exo_unchanging(z) -> bool:
    """True when a house file's exo-camera frame does not change across its rows: the mean pixel
    value varies by less than 1e-6, every 64 px thumbnail is identical, or the median pixel value
    is below 20. In such a house the exo camera sees neither the moved arm nor the moved object,
    and the house is left out of the packed store."""
    px, th = z["exo_px"], z["exo_thumb"]
    return bool(len(px) < 2 or float(px.std()) < 1e-6 or bool(np.all(th == th[0]))
                or float(np.median(px)) < 20)


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
    lab = {k: np.asarray(z[k]) for k in LAB_VEC + LAB_SCA + LAB_INT + LAB_MAT}
    lab["house"] = np.full(n, int(z["house"]), np.int32)
    obj = str(z["obj_name"]) if "obj_name" in z.files else ""
    lab["obj_type"] = np.full(n, obj.split("_")[0].lower() if obj else "", dtype="U24")
    lab["bench"] = np.full(n, str(z["bench"]), dtype="U8")
    return row0, n, lab


# ---------------------------------------------------------------------------------------------- #
# probes (panels a-c)
# ---------------------------------------------------------------------------------------------- #
def _vw(Y, P):
    """Variance-weighted R^2 over all output columns."""
    Y, P = np.asarray(Y, np.float64), np.asarray(P, np.float64)
    return float(1.0 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def _axis(Y, P):
    """Per-column R^2."""
    Y, P = np.asarray(Y, np.float64), np.asarray(P, np.float64)
    return [float(1.0 - ((Y[:, i] - P[:, i]) ** 2).sum() / ((Y[:, i] - Y[:, i].mean()) ** 2).sum())
            for i in range(Y.shape[1])]


def house_center(A, houses):
    """A minus its per-house mean (float64 copy)."""
    out = np.asarray(A, np.float64).copy()
    for h in np.unique(houses):
        m = houses == h
        out[m] -= out[m].mean(0)
    return out


def ridge_label(Xl, Yl, folds, alpha=ALPHA_LABEL):
    """Out-of-house closed-form ridge (with intercept) from one label to another; per-axis R^2."""
    P = np.zeros_like(Yl, dtype=np.float64)
    for tr, te in folds:
        Xa = np.c_[Xl[tr], np.ones(len(tr))]
        W = np.linalg.solve(Xa.T @ Xa + alpha * np.eye(Xa.shape[1]), Xa.T @ Yl[tr])
        P[te] = np.c_[Xl[te], np.ones(len(te))] @ W
    return _axis(Yl, P)


def baselines(lab_g, folds) -> dict:
    """Label-space references on the rows of one grain: the R^2 of predicting each row by its
    house mean, the fraction of variance within houses, the label standard deviation (cm), and
    the out-of-house R^2 of predicting one variable's position from the other's."""
    houses = lab_g["house"]
    out = {}
    for name, key in TARGETS.items():
        Y = lab_g[key].astype(np.float64)
        mu = np.zeros_like(Y)
        for h in np.unique(houses):
            m = houses == h
            mu[m] = Y[m].mean(0)
        out.setdefault("house_mean_oracle_r2", {})[name] = {"vw": _vw(Y, mu), "axis": _axis(Y, mu)}
        out.setdefault("within_house_frac", {})[name] = float(((Y - mu) ** 2).sum()
                                                              / ((Y - Y.mean(0)) ** 2).sum())
        out.setdefault("std_cm", {})[name] = (Y.std(0) * 100).round(3).tolist()
    E, Ob = lab_g["ee"].astype(np.float64), lab_g["obj_base"].astype(np.float64)
    out["label_echo_r2"] = {"obj_from_ee": ridge_label(E, Ob, folds),
                            "ee_from_obj": ridge_label(Ob, E, folds)}
    return out


def probe_cell(X, Ys, houses, folds) -> dict:
    """Out-of-house read-outs of every target in `Ys` from the activations X of one grain.

    Per fold: StandardScaler on the training rows (shared by the targets), RidgeCV (efficient
    leave-one-out over `ALPHAS`) from the scaled activations to the 3 coordinates; reduced-rank
    predictions of rank 1 and 2 from the SVD of the centred training predictions. Pooled over the
    out-of-fold rows: R^2 (all three coordinates, variance weighted), per-axis R^2, rank 1/2/3 R^2,
    the singular values of the training predictions (mean over folds) and their fractions,
    per-axis and 3-D absolute error in cm (mean, fold std, median), and the within-house R^2
    (predictions and labels minus their house means). Returns {target: dict}."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    n = X.shape[0]
    P_full = {k: np.zeros_like(Y, dtype=np.float64) for k, Y in Ys.items()}
    P_rank = {k: {1: np.zeros_like(Y, dtype=np.float64), 2: np.zeros_like(Y, dtype=np.float64)}
              for k, Y in Ys.items()}
    S_all = {k: [] for k in Ys}
    alphas = {k: [] for k in Ys}
    fold_mae = {k: [] for k in Ys}
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        Ztr, Zte = sc.transform(X[tr]), sc.transform(X[te])
        for k, Y in Ys.items():
            rg = RidgeCV(alphas=ALPHAS).fit(Ztr, Y[tr])
            alphas[k].append(float(rg.alpha_))
            Yhat_tr, Yhat_te = rg.predict(Ztr), rg.predict(Zte)
            mu = Yhat_tr.mean(0)
            _, S, Vt = np.linalg.svd(Yhat_tr - mu, full_matrices=False)
            S_all[k].append(S)
            P_full[k][te] = Yhat_te
            for r in (1, 2):
                P_rank[k][r][te] = (Yhat_te - mu) @ Vt[:r].T @ Vt[:r] + mu
            err = np.abs(Y[te] - Yhat_te) * 100
            fold_mae[k].append(list(err.mean(0))
                               + [float(np.linalg.norm(Y[te] - Yhat_te, axis=1).mean() * 100)])
    cells = {}
    for k, Y in Ys.items():
        S = np.mean(np.stack(S_all[k]), 0)
        err = np.abs(Y - P_full[k]) * 100
        e3 = np.linalg.norm(Y - P_full[k], axis=1) * 100
        fm = np.asarray(fold_mae[k])
        cells[k] = {
            "n": int(n),
            "ridge_full": _vw(Y, P_full[k]),
            "per_axis": _axis(Y, P_full[k]),
            "rrr": {"1": _vw(Y, P_rank[k][1]), "2": _vw(Y, P_rank[k][2]), "3": _vw(Y, P_full[k])},
            "sv": S.tolist(), "sv_frac": (S / S.sum()).tolist(),
            "quantities": ["x", "y", "z", "3-D"],
            "mae_cm": list(err.mean(0)) + [float(e3.mean())],
            "mae_cm_fold_std": fm.std(0).tolist(),
            "mae_cm_median": list(np.median(err, 0)) + [float(np.median(e3))],
            "r2_within": _vw(house_center(Y, houses), house_center(P_full[k], houses)),
            "r2_within_axis": _axis(house_center(Y, houses), house_center(P_full[k], houses)),
            "alpha_med": float(np.median(alphas[k])),
        }
    return cells


PROBE_RECIPE = ("5 folds over houses (lists/folds/position.csv); per fold StandardScaler shared "
                "by both targets, RidgeCV activations -> (x, y, z), rank 1/2 read-outs from the "
                "SVD of the training predictions; both targets on the same rows and folds")


def probe_stream_layer(store: PositionStore, folds, stream: str, layer: int,
                       grains=("q1", "q0"), log=print) -> dict:
    """Both targets' read-outs from one (token group, layer) of the packed store, per grain.
    `folds` is a vlageom.probes.FoldMap. Returns the JSON-ready result."""
    import time

    _keep, X, lab = store.load_cell(stream, layer)
    masks = row_masks(lab)
    out = {"stream": stream, "layer": int(layer), "grains": {}, "cells": {}, "targets": TARGETS,
           "recipe": PROBE_RECIPE}
    t0 = time.time()
    for g in grains:
        m = masks[g]
        if m.sum() < 500:
            log(f"L{layer} {g}: only {int(m.sum())} rows, skipped")
            continue
        Xg = X[m]
        lab_g = {k: v[m] for k, v in lab.items()}
        houses = lab_g["house"]
        splits = folds.splits(houses)
        out["grains"][g] = {"n": int(m.sum()), "n_houses": int(len(np.unique(houses))),
                            "baselines": baselines(lab_g, splits)}
        Ys = {name: lab_g[key].astype(np.float64) for name, key in TARGETS.items()}
        cells = probe_cell(Xg, Ys, houses, splits)
        out["cells"][g] = cells
        log(f"{stream}@L{layer} {g} n={len(Xg)}: EE R2 {cells['ee']['ridge_full']:.3f} "
            f"(within {cells['ee']['r2_within']:.3f}, 3-D {cells['ee']['mae_cm'][3]:.1f} cm) | "
            f"object R2 {cells['obj']['ridge_full']:.3f} (within {cells['obj']['r2_within']:.3f}, "
            f"3-D {cells['obj']['mae_cm'][3]:.1f} cm)  [{(time.time() - t0) / 60:.1f} min]")
    return out


def probe_json_path(stream: str, layer: int, root=None) -> Path:
    return (Path(root) if root is not None else probe_dir()) / f"{stream}_L{int(layer):02d}.json"


# ---------------------------------------------------------------------------------------------- #
# equivariance maps (panel d)
# ---------------------------------------------------------------------------------------------- #
def rel_resid(Y, P):
    """Median over rows of |Y - P| / |Y| (rows with |Y| <= 1e-9 left out)."""
    n = np.linalg.norm(Y, axis=1)
    keep = n > 1e-9
    if not keep.any():
        return float("nan")
    return float(np.median(np.linalg.norm(Y - P, axis=1)[keep] / n[keep]))


def r2_vw(Y, P):
    return float(1.0 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def fit_ortho(X, Y):
    """Orthogonal Procrustes with a uniform scale: Y ~ c * X @ R. Returns (R, c)."""
    U, S, Vt = np.linalg.svd(X.T @ Y)
    R = U @ Vt
    c = S.sum() / ((X * X).sum() + 1e-12)
    return R, c


def pairs_within(lab, idx, group_keys, anchor_key, value):
    """(anchor rows, other rows, value[other] - value[anchor]) for every group of rows `idx` that
    share `group_keys`; the anchor of a group is its row with the smallest `anchor_key`. Row
    indices are positions within `idx`."""
    keys = np.stack([lab[k][idx] for k in group_keys], 1)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    ia, io = [], []
    ak = lab[anchor_key][idx]
    for g in range(inv.max() + 1):
        m = np.nonzero(inv == g)[0]
        if len(m) < 2:
            continue
        a = m[np.argmin(ak[m])]
        for o in m:
            if o != a:
                ia.append(a)
                io.append(o)
    ia, io = np.asarray(ia), np.asarray(io)
    return ia, io, value[idx][io] - value[idx][ia]


def procrustes_layer(X: dict, lab: dict, grain: str, folds, streams: dict, rng) -> dict:
    """World displacement -> activation displacement maps at one layer, per variable.

    X: {stream: (N, d) float32} at this layer; lab: labels of the packed store; folds: a FoldMap;
    streams: {variable: stream}; rng: numpy Generator for the shuffled reference.
    Pairs: EE = two arm poses with the same object placement (anchor: the smallest pose_id), target
    = two object placements with the same arm pose (anchor: placement 0; the object moves only in
    x, y, so its displacement is 2-D). Per fold, on the training rows: StandardScaler + RidgeCV
    activations -> position, subspace Q = first 3 columns of the QR factor of the weights;
    activation displacement = (z_other - z_anchor) @ Q. A general linear map (least squares) and a
    rotation with uniform scale (orthogonal Procrustes, the 2-D displacement padded with 0) are fit
    on the training pairs and scored on the pairs of the test houses by the median relative
    residual |dh - map(dp)| / |dh|. Reference: the world displacements permuted over all pairs,
    the linear map refit on the permuted training pairs."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    m = np.nonzero(row_masks(lab)[grain])[0]
    houses = lab["house"][m]
    splits = folds.splits(houses)
    out = {}
    for var, stream in streams.items():
        Xg = np.ascontiguousarray(X[stream][m], dtype=np.float64)
        Y = lab[TARGETS[var]][m].astype(np.float64)
        if var == "ee":
            ia, io, Dp = pairs_within(lab, m, ("house", "config_id"), "pose_id",
                                      lab["ee"].astype(np.float64))
        else:
            ia, io, Dp = pairs_within(lab, m, ("house", "pose_id"), "config_id",
                                      lab["obj_base"].astype(np.float64))
            Dp = Dp[:, :2]
        ph = houses[ia]
        Dp3 = np.c_[Dp, np.zeros(len(Dp))] if Dp.shape[1] == 2 else Dp
        res_lin, res_ort, res_sh, r2s, svf, n_te = [], [], [], [], [], 0
        pred_all, dh_all = [], []
        for tr, te in splits:
            tr_h = set(houses[tr].tolist())
            ptr = np.array([h in tr_h for h in ph])
            pte = ~ptr
            if pte.sum() < 10 or ptr.sum() < 10:
                continue
            sc = StandardScaler().fit(Xg[tr])
            rg = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xg[tr]), Y[tr])
            Q = np.linalg.qr(rg.coef_.T)[0][:, :3]
            Z = (Xg - sc.mean_) / sc.scale_
            Dh = (Z[io] - Z[ia]) @ Q
            M = np.linalg.lstsq(Dp[ptr], Dh[ptr], rcond=None)[0]
            R, c = fit_ortho(Dp3[ptr], Dh[ptr])
            P_lin = Dp[pte] @ M
            res_lin.append(rel_resid(Dh[pte], P_lin))
            r2s.append(r2_vw(Dh[pte], P_lin))
            res_ort.append(rel_resid(Dh[pte], c * (Dp3[pte] @ R)))
            Dp_perm = Dp[rng.permutation(len(Dp))]
            M_sh = np.linalg.lstsq(Dp_perm[ptr], Dh[ptr], rcond=None)[0]
            res_sh.append(rel_resid(Dh[pte], Dp_perm[pte] @ M_sh))
            s = np.linalg.svd(M, compute_uv=False)
            svf.append((s / s.sum()).tolist())
            n_te += int(pte.sum())
            pred_all.append(P_lin)
            dh_all.append(Dh[pte])
        if not pred_all:
            out[var] = {"stream": stream, "n_pairs": int(len(Dp)), "n_pairs_test": 0,
                        "resid_ho": float("nan"), "resid_ho_folds": [], "r2_ho": float("nan"),
                        "ortho_resid": float("nan"), "shuffle_resid": float("nan"),
                        "sv_frac": [float("nan")] * 3, "dp_dim": int(Dp.shape[1])}
            continue
        P_all, D_all = np.concatenate(pred_all), np.concatenate(dh_all)
        out[var] = {"stream": stream, "n_pairs": int(len(Dp)), "n_pairs_test": n_te,
                    "resid_ho": float(np.mean(res_lin)), "resid_ho_folds": res_lin,
                    "r2_ho": r2_vw(D_all, P_all), "ortho_resid": float(np.mean(res_ort)),
                    "shuffle_resid": float(np.mean(res_sh)),
                    "sv_frac": np.mean(np.asarray(svf), 0).tolist(), "dp_dim": int(Dp.shape[1])}
    return out


def procrustes_merge(parts_dir, out_dir, grains=("q1", "q0"), log=print) -> list:
    """Merge the per-layer files {grain}_L{LL}.json of `parts_dir` into
    out_dir/procrustes_{grain}.json (layers, per_layer, streams, best). Returns the written
    paths."""
    parts_dir, out_dir = Path(parts_dir), Path(out_dir)
    written = []
    for g in grains:
        parts = sorted(parts_dir.glob(f"{g}_L*.json"), key=lambda p: int(p.stem.split("_L")[1]))
        if not parts:
            continue
        R = {"grain": g, "layers": [], "per_layer": {}, "streams": {}}
        for p in parts:
            d = json.loads(p.read_text())
            R["layers"].append(d["layer"])
            R["per_layer"][str(d["layer"])] = d["vars"]
            for v, c in d["vars"].items():
                R["streams"][v] = c["stream"]
        best = {}
        for v in ("ee", "obj"):
            per = R["per_layer"]
            cand = [(per[str(L)][v]["resid_ho"], L) for L in R["layers"]
                    if v in per[str(L)] and np.isfinite(per[str(L)][v]["resid_ho"])]
            if cand:
                r, L = min(cand)
                best[v] = {"layer": L, "resid_ho": r,
                           "ortho_resid": R["per_layer"][str(L)][v]["ortho_resid"],
                           "shuffle_resid": R["per_layer"][str(L)][v]["shuffle_resid"]}
        R["best"] = best
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / f"procrustes_{g}.json"
        dst.write_text(json.dumps(R, indent=1, default=float))
        written.append(dst)
        log(f"procrustes_{g}.json: {len(R['layers'])} layers; best {json.dumps(best)}")
    return written


# ---------------------------------------------------------------------------------------------- #
# path curvature (panel e)
# ---------------------------------------------------------------------------------------------- #
def straightness_resid(S):
    """sqrt(1 - sv1^2 / sum sv^2) of the centred path: 0 for a straight line."""
    C = S - S.mean(0)
    sv = np.linalg.svd(C, compute_uv=False)
    e = sv ** 2
    return float(np.sqrt(max(0.0, 1.0 - e[0] / (e.sum() + 1e-12))))


def turning(S):
    """(mean, max) turning angle (rad) between consecutive segments of the path S."""
    seg = np.diff(S, axis=0)
    n = np.linalg.norm(seg, axis=1)
    seg, n = seg[n > 1e-9], n[n > 1e-9]
    if len(seg) < 2:
        return 0.0, 0.0
    u = seg / n[:, None]
    ang = np.arccos(np.clip((u[1:] * u[:-1]).sum(1), -1.0, 1.0))
    return float(ang.mean()), float(ang.max())


def lin_resid(P, S):
    """Median |s - p M| / |s| for the least-squares linear map M (no intercept)."""
    M = np.linalg.lstsq(P, S, rcond=None)[0]
    num = np.linalg.norm(S - P @ M, axis=1)
    den = np.linalg.norm(S, axis=1) + 1e-12
    keep = np.linalg.norm(S, axis=1) > 1e-9
    return float(np.median((num / den)[keep])) if keep.any() else float("nan")


def ortho_resid(P, S):
    """Median |s - c p R| / |s| for the best rotation R with uniform scale c."""
    U, sv, Vt = np.linalg.svd(P.T @ S)
    Rm = U @ Vt
    c = sv.sum() / ((P * P).sum() + 1e-12)
    num = np.linalg.norm(S - c * (P @ Rm), axis=1)
    den = np.linalg.norm(S, axis=1) + 1e-12
    keep = np.linalg.norm(S, axis=1) > 1e-9
    return float(np.median((num / den)[keep])) if keep.any() else float("nan")


def affine_resid(P, S):
    """|s - (A p + b)| / |s - mean(s)| for the least-squares affine map."""
    Ph = np.concatenate([P, np.ones((len(P), 1))], 1)
    pred = Ph @ np.linalg.lstsq(Ph, S, rcond=None)[0]
    return float(np.linalg.norm(S - pred) / (np.linalg.norm(S - S.mean(0)) + 1e-12))


def local_angles(S, stride=1):
    """Turning angle (rad) at the interior waypoints j = stride..n-1-stride between the segments
    S[j] - S[j-stride] and S[j+stride] - S[j] (NaN where a segment has zero length)."""
    ang = np.full(max(len(S) - 2 * stride, 0), np.nan)
    for k, j in enumerate(range(stride, len(S) - stride)):
        u = S[j] - S[j - stride]
        v = S[j + stride] - S[j]
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu > 1e-9 and nv > 1e-9:
            ang[k] = np.arccos(np.clip(np.dot(u / nu, v / nv), -1.0, 1.0))
    return ang


def load_hand_lines(lines_dir, stream, layer):
    """Paths of the hand-line harvest: [{house, kind, P (n, 3) EE positions, dist, X (n, d)}] for
    every line with at least MIN_PTS waypoints, waypoints in line order."""
    out = []
    for f in sorted(Path(lines_dir).glob("h*.npz")):
        z = np.load(f)
        ae = stream == "ae"
        li = list(z["layers_ae" if ae else "layers_llm"]).index(layer)
        for pid in np.unique(z["path_id"]):
            m = z["path_id"] == pid
            if m.sum() < MIN_PTS:
                continue
            order = np.argsort(z["step_k"][m])
            X = (z["ae"] if ae else z[f"X_{stream}"])[m][order][:, li, :].astype(np.float64)
            if not np.isfinite(X).all():
                continue
            out.append({"house": int(z["house"]), "kind": str(z["paths"][pid]),
                        "P": z["ee"][m][order].astype(np.float64),
                        "dist": z["dist"][m][order].astype(np.float64), "X": X})
    return out


def load_object_lines(root, stream, layer):
    """Paths of the object-line harvest (root/shape_{kind}/h*/): [{house, kind, P (n, 3) object
    positions, dist, X (n, d)}] for every line with at least MIN_PTS waypoints."""
    import glob

    out = []
    for kind in OBJECT_LINE_KINDS:
        for mf in sorted(glob.glob(str(Path(root) / f"shape_{kind}" / "h*" / "manifest.json"))):
            man = json.loads(Path(mf).read_text())
            pts = man["points"]
            if len(pts) < MIN_PTS:
                continue
            az = np.load(Path(mf).parent / "acts.npz")
            ae = stream == "ae"
            li = [int(x) for x in az["layers_ae" if ae else "layers_llm"]].index(layer)
            X = (az["ae"] if ae else az[f"X_{stream}"])[:, li, :].astype(np.float64)
            if not np.isfinite(X).all():
                continue
            out.append({"house": int(man["house"]), "kind": kind,
                        "P": np.array([p["obj_base"] for p in pts], float),
                        "dist": np.array([p["dist"] for p in pts], float), "X": X})
    return out


def fit_subspace(store: PositionStore, stream, layer, target_key, grain, exclude_houses):
    """Rank-3 read-out subspace: StandardScaler + RidgeCV on the `grain` rows of every house not in
    `exclude_houses`; Q = first 3 columns of the QR factor of the weights.
    Returns (scaler, Q (d, 3), number of fit rows)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    _keep, X, lab = store.load_cell(stream, layer)
    m = row_masks(lab)[grain] & ~np.isin(lab["house"], list(exclude_houses))
    sc = StandardScaler().fit(X[m])
    rg = RidgeCV(alphas=ALPHAS).fit(sc.transform(X[m]), lab[target_key][m].astype(np.float64))
    Q = np.linalg.qr(rg.coef_.T)[0][:, :3]
    return sc, Q, int(m.sum())


def curvature(paths_, sc, Q, rng_seed) -> dict:
    """Per-path straightness measures and the local turning angle vs hand-object distance.

    Each path's activations are scaled (sc) and projected on Q, both path and projection centred.
    Local angle: `local_angles` with stride = round(SEG_M / median waypoint spacing) (at least
    STRIDE_MIN), for the projection, for the world path, and for N_RAND seeded random orthonormal
    3-D projections (median over them). Angles are pooled over paths and binned by the waypoint's
    hand-object distance into N_BINS quantile bins (bins with < 20 points dropped); per bin the
    median angle in degrees."""
    rng = np.random.default_rng(rng_seed)
    Qrs = [np.linalg.qr(rng.standard_normal((Q.shape[0], 3)))[0][:, :3] for _ in range(N_RAND)]
    per, ang, ang_r, ang_w, dist = [], [], [], [], []
    for p in paths_:
        Z = (p["X"] - sc.mean_) / sc.scale_
        S = Z @ Q
        S = S - S.mean(0)
        P = p["P"] - p["P"].mean(0)
        steps = np.linalg.norm(np.diff(p["P"], axis=0), axis=1)
        nz = steps[steps > 1e-4]
        sp = float(np.median(nz)) if len(nz) else \
            float(np.linalg.norm(p["P"][-1] - p["P"][0]) / max(len(p["P"]) - 1, 1))
        stride = int(max(STRIDE_MIN, round(SEG_M / max(sp, 1e-6))))
        mt, _mx = turning(S)
        per.append({"house": p["house"], "kind": p["kind"], "n": len(S),
                    "straightness": straightness_resid(S), "curv_mean_deg": float(np.degrees(mt)),
                    "proc_resid": lin_resid(P, S), "ortho_resid": ortho_resid(P, S),
                    "affine_resid": affine_resid(P, S),
                    "world_straightness": straightness_resid(P)})
        if len(P) <= 2 * stride:
            continue
        per[-1]["stride"] = stride
        per[-1]["spacing_cm"] = sp * 100
        ang.append(local_angles(S, stride))
        ang_r.append(np.nanmedian(np.stack([local_angles(Z @ Qr, stride) for Qr in Qrs]), 0))
        ang_w.append(local_angles(P, stride))
        dist.append(p["dist"][stride:len(P) - stride])
    ang, ang_r, ang_w, dist = map(np.concatenate, (ang, ang_r, ang_w, dist))
    ok = np.isfinite(ang) & np.isfinite(dist)
    edges = np.quantile(dist[ok], np.linspace(0, 1, N_BINS + 1))
    bins = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        bm = ok & (dist >= lo) & ((dist < hi) if i < N_BINS - 1 else (dist <= hi))
        if bm.sum() < 20:
            continue
        bins.append({"d_lo": float(lo), "d_hi": float(hi), "n": int(bm.sum()),
                     "curv_med_deg": float(np.degrees(np.nanmedian(ang[bm]))),
                     "rand_proj_med_deg": float(np.degrees(np.nanmedian(ang_r[bm]))),
                     "world_med_deg": float(np.degrees(np.nanmedian(ang_w[bm])))})
    med = {k: float(np.median([q[k] for q in per]))
           for k in ("straightness", "curv_mean_deg", "proc_resid", "ortho_resid", "affine_resid",
                     "world_straightness")}
    return {"n_paths": len(per), "n_houses": len({q["house"] for q in per}),
            "kinds": sorted({q["kind"] for q in per}),
            "path_medians": med, "bins": bins, "segment_m": SEG_M,
            "strides": sorted({q.get("stride") for q in per if "stride" in q}),
            "overall_curv_med_deg": float(np.degrees(np.nanmedian(ang[ok]))),
            "overall_rand_med_deg": float(np.degrees(np.nanmedian(ang_r[ok]))), "paths": per}


# ---------------------------------------------------------------------------------------------- #
# top-row panel file
# ---------------------------------------------------------------------------------------------- #
def top_panel(probe: dict, proc: dict, curv: dict, grain: str = FIG_GRAIN) -> dict:
    """The arrays the top row of Figure 2 is drawn from.

    probe: {variable: probe JSON of its cell (probe_stream_layer)}; proc: merged Procrustes JSON
    (procrustes_{PROC_GRAIN}.json); curv: curvature JSON. Rows of every (2, ...) array: EE, target.
      cells, layers           token group and layer of each variable's read-out
      mae_cm, mae_cm_fold_std (2, 4) held-out error x, y, z, 3-D (cm) and its fold std   panel a
      rrr                     (2, 3) held-out R^2 at read-out rank 1, 2, 3                 panel b
      sv_frac                 (2, 3) singular-value fractions                              panel c
      proc_layers, proc_resid, proc_ortho
                              (L,), (2, L), (2, L) general-linear and orthogonal residual  panel d
      curv_ee, curv_obj       (n_bins, 4): d_lo, d_hi (m), median turning angle in the subspace
                              and over random projections (deg)                           panel e
      gripper_cm              the gripper-width line of panel a
      n_rows, n_houses        rows and houses of panels a-c"""
    vars_ = ("ee", "obj")
    cells = [probe[v]["cells"][grain][v] for v in vars_]
    layers = [int(L) for L in proc["layers"]]
    for v in vars_:
        missing = [L for L in layers if v not in proc["per_layer"][str(L)]]
        if missing:
            raise ValueError(f"Procrustes result has no {v} entry at layers {missing}")
    out = {
        "cells": np.array([probe[v]["stream"] for v in vars_]),
        "layers": np.array([probe[v]["layer"] for v in vars_], np.int64),
        "mae_cm": np.array([c["mae_cm"] for c in cells], np.float64),
        "mae_cm_fold_std": np.array([c["mae_cm_fold_std"] for c in cells], np.float64),
        "rrr": np.array([[c["rrr"]["1"], c["rrr"]["2"], c["rrr"]["3"]] for c in cells], np.float64),
        "sv_frac": np.array([c["sv_frac"] for c in cells], np.float64),
        "proc_layers": np.array(layers, np.int64),
        "proc_resid": np.array([[proc["per_layer"][str(L)][v]["resid_ho"] for L in layers]
                                for v in vars_], np.float64),
        "proc_ortho": np.array([[proc["per_layer"][str(L)][v]["ortho_resid"] for L in layers]
                                for v in vars_], np.float64),
        "gripper_cm": np.float64(GRIPPER_CM),
        "n_rows": np.int64(probe["ee"]["grains"][grain]["n"]),
        "n_houses": np.int64(probe["ee"]["grains"][grain]["n_houses"]),
    }
    for v in vars_:
        out[f"curv_{v}"] = np.array([[b["d_lo"], b["d_hi"], b["curv_med_deg"],
                                      b["rand_proj_med_deg"]] for b in curv[v]["bins"]], np.float64)
    return out


# ---------------------------------------------------------------------------------------------- #
# trace grids
# ---------------------------------------------------------------------------------------------- #
def grid_chart(store: PositionStore, stream: str, layer: int, label_key: str,
               exclude_house: int = SHAPE_HOUSE, grain: str = CHART_GRAIN) -> dict:
    """Rank-3 read-out subspace of `label_key` for a trace grid, fit on the `grain` rows of every
    house except `exclude_house`: StandardScaler + RidgeCV (float64), basis B = first 3 columns of
    the QR factor of the weights W (orthonormal), decode matrix A = W B (B^T B)^-1, centre and
    95th-percentile radius of the fit rows in the basis, and the sorted labels (`srt`) that define
    the store colour scale. Returns the arrays and `fitstats` (penalty, rows, houses, in-sample
    R^2, mean absolute error on the left-out house in cm)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    _, X, lab = store.load_cell(stream, layer)
    rows = row_masks(lab)[grain]
    m = rows & (lab["house"] != exclude_house)
    mh = rows & (lab["house"] == exclude_house)
    idx = np.nonzero(m)[0]
    Xf = X[idx].astype(np.float64)
    y = lab[label_key][idx].astype(np.float64)
    sc = StandardScaler().fit(Xf)
    Z = sc.transform(Xf)
    rg = RidgeCV(alphas=ALPHAS).fit(Z, y)
    W, c = rg.coef_, np.asarray(rg.intercept_, float)
    B = np.linalg.qr(W.T)[0][:, :3]
    A = W @ B @ np.linalg.inv(B.T @ B)
    S = Z @ B
    center = S.mean(0)
    scale = float(np.percentile(np.linalg.norm(S - center, axis=1), 95)) or 1.0
    pred = rg.predict(Z)
    r2 = 1 - ((pred - y) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)
    fitstats = {"alpha": float(rg.alpha_), "n_fit": int(len(idx)),
                "n_houses_fit": int(len(np.unique(lab["house"][idx]))),
                "r2_insample": [round(float(v), 3) for v in r2]}
    if mh.any():
        ph = rg.predict(sc.transform(X[mh].astype(np.float64)))
        fitstats["n_heldout"] = int(mh.sum())
        err = np.abs(ph - lab[label_key][mh].astype(np.float64)).mean(0)
        fitstats["heldout_house_mae_cm"] = [round(float(v) * 100, 1) for v in err]
    return {"mean": sc.mean_, "scale_": sc.scale_, "W": W, "c": c, "B": B, "A": A,
            "center": center, "scale": np.float64(scale), "srt": np.sort(y, axis=0),
            "fitstats": fitstats, "cell": f"{stream}@L{layer}", "exclude_house": int(exclude_house)}


def chart_project(chart: dict, acts):
    """Activations -> normalized subspace coordinates ((z @ B - centre) / radius)."""
    Zs = (np.atleast_2d(np.asarray(acts, float)) - chart["mean"]) / chart["scale_"]
    return (Zs @ chart["B"] - chart["center"]) / float(chart["scale"])


def cdf_color(reference):
    """Colour function: per-axis empirical CDF of positions over `reference` (n, 3) (x -> red,
    y -> green, z -> blue), clipped to [0, 1]."""
    srt = [np.sort(np.asarray(reference, float)[:, k]) for k in range(3)] \
        if not isinstance(reference, (list, tuple)) else reference
    yk = np.linspace(0.0, 1.0, len(srt[0]))

    def colorfn(E):
        E = np.atleast_2d(np.asarray(E, float))
        return np.clip(np.stack([np.interp(E[:, k], srt[k], yk) for k in range(3)], 1), 0, 1)
    return colorfn


def store_color(chart: dict):
    """Colour function of the target grid: CDF over the fit rows' labels (`chart["srt"]`)."""
    srt = chart["srt"]
    yk = np.linspace(0.0, 1.0, len(srt))

    def colorfn(E):
        E = np.atleast_2d(np.asarray(E, float))
        return np.clip(np.stack([np.interp(E[:, k], srt[:, k], yk) for k in range(3)], 1), 0, 1)
    return colorfn


def load_ee_color_reference(path=None) -> np.ndarray:
    """(4000, 3) reference hand positions of the EE grid colours
    (lists/position/ee_color_reference.csv)."""
    p = Path(path) if path is not None else ee_color_reference_path()
    return np.loadtxt(p, delimiter=",", skiprows=1, dtype=np.float64)


def ee_color(reference=None):
    """Colour function of the EE grid: per-axis CDF over the reference hand positions, rounded to
    EE_COLOR_DECIMALS decimals, clipped to [0, 1]."""
    ref = load_ee_color_reference() if reference is None else np.asarray(reference, float)
    base = cdf_color(ref)

    def colorfn(E):
        rgb = [[round(float(v), EE_COLOR_DECIMALS) for v in row] for row in base(E)]
        return np.clip(np.asarray(rgb, float), 0, 1)
    return colorfn


def reframe(chart: dict):
    """Rotation aligning the subspace axes with the decoded forward, left and up directions
    (`vlageom.target.reframe_U` on the chart's decode matrix and radius)."""
    from vlageom.target import reframe_U
    return reframe_U(chart["A"], float(chart["scale"]))


def merge_close(P, closed, tol=WAYPOINT_TOL):
    """Indices kept after merging consecutive positions closer than `tol` (and, for a closed path,
    a last point that coincides with the first)."""
    keep = [0]
    for i in range(1, len(P)):
        if np.linalg.norm(P[i] - P[keep[-1]]) > tol:
            keep.append(i)
    if closed and len(keep) > 1 and np.linalg.norm(P[keep[-1]] - P[keep[0]]) <= tol:
        keep.pop()
    return keep


def load_shape(shape_dir, stream, layer, point_key):
    """(positions (n, 3) m, activations (n, d) float64, closed, manifest) of one shape directory
    (acts.npz + manifest.json), with coincident consecutive waypoints merged."""
    shape_dir = Path(shape_dir)
    man = json.loads((shape_dir / "manifest.json").read_text())
    closed = bool(man.get("closed", True))
    az = np.load(shape_dir / "acts.npz")
    if stream == "ae":
        layers = [int(x) for x in az["layers_ae"]]
        acts = az["ae"][:, layers.index(int(layer)), :].astype(np.float64)
    else:
        layers = [int(x) for x in az["layers_llm"]]
        acts = az[f"X_{stream}"][:, layers.index(int(layer)), :].astype(np.float64)
    if not np.isfinite(acts).all():
        raise SystemExit(f"non-finite activations in {shape_dir} {stream}@L{layer}")
    P = np.array([p[point_key] for p in man["points"]], float)
    keep = merge_close(P, closed)
    if len(keep) < len(P):
        log.info("%s/%s: merged %d coincident waypoint(s)", shape_dir.parent.name, shape_dir.name,
                 len(P) - len(keep))
    return P[keep], acts[keep], closed, man


def normf(P):
    """Centre x and y, put the lowest z at 0, divide by the largest axis range. Returns (Pn, S)."""
    lo, hi = P.min(0), P.max(0)
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    S = max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1e-6)
    Pn = np.empty_like(P, float)
    Pn[:, 0] = (P[:, 0] - cx) / S
    Pn[:, 1] = (P[:, 1] - cy) / S
    Pn[:, 2] = (P[:, 2] - lo[2]) / S
    return Pn, S


def grid_panel(chart: dict, shapes_dir, variable: str, house: int = SHAPE_HOUSE,
               colorfn=None) -> dict:
    """Arrays of one trace grid. For each of the five shapes of `variable` ("ee": EE_GRID,
    "obj": OBJ_GRID) in `house`: the realized positions and the projected activations (rotated
    by `reframe`), each normalized by `normf`, and the colour of every waypoint.
    Keys: kinds, labels, closed, n_points, world_{i}, sub_{i}, rgb_{i} (i = column)."""
    stream, layer = chart["cell"].split("@L")
    layer = int(layer)
    point_key = "ee_base" if variable == "ee" else "obj_base"
    spec = EE_GRID if variable == "ee" else OBJ_GRID
    U = reframe(chart)
    if colorfn is None:
        colorfn = ee_color() if variable == "ee" else store_color(chart)
    out = {"kinds": np.array([k for k, _ in spec]), "labels": np.array([lab for _, lab in spec]),
           "closed": [], "n_points": []}
    for i, (kind, _lab) in enumerate(spec):
        d = Path(shapes_dir) / f"{variable}_{kind}" / f"h{house}"
        if not (d / "manifest.json").exists():
            raise SystemExit(f"shape {variable}_{kind} of house {house} missing ({d}): "
                             "run its harvest")
        P, acts, closed, _man = load_shape(d, stream, layer, point_key)
        sub = np.asarray(chart_project(chart, acts)) @ U
        rgb = np.clip(np.asarray(colorfn(P)), 0, 1)
        out[f"world_{i}"] = normf(P)[0]
        out[f"sub_{i}"] = normf(sub)[0]
        out[f"rgb_{i}"] = rgb
        out["closed"].append(closed)
        out["n_points"].append(len(P))
    out["closed"] = np.array(out["closed"], bool)
    out["n_points"] = np.array(out["n_points"], np.int64)
    return out


# ---------------------------------------------------------------------------------------------- #
# path designs (pure geometry, robot base frame unless noted)
# ---------------------------------------------------------------------------------------------- #
def hand_line_targets(args, z_surf_b, obj_b) -> dict:
    """The 4 straight hand lines of a house: {0: forward, 1: sideways, 2: up, 3: toward the object}
    as (n_way, 3) base-frame targets. Lines 0-2 pass through the centre of the pose box (forward
    mid, lateral 0, height surface + mid); line 3 runs from that centre toward the object and stops
    `min_obj_dist` short of it (only when the object is more than min_obj_dist + 5 cm away)."""
    c = np.array([0.5 * (args.pose_fwd_lo + args.pose_fwd_hi), 0.0,
                  z_surf_b + 0.5 * (args.pose_up_lo + args.pose_up_hi)])
    n = args.n_way
    lines = {
        0: np.stack([np.linspace(args.pose_fwd_lo, args.pose_fwd_hi, n), np.zeros(n),
                     np.full(n, c[2])], 1),
        1: np.stack([np.full(n, c[0]), np.linspace(-args.pose_lat, args.pose_lat, n),
                     np.full(n, c[2])], 1),
        2: np.stack([np.full(n, c[0]), np.zeros(n),
                     z_surf_b + np.linspace(args.pose_up_lo, args.pose_up_hi, n)], 1),
    }
    d = obj_b - c
    L = float(np.linalg.norm(d))
    if L > args.min_obj_dist + 0.05:
        end = c + d / L * (L - args.min_obj_dist)
        lines[3] = np.stack([np.linspace(c[i], end[i], n) for i in range(3)], 1)
    return lines


def two_circles_path(corners_ABCD, n, lateral=(0.0, 1.0, 0.0)):
    """A figure-8 of two tangent circles in the plane spanned by the (A+C)/2 - (B+D)/2 diagonal and
    the lateral axis, tangent at the corner centroid, radius = half the centroid-to-(A+C)/2
    distance; n points (n/2 per circle), starting and pinched at the centroid."""
    A, B, C, D = [np.asarray(c, float) for c in corners_ABCD]
    P0 = (A + B + C + D) / 4.0
    far = (A + C) / 2.0
    near = (B + D) / 2.0
    dhat = far - near
    dlen = np.linalg.norm(dhat)
    dhat = dhat / dlen if dlen > 1e-9 else np.array([1.0, 0.0, 0.0])
    yhat = np.asarray(lateral, float)
    yhat = yhat / (np.linalg.norm(yhat) + 1e-12)
    r = float(np.linalg.norm(far - P0)) / 2.0
    C_far, C_near = P0 + r * dhat, P0 - r * dhat
    half = n // 2
    out = []
    for phi in np.linspace(np.pi, 3 * np.pi, half, endpoint=False):
        out.append(C_far + r * np.cos(phi) * dhat + r * np.sin(phi) * yhat)
    for phi in np.linspace(0.0, 2 * np.pi, n - half, endpoint=False):
        out.append(C_near + r * np.cos(phi) * dhat + r * np.sin(phi) * yhat)
    return np.array(out)


def loop_path(corners, k):
    """Closed polygon through `corners`, k points per edge, every corner a waypoint."""
    C = [np.asarray(c, float) for c in corners]
    pts = []
    for p0, p1 in zip(C, C[1:] + C[:1]):
        for t in np.linspace(0.0, 1.0, k, endpoint=False):
            pts.append(p0 * (1 - t) + p1 * t)
    return np.array(pts)


HAND_SHAPE_KINDS = ["fig8", "line_x", "line_y", "line_z", "circle", "square"]
OBJECT_SHAPE_KINDS = ["infinity", "ellipse", "square", "line_fwd", "line_lat", "line_ee"]


def hand_shape_design(kind, z_s, obj_b, a):
    """(targets (n, 3), closed, design dict) of one hand path in the base frame. `a` provides the
    pose box (pose_fwd_lo/hi, pose_lat, pose_up_lo/hi above the surface height z_s), the tilted
    shapes' extents (fig8_fwd, fig8_up, circle_r, sq_fwd, sq_lat, sq_up), closed_lift and the
    waypoint counts (n_lines, n_closed, n_square).

      line_x   forward, pose_fwd_lo..hi, lateral 0, height = box middle
      line_y   sideways, -pose_lat..pose_lat, forward = box middle, same height
      line_z   up, surface + pose_up_lo..hi, forward = box middle, lateral 0
      fig8     two tangent circles (`two_circles_path`) whose low end (surface + fig8_up[0]) is at
               fig8_fwd[1] (far) and high end (surface + fig8_up[1]) at fig8_fwd[0] (near)
      circle   radius circle_r in the figure-8's plane, centred in the box, starting at its far,
               low point
      square   forward sq_fwd x lateral +-sq_lat, low side (surface + sq_up[0]) at -sq_lat and high
               side (surface + sq_up[1]) at +sq_lat, every corner a waypoint
    A closed shape whose low region lies over the object is raised by closed_lift as a whole."""
    def up(h):
        return float(z_s) + float(h)
    fmid = 0.5 * (a.pose_fwd_lo + a.pose_fwd_hi)
    cz = up(0.5 * (a.pose_up_lo + a.pose_up_hi))
    n = a.n_lines
    if kind == "line_x":
        T = np.stack([np.linspace(a.pose_fwd_lo, a.pose_fwd_hi, n), np.zeros(n), np.full(n, cz)], 1)
        return T, False, {"axis": "fwd", "lat": 0.0, "up": cz}
    if kind == "line_y":
        T = np.stack([np.full(n, fmid), np.linspace(-a.pose_lat, a.pose_lat, n), np.full(n, cz)], 1)
        return T, False, {"axis": "lat", "fwd": fmid, "up": cz}
    if kind == "line_z":
        T = np.stack([np.full(n, fmid), np.zeros(n),
                      np.linspace(up(a.pose_up_lo), up(a.pose_up_hi), n)], 1)
        return T, False, {"axis": "up", "fwd": fmid, "lat": 0.0}
    ofwd, olat = float(obj_b[0]), float(obj_b[1])
    if kind == "fig8":
        f_near, f_far = float(a.fig8_fwd[0]), float(a.fig8_fwd[1])
        r = 0.25 * float(np.hypot(f_far - f_near, a.fig8_up[1] - a.fig8_up[0]))
        under = (fmid - 0.04 <= ofwd <= f_far + 0.04) and abs(olat) <= r + 0.04
        lift = float(a.closed_lift) if under else 0.0
        A = [f_far, 0.1, up(a.fig8_up[0] + lift)]
        Cc = [f_far, -0.1, up(a.fig8_up[0] + lift)]
        B = [f_near, -0.1, up(a.fig8_up[1] + lift)]
        D = [f_near, 0.1, up(a.fig8_up[1] + lift)]
        T = two_circles_path([A, B, Cc, D], a.n_closed)
        return T, True, {"corners_ABCD": [A, B, Cc, D], "low_end_fwd": f_far,
                         "high_end_fwd": f_near, "radius": r, "lift": lift}
    if kind == "circle":
        f_near, f_far = float(a.fig8_fwd[0]), float(a.fig8_fwd[1])
        lo_up, hi_up = float(a.fig8_up[0]), float(a.fig8_up[1])
        d = np.array([f_near - f_far, 0.0, hi_up - lo_up])
        d /= np.linalg.norm(d)
        r = float(a.circle_r)
        c = np.array([fmid, 0.0, up(0.5 * (lo_up + hi_up))])
        under = (fmid - 0.02 <= ofwd <= fmid + r * abs(d[0]) + 0.04) and abs(olat) <= r + 0.04
        lift = float(a.closed_lift) if under else 0.0
        c[2] += lift
        th = np.linspace(0.0, 2 * np.pi, a.n_closed, endpoint=False)
        T = c + np.outer(r * np.cos(th), -d) + np.outer(r * np.sin(th), np.array([0.0, 1.0, 0.0]))
        return T, True, {"center": c.tolist(), "radius": r, "plane_diagonal": d.tolist(),
                         "lift": lift}
    if kind == "square":
        f0, f1 = float(a.sq_fwd[0]), float(a.sq_fwd[1])
        w = float(a.sq_lat)
        under = (f0 - 0.04 <= ofwd <= f1 + 0.04) and (-w - 0.06 <= olat <= -w + 0.10)
        lift = float(a.closed_lift) if under else 0.0
        zlo, zhi = up(a.sq_up[0] + lift), up(a.sq_up[1] + lift)
        corners = [[f0, -w, zlo], [f1, -w, zlo], [f1, w, zhi], [f0, w, zhi]]
        T = loop_path(corners, max(2, a.n_square // 4))
        return T, True, {"corners": corners, "low_side_lat": -w, "lift": lift}
    raise ValueError(f"unknown hand shape {kind!r}")


def _square_uv(sa, sb, n):
    """n waypoints on the perimeter of the (+-sa, +-sb) rectangle, counter-clockwise from
    (sa, sb); each side starts at its corner."""
    cs = np.array([[sa, sb], [-sa, sb], [-sa, -sb], [sa, -sb]], float)
    n4, rem = divmod(n, 4)
    out = []
    for k in range(4):
        m = n4 + (1 if k < rem else 0)
        frac = np.linspace(0.0, 1.0, m, endpoint=False)[:, None]
        out.append(cs[k] + frac * (cs[(k + 1) % 4] - cs[k]))
    return np.concatenate(out, 0)


def object_shape_design(kind, surf, R, t, z_nat, n, args, ee_xy=None):
    """A planar n-waypoint object path in world xy inside the valid region (support-surface box
    intersected with the reach envelope, on a 120 x 120 grid), or None if it does not fit.

      infinity  x = a sin t, y = b sin 2t      in the valid region's principal axes, closed
      ellipse   x = a cos t, y = b sin t       same frame, closed
      square    rectangle perimeter, corners exact, same frame, closed
      line_fwd  along the robot's forward axis, open
      line_lat  along the robot's lateral axis, open
      line_ee   along the ray from the natural EE position (world xy `ee_xy`) through the centroid
                of the valid region, open
    a, b = 95th percentile of |offset| along each axis (lines: half the 5-95th percentile range,
    recentred). The scale starts at 0.97 and shrinks by 0.94 until every waypoint is inside the
    surface box and the envelope (not below 0.25). Returns dict(kind, closed, xy, t, center, e1,
    e2, a, b, scale)."""
    from vlageom.sim.scene import envelope_ok, to_base

    lo, hi = np.asarray(surf["xy_min"], float), np.asarray(surf["xy_max"], float)
    gx = np.linspace(lo[0], hi[0], 120)
    gy = np.linspace(lo[1], hi[1], 120)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    P = np.stack([GX.ravel(), GY.ravel(), np.full(GX.size, z_nat)], 1)
    ok = np.array([envelope_ok(to_base(R, t, p), args) for p in P])
    if ok.sum() < 40:
        return None
    V = P[ok][:, :2]
    ctr = V.mean(0)
    Cv = V - ctr
    if kind in ("line_fwd", "line_lat", "line_ee"):
        if kind == "line_ee":
            if ee_xy is None:
                return None
            d2 = ctr - np.asarray(ee_xy, float)
            nrm = float(np.hypot(d2[0], d2[1]))
            e1 = d2 / nrm if nrm > 1e-6 else None
        else:
            ax3 = R[:, 0] if kind == "line_fwd" else R[:, 1]
            nrm = float(np.hypot(ax3[0], ax3[1]))
            e1 = np.asarray([ax3[0], ax3[1]], float) / nrm if nrm > 1e-6 else None
        if e1 is None:
            return None
        e2 = np.array([-e1[1], e1[0]])
        q5, q95 = np.percentile(Cv @ e1, [5, 95])
        ctr = ctr + ((q5 + q95) / 2.0) * e1
        a = float((q95 - q5) / 2.0)
        b = float(np.percentile(np.abs(Cv @ e2), 95))
    else:
        _ev, evec = np.linalg.eigh(np.cov(Cv.T))
        e1, e2 = evec[:, 1], evec[:, 0]
        a = float(np.percentile(np.abs(Cv @ e1), 95))
        b = float(np.percentile(np.abs(Cv @ e2), 95))
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    closed = not kind.startswith("line")
    if kind == "infinity":
        tpar = th

        def gen(s):
            return ctr + (s * a * np.sin(th))[:, None] * e1 + (s * b * np.sin(2 * th))[:, None] * e2
    elif kind == "ellipse":
        tpar = th

        def gen(s):
            return ctr + (s * a * np.cos(th))[:, None] * e1 + (s * b * np.sin(th))[:, None] * e2
    elif kind == "square":
        tpar = np.arange(n, dtype=float) / n

        def gen(s):
            return ctr + _square_uv(s * a, s * b, n) @ np.stack([e1, e2])
    elif kind.startswith("line"):
        tpar = np.linspace(-1.0, 1.0, n)

        def gen(s):
            return ctr + (s * a * tpar)[:, None] * e1
    else:
        raise ValueError(f"unknown object shape {kind!r}")
    s = 0.97
    while s > 0.25:
        W = gen(s)
        in_box = np.all((W >= lo) & (W <= hi), axis=1)
        in_env = np.array([envelope_ok(to_base(R, t, [w[0], w[1], z_nat]), args) for w in W])
        if bool(np.all(in_box & in_env)):
            break
        s *= 0.94
    else:
        return None
    return {"kind": kind, "closed": closed, "xy": W, "t": tpar, "center": ctr,
            "e1": np.asarray(e1, float), "e2": np.asarray(e2, float), "a": a, "b": b, "scale": s}


# ---------------------------------------------------------------------------------------------- #
# simulator-side helpers of the harvests (GPU + EGL; import vlageom.sim first)
# ---------------------------------------------------------------------------------------------- #
GRIP = 7
THUMB = 64
# FR3 arm joint limits (rad). Setting joint angles directly bypasses the simulator's limits, so the
# inverse kinematics keeps every joint inside them (with a 0.02 rad margin).
FR3_LIMITS = np.array([(-2.7437, 2.7437), (-1.7837, 1.7837), (-2.9007, 2.9007),
                       (-3.0421, -0.1518), (-2.8065, 2.8065), (0.5445, 4.5169),
                       (-3.0159, 3.0159)], float)


def arm_qpos(eng) -> np.ndarray:
    """The 7 arm joint angles (float64)."""
    from vlageom.sim import get_arm_movegroup
    rv = eng.task._env.robots[0].robot_view
    return np.asarray(get_arm_movegroup(rv).joint_pos, np.float64)[:GRIP].copy()


def ik_base(eng, q0, target, n_iter=100, tol=0.01, damp=0.05, limits=FR3_LIMITS, null_gain=0.1,
            limit_margin=0.02):
    """Damped-least-squares inverse kinematics in the robot base frame, from joint angles q0 to
    the EE target (3,): each iteration sets the joints, reads the EE position and its base-frame
    Jacobian, and steps q by J^+ err plus a null-space pull (gain null_gain) toward the middle of
    the joint range, clipped to the limits minus limit_margin. Stops when the error is below tol or
    after n_iter iterations. Leaves the arm at the returned joints (the caller restores the state).
    Returns (q float32 (7,), EE position, error in m)."""
    from vlageom.sim import set_arm_qpos

    lo, hi = limits[:, 0] + limit_margin, limits[:, 1] - limit_margin
    q_ref = 0.5 * (lo + hi)
    q = np.clip(np.asarray(q0, np.float64)[:GRIP], lo, hi)
    ee = np.asarray(eng.read_ee(), float)
    for _ in range(n_iter):
        set_arm_qpos(eng, q.astype(np.float32))
        ee = np.asarray(eng.read_ee(), float)
        if ee.size != 3:
            return q.astype(np.float32), ee, float("inf")
        err = target - ee
        if np.linalg.norm(err) < tol:
            break
        J3b, _ = eng._read_ee_jacobian()
        if J3b is None:
            break
        J = np.asarray(J3b, float)[:, :GRIP]
        Jpinv = J.T @ np.linalg.inv(J @ J.T + damp ** 2 * np.eye(3))
        dq = Jpinv @ err + null_gain * ((np.eye(GRIP) - Jpinv @ J) @ (q_ref - q))
        q = np.clip(q + dq, lo, hi)
    return q.astype(np.float32), ee, float(np.linalg.norm(target - ee))


def robot_root_ids(eng) -> set:
    """Root body ids of the robot's move groups (base, arm, gripper)."""
    from vlageom.sim import get_arm_movegroup, mj_handles
    rv = eng.task._env.robots[0].robot_view
    model, _ = mj_handles(eng)
    roots = set()
    for mg in getattr(rv, "_move_groups", {}).values():
        try:
            roots.add(int(model.body_rootid[int(mg.root_body_id)]))
        except Exception:
            pass
    if not roots:
        mg = get_arm_movegroup(rv)
        roots.add(int(model.body_rootid[int(mg.root_body_id)]))
    return roots


def arm_penetrations(model, data, robot_roots, obj_root):
    """(n_scene, n_obj): penetrating contacts (dist < 0) between a robot body and anything else,
    split into the pickup object and the rest of the scene. Needs a preceding mj_forward."""
    n_scene = n_obj = 0
    for i in range(data.ncon):
        c = data.contact[i]
        if float(c.dist) >= 0.0:
            continue
        r1 = int(model.body_rootid[model.geom_bodyid[c.geom[0]]])
        r2 = int(model.body_rootid[model.geom_bodyid[c.geom[1]]])
        a, b = (r1 in robot_roots), (r2 in robot_roots)
        if a ^ b:
            other = r2 if a else r1
            if other == obj_root:
                n_obj += 1
            else:
                n_scene += 1
    return n_scene, n_obj


def sample_poses(eng, rng, q_home, z_surf_b, obj_root, args, audit) -> list:
    """`args.n_poses` arm poses: EE targets drawn uniformly in the box forward pose_fwd_lo..hi,
    lateral +-pose_lat, height z_surf_b + pose_up_lo..hi (base frame), with horizontal radius <=
    pose_r_hi and >= pose_min_sep from every accepted target; each reached by `ik_base` from the
    home joints and accepted when the error is <= ik_tol and no robot body penetrates the scene or
    the object (at its natural placement). At most pose_max_tries draws. Leaves the arm at home.
    Returns [{target, q, ee, ik_err}]."""
    from vlageom.sim import mj_handles, set_arm_qpos

    model, data = mj_handles(eng)
    roots = robot_root_ids(eng)
    poses, tries = [], 0
    while len(poses) < args.n_poses and tries < args.pose_max_tries:
        tries += 1
        tgt = np.array([rng.uniform(args.pose_fwd_lo, args.pose_fwd_hi),
                        rng.uniform(-args.pose_lat, args.pose_lat),
                        z_surf_b + rng.uniform(args.pose_up_lo, args.pose_up_hi)], np.float64)
        if float(np.hypot(tgt[0], tgt[1])) > args.pose_r_hi:
            continue
        if any(float(np.linalg.norm(tgt - p["target"])) < args.pose_min_sep for p in poses):
            continue
        q, ee, err = ik_base(eng, q_home, tgt, n_iter=args.ik_iters, tol=args.ik_tol)
        if not np.isfinite(err) or err > args.ik_tol:
            audit["pose_reject_ik"] += 1
            continue
        set_arm_qpos(eng, q)
        n_scene, n_obj = arm_penetrations(model, data, roots, obj_root)
        if n_scene > 0 or n_obj > 0:
            audit["pose_reject_contact"] += 1
            continue
        poses.append({"target": tgt, "q": q.astype(np.float32), "ee": np.asarray(ee, np.float64),
                      "ik_err": float(err)})
    audit["pose_tries"] = tries
    set_arm_qpos(eng, q_home.astype(np.float32))
    if len(poses) < args.n_poses:
        audit["poses_short"] = args.n_poses - len(poses)
        log.warning("pose sampler: %d/%d poses (tries=%d)", len(poses), args.n_poses, tries)
    return poses


def thumb64(fr):
    """64 x 64 x 3 uint8 thumbnail of an exo frame: block mean when the size allows, else
    strided."""
    if fr is None:
        return np.zeros((THUMB, THUMB, 3), np.uint8)
    H, W = fr.shape[:2]
    hs, ws = H // THUMB, W // THUMB
    if hs >= 1 and ws >= 1:
        f = fr[:hs * THUMB, :ws * THUMB].reshape(THUMB, hs, THUMB, ws, 3).mean((1, 3))
        return np.clip(f, 0, 255).astype(np.uint8)
    return np.asarray(fr[::max(1, hs), ::max(1, ws)][:THUMB, :THUMB], np.uint8)


def encode_query_all(eng, kw, llm_layers, ae_layers, seed):
    """One captured query context -> (llm, ae_tok): llm = {variant: (len(llm_layers), 2560)
    float32 or None} (`vlageom.sim.encode.llm_variant_pools` at every layer), ae_tok =
    (len(ae_layers), n_token, 768) float32 outputs of the action-expert blocks at the final flow
    step (flow noise seeded by `seed`), all blocks read in one forward pass. None if a hook did
    not fire or token pooling failed."""
    from vlageom.sim.encode import llm_variant_pools

    seq_cap: dict = {}
    handles = []

    def mk(L):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            seq_cap[L] = t.detach()[0].float().cpu().numpy()
        return hook

    for L in llm_layers:
        handles.append(eng.model.get_submodule(f"transformer.blocks.{L}").register_forward_hook(mk(L)))
    try:
        cache = eng._build_backbone_cache(kw)
    finally:
        for h in handles:
            h.remove()
    if len(seq_cap) != len(llm_layers):
        return None
    ids = kw["input_ids"][0].detach().cpu().numpy()
    llm: dict = {v: [] for v in VARIANTS_LLM}
    for L in llm_layers:
        try:
            pools = llm_variant_pools(seq_cap[L], ids)
        except Exception as exc:
            log.warning("token pooling failed: %s", exc)
            return None
        for v in VARIANTS_LLM:
            llm[v].append(pools[v])
    for v in VARIANTS_LLM:
        llm[v] = (np.stack(llm[v]).astype(np.float32)
                  if all(x is not None for x in llm[v]) else None)
    seq_cap.clear()

    prefix = eng._compute_prefix(cache, seed)
    ae_cap: dict = {}

    def mk_ae(L):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            ae_cap[L] = t.detach().float()[0].cpu().numpy()
        return hook

    handles = [eng.model.get_submodule(f"action_expert.blocks.{L}").register_forward_hook(mk_ae(L))
               for L in ae_layers]
    try:
        eng._final_step(cache, prefix, None, ae_layers[-1], grad=False, per_token=True)
    finally:
        for h in handles:
            h.remove()
    if len(ae_cap) != len(ae_layers):
        return None
    return llm, np.stack([ae_cap[L] for L in ae_layers]).astype(np.float32)


def run_queries(eng, obs, n_chunks, max_loop):
    """Roll the policy forward from `obs` until `n_chunks` model queries have fired (at most
    `max_loop` env steps), without executing past the last query. Query q uses flow noise seeded
    with eng.scene_seed + 1000 q. Returns (caps, states): caps[q] = {kw, actions, q}; states[q] =
    {obj_w, ee_b, ee44, step, qpos, exo_px, thumb}, read right after the query fired (exo_px and
    thumb from the exo frame of the observation the query read)."""
    import torch

    from vlageom.sim import exo_frame
    caps, states = [], []

    def gen(**kw):
        q = eng._chunk
        eng._chunk += 1
        seed = eng.scene_seed + 1000 * q
        kw2, steps = eng._seeded_kw(kw, seed)
        kwc = {k: (v.detach().clone() if torch.is_tensor(v) else v)
               for k, v in kw2.items() if k != "generator"}
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = eng._gen_orig(num_steps=steps, **kw2)
        caps.append({"kw": kwc, "actions": out[0].detach().float().cpu().numpy(), "q": q})
        return out

    orig = eng.model.generate_actions
    eng.model.generate_actions = gen
    try:
        step = 0
        while step < max_loop and not eng.task.is_done():
            pre = len(caps)
            action = eng.policy.get_action(obs)
            if action is None:
                break
            if len(caps) > pre:
                ow = np.asarray(eng.read_object(), np.float64)
                ee = np.asarray(eng.read_ee(), np.float64)
                if ow.size != 3 or ee.size != 3:
                    break
                fr = exo_frame(obs)
                states.append({"obj_w": ow, "ee_b": ee, "ee44": eng.read_ee_full(), "step": step,
                               "qpos": arm_qpos(eng),
                               "exo_px": float(fr.mean()) if fr is not None else float("nan"),
                               "thumb": thumb64(fr)})
            if len(caps) >= n_chunks:
                break
            obs, *_ = eng.task.step(action)
            step += 1
    finally:
        eng.model.generate_actions = orig
    return caps, states


def place_arm(eng, q, model, data, roots, obj_root):
    """Set the arm to joints q, hold it there (controllers re-anchored), run mj_forward and count
    penetrations. Returns (n_scene, n_obj) as `arm_penetrations`."""
    import mujoco

    from vlageom.sim import hold_robot, set_arm_qpos
    set_arm_qpos(eng, q)
    hold_robot(eng)
    mujoco.mj_forward(model, data)
    return arm_penetrations(model, data, roots, obj_root)


def harvest_object_shape(eng, spec, bench, args, out_root, kind, n, encode) -> bool:
    """Move the pickup object along one designed path (`object_shape_design`) on its support
    surface in the episode loaded in `eng`, with the arm at its home pose, and record the first
    model query at every waypoint.

    Per house: reset and settle with the robot held in place; support surface and reach envelope
    as in the object-teleport harvest. Waypoints where the object would penetrate another object
    are skipped (fewer than 60% passing stops the shape). Per waypoint: restore the settled state,
    move the object, settle, keep it when it stays within 2 cm (xy) / 1 cm (z) of the command on
    the same support geom, render the observation (robot-mounted cameras at the robot's pose),
    reset the policy, run the first query and encode it with `encode(eng, kw, seed)` ->
    (llm {variant: (L, 2560)}, ae_tok (L_ae, 16, 768)). A shape with fewer than half its waypoints
    kept is not written.

    Writes out_root/h{house}/: acts.npz (compressed; X_{variant}, ae_tok, ae, chunk, layers_llm,
    layers_ae, variants), manifest.json (house, shape, closed, design, surface, base pose, audit,
    and per waypoint the realized object, EE and commanded positions, settle displacement and exo
    PNG name) and exo_{i:02d}.png. Returns True when written."""
    import imageio.v2 as imageio
    from molmo_spaces.utils.mujoco_scene_utils import get_supporting_geom

    from vlageom.sim import (as_hw3, bad_contact_count, base_transform, exo_frame, fresh_obs,
                             get_pickup_body_name, mj_handles, obj_root_id, resolve_surface,
                             run_chunks, set_body_pos, settle, to_base)

    house = int(spec.house_index)
    out = Path(out_root) / f"h{house}"
    pickup = get_pickup_body_name(eng.task)
    if not pickup:
        log.error("h%d: no pickup_obj_name", house)
        return False
    model, data = mj_handles(eng)

    obs0, _ = eng.task.reset()
    st_nat = settle(eng, args.settle_max, args.settle_window, args.settle_tol)
    snap0 = eng._snapshot(obs0)
    R, t = base_transform(eng)
    p_nat = np.asarray(eng.read_object(), np.float64)
    z_nat = float(p_nat[2])
    surf = resolve_surface(eng, pickup, args)
    root = obj_root_id(eng, pickup)
    if surf["fallback"]:
        log.error("h%d: no support surface found", house)
        return False
    ee_xy = None
    if kind == "line_ee":
        ee_b = np.asarray(eng.read_ee(), np.float64)
        if ee_b.size != 3:
            log.error("h%d: EE position unavailable", house)
            return False
        ee_xy = (R @ ee_b + t)[:2]
    des = object_shape_design(kind, surf, R, t, z_nat, n, args, ee_xy=ee_xy)
    if des is None:
        log.error("h%d: no %s fits the valid region", house, kind)
        return False
    log.info("h%d: %s center=%s a=%.2f b=%.2f scale=%.2f", house, kind,
             np.round(des["center"], 2).tolist(), des["a"], des["b"], des["scale"])

    pre_ok = []
    for xy in des["xy"]:
        set_body_pos(eng.task, pickup, [float(xy[0]), float(xy[1]), z_nat])
        pre_ok.append(bad_contact_count(model, data, root, surf["support_root"]) == 0)
    set_body_pos(eng.task, pickup, [float(p_nat[0]), float(p_nat[1]), float(p_nat[2])])
    pre_ok = np.asarray(pre_ok)
    if pre_ok.sum() < n * 0.6:
        log.error("h%d %s: only %d/%d waypoints free of contacts", house, kind,
                  int(pre_ok.sum()), n)
        return False

    out.mkdir(parents=True, exist_ok=True)
    audit = {"skipped_contact": int((~pre_ok).sum()), "bad_settle": 0, "geom_changed": 0,
             "encode_fail": 0, "short_rollout": 0, "lean_flags": 0,
             "natural_settle_disp": st_nat["last_disp"]}
    X = {v: [] for v in VARIANTS_LLM}
    AE_TOK, AE_POOL, CHUNKS, pts = [], [], [], []
    for i, (xy, tpar) in enumerate(zip(des["xy"], des["t"])):
        if not pre_ok[i]:
            continue
        eng._restore(snap0)
        data.qacc_warmstart[:] = 0.0
        eng.policy.reset()
        eng._chunk = 0
        set_body_pos(eng.task, pickup, [float(xy[0]), float(xy[1]), z_nat])
        st = settle(eng, args.settle_max, args.settle_window, args.settle_tol)
        ow = np.asarray(eng.read_object(), np.float64)
        if float(np.hypot(*(ow[:2] - xy))) > args.xy_tol or abs(float(ow[2]) - z_nat) > args.z_tol:
            audit["bad_settle"] += 1
            continue
        gid2 = get_supporting_geom(data, root)
        if gid2 is None or int(gid2) != surf["gid"]:
            audit["geom_changed"] += 1
            continue
        lean = bad_contact_count(model, data, root, surf["support_root"])
        if lean:
            audit["lean_flags"] += 1
        obs = fresh_obs(eng)
        img = as_hw3(exo_frame(obs))
        caps, states = run_chunks(eng, obs, 1, max_loop=12)
        if len(caps) < 1 or len(states) < 1:
            audit["short_rollout"] += 1
            continue
        enc = encode(eng, caps[0]["kw"], eng.scene_seed)
        if enc is None or any(enc[0][v] is None for v in VARIANTS_LLM
                              if v not in OPTIONAL_VARIANTS):
            audit["encode_fail"] += 1
            continue
        llm, ae_tok = enc
        png = f"exo_{i:02d}.png"
        imageio.imwrite(out / png, img)
        for v in VARIANTS_LLM:
            X[v].append(llm[v].astype(np.float16) if llm[v] is not None
                        else np.full((len(args.llm_layers), 2560), np.nan, np.float16))
        AE_TOK.append(ae_tok.astype(np.float16))
        AE_POOL.append(ae_tok.mean(1).astype(np.float16))
        CHUNKS.append(caps[0]["actions"].astype(np.float32))
        ob = to_base(R, t, states[0]["obj_w"])
        ee = np.asarray(states[0]["ee_b"], np.float64)
        pts.append({"i": int(i), "t_par": float(tpar), "cmd_xy": [float(xy[0]), float(xy[1])],
                    "obj_base": [float(v) for v in ob],
                    "obj_world": [float(v) for v in states[0]["obj_w"]],
                    "ee_base": [float(v) for v in ee], "dist": float(np.linalg.norm(ob - ee)),
                    "settle_disp": float(st["last_disp"]), "lean": int(lean), "exo_png": png})
    if len(pts) < n * 0.5:
        log.error("h%d %s: only %d/%d waypoints kept (audit=%s)", house, kind, len(pts), n, audit)
        return False

    OB = np.array([p["obj_base"] for p in pts])
    e1b = to_base(R, t, t + np.array([des["e1"][0], des["e1"][1], 0.0]))
    e2b = to_base(R, t, t + np.array([des["e2"][0], des["e2"][1], 0.0]))
    p1, p2 = OB @ e1b, OB @ e2b
    corners = [OB[int(np.argmax(p1))], OB[int(np.argmin(p1))],
               OB[int(np.argmax(p2))], OB[int(np.argmin(p2))]]
    payload = dict(ae_tok=np.stack(AE_TOK), ae=np.stack(AE_POOL), chunk=np.stack(CHUNKS),
                   layers_llm=np.array(args.llm_layers, np.int32),
                   layers_ae=np.array(args.ae_layers, np.int32), variants=np.array(VARIANTS_LLM))
    for v in VARIANTS_LLM:
        payload[f"X_{v}"] = np.stack(X[v])
    save_npz(out / "acts.npz", payload, compressed=True)
    (out / "manifest.json.tmp").write_text(json.dumps(
        {"house": house, "bench": str(bench), "shape": kind, "closed": bool(des["closed"]),
         "n_requested": int(n), "corners": [[float(v) for v in c] for c in corners],
         "center_world": des["center"].tolist(),
         "axes_world": [des["e1"].tolist(), des["e2"].tolist()],
         "half_extents": [des["a"], des["b"]], "scale": des["scale"], "z_nat": z_nat,
         "surface": {"xy_min": np.asarray(surf["xy_min"]).tolist(),
                     "xy_max": np.asarray(surf["xy_max"]).tolist(), "top_z": surf["top_z"]},
         "base_pose": np.vstack([np.hstack([R, t[:, None]]), [0, 0, 0, 1]]).tolist(),
         "audit": audit, "points": pts}, indent=1))
    os.replace(out / "manifest.json.tmp", out / "manifest.json")
    log.info("h%d %s -> %s (%d/%d waypoints; audit=%s)", house, kind, out, len(pts), n, audit)
    return True


def exo_check(shape_dir, dark_px=15.0, static_frac=1e-3) -> dict:
    """Exo-frame check of a written shape directory, from its saved exo PNGs: the shape is
    `exo_static` when the frames do not change (mean pixel value spread < 1e-6, all frames
    identical, every pair of consecutive frames differing in fewer than `static_frac` of their
    pixels, or median pixel value < 20); waypoints whose frame is dark (mean pixel value < dark_px)
    are listed in `dark`. Returns the measures."""
    import imageio.v2 as imageio

    man = json.loads((Path(shape_dir) / "manifest.json").read_text())
    frames = [np.asarray(imageio.imread(Path(shape_dir) / p["exo_png"])) for p in man["points"]]
    px = np.array([float(f.mean()) for f in frames], float)
    same = bool(len(frames) > 3 and all(f.shape == frames[0].shape and np.array_equal(frames[0], f)
                                        for f in frames[1:]))
    fracs = [float((a != b).mean()) for a, b in zip(frames[:-1], frames[1:]) if a.shape == b.shape]
    static = bool(len(fracs) >= 3 and max(fracs) < static_frac)
    return {"exo_px_mean": float(px.mean()), "exo_px_std": float(px.std()),
            "exo_px_median": float(np.median(px)), "frames_identical": same,
            "changed_frac_min": (min(fracs) if fracs else None),
            "changed_frac_median": (float(np.median(fracs)) if fracs else None),
            "n": int(len(px)),
            "exo_static": bool(len(px) > 3 and (px.std() < 1e-6 or same or static
                                                or np.median(px) < 20)),
            "dark": [int(i) for i in np.nonzero(px < dark_px)[0]]}
