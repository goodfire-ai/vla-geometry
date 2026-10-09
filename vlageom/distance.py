"""Distance to the target object: packed stores, the two ruler probes and the Figure 1 panel.

Data
  The distance rows come from two harvests of the same benchmark episodes:
    teleport store  the object-teleport harvest of the target tile (906 houses, 16 object placements
                    per house, model queries q0..q3)
    near store      the near-range harvest of this tile (300 houses, 8 placements per house whose
                    object lies within 0.35 m of the natural end-effector position, queries q0..q7)
  Both are "packed stores": a directory with one float16 array per activation stream
  (`X_<variant>.npy`, rows x layers x features), `labels.npz` (one entry per row) and `meta.json`
  (the layer lists). Rows are in file-name order of the per-house files (`sorted(glob("h*.npz"))`,
  a string sort: h0, h1, h10, h100, ...).

  Placement 0 of the near-range harvest is the object's natural placement, run with the same
  episode seed and flow-noise seeds as the teleport harvest. Its first four queries (q0..q3) are
  therefore the same states as placement 0, q0..q3 of the teleport store, with identical labels
  and activations. `load_distance_rows` takes those rows from the teleport store only
  (`rows_in_teleport_store`), so every state appears once.

Ruler probes
  Two ridge probes read the distance d from last_tok activations at VLM layer 21, fit on every row
  with d < 0.70 m and scored out of house over the shipped fold map `lists/folds/distance.csv`:
    far15 = max(d_cm, 15)   the metre ruler: resolves distances above 15 cm, flat below
    cm15  = min(d_cm, 15)   the centimetre ruler: resolves distances below 15 cm, flat above
  Per fold the activations are standardized on the training rows and the probe is a closed-form
  ridge at alpha 100 (`vlageom.probes.oof_ridge`). The reported scores are R^2 against the true
  distance: far15 on rows with d >= 15 cm, cm15 on rows with d < 15 cm.

Panel
  Figure 1 draws a stratified sample of the out-of-fold read-outs (every row below 15 cm, at most
  400 rows per distance bin above) as points (x = far15 read-out, y = cm15 read-out, z = true
  distance), and the per-bin mean of that sample as a path.

Figure 3 (same activations: last_tok at VLM layer 21)
  erasure       teleport-store rows only (d < 0.70 m, fold map lists/folds/distance_teleport.csv).
                A near probe (rows < 15 cm) and a far probe (rows >= 15 cm) read the distance in
                metres. The directions of one of them are erased one at a time (Gram-Schmidt:
                each refit direction is made orthogonal to those already erased, then projected
                out of the activations), and both probes are refit and scored out of house after
                k = 0..32 erased directions (`erase_curve`). Panel a.
  probe bank    the combined rows of Figure 1 (fold map lists/folds/distance.csv). For every
                boundary c in GRID_CM: round-to-c probes cm_c = min(d, c) and far_c = max(d, c)
                fit on all rows, and range-split probes nr_c / fr_c fit on the rows below / at or
                above c only; plus the full-range probe m (metres). Out-of-fold read-outs, per-fold
                weights in activation units, split-half weight ceilings (`fit_bank`,
                `split_half_ceilings`).
  cutoff table  per c and range (near = rows < c, far = rows >= c): R^2 and MAE of the range-split
                probe, the round-to-c probe and the full-range probe recalibrated per range on
                held-out read-outs (`cutoff_table`). Panel b.
  cosines       per c, the cosine between the near and far probe weights of the range-split and
                the round-to-c pairs, with the split-half ceiling, measured and on a synthetic
                activation set with a single distance direction (`linear_plant`). Panel c.
  plane         the two Figure 1 rulers drawn against each other. Panel d.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vlageom import paths

# ---- constants ------------------------------------------------------------------------------- #
DIST_MAX = 0.70            # rows with d >= 0.70 m are outside the analysed range
RULER_CUT_CM = 15.0        # the two rulers meet at 15 cm
RULER_ALPHA = 100.0        # ridge penalty of both rulers
RULER_CELL = ("last_tok", 21)
NEAR_MAX_Q = 3             # the teleport store holds queries q0..q3

# Distance bins (m) of the plot sample and the mean path: dense near the grasp, coarser far.
CHAN_EDGES = [0.01, 0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35,
              0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
SAMPLE_SEED = 45
SAMPLE_UNIFORM_N = 8000    # uniform sample size
SAMPLE_PER_BIN = 400       # cap per bin in the stratified sample ...
SAMPLE_UNCAPPED_BELOW = 0.15   # ... except bins that end at or below 0.15 m, which keep every row
PATH_MIN_ROWS = 30         # a bin enters the mean path with at least this many sampled rows

# Panel box: x = metre-ruler read-out (m), y = centimetre-ruler read-out (cm),
# z = true distance (m); each axis is mapped to [0, 1] over these limits.
BOX_X_M = (-0.06, 0.76)
BOX_Y_CM = (-1.5, 18.0)
BOX_Z_M = (0.0, 0.71)

# Labels written by the near-range pack stage (plus `house`).
NEAR_LABEL_KEYS = ["obj_base", "ee", "rel_ee", "u", "dist", "config_id", "chunk_q", "step",
                   "fwd", "bearing_deg", "in_env", "grip_width", "contact_other", "ep_done",
                   "success_q"]
NEAR_STREAMS = ["last_tok"]    # VLM token pools kept in the near store (the AE pool is always kept)


# ---- packed stores --------------------------------------------------------------------------- #
def near_root() -> Path:
    """Default root of the near-range store: data_root()/distance/near, holding houses/ (harvest
    output), frames/ (q0 thumbnails) and prep/ (the packed store)."""
    return paths.tile_dir("distance", "near")


def teleport_prep() -> Path:
    """Default packed object-teleport store of the target tile (<teleport store root>/prep)."""
    from vlageom.target import teleport_root
    return teleport_root() / "prep"


def house_files(houses_dir) -> list[Path]:
    """Per-house files `h<house>.npz` of a harvest, in the row order of the packed store
    (string sort of the file names)."""
    return sorted(Path(houses_dir).glob("h*.npz"))


def pack_near_store(houses_dir, out_dir) -> dict:
    """Pack the near-range harvest's per-house files into `out_dir`:
    X_last_tok.npy (N, 13, 2560) f16, X_ae.npy (N, 11, 768) f16 (action-expert token mean),
    labels.npz (NEAR_LABEL_KEYS + house) and meta.json {n_rows, n_houses, layers_llm, layers_ae}.
    Returns the meta dict."""
    files = house_files(houses_dir)
    if not files:
        raise SystemExit(f"no house files h*.npz under {houses_dir}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    z0 = np.load(files[0])
    layers_llm = z0["layers_llm"].tolist()
    layers_ae = z0["layers_ae"].tolist()
    X_llm = {v: [] for v in NEAR_STREAMS}
    X_ae, labs = [], {k: [] for k in NEAR_LABEL_KEYS + ["house"]}
    for f in files:
        z = np.load(f)
        n = len(z["dist"])
        for v in NEAR_STREAMS:
            X_llm[v].append(np.asarray(z[f"X_{v}"], np.float16))
        X_ae.append(np.asarray(z["ae"], np.float16))
        for k in NEAR_LABEL_KEYS:
            labs[k].append(np.asarray(z[k]))
        labs["house"].append(np.full(n, int(z["house"]), np.int32))
    for v in NEAR_STREAMS:
        np.save(out / f"X_{v}.npy", np.concatenate(X_llm[v]))
    np.save(out / "X_ae.npy", np.concatenate(X_ae))
    merged = {k: np.concatenate(v) for k, v in labs.items()}
    np.savez(out / "labels.npz", **merged)
    meta = {"n_rows": int(len(merged["dist"])), "n_houses": int(len(files)),
            "layers_llm": layers_llm, "layers_ae": layers_ae}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def _layer_index(meta, variant, layer):
    key = "layers_ae" if variant in ("ae", "ae_tok") else "layers_llm"
    return int(list(meta[key]).index(int(layer)))


def load_store_cell(store_dir, variant, layer):
    """One (stream, layer) cell of a packed store: (X float32 (n, d), labels dict), restricted to
    the rows whose activations are all finite. `variant` is a VLM token pool such as "last_tok",
    or "ae" for the action-expert token mean."""
    store = Path(store_dir)
    meta = json.loads((store / "meta.json").read_text())
    li = _layer_index(meta, variant, layer)
    name = "X_ae.npy" if variant == "ae" else f"X_{variant}.npy"
    X = np.load(store / name, mmap_mode="r")[:, li, :]
    X = np.ascontiguousarray(X, dtype=np.float32)
    keep = np.isfinite(X).all(1)
    lab = dict(np.load(store / "labels.npz", allow_pickle=False))
    return X[keep], {k: v[keep] for k, v in lab.items()}


def rows_in_teleport_store(near_labels):
    """Near-store rows that are also rows of the teleport store: placement 0 (the natural
    placement) at queries q0..q3."""
    return ((np.asarray(near_labels["config_id"]) == 0)
            & (np.asarray(near_labels["chunk_q"]) <= NEAR_MAX_Q))


def load_distance_rows(teleport_store, near_store, variant="last_tok", layer=21):
    """All distance rows of one (stream, layer) cell: the teleport store's rows, then the near
    store's rows that are not already in the teleport store (`rows_in_teleport_store`), each
    restricted to finite activations and kept in store order.

    Returns (X float32 (n, d), labels) with labels dist (m), house, config_id, chunk_q, obj_base,
    ee (base frame), source (0 = teleport store, 1 = near store), and the near-store-only labels
    grip_width (NaN for teleport rows), contact_other (-1) and ep_done (0)."""
    X1, lab1 = load_store_cell(teleport_store, variant, layer)
    n1 = len(X1)
    out = {k: lab1[k] for k in ("obj_base", "ee", "dist", "chunk_q", "house")}
    out["config_id"] = lab1["config_id"].astype(np.int32)
    out["grip_width"] = np.full(n1, np.nan, np.float32)
    out["contact_other"] = np.full(n1, -1, np.int32)
    out["ep_done"] = np.zeros(n1, np.int32)
    out["source"] = np.zeros(n1, np.int32)

    store = Path(near_store)
    meta = json.loads((store / "meta.json").read_text())
    li = _layer_index(meta, variant, layer)
    name = "X_ae.npy" if variant == "ae" else f"X_{variant}.npy"
    Xx = np.ascontiguousarray(np.load(store / name, mmap_mode="r")[:, li, :], dtype=np.float32)
    labx = dict(np.load(store / "labels.npz", allow_pickle=False))
    keep = np.isfinite(Xx).all(1) & ~rows_in_teleport_store(labx)
    n2 = int(keep.sum())
    add = {"obj_base": labx["obj_base"][keep], "ee": labx["ee"][keep],
           "dist": labx["dist"][keep], "chunk_q": labx["chunk_q"][keep],
           "house": labx["house"][keep],
           "config_id": labx["config_id"][keep].astype(np.int32),
           "grip_width": labx["grip_width"][keep].astype(np.float32),
           "contact_other": labx["contact_other"][keep].astype(np.int32),
           "ep_done": labx["ep_done"][keep].astype(np.int32),
           "source": np.ones(n2, np.int32)}
    X = np.concatenate([X1, Xx[keep]])
    return X, {k: np.concatenate([out[k], add[k]]) for k in out}


# ---- ruler probes ---------------------------------------------------------------------------- #
def ruler_targets(d_cm):
    """{"far15": max(d_cm, 15), "cm15": min(d_cm, 15)} (cm)."""
    return {"far15": np.maximum(d_cm, RULER_CUT_CM), "cm15": np.minimum(d_cm, RULER_CUT_CM)}


def ruler_scores(d_cm, oof):
    """Scores of the two rulers against the true distance d_cm (cm), each on its own side of 15 cm:
    R^2 and mean absolute error (cm), and the row counts of each side."""
    from vlageom.probes import r2
    near = d_cm < RULER_CUT_CM
    far = ~near
    return {
        "far15": {"r2_above15": r2(d_cm[far], oof["far15"][far]),
                  "mae_above15_cm": float(np.mean(np.abs(oof["far15"][far] - d_cm[far])))},
        "cm15": {"r2_below15": r2(d_cm[near], oof["cm15"][near]),
                 "mae_below15_cm": float(np.mean(np.abs(oof["cm15"][near] - d_cm[near])))},
        "n_below15": int(near.sum()), "n_above15": int(far.sum()),
    }


# ---- plot sample and mean path --------------------------------------------------------------- #
def _bin_take(idx, hi, rng):
    if hi <= SAMPLE_UNCAPPED_BELOW or len(idx) <= SAMPLE_PER_BIN:
        return idx
    return rng.choice(idx, SAMPLE_PER_BIN, replace=False)


def plot_samples(d_m, seed=SAMPLE_SEED):
    """Row samples for plotting the ruler read-outs, drawn in this order from one
    numpy default_rng(seed):
      uniform     SAMPLE_UNIFORM_N rows uniformly at random (mostly far rows)
      near_band   the uniform rows plus every row below 0.15 m, in random order
      stratified  per distance bin [0, 0.01) + CHAN_EDGES: every row of bins ending at or below
                  0.15 m, at most SAMPLE_PER_BIN rows (drawn at random) of the bins above
    Returns {name: row-index array}. The stratified sample is the Figure 1 point cloud."""
    y = np.asarray(d_m, np.float64)
    rng = np.random.default_rng(seed)
    uni = rng.choice(len(y), min(SAMPLE_UNIFORM_N, len(y)), replace=False)
    band = rng.permutation(np.union1d(uni, np.nonzero(y < SAMPLE_UNCAPPED_BELOW)[0]))
    strat = []
    edges = [0.0] + CHAN_EDGES
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = np.nonzero((y >= lo) & (y < hi))[0]
        strat += _bin_take(idx, hi, rng).tolist()
    return {"uniform": uni, "near_band": band, "stratified": np.array(strat)}


def mean_path(d_cm, far, cm):
    """Per-bin means over CHAN_EDGES bins with at least PATH_MIN_ROWS rows (bin membership on the
    true distance d_cm): {"far": mean far15 read-out (cm), "cm": mean cm15 read-out (cm),
    "z_true_m": mean true distance (m)}."""
    ds, xs, ys = np.asarray(d_cm), np.asarray(far), np.asarray(cm)
    out = {"far": [], "cm": [], "z_true_m": []}
    for lo, hi in zip(CHAN_EDGES[:-1], CHAN_EDGES[1:]):
        m = (ds >= lo * 100) & (ds < hi * 100)
        if m.sum() >= PATH_MIN_ROWS:
            out["far"].append(float(xs[m].mean()))
            out["cm"].append(float(ys[m].mean()))
            out["z_true_m"].append(float(ds[m].mean() / 100))
    return out


def _unit(v, lim):
    return (np.asarray(v, float) - lim[0]) / (lim[1] - lim[0])


def panel_arrays(d_cm, far, cm):
    """Figure 1 distance panel from the plotted rows (true distance d_cm, far15 and cm15 read-outs,
    all in cm, in plotting order):
      pts     (n, 3) float32  [far15 read-out, cm15 read-out, true distance], each mapped to [0, 1]
              over BOX_X_M (as metres), BOX_Y_CM and BOX_Z_M
      d       (n,) float32    true distance (m)
      path    (k, 3) float32  `mean_path` in the same coordinates
      path_d  (k,) float32    true distance of each path point (m)
      cross   (2,) float64    the 15 cm meeting point of the rulers on the x and y axes"""
    ds = np.asarray(d_cm, float)
    pts = np.stack([_unit(np.asarray(far, float) / 100, BOX_X_M),
                    _unit(cm, BOX_Y_CM), _unit(ds / 100, BOX_Z_M)], 1)
    mp = mean_path(ds, far, cm)
    path = np.stack([_unit(np.asarray(mp["far"], float) / 100, BOX_X_M),
                     _unit(mp["cm"], BOX_Y_CM), _unit(mp["z_true_m"], BOX_Z_M)], 1)
    return dict(pts=pts.astype(np.float32), d=(ds / 100).astype(np.float32),
                path=path.astype(np.float32),
                path_d=np.asarray(mp["z_true_m"], np.float32),
                cross=np.array([_unit(RULER_CUT_CM / 100, BOX_X_M),
                                _unit(RULER_CUT_CM, BOX_Y_CM)], np.float64))


def r2_note(r2_m, r2_cm) -> str:
    """The Figure 1 legend text for the two ruler scores."""
    return f"$R^2$ = {r2_m:.2f} (m) / {r2_cm:.2f} (cm)"


# =============================================================================================== #
# Figure 3
# =============================================================================================== #
FIG3_CELL = ("last_tok", 21)
CUT_M = 0.15               # near / far boundary of the erasure probes (m)
MIN_FIT_ROWS = 150         # a probe is fit only on at least this many rows
MIN_SCORE_ROWS = 30        # a score needs at least this many rows
DEFAULT_ALPHA = 100.0      # ridge penalty when a validation set has too few rows

# Erasure (panel a): teleport-store rows, fold map lists/folds/distance_teleport.csv.
ERASE_FOLDS = "distance_teleport"
ERASE_K_MAX = 32
ERASE_K_SCORE = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32]   # numbers of erased directions scored
ERASE_THREADS = 24         # BLAS threads of the paper's erasure run
ERASE_NAMES = ("far", "near")

# Probe bank, cutoff table and cosines (panels b, c): combined rows, lists/folds/distance.csv.
BANK_FOLDS = "distance"
GRID_CM = [3, 5, 7, 10, 15, 20, 25, 30, 40, 56]          # boundaries c (cm)
BANK_ALPHA = 100.0
BANK_MIN_FIT_HOUSES = 20   # a range-split probe also needs rows from at least this many houses
N_CEIL_DRAWS = 3           # split-half ceilings: number of random house halves
CEIL_SEED = 45
BANK_THREADS = 16          # BLAS threads of the paper's probe bank
CUTOFF_FAMILIES = ("split", "clip", "n0", "n1", "floor")

# Synthetic activations with one distance direction (panel c, dotted curves).
PLANT_FOLDS = "distance_plant"   # house -> fold map of the 120 synthetic houses
PLANT_SEED = 11
PLANT_THREADS = 4         # BLAS threads of the paper's synthetic run


def cosine(a, b):
    """Cosine of two vectors (NaN when either has zero norm)."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


# ---- rows ----------------------------------------------------------------------------------- #
def erasure_rows(teleport_store, folds, cell=FIG3_CELL):
    """Rows of the erasure analysis: the teleport store's finite rows with d < DIST_MAX whose house
    is in the fold map `folds` (a vlageom.probes.FoldMap), in store order.
    Returns (X float32 (n, d), y float64 (n,) true distance in m, houses (n,))."""
    variant, layer = cell
    X, lab = load_store_cell(teleport_store, variant, layer)
    keep = (lab["dist"] < DIST_MAX) & folds.contains(lab["house"])
    return (np.ascontiguousarray(X[keep], np.float32), lab["dist"][keep].astype(np.float64),
            lab["house"][keep])


def bank_rows(teleport_store, near_store, folds, cell=FIG3_CELL):
    """Rows of the probe bank: the combined rows of `load_distance_rows` with d < DIST_MAX whose
    house is in the fold map `folds`, in store order (the rows of the Figure 1 ruler probes).
    Returns (X float32 (n, d), labels dict restricted to these rows)."""
    variant, layer = cell
    X, lab = load_distance_rows(teleport_store, near_store, variant, layer)
    keep = (lab["dist"] < DIST_MAX) & folds.contains(lab["house"])
    return np.ascontiguousarray(X[keep], np.float32), {k: v[keep] for k, v in lab.items()}


# ---- erasure (panel a) ----------------------------------------------------------------------- #
def calib_slope(y_true, y_pred):
    """OLS slope of the true value on the read-out over rows where both are finite (1 = calibrated);
    NaN with fewer than MIN_SCORE_ROWS rows or a constant read-out."""
    ok = np.isfinite(y_pred) & np.isfinite(y_true)
    if ok.sum() < MIN_SCORE_ROWS:
        return float("nan")
    p, t = y_pred[ok], y_true[ok]
    v = p.var()
    return float(((p - p.mean()) * (t - t.mean())).mean() / v) if v > 1e-12 else float("nan")


def range_metrics(y, p):
    """Scores of read-outs p against the true distance y (both m) over rows where p is finite:
    n, R^2, mean absolute error in cm, calibration slope."""
    from vlageom.probes import r2
    ok = np.isfinite(p)
    if ok.sum() < MIN_SCORE_ROWS:
        return {"n": int(ok.sum()), "r2": float("nan"), "mae_cm": float("nan"),
                "slope": float("nan")}
    return {"n": int(ok.sum()), "r2": r2(y[ok].astype(np.float64), p[ok]),
            "mae_cm": float(np.mean(np.abs(y[ok] - p[ok])) * 100),
            "slope": calib_slope(y[ok], p[ok])}


def ridge_predict(Xfit, yfit, Xtest, alpha):
    """Standardize on the fit rows, fit a ridge probe at `alpha`, return its read-outs on Xtest."""
    from vlageom.probes import RidgeEngine, standardize
    Zfit, Zte = standardize(Xfit, Xtest)
    eng = RidgeEngine(Zfit)
    return eng.scores(yfit, alpha, Ztest=Zte, Ktest=(Zte @ Zfit.T if eng.dual else None))


def pick_alpha(Xfit, yfit, Xval, yval):
    """The penalty of ALPHAS_SELECT with the highest validation R^2 (first one on a tie)."""
    from vlageom.probes import ALPHAS_SELECT, RidgeEngine, r2, standardize
    Zfit, Zval = standardize(Xfit, Xval)
    eng = RidgeEngine(Zfit)
    Kv = Zval @ Zfit.T if eng.dual else None
    perf = {a: r2(yval.astype(np.float64), eng.scores(yfit, a, Ztest=Zval, Ktest=Kv))
            for a in ALPHAS_SELECT}
    return max(ALPHAS_SELECT, key=lambda a: np.nan_to_num(perf[a], nan=-9))


def modal_alphas(X, y, houses, splits, masks):
    """Ridge penalty of each range probe. In every fold, every 5th training house (sorted) is a
    validation house; the probe of each range (`masks`: {name: row mask}) is fit on that range's
    rows of the other training houses and the penalty with the best validation R^2 is picked
    (DEFAULT_ALPHA when the range has fewer than MIN_SCORE_ROWS validation rows; no pick with
    fewer than MIN_FIT_ROWS fit rows). The penalty used is the most frequent pick over the folds.
    Returns ({name: penalty}, {name: [pick per fold]})."""
    from collections import Counter
    picks = {b: [] for b in masks}
    for tr, _ in splits:
        gtr = np.unique(houses[tr])
        vmask_tr = np.isin(houses[tr], gtr[::5])
        for b, m in masks.items():
            ifit = tr[~vmask_tr & m[tr]]
            ival = tr[vmask_tr & m[tr]]
            if len(ifit) < MIN_FIT_ROWS:
                continue
            a = (pick_alpha(X[ifit], y[ifit], X[ival], y[ival])
                 if len(ival) >= MIN_SCORE_ROWS else DEFAULT_ALPHA)
            picks[b].append(float(a))
    modal = {b: (Counter(v).most_common(1)[0][0] if v else DEFAULT_ALPHA)
             for b, v in picks.items()}
    return modal, picks


def ridge_direction(Xrows, y, alpha):
    """Ridge weights for y on Xrows, standardized on these rows, mapped back to the units of Xrows
    (weights / per-feature std)."""
    from vlageom.probes import RidgeEngine
    sd = Xrows.std(0) + 1e-8
    Z = (Xrows - Xrows.mean(0, keepdims=True)) / sd
    w = RidgeEngine(Z).weights(y, alpha)
    return np.asarray(w, np.float64) / sd


def erase_curve(X, y, splits, masks, alphas, erase, k_max=ERASE_K_MAX, score_ks=ERASE_K_SCORE):
    """Erase the directions of one range probe and score both range probes after each erasure.

    X (n, d) float32 activations, y (n,) true distance (m), splits [(train, test)] row indices,
    masks {"near": mask, "far": mask} row masks of the two ranges, alphas {"near": a, "far": a},
    erase "far" or "near": the range whose probe directions are erased.

    Per fold the activations are standardized on the training rows. For k = 1..k_max: the probe
    of the erased range is fit on that range's training rows of the current activations
    (`ridge_direction`), made orthogonal to the directions erased before (Gram-Schmidt, applied
    twice), normalized, and projected out of the training and test activations. At every k in
    score_ks (k = 0: nothing erased) both range probes are fit on their training rows of the
    current activations and read out on their test rows. The read-outs are pooled over folds.

    Returns (curve, checks): curve = [{"k", "near": range_metrics, "far": range_metrics}] in k
    order; checks = per fold the largest off-diagonal entry of the Gram matrix of the erased
    directions and the largest |activation . erased direction| left on the erased range's
    training rows."""
    other = "near" if erase == "far" else "far"
    score_ks = sorted(set(int(k) for k in score_ks if int(k) <= k_max) | {0})
    pooled = {k: {r: ([], []) for r in (other, erase)} for k in score_ks}
    checks = {"gram_max_offdiag": [], "resid_max_abs": []}
    for tr, te in splits:
        U = []
        mu = X[tr].mean(0, keepdims=True)
        sd = X[tr].std(0, keepdims=True) + 1e-8
        Z1, Z2 = (X[tr] - mu) / sd, (X[te] - mu) / sd
        ms_tr = masks[erase][tr]
        mt_tr = masks[other][tr]
        if ms_tr.sum() < MIN_FIT_ROWS or mt_tr.sum() < MIN_FIT_ROWS:
            continue
        for k in range(k_max + 1):
            if k > 0:
                w = ridge_direction(Z1[ms_tr], y[tr][ms_tr], alphas[erase])
                if U:
                    Ua = np.asarray(U)
                    for _ in range(2):
                        w = w - Ua.T @ (Ua @ w)
                nw = np.linalg.norm(w)
                if nw == 0:
                    break
                U.append(w / nw)
                w = (w / nw).astype(Z1.dtype)
                Z1 = Z1 - np.outer(Z1 @ w, w)
                Z2 = Z2 - np.outer(Z2 @ w, w)
            if k not in pooled:
                continue
            for r in (other, erase):
                m_tr, m_te = masks[r][tr], masks[r][te]
                if m_te.sum() == 0:
                    continue
                p = ridge_predict(Z1[m_tr], y[tr][m_tr], Z2[m_te], alphas[r])
                pooled[k][r][0].append(y[te][m_te])
                pooled[k][r][1].append(p)
        if U:
            Ua = np.asarray(U)
            G = Ua @ Ua.T
            checks["gram_max_offdiag"].append(float(np.abs(G - np.eye(len(G))).max()))
            checks["resid_max_abs"].append(
                float(np.abs(Z1[ms_tr] @ Ua.T.astype(Z1.dtype)).max()))
    curve = []
    for k in score_ks:
        e = {"k": k}
        for r in ("near", "far"):
            t, p = pooled[k][r]
            if t:
                e[r] = range_metrics(np.concatenate(t), np.concatenate(p))
        curve.append(e)
    return curve, checks


# ---- probe bank (panels b, c) ---------------------------------------------------------------- #
def bank_probe_names(grid=GRID_CM):
    """Probe names: "m", then per boundary c: cm<c>, far<c>, nr<c>, fr<c>."""
    names = ["m"]
    for c in grid:
        names += [f"cm{c}", f"far{c}", f"nr{c}", f"fr{c}"]
    return names


def bank_targets(d_cm, grid=GRID_CM):
    """Targets of the probes fit on all rows: m = d in metres, cm<c> = min(d_cm, c),
    far<c> = max(d_cm, c) (cm). The range-split probes nr<c> / fr<c> read d_cm itself."""
    t = {"m": d_cm / 100.0}
    for c in grid:
        t[f"cm{c}"] = np.minimum(d_cm, c)
        t[f"far{c}"] = np.maximum(d_cm, c)
    return t


def _fit_context(Xrows):
    from vlageom.probes import RidgeEngine
    mu = Xrows.mean(0, keepdims=True)
    sd = Xrows.std(0, keepdims=True) + 1e-8
    return mu, sd, RidgeEngine((Xrows - mu) / sd)


def _weights_predict(mu, sd, eng, t_fit, Xte, alpha):
    w = np.asarray(eng.weights(t_fit, alpha), np.float64)
    return w, ((Xte - mu) / sd) @ w + t_fit.mean()


def fit_bank(X, d_cm, houses, splits, grid=GRID_CM, alpha=BANK_ALPHA):
    """All probes of the bank, fold by fold.

    The full-range probes (m, cm<c>, far<c>) share one standardization and one ridge
    decomposition per fold (all training rows). A range-split probe nr<c> (fr<c>) is fit on the
    training rows with d_cm < c (>= c), standardized on those rows, and only when they number at
    least MIN_FIT_ROWS from at least BANK_MIN_FIT_HOUSES houses. Every probe reads out every
    test row (for a range-split probe also the rows outside its range).

    Returns dict with oof {name: (n,) read-outs in the target's units}, w_raw {name: (5, d)
    weights in activation units (NaN rows for folds without a fit)}, n_fit and n_house
    {name: (5,) fit rows and houses}, fold_id (n,)."""
    from vlageom.probes import N_FOLDS
    n, d = X.shape
    names = bank_probe_names(grid)
    targets = bank_targets(d_cm, grid)
    oof = {k: np.full(n, np.nan) for k in names}
    w_raw = {k: np.full((N_FOLDS, d), np.nan) for k in names}
    n_fit = {k: np.zeros(N_FOLDS, int) for k in names}
    n_house = {k: np.zeros(N_FOLDS, int) for k in names}
    fold_id = np.full(n, -1, int)
    full = ["m"] + [f"{p}{c}" for c in grid for p in ("cm", "far")]
    for f, (tr, te) in enumerate(splits):
        fold_id[te] = f
        Xte = X[te]
        mu, sd, eng = _fit_context(X[tr])
        for k in full:
            w, pred = _weights_predict(mu, sd, eng, targets[k][tr], Xte, alpha)
            oof[k][te] = pred
            w_raw[k][f] = w / sd.ravel()
            n_fit[k][f], n_house[k][f] = len(tr), len(np.unique(houses[tr]))
        del eng
        for c in grid:
            for k, m in ((f"nr{c}", d_cm[tr] < c), (f"fr{c}", d_cm[tr] >= c)):
                rows = tr[m]
                nh = len(np.unique(houses[rows]))
                if len(rows) < MIN_FIT_ROWS or nh < BANK_MIN_FIT_HOUSES:
                    continue
                mu2, sd2, eng2 = _fit_context(X[rows])
                w, pred = _weights_predict(mu2, sd2, eng2, d_cm[rows], Xte, alpha)
                oof[k][te] = pred
                w_raw[k][f] = w / sd2.ravel()
                n_fit[k][f], n_house[k][f] = len(rows), nh
        print(f"  fold {f}: done", flush=True)
    return {"oof": oof, "w_raw": w_raw, "n_fit": n_fit, "n_house": n_house, "fold_id": fold_id}


def split_half_ceilings(X, d_cm, houses, grid=GRID_CM, alpha=BANK_ALPHA, seed=CEIL_SEED,
                        n_draws=N_CEIL_DRAWS):
    """Split-half reliability of every probe's weights: the houses are split into two random
    halves (numpy default_rng(seed).permutation of the sorted unique house ids, first half), each
    probe is fit on each half (same rules as `fit_bank`) and the cosine of the two weight vectors
    (activation units) is recorded; n_draws draws. Returns {name: (n_draws,) cosines}."""
    names = bank_probe_names(grid)
    targets = bank_targets(d_cm, grid)
    full = ["m"] + [f"{p}{c}" for c in grid for p in ("cm", "far")]
    uh = np.unique(houses)
    rng = np.random.default_rng(seed)
    ceil = {k: [] for k in names}
    for draw in range(n_draws):
        half = np.isin(houses, rng.permutation(uh)[: len(uh) // 2])
        w_half = {k: [None, None] for k in names}
        for hi, hm in enumerate((half, ~half)):
            rows_all = np.nonzero(hm)[0]
            mu, sd, eng = _fit_context(X[rows_all])
            for k in full:
                w = np.asarray(eng.weights(targets[k][rows_all], alpha), np.float64)
                w_half[k][hi] = w / sd.ravel()
            del eng
            for c in grid:
                for k, m in ((f"nr{c}", d_cm[rows_all] < c), (f"fr{c}", d_cm[rows_all] >= c)):
                    rows = rows_all[m]
                    nh = len(np.unique(houses[rows]))
                    if len(rows) < MIN_FIT_ROWS or nh < BANK_MIN_FIT_HOUSES:
                        continue
                    mu2, sd2, eng2 = _fit_context(X[rows])
                    w = np.asarray(eng2.weights(d_cm[rows], alpha), np.float64)
                    w_half[k][hi] = w / sd2.ravel()
        for k in names:
            a, b = w_half[k]
            ceil[k].append(cosine(a, b) if a is not None and b is not None else float("nan"))
        print(f"  ceiling draw {draw}: done", flush=True)
    return {k: np.asarray(v) for k, v in ceil.items()}


def pair_names(grid=GRID_CM):
    """The near/far probe pairs compared in panel c: per c, "nr<c>|fr<c>" (range-split) and
    "cm<c>|far<c>" (round-to-c)."""
    return [f"{a}{c}|{b}{c}" for c in grid for a, b in (("cm", "far"), ("nr", "fr"))]


def pair_cosines(w_raw, grid=GRID_CM):
    """Per pair of `pair_names`: mean and std over folds of the cosine between the two probes'
    weights (activation units), over the folds where both were fit."""
    from vlageom.probes import N_FOLDS
    out = {}
    for key in pair_names(grid):
        a, b = key.split("|")
        cs = [cosine(w_raw[a][f], w_raw[b][f]) for f in range(N_FOLDS)
              if np.isfinite(w_raw[a][f]).all() and np.isfinite(w_raw[b][f]).all()]
        if cs:
            out[key] = {"mean": float(np.mean(cs)), "std": float(np.std(cs)), "n_folds": len(cs)}
    return out


def side_scores(y_cm, pred_cm, m):
    """n, MAE (cm) and R^2 of read-outs pred_cm against y_cm (cm) over the rows of mask m where the
    read-out is finite (NaN scores with fewer than MIN_SCORE_ROWS rows)."""
    from vlageom.probes import r2
    m = m & np.isfinite(pred_cm)
    if m.sum() < MIN_SCORE_ROWS:
        return {"n": int(m.sum()), "mae_cm": float("nan"), "r2": float("nan")}
    return {"n": int(m.sum()), "mae_cm": float(np.mean(np.abs(pred_cm[m] - y_cm[m]))),
            "r2": r2(y_cm[m], pred_cm[m])}


def to_cm(name, pred):
    """Read-out of probe `name` in cm (the m probe reads metres)."""
    return pred * 100.0 if name == "m" else pred


def bank_catalog(bank, ceil, d_cm, contact=None, grid=GRID_CM):
    """Per probe: fit rows and houses per fold, split-half ceilings, and R^2 / MAE (cm) against the
    true distance on the rows below and at or above 15 cm. With `contact` (per row: the number of
    contacts between the object and anything but its support, -1 where not recorded) also the
    MAE below 15 cm over the rows with no such contact (mae_below15_precontact_cm)."""
    below = d_cm < RULER_CUT_CM
    cat = {}
    for k in bank_probe_names(grid):
        p = to_cm(k, bank["oof"][k])
        lo, hi = side_scores(d_cm, p, below), side_scores(d_cm, p, ~below)
        cat[k] = {"n_fit": bank["n_fit"][k].tolist(), "n_house_fit": bank["n_house"][k].tolist(),
                  "ceil_raw_cos": ceil[k].tolist(),
                  "r2_below15": lo["r2"], "r2_above15": hi["r2"],
                  "mae_below15_cm": lo["mae_cm"], "mae_above15_cm": hi["mae_cm"]}
        if contact is not None:
            pre = below & (np.asarray(contact) == 0)
            if pre.sum() >= MIN_SCORE_ROWS:
                cat[k]["mae_below15_precontact_cm"] = side_scores(d_cm, p, pre)["mae_cm"]
    return cat


# ---- cutoff table (panel b) ------------------------------------------------------------------ #
def recal_oof(pred_cm, y_cm, fold, regime):
    """Full-range read-outs recalibrated per range: in each fold, the affine map a + b * read-out
    is fit (least squares) on the training rows of the range, using their out-of-fold read-outs,
    and applied to the fold's test rows of the range."""
    from vlageom.probes import N_FOLDS
    out = np.full(len(y_cm), np.nan)
    fin = np.isfinite(pred_cm)
    for f in range(N_FOLDS):
        tr = (fold != f) & regime & fin
        te = (fold == f) & regime & fin
        if tr.sum() < MIN_SCORE_ROWS or te.sum() == 0:
            continue
        p, t = pred_cm[tr], y_cm[tr]
        v = p.var()
        if v < 1e-12:
            a, b = t.mean(), 0.0
        else:
            b = float(((p - p.mean()) * (t - t.mean())).mean() / v)
            a = float(t.mean() - b * p.mean())
        out[te] = a + b * pred_cm[te]
    return out


def floor_pred(y_cm, fold, regime):
    """The training mean of the range, per fold (the score of a probe that reads nothing)."""
    from vlageom.probes import N_FOLDS
    out = np.full(len(y_cm), np.nan)
    for f in range(N_FOLDS):
        tr = (fold != f) & regime
        te = (fold == f) & regime
        if tr.sum() < MIN_SCORE_ROWS or te.sum() == 0:
            continue
        out[te] = y_cm[tr].mean()
    return out


def cutoff_predictions(oof, y_cm, fold, grid=GRID_CM):
    """Read-outs (cm) per boundary c and range ("near": rows < c, "far": rows >= c) of the five
    families: split (nr<c> / fr<c>), clip (cm<c> / far<c>), n0 (the m probe), n1 (the m probe
    recalibrated per range, `recal_oof`) and floor (`floor_pred`)."""
    n0 = to_cm("m", np.asarray(oof["m"], np.float64))
    preds = {}
    for c in grid:
        near = y_cm < c
        far = ~near
        preds[c] = {
            "near": {"split": np.asarray(oof[f"nr{c}"], np.float64),
                     "clip": np.asarray(oof[f"cm{c}"], np.float64),
                     "n0": n0, "n1": recal_oof(n0, y_cm, fold, near),
                     "floor": floor_pred(y_cm, fold, near)},
            "far": {"split": np.asarray(oof[f"fr{c}"], np.float64),
                    "clip": np.asarray(oof[f"far{c}"], np.float64),
                    "n0": n0, "n1": recal_oof(n0, y_cm, fold, far),
                    "floor": floor_pred(y_cm, fold, far)}}
    return preds


def cutoff_table(preds, y_cm):
    """{str(c): {"near"|"far": {"metrics": {family: side_scores on that range}}}}."""
    out = {}
    for c, regs in preds.items():
        out[str(c)] = {}
        for reg, fams in regs.items():
            regime = (y_cm < c) if reg == "near" else (y_cm >= c)
            out[str(c)][reg] = {"metrics": {k: side_scores(y_cm, p, regime)
                                            for k, p in fams.items()}}
    return out


# ---- synthetic activations with one distance direction (panel c) ----------------------------- #
def linear_plant(seed=PLANT_SEED):
    """Synthetic activations that carry the distance along a single direction:
    24,000 rows in 120 houses, 256 features, d_cm ~ U(1, 15) for 30% of the rows and U(1, 70)
    otherwise, X = 0.06 * d_cm * w + 0.5 * N(0, 1) + a per-house offset N(0, 0.3^2) per feature,
    with w a random unit vector. Two further random unit vectors are drawn after w and not used,
    so the generator's sequence (and with it X) is the one of the paper's plant.
    Returns (X float32 (24000, 256), d_cm (24000,), houses (24000,) int, w (256,))."""
    rng = np.random.default_rng(seed)
    n, d, nh = 24000, 256, 120
    houses = rng.integers(0, nh, n)
    d_cm = np.where(rng.random(n) < 0.3, rng.uniform(1, 15, n), rng.uniform(1, 70, n))
    w_sig = rng.normal(size=d)
    w_sig /= np.linalg.norm(w_sig)
    w_b = rng.normal(size=d)
    w_b -= (w_b @ w_sig) * w_sig
    w_b /= np.linalg.norm(w_b)
    w_c = rng.normal(size=d)
    w_c -= (w_c @ w_sig) * w_sig + (w_c @ w_b) * w_b
    w_c /= np.linalg.norm(w_c)
    house_vec = rng.normal(size=(nh, d)) * 0.3
    noise = rng.normal(size=(n, d)) * 0.5 + house_vec[houses]
    X = np.outer(d_cm, w_sig) * 0.06 + noise
    return np.ascontiguousarray(X, np.float32), d_cm, houses, w_sig


def plant_pair_rows(w_raw, ceil, pre_n, pre_f, grid=GRID_CM):
    """Per c: mean, std, min and max over folds of the cosine between probes <pre_n><c> and
    <pre_f><c>, and the smaller of the two probes' median split-half ceilings."""
    from vlageom.probes import N_FOLDS
    rows = []
    for c in grid:
        a, b = f"{pre_n}{c}", f"{pre_f}{c}"
        cs = [cosine(w_raw[a][f], w_raw[b][f]) for f in range(N_FOLDS)
              if np.isfinite(w_raw[a][f]).all() and np.isfinite(w_raw[b][f]).all()]
        cl = [float(np.nanmedian(ceil[k])) for k in (a, b) if np.isfinite(ceil[k]).any()]
        rows.append({"c": c, "cos_mean": float(np.mean(cs)) if cs else None,
                     "cos_std_folds": float(np.std(cs)) if cs else None,
                     "cos_min": float(np.min(cs)) if cs else None,
                     "cos_max": float(np.max(cs)) if cs else None, "n_folds": len(cs),
                     "ceil_weaker_median": min(cl) if cl else None})
    return rows


# ---- Figure 3 panel arrays ------------------------------------------------------------------- #
def erasure_panel(curve_far, curve_near):
    """Panel a from the two erasure curves (`erase_curve` with erase="far" and "near"):
    k (m,), and held-out R^2 of the near and far probes after k far directions were erased
    (near_erase_far, far_erase_far) and after k near directions were erased (near_erase_near,
    far_erase_near)."""
    ks = [e["k"] for e in curve_far]
    if ks != [e["k"] for e in curve_near]:
        raise ValueError("the two erasure curves are scored at different k")
    return {"k": np.asarray(ks, np.int64),
            "near_erase_far": np.asarray([e["near"]["r2"] for e in curve_far], np.float64),
            "far_erase_far": np.asarray([e["far"]["r2"] for e in curve_far], np.float64),
            "near_erase_near": np.asarray([e["near"]["r2"] for e in curve_near], np.float64),
            "far_erase_near": np.asarray([e["far"]["r2"] for e in curve_near], np.float64)}


def cutoff_panel(table, grid):
    """Panel b from the cutoff table: grid_cm (m,) and, per family split / clip / n1, R^2 on the
    near range (<family>_near) and on the far range (<family>_far) at every boundary."""
    out = {"grid_cm": np.asarray(grid, np.int64)}
    for fam in ("split", "clip", "n1"):
        for reg in ("near", "far"):
            out[f"{fam}_{reg}"] = np.asarray(
                [table[str(c)][reg]["metrics"][fam]["r2"] for c in grid], float)
    return out


def cosine_panel(catalog, per_fold, plant, grid):
    """Panel c: per boundary c, for the range-split pair (split_*) and the round-to-c pair
    (round_*): mean and std over folds of the near/far weight cosine (cos, cos_std), the weaker
    probe's median split-half ceiling (ceil), and the plant's mean cosine (plant)."""
    out = {"grid_cm": np.asarray(grid, np.int64)}
    for fam, (pre_n, pre_f) in (("split", ("nr", "fr")), ("round", ("cm", "far"))):
        cm_, cs_, cw_ = [], [], []
        for c in grid:
            e = per_fold.get(f"{pre_n}{c}|{pre_f}{c}") or {}
            cm_.append(e.get("mean", np.nan))
            cs_.append(e.get("std", np.nan))
            cl_n = catalog[f"{pre_n}{c}"]["ceil_raw_cos"]
            cl_f = catalog[f"{pre_f}{c}"]["ceil_raw_cos"]
            cw_.append(min(float(np.nanmedian(cl_n)) if cl_n else np.nan,
                           float(np.nanmedian(cl_f)) if cl_f else np.nan))
        out[f"{fam}_cos"] = np.asarray(cm_)
        out[f"{fam}_cos_std"] = np.asarray(cs_)
        out[f"{fam}_ceil"] = np.asarray(cw_)
        if [r["c"] for r in plant[fam]] != list(grid):
            raise ValueError("plant boundaries differ from the bank boundaries")
        out[f"{fam}_plant"] = np.asarray([r["cos_mean"] for r in plant[fam]], float)
    return out


def query_counts(chunk_q, source, d_m, late_from=NEAR_MAX_Q + 1):
    """Row counts of the combined rows by model query index (chunk_q): per query index over all
    rows, over the teleport-store rows (source 0) and over the near-store rows (source 1); and
    the rows at query index >= late_from (only the near store records them), over all rows and
    over the rows below RULER_CUT_CM."""
    q = np.asarray(chunk_q, np.int64)
    src = np.asarray(source)
    below = np.asarray(d_m, np.float64) * 100 < RULER_CUT_CM
    nq = int(q.max()) + 1 if len(q) else 0
    late = q >= late_from
    return {"n": int(len(q)), "n_below15": int(below.sum()),
            "per_query": np.bincount(q, minlength=nq).tolist(),
            "per_query_teleport": np.bincount(q[src == 0], minlength=nq).tolist(),
            "per_query_near": np.bincount(q[src == 1], minlength=nq).tolist(),
            "late": int(late.sum()), "late_below15": int((late & below).sum())}


def plane_panel(d_m, far, cm):
    """Panel d from the Figure 1 ruler read-outs (stage 30): the stratified plot sample
    (`plot_samples`) as d_cm (true distance, cm), x_m (metre-ruler read-out, m) and y_cm
    (centimetre-ruler read-out, cm), and the per-bin mean path of the sample (`mean_path`)
    as path_x_m and path_y_cm."""
    d_cm = np.asarray(d_m, np.float64) * 100
    s = plot_samples(d_m)["stratified"]
    ds, xs, ys = d_cm[s], np.asarray(far)[s], np.asarray(cm)[s]
    mp = mean_path(ds, xs, ys)
    return {"d_cm": ds, "x_m": xs / 100.0, "y_cm": ys,
            "path_x_m": np.asarray(mp["far"]) / 100.0, "path_y_cm": np.asarray(mp["cm"]),
            "cut_cm": np.float64(RULER_CUT_CM)}
