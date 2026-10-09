"""Figure 7 (gripper control): library code of its four data panels.

All panels start from the gripper edit files of the Figure 1 gripper tile (`vlageom.gripper`,
written by pipelines/gripper/10_harvest_edits.py). Per edit they use the layer-32 activation
without the edit (`base_act`), the edit (`delta`), the decoded gripper command before the edit
(`grip_before`, the chunk mean, -1 open .. +1 closed) and the edit target (+1 close, -1 open).

  panel a  explained-variance ratio of the principal components of the edits of a house split
           (the edits of 80% of the houses; `edit_basis`), and the stable rank 1 / evr1
  panel b  out-of-house ridge probes from the activation to the decoded gripper command: on the
           activation projected onto the top k edit components of the panel-a basis (`topk_probe`)
           and on the full 768-D activation (`full_probe`)
  panel c  the Figure 1 gripper panel file (panels/gripper.npz, `vlageom.gripper.panel_arrays`):
           activations and the binned edit field on PC1-PC2 of all edits
  panel d  arm leakage of edits restricted to delta = mu + V[:, :k] a, with mu and V the mean and
           the principal components of the edits of all other houses (`ksweep_episode`, GPU),
           relative to the unrestricted 768-D edit fitted at the same query (`summarize`)

numpy only at import; torch, scikit-learn and the simulator are imported inside the functions that
need them.
"""
# The low-rank edit fit is vlageom.steer.delta.optimize_delta_basis, run by
# vlageom.sim.BranchEngine.fit_deltas_basis.
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from vlageom import gripper as G

log = logging.getLogger(__name__)

TILE = "gripper_fig7"

# ---- panel a / b: house split, basis and probes ----------------------------------------------- #
TRAIN_FRAC = 0.8          # fraction of houses whose edits make the panel-a/b basis
SPLIT_SEED = 0            # numpy RandomState seed of the house permutation
RIDGE_ALPHA = 10.0        # ridge penalty of the panel-b probes
PROBE_KS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 768)   # top-k edit components of the panel-b curve
PROBE_FOLDS = "gripper_probe"                       # lists/folds/gripper_probe.csv
N_SCREE = 16              # components drawn in panel a

# ---- panel d: low-rank edit fits --------------------------------------------------------------- #
K_LIST = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 640, 768)
LANDMARKS = ("before_grasp", "after_grasp")
POINT = "aw300_sw3000"     # loss-weight tag of the edit files and records
FLIP_TOL = 2.0             # knee: flip rate within this many points of the free-edit flip rate
LEAK_TOL = 1.5             # knee: median leakage at most this multiple of the free-edit leakage



# =============================================================================================== #
# edits
# =============================================================================================== #
def load_listed_edits(edits_dir, episodes=None) -> dict:
    """The edit files of the listed episodes (lists/gripper_episodes.csv, or `episodes`) in
    `edits_dir`, in file-name order, stacked by `vlageom.gripper.load_edits` (H, D, aper, target,
    episode_index, house) plus `files`."""
    episodes = G.read_episodes() if episodes is None else episodes
    files = G.edit_files(edits_dir, episodes)
    if not files:
        raise SystemExit(f"no edit files of the listed episodes in {edits_dir}")
    E = G.load_edits(files)
    E["files"] = files
    return E


def house_split(houses, train_frac=TRAIN_FRAC, seed=SPLIT_SEED):
    """(train houses, held-out houses) as sets: the sorted unique houses permuted by
    numpy RandomState(seed); the first round(n * train_frac) are the train houses."""
    houses = sorted(set(int(h) for h in houses))
    perm = np.random.RandomState(seed).permutation(houses)
    n_train = int(round(len(houses) * train_frac))
    return set(int(h) for h in perm[:n_train]), set(int(h) for h in perm[n_train:])


# =============================================================================================== #
# panel a: basis of the train-split edits
# =============================================================================================== #
def edit_basis(D, target, house, train_frac=TRAIN_FRAC, seed=SPLIT_SEED) -> dict:
    """Principal components of the edits of the train houses (`house_split`).

    D (n, 768) float32 edits, target (n,), house (n,), all in file-name order. The SVD runs in
    torch on the CPU in float32 (`torch.linalg.svd` of the train edits minus their mean).
    Returns mu (768,), V (768, 768) unit columns, evr and sing (768,), mu_close and mu_open (mean
    train edit per target), centered_rank (singular values above 1e-5 * the largest), the house
    split and the edit counts. Column signs of V are those the SVD returns."""
    import torch

    train_h, held_h = house_split(house, train_frac, seed)
    keep = np.isin(np.asarray(house), sorted(train_h))
    Dt = torch.as_tensor(np.ascontiguousarray(D[keep]), device="cpu", dtype=torch.float32)
    tgt = torch.as_tensor(np.asarray(target, np.float32)[keep], device="cpu")
    mu = Dt.mean(0)
    _, S, Vh = torch.linalg.svd(Dt - mu, full_matrices=False)
    centered_rank = int((S > 1e-5 * float(S[0])).sum())
    K = int(Vh.shape[0])
    V = Vh[:K].T.contiguous()
    evr = (S ** 2 / (S ** 2).sum())[:K]
    return dict(
        mu=mu.numpy(), V=V.numpy(), evr=evr.numpy(), sing=S[:K].numpy(),
        mu_close=Dt[tgt > 0].mean(0).numpy(), mu_open=Dt[tgt < 0].mean(0).numpy(),
        centered_rank=np.int64(centered_rank),
        train_houses=np.asarray(sorted(train_h), np.int64),
        heldout_houses=np.asarray(sorted(held_h), np.int64),
        split_seed=np.int64(seed), train_frac=np.float32(train_frac),
        n_train_edits=np.int64(int(keep.sum())), n_edits=np.int64(len(D)))


def stable_rank(sing) -> float:
    """sum(s^2) / s_1^2 (= 1 / evr of PC1), in float64."""
    s2 = np.asarray(sing).astype(np.float64) ** 2
    return float(s2.sum() / s2[0])


# =============================================================================================== #
# panel b: probes
# =============================================================================================== #
def cv_ridge_r2(X, y, splits, alpha=RIDGE_ALPHA):
    """Pooled out-of-fold R^2 of scikit-learn Ridge(alpha) from X (n, ...) to y (n,) over `splits`
    [(train_idx, test_idx)]. Returns (R^2, out-of-fold predictions (n,) float64)."""
    from sklearn.linear_model import Ridge

    X = X.reshape(len(y), -1)
    oof = np.full_like(y, np.nan, dtype=np.float64)
    for tr, te in splits:
        oof[te] = Ridge(alpha=alpha).fit(X[tr], y[tr]).predict(X[te])
    ss_res = float(np.sum((y - oof) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot, oof


def topk_probe(H, aper, splits, mu, V, ks=PROBE_KS, alpha=RIDGE_ALPHA) -> dict:
    """{k: R^2} of ridge probes from (H - mu) @ V[:, :k] to `aper` (panel-b curve)."""
    Z = (H - mu) @ V
    return {int(k): cv_ridge_r2(Z[:, :k], aper, splits, alpha)[0] for k in ks}


def full_probe(H, aper, splits, alpha=RIDGE_ALPHA) -> dict:
    """R^2 and Pearson r of the ridge probe from the full activation H to `aper` (panel-b line)."""
    r2, oof = cv_ridge_r2(H, aper, splits, alpha)
    return dict(r2_full=float(r2), pearson_full=float(np.corrcoef(aper, oof)[0, 1]))


# =============================================================================================== #
# panel d: low-rank edit fits (GPU)
# =============================================================================================== #
def is_opposite(grip_before, target) -> bool:
    """True when the decoded gripper command without the edit (chunk mean) has the sign opposite
    to the edit target, i.e. the edit has a command to flip."""
    return bool(grip_before < 0) if target > 0 else bool(grip_before > 0)


def record_opposite(rec) -> bool:
    """`is_opposite` of a k-sweep record (from its target and the unedited grip_before, which is
    the same in every fit of the record)."""
    if "opposite" in rec:
        return bool(rec["opposite"])
    src = rec.get("ceiling_inrun") or next(iter(rec["results"].values()))
    return is_opposite(float(src["grip_before"]), float(rec["target"]))


class LohoBases:
    """Mean and principal components of the edits of all houses but one, on a torch device.

    D (n, 768), house (n,), target (n,): the basis edits in file-name order. `for_house(h)` returns
    (mu, V (768, K), mu_close, mu_open) of the edits whose house is not h, with
    K = min(k_max, number of components): torch.linalg.svd (float32) of those edits minus their
    mean; mu_close / mu_open are the means of their closing / opening edits."""

    def __init__(self, D, house, target, device, k_max=max(K_LIST)):
        import torch

        self.D = torch.as_tensor(np.asarray(D, np.float32), device=device, dtype=torch.float32)
        self.house = torch.as_tensor(np.asarray(house, np.int64), device=device)
        self.target = torch.as_tensor(np.asarray(target, np.float32), device=device,
                                      dtype=torch.float32)
        self.k_max = int(k_max)

    def for_house(self, house: int):
        import torch

        mask = self.house != int(house)
        Dtr, ttr = self.D[mask], self.target[mask]
        mu = Dtr.mean(0)
        _, _, Vh = torch.linalg.svd(Dtr - mu, full_matrices=False)
        K = int(min(self.k_max, Vh.shape[0]))
        return mu, Vh[:K].T.contiguous(), Dtr[ttr > 0].mean(0), Dtr[ttr < 0].mean(0)


def fit_record(fit, target, coeffs) -> dict:
    """The metrics of one fit as stored in a k-sweep record (coefficients rounded to 4 decimals)."""
    flip = (fit.grip_after > 0) if target > 0 else (fit.grip_after < 0)
    return dict(flip=bool(flip), grip_before=float(fit.grip_before),
                grip_after=float(fit.grip_after), grip_exec=float(fit.grip_exec),
                arm_leakage=float(fit.arm_leakage), arm_vs_natural=float(fit.arm_vs_natural),
                arm_jerk=float(fit.arm_jerk), nat_jerk=float(fit.nat_jerk),
                arm_leak_exec=float(fit.arm_leak_exec), arm_jerk_exec=float(fit.arm_jerk_exec),
                nat_jerk_exec=float(fit.nat_jerk_exec),
                arm_span=float(fit.arm_span), norm=float(fit.norm),
                coeffs=[round(float(c), 4) for c in np.asarray(coeffs).reshape(-1)])


def record_name(episode_index: int, house: int, landmark: str) -> str:
    return f"ep{episode_index}_h{house}_{landmark}.json"


def ksweep_episode(eng, base, bases, episode_index, house, seed, out_dir, *, heldout_houses,
                   k_list=K_LIST, overwrite=False) -> int:
    """Low-rank edit fits of one episode; writes one record per grasp landmark, returns the count.

    `base`: a `BranchEngine.run_baseline` or `load_capture` result. The query of each landmark is
    `vlageom.steer.landmarks.find_landmarks` (one query per landmark). `bases.for_house(house)`
    gives mu, V, mu_close, mu_open. At each query: for every k in `k_list`, Adam on
    delta = mu + V[:, :k] a (start a = V[:, :k]^T (mu_close - mu) for closing edits,
    (mu_open - mu) for opening edits; k = 0 evaluates delta = mu), then the unrestricted 768-D
    edit (`BranchEngine.fit_deltas`); same loss and settings as the gripper edits
    (`vlageom.gripper.FIT`, ARM_WEIGHT, SMOOTH_WEIGHT, arm target = joint angles at the query)."""
    from vlageom.steer.landmarks import find_landmarks

    mu, V, mu_close, mu_open = bases.for_house(int(house))
    kcols = int(V.shape[1])
    k_list_ep = sorted({min(int(k), kcols) for k in k_list})

    def init_coeffs_fn(k, sign):
        if k <= 0:
            return None
        mu_dir = mu_close if sign > 0 else mu_open
        return V[:, :k].T @ (mu_dir - mu)

    lms = find_landmarks(base)
    present = [n for n in LANDMARKS if n in lms]
    log.info("ep%d h%d s%d: %d queries, onset=%s, landmarks=%s", episode_index, house, seed,
             base["n_queries"], lms.get("_grasp_onset_query"),
             {n: lms[n]["chunk"] for n in present})
    if not present:
        log.warning("ep%d h%d: no stable grasp; no records", episode_index, house)
        return 0
    qpos_arm_by_q = base.get("qpos_arm", [])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fit_kw = dict(arm_weight=G.ARM_WEIGHT, grip_weight=G.FIT["grip_weight"],
                  smooth_weight=G.SMOOTH_WEIGHT, iters=G.FIT["iters"], lr=G.FIT["lr"])
    n = 0
    for name in present:
        outpath = out_dir / record_name(episode_index, house, name)
        if outpath.exists() and not overwrite:
            continue
        chunk, target = lms[name]["chunk"], float(lms[name]["target"])
        qarm = qpos_arm_by_q[chunk] if 0 <= chunk < len(qpos_arm_by_q) else None
        freeze = eng.freeze_arm_target(qarm) if qarm is not None else None
        if freeze is None:
            log.warning("ep%d %s: no arm target; the arm term holds the unedited arm actions",
                        episode_index, name)
        res = eng.fit_deltas_basis(chunk, target, G.LAYER, basis=V, mean=mu, k_list=k_list_ep,
                                   arm_target=freeze, init_coeffs_fn=init_coeffs_fn, **fit_kw)
        results = {str(k): {**fit_record(r["fit"], target, r["coeffs"]),
                            "k_eff": int(r["k_eff"]), "loss": float(r["loss"])}
                   for k, r in res.items()}
        free = eng.fit_deltas(chunk, target, [G.LAYER], arm_target=freeze, **fit_kw)[G.LAYER]["fit"]
        rec = dict(ep=int(episode_index), house=int(house), seed=int(seed), landmark=name,
                   target=target, direction=("close" if target > 0 else "open"),
                   layer=int(G.LAYER), branch_query=int(chunk),
                   grasp_onset_query=int(lms.get("_grasp_onset_query", -1)),
                   n_queries=int(base["n_queries"]),
                   is_heldout_split=bool(int(house) in heldout_houses),
                   basis_mode="loho", point=POINT, k_list=list(k_list),
                   arm_mode=("freeze" if freeze is not None else "natural"),
                   arm_weight=float(G.ARM_WEIGHT), smooth_weight=float(G.SMOOTH_WEIGHT),
                   iters=int(G.FIT["iters"]), results=results,
                   ceiling_inrun=fit_record(free, target, []))
        outpath.write_text(json.dumps(rec, indent=2))
        log.info("ep%d h%d %s: wrote k=%s flips=%s", episode_index, house, name, list(results),
                 {k: results[k]["flip"] for k in results})
        n += 1
    return n


# =============================================================================================== #
# panel d: summary of the records (CPU)
# =============================================================================================== #
def read_records(ksweep_dir, episodes=None) -> list[dict]:
    """The k-sweep records (`ep<E>_h<H>_<landmark>.json`, file-name order) of the listed
    episodes."""
    episodes = G.read_episodes() if episodes is None else episodes
    keep = {(int(e[0]), int(e[1])) for e in episodes}
    recs = []
    for f in sorted(Path(ksweep_dir).glob("ep*_*.json")):
        r = json.loads(f.read_text())
        if (int(r["ep"]), int(r["house"])) in keep:
            recs.append(r)
    return recs


def _pct(xs):
    return 100.0 * float(np.mean(xs)) if len(xs) else float("nan")


def summarize_landmark(lm_recs, flip_tol=FLIP_TOL, leak_tol=LEAK_TOL) -> dict:
    """Per-k medians over the records of one landmark, and the unrestricted-edit reference.

    per_k[k]: n, flip_pct (share of fits whose decoded gripper command has the target sign),
    n_opp / flip_pct_opp (the same over the queries where the unedited command is opposite to the
    target), median arm_leakage (`leak`), arm_jerk (`jerk`), grip_after, norm, k_eff.
    ceiling: the same for the 768-D edits fitted in the same run. knee: the smallest k > 0 with a
    flip rate within `flip_tol` points of the ceiling's, median leakage at most `leak_tol` times
    the ceiling's, and median jerk at most the median jerk of the unedited chunk (nat_jerk).
    full_vs_free = leak(k max) / ceiling leak; leak_eff[k] = ceiling leak / leak(k)."""
    opp = [record_opposite(r) for r in lm_recs]
    ks = sorted({int(k) for r in lm_recs for k in r["results"]})
    per_k = {}
    for k in ks:
        pairs = [(r["results"][str(k)], o) for r, o in zip(lm_recs, opp) if str(k) in r["results"]]
        rows = [x for x, _ in pairs]
        per_k[k] = dict(
            n=len(rows),
            flip_pct=_pct([x["flip"] for x in rows]),
            n_opp=int(sum(o for _, o in pairs)),
            flip_pct_opp=_pct([x["flip"] for x, o in pairs if o]),
            leak=float(np.median([x["arm_leakage"] for x in rows])),
            jerk=float(np.median([x["arm_jerk"] for x in rows])),
            grip_after=float(np.median([x["grip_after"] for x in rows])),
            norm=float(np.median([x["norm"] for x in rows])),
            k_eff=int(np.median([x.get("k_eff", k) for x in rows])),
        )
    cpairs = [(r["ceiling_inrun"], o) for r, o in zip(lm_recs, opp) if r.get("ceiling_inrun")]
    ceils = [c for c, _ in cpairs]
    ceil = dict(
        n=len(ceils), source="in-run-free-768",
        flip_pct=_pct([c["flip"] for c in ceils]),
        n_opp=int(sum(o for _, o in cpairs)),
        flip_pct_opp=_pct([c["flip"] for c, o in cpairs if o]),
        leak=float(np.median([c["arm_leakage"] for c in ceils])) if ceils else float("nan"),
        jerk=float(np.median([c["arm_jerk"] for c in ceils])) if ceils else float("nan"),
    )
    nat_jerk = float(np.median([r["results"][str(ks[-1])]["nat_jerk"]
                                for r in lm_recs if str(ks[-1]) in r["results"]]))
    knee = None
    for k in [k for k in ks if k > 0]:
        pk = per_k[k]
        if (pk["flip_pct"] >= ceil["flip_pct"] - flip_tol
                and pk["leak"] <= leak_tol * ceil["leak"]
                and pk["jerk"] <= nat_jerk):
            knee = k
            break
    kmax = max(ks)
    full_vs_free = float(per_k[kmax]["leak"] / ceil["leak"]) if ceil["leak"] else float("nan")
    leak_eff = {k: (ceil["leak"] / per_k[k]["leak"] if per_k[k]["leak"] else float("nan"))
                for k in ks if k > 0}
    return dict(per_k=per_k, ceiling=ceil, nat_jerk=nat_jerk, knee=knee,
                kmax=kmax, kmax_k_eff=per_k[kmax]["k_eff"],
                full_vs_free=full_vs_free, leak_eff=leak_eff, n=len(lm_recs),
                n_opp=int(sum(opp)))


def summarize(recs) -> dict:
    """{landmark: summarize_landmark(records of that landmark) + "opposite_only": the same over the
    records whose unedited command is opposite to the target (None if there are none)}."""
    summary: dict = {}
    for lm in LANDMARKS:
        lm_recs = [r for r in recs if r["landmark"] == lm]
        if not lm_recs:
            continue
        blk = summarize_landmark(lm_recs)
        opp_recs = [r for r in lm_recs if record_opposite(r)]
        blk["opposite_only"] = summarize_landmark(opp_recs) if opp_recs else None
        summary[lm] = blk
    return summary


# =============================================================================================== #
# panel file
# =============================================================================================== #
PANEL_KEYS_AB = ("scree_evr", "stable_rank", "probe_k", "probe_r2", "probe_r2_full")
PANEL_KEYS_D = ("leak_k", "leak_close", "leak_open", "free_leak_close", "free_leak_open",
                "knee_close", "knee_open")
PANEL_KEYS = PANEL_KEYS_AB + PANEL_KEYS_D


def panel_arrays(basis: dict, probe: dict, summary: dict | None) -> dict:
    """Arrays of panels a, b and d (panel c is panels/gripper.npz); without a stage 40 `summary`
    (None) only those of panels a and b.

    scree_evr (16,) float64: explained-variance ratio of the first 16 basis components;
    stable_rank: sum(s^2) / s_1^2 over all components; probe_k, probe_r2: the panel-b curve;
    probe_r2_full: its full-activation line (`probe` is the JSON written by stage 20, with
    "r2_by_k" keyed by str(k)); leak_k (k > 0) and leak_close / leak_open: median
    arm leakage per k of the closing (before the grasp) and opening (after the grasp) edits;
    free_leak_close / free_leak_open: median leakage of the unrestricted 768-D edits; knee_close /
    knee_open (-1 when none)."""
    ks = sorted(int(k) for k in probe["r2_by_k"])
    out = dict(
        scree_evr=np.asarray(basis["evr"]).astype(np.float64)[:N_SCREE],
        stable_rank=np.float64(stable_rank(basis["sing"])),
        probe_k=np.asarray(ks, np.int64),
        probe_r2=np.asarray([float(probe["r2_by_k"][str(k)]) for k in ks], np.float64),
        probe_r2_full=np.float64(probe["r2_full"]))
    if summary is None:
        return out
    sb, sa = summary["before_grasp"], summary["after_grasp"]
    leak_k = sorted({int(k) for s in (sb, sa) for k in s["per_k"] if int(k) > 0})

    def per_k(s, key):
        pk = {int(k): v for k, v in s["per_k"].items()}
        return np.asarray([pk[k][key] for k in leak_k], np.float64)

    return dict(
        out,
        leak_k=np.asarray(leak_k, np.int64),
        leak_close=per_k(sb, "leak"), leak_open=per_k(sa, "leak"),
        free_leak_close=np.float64(sb["ceiling"]["leak"]),
        free_leak_open=np.float64(sa["ceiling"]["leak"]),
        knee_close=np.int64(sb["knee"] if sb["knee"] is not None else -1),
        knee_open=np.int64(sa["knee"] if sa["knee"] is not None else -1))
