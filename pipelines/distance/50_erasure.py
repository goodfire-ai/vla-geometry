"""Stage 50 (CPU): erase the directions of one distance-range probe, score both (Figure 3a).

Rows: the object-teleport store of the target tile (last_tok activations at VLM layer 21),
distance d < 0.70 m, houses of the fold map lists/folds/distance_teleport.csv (5 folds, out of
house). Two ranges: near = rows with d < 15 cm, far = rows with 15 <= d < 0.70 m. A ridge probe of
each range reads d (metres) from the rows of its range; its penalty is chosen on held-out training
houses (vlageom.distance.modal_alphas: the most frequent best penalty over the folds).

For each erased range (--erase far near): per fold, the activations are standardized on the
training rows; then k = 1..32 times the erased range's probe is fit on the current activations,
made orthogonal to the directions already erased (Gram-Schmidt) and projected out of the training
and test activations. After k = 0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32 erased directions both range
probes are fit again on their training rows and read out on their test rows
(vlageom.distance.erase_curve). Scores are pooled over the folds: R^2, MAE (cm) and calibration
slope on each range's held-out rows.

Writes --out-dir (default $VLAGEOM_DATA/distance/erasure)/erase_<range>.json per erased range:
row and house counts per range, the fold map and its sha1, the md5 of the test-row indices of the
5 folds, the penalties, the curve [{"k", "near": scores, "far": scores}] and two checks per fold
(largest off-diagonal Gram entry of the erased directions, about 1e-15; largest |activation .
erased direction| left on the erased range's training rows, about 5e-6).

BLAS threads: 24 (--threads), the count of the paper's run. The last bits of every ridge fit
depend on the number of threads, so other counts change the scores at the level of floating-point
rounding. Compute: about 9 minutes (erasing far directions) and 6 minutes (near) on 24 cores, 3 GB
of memory; on fewer cores the same 24 threads take much longer. OPENBLAS_THREAD_TIMEOUT=4 in the
environment shortens the idle spinning of the BLAS threads and does not change the results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from vlageom import paths
from vlageom.distance import (CUT_M, DIST_MAX, ERASE_FOLDS, ERASE_K_MAX, ERASE_K_SCORE,
                              ERASE_NAMES, ERASE_THREADS, FIG3_CELL, erase_curve, erasure_rows,
                              modal_alphas, teleport_prep)
from vlageom.probes import FoldMap


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--teleport-store", type=Path, default=teleport_prep(),
                    help="packed object-teleport store of the target tile "
                         "(default: $VLAGEOM_DATA/target/teleport/prep)")
    ap.add_argument("--folds", default=ERASE_FOLDS,
                    help="fold map name in lists/folds/ or a house,fold CSV "
                         f"(default {ERASE_FOLDS})")
    ap.add_argument("--erase", nargs="+", choices=ERASE_NAMES, default=list(ERASE_NAMES),
                    help="ranges whose probe directions are erased, one output file each "
                         "(default: far near)")
    ap.add_argument("--k-max", type=int, default=ERASE_K_MAX,
                    help=f"erase up to this many directions (default {ERASE_K_MAX}; a smaller "
                         "value gives the first points of the curve quickly)")
    ap.add_argument("--threads", type=int, default=ERASE_THREADS,
                    help=f"BLAS threads (default {ERASE_THREADS})")
    ap.add_argument("--out-dir", type=Path, default=paths.tile_dir("distance", "erasure"),
                    help="output directory (default: $VLAGEOM_DATA/distance/erasure)")
    a = ap.parse_args()

    from threadpoolctl import threadpool_limits
    variant, layer = FIG3_CELL
    folds = FoldMap.load(a.folds)
    with threadpool_limits(limits=a.threads):
        X, y, houses = erasure_rows(a.teleport_store, folds)
        splits = folds.splits(houses)
        fold_md5 = hashlib.md5(np.concatenate([te.astype(np.int64) for _, te in splits])
                               .tobytes()).hexdigest()
        masks = {"near": y < CUT_M, "far": y >= CUT_M}
        ranges = {"near": {"lo_m": 0.0, "hi_m": CUT_M}, "far": {"lo_m": CUT_M, "hi_m": DIST_MAX}}
        for r, m in masks.items():
            ranges[r].update(n=int(m.sum()), n_houses=int(len(np.unique(houses[m]))))
        print(f"rows: {len(y)} with d < {DIST_MAX} m in {len(np.unique(houses))} houses "
              f"(near {ranges['near']['n']}, far {ranges['far']['n']}); fold map {folds.name} "
              f"sha1 {folds.sha1()[:12]}, test-row md5 {fold_md5}", flush=True)
        t0 = time.time()
        alpha, picks = modal_alphas(X, y, houses, splits, masks)
        print(f"penalties: {alpha} (per fold {picks}), {time.time() - t0:.0f} s", flush=True)

        a.out_dir.mkdir(parents=True, exist_ok=True)
        for erase in a.erase:
            t0 = time.time()
            curve, checks = erase_curve(X, y, splits, masks, alpha, erase, k_max=a.k_max,
                                        score_ks=ERASE_K_SCORE)
            wall = round(time.time() - t0, 1)
            print(f"erase {erase} directions ({wall} s): k " + " ".join(str(e["k"]) for e in curve))
            for r in ("near", "far"):
                print(f"  {r:4s} R^2 " + " ".join(f"{e[r]['r2']:.3f}" for e in curve), flush=True)
            print(f"  checks: Gram max|offdiag| {max(checks['gram_max_offdiag'], default=0):.1e}, "
                  f"max|Z . u| {max(checks['resid_max_abs'], default=0):.1e}", flush=True)
            out = {"cell": f"{variant}@L{layer}", "erase": erase, "cut_m": CUT_M,
                   "dist_max_m": DIST_MAX, "k_max": a.k_max,
                   "score_ks": [e["k"] for e in curve], "n_rows": int(len(y)),
                   "n_houses": int(len(np.unique(houses))), "ranges": ranges,
                   "fold_map": folds.name, "fold_map_sha1": folds.sha1(),
                   "fold_test_rows_md5": fold_md5, "threads": a.threads,
                   "alpha": alpha, "alpha_per_fold": picks, "curve": curve, "checks": checks,
                   "teleport_store": str(a.teleport_store), "wall_s": wall}
            p = a.out_dir / f"erase_{erase}.json"
            p.write_text(json.dumps(out, indent=1))
            print(f"wrote {p}", flush=True)


if __name__ == "__main__":
    main()
