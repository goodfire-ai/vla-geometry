"""Stage 60 (CPU): the distance probe bank across boundaries c = 3..56 cm (Figure 3b, 3c).

Rows: the combined rows of stage 30 (teleport store + near store, without the near rows the
teleport store already holds; d < 0.70 m; houses of lists/folds/distance.csv), last_tok
activations at VLM layer 21. For every boundary c in 3, 5, 7, 10, 15, 20, 25, 30, 40, 56 cm:

  cm<c>   round-to-c, near side: target min(d_cm, c), fit on all rows
  far<c>  round-to-c, far side:  target max(d_cm, c), fit on all rows
  nr<c>   range-split, near:     target d_cm, fit on the rows with d_cm < c only
  fr<c>   range-split, far:      target d_cm, fit on the rows with d_cm >= c only
  m       full range:            target d in metres, fit on all rows (used by stage 70)

Ridge penalty 100. Per fold of the fold map the activations are standardized on the probe's fit
rows; the full-range probes share one fit per fold (vlageom.distance.fit_bank). A range-split
probe is fit only with at least 150 rows from at least 20 houses. Every probe reads out every
held-out row. The split-half ceiling of each probe is the cosine between its weights fit on two
random halves of the houses (3 draws, numpy default_rng(45); vlageom.distance.split_half_ceilings).

Writes --out (default $VLAGEOM_DATA/distance/probe_bank_last_tok_L21.npz) with
  dist (n,) float32 (m), house, fold_id, source, chunk_q, grid_cm,
  oof__<probe> (n,) float32       out-of-fold read-out (metres for m, else cm)
  w_raw__<probe> (5, 2560) float32  weights per fold in activation units (NaN: not fit)
  ceil__<probe> (3,) float64       split-half cosines
and a .json next to it: row counts, fold map, per probe the fit counts, ceilings and the R^2 /
MAE (cm) against the true distance below and at or above 15 cm, and the MAE below 15 cm on the
near-store rows where the object touches nothing but its support ("catalog"), and per
near/far pair of the same c the mean and std over folds of the cosine between their weights
("cosines").

BLAS threads: 16 (--threads), the count of the paper's run; with the same library versions on an
x86-64 CPU with AVX-512 this gives the paper's numbers bit for bit. Compute: about 8 minutes on 16
cores and 9 GB of memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from vlageom import paths
from vlageom.distance import (BANK_ALPHA, BANK_FOLDS, BANK_MIN_FIT_HOUSES, BANK_THREADS,
                              DIST_MAX, FIG3_CELL, GRID_CM, MIN_FIT_ROWS, N_CEIL_DRAWS,
                              bank_catalog, bank_probe_names, bank_rows, fit_bank,
                              near_root, pair_cosines, split_half_ceilings, teleport_prep)
from vlageom.probes import FoldMap


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--teleport-store", type=Path, default=teleport_prep(),
                    help="packed object-teleport store "
                         "(default: $VLAGEOM_DATA/target/teleport/prep)")
    ap.add_argument("--near-store", type=Path, default=near_root() / "prep",
                    help="packed near-range store (default: $VLAGEOM_DATA/distance/near/prep)")
    ap.add_argument("--folds", default=BANK_FOLDS,
                    help="fold map name in lists/folds/ or a house,fold CSV "
                         f"(default {BANK_FOLDS})")
    ap.add_argument("--threads", type=int, default=BANK_THREADS,
                    help=f"BLAS threads (default {BANK_THREADS})")
    ap.add_argument("--out", type=Path,
                    default=paths.tile_dir("distance", "probe_bank_last_tok_L21.npz"))
    a = ap.parse_args()

    from threadpoolctl import threadpool_limits
    variant, layer = FIG3_CELL
    folds = FoldMap.load(a.folds)
    t0 = time.time()
    with threadpool_limits(limits=a.threads):
        X, lab = bank_rows(a.teleport_store, a.near_store, folds)
        d_cm = lab["dist"].astype(np.float64) * 100
        houses = lab["house"]
        print(f"rows: {len(d_cm)} with d < {DIST_MAX} m ({int((lab['source'] == 1).sum())} from "
              f"the near store), {len(np.unique(houses))} houses; fold map {folds.name} "
              f"sha1 {folds.sha1()[:12]}; boundaries {GRID_CM}", flush=True)
        bank = fit_bank(X, d_cm, houses, folds.splits(houses))
        ceil = split_half_ceilings(X, d_cm, houses)
        cat = bank_catalog(bank, ceil, d_cm, lab["contact_other"])
        cos = pair_cosines(bank["w_raw"])

    npz = {"dist": lab["dist"].astype(np.float32), "house": houses.astype(np.int64),
           "fold_id": bank["fold_id"], "source": lab["source"], "chunk_q": lab["chunk_q"],
           "grid_cm": np.asarray(GRID_CM)}
    for k in bank_probe_names():
        npz[f"oof__{k}"] = bank["oof"][k].astype(np.float32)
        npz[f"w_raw__{k}"] = bank["w_raw"][k].astype(np.float32)
        npz[f"ceil__{k}"] = ceil[k]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **npz)
    side = {"cell": f"{variant}@L{layer}", "alpha": BANK_ALPHA, "grid_cm": GRID_CM,
            "dist_max_m": DIST_MAX, "min_fit_rows": MIN_FIT_ROWS,
            "min_fit_houses": BANK_MIN_FIT_HOUSES, "n_ceil_draws": N_CEIL_DRAWS,
            "n_rows": int(len(d_cm)), "n_houses": int(len(np.unique(houses))),
            "n_near_store_rows": int((lab["source"] == 1).sum()),
            "fold_map": folds.name, "fold_map_sha1": folds.sha1(), "threads": a.threads,
            "units": {"m": "m", "default": "cm"},
            "catalog": cat, "cosines": {"per_fold": cos},
            "teleport_store": str(a.teleport_store), "near_store": str(a.near_store),
            "out_sha1": hashlib.sha1(a.out.read_bytes()).hexdigest(),
            "wall_s": round(time.time() - t0, 1)}
    a.out.with_suffix(".json").write_text(json.dumps(side, indent=1))
    for c in GRID_CM:
        sp, rd = cos.get(f"nr{c}|fr{c}", {}), cos.get(f"cm{c}|far{c}", {})
        print(f"  c={c:>2}: cos(nr, fr) {sp.get('mean', float('nan')):+.4f}  "
              f"cos(cm, far) {rd.get('mean', float('nan')):+.4f}  ceilings nr/fr "
              f"{np.nanmedian(ceil[f'nr{c}']):.3f}/{np.nanmedian(ceil[f'fr{c}']):.3f}")
    print(f"wrote {a.out} (+ .json), {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
