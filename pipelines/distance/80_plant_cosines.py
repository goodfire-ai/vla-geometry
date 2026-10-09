"""Stage 80 (CPU): near/far probe cosines on synthetic activations with one distance direction
(Figure 3c, dotted curves).

The synthetic set (vlageom.distance.linear_plant, numpy default_rng(11)): 24,000 rows in 120
houses, 256 features, X = 0.06 * d_cm * w + 0.5 * N(0, 1) noise + a per-house offset, with one
random unit direction w; d_cm is uniform on 1-15 cm for 30% of the rows and on 1-70 cm otherwise.
The probe bank of stage 60 is fit on it with the same rules (vlageom.distance.fit_bank,
split_half_ceilings), over the house -> fold map lists/folds/distance_plant.csv. For every
boundary c the stage records the mean, std, minimum and maximum over folds of the cosine between
the near and far probe weights, for the range-split pair (nr<c>, fr<c>) and the round-to-c pair
(cm<c>, far<c>), and the weaker probe's median split-half ceiling; and the mean |cosine| of the
m, nr15, fr15, cm15 and far15 weights with w.

Writes --out (default $VLAGEOM_DATA/distance/plant_cosines.json):
  {"plant", "fit", "grid_cm", "split": [{"c", "cos_mean", "cos_std_folds", "cos_min",
  "cos_max", "n_folds", "ceil_weaker_median"}], "round": [...], "cos_to_planted_axis", ...}

BLAS threads: 4 (--threads), the count of the paper's run; with the same library versions on an
x86-64 CPU with AVX-512 this gives the paper's values bit for bit (other counts change them at
the level of 1e-15). About 30 s on 4 free cores.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from vlageom import paths
from vlageom.distance import (BANK_ALPHA, GRID_CM, N_CEIL_DRAWS, PLANT_FOLDS, PLANT_SEED,
                              PLANT_THREADS, cosine, fit_bank, linear_plant, plant_pair_rows,
                              split_half_ceilings)
from vlageom.probes import N_FOLDS, FoldMap


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--folds", default=PLANT_FOLDS,
                    help=f"fold map of the synthetic houses (default {PLANT_FOLDS})")
    ap.add_argument("--threads", type=int, default=PLANT_THREADS,
                    help=f"BLAS threads (default {PLANT_THREADS})")
    ap.add_argument("--out", type=Path, default=paths.tile_dir("distance", "plant_cosines.json"))
    a = ap.parse_args()

    from threadpoolctl import threadpool_limits
    t0 = time.time()
    folds = FoldMap.load(a.folds)
    with threadpool_limits(limits=a.threads):
        X, d_cm, houses, w_sig = linear_plant()
        bank = fit_bank(X, d_cm, houses, folds.splits(houses))
        ceil = split_half_ceilings(X, d_cm, houses)
        w_raw = bank["w_raw"]
        to_axis = {k: float(np.mean([abs(cosine(w_raw[k][f], w_sig)) for f in range(N_FOLDS)
                                     if np.isfinite(w_raw[k][f]).all()]))
                   for k in ("m", "nr15", "fr15", "cm15", "far15")}
        out = {"plant": f"default_rng({PLANT_SEED}): X = 0.06*d_cm*w + 0.5*N(0,1) + house "
                        "offsets N(0,0.3^2); n=24000, d=256, 120 houses; one distance direction w",
               "fit": f"probe bank rules (alpha {BANK_ALPHA}, fold map {folds.name}); split-half "
                      f"ceilings ({N_CEIL_DRAWS} house-half draws)",
               "grid_cm": GRID_CM, "fold_map": folds.name, "fold_map_sha1": folds.sha1(),
               "threads": a.threads,
               "split": plant_pair_rows(w_raw, ceil, "nr", "fr"),
               "round": plant_pair_rows(w_raw, ceil, "cm", "far"),
               "cos_to_planted_axis": to_axis}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    for fam in ("split", "round"):
        print(fam, " ".join(f"c{r['c']}:{r['cos_mean']:.4f}/ceil{r['ceil_weaker_median']:.3f}"
                            for r in out[fam] if r["cos_mean"] is not None))
    print("cos to the planted direction:", {k: round(v, 4) for k, v in to_axis.items()})
    print(f"wrote {a.out}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
