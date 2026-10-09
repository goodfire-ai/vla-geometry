"""Stage 30 (CPU): the two distance rulers, scored out of house.

Rows: the teleport store (target tile) followed by the near store (stage 20), without the near rows
that the teleport store already holds (see vlageom.distance.load_distance_rows), restricted to
distance d < 0.70 m and to houses of the fold map. Features: last_tok activations at VLM layer 21
(2560-d, float32). Targets (cm): far15 = max(d_cm, 15) and cm15 = min(d_cm, 15). For each of the 5
folds of lists/folds/distance.csv the features are standardized on the training rows and a ridge
probe (alpha 100) is fit on them; its predictions on the held-out houses form the out-of-fold
read-outs.

Writes --out (default $VLAGEOM_DATA/distance/ruler_probes.npz) with one entry per analysed row:
  house, source (0 teleport store, 1 near store), config_id, chunk_q, d_m (true distance, m),
  far15, cm15 (out-of-fold read-outs, cm)
and a .json next to it with the row counts and the scores: R^2 of far15 against the true distance
on rows with d >= 15 cm (the "m" score of Figure 1) and of cm15 on rows with d < 15 cm (the "cm"
score).

The scores and read-outs are reproducible to the last bit with the same library versions on an
x86-64 CPU with AVX-512 and one BLAS thread (--threads 1, the default). About 2 minutes and 4 GB of
memory at one thread.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from vlageom import paths
from vlageom.distance import (DIST_MAX, RULER_ALPHA, RULER_CELL, load_distance_rows, near_root,
                              ruler_scores, ruler_targets, teleport_prep)
from vlageom.probes import FoldMap, oof_ridge


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--teleport-store", type=Path, default=teleport_prep(),
                    help="packed object-teleport store of the target tile "
                         "(default: $VLAGEOM_DATA/target/teleport/prep)")
    ap.add_argument("--near-store", type=Path, default=near_root() / "prep",
                    help="packed near-range store of stage 20 "
                         "(default: $VLAGEOM_DATA/distance/near/prep)")
    ap.add_argument("--folds", default="distance",
                    help="fold map name in lists/folds/ or a house,fold CSV path "
                         "(default: distance)")
    ap.add_argument("--threads", type=int, default=1, help="BLAS threads (default 1)")
    ap.add_argument("--out", type=Path, default=paths.tile_dir("distance", "ruler_probes.npz"))
    a = ap.parse_args()

    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=a.threads):
        variant, layer = RULER_CELL
        X, lab = load_distance_rows(a.teleport_store, a.near_store, variant, layer)
        folds = FoldMap.load(a.folds)
        in_range = lab["dist"] < DIST_MAX
        mapped = folds.contains(lab["house"])
        gone = sorted(set(np.unique(lab["house"][in_range & ~mapped]).tolist()))
        keep = in_range & mapped
        Xg = np.ascontiguousarray(X[keep], np.float32)
        del X
        y = lab["dist"][keep].astype(np.float64)
        houses = lab["house"][keep]
        d_cm = y * 100
        print(f"rows: {len(y)} with d < {DIST_MAX} m ({int((lab['source'][keep] == 1).sum())} from "
              f"the near store), {len(np.unique(houses))} houses; fold map {folds.name} "
              f"sha1 {folds.sha1()[:12]}", flush=True)
        if gone:
            print(f"  {int((in_range & ~mapped).sum())} rows of {len(gone)} houses outside the "
                  f"fold map left out: {gone[:20]}", flush=True)
        oof = oof_ridge(Xg, ruler_targets(d_cm), houses, folds, alpha=RULER_ALPHA)

    sc = ruler_scores(d_cm, oof)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, house=houses.astype(np.int64), source=lab["source"][keep],
             config_id=lab["config_id"][keep], chunk_q=lab["chunk_q"][keep], d_m=y,
             far15=oof["far15"], cm15=oof["cm15"])
    side = {"cell": f"{variant}@L{layer}", "alpha": RULER_ALPHA, "dist_max_m": DIST_MAX,
            "n_rows": int(len(y)), "n_houses": int(len(np.unique(houses))),
            "n_near_store_rows": int((lab["source"][keep] == 1).sum()),
            "fold_map": folds.name, "fold_map_sha1": folds.sha1(),
            "houses_outside_fold_map": gone, "scores": sc,
            "teleport_store": str(a.teleport_store), "near_store": str(a.near_store),
            "out_sha1": hashlib.sha1(a.out.read_bytes()).hexdigest()}
    a.out.with_suffix(".json").write_text(json.dumps(side, indent=1))
    print(f"far15 (m ruler):  R^2 {sc['far15']['r2_above15']:.4f} on d >= 15 cm "
          f"({sc['n_above15']} rows), MAE {sc['far15']['mae_above15_cm']:.2f} cm")
    print(f"cm15 (cm ruler):  R^2 {sc['cm15']['r2_below15']:.4f} on d < 15 cm "
          f"({sc['n_below15']} rows), MAE {sc['cm15']['mae_below15_cm']:.2f} cm")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
