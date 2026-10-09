"""Stage 20 (CPU): out-of-house ridge probes of the decoded gripper command (Figure 7b).

Rows: the edit files of the listed episodes in --edits, in file-name order; per row the layer-32
activation without the edit (base_act, 768) and the decoded gripper command before the edit
(grip_before, chunk mean, -1 open .. +1 closed). Two probes, both scikit-learn Ridge(alpha=10)
scored by the pooled out-of-fold R^2 over the 5 house folds of lists/folds/gripper_probe.csv:

  curve  the activation minus the basis mean, projected onto the top k principal components of
         stage 10's basis, for k = 1, 2, 4, ..., 256, 768
  line   the full 768-D activation

Writes --out (default $VLAGEOM_DATA/gripper_fig7/probe.json): r2_by_k {k: R^2}, r2_full,
pearson_full, the row and house counts, alpha and the fold map's sha1. With the paper's edits:
1,666 rows, 170 houses, R^2 0.860 at k = 1, 0.9989 at k = 768, full activation 0.9989. Takes about
20 s.

With the pinned library versions on an x86-64 CPU with AVX-512 and 8 BLAS threads (--threads 8,
the default) the scores equal the paper's to the last bit. The folds come from the shipped map, not
from scikit-learn's GroupKFold at run time, whose order of equally sized houses depends on the CPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vlageom import gripper_fig7 as F
from vlageom import paths
from vlageom.probes import FoldMap


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n", 1)[1])
    ap.add_argument("--edits", type=Path, default=paths.tile_dir("gripper", "deltas"),
                    help="directory of edit files (default: $VLAGEOM_DATA/gripper/deltas)")
    ap.add_argument("--episodes", type=Path, default=None,
                    help="episode list CSV (default lists/gripper_episodes.csv)")
    ap.add_argument("--basis", type=Path, default=None,
                    help="stage 10 output (default: $VLAGEOM_DATA/gripper_fig7/basis.npz)")
    ap.add_argument("--folds", default=F.PROBE_FOLDS,
                    help="fold map name under lists/folds/ or a house,fold CSV "
                         "(default %(default)s)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSON (default: $VLAGEOM_DATA/gripper_fig7/probe.json)")
    ap.add_argument("--threads", type=int, default=8, help="BLAS threads (default 8)")
    a = ap.parse_args()

    from threadpoolctl import threadpool_limits

    from vlageom import gripper as G
    E = F.load_listed_edits(a.edits, G.read_episodes(a.episodes))
    H, aper, house = E["H"], E["aper"], E["house"]
    basis_path = a.basis or paths.tile_dir(F.TILE, "basis.npz")
    with np.load(basis_path) as z:
        mu, V = z["mu"].astype(np.float32), z["V"].astype(np.float32)
    folds = FoldMap.load(a.folds)
    splits = folds.splits(house)
    with threadpool_limits(limits=a.threads):
        r2_by_k = F.topk_probe(H, aper, splits, mu, V)
        full = F.full_probe(H, aper, splits)

    out = a.out or paths.tile_dir(F.TILE, create=True) / "probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = dict(n_rows=int(len(H)), n_houses=int(len(np.unique(house))), alpha=F.RIDGE_ALPHA,
               folds=folds.name, folds_sha1=folds.sha1(),
               r2_by_k={str(k): float(v) for k, v in r2_by_k.items()}, **full)
    out.write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(dict(rows=rec["n_rows"], houses=rec["n_houses"],
                          r2_k1=round(r2_by_k[1], 5), r2_k768=round(r2_by_k[768], 5),
                          r2_full=round(full["r2_full"], 7), out=str(out))))


if __name__ == "__main__":
    main()
