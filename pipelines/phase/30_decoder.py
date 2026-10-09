"""Phase, stage 30 (CPU): how well each action-expert layer separates the four phases.

On the successful natural queries of stage 20's table, a four-class LDA (StandardScaler + linear
discriminant analysis, both fit on the training houses of each fold) is scored out of house at each
of the 11 action-expert layers, over the folds of lists/folds/phase.csv (167 houses, 5 folds). The
score is the balanced accuracy (mean of the four per-phase recalls; chance 0.25). Every Figure 5
panel reads layer 24 (vlageom.phase.LAYER), the best layer of this scan (0.920 on the paper's
rollouts; layer 28 is next at 0.919).

Output: --out (default <data>/phase/decoder.json): per layer the balanced and raw accuracy, the
majority-class rate, per-phase recall and the confusion matrix; the best layer; row and house
counts; the sha1 of the fold map. The LDA fits run with 16 BLAS threads (--threads), the count of
the paper's scan: at layers 0 and 4 one or two of the 7,738 out-of-fold predictions depend on the
thread count. About 3-10 minutes.

  python pipelines/phase/30_decoder.py [--cache FILE] [--threads N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from vlageom import paths
from vlageom import phase as P
from vlageom.probes import FoldMap


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path,
                    default=paths.tile_dir("phase", "natural", "cache_ae.npz"),
                    help="natural table of stage 20 (default <data>/phase/natural/cache_ae.npz)")
    ap.add_argument("--folds", default="phase",
                    help="fold map name or house,fold CSV (default phase)")
    ap.add_argument("--threads", type=int, default=16, help="BLAS threads (default 16)")
    ap.add_argument("--out", type=Path, default=paths.tile_dir("phase", "decoder.json"))
    a = ap.parse_args(argv)

    import scipy.linalg  # noqa: F401  (load scipy's BLAS so that the limit below covers it)
    from threadpoolctl import threadpool_limits
    folds = FoldMap.load(a.folds)
    z = P.load_cache(a.cache, ["X", "label", "house", "success", "layers"])
    m = z["success"].astype(bool)
    X = z["X"][m]
    y = z["label"][m].astype(int)
    hs = z["house"][m].astype(int)
    layers = [int(v) for v in z["layers"]]
    out = {"n_rows": int(len(y)), "n_houses": int(len(np.unique(hs))), "folds": folds.name,
           "fold_map_sha1": folds.sha1(), "layers": layers, "per_layer": {}}
    with threadpool_limits(limits=a.threads):
        for li, L in enumerate(layers):
            r = P.lda_cv(X[:, li, :], y, hs, folds)
            out["per_layer"][str(L)] = r
            print(f"layer {L:2d}: balanced accuracy {r['bal']:.5f} (raw {r['raw']:.3f}, "
                  f"majority {r['maj']:.3f})", flush=True)
    best = max(layers, key=lambda L: out["per_layer"][str(L)]["bal"])
    out["best_layer"] = best
    print(f"best layer {best}: {out['per_layer'][str(best)]['bal']:.5f} on {out['n_rows']} rows, "
          f"{out['n_houses']} houses")
    if best != P.LAYER:
        print(f"note: the panels read layer {P.LAYER} (vlageom.phase.LAYER)")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
