"""Contact (Figure 4), stage 50 (CPU): out-of-house LDA probe of contact on the natural rows.

For each requested (pool, layer) cell of the natural cache (default: txt_all and last_tok at layer
24): per fold of lists/folds/contact_natural.csv (5 folds over the 192 houses), StandardScaler and
scikit-learn LinearDiscriminantAnalysis (svd solver) are fit on the rows of the training houses;
the held-out rows get the decision score and LDA's own prediction (threshold from the training
class priors). Reported per cell: acc (fraction of rows predicted correctly; the "baseline" line of
panel b and the "91.3%" of the text for txt_all at layer 24), auc (ROC-AUC of the scores) and
bal_acc (balanced accuracy); plus the majority-class accuracy max(p, 1 - p) of the rows.

BLAS runs on one thread (--threads); scores depend on the thread count in the last bits.

Output: $VLAGEOM_DATA/contact/natural_probe.json (--out). About 2 minutes per cell.

  python pipelines/contact/50_natural_probe.py [--cache FILE] [--cells txt_all@24 last_tok@24]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from vlageom import contact as C  # noqa: E402
from vlageom import paths  # noqa: E402
from vlageom.probes import FoldMap  # noqa: E402

FOLDS = "contact_natural"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n", 1)[1])
    ap.add_argument("--cache", type=Path, default=None,
                    help="natural cache (default $VLAGEOM_DATA/contact/natural_cache.npz)")
    ap.add_argument("--cells", nargs="+", default=[f"{p}@{C.PROBE_LAYER}" for p in C.POOLS],
                    help="pool@layer cells (default txt_all@24 last_tok@24)")
    ap.add_argument("--folds", default=FOLDS, help="fold map name or CSV (default contact_natural)")
    ap.add_argument("--threads", type=int, default=1, help="BLAS threads (default 1)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSON (default $VLAGEOM_DATA/contact/natural_probe.json)")
    a = ap.parse_args(argv)
    import scipy.linalg  # noqa: F401  (load scipy's BLAS so that the limit below covers it)
    from threadpoolctl import threadpool_limits

    cache = a.cache or paths.tile_dir(C.TILE) / "natural_cache.npz"
    out = a.out or paths.tile_dir(C.TILE) / "natural_probe.json"
    with np.load(cache, mmap_mode="r") as z:
        y = np.asarray(z["contact"]).astype(int)
        house = np.asarray(z["house"])
    folds = FoldMap.load(a.folds)
    print(f"[50] {len(y)} rows, {len(np.unique(house))} houses, contact on {y.mean():.4f}; folds "
          f"{folds.name} (sha1 {folds.sha1()[:12]})", flush=True)
    t0 = time.time()
    cells = {}
    with threadpool_limits(limits=a.threads):
        for cell in a.cells:
            pool, layer = cell.split("@")
            X = C.load_layer(cache, f"X_{pool}", int(layer))
            r = C.oof_lda(X, y, house, folds)
            cells[cell] = {"pool": pool, "layer": int(layer), "acc": r["acc"], "auc": r["auc"],
                           "bal_acc": r["bal_acc"]}
            print(f"[50] {cell}: acc {r['acc']:.4f}, auc {r['auc']:.4f}, balanced acc "
                  f"{r['bal_acc']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    res = {"cache": str(cache), "n": int(len(y)), "n_houses": int(len(np.unique(house))),
           "contact_on_frac": float(y.mean()),
           "majority_acc": float(max(y.mean(), 1.0 - y.mean())),
           "folds": folds.name, "folds_sha1": folds.sha1(), "threads": a.threads, "cells": cells}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"[DONE] majority {res['majority_acc']:.4f} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
