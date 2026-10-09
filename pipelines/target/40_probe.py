#!/usr/bin/env python
"""Out-of-house probe score of the target object position (CPU): the "R^2" printed in the target
tile of Figure 1.

Rows: the first query of every placement in the packed object-teleport store (13,145 rows,
906 houses). Features: VLM last-token activation at layer 21 (float32). Targets: object position
x (forward), y (left), z (up) in the robot base frame. Ridge probes on standardized features,
5 folds over houses from lists/folds/target_position.csv; in each fold the penalty (1e0..1e6) is
chosen per target on every 5th training house and the probe is refit on all training rows.
Score = pooled out-of-fold R^2 per axis; the printed value is their mean.

Output: JSON (default $VLAGEOM_DATA/target/probe/last_tok_L21.json) with per-axis R^2, per-fold
R^2, MAE (m), chosen penalties, `note_r2` (mean over x, y, z) and the store provenance.

Runs in one to five minutes. With the default 24 BLAS threads on an x86-64 CPU with AVX-512 the
result equals the paper's to the last bit; other thread counts or CPUs change it at the level of
floating-point rounding (the printed two decimals do not change).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from vlageom.probes import FoldMap
from vlageom.target import BLAS_THREADS, CELL, TeleportStore, probe_json_default, probe_scores
from vlageom.util import blas_threads


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", type=Path, default=None,
                    help="store root holding prep/ (default $VLAGEOM_DATA/target/teleport)")
    ap.add_argument("--folds", default="target_position",
                    help="fold map name in lists/folds/ or a house,fold CSV path")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"output JSON (default $VLAGEOM_DATA/target/probe/{CELL[0]}_L{CELL[1]}.json)")
    ap.add_argument("--blas-threads", type=int, default=BLAS_THREADS,
                    help="BLAS threads (default %(default)s, the paper's setting; 0 = library default)")
    args = ap.parse_args()
    _blas = blas_threads(args.blas_threads)  # noqa: F841  (kept for the whole run)

    store = TeleportStore(args.store)
    folds = FoldMap.load(args.folds)
    out = args.out or probe_json_default()
    t0 = time.time()
    res = probe_scores(store, folds)
    res["seconds"] = round(time.time() - t0, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    r = res["targets"]
    print(f"{res['cell']}: R^2 x {r['fwd']['r2']:.4f}  y {r['lat']['r2']:.4f}  z {r['up']['r2']:.4f}"
          f"  -> mean {res['note_r2']:.4f} (prints {res['note_printed']}); "
          f"{res['n_rows']} rows / {res['n_houses']} houses; fold map {folds.name} "
          f"sha1 {res['fold_map_sha1'][:12]}; wrote {out}", flush=True)


if __name__ == "__main__":
    main()
