#!/usr/bin/env python
"""Out-of-house read-outs of EE and object position (CPU): Figure 2a-c.

For each cell (VLM token group, layer), both variables are read out from the same rows with the
same folds: 5 folds over houses from lists/folds/position.csv; in each fold the activations are
standardized on the training rows and a RidgeCV (efficient leave-one-out over penalties
0.1..1e5) maps them to the (x, y, z) position. Reported per query (q1 = after one executed chunk,
the figure's rows; q0 = right after the move): pooled out-of-fold R^2 (variance weighted over the
three coordinates), per-axis R^2, R^2 at read-out rank 1, 2, 3 (rank 1 and 2 = the predictions
restricted to the top principal directions of the training predictions), the singular values of
the training predictions and their fractions, per-axis and 3-D absolute error in cm (mean, fold
std, median), the within-house R^2 (labels and predictions minus their house means), and label
baselines (house-mean R^2, within-house variance fraction, label std, and the R^2 of predicting
one variable's position from the other's).

The figure's cells are hdr_txt at layer 20 for the EE (the non-image tokens before the
instruction: the "Image N View M" labels, the image start and end markers and the BOS token) and
txt_instr at layer 21 for the object (the instruction tokens); among the VLM token groups at layers
20-21 these give the highest held-out R^2 for their variable. Any other cells can be scored with
--cells.

Output: one JSON per cell, $VLAGEOM_DATA/position/probe/{stream}_L{LL}.json. Both cells take
about 15-35 min on 8 cores (8 BLAS threads): about 15 min of fits, plus the first read of the
33 GB packed store, which takes up to about 20 min from disk. With the default 8 BLAS threads on an
x86-64 CPU with AVX-512 the results equal the paper's bit for bit.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from vlageom.position import (BLAS_THREADS, CELLS, FOLD_MAP, PositionStore, probe_dir,
                              probe_json_path, probe_stream_layer)
from vlageom.probes import FoldMap
from vlageom.util import blas_threads


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--store", type=Path, default=None,
                    help="crossed store root holding prep/ "
                         "(default $VLAGEOM_DATA/position/crossed)")
    ap.add_argument("--cells", nargs="+",
                    default=[f"{s}:{L}" for s, L in (CELLS["ee"], CELLS["obj"])],
                    help="STREAM:LAYER cells (default: %(default)s)")
    ap.add_argument("--grains", default="q1,q0", help="queries, comma separated (default q1,q0)")
    ap.add_argument("--folds", default=FOLD_MAP, help="fold map name in lists/folds/ or a CSV path")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output directory (default $VLAGEOM_DATA/position/probe)")
    ap.add_argument("--blas-threads", type=int, default=BLAS_THREADS,
                    help="BLAS threads (default %(default)s, the paper's setting; "
                         "0 = library default)")
    ap.add_argument("--overwrite", action="store_true", help="recompute cells whose JSON exists")
    args = ap.parse_args()
    _blas = blas_threads(args.blas_threads)  # noqa: F841  (kept for the whole run)

    store = PositionStore(args.store)
    folds = FoldMap.load(args.folds)
    out_dir = args.out_dir or probe_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    grains = [g.strip() for g in args.grains.split(",") if g.strip()]
    for cell in args.cells:
        stream, layer = cell.split(":")
        dst = probe_json_path(stream, int(layer), out_dir)
        if dst.exists() and not args.overwrite:
            print(f"{dst} exists, skipped (--overwrite to recompute)", flush=True)
            continue
        res = probe_stream_layer(store, folds, stream, int(layer), grains,
                                 log=lambda s: print(s, flush=True))
        res["fold_map"] = folds.name
        res["fold_map_sha1"] = folds.sha1()
        res["store"] = store.provenance()
        tmp = dst.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(res, indent=1, default=float))
        os.replace(tmp, dst)
        print(f"wrote {dst}", flush=True)


if __name__ == "__main__":
    main()
