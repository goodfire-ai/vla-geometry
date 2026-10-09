#!/usr/bin/env python
"""Equivariance of the read-out subspaces across layers (CPU): Figure 2d.

Within every house the crossed store holds pairs of states that differ in one variable only: two
arm poses with the same object placement (EE pairs, 3-D displacement) and two object placements
with the same arm pose (target pairs; the object moves only in x and y, so the displacement is
2-D). Pairs are formed against an anchor: the smallest pose_id of each (house, placement) for EE
pairs, placement 0 of each (house, pose) for target pairs.

Per layer, query and fold (lists/folds/position.csv): StandardScaler + RidgeCV on the training
rows give the rank-3 read-out subspace (first 3 columns of the QR factor of the weights); each
pair's activation displacement is projected on it. A general linear map and a rotation with a
uniform scale (orthogonal Procrustes) from world to activation displacement are fit on the
training pairs and scored on the test pairs by the median relative residual |dh - map(dp)| / |dh|
(mean over folds). Reference: world displacements permuted over all pairs, the linear map refit
on the permuted training pairs (about 1.0, no relation). EE pairs use the EE cell's token group
(hdr_txt), target pairs the target cell's (txt_instr), at every layer.

Work split: --task K --n-tasks N takes layers K, K+N, K+2N, ... (the paper: 12 tasks of 3 layers);
--layers L [L ...] runs only those layers. The permutations of the reference come from one numpy
generator (seed 52) per task, drawn in layer order with the queries in --grains order (q1, then
q0), so its values depend on the split into tasks; nothing else does.

Output: $VLAGEOM_DATA/position/procrustes/parts/{grain}_L{LL}.json, then with --merge
procrustes/procrustes_{grain}.json (the figure reads procrustes_q0.json). About 6-8 min per layer
and query on 8 cores; 36 layers x 2 queries about 8 CPU-hours.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from vlageom.position import (BLAS_THREADS, CELLS, FOLD_MAP, PositionStore, procrustes_dir,
                              procrustes_layer, procrustes_merge)
from vlageom.probes import FoldMap
from vlageom.util import blas_threads


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--store", type=Path, default=None,
                    help="crossed store root holding prep/ "
                         "(default $VLAGEOM_DATA/position/crossed)")
    ap.add_argument("--task", type=int, default=None,
                    help="task index (default $SLURM_ARRAY_TASK_ID; all layers if neither is set)")
    ap.add_argument("--n-tasks", type=int, default=12)
    ap.add_argument("--layers", type=int, nargs="+", default=None, help="run only these layers")
    ap.add_argument("--grains", default="q1,q0")
    ap.add_argument("--ee-stream", default=CELLS["ee"][0])
    ap.add_argument("--obj-stream", default=CELLS["obj"][0])
    ap.add_argument("--folds", default=FOLD_MAP)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default $VLAGEOM_DATA/position/procrustes (parts/ below it)")
    ap.add_argument("--merge", action="store_true",
                    help="merge the part files into procrustes_{grain}.json and exit")
    ap.add_argument("--overwrite", action="store_true", help="recompute layers whose part exists")
    ap.add_argument("--blas-threads", type=int, default=BLAS_THREADS,
                    help="BLAS threads (default %(default)s, the paper's setting; "
                         "0 = library default)")
    args = ap.parse_args()
    grains = [g.strip() for g in args.grains.split(",") if g.strip()]
    out_dir = args.out_dir or procrustes_dir()
    parts = out_dir / "parts"
    if args.merge:
        procrustes_merge(parts, out_dir, grains, log=lambda s: print(s, flush=True))
        return
    _blas = blas_threads(args.blas_threads)  # noqa: F841

    store = PositionStore(args.store)
    meta = store.meta()
    folds = FoldMap.load(args.folds)
    if args.layers:
        layers = list(args.layers)
    else:
        task = args.task
        if task is None and os.environ.get("SLURM_ARRAY_TASK_ID", "") != "":
            task = int(os.environ["SLURM_ARRAY_TASK_ID"])
        layers = list(meta["layers_llm"]) if task is None else \
            [L for i, L in enumerate(meta["layers_llm"]) if i % args.n_tasks == task]
    parts.mkdir(parents=True, exist_ok=True)
    lab = store.labels()
    streams = {"ee": args.ee_stream, "obj": args.obj_stream}
    rng = np.random.default_rng(52)
    t0 = time.time()
    print(f"layers {layers}, queries {grains}", flush=True)
    for L in layers:
        X = {s: store.layer_array(s, L, meta) for s in set(streams.values())}
        for g in grains:
            part = parts / f"{g}_L{int(L):02d}.json"
            if part.exists() and not args.overwrite:
                print(f"{part.name} exists, skipped", flush=True)
                continue
            res = procrustes_layer(X, lab, g, folds, streams, rng)
            tmp = part.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"layer": int(L), "grain": g, "vars": res}, indent=1,
                                      default=float))
            os.replace(tmp, part)
            e, o = res["ee"], res["obj"]
            print(f"L{L} {g}: EE resid {e['resid_ho']:.3f} (orthogonal {e['ortho_resid']:.3f}, "
                  f"shuffled {e['shuffle_resid']:.3f}, {e['n_pairs']} pairs) | target resid "
                  f"{o['resid_ho']:.3f} (orthogonal {o['ortho_resid']:.3f}, shuffled "
                  f"{o['shuffle_resid']:.3f}, {o['n_pairs']} pairs) "
                  f"[{(time.time() - t0) / 60:.1f} min]",
                  flush=True)
        del X
    print("done", flush=True)


if __name__ == "__main__":
    main()
