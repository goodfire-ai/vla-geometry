#!/usr/bin/env python
"""Curvature of straight world paths inside the read-out subspaces (CPU): Figure 2e.

Paths:
  EE      the 4 straight hand lines per house of 50_hand_lines.py (47 houses with a line of at
          least 12 waypoints; 188 lines), activations of the EE cell (hdr_txt, layer 20)
  target  the straight object lines per house of 55_object_lines.py (forward, sideways and
          toward-the-hand lines; 120 lines in 43 houses), activations of the target cell
          (txt_instr, layer 21)
Subspace per variable: StandardScaler + RidgeCV from the cell's activations to the position on
the q0 rows of the crossed store, without the houses that have paths (the paths are held out);
basis = first 3 columns of the QR factor of the weights.

Per path: activations standardized and projected on the basis, path and projection centred;
straightness, mean turning angle and general-linear / orthogonal / affine residuals of the world
path vs its projection. Local turning angle at every interior waypoint between the segments to
the waypoint `stride` steps before and after (stride = 7 cm / median waypoint spacing, at least
2), for the projection, the world path and 8 seeded random orthonormal 3-D projections of the
same activations (median over the 8; seed 52 for EE, 37 for the target). The angles are pooled
over paths and binned by that waypoint's hand-object distance into 6 quantile bins; per bin the
median angle in degrees (the figure's solid line: subspace; dashed: random projections).

Output: $VLAGEOM_DATA/position/curvature/curvature_q0.json (per variable: bins, path medians,
every path's measures, the number of rows the subspace was fit on), and the two subspaces
(chart_ee.npz, chart_obj.npz: scaler mean and scale, Q, rows). About 5-15 min on 8 cores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vlageom.position import (BLAS_THREADS, CELLS, CHART_GRAIN, RAND_SEED, PositionStore,
                              curvature, curvature_json_default, fit_subspace, hand_lines_root,
                              load_hand_lines, load_object_lines,
                              object_lines_root)
from vlageom.util import blas_threads


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--store", type=Path, default=None,
                    help="crossed store root holding prep/ "
                         "(default $VLAGEOM_DATA/position/crossed)")
    ap.add_argument("--hand-lines", type=Path, default=None,
                    help="hand-line directory (default $VLAGEOM_DATA/position/hand_lines)")
    ap.add_argument("--object-lines", type=Path, default=None,
                    help="object-line root (default $VLAGEOM_DATA/position/object_lines)")
    ap.add_argument("--grain", default=CHART_GRAIN, help="store rows the subspaces are fit on")
    ap.add_argument("--ee-cell", default=f"{CELLS['ee'][0]}:{CELLS['ee'][1]}")
    ap.add_argument("--obj-cell", default=f"{CELLS['obj'][0]}:{CELLS['obj'][1]}")
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSON (default $VLAGEOM_DATA/position/curvature/curvature_q0.json)")
    ap.add_argument("--blas-threads", type=int, default=BLAS_THREADS,
                    help="BLAS threads (default %(default)s, the paper's setting; "
                         "0 = library default)")
    args = ap.parse_args()
    _blas = blas_threads(args.blas_threads)  # noqa: F841

    store = PositionStore(args.store)
    es, eL = args.ee_cell.split(":")
    os_, oL = args.obj_cell.split(":")
    eL, oL = int(eL), int(oL)
    ee_paths = load_hand_lines(args.hand_lines or hand_lines_root(), es, eL)
    obj_paths = load_object_lines(args.object_lines or object_lines_root(), os_, oL)
    print(f"EE paths {len(ee_paths)} ({len({p['house'] for p in ee_paths})} houses) | target paths "
          f"{len(obj_paths)} ({len({p['house'] for p in obj_paths})} houses)", flush=True)
    out = args.out or curvature_json_default()
    out.parent.mkdir(parents=True, exist_ok=True)
    R = {"grain_for_charts": args.grain, "cells": {"ee": args.ee_cell, "obj": args.obj_cell}}
    for var, paths_, (s, L), key in (("ee", ee_paths, (es, eL), "ee"),
                                     ("obj", obj_paths, (os_, oL), "obj_base")):
        if not paths_:
            continue
        excl = sorted({p["house"] for p in paths_})
        sc, Q, n = fit_subspace(store, s, L, key, args.grain, excl)
        np.savez(out.parent / f"chart_{var}.npz", mean=sc.mean_, scale=sc.scale_, Q=Q, n=n,
                 excluded_houses=np.array(excl, np.int64))
        R[var] = curvature(paths_, sc, Q, RAND_SEED[var])
        R[var]["chart_rows"] = n
    out.write_text(json.dumps(R, indent=1, default=float))
    for v in ("ee", "obj"):
        if v in R:
            b = R[v]
            print(f"{v}: {b['n_paths']} paths / {b['n_houses']} houses; local turning angle "
                  f"{b['overall_curv_med_deg']:.1f} deg vs random projections "
                  f"{b['overall_rand_med_deg']:.1f} deg", flush=True)
            for x in b["bins"]:
                print(f"   d {x['d_lo'] * 100:5.1f}-{x['d_hi'] * 100:5.1f} cm  n={x['n']:4d}  "
                      f"subspace {x['curv_med_deg']:5.1f}  random {x['rand_proj_med_deg']:5.1f}  "
                      f"world {x['world_med_deg']:4.1f}", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
