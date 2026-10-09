#!/usr/bin/env python
"""Panel arrays of the two trace grids of Figure 2 (CPU): $VLAGEOM_DATA/panels/fig2_ee_grid.npz
and fig2_target_grid.npz.

For each variable, the rank-3 read-out subspace is fit on the q0 rows of the crossed store
without the shape house (house 44): StandardScaler + RidgeCV (penalties 0.1..1e5) from the cell's
activations (EE: hdr_txt, layer 20; target: txt_instr, layer 21) to the position, basis = first
3 columns of the QR factor of the weights (`vlageom.position.grid_chart`). Each of the house's
five paths (80_hand_shapes.py / 85_object_shapes.py) is projected into it (centred and divided by
the 95th-percentile radius of the fit rows), rotated so the axes line up with the decoded forward,
left and up directions, and, like the realized path, normalized to a unit floor (`normf`).
Consecutive waypoints closer than 1 mm are merged.

Colours (x -> red, y -> green, z -> blue, each the empirical CDF of that coordinate):
  EE grid      over the 4,000 reference hand positions of lists/position/ee_color_reference.csv
               (hand positions at the model queries of natural MolmoBot-DROID rollouts), rounded
               to 3 decimals
  target grid  over the object positions of the subspace's fit rows

Output keys: kinds, labels, closed, n_points, world_{i}, sub_{i}, rgb_{i} (i = column 0..4).
The fits are also written to $VLAGEOM_DATA/position/grids/chart_{ee,obj}.npz with their
statistics (chart_{ee,obj}.json: penalty, rows, houses, in-sample R^2, mean absolute error on the
held-out house). About 1-3 min per variable on 8 cores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vlageom import paths
from vlageom.position import (BLAS_THREADS, CELLS, SHAPE_HOUSE, PositionStore, TARGETS, grid_chart,
                              grid_panel, shapes_root)
from vlageom.util import blas_threads

NAMES = {"ee": "fig2_ee_grid", "obj": "fig2_target_grid"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--variable", choices=["ee", "obj", "both"], default="both")
    ap.add_argument("--store", type=Path, default=None,
                    help="crossed store root holding prep/ "
                         "(default $VLAGEOM_DATA/position/crossed)")
    ap.add_argument("--shapes", type=Path, default=None,
                    help="shapes root (default $VLAGEOM_DATA/position/shapes)")
    ap.add_argument("--house", type=int, default=SHAPE_HOUSE)
    ap.add_argument("--panels", type=Path, default=None, help="default $VLAGEOM_DATA/panels")
    ap.add_argument("--chart-dir", type=Path, default=None,
                    help="where the fits are written (default $VLAGEOM_DATA/position/grids)")
    ap.add_argument("--blas-threads", type=int, default=BLAS_THREADS,
                    help="BLAS threads (default %(default)s; 0 = library default)")
    args = ap.parse_args()
    _blas = blas_threads(args.blas_threads)  # noqa: F841

    store = PositionStore(args.store)
    pdir = args.panels or paths.panels_dir(create=True)
    pdir.mkdir(parents=True, exist_ok=True)
    cdir = args.chart_dir or paths.tile_dir("position", "grids", create=True)
    cdir.mkdir(parents=True, exist_ok=True)
    for var in (("ee", "obj") if args.variable == "both" else (args.variable,)):
        stream, layer = CELLS[var]
        chart = grid_chart(store, stream, layer, TARGETS[var], args.house)
        np.savez(cdir / f"chart_{var}.npz", **{k: v for k, v in chart.items()
                                              if isinstance(v, np.ndarray) or np.isscalar(v)})
        side = {"cell": chart["cell"], "exclude_house": chart["exclude_house"],
                "fitstats": chart["fitstats"], "store": store.provenance()}
        (cdir / f"chart_{var}.json").write_text(json.dumps(side, indent=1))
        print(f"[{var}] subspace {chart['cell']} without house {args.house}: {chart['fitstats']}",
              flush=True)
        panel = grid_panel(chart, args.shapes or shapes_root(), var, args.house)
        panel["src"] = np.array(json.dumps({"store": str(store.prep),
                                            "shapes": str(args.shapes or shapes_root())}))
        out = pdir / f"{NAMES[var]}.npz"
        np.savez(out, **panel)
        print(f"[{var}] wrote {out}: paths {panel['kinds'].tolist()}, waypoints "
              f"{panel['n_points'].tolist()}", flush=True)


if __name__ == "__main__":
    main()
