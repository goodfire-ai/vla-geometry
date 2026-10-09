"""Wrist tile, stage 3 (CPU): pooled principal components of the 64 wrist sweeps -> panel file.

Takes token 0 of action-expert layer 9 from every sweep file (sorted path order), sorts each house's
rows by joint angle, subtracts that house's own sweep mean, and runs one SVD over all houses
(64 x 39 = 2,496 points). The trace is the sweep of one house (--trace-house, default house 0),
centered the same way and projected into the same basis. Files are left out when they are
unreadable, have a mean exo-camera pixel value below 20, fewer than 10 rows, or an angle range
below 300 deg (no paper sweep is left out).

Output: <data>/panels/wrist.npz with
  Z3 (39, 3), Z2 (39, 2)   trace in PC1-3 / PC1-2            float64
  roll (39,)               trace joint angle (deg)          float64
  evr (3,)                 explained-variance ratio, PC1-3  float64
  nat3 (2496, 3)           all pooled points in PC1-3       float32
  nat_roll (2496,)         their joint angles (deg)         float32
  house (2496,)            their houses                     int64
  n_houses                 number of pooled houses          int64
  files (64,)              pooled sweep files               str
  src                      path, size and modification time of every input file, centering, layer

  python pipelines/wrist/03_panel.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from vlageom import paths, wrist  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweeps-glob", default=None,
                    help="glob of sweep files (default <data>/wrist/sweeps/ep*.npz)")
    ap.add_argument("--trace", default=None,
                    help="sweep file drawn as the trace (default: the matched file of --trace-house)")
    ap.add_argument("--trace-house", type=int, default=0,
                    help="house whose sweep is the trace (default %(default)s)")
    ap.add_argument("--out", type=Path, default=None, help="output npz (default <data>/panels/wrist.npz)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    pattern = args.sweeps_glob or wrist.default_sweep_glob()
    files = wrist.sweep_files(pattern)
    if not files:
        raise SystemExit(f"no sweep files match {pattern}")
    trace = args.trace or wrist.find_trace(files, args.trace_house)
    panel = wrist.pooled_pca(files, trace)

    out = args.out or paths.panels_dir(create=True) / "wrist.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **panel)

    evr = panel["evr"]
    print(f"wrist panel -> {out}")
    print(f"  {panel['n_houses']} houses, {len(panel['nat3'])} points; trace {Path(trace).name}")
    print(f"  explained variance PC1-3: {np.round(100 * evr, 1).tolist()} %  "
          f"(axis labels PC1 {100 * evr[0]:.0f}%, PC2 {100 * evr[1]:.0f}%)")
    print(f"  trace angles {panel['roll'][0]:.1f} to {panel['roll'][-1]:.1f} deg; "
          f"note '{wrist.sweep_note(panel['roll'], panel['Z3'])}'")
    print(f"  trace end gap / path length in PC1-3: {wrist.end_gap_ratio(panel['Z3']):.3f} "
          f"(closed loop at <= {wrist.CLOSED_LOOP_MAX_GAP})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
