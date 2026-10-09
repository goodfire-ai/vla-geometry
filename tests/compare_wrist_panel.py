"""Compare a wrist panel file with a reference panel file, array by array.

For every key of the reference: same dtype, same shape and np.array_equal; for numeric arrays that
differ, the max |difference| is printed. Also prints the numbers the figure shows (axis-label
percentages, colorbar extremes, sweep note) for both files.

  python tests/compare_wrist_panel.py --reference <reference npz> [--panel <data>/panels/wrist.npz]
      [--ignore files,src]     # path-dependent keys, when the inputs were read from another location

Exit status 0 when every compared array is identical.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from vlageom import paths, wrist  # noqa: E402


def compare(a: dict, b: dict, ignore=()) -> bool:
    ok = True
    keys = sorted(set(a) | set(b))
    for k in keys:
        if k not in b:
            print(f"  extra    {k} (not in reference)")
            ok = False
            continue
        if k not in a:
            print(f"  MISSING  {k}")
            ok = False
            continue
        x, y = a[k], b[k]
        same = x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y)
        tag = "equal" if same else ("ignored" if k in ignore else "DIFF")
        msg = f"  {tag:8s} {k:9s} {str(x.dtype):8s} {x.shape}"
        if not same:
            if x.dtype != y.dtype or x.shape != y.shape:
                msg += f" vs {y.dtype} {y.shape}"
            elif np.issubdtype(x.dtype, np.number):
                msg += f" max|diff| {np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))):.3e}"
            if k not in ignore:
                ok = False
        print(msg)
    return ok


def shown_numbers(p: dict) -> str:
    evr, roll = p["evr"], p["roll"]
    return (f"PC1 {100 * evr[0]:.0f}% var, PC2 {100 * evr[1]:.0f}%, colorbar {roll.min():.0f} to "
            f"{roll.max():.0f}, note '{wrist.sweep_note(roll, p['Z3'])}'")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", type=Path, default=paths.panels_dir() / "wrist.npz",
                    help="panel to check (default: $VLAGEOM_DATA/panels/wrist.npz)")
    ap.add_argument("--reference", type=Path, required=True, help="reference panel npz")
    ap.add_argument("--ignore", default="", help="comma-separated keys whose difference is reported only")
    args = ap.parse_args(argv)
    a = dict(np.load(args.panel, allow_pickle=False))
    b = dict(np.load(args.reference, allow_pickle=False))
    print(f"panel     {args.panel}\nreference {args.reference}")
    ok = compare(a, b, ignore={s for s in args.ignore.split(",") if s})
    print(f"  shown (panel):     {shown_numbers(a)}")
    print(f"  shown (reference): {shown_numbers(b)}")
    print("IDENTICAL" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
