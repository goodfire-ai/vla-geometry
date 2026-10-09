"""Compare a gripper panel file with a reference panel file, array by array.

Every key the panel stage writes (vlageom.gripper.PANEL_KEYS) is compared: same dtype, same shape
and np.array_equal; for numeric arrays that differ, the max |difference| is printed. Keys present
only in the reference are listed and not compared. Also prints the numbers the figure shows (the
axis-label percentages of PC1 and PC2) and the edit counts for both files.

  python pipelines/gripper/20_panel.py --edits <edit files> --out <scratch>/gripper.npz
  python tests/compare_gripper_panel.py --panel <scratch>/gripper.npz --reference <reference npz>

Exit status 0 when every compared array is identical.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from vlageom import gripper as G  # noqa: E402
from vlageom import paths  # noqa: E402


def compare(a: dict, b: dict) -> bool:
    ok = True
    for k in G.PANEL_KEYS:
        if k not in a or k not in b:
            print(f"  MISSING  {k} ({'panel' if k not in a else 'reference'})")
            ok = False
            continue
        x, y = np.asarray(a[k]), np.asarray(b[k])
        same = x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y)
        msg = f"  {'equal' if same else 'DIFF':8s} {k:7s} {str(x.dtype):8s} {x.shape}"
        if not same:
            ok = False
            if x.dtype != y.dtype or x.shape != y.shape:
                msg += f" vs {y.dtype} {y.shape}"
            if x.shape == y.shape and np.issubdtype(x.dtype, np.number):
                msg += f" max|diff| {np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))):.3e}"
        print(msg)
    for k in sorted(set(a) - set(G.PANEL_KEYS)):
        print(f"  extra    {k} (panel only; not compared)")
    ref_only = sorted(set(b) - set(G.PANEL_KEYS))
    if ref_only:
        print(f"  reference-only keys, not compared: {', '.join(ref_only)}")
    return ok


def shown(p: dict) -> str:
    evr, t = p["evr"], p["target"]
    return (f"PC1 {evr[0]:.0%} var, PC2 {evr[1]:.0%}; evr {np.round(evr, 5).tolist()}; "
            f"{len(t)} edits ({int((t > 0).sum())} close, {int((t < 0).sum())} open)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", type=Path, default=paths.panels_dir() / "gripper.npz",
                    help="panel file (default: $VLAGEOM_DATA/panels/gripper.npz)")
    ap.add_argument("--reference", type=Path, required=True, help="reference panel npz")
    args = ap.parse_args(argv)
    a = dict(np.load(args.panel, allow_pickle=False))
    b = dict(np.load(args.reference, allow_pickle=False))
    print(f"panel     {args.panel}\nreference {args.reference}")
    ok = compare(a, b)
    print(f"  shown (panel):     {shown(a)}")
    print(f"  shown (reference): {shown(b)}")
    print("IDENTICAL" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
