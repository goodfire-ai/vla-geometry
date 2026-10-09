"""Compare one wrist sweep file (a canary run of pipelines/wrist/01_sweep.py) with a reference sweep
file of the same episode, key by key and row by row.

Every key the sweep stage writes (`vlageom.wrist.SWEEP_KEYS`) must be present in both files with
the same dtype and shape and identical values. Camera images (`frames`, `frames_wrist`) are
compared as uint8 pixels per row; `act` per layer and row, with the max |difference| when not
identical. Other keys of either file are listed and not compared.

  python pipelines/wrist/01_sweep.py --row 0 --out-dir <canary dir>
  python tests/canary_wrist_harvest.py --canary <canary dir>/ep0.npz --reference <reference ep0 file>

Bit-identical activations are expected only on the GPU type the reference was made on (bfloat16).
Exit status 0 when every key is identical.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from vlageom.wrist import SWEEP_KEYS

IMAGE_KEYS = ("frames", "frames_wrist")


def rows_equal_images(a, b):
    """(rows identical, rows total, differing pixels) for two per-row image arrays."""
    n_eq, n_px = 0, 0
    for x, y in zip(a, b):
        x8, y8 = np.asarray(x, np.uint8), np.asarray(y, np.uint8)
        if x8.shape == y8.shape and np.array_equal(x8, y8):
            n_eq += 1
        else:
            n_px += int(np.sum(x8 != y8)) if x8.shape == y8.shape else x8.size
    return n_eq, len(a), n_px


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--canary", type=Path, required=True, help="sweep file written by the canary run")
    ap.add_argument("--reference", type=Path, required=True, help="reference sweep file, same episode")
    args = ap.parse_args(argv)
    a = np.load(args.canary, allow_pickle=True)
    b = np.load(args.reference, allow_pickle=True)
    print(f"canary    {args.canary}\nreference {args.reference}")
    ok = True
    for k in SWEEP_KEYS:
        if k not in a.files or k not in b.files:
            print(f"  MISSING  {k} (in the {'canary' if k not in a.files else 'reference'})")
            ok = False
            continue
        x, y = a[k], b[k]
        head = f"{k:16s} {str(x.dtype):8s} {x.shape}"
        if x.dtype != y.dtype or x.shape != y.shape:
            print(f"  DIFF     {head} vs {y.dtype} {y.shape}")
            ok = False
            continue
        if k in IMAGE_KEYS:
            n_eq, n, n_px = rows_equal_images(x, y)
            same = n_eq == n
            print(f"  {'equal' if same else 'DIFF':8s} {head}  rows identical {n_eq}/{n}"
                  + ("" if same else f", {n_px} pixel values differ"))
        elif k == "act":
            same = np.array_equal(x, y)
            msg = f"  {'equal' if same else 'DIFF':8s} {head}"
            if not same:
                layers = [int(v) for v in b["layers"]]
                per = []
                for li, L in enumerate(layers):
                    d = np.abs(x[:, li].astype(np.float64) - y[:, li].astype(np.float64))
                    rows = int(np.sum(np.all(d.reshape(len(d), -1) == 0, axis=1)))
                    per.append(f"L{L}: max|diff| {d.max():.3e}, rows identical {rows}/{len(d)}")
                msg += "\n           " + "\n           ".join(per)
            print(msg)
        else:
            same = np.array_equal(x, y)
            msg = f"  {'equal' if same else 'DIFF':8s} {head}"
            if not same:
                if np.issubdtype(x.dtype, np.number):
                    msg += f" max|diff| {np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))):.3e}"
                else:
                    msg += f" '{x}' vs '{y}'"
            print(msg)
        ok = ok and same
    for tag, f in (("canary", a), ("reference", b)):
        other = sorted(set(f.files) - set(SWEEP_KEYS))
        if other:
            print(f"  other keys in the {tag} file (not compared): {other}")
    print("IDENTICAL" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
