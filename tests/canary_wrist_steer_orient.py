"""Compare one orientation sweep file (a canary run of pipelines/wrist_steer/30_orient_sweep.py) with a
reference sweep file of the same episode, key by key and row by row.

Every key of `vlageom.wrist_steer.ORIENT_KEYS` must be present in both files with the same dtype and
shape and identical values; for `act` the max |difference| and the identical rows are printed per
layer when it differs. Other keys of either file are listed and not compared.

  python pipelines/wrist_steer/30_orient_sweep.py --task 0 --out-dir $C/orient
  python tests/canary_wrist_steer_orient.py --canary $C/orient/ep0_h0.npz --reference <orient store>/ep0_h0.npz

Bit-identical activations are expected only on the GPU type of the reference (bfloat16).
Exit status 0 when every key is identical.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from vlageom.wrist_steer import ORIENT_KEYS


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--canary", type=Path, required=True, help="sweep file written by the canary run")
    ap.add_argument("--reference", type=Path, required=True, help="reference sweep file, same episode")
    args = ap.parse_args(argv)
    a = np.load(args.canary, allow_pickle=True)
    b = np.load(args.reference, allow_pickle=True)
    print(f"canary    {args.canary}\nreference {args.reference}")
    ok = True
    for k in ORIENT_KEYS:
        if k not in a.files or k not in b.files:
            print(f"  MISSING  {k} (in the {'canary' if k not in a.files else 'reference'})")
            ok = False
            continue
        x, y = a[k], b[k]
        head = f"{k:10s} {str(x.dtype):8s} {x.shape}"
        if x.dtype != y.dtype or x.shape != y.shape:
            print(f"  DIFF     {head} vs {y.dtype} {y.shape}")
            ok = False
            continue
        eq = bool(np.array_equal(x, y))
        msg = f"  {'equal' if eq else 'DIFF':8s} {head}"
        if not eq and k == "act":
            layers = [int(v) for v in b["layers"]]
            per = []
            for li, L in enumerate(layers):
                d = np.abs(x[:, li].astype(np.float64) - y[:, li].astype(np.float64))
                rows = int(np.sum(np.all(d == 0, axis=1)))
                per.append(f"L{L}: max|diff| {d.max():.3e}, rows identical {rows}/{len(d)}")
            msg += "\n           " + "\n           ".join(per)
        elif not eq and np.issubdtype(x.dtype, np.number):
            msg += f" max|diff| {np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))):.3e}"
        print(msg)
        ok = ok and eq
    for tag, f in (("canary", a), ("reference", b)):
        other = sorted(set(f.files) - set(ORIENT_KEYS))
        if other:
            print(f"  other keys in the {tag} file (not compared): {other}")
    print("IDENTICAL" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
