"""Compare a distance panel (pipelines/distance/40_panel.py output) with a reference panel file.

Checks every drawn array of the reference (pts, d, path, path_d, cross) for equal dtype, shape and
values (bit for bit; the maximum absolute difference is printed otherwise), and the panel's R^2
values against the scores printed in the paper's Figure 1 (0.81 for the metre ruler, 0.70 for the
centimetre ruler) and, with --ref-scores, against exact reference values. Keys only in one file
(such as the provenance record `src`) are listed but not compared.

Usage:
  python tests/compare_distance_panel.py --reference <reference panel .npz> \
      [--panel $VLAGEOM_DATA/panels/distance.npz] [--ref-scores R2_M R2_CM]
Exit code 0 when everything matches, 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DRAWN = ["pts", "d", "path", "path_d", "cross"]
PRINTED = {"r2_m": "0.81", "r2_cm": "0.70"}


def main():
    from vlageom import paths

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reference", "--ref", dest="ref", type=Path, required=True,
                    help="reference panel .npz")
    ap.add_argument("--panel", "--new", dest="new", type=Path,
                    default=paths.panels_dir() / "distance.npz",
                    help="panel to check (default: $VLAGEOM_DATA/panels/distance.npz)")
    ap.add_argument("--ref-scores", type=float, nargs=2, metavar=("R2_M", "R2_CM"),
                    help="exact reference R^2 values (metre ruler, centimetre ruler)")
    a = ap.parse_args()

    new = np.load(a.new, allow_pickle=False)
    ref = np.load(a.ref, allow_pickle=False)
    ok = True
    for k in DRAWN:
        if k not in ref.files or k not in new.files:
            print(f"  FAIL {k}: missing in {'reference' if k not in ref.files else 'new panel'}")
            ok = False
            continue
        x, y = new[k], ref[k]
        same = x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y)
        if same:
            print(f"  ok   {k:7s} {str(x.dtype):8s} {x.shape} bit-identical")
        else:
            diff = (float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
                    if x.shape == y.shape else float("nan"))
            print(f"  FAIL {k:7s} new {x.dtype} {x.shape} / ref {y.dtype} {y.shape}; "
                  f"max |diff| {diff:.3g}")
            ok = False
    only_new = sorted(set(new.files) - set(ref.files))
    only_ref = sorted(set(ref.files) - set(new.files))
    other = sorted(set(new.files) & set(ref.files) - set(DRAWN))
    print(f"  keys not compared: both {other}, only new {only_new}, only reference {only_ref}")

    for i, (k, want) in enumerate(PRINTED.items()):
        if k not in new.files:
            print(f"  FAIL {k}: missing")
            ok = False
            continue
        v = float(new[k])
        good = f"{v:.2f}" == want
        line = f"  {'ok  ' if good else 'FAIL'} {k:7s} {v:.10f} -> '{v:.2f}' (paper prints {want})"
        if a.ref_scores is not None:
            exact = v == a.ref_scores[i]
            good &= exact
            line += f"; reference {a.ref_scores[i]!r}: {'equal' if exact else 'DIFFERENT'}"
        print(line)
        ok &= good
    if "note" in new.files:
        print(f"  note: {str(new['note'])}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
