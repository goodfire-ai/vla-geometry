"""Compare two .npz files array by array (the Figure 5 natural cache, a natural episode file, a
freeze-test file, the trajectory capture or a Figure 5 panel file).

For every key: present in both, same dtype, same shape and equal values (np.array_equal; NaN equal
to NaN for float arrays). For keys that differ, the largest absolute difference (numeric arrays) or
the number of differing entries is printed.

  python tests/compare_phase_arrays.py NEW REFERENCE [--ignore KEY ...]

Exit status 0 when every compared key is identical.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def compare_npz(new: Path, ref: Path, ignore=()) -> list[str]:
    """Differences between two npz files, one line each (empty if identical)."""
    a = np.load(new, allow_pickle=True)
    b = np.load(ref, allow_pickle=True)
    out = []
    for k in sorted(set(a.files) - set(b.files) - set(ignore)):
        out.append(f"{k}: only in {new.name}")
    for k in sorted(set(b.files) - set(a.files) - set(ignore)):
        out.append(f"{k}: only in the reference")
    for k in [k for k in b.files if k in a.files and k not in ignore]:
        x, y = a[k], b[k]
        if x.dtype != y.dtype or x.shape != y.shape:
            out.append(f"{k}: {x.dtype} {x.shape} vs {y.dtype} {y.shape}")
            continue
        eq = (np.array_equal(x, y, equal_nan=True) if x.dtype.kind in "fc"
              else np.array_equal(x, y))
        if not eq:
            if x.dtype.kind in "fiub" and x.size:
                diff = np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64)))
                out.append(f"{k}: differs, max |diff| {diff:.3e}, "
                           f"{int((x != y).sum())} of {x.size} entries")
            else:
                out.append(f"{k}: differs ({int((x != y).sum())} of {x.size} entries)")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("new", type=Path)
    ap.add_argument("reference", type=Path)
    ap.add_argument("--ignore", nargs="*", default=[], help="keys not compared")
    a = ap.parse_args(argv)
    diffs = compare_npz(a.new, a.reference, a.ignore)
    n = len(set(np.load(a.reference, allow_pickle=True).files) - set(a.ignore))
    print(f"new       {a.new}\nreference {a.reference}")
    for line in diffs:
        print(f"  DIFF {line}")
    print(f"{n - len(diffs)} of {n} keys identical")
    print("IDENTICAL" if not diffs else "DIFFERENT")
    return 0 if not diffs else 1


if __name__ == "__main__":
    sys.exit(main())
