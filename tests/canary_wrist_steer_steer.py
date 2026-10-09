"""Compare the rows of a steering run (pipelines/wrist_steer/60_steer.py) with a reference run, row by
row.

Rows are matched by (episode, tag, axis, written angle). Only the reference rows of the episodes in
the new run are expected (so a run with --episodes 1 is compared with the first scene's rows). Every
key present in both rows must be identical (floats exactly, NaN equal to NaN, lists element by
element). A key present only in the reference rows is accepted only if it is one of EMPTY_KEYS and
empty there (the held-edit rows of the reference also carry `frames`, rendered frames, and
`achieved_plane_deg`, a field this stage does not use; both are empty in the paper's runs and
stage 60 does not write them); keys present only in the new rows are listed.
The random-generator control rows
(tag random_gen) depend on the operator of stage 40; run the canary with the reference operator
(--operator) to compare them exactly, or give --rtol to accept a relative difference in those rows
(for an operator fitted with another BLAS thread count).

  python pipelines/wrist_steer/60_steer.py --mode persist --episodes 1 --chart <reference chart> \
      --operator <reference operator> --out $C/steer/persist.json
  python tests/canary_wrist_steer_steer.py --new $C/steer/persist.json --reference <reference persist json>
  (the same with --mode oneshot)

Bit-identical rows are expected only on the GPU type of the reference (bfloat16).
Exit status 0 when every compared row is identical (random_gen rows within --rtol).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


EMPTY_KEYS = ("frames", "achieved_plane_deg")


def key(r):
    return (int(r["ep"]), r["tag"], r["axis"], float(r["theta_deg"]))


def leaves(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{prefix}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{prefix}[{i}]")
    else:
        yield prefix, x


def equal(a, b, rtol=0.0):
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b or (rtol > 0 and abs(a - b) <= rtol * max(abs(a), abs(b)))
    return a == b


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", type=Path, required=True, help="json written by the canary run")
    ap.add_argument("--reference", type=Path, required=True, help="reference json of the same mode")
    ap.add_argument("--rtol", type=float, default=0.0,
                    help="relative tolerance for the random_gen rows (default exact)")
    args = ap.parse_args(argv)
    new = json.loads(args.new.read_text())["rows"]
    ref = json.loads(args.reference.read_text())["rows"]
    eps = {int(r["ep"]) for r in new}
    ref = [r for r in ref if int(r["ep"]) in eps]
    rn = {key(r): r for r in new}
    rr = {key(r): r for r in ref}
    ok = True
    only_new, only_ref = set(), set()
    n_leaves = n_eq = 0
    for k in sorted(set(rn) | set(rr)):
        if k not in rn or k not in rr:
            print(f"  MISSING row {k} in the {'new run' if k not in rn else 'reference'}")
            ok = False
            continue
        a, b = rn[k], rr[k]
        only_new |= set(a) - set(b)
        only_ref |= set(b) - set(a)
        tol = args.rtol if k[1] == "random_gen" else 0.0
        bad = [f"{kk} only in the reference" + (" (not empty)" if kk in EMPTY_KEYS else "")
               for kk in sorted(set(b) - set(a)) if kk not in EMPTY_KEYS or b[kk]]
        for kk in sorted(set(a) & set(b)):
            la, lb = list(leaves(a[kk], kk)), list(leaves(b[kk], kk))
            if len(la) != len(lb):
                bad.append(f"{kk} length {len(la)} vs {len(lb)}")
                continue
            for (p, x), (_, y) in zip(la, lb):
                n_leaves += 1
                if equal(x, y, tol):
                    n_eq += 1
                else:
                    bad.append(f"{p}: {x!r} vs {y!r}")
        if bad:
            ok = False
            print(f"  DIFF  {k}: " + "; ".join(bad[:6]) + (" ..." if len(bad) > 6 else ""))
        else:
            print(f"  equal {k}")
    if only_new:
        print(f"  keys only in the new rows (not compared): {sorted(only_new)}")
    if only_ref:
        print(f"  keys only in the reference rows: {sorted(only_ref)} (accepted when empty: "
              f"{list(EMPTY_KEYS)})")
    print(f"{len(rn)} rows, episodes {sorted(eps)}; {n_eq}/{n_leaves} values equal")
    print("IDENTICAL" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
