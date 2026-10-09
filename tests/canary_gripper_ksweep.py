"""Compare the records of a run of pipelines/gripper_fig7/30_ksweep.py with reference records.

For each selected episode the set of record files (one per grasp landmark) must be the same, and in
every pair of records every field must be equal: the scalar fields (episode, house, seed, landmark,
target, branch_query, grasp_onset_query, n_queries, settings ...), and for every k of `results` and
for `ceiling_inrun` every metric (flip, grip_before, grip_after, arm_leakage, arm_jerk, ...,
norm, k_eff, loss) and the rounded coefficients. A field present in only one of the two records
is reported (the comparison is over the reference's fields).

For metrics that differ the script prints, per k, the largest absolute and relative difference
over the compared records.

  python pipelines/gripper_fig7/30_ksweep.py --episode-index 0 --out <scratch>/ksweep ...
  python tests/canary_gripper_ksweep.py --run <scratch>/ksweep --reference <reference records> \
      --episode-index 0

The reference records are those of another run of stage 30 with the same edit files and model
inputs (for example a full run on another node).

Exit status 0 when every selected episode has the same records with equal fields.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from vlageom import gripper as G
from vlageom import gripper_fig7 as F
from vlageom import paths

METRICS = ("grip_before", "grip_after", "grip_exec", "arm_leakage", "arm_vs_natural", "arm_jerk",
           "nat_jerk", "arm_leak_exec", "arm_jerk_exec", "nat_jerk_exec", "arm_span", "norm",
           "loss")


def _eq(u, v) -> bool:
    """Equality that treats two NaN floats as equal (k = 0 records a NaN loss)."""
    if isinstance(u, float) and isinstance(v, float) and math.isnan(u) and math.isnan(v):
        return True
    return u == v


def by_episode(d: Path) -> dict:
    out = defaultdict(dict)
    for f in sorted(d.glob("ep*_*.json")):
        r = json.loads(f.read_text())
        out[(int(r["ep"]), int(r["house"]))][f.name] = r
    return out


def compare_record(a: dict, b: dict, stats: dict) -> list[str]:
    """Differences between run record `a` and reference record `b`, one line each; numeric
    differences of the fit metrics are also collected into `stats[(k, metric)]`."""
    lines = []
    for k in b:
        if k not in a:
            lines.append(f"{k}: only in the reference")
    for k in a:
        if k not in b:
            lines.append(f"{k}: only in the run")
    for k in b:
        if k in a and k not in ("results", "ceiling_inrun") and not _eq(a[k], b[k]):
            lines.append(f"{k}: {a[k]!r} vs {b[k]!r}")
    fits = [(kk, (a.get("results") or {}).get(kk), v) for kk, v in (b.get("results") or {}).items()]
    fits.append(("768-D", a.get("ceiling_inrun"), b.get("ceiling_inrun")))
    for kk, x, y in fits:
        if x is None or y is None:
            if x is not y:
                lines.append(f"k={kk}: fit only in {'the reference' if x is None else 'the run'}")
            continue
        for m in y:
            if m not in x:
                lines.append(f"k={kk} {m}: only in the reference")
            elif not _eq(x[m], y[m]):
                if m in METRICS:
                    d = abs(float(x[m]) - float(y[m]))
                    rel = d / (abs(float(y[m])) + 1e-30)
                    s = stats[(kk, m)]
                    s[0], s[1] = max(s[0], d), max(s[1], rel)
                    lines.append(f"k={kk} {m}: {x[m]!r} vs {y[m]!r}")
                elif m == "coeffs":
                    xa, ya = np.asarray(x[m], float), np.asarray(y[m], float)
                    tail = (f", max|diff| {np.max(np.abs(xa - ya)):.4g}" if xa.shape == ya.shape
                            else f", lengths {len(xa)} vs {len(ya)}")
                    lines.append(f"k={kk} coeffs differ{tail}")
                else:
                    lines.append(f"k={kk} {m}: {x[m]!r} vs {y[m]!r}")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=paths.tile_dir(F.TILE, "ksweep"),
                    help="record directory of the run (default: $VLAGEOM_DATA/gripper_fig7/ksweep)")
    ap.add_argument("--reference", type=Path, required=True, help="reference record directory")
    ap.add_argument("--episode-index", type=int, nargs="+", default=None,
                    help="episodes to compare (default: every episode with records in --run)")
    ap.add_argument("--max-lines", type=int, default=40,
                    help="difference lines printed per record (default 40)")
    a = ap.parse_args(argv)

    run, ref = by_episode(a.run), by_episode(a.reference)
    if a.episode_index:
        listed = {e: h for e, h, *_ in G.read_episodes()}
        keys = sorted((e, listed[e]) for e in a.episode_index if e in listed)
    else:
        keys = sorted(run)
    if not keys:
        print(f"no records to compare in {a.run}")
        return 1

    print(f"run       {a.run}\nreference {a.reference}")
    ok = True
    n_files = n_same = 0
    stats = defaultdict(lambda: [0.0, 0.0])
    for key in keys:
        x, y = run.get(key, {}), ref.get(key, {})
        if set(x) != set(y):
            ok = False
            print(f"  ep{key[0]}_h{key[1]}: record sets differ: run {sorted(x)}, "
                  f"reference {sorted(y)}")
        for name in sorted(set(x) & set(y)):
            n_files += 1
            diffs = compare_record(x[name], y[name], stats)
            if diffs:
                ok = False
                print(f"  DIFF  {name} ({len(diffs)} fields)")
                for line in diffs[:a.max_lines]:
                    print(f"        {line}")
            else:
                n_same += 1
                print(f"  equal {name}")
    if stats:
        print("largest differences of the fit metrics (absolute / relative), per k:")
        order = sorted({k for k, _ in stats},
                       key=lambda s: (s == "768-D", int(s) if s.isdigit() else 0))
        for kk in order:
            row = [f"{m} {stats[(kk, m)][0]:.3g}/{stats[(kk, m)][1]:.3g}"
                   for m in ("arm_leakage", "grip_after", "norm") if (kk, m) in stats]
            if row:
                print(f"  k={kk:>5}: " + ", ".join(row))
    print(f"{n_same}/{n_files} common records identical over {len(keys)} episode(s)")
    print("IDENTICAL" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
