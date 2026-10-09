"""Compare the edit files of a gripper harvest run with reference edit files, file by file.

For each selected episode: the set of edit files (one per landmark and query) must be the same, and
in every pair of files every key must have the same dtype, shape and values (np.array_equal). For
values that differ, the max |difference| is printed, and for the 768-vectors (delta, base_act,
steered_act) also the cosine between the two vectors.

The fits run in bfloat16 on the GPU; a harvest reproduces reference files to the last bit only on
the same GPU type with the pinned library versions.

  python pipelines/gripper/10_harvest_edits.py --episode-index 0 14 --out <scratch>
  python tests/canary_gripper_harvest.py --out <scratch> --reference <reference edits dir> \
      --episode-index 0 14

Exit status 0 when every selected episode has the same files with identical contents.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from vlageom import gripper as G  # noqa: E402
from vlageom import paths  # noqa: E402


def by_episode(d: Path) -> dict:
    out = defaultdict(dict)
    for f in sorted(d.glob("*.npz")):
        m = G.EDIT_RE.fullmatch(f.name)
        if m:
            out[(int(m.group(1)), int(m.group(2)))][f.name] = f
    return out


def compare_file(fa: Path, fb: Path) -> list[str]:
    """Differences between two edit files, one line each (empty if identical)."""
    a, b = np.load(fa), np.load(fb)
    lines = []
    for k in sorted(set(a.files) - set(b.files)):
        lines.append(f"{k}: only in the run")
    for k in sorted(set(b.files) - set(a.files)):
        lines.append(f"{k}: only in the reference")
    for k in [k for k in b.files if k in a.files]:
        x, y = a[k], b[k]
        if x.dtype != y.dtype or x.shape != y.shape:
            lines.append(f"{k}: {x.dtype} {x.shape} vs {y.dtype} {y.shape}")
        elif not np.array_equal(x, y):
            msg = f"{k}: differs"
            if np.issubdtype(x.dtype, np.number) and not np.issubdtype(x.dtype, np.bool_):
                xd, yd = x.astype(np.float64), y.astype(np.float64)
                msg += f", max|diff| {np.max(np.abs(xd - yd)):.3e}"
                if x.ndim == 1 and x.size > 8:
                    cos = float(xd @ yd / (np.linalg.norm(xd) * np.linalg.norm(yd) + 1e-30))
                    msg += f", cos {cos:.7f}"
            else:
                msg += f": {x!r} vs {y!r}"
            lines.append(msg)
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=paths.tile_dir("gripper"),
                    help="harvest output root (with deltas/) or an edits directory "
                         "(default: $VLAGEOM_DATA/gripper)")
    ap.add_argument("--reference", type=Path, required=True, help="reference edits directory")
    ap.add_argument("--episode-index", type=int, nargs="+", default=None,
                    help="episodes to compare (default: every episode with files in --out)")
    args = ap.parse_args(argv)

    run_dir = args.out / "deltas" if (args.out / "deltas").is_dir() else args.out
    run, ref = by_episode(run_dir), by_episode(args.reference)
    if args.episode_index:
        want = set(args.episode_index)
        listed = {e: h for e, h, *_ in G.read_episodes()}
        keys = sorted((e, listed[e]) for e in want if e in listed)
        keys += sorted(k for k in set(run) | set(ref) if k[0] in want and k not in keys)
    else:
        keys = sorted(run)
    if not keys:
        print(f"no edit files to compare in {run_dir}")
        return 1

    print(f"run       {run_dir}\nreference {args.reference}")
    ok = True
    n_files = n_same = 0
    for key in keys:
        a, b = run.get(key, {}), ref.get(key, {})
        tag = f"ep{key[0]}_h{key[1]}"
        if set(a) != set(b):
            ok = False
            print(f"  {tag}: file sets differ")
            for n in sorted(set(a) - set(b)):
                print(f"    only in the run:       {n}")
            for n in sorted(set(b) - set(a)):
                print(f"    only in the reference: {n}")
        for n in sorted(set(a) & set(b)):
            n_files += 1
            diffs = compare_file(a[n], b[n])
            if diffs:
                ok = False
                print(f"  DIFF  {n}")
                for line in diffs:
                    print(f"        {line}")
            else:
                n_same += 1
                print(f"  equal {n}")
    print(f"{n_same}/{n_files} common files identical over {len(keys)} episode(s)")
    print("IDENTICAL" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
