#!/usr/bin/env python
"""Check the Figure 2 lists (lists/position/*.csv, lists/folds/position.csv) against data stores.

Every store location is a command-line argument; each check runs only when its argument is given.

  --crossed-houses DIR  house files h*.npz of the crossed harvest (fields house, bench, bench_idx):
                        crossed_houses.csv must list exactly these houses with these episodes
  --bench               build the benchmark catalog (vlageom.sim.build_house_list(950); needs
                        MolmoSpaces and the benchmark files): crossed_houses.csv must equal it
  --hand-lines DIR      hand-line files h*.npz: line_houses.csv must list exactly these houses
  --object-lines DIR    object-line root (shape_{kind}/h*/): its houses must be in line_houses.csv
  --labels FILE         labels.npz of the packed crossed store: lists/folds/position.csv must equal
                        scikit-learn GroupKFold(5) over house on the q1 rows (and on the q0 rows);
                        GroupKFold orders equally sized houses differently on different CPUs, so
                        this check is meaningful on an x86-64 CPU with AVX-512
  --natural-cache FILE  cache of the natural rollouts' model queries (arrays pos (n, 3) EE position
                        in the base frame, ep (n,) episode): ee_color_reference.csv must equal
                        pos[ep != 57] permuted by numpy default_rng(0) and cut to the first 4,000

Exit status 0 when every requested check passes.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from vlageom import paths
from vlageom.position import read_house_list

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--crossed-houses", type=Path, default=None)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--hand-lines", type=Path, default=None)
    ap.add_argument("--object-lines", type=Path, default=None)
    ap.add_argument("--labels", type=Path, default=None)
    ap.add_argument("--natural-cache", type=Path, default=None)
    a = ap.parse_args(argv)
    L = paths.lists_dir()
    crossed = read_house_list(L / "position" / "crossed_houses.csv")
    lines = read_house_list(L / "position" / "line_houses.csv")
    check("crossed_houses.csv: 947 sorted unique houses",
          len(crossed) == 947 and [h for h, *_ in crossed] == sorted({h for h, *_ in crossed}))
    check("line_houses.csv: 48 houses, all in crossed_houses.csv with the same episode",
          len(lines) == 48 and set(lines) <= set(crossed))

    if a.crossed_houses:
        got = []
        for f in sorted(a.crossed_houses.glob("h*.npz")):
            z = np.load(f, allow_pickle=False)
            got.append((int(z["house"]), str(z["bench"]), int(z["bench_idx"])))
        check("crossed_houses.csv == house files", sorted(got) == crossed, f"{len(got)} files")
    if a.bench:
        import vlageom.sim  # noqa: F401
        from vlageom.sim import build_house_list
        cat = [(h, b, i) for h, b, i, _sp in build_house_list(None, 950)]
        check("crossed_houses.csv == benchmark catalog", cat == crossed, f"{len(cat)} houses")
    if a.hand_lines:
        hs = sorted(int(f.stem[1:]) for f in a.hand_lines.glob("h*.npz"))
        check("line_houses.csv == hand-line houses", hs == [h for h, *_ in lines],
              f"{len(hs)} files")
    if a.object_lines:
        hs = sorted({int(p.name[1:]) for p in a.object_lines.glob("shape_line_*/h*")})
        check("object-line houses in line_houses.csv", set(hs) <= {h for h, *_ in lines},
              f"{len(hs)} houses")
    if a.labels:
        from sklearn.model_selection import GroupKFold
        lab = np.load(a.labels)
        with open(L / "folds" / "position.csv", newline="") as fh:
            fm = {int(r["house"]): int(r["fold"]) for r in csv.DictReader(fh)}
        for g in (1, 0):
            m = lab["chunk_q"] == g
            hh = lab["house"][m]
            got = {}
            z0 = np.zeros(len(hh))
            for k, (_tr, te) in enumerate(GroupKFold(5).split(z0, z0, hh)):
                got.update({int(h): k for h in np.unique(hh[te])})
            check(f"folds/position.csv == GroupKFold(5) on q{g} rows", got == fm,
                  f"{len(got)} houses")
    if a.natural_cache:
        z = np.load(a.natural_cache, allow_pickle=True)
        pos, ep = z["pos"], z["ep"]
        bg = pos[ep != 57]
        ref = bg[np.random.default_rng(0).permutation(len(bg))[:4000]].astype(np.float64)
        shipped = np.loadtxt(L / "position" / "ee_color_reference.csv", delimiter=",", skiprows=1)
        check("ee_color_reference.csv == natural-cache subsample", np.array_equal(ref, shipped),
              f"{len(shipped)} rows")
    print("ALL OK" if not FAILS else f"FAILED ({len(FAILS)})")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
