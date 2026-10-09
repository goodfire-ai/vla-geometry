"""Check the shipped lists (lists/*.csv, lists/folds/*.csv) against existing data stores.

Every store location is a command-line argument; each check runs only when its argument is given.

  --seed-catalog DIR    per-episode rollout records episode_*.npz (fields episode_index,
                        house_index, seed, success)
  --gripper-edits DIR   gripper edit files ep{E}_h{H}_{landmark}_q{Q}_aw300_sw3000.npz
  --wrist-sweeps DIR    wrist sweep files ep{I}.npz (as written by pipelines/wrist/01_sweep.py) or
                        ep{I}/<name>_ep{I}.npz (I = catalog_index)
  --teleport-store DIR  object-teleport store root (prep/meta.json, houses/h{H}.npz)
  --near-store DIR      near-range store root (houses/h{H}.npz, as written by
                        pipelines/distance/10_near_harvest.py; a houses_pick.json house list
                        next to houses/ is checked too when present)
  --fold-npz NAME=PATH  an npz holding a house -> fold assignment (arrays house + fold, or
                        per-row house + fold_id) to compare with lists/folds/NAME.csv; repeatable

Exit status 0 when every requested check passes.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from vlageom.paths import lists_dir  # noqa: E402
from vlageom.probes import FoldMap  # noqa: E402

LISTS = lists_dir()
EDIT_RE = re.compile(r"ep(\d+)_h(\d+)_(before_grasp|after_grasp)_q(\d+)_aw300_sw3000\.npz")
FAILS = []


def read_csv(name):
    with open(LISTS / name, newline="") as f:
        return list(csv.DictReader(f))


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def seed_catalog(d):
    cat = {}
    for f in sorted(Path(d).glob("episode_*.npz")):
        z = np.load(f)
        cat[int(z["episode_index"])] = (int(z["house_index"]), int(z["seed"]), bool(z["success"]))
    return cat


def check_gripper(edits_dir, cat):
    print("gripper_episodes.csv")
    raw = read_csv("gripper_episodes.csv")
    rows = [(int(r["episode_index"]), int(r["house"]), int(r["seed"])) for r in raw]
    check({r["new_sampler"] for r in raw} <= {"0", "1"}, "new_sampler is 0 or 1 "
          f"({sum(r['new_sampler'] == '1' for r in raw)} new, "
          f"{sum(r['new_sampler'] == '0' for r in raw)} existing)")
    eps = {(e, h) for e, h, _ in rows}
    check(len(rows) == len(eps) == 304, f"{len(rows)} episodes (expect 304), unique")
    check(len({h for _, h, _ in rows}) == 170, f"{len({h for _, h, _ in rows})} houses (expect 170)")
    if cat is not None:
        bad = [r for r in rows if cat.get(r[0], (None, None, False))[:2] != (r[1], r[2])
               or not cat[r[0]][2]]
        check(not bad, f"house/seed equal the seed catalog and the episode succeeded ({len(bad)} bad)")
    if edits_dir is not None:
        n_edits, seen, seed_bad = defaultdict(int), set(), 0
        seed_of = {(e, h): s for e, h, s in rows}
        for f in sorted(Path(edits_dir).glob("*.npz")):
            m = EDIT_RE.fullmatch(f.name)
            if not m:
                continue
            key = (int(m.group(1)), int(m.group(2)))
            seen.add(key)
            if key in eps:
                n_edits[key] += 1
                z = np.load(f)
                seed_bad += int(z["seed"]) != seed_of[key]
        check(set(n_edits) == eps, f"every listed episode has edit files ({len(n_edits)}/{len(eps)})")
        check(sum(n_edits.values()) == 1666, f"{sum(n_edits.values())} edits (expect 1666)")
        check(seed_bad == 0, f"edit-file seed field equals the listed seed ({seed_bad} differ)")
        print(f"  info  edit directory holds {len(seen)} episodes; {len(seen - eps)} not listed")


def check_wrist(sweep_dir, cat):
    print("wrist_episodes.csv")
    rows = [(int(r["catalog_index"]), int(r["episode_index"]), int(r["house"]), int(r["seed"]))
            for r in read_csv("wrist_episodes.csv")]
    check(len(rows) == 64 and len({r[2] for r in rows}) == 64, f"{len(rows)} episodes in 64 houses")
    if cat is not None:
        by = defaultdict(list)
        for e, (h, s, ok) in cat.items():
            if ok:
                by[h].append((e, h, s))
        first = [sorted(by[h])[0] for h in sorted(by)]           # first success per house
        bad = [r for r in rows if first[r[0]] != (r[1], r[2], r[3])]
        check(not bad, f"rows equal entry catalog_index of the first-success-per-house catalog "
                       f"({len(first)} houses; {len(bad)} differ)")
    if sweep_dir is not None:
        bad = []
        missing = []
        for i, e, h, _ in rows:
            f = sweep_file(sweep_dir, i)
            if f is None:
                missing.append(i)
                continue
            z = np.load(f, allow_pickle=True)
            if not str(z["meta"]).startswith(f"ep{e} h{h} "):
                bad.append(i)
        check(not missing, f"every listed episode has a sweep file ({len(missing)} missing)")
        check(not bad, f"sweep file meta matches episode/house ({len(bad)} differ)")


def sweep_file(sweep_dir, i):
    """The sweep file of catalog index `i`: <dir>/ep{i}.npz, else the one <dir>/ep{i}/*_ep{i}.npz."""
    d = Path(sweep_dir)
    if (d / f"ep{i}.npz").is_file():
        return d / f"ep{i}.npz"
    hits = sorted((d / f"ep{i}").glob(f"*_ep{i}.npz"))
    return hits[0] if len(hits) == 1 else None


def check_houses(name, root, n_expect):
    print(name)
    rows = [(int(r["house"]), r["bench"], int(r["bench_idx"])) for r in read_csv(name)]
    houses = [r[0] for r in rows]
    check(len(rows) == n_expect and len(set(houses)) == n_expect, f"{len(rows)} houses (expect {n_expect})")
    check(houses == sorted(houses), "sorted by house")
    if root is None:
        return houses
    root = Path(root)
    if name == "teleport_houses.csv":
        meta = json.loads((root / "prep" / "meta.json").read_text())
        check(sorted(meta["houses"]) == houses, f"equal the store's house list ({meta['n_houses']})")
    else:
        have = sorted(int(f.stem[1:]) for f in (root / "houses").glob("h*.npz")
                      if f.stem[1:].isdigit())
        check(have == houses, f"equal the store's house files ({len(have)})")
        if (root / "houses_pick.json").is_file():
            pick = json.loads((root / "houses_pick.json").read_text())
            check(pick == houses, f"equal the store's houses_pick.json ({len(pick)})")
    bad = 0
    for h, b, bi in rows:
        z = np.load(root / "houses" / f"h{h}.npz")
        bad += (str(z["bench"]), int(z["bench_idx"])) != (b, bi)
    check(bad == 0, f"bench / bench_idx equal the house files ({bad} differ)")
    return houses


def check_fold(name, npz_path):
    print(f"folds/{name}.csv")
    fm = FoldMap.load(name)
    z = np.load(npz_path, allow_pickle=False)
    h = z["house"].astype(np.int64)
    f = (z["fold"] if "fold" in z.files else z["fold_id"]).astype(np.int64)
    ref = {}
    consistent = all(ref.setdefault(int(a), int(b)) == int(b) for a, b in zip(h, f))
    check(consistent, "reference assigns one fold per house")
    ours = dict(zip(fm.house.tolist(), fm.fold.tolist()))
    check(ours == ref, f"{len(ours)} houses, identical to the reference ({len(ref)} houses); "
                       f"sha1 {fm.sha1()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed-catalog")
    ap.add_argument("--gripper-edits")
    ap.add_argument("--wrist-sweeps")
    ap.add_argument("--teleport-store")
    ap.add_argument("--near-store")
    ap.add_argument("--fold-npz", action="append", default=[], metavar="NAME=PATH")
    a = ap.parse_args()
    cat = seed_catalog(a.seed_catalog) if a.seed_catalog else None
    check_gripper(a.gripper_edits, cat)
    check_wrist(a.wrist_sweeps, cat)
    tele = check_houses("teleport_houses.csv", a.teleport_store, 906)
    near = check_houses("near_houses.csv", a.near_store, 300)
    check(set(near) <= set(tele), "near-range houses are teleport houses")
    for spec in a.fold_npz:
        name, path = spec.split("=", 1)
        check_fold(name, path)
    for name in ("target_position", "distance", "distance_teleport"):
        fm = FoldMap.load(name)
        print(f"folds/{name}.csv: {len(fm)} houses, per fold {np.bincount(fm.fold).tolist()}, "
              f"sha1 {fm.sha1()}")
    print("PASS" if not FAILS else f"FAIL ({len(FAILS)})")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
