"""Check the Figure 4 lists (lists/contact/*.csv, lists/folds/contact_natural.csv) against the
paper's data stores.

Every store location is a command-line argument; each check runs only when its argument is given.

  --seed-catalog DIR      per-episode rollout records episode_*.npz of the natural rollouts
                          (fields episode_index, house_index, seed, success; 1,011 episodes)
  --episodes-dir DIR      per-episode metadata episode_XXXX.json of the same rollouts (fields
                          task_description, n_steps, query_steps, objects_state)
  --episode-set FILE      an .npz with a per-row `ep` column (e.g. the natural cache): its episodes
                          must be exactly those of natural_episodes.csv
  --natural-cache FILE    natural cache (rows with `house` and `contact`): the fold map must equal
                          scikit-learn GroupKFold(5) over its houses (the result depends on the
                          CPU's sort kernel; the shipped map was assigned on an AVX-512 machine)
  --offschedule-dir DIR   off-schedule harvest files ep<E>_h<H>.npz: a file exists exactly for the
                          rows with outcome "shard"

Checks:
  natural_episodes.csv     house and seed as recorded in the seed catalog; task_description and
                           orig_n_steps as in the metadata; the rollouts queried the model at
                           steps 0, 8, 16, ...; orig_contact[q] = gripper-object contact (a contact
                           record with a geom named robot_0/gripper/... and normal force > 0) at
                           the step of query q; the episodes are those of the first 300
                           metadata files (by name) with at least 5 model queries and object states
  offschedule_episodes.csv the first 128 rows of the catalog that keeps, for each house in
                           increasing order, its successful episode with the lowest episode_index
  folds/contact_natural.csv  192 houses; GroupKFold(5) over the natural cache's rows

Exit status 0 when every requested check passes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from vlageom import contact as C
from vlageom import paths
from vlageom.probes import FoldMap

FAILS = []
GRIP_PREFIX = "robot_0/gripper/"


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def grip_contact(state_step) -> bool:
    for c in state_step.get("pickup_contacts", []):
        g1, g2 = c.get("geom1", ""), c.get("geom2", "")
        if (g1.startswith(GRIP_PREFIX) or g2.startswith(GRIP_PREFIX)) and float(c.get("force_normal", 0.0)) > 0.0:
            return True
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--seed-catalog", type=Path, default=None)
    ap.add_argument("--episodes-dir", type=Path, default=None)
    ap.add_argument("--episode-set", type=Path, default=None)
    ap.add_argument("--natural-cache", type=Path, default=None)
    ap.add_argument("--offschedule-dir", type=Path, default=None)
    a = ap.parse_args(argv)
    nat = C.read_natural_episodes()
    off = C.read_offschedule_episodes()
    folds = FoldMap.load("contact_natural")

    print("natural_episodes.csv")
    eps = [r["episode_index"] for r in nat]
    check(len(nat) == 299 and eps == sorted(set(eps)), "299 episodes, increasing episode_index")
    check(len({r["house"] for r in nat}) == 192, "192 houses")
    check(all(set(r["orig_contact"]) <= {"0", "1"} for r in nat), "orig_contact is a 0/1 string")
    if a.episode_set is not None:
        with np.load(a.episode_set, mmap_mode="r") as z:
            s = sorted(int(e) for e in np.unique(np.asarray(z["ep"])))
        check(s == eps, f"episodes = the episodes of {a.episode_set.name}")
    print("offschedule_episodes.csv")
    check([r["task"] for r in off] == list(range(128)), "tasks 0..127")
    check(sorted(r["outcome"] for r in off if r["outcome"] != "shard")
          == ["no_crossing", "no_crossing", "no_scene", "no_scene", "no_scene"],
          "123 shard, 3 no_scene, 2 no_crossing")
    if a.seed_catalog is not None:
        rec = {}
        for f in sorted(a.seed_catalog.glob("episode_*.npz")):
            d = np.load(f, allow_pickle=False)
            rec[int(d["episode_index"])] = (int(d["house_index"]), int(d["seed"]), bool(d["success"]))
        check(all(rec.get(r["episode_index"], (None, None))[:2] == (r["house"], r["seed"]) for r in nat),
              "natural: house and seed as in the seed catalog")
        by_house = {}
        for e, (h, s, ok) in rec.items():
            if ok:
                by_house.setdefault(h, []).append((e, h, s))
        cat = [sorted(by_house[h])[0] for h in sorted(by_house)]
        check([(r["episode_index"], r["house"], r["seed"]) for r in off] == cat[:128],
              f"off-schedule: first 128 of the {len(cat)}-house first-success catalog")
    if a.episodes_dir is not None:
        bad = []
        for r in nat:
            meta = json.loads((a.episodes_dir / f"episode_{r['episode_index']:04d}.json").read_text())
            ost = meta.get("objects_state", [])
            qs = [int(s) for s in (meta.get("query_steps") or [])]
            want = "".join("1" if grip_contact(ost[s]) else "0" for s in qs)
            if (meta.get("task_description", "") != r["task_description"]
                    or int(meta.get("n_steps") or 1) != r["orig_n_steps"]
                    or len(ost) != r["orig_n_steps"]
                    or qs != [C.ORIG_QUERY_INTERVAL * q for q in range(len(qs))]
                    or want != r["orig_contact"]):
                bad.append(r["episode_index"])
        check(not bad, f"natural: task_description, orig_n_steps, query steps and orig_contact as in "
                       f"the metadata (mismatches: {bad[:10]})")
        first = sorted(a.episodes_dir.glob("episode_*.json"))[:300]
        sel = []
        for f in first:
            meta = json.loads(f.read_text())
            if len(meta.get("query_steps") or []) >= 5 and meta.get("objects_state"):
                sel.append(int(meta.get("episode_index", int(f.stem.split("_")[1]))))
        check(sorted(sel) == eps, f"natural: the episodes among the first 300 with at least 5 model "
                                  f"queries and object states ({len(sel)})")
    if a.offschedule_dir is not None:
        have = {p.name for p in a.offschedule_dir.glob("ep*_h*.npz")}
        check(all((C.shard_name(r["episode_index"], r["house"]) in have) == (r["outcome"] == "shard")
                  for r in off), "off-schedule: a harvest file exists exactly for outcome == shard")
    print("folds/contact_natural.csv")
    check(len(folds) == 192 and folds.sha1() == "98b0e978cdcbb1bbae914a6c3c8e55cbba2527c0",
          f"192 houses, sha1 {folds.sha1()}")
    check(set(folds.house.tolist()) == {r["house"] for r in nat}, "the houses of the natural episodes")
    if a.natural_cache is not None:
        from sklearn.model_selection import GroupKFold
        with np.load(a.natural_cache, mmap_mode="r") as z:
            house = np.asarray(z["house"])
            y = np.asarray(z["contact"]).astype(int)
        got = {}
        for k, (_, te) in enumerate(GroupKFold(n_splits=5).split(np.zeros((len(y), 1)), y, house)):
            for h in np.unique(house[te]):
                got[int(h)] = k
        check(got == dict(zip(folds.house.tolist(), folds.fold.tolist())),
              "GroupKFold(5) over the natural cache rows gives the shipped map")
    print("md5 of the shipped files:")
    import hashlib
    for f in ("contact/natural_episodes.csv", "contact/offschedule_episodes.csv",
              "folds/contact_natural.csv"):
        print(f"  {hashlib.md5((paths.lists_dir() / f).read_bytes()).hexdigest()}  lists/{f}")
    print("PASS" if not FAILS else f"FAIL ({len(FAILS)})")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
