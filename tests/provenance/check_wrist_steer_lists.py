"""Check the Figure 6 lists (lists/wrist_steer/*.csv) against the paper's data stores.

Every store location is a command-line argument; each check runs only when its argument is given.

  --seed-catalog DIR    per-episode rollout records episode_*.npz of the natural rollouts (fields
                        episode_index, house_index, seed, success): the 1,011 natural rollouts of
                        houses 0-199 of which episodes 0-199 are the natural cloud's
  --natural-cache FILE  natural cache (ep per row): the row order of natural_episodes.csv
  --orient-dir DIR      orientation sweep files ep<E>_h<H>.npz
  --rollout-logs DIR    logs (*.out) of the natural rollout processes, with the lines
                        'Episode <i+1> house <h> failed (attempt <a>): House invalid ...'

Checks:
  natural_episodes.csv  episodes 0-199, partition = episode // 16, house and seed as recorded; the
                        seeds and houses follow the sampling attempts of a partition
                        (`vlageom.wrist_steer.sampling_attempts`), and the failed attempts they
                        imply include every one in the rollout logs; the row order is the order
                        in which the episodes appear in the natural cache (episode 194, with 4
                        queries, is not in the cache and follows 193)
  orient_episodes.csv   the first 200 entries of the catalog that keeps, per house in increasing
                        order, the first two successful episodes (by episode index); the sweep store
                        holds all of them but three (houses 30, 104, 106)
  heldout_houses.csv    numpy default_rng(0) shuffle of the sweep store's houses, first third
  steer_episodes.csv    the first successful episode of each held-out house, houses in increasing
                        order, first 8

Exit status 0 when every requested check passes.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

from vlageom import wrist_steer as ws

FAILS = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed-catalog", type=Path, default=None)
    ap.add_argument("--natural-cache", type=Path, default=None)
    ap.add_argument("--orient-dir", type=Path, default=None)
    ap.add_argument("--rollout-logs", type=Path, default=None)
    a = ap.parse_args(argv)
    nat = ws.read_natural_episodes()
    orient = ws.read_orient_episodes()
    held = ws.read_heldout_houses()
    steer = ws.read_steer_episodes()

    print("natural_episodes.csv")
    check(sorted(r["episode_index"] for r in nat) == list(range(200)), "episodes 0..199, each once")
    check(all(r["partition"] == r["episode_index"] // ws.NATURAL_PER_PARTITION for r in nat),
          "partition = episode // 16")
    attempts = {}
    try:
        for p in sorted({r["partition"] for r in nat}):
            eps_p = sorted((r for r in nat if r["partition"] == p), key=lambda r: r["episode_index"])
            for r, att in zip(eps_p, ws.sampling_attempts(nat, p)):
                attempts[r["episode_index"]] = att
        implied = sorted((e, h) for e, att in attempts.items() for h, _s in att[:-1])
        check(True, f"seeds and houses follow the sampling attempts (failed: episode, house {implied})")
    except ValueError as e:
        implied = None
        check(False, f"seeds and houses follow the sampling attempts ({e})")
    if a.rollout_logs is not None and implied is not None:
        logged = set()
        for f in sorted(a.rollout_logs.glob("*.out")):
            for m in re.finditer(r"Episode (\d+) house (\d+) failed \(attempt \d+\): House invalid",
                                 f.read_text(errors="replace")):
                if int(m.group(1)) - 1 < 200:          # the logs count episodes from 1
                    logged.add((int(m.group(1)) - 1, int(m.group(2))))
        absent = sorted(set(implied) - logged)
        check(logged <= set(implied), "every failed attempt in the rollout logs is implied by the list"
              + (f" (implied but not in the given logs: {absent})" if absent else ""))
    cat = {}
    if a.seed_catalog is not None:
        for f in sorted(a.seed_catalog.glob("episode_*.npz")):
            z = np.load(f)
            cat[int(z["episode_index"])] = (int(z["house_index"]), int(z["seed"]), bool(z["success"]))
        bad = [r for r in nat if cat.get(r["episode_index"], (None, None))[:2] != (r["house"], r["seed"])]
        check(not bad, f"house and seed equal the seed catalog ({len(bad)} differ)")
    if a.natural_cache is not None:
        ep = np.load(a.natural_cache)["ep"]
        order = [int(ep[0])] + [int(ep[i]) for i in range(1, len(ep)) if ep[i] != ep[i - 1]]
        listed = [r["episode_index"] for r in nat if r["episode_index"] in set(order)]
        check(listed == order, f"row order = episode order of the natural cache ({len(order)} episodes)")
        check(set(r["episode_index"] for r in nat) - set(order) == {194},
              "only episode 194 is not in the cache")

    print("orient_episodes.csv")
    check([r["task"] for r in orient] == list(range(200)), "tasks 0..199")
    if cat:
        by_house = {}
        for e, (h, s, ok) in sorted(cat.items()):
            if ok:
                by_house.setdefault(h, []).append((e, h, s))
        catalog = [x for h in sorted(by_house) for x in sorted(by_house[h])[:2]]
        check([(r["episode_index"], r["house"], r["seed"]) for r in orient] == catalog[:200],
              f"first 200 of the 2-per-house success catalog ({len(catalog)} entries)")
    files = []
    if a.orient_dir is not None:
        files = ws.orient_files(a.orient_dir)
        names = {Path(f).name for f in files}
        missing = [r for r in orient if f"ep{r['episode_index']}_h{r['house']}.npz" not in names]
        check(sorted(r["house"] for r in missing) == [30, 104, 106],
              f"{len(files)} sweep files; listed but absent: {[(r['task'], r['episode_index'], r['house']) for r in missing]}")

    print("heldout_houses.csv")
    check(len(held) == 34, f"{len(held)} houses")
    if files:
        houses = [int(np.load(f)["house"]) for f in files]
        check(ws.heldout_split(houses) == held,
              f"default_rng(0) shuffle of the {len(set(houses))} sweep houses, first third")

    print("steer_episodes.csv")
    check(len(steer) == 8 and all(r["house"] in set(held) for r in steer),
          "8 episodes of held-out houses")
    if cat:
        first = []
        for h in sorted(set(held)):
            succ = sorted(e for e, (hh, s, ok) in cat.items() if hh == h and ok)
            if succ:
                first.append((succ[0], h, cat[succ[0]][1]))
        check([(r["episode_index"], r["house"], r["seed"]) for r in steer] == first[:8],
              "first successful episode of each held-out house, first 8")
    print("ALL OK" if not FAILS else f"{len(FAILS)} FAILED")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
