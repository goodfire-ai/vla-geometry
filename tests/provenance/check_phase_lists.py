"""Check the Figure 5 lists (lists/phase/*.csv, lists/folds/phase.csv) against the paper's data.

Every location is a command-line argument; each check runs only when its arguments are given.
With --write DIR the lists are written to DIR (same file names) from the given inputs instead of
being compared with the shipped ones.

  --validity FILE      house validity scan of the pick-and-place config (houses 0-199; stage 05
                       --scan writes the same format): {"houses": {house: {"valid": bool}}}
  --rollouts DIR       natural episode files ep<EEEEE>_h<H>.npz
  --accounting FILE    job accounting of the natural rollout jobs, one attempt per line:
                       ARRAY_TASK_ID|STATE|START|END|NODE with ARRAY_TASK_ID = <job>_<partition> and
                       ISO times in the same time zone as the file times (UTC); e.g. the output of
                       `sacct -j <jobs> -D -X -n -P -o JobID,State,Start,End,NodeList`
  --cache FILE         natural cache of stage 20 (the paper's cache of the 530 episodes)
  --freeze-store DIR   freeze-test files ep<E>_h<H>.npz
  --decode-store DIR   a directory whose ep<E>_h<H>.npz names are the episodes of the decode set
  --fold-bank FILE     per-row fold assignment of the successful cache rows (fold_of_row, and
                       house_sha1 of the house column of those rows)
  --trajectory FILE    the trajectory capture (ep, house, seed)

Checks / derivations:
  natural_episodes.csv   catalog = valid houses (increasing) x 3 seed rounds, episode_index =
                         round * n_valid + position, seed = 20,000,000 + 1000 round + house,
                         partition = episode_index // 4; the rows are the episodes with a file.
                         new_sampler = 1 when the episode was the first one the process that wrote
                         it sampled: per attempt of the episode's partition, the first catalog
                         episode of the partition without a file at the attempt's start; the
                         attempt that wrote a file is the one whose [start, end] holds the file's
                         modification time
  freeze_episodes.csv    the cache's successful episodes, at most one per house (the one with a
                         place label first, then one with a transport label, then the lowest
                         index), houses ordered by (no place label, no transport label, house);
                         the first 128, task = row; decode = 1 for the episodes of --decode-store
  folds/phase.csv        one fold per house over the successful cache rows (--fold-bank), whose
                         fold hash (vlageom.phase.fold_hash) is c8b301a92149
  trajectory_episode.csv the capture's ep, house, seed; new_sampler = 0 (the capture was the
                         third episode its process sampled)

Exit status 0 when every requested check passes.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from vlageom import paths
from vlageom import phase as P

FAILS = []
N_ROUNDS = 3
EP_PER_PARTITION = 4
N_FREEZE = 128


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def catalog(validity: Path):
    v = json.loads(validity.read_text())
    houses = sorted(int(h) for h, d in v["houses"].items() if d.get("valid"))
    cat = []
    for r in range(N_ROUNDS):
        for h in houses:
            cat.append((len(cat), h, P.SEED_BASE + 1000 * r + h))
    return cat, houses


def _ts(s):
    if s in ("Unknown", "None", ""):
        return float("inf")
    return dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc).timestamp()


def natural_rows(cat, rollouts: Path, accounting: Path):
    mtime = {}
    for f in rollouts.glob("ep*_h*.npz"):
        mtime[int(f.name[2:7])] = f.stat().st_mtime
    attempts = {}
    for line in accounting.read_text().splitlines():
        if not line.strip():
            continue
        jid, _state, start, end, _node = line.strip().split("|")[:5]
        part = int(jid.split("_")[1])
        attempts.setdefault(part, []).append((_ts(start), _ts(end)))
    rows = []
    for e, h, s in cat:
        if e not in mtime:
            continue
        k = e // EP_PER_PARTITION
        eps = [c[0] for c in cat[k * EP_PER_PARTITION:(k + 1) * EP_PER_PARTITION]]
        hit = [a for a in sorted(attempts.get(k, [])) if a[0] <= mtime[e] <= a[1] + 5]
        if len(hit) != 1:
            raise SystemExit(f"episode {e}: {len(hit)} job attempts hold its file time")
        start = hit[0][0]
        first = next(x for x in eps if not (x in mtime and mtime[x] < start))
        rows.append(dict(episode_index=e, partition=k, house=h, seed=s,
                         new_sampler=int(first == e)))
    return rows


def freeze_rows(cache: Path, decode_eps=None):
    z = P.load_cache(cache, ["ep", "success", "house", "seed", "label"])
    ep_col = z["ep"].astype(int)
    by_house = {}
    for e in np.unique(ep_col):
        m = ep_col == e
        if not bool(z["success"][m].any()):
            continue
        h, s = int(z["house"][m][0]), int(z["seed"][m][0])
        has_p = bool((z["label"][m] == P.PLACE).any())
        has_t = bool((z["label"][m] == P.TRANSPORT).any())
        by_house.setdefault(h, []).append((not has_p, not has_t, int(e), s))
    picks = []
    for h in sorted(by_house):
        no_p, no_t, e, s = sorted(by_house[h])[0]
        picks.append((no_p, no_t, h, e, s))
    picks.sort()
    rows = []
    for t, (_np, _nt, h, e, s) in enumerate(picks[:N_FREEZE]):
        rows.append(dict(task=t, episode_index=e, house=h, seed=s,
                         decode=int(decode_eps is not None and (e, h) in decode_eps)))
    return rows


def fold_rows(cache: Path, bank: Path):
    z = P.load_cache(cache, ["success", "label", "house"])
    m = z["success"] & (z["label"] >= 0)
    hs = z["house"][m]
    b = np.load(bank)
    hh = hashlib.sha1(np.asarray(hs, np.int64).tobytes()).hexdigest()
    check(str(b["house_sha1"]) == hh, "fold bank belongs to these cache rows (house sha1)")
    fa = b["fold_of_row"]
    mp = {}
    for h, f in zip(hs, fa):
        mp.setdefault(int(h), set()).add(int(f))
    check(all(len(v) == 1 for v in mp.values()), f"one fold per house ({len(mp)} houses)")
    check(P.fold_hash(fa) == "c8b301a92149", f"fold hash {P.fold_hash(fa)}")
    return [dict(house=h, fold=next(iter(f))) for h, f in sorted(mp.items())]


def store_eps(d: Path):
    out = set()
    for f in d.glob("ep*_h*.npz"):
        e, h = f.stem[2:].split("_h")
        out.add((int(e), int(h)))
    return out


def write_csv(path: Path, rows, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})


def read_csv(path: Path):
    with open(path, newline="") as f:
        return [{k: int(v) for k, v in r.items()} for r in csv.DictReader(f)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--validity", type=Path)
    ap.add_argument("--rollouts", type=Path)
    ap.add_argument("--accounting", type=Path)
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--freeze-store", type=Path)
    ap.add_argument("--decode-store", type=Path)
    ap.add_argument("--fold-bank", type=Path)
    ap.add_argument("--trajectory", type=Path)
    ap.add_argument("--write", type=Path, default=None, help="write the lists into this directory")
    a = ap.parse_args(argv)
    lists = paths.lists_dir()
    out = {}

    if a.validity and a.rollouts and a.accounting:
        cat, houses = catalog(a.validity)
        print(f"catalog: {len(cat)} entries = {len(houses)} valid houses x {N_ROUNDS} rounds")
        rows = natural_rows(cat, a.rollouts, a.accounting)
        out["phase/natural_episodes.csv"] = (rows, ["episode_index", "partition", "house", "seed",
                                                    "new_sampler"])
        n_new = sum(r["new_sampler"] for r in rows)
        print(f"natural episodes: {len(rows)} (new_sampler=1: {n_new})")
    if a.cache:
        dec = store_eps(a.decode_store) if a.decode_store else None
        rows = freeze_rows(a.cache, dec)
        out["phase/freeze_episodes.csv"] = (rows, ["task", "episode_index", "house", "seed",
                                                   "decode"])
        if dec is not None:
            check(sum(r["decode"] for r in rows) == len(dec),
                  f"decode set: {len(dec)} episodes, all in the catalog")
        if a.freeze_store:
            have = store_eps(a.freeze_store)
            listed = {(r["episode_index"], r["house"]) for r in rows}
            check(have <= listed, f"freeze store: {len(have)} episodes, all listed")
            print(f"  listed but not in the store: {sorted(listed - have)}")
            if dec is not None:
                check(dec <= have, "decode set within the freeze store")
        if a.fold_bank:
            out["folds/phase.csv"] = (fold_rows(a.cache, a.fold_bank), ["house", "fold"])
    if a.trajectory:
        z = np.load(a.trajectory, allow_pickle=True)
        row = dict(episode_index=int(z["ep"]), house=int(z["house"]), seed=int(z["seed"]),
                   new_sampler=0)
        out["phase/trajectory_episode.csv"] = ([row], ["episode_index", "house", "seed",
                                                       "new_sampler"])

    for name, (rows, cols) in out.items():
        if a.write:
            write_csv(a.write / name, rows, cols)
            print(f"wrote {a.write / name} ({len(rows)} rows)")
        else:
            shipped = read_csv(lists / name)
            check(shipped == [{c: int(r[c]) for c in cols} for r in rows],
                  f"lists/{name}: {len(shipped)} rows equal the derivation")
    if FAILS:
        print(f"{len(FAILS)} check(s) FAILED")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
