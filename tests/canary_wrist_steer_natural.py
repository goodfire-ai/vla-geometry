"""Compare natural episode files (a canary run of pipelines/wrist_steer/10_natural_rollouts.py) with a
reference rollout store, episode by episode and query by query.

Reference layout (the paper's natural rollout store):
  <ref-root>/episodes_extra/episode_<index>.npz   per-query records (seed, house, success, query
                                                   steps, predicted chunk, EE pose and Jacobian,
                                                   object position, joint velocities, ...)
  <ref-root>/episodes/episode_<index>.json        per-episode record (n_steps, success, house_index,
                                                   query_steps, task_description, proprioception
                                                   with the pre-step ee_pose of every step)
  --ref-cache FILE                                 natural cache of the reference (X, Xm, R, pos,
                                                   phase, house, ep, q, success, layers)

Per episode it checks: every per-query key of episodes_extra, exactly (NaN equal to NaN); n_steps,
success, house, query steps, task text and the pre-step EE pose of every step against the json;
and the natural-cloud rows of the episode (`vlageom.wrist_steer.natural_cache_rows`: token-0 and
token-mean activations of the 6 layers, R, position, phase) against the reference cache rows.

  python pipelines/wrist_steer/10_natural_rollouts.py --partition 0 --out-dir $C/natural
  python tests/canary_wrist_steer_natural.py --new $C/natural --ref-root <natural rollout store> \
      --ref-cache <natural cache> --episodes 0-15

Bit-identical activations are expected only on the GPU type of the reference (bfloat16).
Exit status 0 when everything compared is identical.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from vlageom import wrist_steer as ws

EXTRA_KEYS = ("seed", "episode_index", "house_index", "success", "n_queries", "n_query_steps",
              "query_steps", "pred_chunk", "sim_time", "qvel", "qvel_groups", "qvel_sizes", "ee_pose",
              "ee_jac", "obj_pos_base", "ee_to_obj", "obj_dist")


def parse_eps(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def same(x, y) -> bool:
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape:
        return False
    if x.dtype.kind in "fc" and y.dtype.kind in "fc":
        return bool(np.array_equal(x, y, equal_nan=True))
    return bool(np.array_equal(x, y))


def maxdiff(x, y) -> str:
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape or x.dtype.kind not in "fciu":
        return f"{x.dtype}{x.shape} vs {y.dtype}{y.shape}"
    d = np.abs(x.astype(np.float64) - y.astype(np.float64))
    return f"max|diff| {np.nanmax(d):.3e}" if d.size else "empty"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", type=Path, required=True, help="directory of the canary's episode files")
    ap.add_argument("--ref-root", type=Path, required=True,
                    help="reference store with episodes/ and episodes_extra/")
    ap.add_argument("--ref-cache", type=Path, default=None, help="reference natural cache npz")
    ap.add_argument("--episodes", default="0-15", help="episode indices, e.g. 0-15 or 192-199")
    args = ap.parse_args(argv)
    cache = dict(np.load(args.ref_cache, allow_pickle=False)) if args.ref_cache else None
    ok = True
    for ep in parse_eps(args.episodes):
        f = ws.natural_episode_path(ep, args.new)
        if not f.exists():
            print(f"episode {ep}: MISSING {f}")
            ok = False
            continue
        a = np.load(f, allow_pickle=False)
        bad = []
        ex = np.load(args.ref_root / "episodes_extra" / f"episode_{ep:04d}.npz", allow_pickle=False)
        for k in EXTRA_KEYS:
            if k not in a.files or k not in ex.files:
                bad.append(f"{k} missing")
            elif not same(a[k], ex[k]):
                bad.append(f"{k} ({maxdiff(a[k], ex[k])})")
        meta = json.loads((args.ref_root / "episodes" / f"episode_{ep:04d}.json").read_text())
        checks = {"n_steps": int(a["n_steps"]) == int(meta["n_steps"]),
                  "success(json)": bool(a["success"]) == bool(meta["success"]),
                  "house(json)": int(a["house_index"]) == int(meta["house_index"]),
                  "query_steps(json)": [int(s) for s in a["query_steps"]] == list(meta["query_steps"]),
                  "task_description": str(a["task_description"]) == meta.get("task_description", "")}
        ref_ee = np.array([np.asarray(p.get("ee_pose"), float) if p.get("ee_pose") is not None
                           else np.full((4, 4), np.nan) for p in meta.get("proprioception", [])])
        checks["step_ee_pose"] = same(a["step_ee_pose"], ref_ee.reshape(-1, 4, 4))
        bad += [k for k, v in checks.items() if not v]
        n_rows = 0
        if cache is not None:
            rows = ws.natural_cache_rows(f)
            m = np.where(cache["ep"] == ep)[0]
            n_rows = len(m)
            if len(rows) != len(m):
                bad.append(f"cache rows {len(rows)} vs reference {len(m)}")
            else:
                for (e, q, house, succ, phase, pos, R, X, Xm), i in zip(rows, m):
                    for name, x, y in (("q", q, cache["q"][i]), ("house", house, cache["house"][i]),
                                       ("success", succ, cache["success"][i]),
                                       ("phase", np.float32(phase), cache["phase"][i]),
                                       ("pos", pos, cache["pos"][i]), ("R", R, cache["R"][i]),
                                       ("X", X, cache["X"][i]), ("Xm", Xm, cache["Xm"][i])):
                        if not same(x, y):
                            bad.append(f"cache row q{q} {name} ({maxdiff(x, y)})")
        head = (f"episode {ep:4d} house {int(a['house_index']):3d} seed {int(a['seed']):4d} "
                f"{len(a['query_steps']):3d} queries {int(a['n_steps']):3d} steps")
        if bad:
            ok = False
            print(f"{head}  DIFF: " + "; ".join(bad[:12]) + (" ..." if len(bad) > 12 else ""))
        else:
            print(f"{head}  identical ({len(EXTRA_KEYS)} record keys, json fields, {n_rows} cache rows)")
    print("IDENTICAL" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
