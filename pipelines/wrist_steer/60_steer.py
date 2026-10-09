"""Wrist steering, stage 60 (GPU + EGL): chart edits in held-out scenes.

Episodes: lists/wrist_steer/steer_episodes.csv, in order (the first successful episode of each of
the first held-out houses), all in one process. Per episode: roll out the policy (at most 140
steps); the edits are written at the first query of a stable grasp, or the middle query when the
hand never holds the object (`vlageom.wrist_steer.find_landmark`). An edit for angle theta about
axis a encodes the token-mean layer-9 activation with the chart, rotates its point on the circle of
axis a by theta, decodes, and adds the decoded difference to all action tokens at the output of
action-expert block 9 at the final flow step.

  --mode oneshot (8 episodes): per episode, the decoded 8-step action chunk of the landmark query
      without an edit, with chart edits of +-20 and +-40 deg about base z, x and y, and with two
      control edits (+-40 deg with a random generator of the linear operator, stage 40); each
      measured as commanded EE rotation (angle and cosine to the written axis), position and
      gripper change. Output: <data>/wrist_steer/steer/oneshot.json.
  --mode persist (6 episodes): from the landmark, 32 env steps of the policy without an edit and
      with the edit about z (+-20, +-40 deg) recomputed from the live activation at every query;
      the hand rotation about base z after every step. Output: <data>/wrist_steer/steer/persist.json.

Each json holds {"args": settings, "rows": one row per condition, "landmarks": per episode the
landmark query and the norm of its unedited layer-9 token-mean activation, "seconds"}. The log
repeats each landmark line ('ep<E> h<H> seed<S>: landmark ... ‖base_act‖=<norm>').

  python pipelines/wrist_steer/60_steer.py --mode oneshot
  python pipelines/wrist_steer/60_steer.py --mode persist
"""
# Mathilde Papillon's steering run, adapted from her code.
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import vlageom.sim  # noqa: E402,F401  (before anything imports molmo_spaces)

from vlageom import paths  # noqa: E402
from vlageom import wrist_steer as ws  # noqa: E402

log = logging.getLogger("wrist_steer.60_steer")
DEFAULTS = {"oneshot": dict(episodes=8, axes=["z", "x", "y"], out="oneshot.json"),
            "persist": dict(episodes=6, axes=["z"], out="persist.json")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["oneshot", "persist"], required=True)
    ap.add_argument("--episodes", type=int, default=None,
                    help="first N listed episodes (default 8 for oneshot, 6 for persist)")
    ap.add_argument("--chart", type=Path, default=None, help="chart (default <data>/wrist_steer/chart.pt)")
    ap.add_argument("--operator", type=Path, default=None,
                    help="operator (default <data>/wrist_steer/operator_L9.npz)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output json (default <data>/wrist_steer/steer/<mode>.json)")
    ap.add_argument("--checkpoint", default=None, help="policy checkpoint (default $VLAGEOM_CHECKPOINT)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import torch

    d = DEFAULTS[args.mode]
    n_eps = args.episodes if args.episodes is not None else d["episodes"]
    chart_path = args.chart or ws.tile_path("chart.pt")
    op_path = args.operator or ws.tile_path("operator_L9.npz")
    out = args.out or ws.tile_path("steer", d["out"], create=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    op = ws.load_operator(op_path)
    chart = ws.Chart(chart_path, "cuda" if torch.cuda.is_available() else "cpu")
    eps = ws.read_steer_episodes()[:n_eps]
    held = {int(h) for h in op["test_houses"]}
    if any(e["house"] not in held for e in eps):
        raise SystemExit(f"listed steering houses {[e['house'] for e in eps]} are not all held out "
                         f"in {op_path}")
    log.info("%s: %d episodes on held-out houses: %s", args.mode, len(eps),
             [(e["episode_index"], e["house"]) for e in eps])
    Lrand = ws.random_generator(op)

    from vlageom.sim import BranchEngine
    eng = BranchEngine({"config": ws.CONFIG, "checkpoint": args.checkpoint,
                        "output_dir": str(paths.tile_dir(ws.TILE, "env"))})
    eng.load()
    rows, landmarks = [], []
    t0 = time.time()
    try:
        for e in eps:
            try:
                r, lm = ws.steer_episode(eng, e["episode_index"], e["house"], e["seed"],
                                         mode=args.mode, chart=chart, op=op, Lrand=Lrand,
                                         axes=tuple(d["axes"]))
            except Exception as exc:                              # noqa: BLE001
                log.exception("ep%d h%d failed: %s", e["episode_index"], e["house"], exc)
                continue
            if r is not None:
                rows.extend(r)
                landmarks.append(lm)
    finally:
        eng.close()
    settings = dict(mode=args.mode, layer=ws.LAYER, thetas=ws.STEER_THETAS, axes=d["axes"],
                    episodes=n_eps, n_exec=ws.STEER_N_EXEC, max_steps=ws.STEER_MAX_STEPS,
                    persist=ws.PERSIST_STEPS if args.mode == "persist" else 0,
                    chart=str(chart_path), operator=str(op_path))
    out.write_text(json.dumps({"args": settings, "rows": rows, "landmarks": landmarks,
                               "seconds": round(time.time() - t0, 1)}, indent=1))
    log.info("wrote %s (%d rows, %d episodes)", out, len(rows), len(landmarks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
