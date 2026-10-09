"""Phase, stage 60 (GPU + EGL): the example trajectory of the figure's top strip and panel b.

Rolls out, in one process, the two episodes of lists/phase/trajectory_preceding.csv and then the
episode of lists/phase/trajectory_episode.csv (house 76, seed 20001076, sampled by the existing task
sampler: new_sampler 0), as in stage 10. The task sampler carries state from one episode to the
next, so the example episode is sampled after the same two episodes as in the paper's capture.
For the example episode it keeps, per model query, the exo and wrist
images the policy read at that query (from its observations), the activations of that query (as in
stage 10) and the phase labels of the rollout's own per-step records (vlageom.phase.label_steps)
at the query steps. In the paper's capture this rollout succeeds after 120 steps with 15 queries:
4 reach, 2 grasp, 5 transport, 4 place.

Output: <data>/phase/trajectory/traj_ep<E>_h<H>.npz (40 arrays, see
vlageom.phase.capture_episode). About 9 minutes on one H200 (three rollouts), plus model loading.

Exit status: 0 written, 1 the rollout produced no usable capture, 2 written but the rollout did
not succeed or has fewer than 2 queries of some phase (the panels need every phase).

  python pipelines/phase/60_trajectory_capture.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import numpy as np  # noqa: E402

import vlageom.sim  # noqa: E402,F401  (before anything imports molmo_spaces)
from vlageom import paths  # noqa: E402
from vlageom import phase as P  # noqa: E402

log = logging.getLogger("phase.60_trajectory_capture")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None,
                    help="default <data>/phase/trajectory/traj_ep<E>_h<H>.npz")
    ap.add_argument("--checkpoint", default=None,
                    help="policy checkpoint (default $VLAGEOM_CHECKPOINT)")
    ap.add_argument("--max-steps", type=int, default=P.MAX_STEPS)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    t = P.read_trajectory_episode()
    out = a.out or paths.tile_dir("phase", "trajectory",
                                  f"traj_ep{t['episode_index']}_h{t['house']}.npz")

    from vlageom.sim.engine import BranchEngine
    eng = BranchEngine({"config": P.CONFIG, "checkpoint": a.checkpoint,
                        "output_dir": str(paths.tile_dir("phase", "env", create=True))})
    eng.load()
    try:
        for p in P.read_trajectory_preceding():
            log.info("preceding rollout ep%d h%d", p["episode_index"], p["house"])
            P.capture_episode(eng, p["episode_index"], p["house"], p["seed"], p["new_sampler"],
                              a.max_steps)
            P.free_episode(eng)
        payload, spans, success = P.capture_episode(eng, t["episode_index"], t["house"], t["seed"],
                                                    t["new_sampler"], a.max_steps)
    finally:
        eng.close()
    if payload is None:
        log.error("no usable capture")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    log.info("saved %s: success %s, %d queries, query counts %s", out, success,
             int(payload["n_queries"]), json.dumps(spans))
    if not success or min(spans.values()) < 2:
        log.warning("the rollout did not succeed or misses a phase: %s", spans)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
