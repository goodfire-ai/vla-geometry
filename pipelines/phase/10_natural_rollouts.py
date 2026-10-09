"""Phase, stage 10 (GPU + EGL): natural pick-and-place rollouts of one partition.

Runs the episodes of one partition of lists/phase/natural_episodes.csv (partitions 0..135, up to 4
episodes each) in one process, in list order. Each episode is sampled with the listed house and
seed and the listed sampler state (vlageom.phase.sample_listed: new_sampler 1 = a task sampler
created for the episode, 0 = an existing one), then MolmoBot-DROID runs with its own flow sampling
(flow noise of query q seeded with seed + 1000 q) until MolmoSpaces judges the task successful
(the rollout stops after that step) or 520 steps. Recorded before every step: the commanded
gripper, the object position, the task info (supported by the receptacle, robot contact,
position error, success), gripper-object contact and the EE position. Per model query, from the
query's own inputs (no extra render): the output of the 11 action-expert blocks
(vlageom.phase.AE_LAYERS) at the final flow step, mean over the 16 action tokens, and the VLM
blocks 21-35 (vlageom.phase.LLM_LAYERS) pooled over four token groups (current-frame patches,
non-patch tokens, the text tail, earlier-frame patches).

Output: <data>/phase/natural/episodes/ep<EEEEE>_h<H>.npz per episode (41 arrays, see
vlageom.phase.collect_episode); existing files are kept (--overwrite replaces them). About 3
minutes per episode on one H200 (rollout, then about 30 s of activation readout).

Exit status: 0 done, 1 an episode failed (the others are written). Listed episodes the sampler
cannot sample (HouseInvalidForTask) do not occur in the list.

  python pipelines/phase/10_natural_rollouts.py --partition K       # K = 0..135 (default: the
                                                                     # SLURM array task id)
  python pipelines/phase/10_natural_rollouts.py --episodes 0 1 2 3   # listed episodes, one process
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

log = logging.getLogger("phase.10_natural_rollouts")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--partition", type=int, default=None,
                    help="partition (default $SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--episodes", type=int, nargs="+", default=None, help="listed episode indices")
    ap.add_argument("--out-dir", type=Path, default=paths.tile_dir("phase", "natural", "episodes"))
    ap.add_argument("--checkpoint", default=None,
                    help="policy checkpoint (default $VLAGEOM_CHECKPOINT)")
    ap.add_argument("--max-steps", type=int, default=P.MAX_STEPS)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    eps = P.read_natural_episodes()
    if a.episodes is not None:
        idx = {e["episode_index"]: e for e in eps}
        bad = [i for i in a.episodes if i not in idx]
        if bad:
            ap.error(f"not listed: {bad}")
        mine = [idx[i] for i in a.episodes]
    else:
        if a.partition is None:
            if os.environ.get("SLURM_ARRAY_TASK_ID", "") == "":
                ap.error("give --partition or --episodes (or run as a SLURM array task)")
            a.partition = int(os.environ["SLURM_ARRAY_TASK_ID"])
        mine = [e for e in eps if e["partition"] == a.partition]
        if not mine:
            ap.error(f"partition {a.partition} has no listed episodes")
    todo = [e for e in mine if a.overwrite
            or not (a.out_dir / P.natural_name(e["episode_index"], e["house"])).exists()]
    log.info("%d episodes, %d to run -> %s", len(mine), len(todo), a.out_dir)
    if not todo:
        return 0

    from vlageom.sim.engine import BranchEngine
    eng = BranchEngine({"config": P.CONFIG, "checkpoint": a.checkpoint,
                        "output_dir": str(paths.tile_dir("phase", "env", create=True))})
    eng.load()
    failed = 0
    try:
        for e in todo:
            out = a.out_dir / P.natural_name(e["episode_index"], e["house"])
            try:
                payload, summary = P.collect_episode(eng, e["episode_index"], e["house"], e["seed"],
                                                     e["new_sampler"], a.max_steps)
                if payload is None:
                    log.error("ep%d h%d: %s", e["episode_index"], e["house"], summary)
                    failed += 1
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out, **payload)
                log.info("ep%d h%d saved: %s", e["episode_index"], e["house"], json.dumps(summary))
            except Exception as exc:
                log.exception("ep%d h%d failed: %s", e["episode_index"], e["house"], exc)
                failed += 1
            finally:
                P.free_episode(eng)
    finally:
        eng.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
