"""Wrist steering, stage 10 (GPU + EGL): natural rollouts of one partition.

Runs the natural MolmoBot-DROID rollouts of one partition of lists/wrist_steer/natural_episodes.csv
(partitions 0..12, 16 episodes each, 8 in partition 12) in one process, in episode order: seed
python `random`, numpy and torch with the listed seed, sample the listed house, roll out with the
policy's own flow sampling until the task succeeds or ends (500 steps). Before an episode whose
listed house is not its first sampling attempt, the attempts that could not be sampled are repeated
with their seeds (`vlageom.wrist_steer.sampling_attempts`). Per model query it keeps the
output of action-expert blocks 0, 4, 9, 16, 24, 32 at the final flow step (rounded to bfloat16 per
token; token 0 and the mean over the 16 action tokens), the predicted action chunk, the EE pose and
Jacobian, the object position and the robot joint velocities; per env step the EE pose
(`vlageom.wrist_steer.NATURAL_KEYS`).

Output: <data>/wrist_steer/natural/episodes/episode_<index>.npz, one per episode. Existing files are
kept (--overwrite replaces them). Since each episode starts from the task sampler its predecessors
in the process used, a partition with a missing file is rolled out from its first episode up to the
last missing one.

Exit status: 0 done, 3 a camera frame of the policy's input at a model query was blank or missing (an
EGL failure of this process; the files written before it are complete; run the partition again, on
another node if it repeats).

  python pipelines/wrist_steer/10_natural_rollouts.py --partition 0     # 0..12 (default $SLURM_ARRAY_TASK_ID)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import vlageom.sim  # noqa: E402,F401  (before anything imports molmo_spaces)

from vlageom import paths  # noqa: E402
from vlageom import wrist_steer as ws  # noqa: E402

log = logging.getLogger("wrist_steer.10_natural_rollouts")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--partition", type=int, help="partition 0..12 (default $SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="episode directory (default <data>/wrist_steer/natural/episodes)")
    ap.add_argument("--checkpoint", default=None, help="policy checkpoint (default $VLAGEOM_CHECKPOINT)")
    ap.add_argument("--min-px", type=float, default=ws.MIN_PX,
                    help="minimum mean pixel value of a query frame (default %(default)s)")
    ap.add_argument("--overwrite", action="store_true", help="replace existing episode files")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    eps = ws.read_natural_episodes()
    parts = sorted({e["partition"] for e in eps})
    if args.partition is None:
        if os.environ.get("SLURM_ARRAY_TASK_ID", "") == "":
            ap.error("give --partition (or run as a SLURM array task)")
        args.partition = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if args.partition not in parts:
        ap.error(f"--partition must be one of {parts}")
    mine = [e for e in eps if e["partition"] == args.partition]
    log.info("partition %d: episodes %d..%d -> %s", args.partition,
             min(e["episode_index"] for e in mine), max(e["episode_index"] for e in mine),
             args.out_dir or ws.tile_path("natural", "episodes"))

    from vlageom.sim.env import build_molmobot_env
    sim_env = build_molmobot_env(ws.CONFIG, args.checkpoint,
                                 output_dir=str(paths.tile_dir(ws.TILE, "env")))
    try:
        written = ws.run_natural_partition(sim_env, args.partition, eps, args.out_dir,
                                           skip_existing=not args.overwrite, min_px=args.min_px)
    except ws.DarkFrames as e:
        log.error("%s: stopping; run the partition again (on another node if it repeats)", e)
        return 3
    finally:
        sim_env.close()
    log.info("partition %d: wrote %d episode files %s", args.partition, len(written), written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
