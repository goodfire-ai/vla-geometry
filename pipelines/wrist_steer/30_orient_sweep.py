"""Wrist steering, stage 30 (GPU + EGL): orientation sweeps of one episode.

For one row of lists/wrist_steer/orient_episodes.csv (tasks 0..199): roll out the episode with
MolmoBot-DROID (at most 600 steps), take 4 open-hand queries spread over the approach, and at each
of them turn the hand about the robot base x, y and z axes, in both directions, in closed-loop steps
of 10 deg (at most 18 per direction) with the EE position held to within 3 cm
(`vlageom.wrist_steer.step_orient`). Every pose, the unrotated one included, is rendered with the
cameras moved to the hand and read from the action expert (token mean of layers 0, 4, 8, 9, 12, 16,
20, 24, 28, 32, 35 at the final flow step, with the flow noise of that query in the rollout).

Output: <data>/wrist_steer/orient/ep<episode_index>_h<house>.npz (keys:
`vlageom.wrist_steer.ORIENT_KEYS`). An existing output is kept unless --overwrite is given. For
three listed episodes MolmoSpaces finds no valid task (houses 30, 104 and 106; tasks 57, 191 and
194); they write no file, so the store holds 197 episodes of 102 houses.

Exit status: 0 written, already present or not sampleable; 3 a rendered camera frame was blank or
missing (an EGL failure of this process; nothing is written; run the task again, on another node if
it repeats).

  python pipelines/wrist_steer/30_orient_sweep.py --task 0     # 0..199 (default $SLURM_ARRAY_TASK_ID)
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
from vlageom.util import save_npz  # noqa: E402

log = logging.getLogger("wrist_steer.30_orient_sweep")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", type=int, help="row of lists/wrist_steer/orient_episodes.csv "
                                             "(default $SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output directory (default <data>/wrist_steer/orient)")
    ap.add_argument("--checkpoint", default=None, help="policy checkpoint (default $VLAGEOM_CHECKPOINT)")
    ap.add_argument("--min-px", type=float, default=ws.MIN_PX,
                    help="minimum mean pixel value of every rendered frame (default %(default)s)")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    rows = ws.read_orient_episodes()
    if args.task is None:
        if os.environ.get("SLURM_ARRAY_TASK_ID", "") == "":
            ap.error("give --task (or run as a SLURM array task)")
        args.task = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if not 0 <= args.task < len(rows):
        ap.error(f"--task must be in 0..{len(rows) - 1}")
    ep = rows[args.task]
    out = ws.orient_path(ep["episode_index"], ep["house"], args.out_dir)
    if out.exists() and not args.overwrite:
        log.info("%s exists; pass --overwrite to replace it", out)
        return 0
    log.info("task %d: episode %d, house %d, seed %d -> %s", args.task, ep["episode_index"],
             ep["house"], ep["seed"], out)

    from vlageom.sim import BranchEngine
    eng = BranchEngine({"config": ws.CONFIG, "checkpoint": args.checkpoint,
                        "output_dir": str(paths.tile_dir(ws.TILE, "env"))})
    eng.load()
    try:
        try:
            eng.sample(ep["house"], ep["seed"])
        except Exception as e:                                    # noqa: BLE001
            log.warning("episode %d house %d cannot be sampled (%s: %s); no file written",
                        ep["episode_index"], ep["house"], type(e).__name__, e)
            return 0
        base = eng.run_baseline(ws.ORIENT_MAX_STEPS)
        payload = ws.harvest_orient_episode(eng, base, ep["episode_index"], ep["house"],
                                            min_px=args.min_px)
    except ws.DarkFrames as e:
        log.error("%s: nothing written; run the task again (on another node if it repeats)", e)
        return 3
    finally:
        eng.close()
    if payload is None:
        log.warning("episode %d house %d: no sweep rows; no file written", ep["episode_index"], ep["house"])
        return 0
    save_npz(out, payload)
    log.info("saved %s (%d rows, %d poses)", out, len(payload["act"]), len(payload["ee0"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
