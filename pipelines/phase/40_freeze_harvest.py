"""Phase, stage 40 (GPU + EGL): the freeze test of one episode.

Task K of lists/phase/freeze_episodes.csv (K = 0..127; one episode per task, sampled by a task
sampler created for it). The episode is rolled out as in stage 10 (stop after the first
successful step, at most 520 steps), the sim and policy state is kept at every step, and its query
steps are labelled from the rollout's own per-query records (vlageom.phase.label_queries). At up to
10 landmark queries (vlageom.phase.pick_landmarks: up to 3 reach, the middle open- and closed-hand
grasp, the first and last transport, the first and last closed-hand place, the first open-hand
place; 3 to 9 in the paper's episodes) the policy is queried again under each condition
(vlageom.phase.perturb_landmark). Every landmark is recorded; the analyses use, of the closed-hand
place landmarks, the last one (vlageom.phase.used_landmarks, stages 50 and 70). Every condition
restores the landmark's sim and policy state and arm pose; the robot-mounted cameras are moved to
the restored pose before the observation is rendered:
  entry     a fresh query on the landmark's observation (current frame and the frame 8 steps
            earlier: "natural")
  pause1/2/4  1, 2 or 4 extra queries on the held observation, then a fresh query: after 2 the
            policy's two-frame input holds the current frame twice ("frozen")
  off, detour  (reach and open-hand grasp landmarks) the arm moved 5 cm along base x in 8 steps
            with a query per step, then put back
The fresh query of every condition uses the landmark query's flow noise. Read per condition: the
11 action-expert blocks (token mean, final flow step), the four VLM token pools and the commanded
action chunk (16 x 8).

Output: <data>/phase/freeze/episodes/ep<E>_h<H>.npz (63 arrays, vlageom.phase.freeze_payload);
an existing file is kept (--overwrite replaces it). About 4 minutes per episode on one H200.
Tasks 24 (episode 206, house 30) and 72 (episode 79, house 87) cannot be sampled by MolmoSpaces
(HouseInvalidForTask) and write no file; the store has 126 episodes.

Exit status: 0 written or present, 1 the episode failed, 3 a rendered camera frame was blank (an
EGL failure of this process; run the task again, on another node if it repeats).

  python pipelines/phase/40_freeze_harvest.py --task K      # 0..127 (default $SLURM_ARRAY_TASK_ID)
"""
from __future__ import annotations

import argparse
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

log = logging.getLogger("phase.40_freeze_harvest")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", type=int, default=None,
                    help="list row 0..127 (default $SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--out-dir", type=Path, default=paths.tile_dir("phase", "freeze", "episodes"))
    ap.add_argument("--checkpoint", default=None,
                    help="policy checkpoint (default $VLAGEOM_CHECKPOINT)")
    ap.add_argument("--max-steps", type=int, default=P.MAX_STEPS)
    ap.add_argument("--min-px", type=float, default=P.MIN_PX,
                    help="blank-frame threshold on the mean pixel value (default %(default)s)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    rows = P.read_freeze_episodes()
    if a.task is None:
        if os.environ.get("SLURM_ARRAY_TASK_ID", "") == "":
            ap.error("give --task (or run as a SLURM array task)")
        a.task = int(os.environ["SLURM_ARRAY_TASK_ID"])
    row = next((r for r in rows if r["task"] == a.task), None)
    if row is None:
        ap.error(f"--task must be 0..{len(rows) - 1}")
    e, h, s = row["episode_index"], row["house"], row["seed"]
    out = a.out_dir / P.freeze_name(e, h)
    if out.exists() and not a.overwrite:
        log.info("%s exists", out)
        return 0

    from vlageom.sim.engine import BranchEngine
    eng = BranchEngine({"config": P.CONFIG, "checkpoint": a.checkpoint,
                        "output_dir": str(paths.tile_dir("phase", "env", create=True))})
    eng.load()
    try:
        P.sample_listed(eng, h, s, 1)
        base, rec = P.rollout(eng, a.max_steps)
        lms, lab = P.run_freeze_episode(eng, base, rec, a.min_px)
        if not lms:
            log.error("ep%d h%d: no landmarks", e, h)
            return 1
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **P.freeze_payload(lms, lab, base, rec, e, h, s))
        log.info("ep%d h%d: %d landmarks -> %s", e, h, len(lms), out)
        return 0
    except P.BlankFrame as exc:
        log.error("ep%d h%d: %s; nothing written, run the task again", e, h, exc)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(3)
    except Exception as exc:
        log.exception("ep%d h%d failed: %s", e, h, exc)
        return 1
    finally:
        eng.close()


if __name__ == "__main__":
    sys.exit(main())
