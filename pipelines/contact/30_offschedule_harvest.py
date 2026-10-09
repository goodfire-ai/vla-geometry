"""Contact (Figure 4), stage 30 (GPU + EGL): off-schedule contact, early touches and induced slips.

For one row of lists/contact/offschedule_episodes.csv (128 episodes in 128 houses: for each house
in increasing order, its successful natural rollout with the lowest episode index; houses 0-132
except 2, 44, 60, 61 and 75, which have no successful rollout):

  1. The episode is sampled by a newly created MolmoSpaces task sampler with all RNGs seeded by
     `seed` and rolled out for at most 600 steps, with per-step snapshots (as in stage 10).
  2. Anchors: up to 4 queries with the hand open (gripper command < 127, not the first or last
     query; vlageom.sim.kinematics.pick_poses) and up to 4 queries where the gripper touches the
     object (vlageom.contact.pick_holding_poses), each spread evenly over the candidates.
  3. Four events per anchor kind, in this order (vlageom.contact.offschedule_payload):
       event 0  early touch, hand open: the object moves toward the hand until it touches
       event 2  early touch, hand forced closed (finger joints at their closed limit)
       event 1  slip, hand closed: the object moves away from the hand until contact ends
       event 3  slip, hand forced open: away from the hand when the opened hand still touches the
                object (branch +1), else toward the hand until it touches again (branch -1)
     At every placement the snapshot of the anchor's step is restored, the arm is set to the
     anchor's joint angles (and the fingers to the forced state), and the object is moved along the
     hand-object line in 1 cm steps, k = 1 .. 24 (40 with a forced hand), until contact changes and
     3 steps more (vlageom.contact.march_boundary). Placements where the object did not follow the
     command are skipped; an (anchor, event) whose placements are all on one side of the contact
     boundary is dropped.
  4. Every kept placement (k = 0, the unmoved scene, included) is rendered with the robot-mounted
     cameras moved to the current hand pose, the policy is queried once with the anchor query's
     flow noise, and the VLM backbone runs on that query's inputs with hooks on
     transformer.blocks.{0,4,8,12,16,20,24,28,32,35} (vlageom.contact.PlacementEncoder). The
     snapshot restored for a placement includes the policy's frame history and the policy is not
     reset, so the query reads the current (re-posed) exo and wrist frames plus the rollout's own
     frames of the previous query, 8 steps earlier; anchors are at query 1 or later, and within one
     pose these earlier frames are the same at every placement.

Output (under --out, default $VLAGEOM_DATA/contact): offschedule/ep<E>_h<H>.npz, keys
vlageom.contact.OFFSCHEDULE_KEYS:
  act (Npts, 10, 392 + T + 1, 2560) f16: exo + wrist patches, text tail, sequence mean;
  per placement: contact, disp (m, > 0 = away from the hand), dist (hand-object distance, m),
  pose_id, step_k, event, phase (anchor step / (n_steps - 1)), qc (anchor query);
  per pose: grasp0, pose_chunk, qpos0, pose_event, pose_qc, pose_sep0 (finger separation, m),
  pose_grasp_forced (contact before any move), pose_dir (+1 away from / -1 toward the hand);
  layers, house, ep, n_txt, txt_ids, txt_str (empty).
About 2.9 GB per episode, 352 GB for the 123 episodes that give a file. The list's `outcome`
column says which episodes give none: houses 30, 104 and 106 have no valid robot placement
("no_scene", MolmoSpaces raises HouseInvalidForTask) and episodes 231 and 525 have no anchor that
crosses the contact boundary ("no_crossing").

Sharding: one listed episode per task; --task K (default $SLURM_ARRAY_TASK_ID) runs row K,
--episode-index runs the given episodes in one process (each sampled by a new task sampler).
Existing outputs are skipped (--overwrite to redo). About 4-6 minutes per episode on one H200.

Exit status: 0 done (including the listed episodes that give no file), 1 an episode failed
unexpectedly, 3 the first query's exo-camera frame is dark (EGL failure of this process).

  python pipelines/contact/30_offschedule_harvest.py --task 0      # rows 0..127
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import vlageom.sim  # noqa: E402,F401  (first: sets up the file locks before molmo_spaces is imported)

import argparse  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from vlageom import contact as C  # noqa: E402
from vlageom import paths  # noqa: E402
from vlageom.sim import GripperControl  # noqa: E402

log = logging.getLogger("contact.30_offschedule_harvest")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n", 1)[1])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--task", type=int, default=None,
                   help="row of lists/contact/offschedule_episodes.csv (default $SLURM_ARRAY_TASK_ID)")
    g.add_argument("--episode-index", type=int, nargs="+", default=None,
                   help="run these listed episodes (by episode_index)")
    ap.add_argument("--episodes", type=Path, default=None,
                    help="episode list (default lists/contact/offschedule_episodes.csv)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output root (default $VLAGEOM_DATA/contact); files go to <out>/offschedule")
    ap.add_argument("--checkpoint", default=None, help="policy checkpoint (default $VLAGEOM_CHECKPOINT)")
    ap.add_argument("--max-steps", type=int, default=C.MAX_STEPS)
    ap.add_argument("--n-pose", type=int, default=C.N_POSE, help="anchors per kind (default 4)")
    ap.add_argument("--step-m", type=float, default=C.STEP_M, help="object step (m, default 0.01)")
    ap.add_argument("--k-max", type=int, default=C.K_MAX, help="step cap, hand as it is (default 24)")
    ap.add_argument("--k-max-forced", type=int, default=C.K_MAX_FORCED,
                    help="step cap, forced hand state (default 40)")
    ap.add_argument("--min-px", type=float, default=C.MIN_PX,
                    help="minimum mean exo-camera pixel value of the first query (default %(default)s)")
    ap.add_argument("--verbose", action="store_true", help="log every placement")
    ap.add_argument("--overwrite", action="store_true", help="redo episodes whose file exists")
    a = ap.parse_args(argv)

    rows = C.read_offschedule_episodes(a.episodes)
    if a.episode_index:
        want = set(a.episode_index)
        sel = [r for r in rows if r["episode_index"] in want]
        missing = want - {r["episode_index"] for r in sel}
        if missing:
            ap.error(f"episode_index not in the list: {sorted(missing)}")
    else:
        task = a.task
        if task is None:
            env_id = os.environ.get("SLURM_ARRAY_TASK_ID", "")
            if env_id == "":
                ap.error("give --task or --episode-index (or run as a SLURM array task)")
            task = int(env_id)
        if not 0 <= task < len(rows):
            ap.error(f"--task must be in 0..{len(rows) - 1}")
        sel = [rows[task]]
    out_dir = C.offschedule_dir(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for r in sel:
        if not a.overwrite and (out_dir / C.shard_name(r["episode_index"], r["house"])).exists():
            log.info("ep%d h%d: file exists, skipped", r["episode_index"], r["house"])
        else:
            todo.append(r)
    if not todo:
        return 0

    from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask

    from vlageom.gripper import sample_episode
    from vlageom.sim import BranchEngine
    root = Path(a.out) if a.out is not None else paths.tile_dir(C.TILE)
    eng = BranchEngine({"config": C.CONFIG, "checkpoint": a.checkpoint,
                        "output_dir": str(root / "_env")})
    eng.load()
    encode = C.PlacementEncoder(C.LAYERS)
    gripper = GripperControl(eng)
    n_fail = 0
    try:
        for r in todo:
            ep, house, seed = r["episode_index"], r["house"], r["seed"]
            dst = out_dir / C.shard_name(ep, house)
            try:
                try:
                    sample_episode(eng, house, seed, new_sampler=1)
                except HouseInvalidForTask as e:
                    log.warning("ep%d h%d: no valid scene (%s); nothing written (listed outcome: %s)",
                                ep, house, e, r["outcome"])
                    if r["outcome"] != "no_scene":
                        n_fail += 1
                    continue
                gripper.reset()
                base = eng.run_baseline(a.max_steps)
                log.info("ep%d h%d seed%d: %d steps, %d queries, %d holding", ep, house, seed,
                         base["n_steps"], base["n_queries"], sum(bool(g) for g in base["grasp_q"]))
                if base["query_steps"]:
                    px = C.exo_mean_px(eng._baseline_snaps[base["query_steps"][0]]["obs"])
                    if np.isfinite(px) and px < a.min_px:
                        log.error("ep%d h%d: first-query exo frame is dark (mean %.1f < %.1f); "
                                  "nothing written", ep, house, px, a.min_px)
                        return 3
                payload = C.offschedule_payload(eng, base, ep, house, encode, gripper,
                                                n_pose=a.n_pose, step_m=a.step_m, k_max=a.k_max,
                                                k_max_forced=a.k_max_forced, verbose=a.verbose)
                if payload is None:
                    log.warning("ep%d h%d: no anchor crosses the contact boundary; nothing written "
                                "(listed outcome: %s)", ep, house, r["outcome"])
                    if r["outcome"] != "no_crossing":
                        n_fail += 1
                    continue
                tmp = dst.with_name("_tmp_" + dst.name)
                np.savez(tmp, **payload)
                os.replace(tmp, dst)
                ev = payload["event"]
                log.info("ep%d h%d -> %s (%d placements; events 0/1/2/3: %d/%d/%d/%d)", ep, house,
                         dst, len(payload["act"]), *(int((ev == e).sum()) for e in (0, 1, 2, 3)))
            except Exception as e:
                n_fail += 1
                log.exception("ep%d h%d failed: %s", ep, house, e)
    finally:
        eng.close()
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
