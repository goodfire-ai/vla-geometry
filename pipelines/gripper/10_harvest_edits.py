"""Stage 10 (GPU + EGL): policy rollouts and gripper edits at action-expert layer 32.

For each episode (episode_index, house, seed, new_sampler) of lists/gripper_episodes.csv:

  1. The episode is sampled (FrankaPickDroidDataGenConfig, all RNGs seeded with `seed`, by a newly
     created or an existing MolmoSpaces task sampler as `new_sampler` says; see
     vlageom.gripper.sample_episode) and MolmoBot-DROID runs closed loop for up to 600 steps. At
     every model query (every 8 steps) the model inputs are kept, together with the executed
     gripper command, gripper-object contact, object height and arm joint angles.
  2. Stable-grasp onset = the first query where the gripper holds the object for 3 queries in a row
     (with a lift of more than 2 cm when the object is lifted at all). Up to 3 gripper-open queries
     3-9 queries before the onset (before_grasp, edit target: close) and up to 3 closed-and-holding
     queries 3-9 queries after it (after_grasp, edit target: open) are picked, spread evenly; when
     none lies in that range, any open query before / closed-and-holding query after the onset is
     used (vlageom.steer.landmarks.find_landmark_windows). Episodes without a stable grasp write
     nothing.
  3. At each picked query the edit (768,) added to the output of action-expert block 32 at the last
     flow step is fitted with Adam (1000 iterations, lr 0.1): loss = 1e-4 |delta|^2
     + 1 * mean((gripper - target)^2) + 300 * mean((arm - arm at this query)^2)
     + 3000 * mean((arm[t+1] - arm[t])^2) over the 16 decoded actions. The flow noise of the query is
     the one the rollout used (seed + 1000 * query).

Output (under --out, default $VLAGEOM_DATA/gripper):
  deltas/ep<E>_h<H>_<landmark>_q<Q>_aw300_sw3000.npz   one file per edit (keys: vlageom.gripper.EDIT_KEYS)
  captures/ep<E>_h<H>.pt   with --save-captures: the per-query model inputs of each grasping episode
                           (about 110 MB each), from which --from-captures fits again without a rollout

The scene of an episode is set by its (house, seed, new_sampler) row alone, so episodes can be
split into tasks in any way. Episodes that already have edit files are skipped (--overwrite fits
them again), so a job array task can simply be restarted. With --save-captures an episode is
skipped only when its model-input file exists as well; otherwise it is rolled out again and only
its model-input file is written (its edit files are kept).

Sharding: the listed episodes (in list order) are split into tasks of --per-task episodes; run task
--task (default $SLURM_ARRAY_TASK_ID, else 0). --episode-index runs the given episodes instead.
About 10 minutes per episode on one H200 (a 2-minute rollout and about 1.5 minutes per edit), plus
about 2 minutes to load the model per process.
Example (one episode): MUJOCO_GL=egl python pipelines/gripper/10_harvest_edits.py --episode-index 0
"""
from __future__ import annotations

import vlageom.sim  # noqa: F401  (first: sets up the file locks before molmo_spaces is imported)

import argparse
import logging
import os
import sys
from pathlib import Path

from vlageom import gripper as G
from vlageom import paths
from vlageom.sim import BranchEngine

log = logging.getLogger("gripper.harvest_edits")


def select(episodes, a):
    if a.episode_index:
        want = set(a.episode_index)
        sel = [e for e in episodes if e[0] in want]
        missing = want - {e[0] for e in sel}
        if missing:
            raise SystemExit(f"episode_index not in the list: {sorted(missing)}")
        return sel
    task = a.task
    if task is None:
        env_id = os.environ.get("SLURM_ARRAY_TASK_ID", "")
        task = int(env_id) if env_id else 0
    return episodes[task * a.per_task:(task + 1) * a.per_task]


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n", 1)[1])
    ap.add_argument("--episodes", type=Path, default=None,
                    help="episode list CSV (episode_index, house, seed); "
                         "default lists/gripper_episodes.csv")
    ap.add_argument("--out", type=Path, default=paths.tile_dir("gripper"),
                    help="output root (default: $VLAGEOM_DATA/gripper)")
    ap.add_argument("--checkpoint", default=None,
                    help="policy checkpoint (default $VLAGEOM_CHECKPOINT)")
    ap.add_argument("--task", type=int, default=None,
                    help="task index (default: $SLURM_ARRAY_TASK_ID, else 0)")
    ap.add_argument("--per-task", type=int, default=2, help="episodes per task (default 2)")
    ap.add_argument("--episode-index", type=int, nargs="+", default=None,
                    help="run these listed episodes (by episode_index) instead of a task")
    ap.add_argument("--save-captures", action="store_true",
                    help="also save the per-query model inputs to <out>/captures/ep<E>_h<H>.pt")
    ap.add_argument("--from-captures", type=Path, default=None,
                    help="directory of ep<E>_h<H>.pt files: fit from these model inputs instead of "
                         "running the rollout (no per-step rendering)")
    ap.add_argument("--overwrite", action="store_true",
                    help="fit episodes that already have edit files again")
    a = ap.parse_args()

    episodes = G.read_episodes(a.episodes)
    sel = select(episodes, a)
    if not sel:
        log.warning("no episodes for task %s (%d listed, %d per task)", a.task, len(episodes),
                    a.per_task)
        return
    edits_dir = a.out / "deltas"
    edits_dir.mkdir(parents=True, exist_ok=True)
    cap_dir = a.out / "captures"
    if a.save_captures:
        cap_dir.mkdir(parents=True, exist_ok=True)

    eng = BranchEngine({"config": G.CONFIG, "checkpoint": a.checkpoint,
                        "output_dir": str(a.out / "_env")})
    eng.load()
    log.info("%d episode(s): %s; layer %d, %s; mode %s", len(sel),
             " ".join(f"ep{e}_h{h}" for e, h, *_ in sel), G.LAYER, G.FIT,
             f"from captures in {a.from_captures}" if a.from_captures else "rollout")

    n_fail = 0
    for ep, house, seed, new_sampler in sel:
        have_edits = bool(list(edits_dir.glob(f"ep{ep}_h{house}_*.npz")))
        need_capture = (a.save_captures and a.from_captures is None
                        and not (cap_dir / f"ep{ep}_h{house}.pt").exists())
        if not a.overwrite and have_edits and not need_capture:
            log.info("ep%d house%d seed%d: edits exist, skipped", ep, house, seed)
            continue
        try:
            G.sample_episode(eng, house, seed, new_sampler)
            cap_path = None
            if a.from_captures is not None:
                src = a.from_captures / f"ep{ep}_h{house}.pt"
                if not src.exists():
                    log.warning("ep%d house%d seed%d: no capture %s, skipped", ep, house, seed, src)
                    continue
                base = eng.load_capture(src)
            else:
                base = eng.run_baseline(G.MAX_STEPS)
                if a.save_captures:
                    cap_path = cap_dir / f"ep{ep}_h{house}.pt"
            G.fit_episode(eng, base, ep, house, seed, edits_dir, capture_path=cap_path,
                          overwrite=a.overwrite, fit=a.overwrite or not have_edits)
        except Exception as e:
            n_fail += 1
            log.exception("ep%d house%d seed%d failed: %s", ep, house, seed, e)
    eng.close()
    log.info("done -> %s", edits_dir)
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
