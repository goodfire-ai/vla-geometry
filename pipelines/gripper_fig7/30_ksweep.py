"""Stage 30 (GPU): edits restricted to the top k components of the other houses' edits (Figure 7d).

For each listed episode (lists/gripper_episodes.csv):

  1. Bases: the edits of the listed episodes (their edit files in --edits) of all houses except
     the episode's house, their mean mu and principal components V (torch.linalg.svd on the GPU,
     float32, of the edits minus mu), and the mean closing and opening edit.
  2. Model inputs: the per-query inputs of the episode's rollout, read from
     <--from-captures>/ep<E>_h<H>.pt (written by pipelines/gripper/10_harvest_edits.py
     --save-captures; no rendering), or, with --rollout, from a new rollout of the episode
     (GPU + EGL). The scene is sampled as in pipelines/gripper/10_harvest_edits.py
     (vlageom.gripper.sample_episode); with stored inputs the fits do not depend on the scene.
  3. One query per grasp landmark (vlageom.steer.landmarks.find_landmarks): before_grasp, the
     query 3 before the stable-grasp onset (or the nearest earlier query with the gripper open),
     target close; after_grasp, the query 3 after the onset within the closed-and-holding run,
     target open.
  4. At each query, for k = 0, 1, 2, 4, ..., 256, 384, 512, 640, 768: the gripper-edit loss of
     pipelines/gripper/10_harvest_edits.py (weights 1 / 300 / 3000, norm 1e-4, arm target = joint
     angles at the query) minimized over a in delta = mu + V[:, :k] a with Adam (1000 iterations,
     lr 0.1), starting from the coordinates of the mean closing (or opening) edit; k = 0 evaluates
     delta = mu. Then the unrestricted 768-D edit at the same query (the reference of Figure 7d).

Output (--out, default $VLAGEOM_DATA/gripper_fig7/ksweep): one JSON record per landmark,
ep<E>_h<H>_<landmark>.json, with the episode, landmark and query, `results` {k: metrics of the fit
(flip, grip_before, grip_after, arm_leakage, arm_jerk, nat_jerk, ..., norm, coeffs, k_eff, loss)}
and `ceiling_inrun` (the same metrics of the 768-D edit).

Sharding: the listed episodes (in list order) are split into tasks of --per-task episodes (default
1); run task --task (default $SLURM_ARRAY_TASK_ID, else 0), or --episode-index for given episodes.
Records that exist are kept (--overwrite fits again). About 40 minutes per episode on one H200 (two
landmarks x 14 restricted fits + the 768-D fit, 1000 iterations each), plus about 2 minutes to load
the model per process; 304 episodes, about 210 GPU-hours.
Example (one episode): python pipelines/gripper_fig7/30_ksweep.py --episode-index 0
"""
from __future__ import annotations

import vlageom.sim  # noqa: F401  (first: sets up the file locks before molmo_spaces is imported)

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

from vlageom import gripper as G
from vlageom import gripper_fig7 as F
from vlageom import paths
from vlageom.sim import BranchEngine

log = logging.getLogger("gripper_fig7.ksweep")


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
                    help="episode list CSV (default lists/gripper_episodes.csv)")
    ap.add_argument("--edits", type=Path, default=paths.tile_dir("gripper", "deltas"),
                    help="directory of the listed episodes' edit files, from which the bases are "
                         "built (default: $VLAGEOM_DATA/gripper/deltas)")
    ap.add_argument("--from-captures", type=Path, default=paths.tile_dir("gripper", "captures"),
                    help="directory of ep<E>_h<H>.pt model-input files "
                         "(default: $VLAGEOM_DATA/gripper/captures)")
    ap.add_argument("--rollout", action="store_true",
                    help="roll the episodes out instead of reading --from-captures (GPU + EGL)")
    ap.add_argument("--out", type=Path, default=paths.tile_dir(F.TILE, "ksweep"),
                    help="record directory (default: $VLAGEOM_DATA/gripper_fig7/ksweep)")
    ap.add_argument("--checkpoint", default=None,
                    help="policy checkpoint (default $VLAGEOM_CHECKPOINT)")
    ap.add_argument("--task", type=int, default=None,
                    help="task index (default: $SLURM_ARRAY_TASK_ID, else 0)")
    ap.add_argument("--per-task", type=int, default=1, help="episodes per task (default 1)")
    ap.add_argument("--episode-index", type=int, nargs="+", default=None,
                    help="run these listed episodes (by episode_index) instead of a task")
    ap.add_argument("--overwrite", action="store_true", help="fit records that exist again")
    a = ap.parse_args()

    import torch

    episodes = G.read_episodes(a.episodes)
    sel = select(episodes, a)
    if not sel:
        log.warning("no episodes for task %s (%d listed, %d per task)", a.task, len(episodes),
                    a.per_task)
        return

    # bases
    files = G.edit_files(a.edits, episodes)
    if not files:
        raise SystemExit(f"no edit files in {a.edits}")
    E = G.load_edits(files)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bases = F.LohoBases(E["D"], E["house"], E["target"], dev)
    _, heldout = F.house_split({h for _, h, *_ in episodes})
    log.info("bases from %d edits (%d houses) in %s; %d episode(s): %s; device %s", len(files),
             len(np.unique(E["house"])), a.edits, len(sel),
             " ".join(f"ep{e}_h{h}" for e, h, *_ in sel), dev)

    a.out.mkdir(parents=True, exist_ok=True)
    eng = BranchEngine({"config": G.CONFIG, "checkpoint": a.checkpoint,
                        "output_dir": str(a.out.parent / "_env")})
    eng.load()

    n_fail = 0
    for ep, house, seed, new_sampler in sel:
        names = [F.record_name(ep, house, lm) for lm in F.LANDMARKS]
        if not a.overwrite and all((a.out / n).exists() for n in names):
            log.info("ep%d house%d: records exist, skipped", ep, house)
            continue
        try:
            G.sample_episode(eng, house, seed, new_sampler)
            if a.rollout:
                base = eng.run_baseline(G.MAX_STEPS)
            else:
                src = a.from_captures / f"ep{ep}_h{house}.pt"
                if not src.exists():
                    n_fail += 1
                    log.error("ep%d house%d: no model-input file %s", ep, house, src)
                    continue
                base = eng.load_capture(src)
            F.ksweep_episode(eng, base, bases, ep, house, seed, a.out, heldout_houses=heldout,
                             overwrite=a.overwrite)
        except Exception as e:
            n_fail += 1
            log.exception("ep%d house%d seed%d failed: %s", ep, house, seed, e)
    eng.close()
    log.info("done -> %s", a.out)
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
