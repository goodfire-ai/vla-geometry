"""Contact (Figure 4), stage 10 (GPU + EGL): natural rollouts, unpooled VLM tokens at every query.

For each listed episode of lists/contact/natural_episodes.csv (299 episodes, 192 houses):

  1. The episode is sampled by a newly created MolmoSpaces task sampler with all RNGs seeded by
     `seed` (FrankaPickDroidDataGenConfig; vlageom.gripper.sample_episode with new_sampler=1) and
     MolmoBot-DROID runs closed loop for at most 600 steps, with a snapshot of the simulator and
     policy state at every step and the model inputs of every query (one query every 8 steps).
  2. Contact at every step: each snapshot is restored and gripper-object contact is read
     (`contact_series`; vlageom.contact.replay_contact_series).
  3. For every query, the VLM backbone is run again on the query's inputs with forward hooks on
     transformer.blocks.{0,4,8,12,16,20,24,28,32,35}. The input holds the current exo and wrist
     frames and, from query 1 on, the frames of the previous query (8 steps earlier). Stored per
     query in float16: the outputs at the current exo and wrist image patches (196 + 196), at the
     text tail, and the mean over all tokens of the sequence.

Output (under --out, default $VLAGEOM_DATA/contact): natural/ep<E>_h<H>.npz, keys
vlageom.contact.NATURAL_KEYS:
  act_img (P, 10, 392, 2560) f16, act_txt (P, 10, T, 2560) f16, act_seqmean (P, 10, 2560) f16,
  txt_ids (T,), txt_str (T,) decoded tail tokens, contact_series (n_steps,), q_idx,
  query_steps_kept, layout_imgs (2 image blocks at query 0, 4 later), replay_grasp (contact at
  the query, read during the rollout), replay_grip (executed gripper command), layers, house, ep,
  seed, n_queries, n_steps, base_success, descr (the listed task description).
About 1.2 GB per episode, 369 GB for the 299 episodes.

Sharding: one listed episode per task; --task K (default $SLURM_ARRAY_TASK_ID) runs row K of the
list, --episode-index runs the given episodes in one process (each sampled by a new task sampler).
Existing outputs are skipped (--overwrite to redo). 2-4 minutes per episode on one H200, including
about 30 s to load the model.

Exit status: 0 done, 1 an episode failed, 3 the first query's exo-camera frame is dark (mean pixel
value below 20, an EGL failure of this process; nothing is written, run it again on another node).

  python pipelines/contact/10_natural_harvest.py --task 0          # rows 0..298
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
from vlageom.sim import get_tokenizer  # noqa: E402

log = logging.getLogger("contact.10_natural_harvest")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n", 1)[1])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--task", type=int, default=None,
                   help="row of lists/contact/natural_episodes.csv (default $SLURM_ARRAY_TASK_ID)")
    g.add_argument("--episode-index", type=int, nargs="+", default=None,
                   help="run these listed episodes (by episode_index)")
    ap.add_argument("--episodes", type=Path, default=None,
                    help="episode list (default lists/contact/natural_episodes.csv)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output root (default $VLAGEOM_DATA/contact); files go to <out>/natural")
    ap.add_argument("--checkpoint", default=None, help="policy checkpoint (default $VLAGEOM_CHECKPOINT)")
    ap.add_argument("--max-steps", type=int, default=C.MAX_STEPS)
    ap.add_argument("--n-queries", type=int, default=C.N_QUERIES,
                    help="queries kept per episode, evenly spaced (default 128 = all)")
    ap.add_argument("--min-px", type=float, default=C.MIN_PX,
                    help="minimum mean exo-camera pixel value of the first query (default %(default)s)")
    ap.add_argument("--overwrite", action="store_true", help="redo episodes whose file exists")
    a = ap.parse_args(argv)

    rows = C.read_natural_episodes(a.episodes)
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
    out_dir = C.natural_dir(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in sel if a.overwrite or not (out_dir / C.shard_name(r["episode_index"], r["house"])).exists()]
    for r in sel:
        if r not in todo:
            log.info("ep%d h%d: file exists, skipped", r["episode_index"], r["house"])
    if not todo:
        return 0

    from vlageom.gripper import sample_episode
    from vlageom.sim import BranchEngine
    root = Path(a.out) if a.out is not None else paths.tile_dir(C.TILE)
    eng = BranchEngine({"config": C.CONFIG, "checkpoint": a.checkpoint,
                        "output_dir": str(root / "_env")})
    eng.load()
    tok = get_tokenizer(a.checkpoint)
    n_fail = 0
    try:
        for r in todo:
            ep, house, seed = r["episode_index"], r["house"], r["seed"]
            dst = out_dir / C.shard_name(ep, house)
            try:
                sample_episode(eng, house, seed, new_sampler=1)
                base = eng.run_baseline(a.max_steps)
                log.info("ep%d h%d seed%d: %d steps, %d queries, success=%s", ep, house, seed,
                         base["n_steps"], base["n_queries"], base.get("success"))
                if base["query_steps"]:
                    px = C.exo_mean_px(eng._baseline_snaps[base["query_steps"][0]]["obs"])
                    if np.isfinite(px) and px < a.min_px:
                        log.error("ep%d h%d: first-query exo frame is dark (mean %.1f < %.1f); "
                                  "nothing written", ep, house, px, a.min_px)
                        return 3
                payload = C.natural_payload(eng, tok, base, ep, house, seed, r["task_description"],
                                            C.LAYERS, a.n_queries)
                if payload is None:
                    log.warning("ep%d h%d: no queries; nothing written", ep, house)
                    continue
                tmp = dst.with_name("_tmp_" + dst.name)
                np.savez(tmp, **payload)
                os.replace(tmp, dst)
                log.info("ep%d h%d -> %s (%d queries, T=%d, contact on %.3f of the steps)", ep, house,
                         dst, len(payload["q_idx"]), len(payload["txt_ids"]),
                         float(payload["contact_series"].mean()))
            except Exception as e:
                n_fail += 1
                log.exception("ep%d h%d failed: %s", ep, house, e)
    finally:
        eng.close()
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
