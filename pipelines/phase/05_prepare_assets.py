"""Phase, stage 05 (one process, real file locks): install the pick-and-place receptacle assets
and the scenes of the listed houses; optionally scan which houses can host the task.

1. Receptacles. The pick-and-place task sampler picks the target receptacles of a house from the
   objaverse receptacle pool and installs them into the MolmoSpaces asset cache when it first
   needs them. The harvest stages run with no-op file locks (scripts/harvest_env.sh), so this stage
   installs, once and with the real locks, every receptacle the sampler can pick: the
   synset-valid receptacles that can be picked up, with a bounding box of at most 0.5 x 0.5 x
   0.15 m and more than 0.17 m in x and y. It stops with status 2 if an install fails.
2. Scenes. The scene of every house of lists/phase/natural_episodes.csv (181 houses; the
   freeze-test and example-rollout houses are among them), with its objects and grasps, as
   scripts/prefetch_assets.py does for the houses of the other lists. It stops with status 2 if
   an install fails.
3. --scan (GPU + EGL): the pick-and-place task sampler samples houses 0-199 (seed 100,000 +
   house before each); a house where the sampler raises HouseInvalidForTask is not valid. Writes
   <data>/phase/house_validity.json ({"houses": {house: {"valid", "reason", "pickup",
   "receptacle", "description"}}}). The paper's scan found 181 valid houses; the natural rollout
   catalog (lists/phase/natural_episodes.csv) is those houses x 3 seed rounds
   (tests/provenance/check_phase_lists.py --validity re-derives it from this file).

Run it once before the harvest stages of this figure, in a process with network access (after
scripts/prefetch_assets.py), with nothing else writing to the asset cache.

  python pipelines/phase/05_prepare_assets.py [--scan] [--skip-receptacles] [--skip-scenes]
"""
from __future__ import annotations

import os

os.environ["VLAGEOM_NOOP_FILELOCK"] = "0"          # real locks while the cache is written
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import vlageom.sim  # noqa: E402,F401  (before anything imports molmo_spaces)
from vlageom import paths  # noqa: E402
from vlageom import phase as P  # noqa: E402

log = logging.getLogger("phase.05_prepare_assets")
MAX_SIZE = np.array([0.5, 0.5, 0.15])
MIN_SIZE = np.array([0.17, 0.17, -1])


def install_receptacles(grasp_source: str = "droid_objaverse") -> dict:
    from molmo_spaces.utils.lazy_loading_utils import install_uid
    from molmo_spaces.utils.synset_utils import get_valid_receptacle_uids

    def valid(anno):
        xyz = [anno["boundingBox"][x] for x in "xyz"]
        return (anno["receptacle"] and anno["primaryProperty"] == "CanPickup"
                and MAX_SIZE[2] >= xyz[2]
                and MAX_SIZE[0] >= xyz[0] > MIN_SIZE[0]
                and MAX_SIZE[1] >= xyz[1] > MIN_SIZE[1])

    pool = {uid: anno for uid, anno in get_valid_receptacle_uids().items() if valid(anno)}
    log.info("receptacle pool: %d objects", len(pool))
    ok, fail = 0, []
    t0 = time.time()
    for i, uid in enumerate(sorted(pool)):
        try:
            install_uid(uid, grasp_source=grasp_source)
            ok += 1
        except Exception as e:
            fail.append(uid)
            log.warning("install_uid(%s) failed: %s", uid, e)
        if (i + 1) % 50 == 0:
            log.info("  %d/%d installed (%.0f s)", i + 1, len(pool), time.time() - t0)
    return {"n_pool": len(pool), "n_ok": ok, "failed": fail}


def install_scenes(houses) -> dict:
    """Install the scene (ceiling variant, the one the task sampler loads) of every house in
    `houses`, with its objects and grasps."""
    from molmo_spaces.utils.lazy_loading_utils import (
        install_scene_with_objects_and_grasps_from_path)

    from vlageom.sim.env import make_config

    cfg = make_config(P.CONFIG)
    sampler = cfg.task_sampler_config.task_sampler_class(cfg)
    ok, fail = 0, []
    t0 = time.time()
    for i, h in enumerate(houses):
        try:
            sampler._increment_task_and_reset_house(force_advance_scene=False, house_index=int(h))
            scene = sampler._current_house_scene_path(variant="ceiling")
            install_scene_with_objects_and_grasps_from_path(scene)
            ok += 1
            log.info("house %d (%d/%d): %s (%.0f s)", h, i + 1, len(houses), Path(scene).name,
                     time.time() - t0)
        except Exception as e:
            fail.append(int(h))
            log.warning("scene of house %d failed: %s", h, e)
    return {"n_houses": len(houses), "n_ok": ok, "failed": fail}


def scan_houses(houses, out_dir: str) -> dict:
    from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask

    from vlageom.sim.engine import seed_all
    from vlageom.sim.env import SimEnv

    sim_env = SimEnv(P.CONFIG, output_dir=out_dir)
    sampler = sim_env.exp_config.task_sampler_config.task_sampler_class(sim_env.exp_config)
    res = {}
    for h in houses:
        t0 = time.time()
        seed_all(100_000 + int(h))
        try:
            task = sampler.sample_task(house_index=int(h))
            tc = task.config.task_config
            res[str(h)] = {"valid": True, "reason": "",
                           "pickup": getattr(tc, "pickup_obj_name", ""),
                           "receptacle": getattr(tc, "place_receptacle_name", ""),
                           "description": task.get_task_description(),
                           "t_s": round(time.time() - t0, 1)}
            del task
        except HouseInvalidForTask as e:
            res[str(h)] = {"valid": False, "reason": f"HouseInvalidForTask: {e}",
                           "t_s": round(time.time() - t0, 1)}
        except Exception as e:
            res[str(h)] = {"valid": False, "reason": f"{type(e).__name__}: {e}",
                           "t_s": round(time.time() - t0, 1)}
        log.info("house %d: %s", h, "valid" if res[str(h)]["valid"] else res[str(h)]["reason"])
        gc.collect()
    try:
        sim_env.close()
    except Exception:
        pass
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", action="store_true", help="also scan houses 0-199 (GPU + EGL)")
    ap.add_argument("--houses", type=int, nargs="+", default=None,
                    help="houses to scan (default 0..199)")
    ap.add_argument("--skip-receptacles", action="store_true")
    ap.add_argument("--skip-scenes", action="store_true")
    ap.add_argument("--out", type=Path, default=paths.tile_dir("phase", "house_validity.json"))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    pool = {} if a.skip_receptacles else install_receptacles()
    if pool:
        log.info("receptacles: %d installed, %d failed", pool["n_ok"], len(pool["failed"]))
        if pool["failed"]:
            return 2
    if not a.skip_scenes:
        houses = sorted({e["house"] for e in P.read_natural_episodes()})
        sc = install_scenes(houses)
        log.info("scenes: %d of %d installed, failed: %s", sc["n_ok"], sc["n_houses"], sc["failed"])
        if sc["failed"]:
            return 2
    if not a.scan:
        return 0
    houses = a.houses if a.houses else list(range(200))
    res = scan_houses(houses, str(paths.tile_dir("phase", "env", create=True)))
    n_valid = sum(1 for v in res.values() if v["valid"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"config": P.CONFIG, "n_valid": n_valid, "n_scanned": len(res),
                                 "pool": pool, "houses": res}, indent=1))
    log.info("%d of %d houses valid -> %s", n_valid, len(res), a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
