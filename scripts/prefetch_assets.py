#!/usr/bin/env python
"""One-time download of everything the harvest stages read. Run it once, in one process and with
network access, before any harvest job:

  1. MolmoSpaces assets: the resource manager's start-up installs (robots, benchmark episodes,
     THOR objects, DROID grasps, small scene sets), then the scene, its objects and its grasps for
     every house in the shipped lists: the rollout houses of lists/gripper_episodes.csv,
     lists/wrist_episodes.csv, lists/contact/*.csv and lists/wrist_steer/*.csv (houses 0-199,
     including the houses the wrist-steering natural rollouts sample before their listed house),
     and the benchmark houses of lists/teleport_houses.csv, lists/near_houses.csv and
     lists/position/crossed_houses.csv; then the object-metadata database. The pick-and-place
     receptacles and scenes of the phase figure are installed by
     pipelines/phase/05_prepare_assets.py, run after this script;
  2. the NLTK WordNet corpora that MolmoSpaces reads when it is imported;
  3. the MolmoBot-DROID checkpoint (pinned revision) into $VLAGEOM_CHECKPOINT (default
     ~/checkpoints/MolmoBot-DROID);
  4. the Qwen3 tokenizer the policy loads (Qwen/Qwen3-4B-Instruct-2507) at its pinned revision into
     the Hugging Face cache. The policy loads the tokenizer by name, and an offline load by name
     reads the revision in the cache's refs/main, so refs/main is set to the pinned revision.

The MolmoSpaces assets go into two directories: the resource cache (MLSPACES_CACHE_DIR, default
~/.cache/molmo-spaces-resources) and the assets directory (MLSPACES_ASSETS_DIR, default
<site-packages>/assets of this environment), which holds links into the cache and the object
database. Both must be writable here, and the harvest jobs must use the same two directories; the
script prints them.

The script keeps the real file locks of the asset cache (it sets VLAGEOM_NOOP_FILELOCK=0 before
anything is imported), so it is safe to run while nothing else uses the cache. The harvest jobs
(scripts/harvest_env.sh) turn the locks off and only read the cache afterwards.

Steps whose results are already present are skipped by the libraries (asset archives are
extracted once, files already in the Hugging Face cache are not downloaded again), so the script
can be run again to complete an interrupted download.

  python scripts/prefetch_assets.py                    # everything
  python scripts/prefetch_assets.py --skip-checkpoint  # assets and tokenizer only
"""
from __future__ import annotations

import os

os.environ["VLAGEOM_NOOP_FILELOCK"] = "0"      # real locks while the caches are written
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

import argparse  # noqa: E402
import csv  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from vlageom import paths  # noqa: E402

log = logging.getLogger("prefetch")

ROLLOUT_CONFIG = "FrankaPickDroidDataGenConfig"     # pick rollouts (Figures 1, 4, 6, 7)
TOKENIZER = "Qwen/Qwen3-4B-Instruct-2507"
TOKENIZER_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"   # the revision the paper used
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                   "special_tokens_map.json", "added_tokens.json", "chat_template.jinja"]
NLTK_CORPORA = ("wordnet", "wordnet2022")


def _houses(name: str) -> list[int]:
    with open(paths.lists_dir() / name, newline="") as f:
        return sorted({int(r["house"]) for r in csv.DictReader(f)})


def prefetch_assets(limit: int | None, dry_run: bool = False) -> None:
    from molmo_spaces.molmo_spaces_constants import get_resource_manager
    from molmo_spaces.utils.lazy_loading_utils import (
        install_scene_with_objects_and_grasps_from_path, install_uid)
    from vlageom.sim.bench import BENCH_CONFIGS, build_house_list
    from vlageom.sim.env import make_config

    from molmo_spaces.molmo_spaces_constants import ASSETS_DIR, DATA_CACHE_DIR
    log.info("MolmoSpaces resource cache: %s (MLSPACES_CACHE_DIR)", DATA_CACHE_DIR)
    log.info("MolmoSpaces assets directory: %s (MLSPACES_ASSETS_DIR)", ASSETS_DIR)
    t0 = time.time()
    if dry_run:
        def install_scene_with_objects_and_grasps_from_path(scene):   # noqa: F811
            log.info("would install %s with its objects and grasps", scene)

        def install_uid(uid):                                           # noqa: F811
            log.info("would install object %s", uid)
    else:
        get_resource_manager()
        log.info("resource manager ready (%.0f s)", time.time() - t0)

    # Rollout houses: the scene the task sampler of the rollout config loads for each house.
    from vlageom.wrist_steer import rollout_houses

    rollout = sorted(set(_houses("gripper_episodes.csv")) | set(_houses("wrist_episodes.csv"))
                     | set(_houses("contact/natural_episodes.csv"))
                     | set(_houses("contact/offschedule_episodes.csv")) | set(rollout_houses()))
    cfg = make_config(ROLLOUT_CONFIG)
    sampler = cfg.task_sampler_config.task_sampler_class(cfg)
    for i, h in enumerate(rollout[:limit]):
        sampler._increment_task_and_reset_house(force_advance_scene=False, house_index=h)
        scene = sampler._current_house_scene_path(variant="ceiling")
        install_scene_with_objects_and_grasps_from_path(scene)
        log.info("rollout house %d (%d/%d): %s", h, i + 1, len(rollout), Path(scene).name)

    # Benchmark houses: the scene of each house's benchmark episode, plus its added objects.
    from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler

    want = (set(_houses("teleport_houses.csv")) | set(_houses("near_houses.csv"))
            | set(_houses("position/crossed_houses.csv")))
    specs = [(h, b, sp) for h, b, _, sp in build_house_list() if h in want]
    missing = want - {h for h, _, _ in specs}
    if missing:
        raise SystemExit(f"houses not in the benchmark episodes: {sorted(missing)[:10]} ...")
    for i, (h, bench, spec) in enumerate(specs[:limit]):
        s = JsonEvalTaskSampler(make_config(BENCH_CONFIGS[bench]), episode_spec=spec)
        scene = s._current_house_scene_path(variant="ceiling")
        install_scene_with_objects_and_grasps_from_path(scene)
        for rel in spec.scene_modifications.added_objects.values():
            if not (ASSETS_DIR / rel).is_file():
                install_uid(Path(rel).stem)
        log.info("benchmark house %d (%d/%d): %s", h, i + 1, len(specs), Path(scene).name)

    if not dry_run:
        from molmo_spaces.utils.object_metadata import get_db
        get_db()
    log.info("assets done (%.0f s)", time.time() - t0)


def prefetch_nltk() -> None:
    import nltk
    for c in NLTK_CORPORA:
        nltk.download(c, quiet=True)
    log.info("NLTK corpora: %s", ", ".join(NLTK_CORPORA))


def prefetch_checkpoint() -> None:
    from huggingface_hub import snapshot_download
    ck = paths.checkpoint()
    local = Path(ck).expanduser()
    if "/" in ck and not ck.startswith(("/", "~", ".")) and not local.exists():
        snapshot_download(ck, revision=paths.CHECKPOINT_REVISION if ck == paths.CHECKPOINT_REPO
                          else None)
        log.info("checkpoint %s in the Hugging Face cache", ck)
        return
    if all((local / f).is_file() for f in ("model.pt", "config.yaml")):
        log.info("checkpoint present: %s", local)
        return
    local.mkdir(parents=True, exist_ok=True)
    snapshot_download(paths.CHECKPOINT_REPO, revision=paths.CHECKPOINT_REVISION,
                      local_dir=str(local))
    log.info("checkpoint %s@%s -> %s", paths.CHECKPOINT_REPO, paths.CHECKPOINT_REVISION, local)


def prefetch_tokenizer() -> None:
    from huggingface_hub import snapshot_download, try_to_load_from_cache
    from transformers import AutoTokenizer
    snap = Path(snapshot_download(TOKENIZER, revision=TOKENIZER_REVISION,
                                  allow_patterns=TOKENIZER_FILES))
    # A load at the revision also records which optional tokenizer files the revision lacks, so
    # that an offline load does not look for them.
    AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION)
    ref = snap.parent.parent / "refs" / "main"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(TOKENIZER_REVISION)
    hit = try_to_load_from_cache(TOKENIZER, "tokenizer.json")
    if not isinstance(hit, str) or f"/snapshots/{TOKENIZER_REVISION}/" not in hit:
        raise SystemExit(f"tokenizer {TOKENIZER}: a load by name resolves to {hit}, not to "
                         f"revision {TOKENIZER_REVISION}")
    log.info("tokenizer %s cached at revision %s (refs/main)", TOKENIZER, TOKENIZER_REVISION)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--skip-assets", action="store_true",
                    help="skip the MolmoSpaces assets and the NLTK corpora")
    ap.add_argument("--skip-checkpoint", action="store_true", help="skip the policy checkpoint")
    ap.add_argument("--skip-tokenizer", action="store_true", help="skip the Qwen3 tokenizer")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N houses of each house set (for a quick check)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the scenes and objects of the asset step without installing them")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if not args.skip_assets:
        prefetch_assets(args.limit, args.dry_run)
        if not args.dry_run:
            prefetch_nltk()
    if args.dry_run:
        return 0
    if not args.skip_checkpoint:
        prefetch_checkpoint()
    if not args.skip_tokenizer:
        prefetch_tokenizer()
    log.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
