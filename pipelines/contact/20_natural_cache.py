"""Contact (Figure 4), stage 20 (CPU): natural cache, one row per query of the natural harvest.

Reads every natural harvest file <harvest>/ep<E>_h<H>.npz (stage 10) and pools, per query and
layer, six token sets of the stored float16 activations (float32 mean, stored as float16):

  X_seq_all    mean over all tokens of the sequence
  X_img_all    mean over the 392 current-frame image patches (exo + wrist)
  X_img_exo    mean over the 196 exo patches
  X_img_wrist  mean over the 196 wrist patches
  X_txt_all    mean over the text-tail tokens other than the chat and multimodal control tokens
               (the text-token mean of Figure 4)
  X_last_tok   the final token

Each (N, 10, 2560) float16. Labels per row, all from the harvest's own rollout: contact (contact
at the query's step), delta (step - first contact step; nan without contact), grip (executed
gripper command / 255), phase (step / (n_steps - 1)), q, step, ep, house, success, onset,
onset_phase, n_steps_ep, replay (True). From the listed rollout of each episode
(lists/contact/natural_episodes.csv): orig_n_steps, and agree (1 / 0 when that rollout queried the
model at the same step and its contact label agrees / disagrees, -1 otherwise). Rows are ordered
by episode, then query. `layers`, `variants` and `harvest` (the input directory) are stored too.

Output: $VLAGEOM_DATA/contact/natural_cache.npz (--out), 5.8 GB for the 299 episodes (18,837 rows,
192 houses). Reads the whole harvest (369 GB); about 1.3 GB of memory per worker plus 12 GB for
the merge. BLAS is not used.

  python pipelines/contact/20_natural_cache.py [--harvest DIR] [--out FILE] [--workers 4]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from vlageom import contact as C  # noqa: E402
from vlageom import paths  # noqa: E402
from vlageom.util import save_npz  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n", 1)[1])
    ap.add_argument("--harvest", type=Path, default=None,
                    help="natural harvest directory (default $VLAGEOM_DATA/contact/natural)")
    ap.add_argument("--episodes", type=Path, default=None,
                    help="episode list (default lists/contact/natural_episodes.csv)")
    ap.add_argument("--out", type=Path, default=None,
                    help="cache file (default $VLAGEOM_DATA/contact/natural_cache.npz)")
    ap.add_argument("--workers", type=int, default=4, help="worker processes (default 4)")
    a = ap.parse_args(argv)

    harvest = a.harvest or C.natural_dir()
    out = a.out or paths.tile_dir(C.TILE) / "natural_cache.npz"
    meta = {r["episode_index"]: r for r in C.read_natural_episodes(a.episodes)}
    shards = sorted(Path(harvest).glob("ep*_h*.npz"))
    if not shards:
        raise SystemExit(f"no natural harvest files under {harvest}")
    eps = [int(s.name.split("_")[0][2:]) for s in shards]
    unknown = sorted(set(eps) - set(meta))
    if unknown:
        raise SystemExit(f"harvest files of episodes not in the list: {unknown[:10]}")
    missing = sorted(set(meta) - set(eps))
    print(f"[20] {len(shards)} harvest files in {harvest}"
          + (f"; {len(missing)} listed episodes have none: {missing[:10]}" if missing else ""),
          flush=True)

    t0 = time.time()
    parts, n_failed = [], 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(C.pool_natural_shard, str(s), meta[e]["orig_n_steps"],
                          meta[e]["orig_contact"]): s for s, e in zip(shards, eps)}
        for i, f in enumerate(as_completed(futs)):
            try:
                r = f.result()
            except Exception as e:
                print(f"[20] {futs[f].name} FAILED: {e}", flush=True)
                n_failed += 1
                continue
            if r is None:
                print(f"[20] {futs[f].name}: no usable rows, skipped", flush=True)
                continue
            parts.append(r)
            if (i + 1) % 25 == 0:
                print(f"[20]  {i + 1}/{len(shards)} files ({time.time() - t0:.0f}s)", flush=True)
    if not parts:
        raise SystemExit("[20] no usable episodes")
    if n_failed > max(2, len(shards) // 20):
        raise SystemExit(f"[20] {n_failed}/{len(shards)} files failed; no cache written")
    m = C.merge_natural_parts(parts)
    if m["dropped"]:
        print(f"[20] layer grid differs from the first episode's; dropped episodes {m['dropped']}",
              flush=True)
    merged = m["merged"]
    save_npz(out, dict(layers=m["layers"], variants=m["variants"], harvest=str(harvest), **merged))
    N = len(merged["contact"])
    ag = merged["agree"]
    ag_frac = float((ag[ag >= 0] == 1).mean()) if (ag >= 0).any() else float("nan")
    print(f"[DONE] {N} rows, {m['n_eps']} episodes, {len(np.unique(merged['house']))} houses, "
          f"layers {[int(x) for x in m['layers']]}, contact on {merged['contact'].mean():.3f}, "
          f"with a contact onset {np.isfinite(merged['delta']).mean():.3f}, "
          f"rows outside the step series {m['step_mismatch']}, agreement with the listed rollouts "
          f"{ag_frac:.4f}, {time.time() - t0:.0f}s -> {out}", flush=True)
    print(f"[20] the rollout's instruction differs from the listed task description in "
          f"{m['n_instruction_differs']} of {m['n_eps']} episodes", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
