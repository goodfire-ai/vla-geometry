"""Contact (Figure 4), stage 40 (CPU): layer-24 text-token pools of the off-schedule placements.

Reads the off-schedule harvest files <harvest>/ep<E>_h<H>.npz (stage 30) and keeps, per placement,
two pools of the stored float16 tokens at layer 24 (--layer):

  X_txt_all   float32 mean over the text-tail tokens other than the chat and multimodal control
              tokens (the text-token mean of Figure 4)
  X_last_tok  the final text token, as float32

Only the text-tail rows of the one layer are read from each file (row by row, from the offsets of
the uncompressed `act` member), about 1.4 GB in all instead of the whole 352 GB.

Per-row fields: ep, house, pose_id, gpose (= ep * 1000 + pose_id, unique per pose), event, pdir
(the pose's branch: +1 object moved away from the hand, -1 toward it), qc, step_k, contact, disp,
dist, phase, pose_chunk, grasp0, pose_grasp_forced, pose_sep0 (pose fields broadcast to the
pose's rows), n_txt and shard_idx (index into `shards`, the file names in sorted order). Also
`layer` and `subdir` (the harvest directory's name).

Output: $VLAGEOM_DATA/contact/offschedule_L24.npz (--out); 17,975 rows from the 123 files, 369 MB.
About 1-2 minutes.

  python pipelines/contact/40_offschedule_pool.py [--harvest DIR] [--out FILE]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
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
                    help="off-schedule harvest directory (default $VLAGEOM_DATA/contact/offschedule)")
    ap.add_argument("--layer", type=int, default=C.PROBE_LAYER, help="VLM layer (default 24)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output file (default $VLAGEOM_DATA/contact/offschedule_L<layer>.npz)")
    a = ap.parse_args(argv)

    harvest = Path(a.harvest or C.offschedule_dir())
    out = a.out or paths.tile_dir(C.TILE) / f"offschedule_L{a.layer}.npz"
    shards = sorted(harvest.glob("ep*_h*.npz"), key=lambda p: str(p))
    if not shards:
        raise SystemExit(f"no off-schedule harvest files under {harvest}")
    print(f"[40] {len(shards)} files in {harvest}, layer {a.layer}", flush=True)
    t0 = time.time()
    X = {p: [] for p in C.POOLS}
    cols = {k: [] for k in C.POOL_FIELDS}
    for si, sp in enumerate(shards):
        pools, c = C.pool_offschedule_shard(sp, a.layer)
        for p in C.POOLS:
            X[p].append(pools[p])
        for k, v in c.items():
            cols[k].append(v)
        cols["shard_idx"].append(np.full(len(c["contact"]), si, np.int16))
        if (si + 1) % 20 == 0:
            print(f"[40]  {si + 1}/{len(shards)} files ({time.time() - t0:.0f}s)", flush=True)
    arrays = {f"X_{p}": np.concatenate(X[p]) for p in C.POOLS}
    arrays.update({k: np.concatenate(v) for k, v in cols.items()})
    save_npz(out, dict(layer=a.layer, subdir=harvest.name,
                       shards=np.array([s.name for s in shards]), **arrays))
    print(f"[DONE] {len(arrays['contact'])} rows, {len(shards)} episodes, {time.time() - t0:.0f}s "
          f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
