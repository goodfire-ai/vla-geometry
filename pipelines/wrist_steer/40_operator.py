"""Wrist steering, stage 40 (CPU): held-out house split and linear rotation operator.

Reads the orientation sweep files of stage 30 (sorted path order, action-expert layer 9). The houses
are shuffled with numpy default_rng(0) and the first third is held out (34 of 102); the split is
compared with lists/wrist_steer/heldout_houses.csv. On the rows of the other houses: mean, 64-d PCA,
and per rotation axis a ridge-regressed 10-degree map between rows of one sweep whose rotation
differs by 10 +- 3 deg, whose matrix logarithm / 10 deg is the axis's generator
(`vlageom.wrist_steer.fit_operator`). The steering stage reads the held-out houses from this file
and uses the generators' mean norm to scale its random-generator control edit.

Output: <data>/wrist_steer/operator_L9.npz (V (768, 64), c (768,), Lhat (3, 64, 64), test_houses (34,)).

  python pipelines/wrist_steer/40_operator.py

The linear algebra runs with 24 BLAS threads (--blas-threads), the number the paper's operator was
computed with: the last digits of V and Lhat depend on it (LAPACK splits its work by thread count);
the split does not. With 24 threads on an x86-64 CPU with AVX-512 the file is the paper's, byte for
byte.
"""
# Adapted from Mathilde Papillon's code.
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from vlageom import wrist_steer as ws

log = logging.getLogger("wrist_steer.40_operator")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orient-dir", type=Path, default=None,
                    help="orientation sweep files (default <data>/wrist_steer/orient)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output npz (default <data>/wrist_steer/operator_L9.npz)")
    ap.add_argument("--blas-threads", type=int, default=ws.OPERATOR_BLAS_THREADS,
                    help="BLAS threads of the fit (default %(default)s)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    files = ws.orient_files(args.orient_dir)
    if not files:
        raise SystemExit(f"no orientation sweep files under {args.orient_dir or ws.tile_path('orient')}")
    op = ws.fit_operator(files, ws.LAYER, blas_threads=args.blas_threads)
    out = args.out or ws.tile_path("operator_L9.npz", create=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{k: op[k] for k in ("V", "c", "Lhat", "test_houses")})
    held = [int(h) for h in op["test_houses"]]
    listed = ws.read_heldout_houses()
    print(f"operator -> {out}")
    print(f"  {len(files)} sweep files; {len(held)} held-out houses "
          f"({'equal to' if held == listed else 'DIFFERENT from'} lists/wrist_steer/heldout_houses.csv)")
    print(f"  |Lhat| per axis x, y, z: {[round(float(np.linalg.norm(L)), 2) for L in op['Lhat']]}")
    return 0 if held == listed else 1


if __name__ == "__main__":
    sys.exit(main())
