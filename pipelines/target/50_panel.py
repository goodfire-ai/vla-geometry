#!/usr/bin/env python
"""Target tile panel arrays for Figure 1 (CPU): $VLAGEOM_DATA/panels/target.npz.

Inputs: the packed object-teleport store (20_pack_store.py), the object path of house 44
(30_square_path_harvest.py) and the probe score (40_probe.py).

  1. Read-out basis: StandardScaler + RidgeCV (efficient leave-one-out over penalties
     0.1..1e4) from the VLM last-token activation at layer 21 to the object position (x, y, z),
     on the first-query rows of every house except house 44. The 3 weight vectors are
     orthonormalized (QR) into a 3-D basis.
  2. Cloud: all first-query rows projected on the basis, centred, divided by the 95th-percentile
     radius; coloured by the per-axis rank of the object position (x -> red, y -> green,
     z -> blue); thinned to about one point per 0.06 voxel and capped at 4,000 points.
  3. Path: the 64 waypoint activations of house 44 projected the same way (consecutive waypoints
     within 1 mm merged), and a periodic cubic smoothing spline through them (s = 0.02, 600 samples).
  4. Everything is rotated so the axes line up with the decoded forward, left and up directions.

Output keys: bg, rgb (cloud), sub, sub_rgb (waypoints), curve, curve_rgb (spline), origin
(projection of the zero vector), src (provenance string), note_r2 (printed R^2, mean over x, y, z)
and note_r2_xyz. Runs in about 1-5 min. With the default 24 BLAS threads on an x86-64 CPU with
AVX-512 the arrays equal the paper's bit for bit; other thread counts change them by about 1e-11.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vlageom import paths
from vlageom.target import (BLAS_THREADS, HELDOUT_HOUSE, TeleportStore, probe_json_default,
                            square_path_root, target_panel)
from vlageom.util import blas_threads


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", type=Path, default=None,
                    help="store root holding prep/ (default $VLAGEOM_DATA/target/teleport)")
    ap.add_argument("--path-dir", type=Path, default=None,
                    help=f"object-path directory with acts.npz + manifest.json "
                         f"(default $VLAGEOM_DATA/target/shape_square/h{HELDOUT_HOUSE})")
    ap.add_argument("--probe-json", type=Path, default=None,
                    help="40_probe.py output (default $VLAGEOM_DATA/target/probe/last_tok_L21.json)")
    ap.add_argument("--out", type=Path, default=None,
                    help="panel file (default $VLAGEOM_DATA/panels/target.npz)")
    ap.add_argument("--blas-threads", type=int, default=BLAS_THREADS,
                    help="BLAS threads (default %(default)s, the paper's setting; 0 = library default)")
    args = ap.parse_args()
    _blas = blas_threads(args.blas_threads)  # noqa: F841  (kept for the whole run)

    store = TeleportStore(args.store)
    path_dir = args.path_dir or square_path_root() / f"h{HELDOUT_HOUSE}"
    pj = args.probe_json or probe_json_default()
    note = json.loads(Path(pj).read_text())
    if note["store"]["labels_md5"] != store.provenance()["labels_md5"]:
        raise SystemExit(f"{pj} was computed on another store ({note['store']['store']})")
    panel = target_panel(store, path_dir, note=note)
    out = args.out or paths.panels_dir(create=True) / "target.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **panel)
    print(f"wrote {out}: " + ", ".join(f"{k} {v.shape}" for k, v in panel.items())
          + f"; note R^2 = {float(panel['note_r2']):.4f} (prints {float(panel['note_r2']):.2f})",
          flush=True)


if __name__ == "__main__":
    main()
