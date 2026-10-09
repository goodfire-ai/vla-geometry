"""Stage 20 (CPU): pack the near-range harvest into a store that can be memory-mapped.

Reads <near store>/houses/h*.npz (stage 10) in file-name order and writes to --out
(default <near store>/prep, near store = $VLAGEOM_DATA/distance/near):
  X_last_tok.npy  (N, 13, 2560) float16   VLM last-token activations at the 13 stored layers
  X_ae.npy        (N, 11, 768)  float16   action-expert activations (mean over action tokens)
  labels.npz      per-row labels: obj_base, ee, rel_ee, u, dist, config_id, chunk_q, step, fwd,
                  bearing_deg, in_env, grip_width, contact_other, ep_done, success_q, house
  meta.json       {n_rows, n_houses, layers_llm, layers_ae}

Takes about 3 minutes for the 300 houses (14 GB of house files) on one core.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlageom.distance import near_root, pack_near_store


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--houses", type=Path, default=near_root() / "houses",
                    help="per-house files of stage 10 "
                         "(default: $VLAGEOM_DATA/distance/near/houses)")
    ap.add_argument("--out", type=Path, default=near_root() / "prep",
                    help="packed store (default: $VLAGEOM_DATA/distance/near/prep)")
    a = ap.parse_args()
    if (a.out / "meta.json").exists():
        raise SystemExit(f"{a.out} already holds a packed store; remove it to rebuild")
    meta = pack_near_store(a.houses, a.out)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
