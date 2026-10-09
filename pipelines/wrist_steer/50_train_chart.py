"""Wrist steering, stage 50 (GPU): train the circular autoencoder chart.

Training data: the token-mean layer-9 activations of the orientation sweep files of stage 30 (sorted
path order; 25,748 rows of 102 houses in the paper's store), standardized with the training rows' mean
and RMS norm, and every ordered pair of rows of one sweep whose rotation differs by 4..340 deg. Rows
of the held-out houses (numpy default_rng(0) shuffle, first third) are not trained on.

Model: encoder 768-512-256-32 (Linear, LayerNorm, GELU) whose code pairs (0,1), (2,3), (4,5) are
projected onto circles of learned radius, one per rotation axis (base x, y, z); decoder 32-256-512-768.
Loss per step (8,000 steps, batches of 512 rows and 512 pairs, Adam lr 1e-3): reconstruction +
code-rotation (the code of row i rotated by the pair's angle on its axis's circle versus the code of
row j) + 2 x decoded rotation (decoded rotated code versus row j). Seeds: torch and numpy 0.

Outputs: <data>/wrist_steer/chart.pt (enc_trunk, log_r, dec state dicts; mean, scale, z_dim,
version "circular2", args) and chart_report.json (held-out reconstruction R^2, held-out
decode-transport R^2 per axis, radii).

The paper's chart was trained on one H200 (CUDA) with torch 2.7.1; a run on another device gives
a different chart.

  python pipelines/wrist_steer/50_train_chart.py
"""
# The chart is Mathilde Papillon's; adapted from her code.
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from vlageom import wrist_steer as ws

log = logging.getLogger("wrist_steer.50_train_chart")


def main(argv=None) -> int:
    a0 = ws.CHART_ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orient-dir", type=Path, default=None,
                    help="orientation sweep files (default <data>/wrist_steer/orient)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output directory for chart.pt and chart_report.json (default <data>/wrist_steer)")
    ap.add_argument("--device", default=None, help="torch device (default cuda when available)")
    ap.add_argument("--z-dim", type=int, default=a0["z_dim"])
    ap.add_argument("--steps", type=int, default=a0["steps"])
    ap.add_argument("--batch", type=int, default=a0["batch"])
    ap.add_argument("--lr", type=float, default=a0["lr"])
    ap.add_argument("--w-recon", type=float, default=a0["w_recon"])
    ap.add_argument("--w-latent", type=float, default=a0["w_latent"])
    ap.add_argument("--w-decode", type=float, default=a0["w_decode"])
    ap.add_argument("--max-pair-deg", type=float, default=a0["max_pair_deg"])
    ap.add_argument("--seed", type=int, default=a0["seed"])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import torch

    files = ws.orient_files(args.orient_dir)
    if not files:
        raise SystemExit(f"no orientation sweep files under {args.orient_dir or ws.tile_path('orient')}")
    X, is_te, pairs, pair_te = ws.load_chart_pairs(files, ws.LAYER, seed=args.seed,
                                                   max_pair_deg=args.max_pair_deg)
    ck, report = ws.train_chart(X, is_te, pairs, pair_te, z_dim=args.z_dim, steps=args.steps,
                                batch=args.batch, lr=args.lr, w_recon=args.w_recon,
                                w_latent=args.w_latent, w_decode=args.w_decode,
                                max_pair_deg=args.max_pair_deg, seed=args.seed, device=args.device)
    out = args.out_dir or ws.tile_path("chart.pt", create=True).parent
    out.mkdir(parents=True, exist_ok=True)
    torch.save(ck, out / "chart.pt")
    (out / "chart_report.json").write_text(json.dumps(report, indent=1, default=float))
    print(f"chart -> {out / 'chart.pt'}, {out / 'chart_report.json'}")
    print(f"  held-out reconstruction R2 {report['recon_r2_test']:.3f}; decode-transport R2 "
          f"{report['decode_transport_r2_test']}; radii {[round(r, 3) for r in report['radii']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
