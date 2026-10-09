"""Compare a chart trained by pipelines/wrist_steer/50_train_chart.py with a reference chart.

Compared: every tensor of the encoder trunk and decoder state dicts and log_r (dtype, shape, values),
the input mean (array) and scale, z_dim and the training arguments; with --new-report and
--ref-report also the report values (held-out reconstruction R^2, decode-transport R^2, radii).

  python pipelines/wrist_steer/50_train_chart.py --orient-dir <reference orient store> --out-dir $C/chart
  python tests/canary_wrist_steer_chart.py --new $C/chart/chart.pt --reference <reference chart .pt> \
      [--new-report $C/chart/chart_report.json --ref-report <reference report .json>]

A bit-identical chart is expected only on the GPU type, CUDA and torch versions of the reference.
Exit status 0 when everything compared is identical.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ARG_KEYS = ("z_dim", "steps", "batch", "lr", "w_recon", "w_latent", "w_decode", "max_pair_deg", "seed")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--new-report", type=Path, default=None)
    ap.add_argument("--ref-report", type=Path, default=None)
    args = ap.parse_args(argv)
    import torch
    a = torch.load(args.new, map_location="cpu", weights_only=False)
    b = torch.load(args.reference, map_location="cpu", weights_only=False)
    ok = True
    for part in ("enc_trunk", "dec"):
        for k in b[part]:
            x, y = a[part].get(k), b[part][k]
            eq = x is not None and x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)
            msg = f"  {'equal' if eq else 'DIFF':6s} {part}.{k} {tuple(y.shape)}"
            if not eq and x is not None and x.shape == y.shape:
                msg += f" max|diff| {(x.double() - y.double()).abs().max().item():.3e}"
            print(msg)
            ok = ok and eq
    eq = torch.equal(a["log_r"], b["log_r"])
    print(f"  {'equal' if eq else 'DIFF':6s} log_r {a['log_r'].tolist()} vs {b['log_r'].tolist()}")
    ok = ok and eq
    eq = np.array_equal(np.asarray(a["mean"]), np.asarray(b["mean"])) and a["scale"] == b["scale"]
    print(f"  {'equal' if eq else 'DIFF':6s} mean, scale ({a['scale']!r} vs {b['scale']!r})")
    ok = ok and eq
    ea = {k: a["args"].get(k) for k in ARG_KEYS}
    eb = {k: b["args"].get(k) for k in ARG_KEYS}
    eq = ea == eb and a["z_dim"] == b["z_dim"]
    print(f"  {'equal' if eq else 'DIFF':6s} z_dim, training arguments {ea}")
    ok = ok and eq
    if args.new_report and args.ref_report:
        ra = json.loads(args.new_report.read_text())
        rb = json.loads(args.ref_report.read_text())
        for k in ("recon_r2_test", "decode_transport_r2_test", "radii"):
            eq = ra.get(k) == rb.get(k)
            print(f"  {'equal' if eq else 'DIFF':6s} report {k}: {ra.get(k)} vs {rb.get(k)}")
            ok = ok and eq
    print("IDENTICAL" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
