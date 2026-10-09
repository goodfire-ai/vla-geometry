"""Wrist steering, stage 70 (CPU): panel arrays of Figure 6.

Panel a (--panel a): the wrist roll sweep of house 0 (<data>/wrist/sweeps/ep0.npz, written by
pipelines/wrist/01_sweep.py --row 0): rows sorted by joint angle, action token 0 of layer 9 minus
the sweep's mean, numpy SVD; the 39 rows on PC1-2.
  -> <data>/panels/fig6_a.npz: Z2 (39, 2), roll (39,) joint angle (deg), evr (2,), src
  -> <data>/panels/fig6_a_metrics.json: shape measures of the sweep (closure, coverage of 345.6 deg,
     circle fit, phase advance, largest H1 persistence with ripser when installed), PC1-2 variance

Panels b-d (--panel bcd; needs torch, CPU is enough):
  b  the natural cloud (stage 20) encoded with the chart (stage 50); its point on the circle for
     rotation about base z, the phase shifted so that it lines up with the hand's yaw, and the
     circular mean of the phase in 36 yaw bins
  c  one-shot rows (stage 60, --mode oneshot) with chart edits about z of -40, -20, +20, +40 deg,
     scenes whose unedited chunk turns the wrist by more than 10 deg left out (on the paper's data
     scene 6, 22.9 deg; the others turn it by at most 2.6): rotation of the decoded chunk along the
     written direction, sign(theta) x cos x per-step rotation x 8 steps
  d  held-edit rows (stage 60, --mode persist) of house 10: the hand's rotation about base z after
     each of 32 steps, for -40, -20, +20, +40 deg and the unedited rollout
  -> <data>/panels/fig6_bcd.npz: ph, yaw (N,), r, bin_ph, bin_ctr (36,), c_theta, c_resp (28,),
     d_theta (5,), d_rot (5, 32)

  python pipelines/wrist_steer/70_panels.py [--panel a|bcd|all]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from vlageom import paths
from vlageom import wrist as tile_wrist
from vlageom import wrist_steer as ws

log = logging.getLogger("wrist_steer.70_panels")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", choices=["a", "bcd", "all"], default="all")
    ap.add_argument("--sweep", type=Path, default=None,
                    help="panel a: roll sweep file (default <data>/wrist/sweeps/ep0.npz)")
    ap.add_argument("--chart", type=Path, default=None, help="default <data>/wrist_steer/chart.pt")
    ap.add_argument("--natural", type=Path, default=None,
                    help="default <data>/wrist_steer/natural/natural.npz")
    ap.add_argument("--oneshot", type=Path, default=None,
                    help="default <data>/wrist_steer/steer/oneshot.json")
    ap.add_argument("--persist", type=Path, default=None,
                    help="default <data>/wrist_steer/steer/persist.json")
    ap.add_argument("--out-dir", type=Path, default=None, help="default <data>/panels")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = args.out_dir or paths.panels_dir(create=True)
    out.mkdir(parents=True, exist_ok=True)

    if args.panel in ("a", "all"):
        sweep = args.sweep or tile_wrist.sweep_path(0)
        panel, metrics = ws.panel_a(sweep, ws.LAYER)
        np.savez(out / "fig6_a.npz", **panel)
        (out / "fig6_a_metrics.json").write_text(json.dumps(metrics, indent=1))
        evr = panel["evr"]
        print(f"panel a -> {out / 'fig6_a.npz'}, {out / 'fig6_a_metrics.json'}")
        print(f"  {len(panel['roll'])} angles {panel['roll'].min():.1f}..{panel['roll'].max():.1f} deg "
              f"(coverage {metrics['coverage_deg']} deg, ends {360 - metrics['coverage_deg']:.1f} deg "
              f"apart); PC1 {100 * evr[0]:.0f}% var, PC2 {100 * evr[1]:.0f}% var")

    if args.panel in ("bcd", "all"):
        chart = ws.Chart(args.chart or ws.tile_path("chart.pt"), "cpu")
        with np.load(args.natural or ws.tile_path("natural", "natural.npz"), allow_pickle=False) as z:
            natural = {k: z[k] for k in z.files}
        oneshot = ws.read_json(args.oneshot or ws.tile_path("steer", "oneshot.json"))["rows"]
        persist = ws.read_json(args.persist or ws.tile_path("steer", "persist.json"))["rows"]
        panel = ws.panel_bcd(chart, natural, persist, oneshot)
        np.savez(out / "fig6_bcd.npz", **panel)
        means = {float(t): float(np.mean(panel["c_resp"][panel["c_theta"] == t]))
                 for t in sorted(set(panel["c_theta"].tolist()))}
        print(f"panels b-d -> {out / 'fig6_bcd.npz'}")
        print(f"  b: {len(panel['ph'])} natural queries, circle radius {float(panel['r']):.4f}")
        rot = ws.noop_chunk_rotation(oneshot)
        print(f"  c: {len(panel['c_resp'])} edits; scene(s) left out (unedited chunk rotation > "
              f"{ws.DRIFT_MAX_DEG:g} deg): " + (", ".join(
                  f"{e} ({rot[e]:.1f} deg)" for e in ws.drift_episodes(oneshot)) or "none"))
        print("     mean response per written angle (deg): "
              + ", ".join(f"{t:+.0f}: {m:+.2f}" for t, m in means.items()))
        print("  d: rotation at step 32 (deg): " + ", ".join(
            f"{t:+.0f}: {r[-1]:+.2f}" for t, r in zip(panel["d_theta"], panel["d_rot"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
