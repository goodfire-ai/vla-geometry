"""Wrist steering, stage 80 (CPU): the numbers of Figure 6 and its text.

  compensation   -response / theta over the panel-c edits (28 on the paper's data: 7 scenes x 4
                 angles; scenes whose unedited chunk turns the wrist by more than 10 deg are left
                 out): mean +- s.d. (ddof 0), and how many oppose the written angle
                 ("0.33 +- 0.08", 28 of 28)
  held edit      per written angle the mean over the 6 scenes of -net / theta, net = rotation after
                 32 steps minus the unedited rollout's ("1.0-1.3x"), and how many have the opposing
                 sign (24 of 24)
  edit size      edit norm / norm of the unedited token-mean activation over all one-shot chart
                 edits about z (32 on the paper's data); the activation norms are rounded to one
                 decimal ("19% +- 7%")
  calibration    median |slope| of the chart's z phase against the hand's rotation about base z over
                 the held-out z sweeps spanning more than 120 deg (171 sweeps; "one degree on the
                 circle corresponds to one degree")
  held-out split houses and held-out houses of the orientation sweeps (34 of 102)
  chart          held-out reconstruction R^2, decode-transport R^2, circle radii

The activation norms are read, rounded to one decimal, from the "landmarks" of the one-shot json,
or with --oneshot-log from the landmark lines of a one-shot run's log.

Output: <data>/wrist_steer/fig6_numbers.json (and the printed summary).

  python pipelines/wrist_steer/80_numbers.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from vlageom import wrist_steer as ws

log = logging.getLogger("wrist_steer.80_numbers")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oneshot", type=Path, default=None, help="default <data>/wrist_steer/steer/oneshot.json")
    ap.add_argument("--persist", type=Path, default=None, help="default <data>/wrist_steer/steer/persist.json")
    ap.add_argument("--oneshot-log", type=Path, default=None,
                    help="read the activation norms (one decimal) from the landmark lines of this "
                         "one-shot log")
    ap.add_argument("--chart", type=Path, default=None, help="default <data>/wrist_steer/chart.pt")
    ap.add_argument("--chart-report", type=Path, default=None,
                    help="default <data>/wrist_steer/chart_report.json")
    ap.add_argument("--orient-dir", type=Path, default=None, help="default <data>/wrist_steer/orient")
    ap.add_argument("--out", type=Path, default=None, help="default <data>/wrist_steer/fig6_numbers.json")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    one = ws.read_json(args.oneshot or ws.tile_path("steer", "oneshot.json"))
    per = ws.read_json(args.persist or ws.tile_path("steer", "persist.json"))
    if args.oneshot_log is not None:
        norms = ws.base_norms_from_log(args.oneshot_log.read_text())
    else:
        norms = {int(lm["ep"]): round(float(lm["base_act_norm"]), 1) for lm in one.get("landmarks", [])}
    chart = ws.Chart(args.chart or ws.tile_path("chart.pt"), "cpu")
    report = ws.read_json(args.chart_report or ws.tile_path("chart_report.json"))
    files = ws.orient_files(args.orient_dir)
    nums = ws.fig6_numbers(one["rows"], per["rows"], norms, chart, files, report)
    out = args.out or ws.tile_path("fig6_numbers.json", create=True)
    out.write_text(json.dumps(nums, indent=1, default=float))

    o, p, e, c = nums["oneshot"], nums["persist"], nums["edit_size"], nums["calibration"]
    print(f"numbers -> {out}")
    print(f"  one-shot compensation {o['comp_mean']:.4f} +- {o['comp_sd0']:.4f} (s.d., ddof 0) over "
          f"{o['n_dots']} edits in {o['n_scenes']} scenes; {o['opposing']}/{o['n_dots']} oppose the "
          f"written angle")
    print(f"  panel c leaves out scene(s) (unedited chunk rotation > {o['drift_max_deg']:g} deg): "
          + (", ".join(f"{e} ({o['noop_chunk_rot_deg'][e]:.1f} deg)" for e in o["left_out"])
             or "none"))
    print("  panel c means (deg): " + ", ".join(f"{t:+.0f}: {v['mean_resp']:+.2f}"
                                                for t, v in sorted(o["per_angle"].items())))
    print("  held edit ratio -net/theta per angle: " + ", ".join(
        f"{t:+.0f}: {v['ratio']:.3f}" for t, v in sorted(p["per_angle"].items()))
        + f"  (range {p['ratio_range'][0]:.2f}-{p['ratio_range'][1]:.2f}); "
          f"{p['sign_correct']}/{p['n_cond']} opposing")
    d = p["fig6d_example"]
    print(f"  panel d, house {d['house']} rotation at step 32 (deg): " + ", ".join(
        f"{t:+.0f}: {v['raw']:+.2f}" for t, v in sorted(d["final_rot"].items()))
        + f"; unedited {p['fig6d_ref_final']:+.2f}")
    if "frac_mean_all" in e:
        print(f"  edit size {100 * e['frac_mean_all']:.2f}% +- {100 * e['frac_sd_all']:.2f}% of "
              f"the activation norm ({e['n_all']} edits)")
    else:
        print(f"  edit size: {e.get('note')}")
    print(f"  calibration slope median {c['slope_median']:.4f} (IQR {c['slope_iqr'][0]:.3f}-"
          f"{c['slope_iqr'][1]:.3f}) over {c['n_fibers']} held-out z sweeps; {c['n_test_houses']} of "
          f"{c['n_houses']} houses held out, {c['n_eps']} sweep files")
    print(f"  chart: {nums['chart_report']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
