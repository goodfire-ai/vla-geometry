"""Wrist tile, stage 2 (CPU): check the sweep files before pooling.

For every file matching --sweeps-glob, checks that:
  * it holds every sweep key (`vlageom.wrist.SWEEP_KEYS`) and 39 rows (38 angles + the episode's own),
  * the swept joint is arm joint 6 and the swept range equals --range-rad (the FR3 joint 7 limits),
    with the first and last angle at the two limits,
  * the EE point stayed within --max-drift-mm (the roll joint turns the hand in place) and the
    wrist-camera images at the two ends of the sweep differ (the camera turns with the hand),
  * the mean exo-camera pixel value is at least --min-px (not dark),
  * the episode and house in `meta` equal the file's row of lists/wrist_episodes.csv (the catalog
    index is read from the file name, ...ep<catalog_index>.npz).
It also reports listed episodes without a file. Exit status 0 when every check passes.

  python pipelines/wrist/02_check_sweeps.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

from vlageom import wrist  # noqa: E402


def check_file(f, row, args):
    """List of failed-check messages for one sweep file (empty when it passes) and a summary dict."""
    bad = []
    z = np.load(f, allow_pickle=True)
    missing = [k for k in wrist.SWEEP_KEYS if k not in z.files]
    if missing:
        return [f"missing keys {missing}"], {}
    n = len(z["jval"])
    jv = np.asarray(z["jval"], np.float64)
    rng = np.asarray(z["sweep_range_rad"], np.float64)
    drift = float(np.max(z["drift_mm"]))
    px = float(np.mean([np.asarray(x, np.uint8).mean() for x in z["frames"]]))
    fw = z["frames_wrist"]
    wrist_differs = not np.array_equal(np.asarray(fw[0], np.uint8), np.asarray(fw[-1], np.uint8))
    meta = str(z["meta"])
    if n != wrist.N_STEPS + 1:
        bad.append(f"{n} rows (expect {wrist.N_STEPS + 1})")
    if z["act"].shape[:1] != (n,) or [int(x) for x in z["layers"]] != wrist.LAYERS:
        bad.append(f"act {z['act'].shape} / layers {z['layers'].tolist()}")
    if int(z["roll_joint"]) != wrist.ROLL_JOINT:
        bad.append(f"roll joint {int(z['roll_joint'])} (expect {wrist.ROLL_JOINT})")
    if not np.allclose(rng, args.range_rad, atol=1e-4):
        bad.append(f"range {rng.tolist()} (expect {args.range_rad})")
    if not (np.all(np.diff(jv) >= 0) and np.isclose(jv[0], rng[0], atol=1e-6)
            and np.isclose(jv[-1], rng[1], atol=1e-6)):
        bad.append("angles not ascending from limit to limit")
    if drift > args.max_drift_mm:
        bad.append(f"EE drift {drift:.3f} mm")
    if not wrist_differs:
        bad.append("wrist-camera images identical at both ends")
    if px < args.min_px:
        bad.append(f"mean exo px {px:.1f}")
    if row is None:
        bad.append("catalog index not in lists/wrist_episodes.csv")
    elif not meta.startswith(f"ep{row['episode_index']} h{row['house']} "):
        bad.append(f"meta '{meta}' (expect ep{row['episode_index']} h{row['house']})")
    return bad, dict(n=n, span=float(np.degrees(np.ptp(jv))), drift=drift, px=px,
                     q0=float(np.degrees(z["q0"][wrist.ROLL_JOINT])), meta=meta)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweeps-glob", default=None,
                    help="glob of sweep files (default <data>/wrist/sweeps/ep*.npz)")
    ap.add_argument("--range-rad", type=float, nargs=2, default=[-3.0159, 3.0159],
                    help="expected joint limits (rad) of the swept joint (default %(default)s)")
    ap.add_argument("--min-px", type=float, default=wrist.MIN_PX,
                    help="minimum mean exo-camera pixel value (default %(default)s)")
    ap.add_argument("--max-drift-mm", type=float, default=1.0,
                    help="maximum EE position change over the sweep (default %(default)s mm)")
    args = ap.parse_args(argv)

    pattern = args.sweeps_glob or wrist.default_sweep_glob()
    files = wrist.sweep_files(pattern)
    by_idx = {r["catalog_index"]: r for r in wrist.read_episodes()}
    seen, n_bad, summ = set(), 0, []
    for f in files:
        m = re.search(r"ep(\d+)\.npz$", f)
        idx = int(m.group(1)) if m else None
        seen.add(idx)
        try:
            bad, s = check_file(f, by_idx.get(idx), args)
        except Exception as e:                                   # noqa: BLE001
            bad, s = [f"unreadable: {type(e).__name__}: {e}"], {}
        if bad:
            n_bad += 1
            print(f"FAIL {Path(f).name}: " + "; ".join(bad))
        if s:
            summ.append(s)
    missing = sorted(set(by_idx) - seen)
    print(f"{len(files)} sweep files matched {pattern}")
    if summ:
        a = {k: np.array([s[k] for s in summ]) for k in ("n", "span", "drift", "px", "q0")}
        print(f"  rows {sorted(set(a['n'].tolist()))}; angle range {a['span'].min():.1f}-"
              f"{a['span'].max():.1f} deg; max EE drift {a['drift'].max():.3f} mm; "
              f"mean exo px >= {a['px'].min():.1f}; episode angle {a['q0'].min():.2f} to "
              f"{a['q0'].max():.2f} deg")
    if missing:
        print(f"  {len(missing)} listed episodes have no file: catalog_index {missing}")
    ok = n_bad == 0 and not missing and len(files) > 0
    print("all checks passed" if ok else f"{n_bad} files failed, {len(missing)} missing")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
