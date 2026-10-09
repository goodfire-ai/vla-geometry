"""Stage 40 (CPU): per-k medians of the restricted gripper edits (Figure 7d).

Reads the stage 30 records (ep<E>_h<H>_<landmark>.json, file-name order) of the listed episodes
(lists/gripper_episodes.csv) from --ksweep and summarizes each landmark
(vlageom.gripper_fig7.summarize):

  per_k[k]   n, flip_pct (share of fits whose chunk-mean gripper command has the target's sign),
             n_opp / flip_pct_opp (the same over the queries whose unedited command is opposite to
             the target), medians of arm_leakage (leak), arm_jerk (jerk), grip_after, norm
  ceiling    the same for the unrestricted 768-D edits fitted in the same run
  knee       smallest k > 0 whose flip rate is within 2 points of the ceiling's, whose median
             leakage is at most 1.5 x the ceiling's and whose median jerk is at most that of the
             unedited chunk (not drawn)
  opposite_only  the same summary over the queries whose unedited command is opposite to the
             target

Writes --out (default $VLAGEOM_DATA/gripper_fig7/ksweep_summary.json) and prints, per landmark,
the leakage relative to the 768-D edit, 100 x leak(k) / ceiling leak, which Figure 7d draws, and
the listed episodes without records. Takes seconds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlageom import gripper as G
from vlageom import gripper_fig7 as F
from vlageom import paths


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n", 1)[1])
    ap.add_argument("--ksweep", type=Path, default=paths.tile_dir(F.TILE, "ksweep"),
                    help="stage 30 record directory (default: $VLAGEOM_DATA/gripper_fig7/ksweep)")
    ap.add_argument("--episodes", type=Path, default=None,
                    help="episode list CSV (default lists/gripper_episodes.csv)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSON (default: $VLAGEOM_DATA/gripper_fig7/ksweep_summary.json)")
    a = ap.parse_args()

    episodes = G.read_episodes(a.episodes)
    recs = F.read_records(a.ksweep, episodes)
    if not recs:
        raise SystemExit(f"no records of the listed episodes in {a.ksweep}")
    summary = F.summarize(recs)
    out = a.out or paths.tile_dir(F.TILE, create=True) / "ksweep_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    have = {(r["ep"], r["house"]) for r in recs}
    missing = [f"ep{e}_h{h}" for e, h, *_ in episodes if (e, h) not in have]
    print(f"{len(recs)} records of {len(have)} listed episodes in {a.ksweep}")
    for lm, s in summary.items():
        ks = [k for k in s["per_k"] if k > 0]
        rel = " ".join(f"{k}:{100.0 * s['per_k'][k]['leak'] / s['ceiling']['leak']:.1f}"
                       for k in ks)
        print(f"{lm}: n {s['n']}, 768-D edit: flip {s['ceiling']['flip_pct']:.2f}%, "
              f"median leak {s['ceiling']['leak']:.5f}; knee {s['knee']}")
        print(f"  leak / 768-D leak (%) by k: {rel}")
    if missing:
        print(f"listed episodes without records ({len(missing)}): {' '.join(missing[:20])}"
              + (" ..." if len(missing) > 20 else ""))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
