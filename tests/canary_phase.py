"""Compare one unit of a Figure 5 harvest stage with the same unit of a reference store. The npz
units are compared file by file and array by array (tests/compare_phase_arrays.compare_npz: every
key, dtype, shape and value, NaN equal to NaN).

  scan        stage 05 house scan: per scanned house, valid, reason, pickup object, receptacle
              and task description (the scan time is not compared)
  natural     stage 10 episode files ep<EEEEE>_h<H>.npz of the listed episodes of --partition (or
              --episodes)
  freeze      stage 40 file ep<E>_h<H>.npz of --task
  trajectory  stage 60 capture traj_ep<E>_h<H>.npz

  python pipelines/phase/05_prepare_assets.py --skip-receptacles --skip-scenes --scan \
      --houses 0 1 2 3 --out $C/house_validity.json
  python tests/canary_phase.py scan --new $C/house_validity.json --ref <house validity scan>

  python pipelines/phase/10_natural_rollouts.py --partition 0 --out-dir $C/natural
  python tests/canary_phase.py natural --new $C/natural --ref <natural episode store> --partition 0

  python pipelines/phase/40_freeze_harvest.py --task 0 --out-dir $C/freeze
  python tests/canary_phase.py freeze --new $C/freeze --ref <freeze-test store> --task 0

  python pipelines/phase/60_trajectory_capture.py --out $C/trajectory/traj_ep249_h76.npz
  python tests/canary_phase.py trajectory --new $C/trajectory/traj_ep249_h76.npz \
      --ref <reference capture>

The policy runs in bfloat16 on the GPU and renders with EGL; a unit reproduces the reference to
the last bit only on the same GPU type, driver and library versions. Exit status 0 when every
compared file is identical.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare_phase_arrays import compare_npz  # noqa: E402

from vlageom import phase as P  # noqa: E402


def compare_pairs(pairs) -> bool:
    ok = True
    for new, ref in pairs:
        if not new.exists() or not ref.exists():
            print(f"  MISSING {new if not new.exists() else ref}")
            ok = False
            continue
        diffs = compare_npz(new, ref)
        if diffs:
            ok = False
            print(f"  DIFF  {new.name}")
            for line in diffs:
                print(f"        {line}")
        else:
            print(f"  equal {new.name}")
    return ok


SCAN_FIELDS = ("valid", "reason", "pickup", "receptacle", "description")


def compare_scan(new: Path, ref: Path) -> bool:
    """Per house of the new scan: the SCAN_FIELDS equal those of the reference scan."""
    a = json.loads(new.read_text())["houses"]
    b = json.loads(ref.read_text())["houses"]
    ok = True
    for h in sorted(a, key=int):
        if h not in b:
            print(f"  MISSING house {h} in the reference")
            ok = False
            continue
        diff = [k for k in SCAN_FIELDS if a[h].get(k) != b[h].get(k)]
        if diff:
            ok = False
            print(f"  DIFF  house {h}: "
                  + "; ".join(f"{k} {a[h].get(k)!r} vs {b[h].get(k)!r}" for k in diff))
        else:
            print(f"  equal house {h} ({'valid' if a[h]['valid'] else 'not valid'})")
    return ok and len(a) > 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("stage", choices=("scan", "natural", "freeze", "trajectory"))
    ap.add_argument("--new", type=Path, required=True,
                    help="output directory (file for scan and trajectory)")
    ap.add_argument("--ref", type=Path, required=True,
                    help="reference directory (file for scan and trajectory)")
    ap.add_argument("--partition", type=int, default=None)
    ap.add_argument("--episodes", type=int, nargs="+", default=None)
    ap.add_argument("--task", type=int, default=None)
    a = ap.parse_args(argv)

    if a.stage == "scan":
        print(f"scan\n  new {a.new}\n  ref {a.ref}")
        ok = compare_scan(a.new, a.ref)
        print("IDENTICAL" if ok else "DIFFERENT")
        return 0 if ok else 1
    if a.stage == "natural":
        eps = P.read_natural_episodes()
        if a.episodes:
            sel = [e for e in eps if e["episode_index"] in set(a.episodes)]
        elif a.partition is not None:
            sel = [e for e in eps if e["partition"] == a.partition]
        else:
            ap.error("give --partition or --episodes")
        names = [P.natural_name(e["episode_index"], e["house"]) for e in sel]
        pairs = [(a.new / n, a.ref / n) for n in names]
    elif a.stage == "freeze":
        if a.task is None:
            ap.error("give --task")
        r = next(r for r in P.read_freeze_episodes() if r["task"] == a.task)
        n = P.freeze_name(r["episode_index"], r["house"])
        pairs = [(a.new / n, a.ref / n)]
    else:
        pairs = [(a.new, a.ref)]
    print(f"{a.stage}: {len(pairs)} file(s)\n  new {a.new}\n  ref {a.ref}")
    ok = compare_pairs(pairs)
    print("IDENTICAL" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
