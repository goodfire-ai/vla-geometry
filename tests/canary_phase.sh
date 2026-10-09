#!/bin/bash
# GPU canaries of the Figure 5 harvest stages: one unit of each, compared with the reference
# stores. Run from the repository root on one GPU node with EGL, e.g.
#
#   C=<empty dir> REF_VALIDITY=... REF_NATURAL=... REF_FREEZE=... REF_TRAJECTORY=... \
#       sbatch --gpus=1 --time=01:00:00 tests/canary_phase.sh [scan|natural|freeze|trajectory|all]
#
# Reference locations (all read only):
#   REF_VALIDITY     house validity scan (json) of stage 05 --scan
#   REF_NATURAL      natural episode store (ep<EEEEE>_h<H>.npz) of stage 10
#   REF_FREEZE       freeze-test store (ep<E>_h<H>.npz) of stage 40
#   REF_TRAJECTORY   trajectory capture (traj_ep249_h76.npz) of stage 60
# Units: scan = houses 0-3 (house 2 cannot host the task); natural = partition 0 (episodes 0-3,
# houses 0, 1, 3, 4; episode 0 sampled by a new task sampler, 1-3 by the existing one); freeze =
# task 0 (episode 0, house 0); trajectory = the listed example rollout. Each unit writes under $C
# (use an empty directory) and prints IDENTICAL or DIFFERENT; the script exits non-zero if any
# compared unit differs. Times on one H200 including model loading: scan about 4 min, natural about
# 15 min, freeze about 8 min, trajectory about 6 min.
set -uo pipefail
source scripts/harvest_env.sh
PY="${PYTHON:-python}"
UNIT="${1:-all}"
: "${C:?set C to an empty scratch directory}"
mkdir -p "$C"
rc=0

if [[ "$UNIT" == scan || "$UNIT" == all ]]; then
  "$PY" pipelines/phase/05_prepare_assets.py --skip-receptacles --skip-scenes --scan \
      --houses 0 1 2 3 --out "$C/house_validity.json" || rc=1
  "$PY" tests/canary_phase.py scan --new "$C/house_validity.json" --ref "$REF_VALIDITY" || rc=1
fi
if [[ "$UNIT" == natural || "$UNIT" == all ]]; then
  "$PY" pipelines/phase/10_natural_rollouts.py --partition 0 --out-dir "$C/natural" || rc=1
  "$PY" tests/canary_phase.py natural --new "$C/natural" --ref "$REF_NATURAL" --partition 0 || rc=1
fi
if [[ "$UNIT" == freeze || "$UNIT" == all ]]; then
  "$PY" pipelines/phase/40_freeze_harvest.py --task 0 --out-dir "$C/freeze" || rc=1
  "$PY" tests/canary_phase.py freeze --new "$C/freeze" --ref "$REF_FREEZE" --task 0 || rc=1
fi
if [[ "$UNIT" == trajectory || "$UNIT" == all ]]; then
  "$PY" pipelines/phase/60_trajectory_capture.py --out "$C/trajectory/traj_ep249_h76.npz"; s=$?
  [[ $s == 0 || $s == 2 ]] || rc=1
  "$PY" tests/canary_phase.py trajectory --new "$C/trajectory/traj_ep249_h76.npz" \
      --ref "$REF_TRAJECTORY" || rc=1
fi
exit $rc
