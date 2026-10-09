#!/bin/bash
# GPU canaries of the Figure 6 harvest and training stages: one unit of each, compared with the
# reference stores. Run from the repository root on one GPU node with EGL, e.g.
#
#   C=<empty dir> REF_NATURAL_ROOT=... REF_NATURAL_CACHE=... REF_ORIENT=... REF_CHART=... \
#   REF_CHART_REPORT=... REF_OPERATOR=... REF_PERSIST=... REF_ONESHOT=... \
#       sbatch --gpus=1 --time=01:30:00 tests/canary_wrist_steer.sh [natural|orient|chart|steer|all]
#
# Reference locations (all read only):
#   REF_NATURAL_ROOT   natural rollout store with episodes/ (json) and episodes_extra/ (npz)
#   REF_NATURAL_CACHE  its natural cache (X, Xm, R, pos, phase, house, ep, q, success, layers)
#   REF_ORIENT         orientation sweep store (ep<E>_h<H>.npz)
#   REF_CHART, REF_CHART_REPORT   chart checkpoint and its report json
#   REF_OPERATOR       operator npz (V, c, Lhat, test_houses)
#   REF_PERSIST, REF_ONESHOT      steering jsons of the two modes
# Each unit writes under $C (use an empty directory) and prints IDENTICAL or DIFFERENT; the script
# exits non-zero if any compared unit differs. Times on one H200: natural about 30 min, orient about
# 8 min, chart about 4 min, steer about 10 min.
set -uo pipefail
source scripts/harvest_env.sh
PY="${PYTHON:-python}"
UNIT="${1:-all}"
: "${C:?set C to an empty scratch directory}"
mkdir -p "$C"
rc=0

if [[ "$UNIT" == natural || "$UNIT" == all ]]; then
  "$PY" pipelines/wrist_steer/10_natural_rollouts.py --partition 0 --out-dir "$C/natural" || rc=1
  "$PY" tests/canary_wrist_steer_natural.py --new "$C/natural" --ref-root "$REF_NATURAL_ROOT" \
      --ref-cache "$REF_NATURAL_CACHE" --episodes 0-15 || rc=1
fi
if [[ "$UNIT" == orient || "$UNIT" == all ]]; then
  "$PY" pipelines/wrist_steer/30_orient_sweep.py --task 0 --out-dir "$C/orient" || rc=1
  "$PY" tests/canary_wrist_steer_orient.py --canary "$C/orient/ep0_h0.npz" \
      --reference "$REF_ORIENT/ep0_h0.npz" || rc=1
fi
if [[ "$UNIT" == chart || "$UNIT" == all ]]; then
  "$PY" pipelines/wrist_steer/50_train_chart.py --orient-dir "$REF_ORIENT" --out-dir "$C/chart" || rc=1
  "$PY" tests/canary_wrist_steer_chart.py --new "$C/chart/chart.pt" --reference "$REF_CHART" \
      --new-report "$C/chart/chart_report.json" --ref-report "$REF_CHART_REPORT" || rc=1
fi
if [[ "$UNIT" == steer || "$UNIT" == all ]]; then
  for mode in persist oneshot; do
    ref="$REF_PERSIST"; [[ "$mode" == oneshot ]] && ref="$REF_ONESHOT"
    "$PY" pipelines/wrist_steer/60_steer.py --mode "$mode" --episodes 1 --chart "$REF_CHART" \
        --operator "$REF_OPERATOR" --out "$C/steer/$mode.json" || rc=1
    "$PY" tests/canary_wrist_steer_steer.py --new "$C/steer/$mode.json" --reference "$ref" || rc=1
  done
fi
exit $rc
