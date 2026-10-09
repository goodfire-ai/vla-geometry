#!/bin/bash
# GPU canaries of the Figure 4 harvest stages: one episode of each, compared with the reference
# stores. Run from the repository root on one GPU node with EGL, e.g.
#
#   C=<empty dir> REF_NATURAL=... REF_OFFSCHEDULE=... \
#       sbatch --gpus=1 --time=00:30:00 tests/canary_contact.sh [natural|offschedule|all]
#
# Reference locations (read only):
#   REF_NATURAL      natural harvest files (ep<E>_h<H>.npz) of stage 10
#   REF_OFFSCHEDULE  off-schedule harvest files (ep<E>_h<H>.npz) of stage 30
# Units: natural = task 0 (episode 0, house 0, 63 queries); offschedule = task 0 (episode 0,
# house 0, 131 placements). Each unit writes under $C (use an empty directory) and the comparison
# prints PASS or FAIL; the script exits non-zero if any unit fails. On the GPU type of the
# reference run the natural unit must repeat the rollout exactly (every query within the tolerance,
# the rollout fields bit for bit); on other hardware set LENIENT=1 to require only its first query
# (see tests/canary_contact_harvest.py). Times on one H200 including model loading: natural about
# 3 min, offschedule about 6 min.
set -uo pipefail
source scripts/harvest_env.sh
PY="${PYTHON:-python}"
UNIT="${1:-all}"
: "${C:?set C to an empty scratch directory}"
mkdir -p "$C"
rc=0

if [[ "$UNIT" == natural || "$UNIT" == all ]]; then
  : "${REF_NATURAL:?set REF_NATURAL to the reference natural harvest directory}"
  "$PY" pipelines/contact/10_natural_harvest.py --task 0 --out "$C/contact" || rc=1
  "$PY" tests/canary_contact_harvest.py natural --new "$C/contact/natural/ep0_h0.npz" \
      --reference "$REF_NATURAL/ep0_h0.npz" ${LENIENT:+--lenient} || rc=1
fi
if [[ "$UNIT" == offschedule || "$UNIT" == all ]]; then
  : "${REF_OFFSCHEDULE:?set REF_OFFSCHEDULE to the reference off-schedule harvest directory}"
  "$PY" pipelines/contact/30_offschedule_harvest.py --task 0 --out "$C/contact" || rc=1
  "$PY" tests/canary_contact_harvest.py offschedule --new "$C/contact/offschedule/ep0_h0.npz" \
      --reference "$REF_OFFSCHEDULE/ep0_h0.npz" || rc=1
fi
exit $rc
