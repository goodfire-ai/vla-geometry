#!/bin/bash
# GPU canary of pipelines/gripper_fig7/30_ksweep.py (Figure 7d): episode 0 (house 0), both grasp
# landmarks, compared field by field with the records of another run of stage 30 by
# tests/canary_gripper_ksweep.py. Run from the repository root on one GPU node (no rendering: the
# fits read stored model inputs), e.g.
#
#   C=<empty dir> REF_EDITS=... REF_CAPTURES=... REF_RECORDS=... \
#       sbatch --gpus=1 --time=02:00:00 tests/canary_gripper_ksweep.sh
#
# Reference locations (all read only):
#   REF_EDITS     directory of the gripper edit files of the listed episodes (the bases are built
#                 from them, as in stage 30)
#   REF_CAPTURES  directory of the per-episode model-input files (ep<E>_h<H>.pt)
#   REF_RECORDS   directory of reference records of stage 30 (ep<E>_h<H>_<landmark>.json) fitted
#                 from the same edit files and model inputs
# Output goes to $C/ksweep (use an empty directory). Exit status 0 when both records are
# identical. About 45 min on one H200, including about 2 min to load the model.
set -uo pipefail
source scripts/harvest_env.sh
PY="${PYTHON:-python}"
: "${C:?set C to an empty scratch directory}"
: "${REF_EDITS:?set REF_EDITS}" "${REF_CAPTURES:?set REF_CAPTURES}" "${REF_RECORDS:?set REF_RECORDS}"
mkdir -p "$C"
"$PY" pipelines/gripper_fig7/30_ksweep.py --episode-index 0 --edits "$REF_EDITS" \
    --from-captures "$REF_CAPTURES" --out "$C/ksweep" || exit 1
"$PY" tests/canary_gripper_ksweep.py --run "$C/ksweep" --reference "$REF_RECORDS" --episode-index 0
