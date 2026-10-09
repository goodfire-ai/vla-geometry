#!/bin/bash
# GPU canaries of the Figure 2 harvests: one unit of each, compared row for row with the reference
# stores. Run from the repository root on one GPU node with EGL, e.g.
#
#   C=<empty dir> REF_CROSSED=... REF_HAND_LINES=... REF_OBJECT_LINES=... REF_SHAPES=... \
#       sbatch --gpus=1 --time=01:30:00 tests/canary_position.sh [crossed|hand_lines|object_lines|hand_shapes|object_shapes|all]
#
# Reference locations (all read only):
#   REF_CROSSED       crossed store root (houses/h44.npz)
#   REF_HAND_LINES    hand-line directory (h44.npz)
#   REF_OBJECT_LINES  object-line root (shape_line_ee/h44/)
#   REF_SHAPES        trace-grid path root (ee_{kind}/h44/, obj_{kind}/h44/)
# Units (house 44 throughout, the house of the trace grids):
#   crossed        one house, 5 placements x 10 poses x 2 queries        about 5 min
#   hand_lines     one house, 4 lines x 24 waypoints                     about 12 min
#   object_lines   one house, the line toward the hand (24 waypoints)    about 4 min
#   hand_shapes    one house, the x line of the EE grid (24 waypoints)   about 4 min
#   object_shapes  one house, the x line of the target grid              about 4 min
# Times include loading the model (about 2 min) on one H200. Each unit writes under $C (use an
# empty directory) and prints IDENTICAL or DIFFERENT; the script exits non-zero if any compared
# unit differs.
set -uo pipefail
source scripts/harvest_env.sh
PY="${PYTHON:-python}"
UNIT="${1:-all}"
: "${C:?set C to an empty scratch directory}"
mkdir -p "$C"
rc=0
want() { [[ "$UNIT" == "$1" || "$UNIT" == all ]]; }

if want crossed; then
  "$PY" pipelines/position/10_crossed_harvest.py --houses 44 --out "$C/crossed" || rc=1
  "$PY" tests/canary_position_harvest.py crossed --house 44 --new "$C/crossed" --ref "$REF_CROSSED" || rc=1
fi
if want hand_lines; then
  "$PY" pipelines/position/50_hand_lines.py --houses 44 --out "$C/hand_lines" || rc=1
  "$PY" tests/canary_position_harvest.py lines --house 44 --new "$C/hand_lines" --ref "$REF_HAND_LINES" || rc=1
fi
if want object_lines; then
  "$PY" pipelines/position/55_object_lines.py --houses 44 --shapes line_ee --out "$C/object_lines" || rc=1
  "$PY" tests/canary_position_harvest.py shape --new "$C/object_lines/shape_line_ee/h44" \
      --ref "$REF_OBJECT_LINES/shape_line_ee/h44" || rc=1
fi
if want hand_shapes; then
  "$PY" pipelines/position/80_hand_shapes.py --house 44 --shapes line_x --out "$C/shapes" || rc=1
  "$PY" tests/canary_position_harvest.py shape --new "$C/shapes/ee_line_x/h44" \
      --ref "$REF_SHAPES/ee_line_x/h44" || rc=1
fi
if want object_shapes; then
  "$PY" pipelines/position/85_object_shapes.py --house 44 --shapes line_fwd --out "$C/shapes" || rc=1
  "$PY" tests/canary_position_harvest.py shape --new "$C/shapes/obj_line_fwd/h44" \
      --ref "$REF_SHAPES/obj_line_fwd/h44" || rc=1
fi
exit $rc
