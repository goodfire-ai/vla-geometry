#!/bin/bash
# Example SLURM job array for a harvest stage: one GPU per array task. Each stage picks its unit of
# work (shard, task or row) from SLURM_ARRAY_TASK_ID. Submit from the repository root, e.g.:
#
#   Figure 1 (and the inputs of Figures 3, 6a and 7)
#   sbatch --array=0-151 scripts/harvest_array.sbatch pipelines/gripper/10_harvest_edits.py --save-captures
#   sbatch --array=0-63  scripts/harvest_array.sbatch pipelines/wrist/01_sweep.py
#   sbatch --array=0-113 scripts/harvest_array.sbatch pipelines/target/10_teleport_harvest.py
#   sbatch               scripts/harvest_array.sbatch pipelines/target/30_square_path_harvest.py
#   sbatch --array=0-74  scripts/harvest_array.sbatch pipelines/distance/10_near_harvest.py
#   Figure 2
#   sbatch --array=0-157 scripts/harvest_array.sbatch pipelines/position/10_crossed_harvest.py
#   sbatch --array=0-7   scripts/harvest_array.sbatch pipelines/position/50_hand_lines.py
#   sbatch --array=0-7   scripts/harvest_array.sbatch pipelines/position/55_object_lines.py
#   sbatch               scripts/harvest_array.sbatch pipelines/position/80_hand_shapes.py
#   sbatch               scripts/harvest_array.sbatch pipelines/position/85_object_shapes.py
#   Figure 4
#   sbatch --array=0-298 scripts/harvest_array.sbatch pipelines/contact/10_natural_harvest.py
#   sbatch --array=0-127 scripts/harvest_array.sbatch pipelines/contact/30_offschedule_harvest.py
#   Figure 5 (after pipelines/phase/05_prepare_assets.py)
#   sbatch --array=0-135 scripts/harvest_array.sbatch pipelines/phase/10_natural_rollouts.py
#   sbatch --array=0-127 scripts/harvest_array.sbatch pipelines/phase/40_freeze_harvest.py
#   sbatch               scripts/harvest_array.sbatch pipelines/phase/60_trajectory_capture.py
#   Figure 6
#   sbatch --array=0-12  scripts/harvest_array.sbatch pipelines/wrist_steer/10_natural_rollouts.py
#   sbatch --array=0-199 scripts/harvest_array.sbatch pipelines/wrist_steer/30_orient_sweep.py
#   sbatch               scripts/harvest_array.sbatch pipelines/wrist_steer/50_train_chart.py
#   sbatch               scripts/harvest_array.sbatch pipelines/wrist_steer/60_steer.py --mode oneshot
#   sbatch               scripts/harvest_array.sbatch pipelines/wrist_steer/60_steer.py --mode persist
#   Figure 7 (one episode per task; reads the model inputs saved by the gripper harvest)
#   sbatch --array=0-303 scripts/harvest_array.sbatch pipelines/gripper_fig7/30_ksweep.py
#
# Arguments after the script path are passed to it. Stages skip units whose output already exists,
# so a failed task can be submitted again; exit status 3 means the EGL renderer returned blank
# frames on that node, and the task should be run again (on another node if it repeats).
# Set PYTHON to the interpreter of the environment where this repository is installed with the
# [sim] extra (default: python). Run scripts/prefetch_assets.py once before the first submission.
# Each array task writes its log to vlageom-harvest_<job>_<task>.out in the submit directory
# (ignored by git); add e.g. --output=logs/%x_%A_%a.out (with an existing logs/) to collect them.
#
#SBATCH --job-name=vlageom-harvest
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=vlageom-harvest_%A_%a.out

set -euo pipefail
source scripts/harvest_env.sh
"${PYTHON:-python}" "$@"
