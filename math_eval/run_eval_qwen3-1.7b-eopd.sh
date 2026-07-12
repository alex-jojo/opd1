#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EOPD_EXPERIMENT_NAME="${EOPD_EXPERIMENT_NAME:-eopd_217_qwen3_4b_eopd_teacher_qwen3_4b_non_thinking_rl_math}"
EOPD_SAVE_FREQ="${EOPD_SAVE_FREQ:-50}"
EOPD_CKPT_DIR="${EOPD_CKPT_DIR:-/EOPD-checkpoints/${EOPD_EXPERIMENT_NAME}_save_step_${EOPD_SAVE_FREQ}}"
EOPD_STEP="${EOPD_STEP:-110}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
# run_eval_math.sh uses this generic checkpoint root when auto-selecting a step.
export G_OPD_CKPT_DIR="$EOPD_CKPT_DIR"

export FSDP_CKPT_DIR="${FSDP_CKPT_DIR:-$EOPD_CKPT_DIR/global_step_${EOPD_STEP}/actor}"

export DATASETS="${DATASETS:-aime24 aime25 aime26 hmmt26 amc23 math500}"
export GPU_GROUP_0="${GPU_GROUP_0:-0,1}"
export GPU_GROUP_1="${GPU_GROUP_1:-2,3}"
export SHOW_SUMMARY="${SHOW_SUMMARY:-1}"

echo "[eval-eopd] experiment=$EOPD_EXPERIMENT_NAME"
echo "[eval-eopd] ckpt_dir=$EOPD_CKPT_DIR"
echo "[eval-eopd] step=$EOPD_STEP"
echo "[eval-eopd] datasets=$DATASETS"
echo "[eval-eopd] gpu_group_0=$GPU_GROUP_0"
echo "[eval-eopd] gpu_group_1=$GPU_GROUP_1"

exec bash "$SCRIPT_DIR/run_eval_math.sh" "$@"
