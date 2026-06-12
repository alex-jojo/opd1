#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXOPD_LAMBDA="${EXOPD_LAMBDA:-1.25}"
EXOPD_LAMBDA_TAG="${EXOPD_LAMBDA//./p}"

export G_OPD_EXPERIMENT_NAME="${G_OPD_EXPERIMENT_NAME:-146_qwen3_4b_teacher_qwen3_4b_single_teacher_exopd_lambda_${EXOPD_LAMBDA_TAG}}"
export G_OPD_SAVE_FREQ="${G_OPD_SAVE_FREQ:-50}"
export G_OPD_CKPT_DIR="${G_OPD_CKPT_DIR:-/G-OPD-checkpoints/${G_OPD_EXPERIMENT_NAME}_save_step_${G_OPD_SAVE_FREQ}}"
EXOPD_STEP="${EXOPD_STEP:-110}"
export FSDP_CKPT_DIR="${FSDP_CKPT_DIR:-$G_OPD_CKPT_DIR/global_step_${EXOPD_STEP}/actor}"

export DATASETS="${DATASETS:-aime24 aime25 aime26 hmmt26 amc23 math500}"
export GPU_GROUP_0="${GPU_GROUP_0:-0,1}"
export GPU_GROUP_1="${GPU_GROUP_1:-2,3}"
export SHOW_SUMMARY="${SHOW_SUMMARY:-1}"

echo "[eval-exopd] experiment=$G_OPD_EXPERIMENT_NAME"
echo "[eval-exopd] save_freq=$G_OPD_SAVE_FREQ"
echo "[eval-exopd] step=$EXOPD_STEP"
echo "[eval-exopd] datasets=$DATASETS"
echo "[eval-exopd] gpu_group_0=$GPU_GROUP_0"
echo "[eval-exopd] gpu_group_1=$GPU_GROUP_1"
if [ -n "${G_OPD_CKPT_DIR:-}" ]; then
    echo "[eval-exopd] ckpt_dir=$G_OPD_CKPT_DIR"
fi
if [ -n "${FSDP_CKPT_DIR:-}" ]; then
    echo "[eval-exopd] fsdp_ckpt_dir=$FSDP_CKPT_DIR"
fi
if [ -n "${MODEL_PATH:-}" ]; then
    echo "[eval-exopd] model_path=$MODEL_PATH"
fi

exec bash "$SCRIPT_DIR/run_eval_math.sh" "$@"
