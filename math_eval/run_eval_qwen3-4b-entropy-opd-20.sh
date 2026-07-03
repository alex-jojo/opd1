#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENTROPY_OPD_FRACTION="${ENTROPY_OPD_FRACTION:-0.2}"
ENTROPY_OPD_FRACTION_TAG="${ENTROPY_OPD_FRACTION//./p}"

if [ "$ENTROPY_OPD_FRACTION" = "0.2" ]; then
    DEFAULT_EXPERIMENT_NAME="146_qwen3_1.7b_teacher_qwen3_4b_entropy_opd_20"
else
    DEFAULT_EXPERIMENT_NAME="146_qwen3_1.7b_teacher_qwen3_4b_entropy_opd_${ENTROPY_OPD_FRACTION_TAG}"
fi

export G_OPD_EXPERIMENT_NAME="${G_OPD_EXPERIMENT_NAME:-$DEFAULT_EXPERIMENT_NAME}"
export G_OPD_SAVE_FREQ="${G_OPD_SAVE_FREQ:-50}"

export DATASETS="${DATASETS:-aime24 aime25 aime26 hmmt26 amc23 math500}"
export GPU_GROUP_0="${GPU_GROUP_0:-0,1}"
export GPU_GROUP_1="${GPU_GROUP_1:-2,3}"
export SHOW_SUMMARY="${SHOW_SUMMARY:-1}"

echo "[eval-entropy-opd] experiment=$G_OPD_EXPERIMENT_NAME"
echo "[eval-entropy-opd] fraction=$ENTROPY_OPD_FRACTION"
echo "[eval-entropy-opd] save_freq=$G_OPD_SAVE_FREQ"
echo "[eval-entropy-opd] datasets=$DATASETS"
echo "[eval-entropy-opd] gpu_group_0=$GPU_GROUP_0"
echo "[eval-entropy-opd] gpu_group_1=$GPU_GROUP_1"
if [ -n "${G_OPD_CKPT_DIR:-}" ]; then
    echo "[eval-entropy-opd] ckpt_dir=$G_OPD_CKPT_DIR"
fi
if [ -n "${FSDP_CKPT_DIR:-}" ]; then
    echo "[eval-entropy-opd] fsdp_ckpt_dir=$FSDP_CKPT_DIR"
fi
if [ -n "${MODEL_PATH:-}" ]; then
    echo "[eval-entropy-opd] model_path=$MODEL_PATH"
fi

exec bash "$SCRIPT_DIR/run_eval_math.sh" "$@"
