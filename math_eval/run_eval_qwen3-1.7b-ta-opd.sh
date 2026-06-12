#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TA_OPD_METHOD="${TA_OPD_METHOD:-teachability}"
TA_OPD_RATIO="${TA_OPD_RATIO:-0.1}"
TA_OPD_TOPK="${TA_OPD_TOPK:-16}"
TA_OPD_SEED="${TA_OPD_SEED:-42}"
TA_OPD_EXPERIMENT_NAME="${TA_OPD_EXPERIMENT_NAME:-qwen3_4b_ta_opd_${TA_OPD_METHOD}_ratio${TA_OPD_RATIO}_k${TA_OPD_TOPK}_seed${TA_OPD_SEED}_teacher_qwen3_4b_non_thinking_rl_math}"
TA_OPD_SAVE_FREQ="${TA_OPD_SAVE_FREQ:-50}"
TA_OPD_CKPT_DIR="${TA_OPD_CKPT_DIR:-/TA_OPD-checkpoints/${TA_OPD_EXPERIMENT_NAME}_save_step_${TA_OPD_SAVE_FREQ}}"
TA_OPD_STEP="${TA_OPD_STEP:-110}"

# run_eval_math.sh uses these generic variables to locate and merge checkpoints.
export G_OPD_CKPT_DIR="$TA_OPD_CKPT_DIR"
export FSDP_CKPT_DIR="${FSDP_CKPT_DIR:-$TA_OPD_CKPT_DIR/global_step_${TA_OPD_STEP}/actor}"

export DATASETS="${DATASETS:-aime24 aime25 aime26 hmmt26 amc23 math500}"
export GPU_GROUP_0="${GPU_GROUP_0:-0,1}"
export GPU_GROUP_1="${GPU_GROUP_1:-2,3}"
export SHOW_SUMMARY="${SHOW_SUMMARY:-1}"

echo "[eval-ta-opd] experiment=$TA_OPD_EXPERIMENT_NAME"
echo "[eval-ta-opd] method=$TA_OPD_METHOD"
echo "[eval-ta-opd] ratio=$TA_OPD_RATIO"
echo "[eval-ta-opd] topk=$TA_OPD_TOPK"
echo "[eval-ta-opd] seed=$TA_OPD_SEED"
echo "[eval-ta-opd] ckpt_dir=$TA_OPD_CKPT_DIR"
echo "[eval-ta-opd] step=$TA_OPD_STEP"
echo "[eval-ta-opd] fsdp_ckpt_dir=$FSDP_CKPT_DIR"
echo "[eval-ta-opd] datasets=$DATASETS"
echo "[eval-ta-opd] gpu_group_0=$GPU_GROUP_0"
echo "[eval-ta-opd] gpu_group_1=$GPU_GROUP_1"

exec bash "$SCRIPT_DIR/run_eval_math.sh" "$@"
