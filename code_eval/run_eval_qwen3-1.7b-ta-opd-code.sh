#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TA_OPD_METHOD="${TA_OPD_METHOD:-teachability}"
TA_OPD_RATIO="${TA_OPD_RATIO:-0.1}"
TA_OPD_TOPK="${TA_OPD_TOPK:-16}"
TA_OPD_SEED="${TA_OPD_SEED:-42}"
TA_OPD_EXPERIMENT_NAME="${TA_OPD_EXPERIMENT_NAME:-qwen3_1_7b_ta_opd_${TA_OPD_METHOD}_ratio${TA_OPD_RATIO}_k${TA_OPD_TOPK}_seed${TA_OPD_SEED}_teacher_qwen3_4b_instruct_2507_code}"
TA_OPD_SAVE_FREQ="${TA_OPD_SAVE_FREQ:-109}"
TA_OPD_CKPT_DIR="${TA_OPD_CKPT_DIR:-/TA_OPD-checkpoints/${TA_OPD_EXPERIMENT_NAME}_save_step_${TA_OPD_SAVE_FREQ}}"
TA_OPD_STEP="${TA_OPD_STEP:-109}"
export G_OPD_CKPT_DIR="$TA_OPD_CKPT_DIR"
export FSDP_CKPT_DIR="${FSDP_CKPT_DIR:-$TA_OPD_CKPT_DIR/global_step_${TA_OPD_STEP}/actor}"
export MODEL_NAME="${MODEL_NAME:-${TA_OPD_EXPERIMENT_NAME}_save_step_${TA_OPD_SAVE_FREQ}_global_step_${TA_OPD_STEP}_code_n8}"

echo "[code-eval-ta-opd] experiment=$TA_OPD_EXPERIMENT_NAME"
echo "[code-eval-ta-opd] ckpt_dir=$TA_OPD_CKPT_DIR"
echo "[code-eval-ta-opd] step=$TA_OPD_STEP"
echo "[code-eval-ta-opd] fsdp_ckpt_dir=$FSDP_CKPT_DIR"
exec bash "$SCRIPT_DIR/run_eval_code.sh" "$@"
