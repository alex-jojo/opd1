#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXOPD_LAMBDA="${EXOPD_LAMBDA:-1.25}"
EXOPD_LAMBDA_TAG="${EXOPD_LAMBDA//./p}"
export G_OPD_EXPERIMENT_NAME="${G_OPD_EXPERIMENT_NAME:-146_qwen3_4b_teacher_qwen3_4b_instruct_2507_code_single_teacher_exopd_lambda_${EXOPD_LAMBDA_TAG}}"
export G_OPD_SAVE_FREQ="${G_OPD_SAVE_FREQ:-109}"
export G_OPD_CKPT_DIR="${G_OPD_CKPT_DIR:-/G-OPD-checkpoints/${G_OPD_EXPERIMENT_NAME}_save_step_${G_OPD_SAVE_FREQ}}"
EXOPD_STEP="${EXOPD_STEP:-109}"
export FSDP_CKPT_DIR="${FSDP_CKPT_DIR:-$G_OPD_CKPT_DIR/global_step_${EXOPD_STEP}/actor}"
export MODEL_NAME="${MODEL_NAME:-${G_OPD_EXPERIMENT_NAME}_save_step_${G_OPD_SAVE_FREQ}_global_step_${EXOPD_STEP}_code_n8}"

echo "[code-eval-exopd] experiment=$G_OPD_EXPERIMENT_NAME"
echo "[code-eval-exopd] ckpt_dir=$G_OPD_CKPT_DIR"
echo "[code-eval-exopd] step=$EXOPD_STEP"
echo "[code-eval-exopd] fsdp_ckpt_dir=$FSDP_CKPT_DIR"
exec bash "$SCRIPT_DIR/run_eval_code.sh" "$@"
