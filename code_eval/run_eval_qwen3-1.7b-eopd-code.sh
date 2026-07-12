#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EOPD_EXPERIMENT_NAME="${EOPD_EXPERIMENT_NAME:-eopd_217_qwen3_1_7b_eopd_teacher_qwen3_4b_instruct_2507_code}"
EOPD_SAVE_FREQ="${EOPD_SAVE_FREQ:-109}"
EOPD_CKPT_DIR="${EOPD_CKPT_DIR:-/EOPD-checkpoints/${EOPD_EXPERIMENT_NAME}_save_step_${EOPD_SAVE_FREQ}}"
EOPD_STEP="${EOPD_STEP:-109}"
EVAL_LCB_N="${LCB_N:-8}"
export G_OPD_CKPT_DIR="$EOPD_CKPT_DIR"
export FSDP_CKPT_DIR="${FSDP_CKPT_DIR:-$EOPD_CKPT_DIR/global_step_${EOPD_STEP}/actor}"
export MODEL_NAME="${MODEL_NAME:-${EOPD_EXPERIMENT_NAME}_save_step_${EOPD_SAVE_FREQ}_global_step_${EOPD_STEP}_code_n${EVAL_LCB_N}}"

echo "[code-eval-eopd] experiment=$EOPD_EXPERIMENT_NAME"
echo "[code-eval-eopd] ckpt_dir=$EOPD_CKPT_DIR"
echo "[code-eval-eopd] step=$EOPD_STEP"
echo "[code-eval-eopd] fsdp_ckpt_dir=$FSDP_CKPT_DIR"
exec bash "$SCRIPT_DIR/run_eval_code.sh" "$@"
