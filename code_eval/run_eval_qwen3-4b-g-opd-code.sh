#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export G_OPD_EXPERIMENT_NAME="${G_OPD_EXPERIMENT_NAME:-146_qwen3_4b_teacher_qwen3_4b_instruct_2507_code_vanilla_opd}"
export G_OPD_SAVE_FREQ="${G_OPD_SAVE_FREQ:-109}"
export G_OPD_CKPT_DIR="${G_OPD_CKPT_DIR:-/G-OPD-checkpoints/${G_OPD_EXPERIMENT_NAME}_save_step_${G_OPD_SAVE_FREQ}}"
G_OPD_STEP="${G_OPD_STEP:-109}"
EVAL_LCB_N="${LCB_N:-8}"
export FSDP_CKPT_DIR="${FSDP_CKPT_DIR:-$G_OPD_CKPT_DIR/global_step_${G_OPD_STEP}/actor}"
export MODEL_NAME="${MODEL_NAME:-${G_OPD_EXPERIMENT_NAME}_save_step_${G_OPD_SAVE_FREQ}_global_step_${G_OPD_STEP}_code_n${EVAL_LCB_N}}"

echo "[code-eval-g-opd] experiment=$G_OPD_EXPERIMENT_NAME"
echo "[code-eval-g-opd] ckpt_dir=$G_OPD_CKPT_DIR"
echo "[code-eval-g-opd] step=$G_OPD_STEP"
echo "[code-eval-g-opd] fsdp_ckpt_dir=$FSDP_CKPT_DIR"
exec bash "$SCRIPT_DIR/run_eval_code.sh" "$@"
