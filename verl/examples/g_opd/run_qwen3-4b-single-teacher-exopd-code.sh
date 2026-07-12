#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
. "$SCRIPT_DIR/run_qwen3-code-common.sh"

EXOPD_LAMBDA="${EXOPD_LAMBDA:-1.25}"
EXOPD_LAMBDA_TAG="${EXOPD_LAMBDA//./p}"
export G_OPD_EXPERIMENT_NAME="${G_OPD_EXPERIMENT_NAME:-146_qwen3_4b_teacher_qwen3_4b_instruct_2507_code_single_teacher_exopd_lambda_${EXOPD_LAMBDA_TAG}}"
export G_OPD_SAVE_FREQ="${G_OPD_SAVE_FREQ:-$CODE_SAVE_FREQ}"
export G_OPD_CKPT_DIR="${G_OPD_CKPT_DIR:-/G-OPD-checkpoints/${G_OPD_EXPERIMENT_NAME}_save_step_${G_OPD_SAVE_FREQ}}"
export G_OPD_RESUME_MODE="${G_OPD_RESUME_MODE:-disable}"

prepare_code_training_inputs

exec bash "$SCRIPT_DIR/run_qwen3-4b-single-teacher-exopd.sh" \
    trainer.total_training_steps="$CODE_TOTAL_STEPS" \
    trainer.save_freq="$G_OPD_SAVE_FREQ" \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    actor_rollout_ref.actor.checkpoint.save_contents='["model"]' \
    "$@"
