#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STUDENT_MODEL="${STUDENT_MODEL:-/workspace/models/Qwen3-1.7B}"
export STUDENT_MODEL_REPO="${STUDENT_MODEL_REPO:-Qwen/Qwen3-1.7B}"
export TEACHER_MODEL="${TEACHER_MODEL:-/workspace/models/Qwen3-4B-Instruct-2507}"
export TEACHER_MODEL_REPO="${TEACHER_MODEL_REPO:-Qwen/Qwen3-4B-Instruct-2507}"
# shellcheck source=/dev/null
. "$SCRIPT_DIR/run_qwen3-code-common.sh"

export TA_OPD_METHOD="${TA_OPD_METHOD:-teachability}"
export TA_OPD_RATIO="${TA_OPD_RATIO:-0.1}"
export TA_OPD_TOPK="${TA_OPD_TOPK:-16}"
export TA_OPD_SEED="${TA_OPD_SEED:-42}"
export TA_OPD_EXPERIMENT_NAME="${TA_OPD_EXPERIMENT_NAME:-qwen3_1_7b_ta_opd_${TA_OPD_METHOD}_ratio${TA_OPD_RATIO}_k${TA_OPD_TOPK}_seed${TA_OPD_SEED}_teacher_qwen3_4b_instruct_2507_code}"
export TA_OPD_SAVE_FREQ="${TA_OPD_SAVE_FREQ:-$CODE_SAVE_FREQ}"
export TA_OPD_CKPT_DIR="${TA_OPD_CKPT_DIR:-/TA_OPD-checkpoints/${TA_OPD_EXPERIMENT_NAME}_save_step_${TA_OPD_SAVE_FREQ}}"
export TA_OPD_RESUME_MODE="${TA_OPD_RESUME_MODE:-disable}"

prepare_code_training_inputs

exec bash "$SCRIPT_DIR/run_qwen3-1.7b-ta-opd.sh" \
    trainer.total_training_steps="$CODE_TOTAL_STEPS" \
    trainer.save_freq="$TA_OPD_SAVE_FREQ" \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    actor_rollout_ref.actor.checkpoint.save_contents='["model"]' \
    "$@"
