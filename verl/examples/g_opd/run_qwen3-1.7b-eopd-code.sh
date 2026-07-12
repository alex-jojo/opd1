#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STUDENT_MODEL="${STUDENT_MODEL:-/workspace/models/Qwen3-1.7B}"
export STUDENT_MODEL_REPO="${STUDENT_MODEL_REPO:-Qwen/Qwen3-1.7B}"
export TEACHER_MODEL="${TEACHER_MODEL:-/workspace/models/Qwen3-4B-Instruct-2507}"
export TEACHER_MODEL_REPO="${TEACHER_MODEL_REPO:-Qwen/Qwen3-4B-Instruct-2507}"
# shellcheck source=/dev/null
. "$SCRIPT_DIR/run_qwen3-code-common.sh"

export EOPD_EXPERIMENT_NAME="${EOPD_EXPERIMENT_NAME:-eopd_217_qwen3_1_7b_eopd_teacher_qwen3_4b_instruct_2507_code}"
export EOPD_SAVE_FREQ="${EOPD_SAVE_FREQ:-$CODE_SAVE_FREQ}"
export EOPD_CKPT_DIR="${EOPD_CKPT_DIR:-/EOPD-checkpoints/${EOPD_EXPERIMENT_NAME}_save_step_${EOPD_SAVE_FREQ}}"
export EOPD_RESUME_MODE="${EOPD_RESUME_MODE:-disable}"

prepare_code_training_inputs

exec bash "$SCRIPT_DIR/run_qwen3-1.7b-eopd.sh" \
    trainer.total_training_steps="$CODE_TOTAL_STEPS" \
    trainer.save_freq="$EOPD_SAVE_FREQ" \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    actor_rollout_ref.actor.checkpoint.save_contents='["model"]' \
    "$@"
