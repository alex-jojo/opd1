#!/usr/bin/env bash
set -Eeuo pipefail

# Qwen3-4B student entry point for the sequential code-baseline pipeline.
# Keep this wrapper beside the shared 1.7B runner in code_sequential/.
# The shared runner owns training, evaluation, checkpoint validation, and cleanup;
# this wrapper only selects the student model and collision-free experiment names.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export STUDENT_MODEL_REPO="${STUDENT_MODEL_REPO:-Qwen/Qwen3-4B}"
export STUDENT_MODEL="${STUDENT_MODEL:-/workspace/models/Qwen3-4B}"

export EOPD_EXPERIMENT="${EOPD_EXPERIMENT:-eopd_217_qwen3_4b_eopd_teacher_qwen3_4b_code_step300}"
export OPD_EXPERIMENT="${OPD_EXPERIMENT:-146_qwen3_4b_teacher_qwen3_4b_code_step300_vanilla_opd}"
export TA_OPD_EXPERIMENT="${TA_OPD_EXPERIMENT:-qwen3_4b_ta_opd_teachability_ratio0.1_k16_seed42_teacher_qwen3_4b_code_step300}"
export EXOPD_EXPERIMENT="${EXOPD_EXPERIMENT:-146_qwen3_4b_teacher_qwen3_4b_code_step300_single_teacher_exopd_lambda_1p25}"

exec bash "$SCRIPT_DIR/run_three_1.7_code_baselines_sequential_qwen3_4b.sh" "$@"
