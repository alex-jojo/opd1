#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Vanilla OPD should report the task's executable code reward. The shared
# code defaults keep a constant reward for distillation-only baselines, so set
# the vanilla default before sourcing them while still allowing an explicit
# caller override.
export CODE_CONSTANT_REWARD="${CODE_CONSTANT_REWARD:-0}"
# shellcheck source=/dev/null
. "$SCRIPT_DIR/run_qwen3-code-common.sh"

export G_OPD_EXPERIMENT_NAME="${G_OPD_EXPERIMENT_NAME:-146_qwen3_4b_teacher_qwen3_4b_instruct_2507_code_vanilla_opd}"
export G_OPD_SAVE_FREQ="${G_OPD_SAVE_FREQ:-$CODE_SAVE_FREQ}"
export G_OPD_CKPT_DIR="${G_OPD_CKPT_DIR:-/G-OPD-checkpoints/${G_OPD_EXPERIMENT_NAME}_save_step_${G_OPD_SAVE_FREQ}}"
export G_OPD_RESUME_MODE="${G_OPD_RESUME_MODE:-disable}"

prepare_code_training_inputs

exec bash "$SCRIPT_DIR/run_qwen3-4b-g-opd.sh" \
    trainer.total_training_steps="$CODE_TOTAL_STEPS" \
    trainer.save_freq="$G_OPD_SAVE_FREQ" \
    actor_rollout_ref.actor.checkpoint.save_contents='["model"]' \
    "$@"
