#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TROPD_BASELINE_PATCH_DIR="${TROPD_BASELINE_PATCH_DIR:-$SCRIPT_DIR/tropd_baseline_patch}"
TROPD_BASELINE_VERBOSE="${TROPD_BASELINE_VERBOSE:-1}"
TROPD_TRUST_REGION_THRESHOLD="${TROPD_TRUST_REGION_THRESHOLD:-0.5}"
TROPD_OUTLIER_MODE="${TROPD_OUTLIER_MODE:-passthrough}"
TROPD_OFF_POLICY_GUIDANCE_ENABLE="${TROPD_OFF_POLICY_GUIDANCE_ENABLE:-0}"
TROPD_BASELINE_VERL_DIR="${TROPD_BASELINE_VERL_DIR:-/workspace/opd1/verl}"

export TROPD_BASELINE_ENABLE=1
export TROPD_BASELINE_PATCH_DIR
export TROPD_BASELINE_VERBOSE
export TROPD_TRUST_REGION_THRESHOLD
export TROPD_OUTLIER_MODE
export TROPD_OFF_POLICY_GUIDANCE_ENABLE
export TROPD_BASELINE_VERL_DIR
export PYTHONPATH="$TROPD_BASELINE_PATCH_DIR:$TROPD_BASELINE_VERL_DIR:${PYTHONPATH:-}"

G_OPD_EXPERIMENT_NAME="${G_OPD_EXPERIMENT_NAME:-146_qwen3_1.7b_teacher_qwen3_4b_tropd_baseline_scaffold}"
export G_OPD_EXPERIMENT_NAME

echo "[tropd-baseline] experiment=$G_OPD_EXPERIMENT_NAME"
echo "[tropd-baseline] patch_dir=$TROPD_BASELINE_PATCH_DIR"
echo "[tropd-baseline] current implementation=vanilla OPD passthrough"

exec "$SCRIPT_DIR/run_qwen3-4b-g-opd.sh" \
        +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="$PYTHONPATH" \
        +ray_kwargs.ray_init.runtime_env.env_vars.TROPD_BASELINE_ENABLE="$TROPD_BASELINE_ENABLE" \
        +ray_kwargs.ray_init.runtime_env.env_vars.TROPD_BASELINE_PATCH_DIR="$TROPD_BASELINE_PATCH_DIR" \
        +ray_kwargs.ray_init.runtime_env.env_vars.TROPD_BASELINE_VERBOSE="$TROPD_BASELINE_VERBOSE" \
        +ray_kwargs.ray_init.runtime_env.env_vars.TROPD_TRUST_REGION_THRESHOLD="$TROPD_TRUST_REGION_THRESHOLD" \
        +ray_kwargs.ray_init.runtime_env.env_vars.TROPD_OUTLIER_MODE="$TROPD_OUTLIER_MODE" \
        +ray_kwargs.ray_init.runtime_env.env_vars.TROPD_OFF_POLICY_GUIDANCE_ENABLE="$TROPD_OFF_POLICY_GUIDANCE_ENABLE" \
        +ray_kwargs.ray_init.runtime_env.env_vars.TROPD_BASELINE_VERL_DIR="$TROPD_BASELINE_VERL_DIR" \
        "$@"
