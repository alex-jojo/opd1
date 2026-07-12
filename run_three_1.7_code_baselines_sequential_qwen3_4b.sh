#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$ROOT_DIR/verl/examples/g_opd"
EVAL_DIR="$ROOT_DIR/code_eval"
if [ -d /venv/verl/bin ]; then
    export PATH="/venv/verl/bin:$PATH"
fi
if [ -d /venv/main/bin ]; then
    export PATH="$PATH:/venv/main/bin"
fi

TOTAL_STEPS="${TOTAL_STEPS:-109}"
SAVE_FREQ="${SAVE_FREQ:-109}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/baseline_runs/$RUN_ID}"
DRY_RUN="${DRY_RUN:-0}"
KEEP_CKPT="${KEEP_CKPT:-0}"
ALLOW_EXISTING_CKPT="${ALLOW_EXISTING_CKPT:-0}"

LCB_RELEASE="${LCB_RELEASE:-v5}"
LCB_N="${LCB_N:-8}"
RUN_LCB="${RUN_LCB:-1}"
RUN_EVALPLUS="${RUN_EVALPLUS:-1}"
RUN_HUMANEVAL="${RUN_HUMANEVAL:-1}"
RUN_MBPP="${RUN_MBPP:-1}"
EVALPLUS_ALLOW_OVERWRITE="${EVALPLUS_ALLOW_OVERWRITE:-1}"
LCB_AUTO_DOWNLOAD_DATA="${LCB_AUTO_DOWNLOAD_DATA:-1}"

TEACHER_MODEL_REPO="${TEACHER_MODEL_REPO:-Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step300}"
TEACHER_MODEL="${TEACHER_MODEL:-/workspace/models/Qwen3-4B-Non-Thinking-RL-Code-Step300}"
STUDENT_MODEL_REPO="${STUDENT_MODEL_REPO:-Qwen/Qwen3-1.7B}"
STUDENT_MODEL="${STUDENT_MODEL:-/workspace/models/Qwen3-1.7B}"
TRAIN_DATA_REPO="${TRAIN_DATA_REPO:-Skywork/Skywork-OR1-RL-Data}"
TRAIN_DATA_SPLIT="${TRAIN_DATA_SPLIT:-code}"
TRAIN_SRC="${TRAIN_SRC:-$ROOT_DIR/data/skywork_or1_rl_data_code.parquet}"
TRAIN_FILE="${TRAIN_FILE:-$ROOT_DIR/verl/train_skywork_or1_rl_data_code_verl.parquet}"
CONSTANT_REWARD_FN="${CONSTANT_REWARD_FN:-$TRAIN_DIR/constant_reward.py}"
BASELINE_CONSTANT_REWARD_VALUE="${BASELINE_CONSTANT_REWARD_VALUE:-1.0}"

EOPD_EXPERIMENT="${EOPD_EXPERIMENT:-eopd_217_qwen3_1_7b_eopd_teacher_qwen3_4b_code_step300}"
OPD_EXPERIMENT="${OPD_EXPERIMENT:-146_qwen3_1_7b_teacher_qwen3_4b_code_step300_vanilla_opd}"
TA_OPD_EXPERIMENT="${TA_OPD_EXPERIMENT:-qwen3_1_7b_ta_opd_teachability_ratio0.1_k16_seed42_teacher_qwen3_4b_code_step300}"
EXOPD_EXPERIMENT="${EXOPD_EXPERIMENT:-146_qwen3_1_7b_teacher_qwen3_4b_code_step300_single_teacher_exopd_lambda_1p25}"

EOPD_CKPT="${EOPD_CKPT:-/EOPD-checkpoints/${EOPD_EXPERIMENT}_save_step_${SAVE_FREQ}}"
OPD_CKPT="${OPD_CKPT:-/G-OPD-checkpoints/${OPD_EXPERIMENT}_save_step_${SAVE_FREQ}}"
TA_OPD_CKPT="${TA_OPD_CKPT:-/TA_OPD-checkpoints/${TA_OPD_EXPERIMENT}_save_step_${SAVE_FREQ}}"
EXOPD_CKPT="${EXOPD_CKPT:-/G-OPD-checkpoints/${EXOPD_EXPERIMENT}_save_step_${SAVE_FREQ}}"

STATUS_FILE="$RUN_DIR/status.tsv"
COMBINED_SUMMARY="$RUN_DIR/all_results.csv"
CURRENT_BASELINE="pipeline"
CURRENT_STAGE="setup"

mkdir -p "$RUN_DIR"
if [ ! -f "$CONSTANT_REWARD_FN" ]; then
    echo "ERROR: constant reward function not found: $CONSTANT_REWARD_FN"
    exit 1
fi
printf 'timestamp\tbaseline\tstage\tstatus\tpath\n' > "$STATUS_FILE"

timestamp() {
    date -u +%Y-%m-%dT%H:%M:%SZ
}

record_status() {
    local baseline="$1"
    local stage="$2"
    local status="$3"
    local path="${4:-}"
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$(timestamp)" "$baseline" "$stage" "$status" "$path" >> "$STATUS_FILE"
}

on_error() {
    local exit_code=$?
    trap - ERR
    record_status "$CURRENT_BASELINE" "$CURRENT_STAGE" failed
    echo "ERROR: $CURRENT_BASELINE failed during $CURRENT_STAGE (exit $exit_code)."
    echo "Checkpoint cleanup was not run for the failed baseline."
    exit "$exit_code"
}
trap on_error ERR

run_logged() {
    local log_file="$1"
    shift

    if [ "$DRY_RUN" = "1" ]; then
        printf '[dry-run]'
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi

    "$@" 2>&1 | tee "$log_file"
}

stop_ray() {
    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] ray stop --force"
        return
    fi

    ray stop --force >/dev/null 2>&1 || true
}

validate_new_ckpt_dir() {
    local ckpt_dir="$1"
    local expected_prefix="$2"

    case "$ckpt_dir" in
        "$expected_prefix"/*) ;;
        *)
            echo "ERROR: refusing checkpoint path outside $expected_prefix: $ckpt_dir"
            exit 1
            ;;
    esac

    if [ -e "$ckpt_dir" ] && [ "$ALLOW_EXISTING_CKPT" != "1" ]; then
        echo "ERROR: checkpoint directory already exists: $ckpt_dir"
        echo "Set ALLOW_EXISTING_CKPT=1 only if this run may reuse and later delete it."
        exit 1
    fi
}

require_step_checkpoint() {
    local ckpt_dir="$1"
    local actor_dir="$ckpt_dir/global_step_${TOTAL_STEPS}/actor"

    if [ "$DRY_RUN" = "1" ]; then
        return
    fi

    if [ ! -f "$actor_dir/fsdp_config.json" ]; then
        echo "ERROR: expected step checkpoint is missing: $actor_dir/fsdp_config.json"
        exit 1
    fi
}

validate_code_eval_results() {
    local summary_file="$1"

    python3 - "$summary_file" <<'PYCHECK'
import csv
import sys

summary_file = sys.argv[1]
required = [
    "LiveCodeBenchv5 AVG@8",
    "LiveCodeBenchv5 P@8",
    "LiveCodeBenchv5 maj@8",
    "HumanEval+ AVG@8",
    "HumanEval+ P@8",
    "HumanEval+ maj@8",
    "MBPP AVG@8",
    "MBPP P@8",
    "MBPP maj@8",
]
with open(summary_file, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit("summary has no rows")
for row in rows:
    missing = [key for key in required if row.get(key) in (None, "", "NA")]
    if missing:
        raise SystemExit(f"summary row missing metrics {missing}: {row}")
PYCHECK
}

append_summary() {
    local summary_file="$1"

    if [ ! -f "$COMBINED_SUMMARY" ]; then
        cp "$summary_file" "$COMBINED_SUMMARY"
    else
        tail -n +2 "$summary_file" >> "$COMBINED_SUMMARY"
    fi
}

delete_checkpoint() {
    local baseline="$1"
    local ckpt_dir="$2"
    local expected_prefix="$3"

    if [ "$KEEP_CKPT" = "1" ]; then
        echo "[$baseline] KEEP_CKPT=1, retaining $ckpt_dir"
        record_status "$baseline" cleanup kept "$ckpt_dir"
        return
    fi

    case "$ckpt_dir" in
        "$expected_prefix"/*) ;;
        *)
            echo "ERROR: refusing to delete path outside $expected_prefix: $ckpt_dir"
            exit 1
            ;;
    esac

    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] rm -rf -- $ckpt_dir"
    else
        rm -rf -- "$ckpt_dir"
    fi
    record_status "$baseline" cleanup deleted "$ckpt_dir"
}

eval_common_env() {
    env \
        -u FSDP_CKPT_DIR -u MODEL_PATH -u MODEL -u MODEL_NAME \
        RUN_LCB="$RUN_LCB" \
        RUN_EVALPLUS="$RUN_EVALPLUS" \
        RUN_HUMANEVAL="$RUN_HUMANEVAL" \
        RUN_MBPP="$RUN_MBPP" \
        LCB_RELEASE="$LCB_RELEASE" \
        LCB_N="$LCB_N" \
        LCB_AUTO_DOWNLOAD_DATA="$LCB_AUTO_DOWNLOAD_DATA" \
        EVALPLUS_ALLOW_OVERWRITE="$EVALPLUS_ALLOW_OVERWRITE" \
        "$@"
}


run_eopd() {
    local baseline="eopd"
    local result_dir="$RUN_DIR/$baseline"
    local output_dir="$result_dir/eval_outputs"

    mkdir -p "$result_dir" "$output_dir"
    CURRENT_BASELINE="$baseline"
    CURRENT_STAGE="training"
    record_status "$baseline" train started "$EOPD_CKPT"

    run_logged "$result_dir/train.log" env \
        STUDENT_MODEL="$STUDENT_MODEL" \
        STUDENT_MODEL_REPO="$STUDENT_MODEL_REPO" \
        TEACHER_MODEL="$TEACHER_MODEL" \
        TEACHER_MODEL_REPO="$TEACHER_MODEL_REPO" \
        TRAIN_SRC="$TRAIN_SRC" \
        TRAIN_FILE="$TRAIN_FILE" \
        EOPD_EXPERIMENT_NAME="$EOPD_EXPERIMENT" \
        EOPD_SAVE_FREQ="$SAVE_FREQ" \
        EOPD_CKPT_DIR="$EOPD_CKPT" \
        EOPD_RESUME_MODE=disable \
        bash "$TRAIN_DIR/run_qwen3-1.7b-eopd-code.sh" \
        trainer.total_training_steps="$TOTAL_STEPS"

    require_step_checkpoint "$EOPD_CKPT"
    record_status "$baseline" train completed "$EOPD_CKPT/global_step_${TOTAL_STEPS}/actor"
    stop_ray

    CURRENT_STAGE="evaluation"
    record_status "$baseline" eval started "$output_dir"
    run_logged "$result_dir/eval.log" eval_common_env \
        OUTPUT_DIR="$output_dir" \
        SUMMARY_FILE="$output_dir/summary.csv" \
        EOPD_EXPERIMENT_NAME="$EOPD_EXPERIMENT" \
        EOPD_SAVE_FREQ="$SAVE_FREQ" \
        EOPD_CKPT_DIR="$EOPD_CKPT" \
        EOPD_STEP="$TOTAL_STEPS" \
        bash "$EVAL_DIR/run_eval_qwen3-1.7b-eopd-code.sh"

    if [ "$DRY_RUN" != "1" ]; then
        validate_code_eval_results "$output_dir/summary.csv"
        append_summary "$output_dir/summary.csv"
    fi
    record_status "$baseline" eval completed "$output_dir/summary.csv"

    CURRENT_STAGE="cleanup"
    delete_checkpoint "$baseline" "$EOPD_CKPT" /EOPD-checkpoints
}

run_opd() {
    local baseline="opd"
    local result_dir="$RUN_DIR/$baseline"
    local output_dir="$result_dir/eval_outputs"

    mkdir -p "$result_dir" "$output_dir"
    CURRENT_BASELINE="$baseline"
    CURRENT_STAGE="training"
    record_status "$baseline" train started "$OPD_CKPT"

    run_logged "$result_dir/train.log" env \
        STUDENT_MODEL="$STUDENT_MODEL" \
        STUDENT_MODEL_REPO="$STUDENT_MODEL_REPO" \
        TEACHER_MODEL="$TEACHER_MODEL" \
        TEACHER_MODEL_REPO="$TEACHER_MODEL_REPO" \
        TRAIN_SRC="$TRAIN_SRC" \
        TRAIN_FILE="$TRAIN_FILE" \
        G_OPD_EXPERIMENT_NAME="$OPD_EXPERIMENT" \
        G_OPD_SAVE_FREQ="$SAVE_FREQ" \
        G_OPD_CKPT_DIR="$OPD_CKPT" \
        G_OPD_RESUME_MODE=disable \
        bash "$TRAIN_DIR/run_qwen3-4b-g-opd-code.sh" \
        trainer.total_training_steps="$TOTAL_STEPS" \
        actor_rollout_ref.actor.checkpoint.save_contents='["model"]'

    require_step_checkpoint "$OPD_CKPT"
    record_status "$baseline" train completed "$OPD_CKPT/global_step_${TOTAL_STEPS}/actor"
    stop_ray

    CURRENT_STAGE="evaluation"
    record_status "$baseline" eval started "$output_dir"
    run_logged "$result_dir/eval.log" eval_common_env \
        OUTPUT_DIR="$output_dir" \
        SUMMARY_FILE="$output_dir/summary.csv" \
        G_OPD_EXPERIMENT_NAME="$OPD_EXPERIMENT" \
        G_OPD_SAVE_FREQ="$SAVE_FREQ" \
        G_OPD_CKPT_DIR="$OPD_CKPT" \
        G_OPD_STEP="$TOTAL_STEPS" \
        bash "$EVAL_DIR/run_eval_qwen3-4b-g-opd-code.sh"

    if [ "$DRY_RUN" != "1" ]; then
        validate_code_eval_results "$output_dir/summary.csv"
        append_summary "$output_dir/summary.csv"
    fi
    record_status "$baseline" eval completed "$output_dir/summary.csv"

    CURRENT_STAGE="cleanup"
    delete_checkpoint "$baseline" "$OPD_CKPT" /G-OPD-checkpoints
}

run_ta_opd() {
    local baseline="ta_opd"
    local result_dir="$RUN_DIR/$baseline"
    local output_dir="$result_dir/eval_outputs"

    mkdir -p "$result_dir" "$output_dir"
    CURRENT_BASELINE="$baseline"
    CURRENT_STAGE="training"
    record_status "$baseline" train started "$TA_OPD_CKPT"

    run_logged "$result_dir/train.log" env \
        STUDENT_MODEL="$STUDENT_MODEL" \
        STUDENT_MODEL_REPO="$STUDENT_MODEL_REPO" \
        TEACHER_MODEL="$TEACHER_MODEL" \
        TEACHER_MODEL_REPO="$TEACHER_MODEL_REPO" \
        TRAIN_SRC="$TRAIN_SRC" \
        TRAIN_FILE="$TRAIN_FILE" \
        TA_OPD_EXPERIMENT_NAME="$TA_OPD_EXPERIMENT" \
        TA_OPD_SAVE_FREQ="$SAVE_FREQ" \
        TA_OPD_CKPT_DIR="$TA_OPD_CKPT" \
        TA_OPD_RESUME_MODE=disable \
        CODE_CONSTANT_REWARD=1 \
        CODE_CONSTANT_REWARD_VALUE="$BASELINE_CONSTANT_REWARD_VALUE" \
        bash "$TRAIN_DIR/run_qwen3-1.7b-ta-opd-code.sh" \
        trainer.total_training_steps="$TOTAL_STEPS" \
        custom_reward_function.path="$CONSTANT_REWARD_FN" \
        custom_reward_function.name=compute_score \
        actor_rollout_ref.actor.checkpoint.save_contents='["model"]'

    require_step_checkpoint "$TA_OPD_CKPT"
    record_status "$baseline" train completed "$TA_OPD_CKPT/global_step_${TOTAL_STEPS}/actor"
    stop_ray

    CURRENT_STAGE="evaluation"
    record_status "$baseline" eval started "$output_dir"
    run_logged "$result_dir/eval.log" eval_common_env \
        OUTPUT_DIR="$output_dir" \
        SUMMARY_FILE="$output_dir/summary.csv" \
        TA_OPD_EXPERIMENT_NAME="$TA_OPD_EXPERIMENT" \
        TA_OPD_SAVE_FREQ="$SAVE_FREQ" \
        TA_OPD_CKPT_DIR="$TA_OPD_CKPT" \
        TA_OPD_STEP="$TOTAL_STEPS" \
        bash "$EVAL_DIR/run_eval_qwen3-1.7b-ta-opd-code.sh"

    if [ "$DRY_RUN" != "1" ]; then
        validate_code_eval_results "$output_dir/summary.csv"
        append_summary "$output_dir/summary.csv"
    fi
    record_status "$baseline" eval completed "$output_dir/summary.csv"

    CURRENT_STAGE="cleanup"
    delete_checkpoint "$baseline" "$TA_OPD_CKPT" /TA_OPD-checkpoints
}

run_exopd() {
    local baseline="single_teacher_exopd"
    local result_dir="$RUN_DIR/$baseline"
    local output_dir="$result_dir/eval_outputs"

    mkdir -p "$result_dir" "$output_dir"
    CURRENT_BASELINE="$baseline"
    CURRENT_STAGE="training"
    record_status "$baseline" train started "$EXOPD_CKPT"

    run_logged "$result_dir/train.log" env \
        STUDENT_MODEL="$STUDENT_MODEL" \
        STUDENT_MODEL_REPO="$STUDENT_MODEL_REPO" \
        TEACHER_MODEL="$TEACHER_MODEL" \
        TEACHER_MODEL_REPO="$TEACHER_MODEL_REPO" \
        TRAIN_SRC="$TRAIN_SRC" \
        TRAIN_FILE="$TRAIN_FILE" \
        EXOPD_LAMBDA=1.25 \
        G_OPD_EXPERIMENT_NAME="$EXOPD_EXPERIMENT" \
        G_OPD_SAVE_FREQ="$SAVE_FREQ" \
        G_OPD_CKPT_DIR="$EXOPD_CKPT" \
        G_OPD_RESUME_MODE=disable \
        CODE_CONSTANT_REWARD=1 \
        CODE_CONSTANT_REWARD_VALUE="$BASELINE_CONSTANT_REWARD_VALUE" \
        bash "$TRAIN_DIR/run_qwen3-4b-single-teacher-exopd-code.sh" \
        trainer.total_training_steps="$TOTAL_STEPS" \
        custom_reward_function.path="$CONSTANT_REWARD_FN" \
        custom_reward_function.name=compute_score \
        actor_rollout_ref.actor.checkpoint.save_contents='["model"]'

    require_step_checkpoint "$EXOPD_CKPT"
    record_status "$baseline" train completed "$EXOPD_CKPT/global_step_${TOTAL_STEPS}/actor"
    stop_ray

    CURRENT_STAGE="evaluation"
    record_status "$baseline" eval started "$output_dir"
    run_logged "$result_dir/eval.log" eval_common_env \
        OUTPUT_DIR="$output_dir" \
        SUMMARY_FILE="$output_dir/summary.csv" \
        EXOPD_LAMBDA=1.25 \
        EXOPD_STEP="$TOTAL_STEPS" \
        G_OPD_EXPERIMENT_NAME="$EXOPD_EXPERIMENT" \
        G_OPD_SAVE_FREQ="$SAVE_FREQ" \
        G_OPD_CKPT_DIR="$EXOPD_CKPT" \
        bash "$EVAL_DIR/run_eval_qwen3-4b-single-teacher-exopd-code.sh"

    if [ "$DRY_RUN" != "1" ]; then
        validate_code_eval_results "$output_dir/summary.csv"
        append_summary "$output_dir/summary.csv"
    fi
    record_status "$baseline" eval completed "$output_dir/summary.csv"

    CURRENT_STAGE="cleanup"
    delete_checkpoint "$baseline" "$EXOPD_CKPT" /G-OPD-checkpoints
}

for required_script in \
    "$TRAIN_DIR/run_qwen3-1.7b-eopd-code.sh" \
    "$TRAIN_DIR/run_qwen3-4b-g-opd-code.sh" \
    "$TRAIN_DIR/run_qwen3-1.7b-ta-opd-code.sh" \
    "$TRAIN_DIR/run_qwen3-4b-single-teacher-exopd-code.sh" \
    "$EVAL_DIR/run_eval_qwen3-1.7b-eopd-code.sh" \
    "$EVAL_DIR/run_eval_qwen3-4b-g-opd-code.sh" \
    "$EVAL_DIR/run_eval_qwen3-1.7b-ta-opd-code.sh" \
    "$EVAL_DIR/run_eval_qwen3-4b-single-teacher-exopd-code.sh"; do
    if [ ! -f "$required_script" ]; then
        echo "ERROR: required script not found: $required_script"
        exit 1
    fi
done

validate_new_ckpt_dir "$EOPD_CKPT" /EOPD-checkpoints
validate_new_ckpt_dir "$OPD_CKPT" /G-OPD-checkpoints
validate_new_ckpt_dir "$TA_OPD_CKPT" /TA_OPD-checkpoints
validate_new_ckpt_dir "$EXOPD_CKPT" /G-OPD-checkpoints

echo "[run] results: $RUN_DIR"
echo "[run] baselines: eopd ta_opd single_teacher_exopd opd"
echo "[run] total steps per baseline: $TOTAL_STEPS"
echo "[run] train data repo: $TRAIN_DATA_REPO split=$TRAIN_DATA_SPLIT"
echo "[run] train src: $TRAIN_SRC"
echo "[run] train file: $TRAIN_FILE"
echo "[run] student model: $STUDENT_MODEL_REPO -> $STUDENT_MODEL"
echo "[run] teacher model: $TEACHER_MODEL_REPO -> $TEACHER_MODEL"
echo "[run] eval: LCB_RELEASE=$LCB_RELEASE LCB_N=$LCB_N RUN_LCB=$RUN_LCB RUN_EVALPLUS=$RUN_EVALPLUS RUN_HUMANEVAL=$RUN_HUMANEVAL RUN_MBPP=$RUN_MBPP"
echo "[run] checkpoint cleanup: $([ "$KEEP_CKPT" = "1" ] && echo disabled || echo enabled)"

run_eopd
run_ta_opd
run_exopd
run_opd

CURRENT_BASELINE="pipeline"
CURRENT_STAGE="completed"
record_status all pipeline completed "$COMBINED_SUMMARY"
echo "[done] all baselines completed"
echo "[done] combined summary: $COMBINED_SUMMARY"
