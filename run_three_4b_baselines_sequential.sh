#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$ROOT_DIR/verl/examples/g_opd"
EVAL_DIR="$ROOT_DIR/math_eval"

TOTAL_STEPS="${TOTAL_STEPS:-110}"
SAVE_FREQ="${SAVE_FREQ:-50}"
DATASETS="${DATASETS:-aime24 aime25 aime26 hmmt26 amc23 math500}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/baseline_runs/$RUN_ID}"
DRY_RUN="${DRY_RUN:-0}"
KEEP_CKPT="${KEEP_CKPT:-0}"
ALLOW_EXISTING_CKPT="${ALLOW_EXISTING_CKPT:-0}"
EVAL_SKIP_DOWNLOAD="${EVAL_SKIP_DOWNLOAD:-1}"

EOPD_EXPERIMENT="eopd_217_qwen3_4b_eopd_teacher_qwen3_4b_non_thinking_rl_math"
TA_OPD_EXPERIMENT="qwen3_4b_ta_opd_teachability_ratio0.1_k16_seed42_teacher_qwen3_4b_non_thinking_rl_math"
EXOPD_EXPERIMENT="146_qwen3_4b_teacher_qwen3_4b_single_teacher_exopd_lambda_1p25"

EOPD_CKPT="/EOPD-checkpoints/${EOPD_EXPERIMENT}_save_step_${SAVE_FREQ}"
TA_OPD_CKPT="/TA_OPD-checkpoints/${TA_OPD_EXPERIMENT}_save_step_${SAVE_FREQ}"
EXOPD_CKPT="/G-OPD-checkpoints/${EXOPD_EXPERIMENT}_save_step_${SAVE_FREQ}"

STATUS_FILE="$RUN_DIR/status.tsv"
COMBINED_SUMMARY="$RUN_DIR/all_results.csv"
CURRENT_BASELINE="pipeline"
CURRENT_STAGE="setup"

mkdir -p "$RUN_DIR"
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

validate_eval_results() {
    local summary_file="$1"
    local expected_count=0
    local dataset

    for dataset in $DATASETS; do
        expected_count=$((expected_count + 1))
    done

    awk -F, -v expected="$expected_count" '
        NR == 1 { next }
        {
            count += 1
            if ($8 != "ok") {
                bad = 1
            }
        }
        END {
            if (count != expected || bad) {
                exit 1
            }
        }
    ' "$summary_file"
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

run_eopd() {
    local baseline="eopd"
    local result_dir="$RUN_DIR/$baseline"
    local output_dir="$result_dir/eval_outputs"

    mkdir -p "$result_dir"
    CURRENT_BASELINE="$baseline"
    CURRENT_STAGE="training"
    record_status "$baseline" train started "$EOPD_CKPT"

    run_logged "$result_dir/train.log" env \
        STUDENT_MODEL=/workspace/models/Qwen3-4B \
        EOPD_EXPERIMENT_NAME="$EOPD_EXPERIMENT" \
        EOPD_SAVE_FREQ="$SAVE_FREQ" \
        EOPD_CKPT_DIR="$EOPD_CKPT" \
        EOPD_RESUME_MODE=disable \
        bash "$TRAIN_DIR/run_qwen3-1.7b-eopd.sh" \
        trainer.total_training_steps="$TOTAL_STEPS" \
        actor_rollout_ref.actor.checkpoint.save_contents='["model"]'

    require_step_checkpoint "$EOPD_CKPT"
    record_status "$baseline" train completed "$EOPD_CKPT/global_step_${TOTAL_STEPS}/actor"
    stop_ray

    CURRENT_STAGE="evaluation"
    record_status "$baseline" eval started "$output_dir"
    run_logged "$result_dir/eval.log" env \
        -u FSDP_CKPT_DIR -u MODEL_PATH -u MODEL -u MODEL_NAME \
        OUTPUT_DIR="$output_dir" \
        SKIP_DOWNLOAD="$EVAL_SKIP_DOWNLOAD" \
        DATASETS="$DATASETS" \
        EOPD_EXPERIMENT_NAME="$EOPD_EXPERIMENT" \
        EOPD_SAVE_FREQ="$SAVE_FREQ" \
        EOPD_CKPT_DIR="$EOPD_CKPT" \
        EOPD_STEP="$TOTAL_STEPS" \
        bash "$EVAL_DIR/run_eval_qwen3-1.7b-eopd.sh"

    if [ "$DRY_RUN" != "1" ]; then
        validate_eval_results "$output_dir/summary.csv"
        append_summary "$output_dir/summary.csv"
    fi
    record_status "$baseline" eval completed "$output_dir/summary.csv"

    CURRENT_STAGE="cleanup"
    delete_checkpoint "$baseline" "$EOPD_CKPT" /EOPD-checkpoints
}

run_ta_opd() {
    local baseline="ta_opd"
    local result_dir="$RUN_DIR/$baseline"
    local output_dir="$result_dir/eval_outputs"

    mkdir -p "$result_dir"
    CURRENT_BASELINE="$baseline"
    CURRENT_STAGE="training"
    record_status "$baseline" train started "$TA_OPD_CKPT"

    run_logged "$result_dir/train.log" env \
        STUDENT_MODEL=/workspace/models/Qwen3-4B \
        TA_OPD_EXPERIMENT_NAME="$TA_OPD_EXPERIMENT" \
        TA_OPD_SAVE_FREQ="$SAVE_FREQ" \
        TA_OPD_CKPT_DIR="$TA_OPD_CKPT" \
        TA_OPD_RESUME_MODE=disable \
        bash "$TRAIN_DIR/run_qwen3-1.7b-ta-opd.sh" \
        trainer.total_training_steps="$TOTAL_STEPS" \
        actor_rollout_ref.actor.checkpoint.save_contents='["model"]'

    require_step_checkpoint "$TA_OPD_CKPT"
    record_status "$baseline" train completed "$TA_OPD_CKPT/global_step_${TOTAL_STEPS}/actor"
    stop_ray

    CURRENT_STAGE="evaluation"
    record_status "$baseline" eval started "$output_dir"
    run_logged "$result_dir/eval.log" env \
        -u FSDP_CKPT_DIR -u MODEL_PATH -u MODEL -u MODEL_NAME \
        OUTPUT_DIR="$output_dir" \
        SKIP_DOWNLOAD="$EVAL_SKIP_DOWNLOAD" \
        DATASETS="$DATASETS" \
        TA_OPD_EXPERIMENT_NAME="$TA_OPD_EXPERIMENT" \
        TA_OPD_SAVE_FREQ="$SAVE_FREQ" \
        TA_OPD_CKPT_DIR="$TA_OPD_CKPT" \
        TA_OPD_STEP="$TOTAL_STEPS" \
        bash "$EVAL_DIR/run_eval_qwen3-1.7b-ta-opd.sh"

    if [ "$DRY_RUN" != "1" ]; then
        validate_eval_results "$output_dir/summary.csv"
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

    mkdir -p "$result_dir"
    CURRENT_BASELINE="$baseline"
    CURRENT_STAGE="training"
    record_status "$baseline" train started "$EXOPD_CKPT"

    run_logged "$result_dir/train.log" env \
        STUDENT_MODEL=/workspace/models/Qwen3-4B \
        EXOPD_LAMBDA=1.25 \
        G_OPD_EXPERIMENT_NAME="$EXOPD_EXPERIMENT" \
        G_OPD_SAVE_FREQ="$SAVE_FREQ" \
        G_OPD_CKPT_DIR="$EXOPD_CKPT" \
        G_OPD_RESUME_MODE=disable \
        bash "$TRAIN_DIR/run_qwen3-4b-single-teacher-exopd.sh" \
        trainer.total_training_steps="$TOTAL_STEPS" \
        actor_rollout_ref.actor.checkpoint.save_contents='["model"]'

    require_step_checkpoint "$EXOPD_CKPT"
    record_status "$baseline" train completed "$EXOPD_CKPT/global_step_${TOTAL_STEPS}/actor"
    stop_ray

    CURRENT_STAGE="evaluation"
    record_status "$baseline" eval started "$output_dir"
    run_logged "$result_dir/eval.log" env \
        -u FSDP_CKPT_DIR -u MODEL_PATH -u MODEL -u MODEL_NAME \
        OUTPUT_DIR="$output_dir" \
        SKIP_DOWNLOAD="$EVAL_SKIP_DOWNLOAD" \
        DATASETS="$DATASETS" \
        EXOPD_LAMBDA=1.25 \
        EXOPD_STEP="$TOTAL_STEPS" \
        G_OPD_EXPERIMENT_NAME="$EXOPD_EXPERIMENT" \
        G_OPD_SAVE_FREQ="$SAVE_FREQ" \
        G_OPD_CKPT_DIR="$EXOPD_CKPT" \
        bash "$EVAL_DIR/run_eval_qwen3-4b-single-teacher-exopd.sh"

    if [ "$DRY_RUN" != "1" ]; then
        validate_eval_results "$output_dir/summary.csv"
        append_summary "$output_dir/summary.csv"
    fi
    record_status "$baseline" eval completed "$output_dir/summary.csv"

    CURRENT_STAGE="cleanup"
    delete_checkpoint "$baseline" "$EXOPD_CKPT" /G-OPD-checkpoints
}

for required_script in \
    "$TRAIN_DIR/run_qwen3-1.7b-eopd.sh" \
    "$TRAIN_DIR/run_qwen3-1.7b-ta-opd.sh" \
    "$TRAIN_DIR/run_qwen3-4b-single-teacher-exopd.sh" \
    "$EVAL_DIR/run_eval_qwen3-1.7b-eopd.sh" \
    "$EVAL_DIR/run_eval_qwen3-1.7b-ta-opd.sh" \
    "$EVAL_DIR/run_eval_qwen3-4b-single-teacher-exopd.sh"; do
    if [ ! -f "$required_script" ]; then
        echo "ERROR: required script not found: $required_script"
        exit 1
    fi
done

if [ "$EVAL_SKIP_DOWNLOAD" = "1" ]; then
    for dataset in $DATASETS; do
        if [ ! -f "$ROOT_DIR/data/$dataset/test.jsonl" ]; then
            echo "ERROR: dataset missing while EVAL_SKIP_DOWNLOAD=1: $ROOT_DIR/data/$dataset/test.jsonl"
            exit 1
        fi
    done
fi

validate_new_ckpt_dir "$EOPD_CKPT" /EOPD-checkpoints
validate_new_ckpt_dir "$TA_OPD_CKPT" /TA_OPD-checkpoints
validate_new_ckpt_dir "$EXOPD_CKPT" /G-OPD-checkpoints

echo "[run] results: $RUN_DIR"
echo "[run] total steps per baseline: $TOTAL_STEPS"
echo "[run] datasets: $DATASETS"
echo "[run] checkpoint cleanup: $([ "$KEEP_CKPT" = "1" ] && echo disabled || echo enabled)"

run_eopd
run_ta_opd
run_exopd

CURRENT_BASELINE="pipeline"
CURRENT_STAGE="completed"
record_status all pipeline completed "$COMBINED_SUMMARY"
echo "[done] all baselines completed"
echo "[done] combined summary: $COMBINED_SUMMARY"
