#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_EVAL_DIR="$PROJECT_ROOT/code_eval"
OUTPUT_DIR="${OUTPUT_DIR:-$CODE_EVAL_DIR/results}"
mkdir -p "$OUTPUT_DIR"

if [ -d /venv/verl/bin ]; then
    export PATH="/venv/verl/bin:$PATH"
fi

if [ -z "${FSDP_CKPT_DIR:-}" ]; then
    if [ -z "${G_OPD_CKPT_DIR:-}" ]; then
        echo "ERROR: set FSDP_CKPT_DIR or G_OPD_CKPT_DIR before running code eval"
        exit 1
    fi
    LATEST_STEP_DIR="$(
        find "$G_OPD_CKPT_DIR" -maxdepth 3 -type f -path '*/actor/model_world_size_*_rank_0.pt' 2>/dev/null \
            | sed 's#/actor/model_world_size_[^/]*_rank_0\.pt$##' \
            | awk -F'global_step_' '/global_step_[0-9]+$/ {print $2 "\t" $0}' \
            | sort -n \
            | tail -1 \
            | cut -f2-
    )"
    if [ -z "$LATEST_STEP_DIR" ]; then
        echo "ERROR: no actor checkpoints found under: $G_OPD_CKPT_DIR"
        exit 1
    fi
    FSDP_CKPT_DIR="$LATEST_STEP_DIR/actor"
fi

CKPT_STEP="$(basename "$(dirname "$FSDP_CKPT_DIR")")"
if [ -n "${G_OPD_CKPT_DIR:-}" ]; then
    DEFAULT_MODEL_BASE="$(basename "$G_OPD_CKPT_DIR")"
else
    DEFAULT_MODEL_BASE="$(basename "$(dirname "$(dirname "$FSDP_CKPT_DIR")")")"
fi
MODEL="${MODEL:-${DEFAULT_MODEL_BASE}_${CKPT_STEP}}"
MODEL_PATH="${MODEL_PATH:-$FSDP_CKPT_DIR/merged_hf}"
MODEL_NAME="${MODEL_NAME:-${MODEL}_code_lcbv5_n8}"
LCB_MODEL_ID="${LCB_MODEL_ID:-Qwen3-4B-NonThinking}"

find_merged_weight() {
    find "$MODEL_PATH" -maxdepth 1 -type f \
        \( -name 'model*.safetensors' -o -name 'pytorch_model*.bin' \) \
        -print -quit 2>/dev/null || true
}

MERGED_WEIGHT="$(find_merged_weight)"
if [ -z "$MERGED_WEIGHT" ] || [ ! -f "$MODEL_PATH/config.json" ] || [ ! -f "$MODEL_PATH/tokenizer_config.json" ]; then
    if [ ! -f "$FSDP_CKPT_DIR/fsdp_config.json" ]; then
        echo "ERROR: FSDP checkpoint is not ready: $FSDP_CKPT_DIR"
        echo "Expected fsdp_config.json and model_world_size_<N>_rank_<R>.pt shards."
        exit 1
    fi

    FSDP_WORLD_SIZE="$(python3 - "$FSDP_CKPT_DIR/fsdp_config.json" <<'PYFS'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as f:
    world_size = json.load(f).get("world_size")
if not isinstance(world_size, int) or world_size <= 0:
    raise SystemExit(f"invalid world_size in {sys.argv[1]}: {world_size!r}")
print(world_size)
PYFS
)"

    MISSING_SHARDS=0
    for ((rank = 0; rank < FSDP_WORLD_SIZE; rank++)); do
        shard="$FSDP_CKPT_DIR/model_world_size_${FSDP_WORLD_SIZE}_rank_${rank}.pt"
        if [ ! -f "$shard" ]; then
            echo "ERROR: missing FSDP checkpoint shard: $shard"
            MISSING_SHARDS=1
        fi
    done
    if [ "$MISSING_SHARDS" = "1" ]; then
        echo "Checkpoint may still be saving; retry after global_step checkpoint creation finishes."
        exit 1
    fi

    echo "[merge] $FSDP_CKPT_DIR -> $MODEL_PATH"
    (
        cd "$PROJECT_ROOT/verl"
        python3 -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "$FSDP_CKPT_DIR" \
            --target_dir "$MODEL_PATH"
    )
fi

MERGED_WEIGHT="$(find_merged_weight)"
if [ -z "$MERGED_WEIGHT" ] || [ ! -f "$MODEL_PATH/config.json" ] || [ ! -f "$MODEL_PATH/tokenizer_config.json" ]; then
    echo "ERROR: MODEL_PATH is not a Hugging Face model directory: $MODEL_PATH"
    exit 1
fi

echo "[code-eval] model_path=$MODEL_PATH"
echo "[code-eval] model_name=$MODEL_NAME"
echo "[code-eval] output_dir=$OUTPUT_DIR"

RUN_LCB="${RUN_LCB:-1}"
RUN_EVALPLUS="${RUN_EVALPLUS:-1}"
RUN_HUMANEVAL="${RUN_HUMANEVAL:-1}"
RUN_MBPP="${RUN_MBPP:-1}"
LCB_RELEASE="${LCB_RELEASE:-v5}"
LCB_N="${LCB_N:-8}"
LCB_TEMPERATURE="${LCB_TEMPERATURE:-1.0}"
LCB_TOP_P="${LCB_TOP_P:-1.0}"
LCB_MAX_TOKENS="${LCB_MAX_TOKENS:-2048}"
LCB_GPUS="${LCB_GPUS:-0,1,2,3}"
LCB_TP="${LCB_TP:-4}"
LCB_BATCH_SIZE="${LCB_BATCH_SIZE:-64}"
LCB_MODEL_PATH="${LCB_MODEL_PATH:-$OUTPUT_DIR/lcb_model_paths/$MODEL_NAME}"
LCB_QWEN3_TOKENIZER="${LCB_QWEN3_TOKENIZER:-$MODEL_PATH}"
mkdir -p "$(dirname "$LCB_MODEL_PATH")"
if [ ! -e "$LCB_MODEL_PATH" ]; then
    ln -s "$MODEL_PATH" "$LCB_MODEL_PATH"
fi
LCB_OUTPUT_MODEL_DIR="${LCB_MODEL_PATH##*/}"
LCB_SCENARIO_OUTPUT="${LCB_SCENARIO_OUTPUT:-Scenario.codegeneration}"
LCB_EVAL_ALL_FILE="${LCB_EVAL_ALL_FILE:-$CODE_EVAL_DIR/coding/LiveCodeBench/lcb_outputs/$LCB_OUTPUT_MODEL_DIR/${LCB_SCENARIO_OUTPUT}_${LCB_N}_${LCB_TEMPERATURE}_eval_all.json}"
LCB_DATASET_DIR="${LCB_DATASET_DIR:-$CODE_EVAL_DIR/coding/LiveCodeBench/code_generation_lite}"
LCB_DATASET_REPO="${LCB_DATASET_REPO:-livecodebench/code_generation_lite}"
LCB_AUTO_DOWNLOAD_DATA="${LCB_AUTO_DOWNLOAD_DATA:-1}"
SUMMARY_FILE="${SUMMARY_FILE:-$OUTPUT_DIR/${MODEL_NAME}_code_summary.csv}"

EVALPLUS_ROOT="${EVALPLUS_ROOT:-$OUTPUT_DIR/evalplus_results}"
EVALPLUS_MODEL="${EVALPLUS_MODEL:-$MODEL_PATH}"
EVALPLUS_BACKEND="${EVALPLUS_BACKEND:-vllm}"
EVALPLUS_GPUS="${EVALPLUS_GPUS:-$LCB_GPUS}"
EVALPLUS_TP="${EVALPLUS_TP:-$LCB_TP}"
EVALPLUS_GREEDY="${EVALPLUS_GREEDY:-0}"
if [ "$EVALPLUS_GREEDY" = "1" ]; then
    EVALPLUS_TEMPERATURE="0.0"
    EVALPLUS_N_SAMPLES="1"
else
    EVALPLUS_TEMPERATURE="${EVALPLUS_TEMPERATURE:-1.0}"
    EVALPLUS_N_SAMPLES="${EVALPLUS_N_SAMPLES:-8}"
fi
EVALPLUS_BATCH_SIZE="${EVALPLUS_BATCH_SIZE:-}"
EVALPLUS_MAX_TOKENS="${EVALPLUS_MAX_TOKENS:-4096}"
EVALPLUS_DTYPE="${EVALPLUS_DTYPE:-bfloat16}"
EVALPLUS_TRUST_REMOTE_CODE="${EVALPLUS_TRUST_REMOTE_CODE:-1}"
EVALPLUS_FORCE_BASE_PROMPT="${EVALPLUS_FORCE_BASE_PROMPT:-0}"
EVALPLUS_ENABLE_PREFIX_CACHING="${EVALPLUS_ENABLE_PREFIX_CACHING:-0}"
EVALPLUS_ENABLE_CHUNKED_PREFILL="${EVALPLUS_ENABLE_CHUNKED_PREFILL:-0}"
EVALPLUS_MIN_TIME_LIMIT="${EVALPLUS_MIN_TIME_LIMIT:-10.0}"
EVALPLUS_GT_TIME_LIMIT_FACTOR="${EVALPLUS_GT_TIME_LIMIT_FACTOR:-8.0}"
EVALPLUS_PARALLEL="${EVALPLUS_PARALLEL:-}"
EVALPLUS_ALLOW_OVERWRITE="${EVALPLUS_ALLOW_OVERWRITE:-1}"
EVALPLUS_PYTHONPATH="$CODE_EVAL_DIR/coding/evalplus${PYTHONPATH:+:$PYTHONPATH}"
HUMANEVAL_RESULT_FILE="${HUMANEVAL_RESULT_FILE:-$EVALPLUS_ROOT/humaneval/${MODEL_NAME}_eval_results.json}"
MBPP_RESULT_FILE="${MBPP_RESULT_FILE:-$EVALPLUS_ROOT/mbpp/${MODEL_NAME}_eval_results.json}"
lcb_release_files() {
    local release="$1"
    local start end idx

    case "$release" in
        release_v1)
            echo "test.jsonl"
            return
            ;;
        release_v2)
            echo "test.jsonl test2.jsonl"
            return
            ;;
        release_v3)
            echo "test.jsonl test2.jsonl test3.jsonl"
            return
            ;;
        release_v4)
            echo "test.jsonl test2.jsonl test3.jsonl test4.jsonl"
            return
            ;;
        release_v5)
            echo "test.jsonl test2.jsonl test3.jsonl test4.jsonl test5.jsonl"
            return
            ;;
        release_v6 | release_latest)
            echo "test.jsonl test2.jsonl test3.jsonl test4.jsonl test5.jsonl test6.jsonl"
            return
            ;;
        v1)
            echo "test.jsonl"
            return
            ;;
        v2 | v3 | v4 | v5 | v6)
            echo "test${release#v}.jsonl"
            return
            ;;
    esac

    if [[ "$release" =~ ^v([1-6])_v([1-6])$ ]]; then
        start="${BASH_REMATCH[1]}"
        end="${BASH_REMATCH[2]}"
        if (( start >= end )); then
            echo "ERROR: invalid LCB_RELEASE range: $release" >&2
            return 1
        fi
        for ((idx = start; idx <= end; idx++)); do
            if (( idx == 1 )); then
                printf 'test.jsonl'
            else
                printf 'test%s.jsonl' "$idx"
            fi
            if (( idx < end )); then
                printf ' '
            fi
        done
        printf '\n'
        return
    fi

    echo "ERROR: unknown LCB_RELEASE=$release" >&2
    return 1
}

download_lcb_dataset_files() {
    local -a missing_files=("$@")
    local -a include_args=()
    local file

    mkdir -p "$LCB_DATASET_DIR"
    for file in "${missing_files[@]}"; do
        include_args+=(--include "$file")
    done

    if command -v hf >/dev/null 2>&1; then
        hf download "$LCB_DATASET_REPO" \
            --repo-type dataset \
            "${include_args[@]}" \
            --local-dir "$LCB_DATASET_DIR"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download "$LCB_DATASET_REPO" \
            --repo-type dataset \
            "${include_args[@]}" \
            --local-dir "$LCB_DATASET_DIR" \
            --local-dir-use-symlinks False
    else
        LCB_MISSING_FILES="${missing_files[*]}" \
        LCB_DATASET_REPO="$LCB_DATASET_REPO" \
        LCB_DATASET_DIR="$LCB_DATASET_DIR" \
        python3 - <<'PYLCB'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["LCB_DATASET_REPO"],
    repo_type="dataset",
    local_dir=os.environ["LCB_DATASET_DIR"],
    allow_patterns=os.environ["LCB_MISSING_FILES"].split(),
    local_dir_use_symlinks=False,
)
PYLCB
    fi
}

ensure_lcb_dataset() {
    local -a required_files=()
    local -a missing_files=()
    local file

    read -r -a required_files <<<"$(lcb_release_files "$LCB_RELEASE")"
    if [ "${#required_files[@]}" -eq 0 ]; then
        echo "ERROR: no LiveCodeBench files resolved for LCB_RELEASE=$LCB_RELEASE"
        exit 1
    fi

    for file in "${required_files[@]}"; do
        if [ ! -s "$LCB_DATASET_DIR/$file" ]; then
            missing_files+=("$file")
        fi
    done

    if [ "${#missing_files[@]}" -eq 0 ]; then
        return
    fi

    if [ "$LCB_AUTO_DOWNLOAD_DATA" != "1" ]; then
        echo "ERROR: missing LiveCodeBench data files under $LCB_DATASET_DIR: ${missing_files[*]}"
        echo "Set LCB_AUTO_DOWNLOAD_DATA=1 or download $LCB_DATASET_REPO into $LCB_DATASET_DIR."
        exit 1
    fi

    echo "[code-eval] missing LiveCodeBench data: ${missing_files[*]}"
    echo "[code-eval] downloading $LCB_DATASET_REPO to $LCB_DATASET_DIR"
    download_lcb_dataset_files "${missing_files[@]}"

    missing_files=()
    for file in "${required_files[@]}"; do
        if [ ! -s "$LCB_DATASET_DIR/$file" ]; then
            missing_files+=("$file")
        fi
    done

    if [ "${#missing_files[@]}" -ne 0 ]; then
        echo "ERROR: LiveCodeBench dataset download did not produce: ${missing_files[*]}"
        echo "Expected files under: $LCB_DATASET_DIR"
        exit 1
    fi
}

evalplus_model_identifier() {
    local model="$1"
    local backend="$2"
    local temperature="$3"
    local stripped="$model"

    while [ -n "$stripped" ]; do
        case "${stripped:0:1}" in
            . | /) stripped="${stripped:1}" ;;
            *) break ;;
        esac
    done
    while [ -n "$stripped" ]; do
        case "${stripped: -1}" in
            . | /) stripped="${stripped:0:${#stripped}-1}" ;;
            *) break ;;
        esac
    done

    stripped="${stripped//\//--}"
    printf '%s_%s_temp_%s' "$stripped" "$backend" "$temperature"
}

evalplus_samples_file() {
    local dataset="$1"
    local identifier
    identifier="$(evalplus_model_identifier "$EVALPLUS_MODEL" "$EVALPLUS_BACKEND" "$EVALPLUS_TEMPERATURE")"
    printf '%s/%s/%s.jsonl' "$EVALPLUS_ROOT" "$dataset" "$identifier"
}

configure_evalplus_data_overrides() {
    local humaneval_data="$CODE_EVAL_DIR/data/HumanEvalPlus.jsonl"
    local mbpp_data="$CODE_EVAL_DIR/data/MbppPlus.jsonl"

    if [ -z "${HUMANEVAL_OVERRIDE_PATH:-}" ] && [ -f "$humaneval_data" ]; then
        export HUMANEVAL_OVERRIDE_PATH="$humaneval_data"
    fi
    if [ -z "${MBPP_OVERRIDE_PATH:-}" ] && [ -f "$mbpp_data" ]; then
        export MBPP_OVERRIDE_PATH="$mbpp_data"
    fi
}

run_evalplus_dataset() {
    local dataset="$1"
    local result_file="$2"
    local samples_file
    local -a codegen_args
    local -a evaluate_args

    mkdir -p "$(dirname "$result_file")"
    if [ -f "$result_file" ] && [ "$EVALPLUS_ALLOW_OVERWRITE" != "1" ]; then
        echo "[code-eval] using existing EvalPlus $dataset result file: $result_file"
        return
    fi

    echo "[code-eval] running EvalPlus $dataset model=$EVALPLUS_MODEL greedy=$EVALPLUS_GREEDY n=$EVALPLUS_N_SAMPLES max_tokens=$EVALPLUS_MAX_TOKENS"
    codegen_args=(
        -m evalplus.codegen
        --model "$EVALPLUS_MODEL"
        --dataset "$dataset"
        --root "$EVALPLUS_ROOT"
        --backend "$EVALPLUS_BACKEND"
        --temperature "$EVALPLUS_TEMPERATURE"
        --n_samples "$EVALPLUS_N_SAMPLES"
        --max_new_tokens "$EVALPLUS_MAX_TOKENS"
        --tp "$EVALPLUS_TP"
        --dtype "$EVALPLUS_DTYPE"
    )
    if [ "$EVALPLUS_GREEDY" = "1" ]; then
        codegen_args+=(--greedy)
    fi
    if [ -n "$EVALPLUS_BATCH_SIZE" ]; then
        codegen_args+=(--bs "$EVALPLUS_BATCH_SIZE")
    fi
    if [ "$EVALPLUS_TRUST_REMOTE_CODE" = "1" ]; then
        codegen_args+=(--trust_remote_code)
    fi
    if [ "$EVALPLUS_FORCE_BASE_PROMPT" = "1" ]; then
        codegen_args+=(--force_base_prompt)
    fi
    if [ "$EVALPLUS_ENABLE_PREFIX_CACHING" = "1" ]; then
        codegen_args+=(--enable_prefix_caching)
    fi
    if [ "$EVALPLUS_ENABLE_CHUNKED_PREFILL" = "1" ]; then
        codegen_args+=(--enable_chunked_prefill)
    fi
    CUDA_VISIBLE_DEVICES="$EVALPLUS_GPUS" PYTHONPATH="$EVALPLUS_PYTHONPATH" python3 "${codegen_args[@]}"

    samples_file="$(evalplus_samples_file "$dataset")"
    if [ ! -f "$samples_file" ]; then
        echo "ERROR: EvalPlus samples file not found after codegen: $samples_file"
        exit 1
    fi

    if [ -f "$result_file" ] && [ "$EVALPLUS_ALLOW_OVERWRITE" = "1" ]; then
        local backup_file="${result_file}.bak.$(date +%Y%m%d%H%M%S)"
        mv "$result_file" "$backup_file"
        echo "[code-eval] backed up existing EvalPlus result to: $backup_file"
    fi

    evaluate_args=(
        -m evalplus.evaluate
        --dataset "$dataset"
        --samples "$samples_file"
        --output_file "$result_file"
        --min_time_limit "$EVALPLUS_MIN_TIME_LIMIT"
        --gt_time_limit_factor "$EVALPLUS_GT_TIME_LIMIT_FACTOR"
    )
    if [ -n "$EVALPLUS_PARALLEL" ]; then
        evaluate_args+=(--parallel "$EVALPLUS_PARALLEL")
    fi
    PYTHONPATH="$EVALPLUS_PYTHONPATH" python3 "${evaluate_args[@]}"

    if [ -f "$result_file" ]; then
        echo "[code-eval] EvalPlus $dataset result file: $result_file"
    else
        echo "[warn] EvalPlus $dataset result file not found after evaluation: $result_file"
    fi
}

if [ "$RUN_LCB" = "1" ]; then
    ensure_lcb_dataset
    echo "[code-eval] running LiveCodeBench $LCB_RELEASE n=$LCB_N max_tokens=$LCB_MAX_TOKENS"
    (
        cd "$CODE_EVAL_DIR/coding/LiveCodeBench"
        LCB_QWEN3_TOKENIZER="$LCB_QWEN3_TOKENIZER" CUDA_VISIBLE_DEVICES="$LCB_GPUS" python3 -m lcb_runner.runner.main \
            --model "$LCB_MODEL_ID" \
            --local_model_path "$LCB_MODEL_PATH" \
            --trust_remote_code \
            --scenario codegeneration \
            --release_version "$LCB_RELEASE" \
            --tensor_parallel_size "$LCB_TP" \
            --use_cache \
            --cache_batch_size "$LCB_BATCH_SIZE" \
            --n "$LCB_N" \
            --temperature "$LCB_TEMPERATURE" \
            --top_p "$LCB_TOP_P" \
            --max_tokens "$LCB_MAX_TOKENS" \
            --custom_output_save_name "$MODEL_NAME" \
            --timeout 60 \
            --evaluate --continue_existing --continue_existing_with_eval
    )
fi

if [ "$RUN_EVALPLUS" = "1" ]; then
    configure_evalplus_data_overrides
    if [ "$RUN_HUMANEVAL" = "1" ]; then
        run_evalplus_dataset humaneval "$HUMANEVAL_RESULT_FILE"
    fi
    if [ "$RUN_MBPP" = "1" ]; then
        run_evalplus_dataset mbpp "$MBPP_RESULT_FILE"
    fi
fi

if [ ! -f "$LCB_EVAL_ALL_FILE" ]; then
    echo "[warn] LiveCodeBench eval_all file not found: $LCB_EVAL_ALL_FILE"
    echo "[warn] LiveCodeBench columns will be NA unless this file exists."
fi

SUMMARY_ARGS=(
    --model-name "$MODEL_NAME"
    --lcb-eval-all-file "$LCB_EVAL_ALL_FILE"
    --humaneval-result-file "$HUMANEVAL_RESULT_FILE"
    --mbpp-result-file "$MBPP_RESULT_FILE"
    --output-file "$SUMMARY_FILE"
)
python3 "$CODE_EVAL_DIR/summarize_code_eval.py" "${SUMMARY_ARGS[@]}"

echo "[code-eval] summary=$SUMMARY_FILE"
echo "[code-eval] done"
