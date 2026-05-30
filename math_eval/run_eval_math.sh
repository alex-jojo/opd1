#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-"Qwen3-1.7B"}
MODEL_PATH=${MODEL_PATH:-"/workspace/models/Qwen3-1.7B"}
MODEL_NAME=${MODEL_NAME:-$MODEL}

MAX_TOKENS=${MAX_TOKENS:-16384}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
N=${N:-16}
SEED=${SEED:-42}

echo "$MODEL_PATH"
echo "$MODEL_NAME"

run_eval() {
    local dataset=$1
    local input_file=$2
    local cuda_devices=$3

    if [ ! -f "$input_file" ]; then
        echo "ERROR: missing dataset file for ${dataset}: ${input_file}"
        exit 1
    fi

    mkdir -p "./eval_outputs/${dataset}"

    CUDA_VISIBLE_DEVICES=$cuda_devices python3 eval_math.py \
        --input_file "$input_file" \
        --model_path "$MODEL_PATH" \
        --output_file "./eval_outputs/${dataset}/${MODEL_NAME}.jsonl" \
        --max_tokens "$MAX_TOKENS" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --max_num_seqs "$MAX_NUM_SEQS" \
        --n "$N" \
        --begin_idx -1 \
        --end_idx -1 \
        --seed "$SEED" &
}

# Run sequentially on the same two physical GPUs used by training.
run_eval aime24 ../data/aime24/test.jsonl 2,3
wait

run_eval aime25 ../data/aime25/test.jsonl 2,3
wait

run_eval aime26 ../data/aime26/test.jsonl 2,3
wait

run_eval hmmt26 ../data/hmmt26/test.jsonl 2,3
wait

run_eval amc23 ../data/amc23/test.jsonl 2,3
wait

run_eval math500 ../data/math500/test.jsonl 2,3
wait

echo "Model $MODEL_NAME done!"
