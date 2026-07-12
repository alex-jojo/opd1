#!/usr/bin/env bash

if [ -d /venv/verl/bin ]; then
    export PATH="/venv/verl/bin:$PATH"
fi
if [ -d /venv/main/bin ]; then
    export PATH="$PATH:/venv/main/bin"
fi

CODE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_VERL_DIR="$(cd "$CODE_SCRIPT_DIR/../.." && pwd)"
CODE_ROOT_DIR="$(cd "$CODE_VERL_DIR/.." && pwd)"

export CODE_TOTAL_STEPS="${TOTAL_STEPS:-109}"
export CODE_SAVE_FREQ="${SAVE_FREQ:-109}"
export CODE_MAX_PROMPT_LENGTH="${CODE_MAX_PROMPT_LENGTH:-2048}"
export CODE_MAX_RESPONSE_LENGTH="${CODE_MAX_RESPONSE_LENGTH:-2048}"
export EOPD_MAX_PROMPT_LENGTH="${EOPD_MAX_PROMPT_LENGTH:-$CODE_MAX_PROMPT_LENGTH}"
export EOPD_MAX_RESPONSE_LENGTH="${EOPD_MAX_RESPONSE_LENGTH:-$CODE_MAX_RESPONSE_LENGTH}"
export TA_OPD_MAX_PROMPT_LENGTH="${TA_OPD_MAX_PROMPT_LENGTH:-$CODE_MAX_PROMPT_LENGTH}"
export TA_OPD_MAX_RESPONSE_LENGTH="${TA_OPD_MAX_RESPONSE_LENGTH:-$CODE_MAX_RESPONSE_LENGTH}"
export G_OPD_MAX_RESPONSE_LENGTH="${G_OPD_MAX_RESPONSE_LENGTH:-$CODE_MAX_RESPONSE_LENGTH}"
export CODE_CONSTANT_REWARD="${CODE_CONSTANT_REWARD:-1}"
export CODE_CONSTANT_REWARD_VALUE="${CODE_CONSTANT_REWARD_VALUE:-1.0}"

export STUDENT_MODEL="${STUDENT_MODEL:-/workspace/models/Qwen3-4B}"
export STUDENT_MODEL_REPO="${STUDENT_MODEL_REPO:-Qwen/Qwen3-4B}"
export TEACHER_MODEL="${TEACHER_MODEL:-/workspace/models/Qwen3-4B-Instruct-2507}"
export TEACHER_MODEL_REPO="${TEACHER_MODEL_REPO:-Qwen/Qwen3-4B-Instruct-2507}"

export TRAIN_DATA_REPO="${TRAIN_DATA_REPO:-Skywork/Skywork-OR1-RL-Data}"
export TRAIN_DATA_SPLIT="${TRAIN_DATA_SPLIT:-code}"
export TRAIN_SRC="${TRAIN_SRC:-$CODE_ROOT_DIR/data/skywork_or1_rl_data_code.parquet}"
export TRAIN_FILE="${TRAIN_FILE:-$CODE_VERL_DIR/train_skywork_or1_rl_data_code_verl.parquet}"

download_code_hf_repo() {
    local repo="$1"
    local local_dir="$2"

    mkdir -p "$(dirname "$local_dir")"
    if command -v hf >/dev/null 2>&1; then
        hf download "$repo" --local-dir "$local_dir"
    else
        huggingface-cli download "$repo" \
            --local-dir "$local_dir" \
            --local-dir-use-symlinks False
    fi
}

is_code_model_complete() {
    local local_dir="$1"

    [ -f "$local_dir/config.json" ] || return 1
    [ -f "$local_dir/tokenizer_config.json" ] || return 1

    python3 - "$local_dir" <<'PYMODEL'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])

index_files = [
    model_dir / "model.safetensors.index.json",
    model_dir / "pytorch_model.bin.index.json",
]
for index_file in index_files:
    if index_file.exists():
        data = json.loads(index_file.read_text())
        shards = sorted(set(data.get("weight_map", {}).values()))
        if not shards:
            raise SystemExit(1)
        missing = [shard for shard in shards if not (model_dir / shard).is_file()]
        raise SystemExit(1 if missing else 0)

single_weights = list(model_dir.glob("model*.safetensors")) + list(model_dir.glob("pytorch_model*.bin"))
raise SystemExit(0 if single_weights else 1)
PYMODEL
}

ensure_code_model() {
    local repo="$1"
    local local_dir="$2"

    if ! is_code_model_complete "$local_dir"; then
        echo "[code setup] model missing or incomplete, downloading $repo to $local_dir"
        download_code_hf_repo "$repo" "$local_dir"
    fi

    if ! is_code_model_complete "$local_dir"; then
        echo "ERROR: model download did not produce a complete Hugging Face model: $local_dir"
        echo "Expected config/tokenizer files and all weight shards listed by model.safetensors.index.json."
        exit 1
    fi
}

ensure_code_train_data() {
    if [ -f "$TRAIN_SRC" ]; then
        return
    fi

    echo "[code setup] training data missing, downloading $TRAIN_DATA_REPO split=$TRAIN_DATA_SPLIT to $TRAIN_SRC"
    local cache_dir="$CODE_ROOT_DIR/data/skywork_or1_rl_data_${TRAIN_DATA_SPLIT}_shards"
    mkdir -p "$cache_dir"

    if command -v hf >/dev/null 2>&1; then
        hf download "$TRAIN_DATA_REPO" \
            --repo-type dataset \
            --include "data/${TRAIN_DATA_SPLIT}-*.parquet" \
            --local-dir "$cache_dir"
    else
        huggingface-cli download "$TRAIN_DATA_REPO" \
            --repo-type dataset \
            --include "data/${TRAIN_DATA_SPLIT}-*.parquet" \
            --local-dir "$cache_dir" \
            --local-dir-use-symlinks False
    fi

    env \
        TRAIN_DATA_REPO="$TRAIN_DATA_REPO" \
        TRAIN_DATA_SPLIT="$TRAIN_DATA_SPLIT" \
        TRAIN_SRC="$TRAIN_SRC" \
        TRAIN_DATA_CACHE_DIR="$cache_dir" \
        python3 - <<'PYDATA'
import os
from pathlib import Path

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit("ERROR: Python package 'pandas' is required to merge Skywork parquet shards") from exc

repo = os.environ["TRAIN_DATA_REPO"]
split = os.environ["TRAIN_DATA_SPLIT"]
out = Path(os.environ["TRAIN_SRC"])
cache_dir = Path(os.environ["TRAIN_DATA_CACHE_DIR"])
shards = sorted((cache_dir / "data").glob(f"{split}-*.parquet"))
if not shards:
    raise SystemExit(f"ERROR: no parquet shards found for {repo} split={split} under {cache_dir}/data")

out.parent.mkdir(parents=True, exist_ok=True)
df = pd.concat((pd.read_parquet(shard) for shard in shards), ignore_index=True)
if df.empty:
    raise SystemExit(f"ERROR: {repo} split={split} is empty")

df.to_parquet(out, index=False)
print(f"saved: {out}")
print(f"rows: {len(df)}")
print(f"columns: {list(df.columns)}")
PYDATA

    if [ ! -f "$TRAIN_SRC" ]; then
        echo "ERROR: training data download did not produce parquet: $TRAIN_SRC"
        exit 1
    fi
}

prepare_code_training_inputs() {
    export TRAIN_SRC TRAIN_FILE
    export STUDENT_MODEL STUDENT_MODEL_REPO
    export TEACHER_MODEL TEACHER_MODEL_REPO

    ensure_code_train_data
    ensure_code_model "$STUDENT_MODEL_REPO" "$STUDENT_MODEL"
    ensure_code_model "$TEACHER_MODEL_REPO" "$TEACHER_MODEL"
}
