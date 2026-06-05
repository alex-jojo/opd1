#!/bin/bash
set -e

if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
    pip install -U datasets "huggingface_hub<1.0,>=0.34.0"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$SCRIPT_DIR/eval_math.py" ]; then
    MATH_EVAL_DIR="$SCRIPT_DIR"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    PROJECT_ROOT="$SCRIPT_DIR"
    MATH_EVAL_DIR="$PROJECT_ROOT/math_eval"
fi

DATA_DIR="$PROJECT_ROOT/data"
OUTPUT_DIR="$MATH_EVAL_DIR/eval_outputs"

cd "$PROJECT_ROOT"

cat > "$PROJECT_ROOT/download_math_datasets.py" <<'PY'
import os
import re
import json
from datasets import load_dataset

DATASETS = {
    "aime24": {
        "repo": "math-ai/aime24",
        "out": "data/aime24/test.jsonl",
    },
    "aime25": {
        "repo": "math-ai/aime25",
        "out": "data/aime25/test.jsonl",
    },
    "aime26": {
        "repo": "math-ai/aime26",
        "out": "data/aime26/test.jsonl",
    },
    "hmmt26": {
        "repo": "MathArena/hmmt_feb_2026",
        "out": "data/hmmt26/test.jsonl",
    },
    "amc23": {
        "repo": "math-ai/amc23",
        "out": "data/amc23/test.jsonl",
    },
    "math500": {
        "repo": "HuggingFaceH4/MATH-500",
        "out": "data/math500/test.jsonl",
    },
}


def pick_split(ds):
    for split in ["test", "train", "validation", "dev"]:
        if split in ds:
            return ds[split]
    return ds[list(ds.keys())[0]]


def extract_problem(row):
    for key in ["problem", "question", "Problem", "Question", "input", "prompt"]:
        if key in row and row[key] is not None:
            return str(row[key])
    raise KeyError(f"Cannot find problem field. Keys: {list(row.keys())}")


def extract_answer(row):
    for key in ["answer", "Answer", "final_answer", "target", "ground_truth", "solution", "Solution"]:
        if key in row and row[key] is not None:
            value = row[key]

            if isinstance(value, dict):
                for k in ["answer", "ground_truth", "value"]:
                    if k in value:
                        return str(value[k])
                return json.dumps(value, ensure_ascii=False)

            if isinstance(value, list):
                return str(value[0]) if value else ""

            return str(value)

    raise KeyError(f"Cannot find answer field. Keys: {list(row.keys())}")


def strip_boxed(ans):
    ans = str(ans).strip()

    boxed = re.findall(r"\\(?:boxed|fbox)\{([^{}]+)\}", ans, flags=re.DOTALL)
    if boxed:
        ans = boxed[-1].strip()

    m = re.fullmatch(r"\\boxed\{(.+)\}", ans, flags=re.DOTALL)
    if m:
        ans = m.group(1).strip()

    m = re.fullmatch(r"\\fbox\{(.+)\}", ans, flags=re.DOTALL)
    if m:
        ans = m.group(1).strip()

    try:
        f = float(ans)
        if f.is_integer():
            ans = str(int(f))
    except Exception:
        pass

    return ans


def convert_one(name, repo, out_path):
    print(f"[download] {name}: {repo}")

    ds = load_dataset(repo)
    split = pick_split(ds)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in split:
            item = {
                "problem": extract_problem(row),
                "answer": strip_boxed(extract_answer(row)),
            }

            for k in [
                "id",
                "ID",
                "url",
                "subject",
                "level",
                "unique_id",
                "problem_idx",
                "problem_type",
            ]:
                if k in row:
                    item[k] = row[k]

            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1

    print(f"[saved] {out_path}: {n} rows")


def main():
    for name, cfg in DATASETS.items():
        convert_one(name, cfg["repo"], cfg["out"])


if __name__ == "__main__":
    main()
PY

if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
    python3 "$PROJECT_ROOT/download_math_datasets.py"
else
    echo "[skip] dataset download because SKIP_DOWNLOAD=1"
fi

MODEL="${MODEL:-qwen3_1.7b_non_thinking_teacher_qwen3_4b_non_thinking_rl_math_rubric_40_save_step_50}"
FSDP_CKPT_DIR="${FSDP_CKPT_DIR:-/G-OPD-checkpoints/qwen3_1.7b_non_thinking_teacher_qwen3_4b_non_thinking_rl_math_rubric_40_save_step_50/global_step_110/actor}"
MODEL_PATH="${MODEL_PATH:-$FSDP_CKPT_DIR/merged_hf}"
MODEL_NAME="${MODEL_NAME:-${MODEL}_8k_n8}"

if ! ls "$MODEL_PATH"/model*.safetensors "$MODEL_PATH"/pytorch_model*.bin >/dev/null 2>&1; then
    if [ ! -f "$FSDP_CKPT_DIR/model_world_size_4_rank_0.pt" ]; then
        echo "ERROR: missing FSDP checkpoint shards: $FSDP_CKPT_DIR"
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

if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "ERROR: MODEL_PATH is not a Hugging Face model directory: $MODEL_PATH"
    exit 1
fi

echo "$MODEL_PATH"
echo "$MODEL_NAME"

GPU_GROUP_0="${GPU_GROUP_0:-0,1}"
GPU_GROUP_1="${GPU_GROUP_1:-2,3}"
DATASETS_TO_EVAL="${DATASETS:-aime24 aime25 aime26 hmmt26 amc23 math500}"

mkdir -p "$OUTPUT_DIR/aime24"
mkdir -p "$OUTPUT_DIR/aime25"
mkdir -p "$OUTPUT_DIR/aime26"
mkdir -p "$OUTPUT_DIR/hmmt26"
mkdir -p "$OUTPUT_DIR/amc23"
mkdir -p "$OUTPUT_DIR/math500"

run_eval() {
    local dataset="$1"
    local gpu_devices="$2"

    echo "[eval] $dataset on GPUs $gpu_devices"

    CUDA_VISIBLE_DEVICES="$gpu_devices" python3 "$MATH_EVAL_DIR/eval_math.py" \
        --input_file "$DATA_DIR/$dataset/test.jsonl" \
        --model_path "$MODEL_PATH" \
        --output_file "$OUTPUT_DIR/$dataset/${MODEL_NAME}.jsonl" \
        --max_tokens 8192 \
        --temperature 1.0 \
        --top_p 1.0 \
        --max_num_seqs 256 \
        --n 8 \
        --begin_idx -1 \
        --end_idx -1 \
        --seed 42 &
}

run_selected_evals() {
    local datasets=("$@")
    local i=0

    while [ $i -lt ${#datasets[@]} ]; do
        run_eval "${datasets[$i]}" "$GPU_GROUP_0"

        if [ $((i + 1)) -lt ${#datasets[@]} ]; then
            run_eval "${datasets[$((i + 1))]}" "$GPU_GROUP_1"
        fi

        wait
        i=$((i + 2))
    done
}

# shellcheck disable=SC2086
run_selected_evals $DATASETS_TO_EVAL

echo "Model $MODEL_NAME done!"

if [ "${SHOW_SUMMARY:-1}" = "1" ]; then
    python3 "$MATH_EVAL_DIR/summarize_eval.py" \
        --output-dir "$OUTPUT_DIR" \
        --models "$MODEL_NAME"
fi


