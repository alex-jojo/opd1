#!/usr/bin/env bash
set -euo pipefail

if [ "${G_OPD_SHELL_DEBUG:-0}" = "1" ]; then
    set -x
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_WORKSPACE="${VERL_WORKSPACE:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
OPD_REPO_ROOT="${OPD_REPO_ROOT:-$(cd "$VERL_WORKSPACE/.." && pwd)}"


export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=online
export USED_MODEL="no_api"
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export HYDRA_FULL_ERROR=1
export VERL_PRINT_CONFIG="${VERL_PRINT_CONFIG:-0}"
export G_OPD_PROGRESS_DEBUG="${G_OPD_PROGRESS_DEBUG:-1}"
export GPT_ROLLOUT_SCORE_VERBOSE="${GPT_ROLLOUT_SCORE_VERBOSE:-1}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
cd "$VERL_WORKSPACE"

G_OPD_ENV_FILE="${G_OPD_ENV_FILE:-$VERL_WORKSPACE/.env}"
if [ -f "$G_OPD_ENV_FILE" ]; then
    case "$-" in
        *x*) G_OPD_RESTORE_XTRACE=1; set +x ;;
        *) G_OPD_RESTORE_XTRACE=0 ;;
    esac
    set -a
    # shellcheck disable=SC1090
    . "$G_OPD_ENV_FILE"
    set +a
    if [ "$G_OPD_RESTORE_XTRACE" = 1 ]; then
        set -x
    fi
    unset G_OPD_RESTORE_XTRACE
fi

TRAIN_SRC="${TRAIN_SRC:-$OPD_REPO_ROOT/data/train-00000-of-00001.parquet}"
TRAIN_FILE="${TRAIN_FILE:-$VERL_WORKSPACE/train_verl.parquet}"
export TRAIN_SRC TRAIN_FILE

AIME26_JSONL="${AIME26_JSONL:-/workspace/G-OPD/data/aime26/test.jsonl}"
AIME26_PARQUET="${AIME26_PARQUET:-/workspace/G-OPD/data/aime26/test_verl.parquet}"
export AIME26_JSONL AIME26_PARQUET

STUDENT_MODEL="${STUDENT_MODEL:-/workspace/models/Qwen3-1.7B}"
TEACHER_MODEL="${TEACHER_MODEL:-/workspace/models/Qwen3-4B-Non-Thinking-RL-Math-Step500}"
STUDENT_MODEL_REPO="${STUDENT_MODEL_REPO:-Qwen/Qwen3-1.7B}"
TEACHER_MODEL_REPO="${TEACHER_MODEL_REPO:-Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500}"

if [ ! -f "$AIME26_JSONL" ]; then
    mkdir -p "$(dirname "$AIME26_JSONL")"
    wget -O "$AIME26_JSONL" \
        "https://huggingface.co/datasets/math-ai/aime26/resolve/main/aime2026.jsonl"
fi

if [ ! -f "$AIME26_JSONL" ]; then
    echo "ERROR: AIME26 jsonl not found: $AIME26_JSONL"
    exit 1
fi

if [ ! -f "$TRAIN_SRC" ]; then
    mkdir -p "$(dirname "$TRAIN_SRC")"
    wget -O "$TRAIN_SRC" \
        "https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed/resolve/main/en/train-00000-of-00001.parquet"
fi

if [ ! -f "$TRAIN_SRC" ]; then
    echo "ERROR: training file not found: $TRAIN_SRC"
    exit 1
fi

download_hf_repo() {
    local repo="$1"
    local local_dir="$2"
    mkdir -p /workspace/models
    if command -v hf >/dev/null 2>&1; then
        hf download "$repo" --local-dir "$local_dir"
    else
        huggingface-cli download "$repo" \
            --local-dir "$local_dir" \
            --local-dir-use-symlinks False
    fi
}

if [ ! -f "$STUDENT_MODEL/config.json" ]; then
    rm -rf "$STUDENT_MODEL"
    download_hf_repo "$STUDENT_MODEL_REPO" "$STUDENT_MODEL"
fi

if [ ! -f "$TEACHER_MODEL/config.json" ]; then
    rm -rf "$TEACHER_MODEL"
    download_hf_repo "$TEACHER_MODEL_REPO" "$TEACHER_MODEL"
fi

python3 - <<PY
import os
import wandb

key = os.environ.get("WANDB_API_KEY")
if not key:
    raise RuntimeError("WANDB_API_KEY is empty")
wandb.login(key=key, relogin=True)
PY

python3 - <<'PY'
import json
import os
from pathlib import Path

import pandas as pd


def pick_text(x):
    return (
        x.get("prompt")
        or x.get("problem")
        or x.get("question")
        or x.get("input")
        or x.get("query")
    )


def pick_answer(x):
    reward_model = x.get("reward_model")
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
        return reward_model.get("ground_truth")

    return (
        x.get("answer")
        or x.get("final_answer")
        or x.get("target")
        or x.get("gt")
        or x.get("ground_truth")
        or x.get("solution")
    )


def json_safe(v):
    if hasattr(v, "tolist"):
        return json_safe(v.tolist())
    if isinstance(v, dict):
        return {str(k): json_safe(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(item) for item in v]
    return v


def normalize_prompt(v):
    v = json_safe(v)
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        return [{"role": "user", "content": v}]
    return [{"role": "user", "content": str(v)}]


def normalize_parquet(src, dst, data_source, split):
    src = Path(src)
    dst = Path(dst)

    df = pd.read_parquet(src)
    rows = []

    for i, row in df.iterrows():
        x = row.to_dict()

        problem = pick_text(x)
        answer = pick_answer(x)

        if problem is None:
            raise KeyError(f"{src} row {i} has no prompt/problem/question/input/query field. keys={list(x.keys())}")

        reward_model = json_safe(x.get("reward_model"))
        if not isinstance(reward_model, dict):
            reward_model = {
                "style": "rule",
                "ground_truth": "" if answer is None else str(answer),
            }

        extra_info = json_safe(x.get("extra_info"))
        if not isinstance(extra_info, dict):
            extra_info = {}

        extra_info.update({
            "split": split,
            "index": int(i),
            "answer": "" if answer is None else str(answer),
            "problem": problem if isinstance(problem, str) else json.dumps(json_safe(problem), ensure_ascii=False),
        })

        rows.append({
            "data_source": x.get("data_source", data_source),
            "prompt": normalize_prompt(problem),
            "ability": x.get("ability", "math"),
            "reward_model": reward_model,
            "extra_info": extra_info,
        })

    out = pd.DataFrame(rows)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dst, index=False)

    print("saved:", dst)
    print("rows:", len(out))
    print("columns:", list(out.columns))
    print(out.head(1).to_dict("records")[0])


def normalize_jsonl(src, dst, data_source, split):
    src = Path(src)
    dst = Path(dst)

    rows = []
    with src.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue

            x = json.loads(line)
            problem = pick_text(x)
            answer = pick_answer(x)

            if problem is None:
                raise KeyError(f"{src} row {i} has no prompt/problem/question/input/query field. keys={list(x.keys())}")

            rows.append({
                "data_source": data_source,
                "prompt": normalize_prompt(problem),
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": "" if answer is None else str(answer),
                },
                "extra_info": {
                    "split": split,
                    "index": i,
                    "answer": "" if answer is None else str(answer),
                    "problem": problem if isinstance(problem, str) else json.dumps(json_safe(problem), ensure_ascii=False),
                },
            })

    out = pd.DataFrame(rows)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dst, index=False)

    print("saved:", dst)
    print("rows:", len(out))
    print("columns:", list(out.columns))
    print(out.head(1).to_dict("records")[0])


normalize_parquet(
    os.environ["TRAIN_SRC"],
    os.environ["TRAIN_FILE"],
    data_source="dapo_math_17k",
    split="train",
)

normalize_jsonl(
    os.environ["AIME26_JSONL"],
    os.environ["AIME26_PARQUET"],
    data_source="aime26",
    split="test",
)
PY

ray stop --force || true
rm -rf /workspace/ray_tmp
mkdir -p /workspace/ray_tmp

G_OPD_EXPERIMENT_NAME="${G_OPD_EXPERIMENT_NAME:-146_qwen3_1.7b_teacher_qwen3_4b_vanilla_opd}"
G_OPD_LR="${G_OPD_LR:-2e-6}"
G_OPD_SAVE_FREQ="${G_OPD_SAVE_FREQ:-50}"
G_OPD_DEFAULT_CKPT_DIR="/G-OPD-checkpoints/${G_OPD_EXPERIMENT_NAME}_save_step_${G_OPD_SAVE_FREQ}"
G_OPD_CKPT_DIR="${G_OPD_CKPT_DIR:-$G_OPD_DEFAULT_CKPT_DIR}"
G_OPD_RESUME_MODE="${G_OPD_RESUME_MODE:-disable}"
G_OPD_RESUME_FROM_PATH="${G_OPD_RESUME_FROM_PATH:-null}"

case "$G_OPD_RESUME_MODE" in
    auto|disable|resume_path) ;;
    *)
        echo "ERROR: G_OPD_RESUME_MODE must be one of: auto, disable, resume_path"
        exit 1
        ;;
esac

GPT_ROLLOUT_SCORE_ENABLE="${GPT_ROLLOUT_SCORE_ENABLE:-False}"
GPT_ROLLOUT_SCORE_MODEL="${GPT_ROLLOUT_SCORE_MODEL:-gpt-5.4-mini}"
GPT_ROLLOUT_SCORE_REASONING_EFFORT="${GPT_ROLLOUT_SCORE_REASONING_EFFORT:-none}"
GPT_ROLLOUT_SCORE_MAX_WORKERS="${GPT_ROLLOUT_SCORE_MAX_WORKERS:-128}"
GPT_ROLLOUT_SCORE_TIMEOUT="${GPT_ROLLOUT_SCORE_TIMEOUT:-60}"
GPT_ROLLOUT_SCORE_RETRIES="${GPT_ROLLOUT_SCORE_RETRIES:-2}"
GPT_ROLLOUT_SCORE_MAX_PROMPT_CHARS="${GPT_ROLLOUT_SCORE_MAX_PROMPT_CHARS:-2048}"
GPT_ROLLOUT_SCORE_MAX_RESPONSE_CHARS="${GPT_ROLLOUT_SCORE_MAX_RESPONSE_CHARS:-2048}"
GPT_ROLLOUT_SCORE_MAX_OUTPUT_TOKENS="${GPT_ROLLOUT_SCORE_MAX_OUTPUT_TOKENS:-768}"
GPT_ROLLOUT_SCORE_HINT_RETRIES="${GPT_ROLLOUT_SCORE_HINT_RETRIES:-0}"
GPT_ROLLOUT_SCORE_HINT_MAX_OUTPUT_TOKENS="${GPT_ROLLOUT_SCORE_HINT_MAX_OUTPUT_TOKENS:-256}"
GPT_ROLLOUT_SCORE_REROLL_MAX_OUTPUT_TOKENS="${GPT_ROLLOUT_SCORE_REROLL_MAX_OUTPUT_TOKENS:-512}"
GPT_ROLLOUT_SCORE_MIN_SCORE_100="${GPT_ROLLOUT_SCORE_MIN_SCORE_100:-50}"
GPT_ROLLOUT_SCORE_MAX_REROLLOUT_ATTEMPTS="${GPT_ROLLOUT_SCORE_MAX_REROLLOUT_ATTEMPTS:-1}"
GPT_ROLLOUT_SCORE_MAX_REROLL_CONTEXT_TOKENS="${GPT_ROLLOUT_SCORE_MAX_REROLL_CONTEXT_TOKENS:-256}"
GPT_ROLLOUT_SCORE_ORIG_LOSS_WEIGHT="${GPT_ROLLOUT_SCORE_ORIG_LOSS_WEIGHT:-1.0}"
GPT_ROLLOUT_SCORE_REROLL_HINT_LOSS_WEIGHT="${GPT_ROLLOUT_SCORE_REROLL_HINT_LOSS_WEIGHT:-0.5}"
GPT_ROLLOUT_SCORE_REROLL_APPEND_REQUIRE_IMPROVEMENT="${GPT_ROLLOUT_SCORE_REROLL_APPEND_REQUIRE_IMPROVEMENT:-True}"
GPT_ROLLOUT_SCORE_REROLL_APPEND_MIN_GAIN="${GPT_ROLLOUT_SCORE_REROLL_APPEND_MIN_GAIN:-20.0}"
GPT_ROLLOUT_SCORE_REROLL_APPEND_MIN_SCORE="${GPT_ROLLOUT_SCORE_REROLL_APPEND_MIN_SCORE:-75.0}"
GPT_ROLLOUT_SCORE_REROLL_SFT_ENABLE="${GPT_ROLLOUT_SCORE_REROLL_SFT_ENABLE:-False}"
GPT_ROLLOUT_SCORE_REROLL_SFT_KEEP_HINT_OPD="${GPT_ROLLOUT_SCORE_REROLL_SFT_KEEP_HINT_OPD:-False}"
GPT_ROLLOUT_SCORE_REROLL_SFT_LAMBDA="${GPT_ROLLOUT_SCORE_REROLL_SFT_LAMBDA:-0.05}"
GPT_ROLLOUT_SCORE_REROLL_SFT_ALPHA="${GPT_ROLLOUT_SCORE_REROLL_SFT_ALPHA:-0.5}"
GPT_ROLLOUT_SCORE_REROLL_SFT_Z_CLIP="${GPT_ROLLOUT_SCORE_REROLL_SFT_Z_CLIP:-2.0}"
GPT_ROLLOUT_SCORE_REROLL_SFT_WEIGHT_MIN="${GPT_ROLLOUT_SCORE_REROLL_SFT_WEIGHT_MIN:-0.1}"
GPT_ROLLOUT_SCORE_REROLL_SFT_WEIGHT_MAX="${GPT_ROLLOUT_SCORE_REROLL_SFT_WEIGHT_MAX:-4.0}"
GPT_ROLLOUT_SCORE_REROLL_SFT_NORMALIZE_WEIGHTS="${GPT_ROLLOUT_SCORE_REROLL_SFT_NORMALIZE_WEIGHTS:-True}"
GPT_ROLLOUT_SCORE_REROLL_SFT_STD_FLOOR="${GPT_ROLLOUT_SCORE_REROLL_SFT_STD_FLOOR:-1e-6}"
GPT_ROLLOUT_SCORE_REROLL_SFT_SCORE_COEF="${GPT_ROLLOUT_SCORE_REROLL_SFT_SCORE_COEF:-0.5}"
GPT_ROLLOUT_SCORE_REROLL_SFT_GAIN_COEF="${GPT_ROLLOUT_SCORE_REROLL_SFT_GAIN_COEF:-0.5}"
GPT_ROLLOUT_SCORE_REROLL_NOHINT_ENABLE="${GPT_ROLLOUT_SCORE_REROLL_NOHINT_ENABLE:-True}"
GPT_ROLLOUT_SCORE_REROLL_NOHINT_MIN_SCORE="${GPT_ROLLOUT_SCORE_REROLL_NOHINT_MIN_SCORE:-50}"
GPT_ROLLOUT_SCORE_REROLL_NOHINT_MIN_GAIN="${GPT_ROLLOUT_SCORE_REROLL_NOHINT_MIN_GAIN:-10}"
GPT_ROLLOUT_SCORE_REROLL_NOHINT_MAX_WEIGHT="${GPT_ROLLOUT_SCORE_REROLL_NOHINT_MAX_WEIGHT:-0.5}"
GPT_ROLLOUT_SCORE_REROLL_NOHINT_GAIN_NORM="${GPT_ROLLOUT_SCORE_REROLL_NOHINT_GAIN_NORM:-50}"
GPT_ROLLOUT_SCORE_RUBRIC_ADV_SHIFT_ENABLE="${GPT_ROLLOUT_SCORE_RUBRIC_ADV_SHIFT_ENABLE:-False}"
GPT_ROLLOUT_SCORE_RUBRIC_ADV_SHIFT_MODE="${GPT_ROLLOUT_SCORE_RUBRIC_ADV_SHIFT_MODE:-rank_residual}"
GPT_ROLLOUT_SCORE_RUBRIC_ADV_SHIFT_COEF="${GPT_ROLLOUT_SCORE_RUBRIC_ADV_SHIFT_COEF:-0.10}"
GPT_ROLLOUT_SCORE_RUBRIC_ADV_SHIFT_CLIP="${GPT_ROLLOUT_SCORE_RUBRIC_ADV_SHIFT_CLIP:-0.20}"
GPT_ROLLOUT_SCORE_HISTORY_NUM_BINS="${GPT_ROLLOUT_SCORE_HISTORY_NUM_BINS:-12}"
GPT_ROLLOUT_SCORE_HISTORY_BUCKET_MODE="${GPT_ROLLOUT_SCORE_HISTORY_BUCKET_MODE:-label}"
GPT_ROLLOUT_SCORE_HISTORY_RANDOM_BUCKET_SEED="${GPT_ROLLOUT_SCORE_HISTORY_RANDOM_BUCKET_SEED:-42}"
GPT_ROLLOUT_SCORE_HISTORY_SIZE="${GPT_ROLLOUT_SCORE_HISTORY_SIZE:-2048}"
GPT_ROLLOUT_SCORE_HISTORY_WARMUP_STEPS="${GPT_ROLLOUT_SCORE_HISTORY_WARMUP_STEPS:-5}"
GPT_ROLLOUT_SCORE_HISTORY_MIN_BIN_COUNT="${GPT_ROLLOUT_SCORE_HISTORY_MIN_BIN_COUNT:-64}"
GPT_ROLLOUT_SCORE_HISTORY_GLOBAL_MIN_COUNT="${GPT_ROLLOUT_SCORE_HISTORY_GLOBAL_MIN_COUNT:-256}"
GPT_ROLLOUT_SCORE_HISTORY_STD_FLOOR="${GPT_ROLLOUT_SCORE_HISTORY_STD_FLOOR:-8.0}"
GPT_ROLLOUT_SCORE_HISTORY_Z_CLIP="${GPT_ROLLOUT_SCORE_HISTORY_Z_CLIP:-2.0}"
GPT_ROLLOUT_SCORE_HISTORY_NEGATIVE_COEF_SCALE="${GPT_ROLLOUT_SCORE_HISTORY_NEGATIVE_COEF_SCALE:-0.8}"
GPT_ROLLOUT_SCORE_RANK_GAP_DROP_ENABLE="${GPT_ROLLOUT_SCORE_RANK_GAP_DROP_ENABLE:-True}"
GPT_ROLLOUT_SCORE_RANK_GAP_DROP_THRESHOLD="${GPT_ROLLOUT_SCORE_RANK_GAP_DROP_THRESHOLD:-0.75}"
GPT_ROLLOUT_SCORE_RANK_GAP_DROP_SCOPE="${GPT_ROLLOUT_SCORE_RANK_GAP_DROP_SCOPE:-all}"
GPT_ROLLOUT_SCORE_VERBOSE="${GPT_ROLLOUT_SCORE_VERBOSE:-1}"
G_OPD_LOG_DIR="${G_OPD_LOG_DIR:-/workspace/G-OPD-logs/${G_OPD_EXPERIMENT_NAME}_save_step_${G_OPD_SAVE_FREQ}}"
GPT_ROLLOUT_DATA_DIR="${GPT_ROLLOUT_DATA_DIR:-$G_OPD_LOG_DIR/rollout_data}"
GPT_ROLLOUT_SCORE_CASE_STUDY_DIR="${GPT_ROLLOUT_SCORE_CASE_STUDY_DIR:-$G_OPD_LOG_DIR/gpt_low_score_cases}"
GPT_ROLLOUT_SCORE_RANK_GAP_CASE_STUDY_DIR="${GPT_ROLLOUT_SCORE_RANK_GAP_CASE_STUDY_DIR:-$G_OPD_LOG_DIR/gpt_rank_gap_cases}"
GPT_ROLLOUT_SCORE_CASE_STUDY_MAX_PER_STEP="${GPT_ROLLOUT_SCORE_CASE_STUDY_MAX_PER_STEP:-20}"
GPT_ROLLOUT_SCORE_CASE_STUDY_THRESHOLD_100="${GPT_ROLLOUT_SCORE_CASE_STUDY_THRESHOLD_100:-$GPT_ROLLOUT_SCORE_MIN_SCORE_100}"
GPT_ROLLOUT_SCORE_CASE_STUDY_INCLUDE_ERRORS="${GPT_ROLLOUT_SCORE_CASE_STUDY_INCLUDE_ERRORS:-True}"
RUBRIC_PROBE_DATA_ENABLE="${RUBRIC_PROBE_DATA_ENABLE:-False}"
RUBRIC_PROBE_DATA_DIR="${RUBRIC_PROBE_DATA_DIR:-$G_OPD_LOG_DIR/rubric_probe_data}"
RUBRIC_PROBE_HIDDEN_DTYPE="${RUBRIC_PROBE_HIDDEN_DTYPE:-float16}"
RUBRIC_PROBE_STUDENT_HIDDEN_SIZE="${RUBRIC_PROBE_STUDENT_HIDDEN_SIZE:-2048}"
RUBRIC_PROBE_TEACHER_HIDDEN_SIZE="${RUBRIC_PROBE_TEACHER_HIDDEN_SIZE:-2560}"
RUBRIC_PROBE_PROMPT_VERSION="${RUBRIC_PROBE_PROMPT_VERSION:-math_7rubric_v1}"

case "$GPT_ROLLOUT_SCORE_ENABLE" in
    True|true|1|yes|YES)
        if [ -z "${OPENAI_API_KEY:-}" ]; then
            echo "ERROR: OPENAI_API_KEY is required when GPT rollout scoring is enabled"
            exit 1
        fi
        ;;
esac

python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        algorithm.rollout_correction.rollout_is=token \
        algorithm.rollout_correction.rollout_is_threshold=5.0 \
        algorithm.rollout_correction.rollout_rs=null \
        algorithm.rollout_correction.bypass_mode=false \
        actor_rollout_ref.rollout.calculate_log_probs=true \
        data.train_files="$TRAIN_FILE" \
        data.val_files="$AIME26_PARQUET" \
        data.train_batch_size=128 \
        data.max_prompt_length=2048 \
        data.max_response_length="${G_OPD_MAX_RESPONSE_LENGTH:-2048}" \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.shuffle=True \
        data.seed=42 \
        data.return_raw_chat=True \
        +data.apply_chat_template_kwargs.enable_thinking=False \
        actor_rollout_ref.model.path="$STUDENT_MODEL" \
        +actor_rollout_ref.ref.model.path="$TEACHER_MODEL" \
        actor_rollout_ref.actor.optim.lr="$G_OPD_LR" \
        data.filter_overlong_prompts_workers=4 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True \
        actor_rollout_ref.actor.policy_loss.lambda_vals=1.0 \
        actor_rollout_ref.actor.policy_loss.multi_teacher_distill=False \
        actor_rollout_ref.actor.ppo_mini_batch_size=128 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.actor.checkpoint.save_contents='["model"]' \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
        actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        reward_model.reward_manager=naive \
        trainer.rubric_probe_data.enable="$RUBRIC_PROBE_DATA_ENABLE" \
        trainer.rubric_probe_data.output_dir="$RUBRIC_PROBE_DATA_DIR" \
        trainer.rubric_probe_data.hidden_dtype="$RUBRIC_PROBE_HIDDEN_DTYPE" \
        trainer.rubric_probe_data.expected_student_hidden_size="$RUBRIC_PROBE_STUDENT_HIDDEN_SIZE" \
        trainer.rubric_probe_data.expected_teacher_hidden_size="$RUBRIC_PROBE_TEACHER_HIDDEN_SIZE" \
        trainer.rubric_probe_data.rubric_prompt_version="$RUBRIC_PROBE_PROMPT_VERSION" \
        trainer.gpt_rollout_score.enable="$GPT_ROLLOUT_SCORE_ENABLE" \
        trainer.gpt_rollout_score.model="$GPT_ROLLOUT_SCORE_MODEL" \
        trainer.gpt_rollout_score.reasoning_effort="$GPT_ROLLOUT_SCORE_REASONING_EFFORT" \
        trainer.gpt_rollout_score.max_workers="$GPT_ROLLOUT_SCORE_MAX_WORKERS" \
        trainer.gpt_rollout_score.timeout="$GPT_ROLLOUT_SCORE_TIMEOUT" \
        trainer.gpt_rollout_score.retries="$GPT_ROLLOUT_SCORE_RETRIES" \
        trainer.gpt_rollout_score.max_prompt_chars="$GPT_ROLLOUT_SCORE_MAX_PROMPT_CHARS" \
        trainer.gpt_rollout_score.max_response_chars="$GPT_ROLLOUT_SCORE_MAX_RESPONSE_CHARS" \
        trainer.gpt_rollout_score.max_output_tokens="$GPT_ROLLOUT_SCORE_MAX_OUTPUT_TOKENS" \
        trainer.gpt_rollout_score.hint_retries="$GPT_ROLLOUT_SCORE_HINT_RETRIES" \
        trainer.gpt_rollout_score.hint_max_output_tokens="$GPT_ROLLOUT_SCORE_HINT_MAX_OUTPUT_TOKENS" \
        trainer.gpt_rollout_score.reroll_max_output_tokens="$GPT_ROLLOUT_SCORE_REROLL_MAX_OUTPUT_TOKENS" \
        trainer.gpt_rollout_score.min_score_100="$GPT_ROLLOUT_SCORE_MIN_SCORE_100" \
        trainer.gpt_rollout_score.max_rerollout_attempts="$GPT_ROLLOUT_SCORE_MAX_REROLLOUT_ATTEMPTS" \
        trainer.gpt_rollout_score.max_reroll_context_tokens="$GPT_ROLLOUT_SCORE_MAX_REROLL_CONTEXT_TOKENS" \
        trainer.gpt_rollout_score.orig_loss_weight="$GPT_ROLLOUT_SCORE_ORIG_LOSS_WEIGHT" \
        trainer.gpt_rollout_score.reroll_hint_loss_weight="$GPT_ROLLOUT_SCORE_REROLL_HINT_LOSS_WEIGHT" \
        trainer.gpt_rollout_score.reroll_append_require_improvement="$GPT_ROLLOUT_SCORE_REROLL_APPEND_REQUIRE_IMPROVEMENT" \
        trainer.gpt_rollout_score.reroll_append_min_gain="$GPT_ROLLOUT_SCORE_REROLL_APPEND_MIN_GAIN" \
        trainer.gpt_rollout_score.reroll_append_min_score="$GPT_ROLLOUT_SCORE_REROLL_APPEND_MIN_SCORE" \
        trainer.gpt_rollout_score.reroll_sft_enable="$GPT_ROLLOUT_SCORE_REROLL_SFT_ENABLE" \
        trainer.gpt_rollout_score.reroll_sft_keep_hint_opd="$GPT_ROLLOUT_SCORE_REROLL_SFT_KEEP_HINT_OPD" \
        trainer.gpt_rollout_score.reroll_sft_lambda="$GPT_ROLLOUT_SCORE_REROLL_SFT_LAMBDA" \
        trainer.gpt_rollout_score.reroll_sft_alpha="$GPT_ROLLOUT_SCORE_REROLL_SFT_ALPHA" \
        trainer.gpt_rollout_score.reroll_sft_z_clip="$GPT_ROLLOUT_SCORE_REROLL_SFT_Z_CLIP" \
        trainer.gpt_rollout_score.reroll_sft_weight_min="$GPT_ROLLOUT_SCORE_REROLL_SFT_WEIGHT_MIN" \
        trainer.gpt_rollout_score.reroll_sft_weight_max="$GPT_ROLLOUT_SCORE_REROLL_SFT_WEIGHT_MAX" \
        trainer.gpt_rollout_score.reroll_sft_normalize_weights="$GPT_ROLLOUT_SCORE_REROLL_SFT_NORMALIZE_WEIGHTS" \
        trainer.gpt_rollout_score.reroll_sft_std_floor="$GPT_ROLLOUT_SCORE_REROLL_SFT_STD_FLOOR" \
        trainer.gpt_rollout_score.reroll_sft_score_coef="$GPT_ROLLOUT_SCORE_REROLL_SFT_SCORE_COEF" \
        trainer.gpt_rollout_score.reroll_sft_gain_coef="$GPT_ROLLOUT_SCORE_REROLL_SFT_GAIN_COEF" \
        trainer.gpt_rollout_score.reroll_nohint_enable="$GPT_ROLLOUT_SCORE_REROLL_NOHINT_ENABLE" \
        trainer.gpt_rollout_score.reroll_nohint_min_score="$GPT_ROLLOUT_SCORE_REROLL_NOHINT_MIN_SCORE" \
        trainer.gpt_rollout_score.reroll_nohint_min_gain="$GPT_ROLLOUT_SCORE_REROLL_NOHINT_MIN_GAIN" \
        trainer.gpt_rollout_score.reroll_nohint_max_weight="$GPT_ROLLOUT_SCORE_REROLL_NOHINT_MAX_WEIGHT" \
        trainer.gpt_rollout_score.reroll_nohint_gain_norm="$GPT_ROLLOUT_SCORE_REROLL_NOHINT_GAIN_NORM" \
        trainer.gpt_rollout_score.rubric_adv_shift_enable="$GPT_ROLLOUT_SCORE_RUBRIC_ADV_SHIFT_ENABLE" \
        trainer.gpt_rollout_score.rubric_adv_shift_mode="$GPT_ROLLOUT_SCORE_RUBRIC_ADV_SHIFT_MODE" \
        trainer.gpt_rollout_score.rubric_adv_shift_coef="$GPT_ROLLOUT_SCORE_RUBRIC_ADV_SHIFT_COEF" \
        trainer.gpt_rollout_score.rubric_adv_shift_clip="$GPT_ROLLOUT_SCORE_RUBRIC_ADV_SHIFT_CLIP" \
        trainer.gpt_rollout_score.history_num_bins="$GPT_ROLLOUT_SCORE_HISTORY_NUM_BINS" \
        trainer.gpt_rollout_score.history_bucket_mode="$GPT_ROLLOUT_SCORE_HISTORY_BUCKET_MODE" \
        trainer.gpt_rollout_score.history_random_bucket_seed="$GPT_ROLLOUT_SCORE_HISTORY_RANDOM_BUCKET_SEED" \
        trainer.gpt_rollout_score.history_size="$GPT_ROLLOUT_SCORE_HISTORY_SIZE" \
        trainer.gpt_rollout_score.history_warmup_steps="$GPT_ROLLOUT_SCORE_HISTORY_WARMUP_STEPS" \
        trainer.gpt_rollout_score.history_min_bin_count="$GPT_ROLLOUT_SCORE_HISTORY_MIN_BIN_COUNT" \
        trainer.gpt_rollout_score.history_global_min_count="$GPT_ROLLOUT_SCORE_HISTORY_GLOBAL_MIN_COUNT" \
        trainer.gpt_rollout_score.history_std_floor="$GPT_ROLLOUT_SCORE_HISTORY_STD_FLOOR" \
        trainer.gpt_rollout_score.history_z_clip="$GPT_ROLLOUT_SCORE_HISTORY_Z_CLIP" \
        trainer.gpt_rollout_score.history_negative_coef_scale="$GPT_ROLLOUT_SCORE_HISTORY_NEGATIVE_COEF_SCALE" \
        trainer.gpt_rollout_score.rank_gap_drop_enable="$GPT_ROLLOUT_SCORE_RANK_GAP_DROP_ENABLE" \
        trainer.gpt_rollout_score.rank_gap_drop_threshold="$GPT_ROLLOUT_SCORE_RANK_GAP_DROP_THRESHOLD" \
        trainer.gpt_rollout_score.rank_gap_drop_scope="$GPT_ROLLOUT_SCORE_RANK_GAP_DROP_SCOPE" \
        trainer.gpt_rollout_score.rank_gap_case_study_dir="$GPT_ROLLOUT_SCORE_RANK_GAP_CASE_STUDY_DIR" \
        trainer.gpt_rollout_score.case_study_dir="$GPT_ROLLOUT_SCORE_CASE_STUDY_DIR" \
        trainer.gpt_rollout_score.case_study_max_per_step="$GPT_ROLLOUT_SCORE_CASE_STUDY_MAX_PER_STEP" \
        trainer.gpt_rollout_score.case_study_threshold_100="$GPT_ROLLOUT_SCORE_CASE_STUDY_THRESHOLD_100" \
        trainer.gpt_rollout_score.case_study_include_errors="$GPT_ROLLOUT_SCORE_CASE_STUDY_INCLUDE_ERRORS" \
        trainer.gpt_rollout_score.verbose="$GPT_ROLLOUT_SCORE_VERBOSE" \
        trainer.rollout_data_dir="$GPT_ROLLOUT_DATA_DIR" \
        trainer.critic_warmup=0 \
        trainer.val_before_train=True \
        trainer.logger='["console","wandb"]' \
        trainer.log_val_generations=10 \
        trainer.project_name='on-policy-distillation' \
        trainer.experiment_name="$G_OPD_EXPERIMENT_NAME" \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        trainer.save_freq="$G_OPD_SAVE_FREQ" \
        trainer.resume_mode="$G_OPD_RESUME_MODE" \
        trainer.resume_from_path="$G_OPD_RESUME_FROM_PATH" \
        trainer.default_local_dir="$G_OPD_CKPT_DIR" \
        trainer.max_actor_ckpt_to_keep=1 \
        trainer.max_critic_ckpt_to_keep=1 \
        trainer.test_freq=30 \
        trainer.total_epochs=1 \
        "$@"
