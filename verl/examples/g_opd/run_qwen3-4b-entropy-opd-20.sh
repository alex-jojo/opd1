#!/usr/bin/env bash
set -euo pipefail

if [ "${G_OPD_SHELL_DEBUG:-0}" = "1" ]; then
    set -x
fi

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE="${WANDB_MODE:-online}"
export USED_MODEL="${USED_MODEL:-no_api}"
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export HYDRA_FULL_ERROR=1
export VERL_PRINT_CONFIG="${VERL_PRINT_CONFIG:-0}"
export G_OPD_PROGRESS_DEBUG="${G_OPD_PROGRESS_DEBUG:-1}"
export GPT_ROLLOUT_SCORE_VERBOSE="${GPT_ROLLOUT_SCORE_VERBOSE:-1}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

cd /workspace/opd1/verl

G_OPD_ENV_FILE="${G_OPD_ENV_FILE:-/workspace/opd1/verl/.env}"
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

TRAIN_SRC="${TRAIN_SRC:-/workspace/opd1/data/train-00000-of-00001.parquet}"
TRAIN_FILE="${TRAIN_FILE:-/workspace/opd1/verl/train_verl.parquet}"
export TRAIN_SRC TRAIN_FILE

AIME26_JSONL="${AIME26_JSONL:-/workspace/G-OPD/data/aime26/test.jsonl}"
AIME26_PARQUET="${AIME26_PARQUET:-/workspace/G-OPD/data/aime26/test_verl.parquet}"
export AIME26_JSONL AIME26_PARQUET

STUDENT_MODEL="${STUDENT_MODEL:-/workspace/models/Qwen3-1.7B}"
TEACHER_MODEL="${TEACHER_MODEL:-/workspace/models/Qwen3-4B-Non-Thinking-RL-Math-Step500}"

ENTROPY_OPD_FRACTION="${ENTROPY_OPD_FRACTION:-0.2}"
ENTROPY_OPD_VERBOSE="${ENTROPY_OPD_VERBOSE:-1}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

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
    download_hf_repo Qwen/Qwen3-1.7B "$STUDENT_MODEL"
fi

if [ ! -f "$TEACHER_MODEL/config.json" ]; then
    rm -rf "$TEACHER_MODEL"
    download_hf_repo Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500 "$TEACHER_MODEL"
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
    python3 - <<'PY'
import os
import wandb

wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
PY
else
    echo "[warn] WANDB_API_KEY is empty. Set WANDB_MODE=offline or provide a key if wandb logging fails."
fi

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


def normalize_prompt(v):
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

        reward_model = x.get("reward_model")
        if not isinstance(reward_model, dict):
            reward_model = {
                "style": "rule",
                "ground_truth": "" if answer is None else str(answer),
            }

        extra_info = x.get("extra_info")
        if not isinstance(extra_info, dict):
            extra_info = {}

        extra_info.update({
            "split": split,
            "index": int(i),
            "answer": "" if answer is None else str(answer),
            "problem": problem if isinstance(problem, str) else json.dumps(problem, ensure_ascii=False),
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
                    "problem": problem if isinstance(problem, str) else json.dumps(problem, ensure_ascii=False),
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

G_OPD_EXPERIMENT_NAME="${G_OPD_EXPERIMENT_NAME:-146_qwen3_1.7b_teacher_qwen3_4b_entropy_opd_20}"
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
GPT_ROLLOUT_SCORE_MAX_RESPONSE_CHARS="${GPT_ROLLOUT_SCORE_MAX_RESPONSE_CHARS:-4096}"
GPT_ROLLOUT_SCORE_MAX_OUTPUT_TOKENS="${GPT_ROLLOUT_SCORE_MAX_OUTPUT_TOKENS:-1024}"
GPT_ROLLOUT_SCORE_MIN_SCORE_100="${GPT_ROLLOUT_SCORE_MIN_SCORE_100:-50}"
GPT_ROLLOUT_SCORE_MAX_REROLLOUT_ATTEMPTS="${GPT_ROLLOUT_SCORE_MAX_REROLLOUT_ATTEMPTS:-1}"
GPT_ROLLOUT_SCORE_MAX_REROLL_CONTEXT_TOKENS="${GPT_ROLLOUT_SCORE_MAX_REROLL_CONTEXT_TOKENS:-256}"
GPT_ROLLOUT_SCORE_REROLL_SUMMARY_MODEL="${GPT_ROLLOUT_SCORE_REROLL_SUMMARY_MODEL:-$GPT_ROLLOUT_SCORE_MODEL}"
GPT_ROLLOUT_SCORE_REROLL_SUMMARY_MAX_OUTPUT_TOKENS="${GPT_ROLLOUT_SCORE_REROLL_SUMMARY_MAX_OUTPUT_TOKENS:-1024}"
GPT_ROLLOUT_SCORE_VERBOSE="${GPT_ROLLOUT_SCORE_VERBOSE:-1}"
G_OPD_LOG_DIR="${G_OPD_LOG_DIR:-/workspace/G-OPD-logs/${G_OPD_EXPERIMENT_NAME}_save_step_${G_OPD_SAVE_FREQ}}"
GPT_ROLLOUT_DATA_DIR="${GPT_ROLLOUT_DATA_DIR:-$G_OPD_LOG_DIR/rollout_data}"
GPT_ROLLOUT_SCORE_CASE_STUDY_DIR="${GPT_ROLLOUT_SCORE_CASE_STUDY_DIR:-$G_OPD_LOG_DIR/gpt_low_score_cases}"
GPT_ROLLOUT_SCORE_CASE_STUDY_MAX_PER_STEP="${GPT_ROLLOUT_SCORE_CASE_STUDY_MAX_PER_STEP:-20}"
GPT_ROLLOUT_SCORE_CASE_STUDY_THRESHOLD_100="${GPT_ROLLOUT_SCORE_CASE_STUDY_THRESHOLD_100:-$GPT_ROLLOUT_SCORE_MIN_SCORE_100}"
GPT_ROLLOUT_SCORE_CASE_STUDY_INCLUDE_ERRORS="${GPT_ROLLOUT_SCORE_CASE_STUDY_INCLUDE_ERRORS:-True}"

case "$GPT_ROLLOUT_SCORE_ENABLE" in
    True|true|1|yes|YES)
        if [ -z "${OPENAI_API_KEY:-}" ]; then
            echo "ERROR: OPENAI_API_KEY is required when GPT rollout scoring is enabled"
            exit 1
        fi
        ;;
esac

echo "[entropy-opd] student=$STUDENT_MODEL"
echo "[entropy-opd] teacher=$TEACHER_MODEL"
echo "[entropy-opd] top_fraction=$ENTROPY_OPD_FRACTION"

export ENTROPY_OPD_ENABLE=1
export ENTROPY_OPD_FRACTION
export ENTROPY_OPD_VERBOSE

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
        data.max_response_length=4096 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.shuffle=True \
        data.seed=42 \
        data.return_raw_chat=True \
        +data.apply_chat_template_kwargs.enable_thinking=False \
        actor_rollout_ref.model.path="$STUDENT_MODEL" \
        +actor_rollout_ref.ref.model.path="$TEACHER_MODEL" \
        actor_rollout_ref.actor.optim.lr=2e-6 \
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
        trainer.gpt_rollout_score.enable="$GPT_ROLLOUT_SCORE_ENABLE" \
        trainer.gpt_rollout_score.model="$GPT_ROLLOUT_SCORE_MODEL" \
        trainer.gpt_rollout_score.reasoning_effort="$GPT_ROLLOUT_SCORE_REASONING_EFFORT" \
        trainer.gpt_rollout_score.max_workers="$GPT_ROLLOUT_SCORE_MAX_WORKERS" \
        trainer.gpt_rollout_score.timeout="$GPT_ROLLOUT_SCORE_TIMEOUT" \
        trainer.gpt_rollout_score.retries="$GPT_ROLLOUT_SCORE_RETRIES" \
        trainer.gpt_rollout_score.max_prompt_chars="$GPT_ROLLOUT_SCORE_MAX_PROMPT_CHARS" \
        trainer.gpt_rollout_score.max_response_chars="$GPT_ROLLOUT_SCORE_MAX_RESPONSE_CHARS" \
        trainer.gpt_rollout_score.max_output_tokens="$GPT_ROLLOUT_SCORE_MAX_OUTPUT_TOKENS" \
        trainer.gpt_rollout_score.min_score_100="$GPT_ROLLOUT_SCORE_MIN_SCORE_100" \
        trainer.gpt_rollout_score.max_rerollout_attempts="$GPT_ROLLOUT_SCORE_MAX_REROLLOUT_ATTEMPTS" \
        trainer.gpt_rollout_score.max_reroll_context_tokens="$GPT_ROLLOUT_SCORE_MAX_REROLL_CONTEXT_TOKENS" \
        trainer.gpt_rollout_score.reroll_summary_model="$GPT_ROLLOUT_SCORE_REROLL_SUMMARY_MODEL" \
        trainer.gpt_rollout_score.reroll_summary_max_output_tokens="$GPT_ROLLOUT_SCORE_REROLL_SUMMARY_MAX_OUTPUT_TOKENS" \
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
