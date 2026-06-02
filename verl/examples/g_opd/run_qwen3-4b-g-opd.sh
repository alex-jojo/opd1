#!/usr/bin/env bash
set -x
set -euo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_API_KEY='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'

export WANDB_MODE=online
export USED_MODEL="no_api"
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export HYDRA_FULL_ERROR=1

cd /workspace/opd1/verl

TRAIN_SRC="${TRAIN_SRC:-/workspace/opd1/data/train-00000-of-00001.parquet}"
TRAIN_FILE="${TRAIN_FILE:-/workspace/opd1/verl/train_verl.parquet}"
export TRAIN_SRC TRAIN_FILE

AIME24_JSONL="/workspace/opd1/data/aime24/test.jsonl"
AIME24_PARQUET="/workspace/opd1/data/aime24/test_verl.parquet"

STUDENT_MODEL="/workspace/models/Qwen3-1.7B"
TEACHER_MODEL="/workspace/models/Qwen3-4B-Non-Thinking-RL-Math-Step500"

if [ ! -f "$TRAIN_SRC" ]; then
    echo "ERROR: training file not found: $TRAIN_SRC"
    exit 1
fi

if [ ! -f "$AIME24_JSONL" ]; then
    echo "ERROR: AIME24 jsonl not found: $AIME24_JSONL"
    exit 1
fi

if [ ! -d "$STUDENT_MODEL" ]; then
    mkdir -p /workspace/models
    huggingface-cli download Qwen/Qwen3-1.7B \
        --local-dir "$STUDENT_MODEL" \
        --local-dir-use-symlinks False
fi

if [ ! -d "$TEACHER_MODEL" ]; then
    mkdir -p /workspace/models
    huggingface-cli download Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500 \
        --local-dir "$TEACHER_MODEL" \
        --local-dir-use-symlinks False
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
    "/workspace/opd1/data/aime24/test.jsonl",
    "/workspace/opd1/data/aime24/test_verl.parquet",
    data_source="aime24",
    split="test",
)
PY

test_files="['$AIME24_PARQUET']"

ray stop --force || true
rm -rf /workspace/ray_tmp
mkdir -p /workspace/ray_tmp

GPT_ROLLOUT_SCORE_ENABLE="${GPT_ROLLOUT_SCORE_ENABLE:-True}"
GPT_ROLLOUT_SCORE_MODEL="${GPT_ROLLOUT_SCORE_MODEL:-gpt-4.1-mini}"
GPT_ROLLOUT_SCORE_MAX_WORKERS="${GPT_ROLLOUT_SCORE_MAX_WORKERS:-8}"
GPT_ROLLOUT_SCORE_TIMEOUT="${GPT_ROLLOUT_SCORE_TIMEOUT:-60}"
GPT_ROLLOUT_SCORE_RETRIES="${GPT_ROLLOUT_SCORE_RETRIES:-2}"
GPT_ROLLOUT_SCORE_MAX_PROMPT_CHARS="${GPT_ROLLOUT_SCORE_MAX_PROMPT_CHARS:-8000}"
GPT_ROLLOUT_SCORE_MAX_RESPONSE_CHARS="${GPT_ROLLOUT_SCORE_MAX_RESPONSE_CHARS:-16000}"
GPT_ROLLOUT_SCORE_MAX_OUTPUT_TOKENS="${GPT_ROLLOUT_SCORE_MAX_OUTPUT_TOKENS:-1024}"
GPT_ROLLOUT_SCORE_MIN_SCORE_100="${GPT_ROLLOUT_SCORE_MIN_SCORE_100:-50}"
GPT_ROLLOUT_SCORE_MAX_REROLLOUT_ATTEMPTS="${GPT_ROLLOUT_SCORE_MAX_REROLLOUT_ATTEMPTS:-1}"
GPT_ROLLOUT_SCORE_MAX_REROLL_FEEDBACK_TOKENS="${GPT_ROLLOUT_SCORE_MAX_REROLL_FEEDBACK_TOKENS:-512}"
GPT_ROLLOUT_DATA_DIR="${GPT_ROLLOUT_DATA_DIR:-/G-OPD-checkpoints/Qwen3-1.7B-Standard-OPD-DAPO-Math-17K/rollout_data}"

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
        data.val_files="$test_files" \
        data.train_batch_size=128 \
        data.max_prompt_length=2048 \
        data.max_response_length=8192 \
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
        +env.rollout.n=8 \
        actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
        actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.n=32 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        reward_model.reward_manager=naive \
        trainer.gpt_rollout_score.enable="$GPT_ROLLOUT_SCORE_ENABLE" \
        trainer.gpt_rollout_score.model="$GPT_ROLLOUT_SCORE_MODEL" \
        trainer.gpt_rollout_score.max_workers="$GPT_ROLLOUT_SCORE_MAX_WORKERS" \
        trainer.gpt_rollout_score.timeout="$GPT_ROLLOUT_SCORE_TIMEOUT" \
        trainer.gpt_rollout_score.retries="$GPT_ROLLOUT_SCORE_RETRIES" \
        trainer.gpt_rollout_score.max_prompt_chars="$GPT_ROLLOUT_SCORE_MAX_PROMPT_CHARS" \
        trainer.gpt_rollout_score.max_response_chars="$GPT_ROLLOUT_SCORE_MAX_RESPONSE_CHARS" \
        trainer.gpt_rollout_score.max_output_tokens="$GPT_ROLLOUT_SCORE_MAX_OUTPUT_TOKENS" \
        trainer.gpt_rollout_score.min_score_100="$GPT_ROLLOUT_SCORE_MIN_SCORE_100" \
        trainer.gpt_rollout_score.max_rerollout_attempts="$GPT_ROLLOUT_SCORE_MAX_REROLLOUT_ATTEMPTS" \
        trainer.gpt_rollout_score.max_reroll_feedback_tokens="$GPT_ROLLOUT_SCORE_MAX_REROLL_FEEDBACK_TOKENS" \
        trainer.rollout_data_dir="$GPT_ROLLOUT_DATA_DIR" \
        trainer.critic_warmup=0 \
        trainer.val_before_train=True \
        trainer.logger='["console","wandb"]' \
        trainer.log_val_generations=10 \
        trainer.project_name='on-policy-distillation' \
        trainer.experiment_name='qwen3_1.7b_non_thinking_teacher_qwen3_4b_non_thinking_rl_math_standard_opd_dapo_math_17k' \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        trainer.save_freq=50 \
        trainer.default_local_dir=/G-OPD-checkpoints/Qwen3-1.7B-Standard-OPD-DAPO-Math-17K \
        trainer.max_actor_ckpt_to_keep=1 \
        trainer.max_critic_ckpt_to_keep=1 \
        trainer.test_freq=10 \
        trainer.total_epochs=1 \
        "$@"
