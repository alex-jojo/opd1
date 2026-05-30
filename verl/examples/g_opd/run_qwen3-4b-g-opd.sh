#!/usr/bin/env bash
set -x
set -euo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}

export WANDB_API_KEY=${WANDB_API_KEY:-""}
export WANDB_MODE=${WANDB_MODE:-online}
export USED_MODEL=${USED_MODEL:-"no_api"}
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export HYDRA_FULL_ERROR=1

cd /workspace/opd1/verl

TRAIN_FILE="/workspace/opd1/verl/train-00000-of-00001.parquet"

AIME24_JSONL="/workspace/opd1/data/aime24/test.jsonl"
AIME24_PARQUET="/workspace/opd1/data/aime24/test_verl.parquet"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "ERROR: training file not found: $TRAIN_FILE"
    exit 1
fi

if [ ! -f "$AIME24_PARQUET" ]; then
    if [ ! -f "$AIME24_JSONL" ]; then
        echo "ERROR: AIME24 jsonl not found: $AIME24_JSONL"
        exit 1
    fi

    python3 - <<PY
import json
import pandas as pd
from pathlib import Path

src = Path("$AIME24_JSONL")
dst = Path("$AIME24_PARQUET")

rows = []
with src.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if not line.strip():
            continue
        x = json.loads(line)

        problem = (
            x.get("prompt")
            or x.get("problem")
            or x.get("question")
            or x.get("input")
            or x.get("query")
        )

        answer = (
            x.get("answer")
            or x.get("final_answer")
            or x.get("target")
            or x.get("gt")
            or x.get("ground_truth")
        )

        if problem is None:
            raise KeyError(f"row {i} has no prompt/problem/question/input/query field, keys={list(x.keys())}")

        rows.append({
            "data_source": "aime24",
            "prompt": [
                {
                    "role": "user",
                    "content": str(problem),
                }
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": "" if answer is None else str(answer),
            },
            "extra_info": {
                "split": "test",
                "index": i,
                "answer": "" if answer is None else str(answer),
            },
        })

df = pd.DataFrame(rows)
print(df.head())
print("rows:", len(df))
print("columns:", list(df.columns))

dst.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(dst, index=False)
print("saved:", dst)
PY
fi

test_files="['$AIME24_PARQUET']"

ray stop --force || true
rm -rf /workspace/ray_tmp
mkdir -p /workspace/ray_tmp

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
        data.max_response_length=16384 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.shuffle=True \
        data.seed=42 \
        data.return_raw_chat=True \
        +data.apply_chat_template_kwargs.enable_thinking=False \
        actor_rollout_ref.model.path=/workspace/models/Qwen3-1.7B \
        +actor_rollout_ref.ref.model.path=/workspace/models/Qwen3-4B-Non-Thinking-RL-Math-Step500 \
        actor_rollout_ref.actor.optim.lr=2e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True \
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
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
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
        trainer.critic_warmup=0 \
        trainer.val_before_train=True \
        trainer.logger='["console","wandb"]' \
        trainer.log_val_generations=10 \
        trainer.project_name='on-policy-distillation' \
        trainer.experiment_name='qwen3_1.7b_non_thinking_teacher_qwen3_4b_non_thinking_rl_math_standard_opd_dapo_math_17k' \
        trainer.n_gpus_per_node=2 \
        trainer.nnodes=1 \
        trainer.save_freq=50 \
        trainer.default_local_dir=/G-OPD-checkpoints/Qwen3-1.7B-Standard-OPD-DAPO-Math-17K \
        trainer.max_actor_ckpt_to_keep=1 \
        trainer.max_critic_ckpt_to_keep=1 \
        trainer.test_freq=10 \
        trainer.total_epochs=1 \
        "$@"
