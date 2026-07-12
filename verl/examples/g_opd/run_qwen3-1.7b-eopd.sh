#!/usr/bin/env bash
set -euo pipefail

if [ "${EOPD_SHELL_DEBUG:-0}" = "1" ]; then
    set -x
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_WORKSPACE="${VERL_WORKSPACE:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
OPD_REPO_ROOT="${OPD_REPO_ROOT:-$(cd "$VERL_WORKSPACE/.." && pwd)}"


export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_MODE="${WANDB_MODE:-online}"
export USED_MODEL="${USED_MODEL:-no_api}"
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export VERL_PRINT_CONFIG="${VERL_PRINT_CONFIG:-0}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

count_visible_cuda_devices() {
    local devices="$1"
    if [ -z "$devices" ]; then
        echo 1
        return
    fi

    local IFS=','
    local -a parts
    read -r -a parts <<< "$devices"
    echo "${#parts[@]}"
}

cd "$VERL_WORKSPACE"

EOPD_ENV_FILE="${EOPD_ENV_FILE:-$VERL_WORKSPACE/.env}"
if [ -f "$EOPD_ENV_FILE" ]; then
    case "$-" in
        *x*) EOPD_RESTORE_XTRACE=1; set +x ;;
        *) EOPD_RESTORE_XTRACE=0 ;;
    esac
    set -a
    # shellcheck disable=SC1090
    . "$EOPD_ENV_FILE"
    set +a
    if [ "$EOPD_RESTORE_XTRACE" = 1 ]; then
        set -x
    fi
    unset EOPD_RESTORE_XTRACE
fi

EOPD_N_VISIBLE_GPUS="$(count_visible_cuda_devices "$CUDA_VISIBLE_DEVICES")"

TRAIN_SRC="${TRAIN_SRC:-$OPD_REPO_ROOT/data/train-00000-of-00001.parquet}"
TRAIN_FILE="${TRAIN_FILE:-$VERL_WORKSPACE/train_verl.parquet}"
export TRAIN_SRC TRAIN_FILE

AIME26_JSONL="${AIME26_JSONL:-/workspace/G-OPD/data/aime26/test.jsonl}"
AIME26_PARQUET="${AIME26_PARQUET:-/workspace/G-OPD/data/aime26/test_verl.parquet}"
export AIME26_JSONL AIME26_PARQUET

STUDENT_MODEL="${STUDENT_MODEL:-/workspace/models/Qwen3-4B}"
TEACHER_MODEL="${TEACHER_MODEL:-/workspace/models/Qwen3-4B-Non-Thinking-RL-Math-Step500}"
STUDENT_MODEL_REPO="${STUDENT_MODEL_REPO:-Qwen/Qwen3-4B}"
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

EOPD_EXPERIMENT_NAME="${EOPD_EXPERIMENT_NAME:-eopd_217_qwen3_4b_eopd_teacher_qwen3_4b_non_thinking_rl_math}"
EOPD_SAVE_FREQ="${EOPD_SAVE_FREQ:-50}"
EOPD_DEFAULT_CKPT_DIR="/EOPD-checkpoints/${EOPD_EXPERIMENT_NAME}_save_step_${EOPD_SAVE_FREQ}"
EOPD_CKPT_DIR="${EOPD_CKPT_DIR:-$EOPD_DEFAULT_CKPT_DIR}"
EOPD_RESUME_MODE="${EOPD_RESUME_MODE:-disable}"
EOPD_RESUME_FROM_PATH="${EOPD_RESUME_FROM_PATH:-null}"

case "$EOPD_RESUME_MODE" in
    auto|disable|resume_path) ;;
    *)
        echo "ERROR: EOPD_RESUME_MODE must be one of: auto, disable, resume_path"
        exit 1
        ;;
esac

EOPD_LOGGER="${EOPD_LOGGER:-[\"console\",\"wandb\"]}"
if [[ "$EOPD_LOGGER" == *wandb* ]]; then
    python3 - <<'PY'
import os
import wandb

key = os.environ.get("WANDB_API_KEY")
if not key:
    raise RuntimeError("WANDB_API_KEY is empty. Set it in the environment or .env, or set EOPD_LOGGER='[\"console\"]'.")
wandb.login(key=key, relogin=True)
PY
fi

EOPD_TOPK="${EOPD_TOPK:-16}"
EOPD_ENTROPY_THRESHOLD="${EOPD_ENTROPY_THRESHOLD:-0.8}"
EOPD_ALPHA="${EOPD_ALPHA:-1.0}"
EOPD_ROLLOUT_GPU_MEMORY_UTILIZATION="${EOPD_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
EOPD_PROGRESS_DEBUG="${EOPD_PROGRESS_DEBUG:-True}"
EOPD_DEBUG_ENABLE="${EOPD_DEBUG_ENABLE:-False}"
EOPD_LOG_DIR="${EOPD_LOG_DIR:-/workspace/G-OPD-logs/${EOPD_EXPERIMENT_NAME}_save_step_${EOPD_SAVE_FREQ}}"
EOPD_DEBUG_ROOT="${EOPD_DEBUG_ROOT:-$EOPD_LOG_DIR}"
EOPD_DEBUG_DIR="${EOPD_DEBUG_DIR:-$EOPD_DEBUG_ROOT/token_debug}"
EOPD_DEBUG_MAX_SAMPLES="${EOPD_DEBUG_MAX_SAMPLES:-1}"
EOPD_DEBUG_MAX_TOKENS_PER_SAMPLE="${EOPD_DEBUG_MAX_TOKENS_PER_SAMPLE:-16}"
EOPD_ROLLOUT_DATA_DIR="${EOPD_ROLLOUT_DATA_DIR:-$EOPD_DEBUG_ROOT/rollout_data}"

echo "[EOPD] checkpoints: $EOPD_CKPT_DIR"
echo "[EOPD] log dir: $EOPD_LOG_DIR"
echo "[EOPD] debug root: $EOPD_DEBUG_ROOT"
echo "[EOPD] progress debug: $EOPD_PROGRESS_DEBUG"
echo "[EOPD] token debug: $EOPD_DEBUG_ENABLE -> $EOPD_DEBUG_DIR"
echo "[EOPD] rollout data: $EOPD_ROLLOUT_DATA_DIR"

python3 -m recipe.eopd_baseline.main_eopd \
        algorithm.adv_estimator=grpo \
        algorithm.rollout_correction.rollout_is=token \
        algorithm.rollout_correction.rollout_is_threshold=5.0 \
        algorithm.rollout_correction.rollout_rs=null \
        algorithm.rollout_correction.bypass_mode=false \
        actor_rollout_ref.rollout.calculate_log_probs=true \
        data.train_files="$TRAIN_FILE" \
        data.val_files="$AIME26_PARQUET" \
        data.train_batch_size="${EOPD_TRAIN_BATCH_SIZE:-128}" \
        data.max_prompt_length="${EOPD_MAX_PROMPT_LENGTH:-2048}" \
        data.max_response_length="${EOPD_MAX_RESPONSE_LENGTH:-2048}" \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.shuffle=True \
        data.seed="${EOPD_SEED:-42}" \
        data.return_raw_chat=True \
        +data.apply_chat_template_kwargs.enable_thinking=False \
        actor_rollout_ref.model.path="$STUDENT_MODEL" \
        +actor_rollout_ref.ref.model.path="$TEACHER_MODEL" \
        actor_rollout_ref.actor.optim.lr="${EOPD_LR:-2e-6}" \
        data.filter_overlong_prompts_workers=4 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.use_fused_kernels=False \
        actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True \
        actor_rollout_ref.actor.policy_loss.lambda_vals=1.0 \
        actor_rollout_ref.actor.policy_loss.multi_teacher_distill=False \
        actor_rollout_ref.actor.policy_loss.eopd.enable=True \
        actor_rollout_ref.actor.policy_loss.eopd.topk="$EOPD_TOPK" \
        actor_rollout_ref.actor.policy_loss.eopd.entropy_threshold="$EOPD_ENTROPY_THRESHOLD" \
        actor_rollout_ref.actor.policy_loss.eopd.alpha="$EOPD_ALPHA" \
        actor_rollout_ref.actor.policy_loss.eopd.debug.enable="$EOPD_DEBUG_ENABLE" \
        actor_rollout_ref.actor.policy_loss.eopd.debug.dir="$EOPD_DEBUG_DIR" \
        actor_rollout_ref.actor.policy_loss.eopd.debug.max_samples="$EOPD_DEBUG_MAX_SAMPLES" \
        actor_rollout_ref.actor.policy_loss.eopd.debug.max_tokens_per_sample="$EOPD_DEBUG_MAX_TOKENS_PER_SAMPLE" \
        actor_rollout_ref.actor.ppo_mini_batch_size="${EOPD_PPO_MINI_BATCH_SIZE:-128}" \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${EOPD_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}" \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${EOPD_PPO_MAX_TOKEN_LEN_PER_GPU:-32768}" \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.actor.checkpoint.save_contents='["model"]' \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${EOPD_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}" \
        actor_rollout_ref.rollout.tensor_model_parallel_size="${EOPD_ROLLOUT_TP:-4}" \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization="$EOPD_ROLLOUT_GPU_MEMORY_UTILIZATION" \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.rollout.max_num_batched_tokens="${EOPD_MAX_NUM_BATCHED_TOKENS:-32768}" \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
        actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${EOPD_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}" \
        actor_rollout_ref.ref.topk_logits="$EOPD_TOPK" \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        reward_model.reward_manager=naive \
        trainer.critic_warmup=0 \
        trainer.val_before_train=True \
        trainer.progress_debug="$EOPD_PROGRESS_DEBUG" \
        trainer.logger="$EOPD_LOGGER" \
        trainer.rollout_data_dir="$EOPD_ROLLOUT_DATA_DIR" \
        trainer.log_val_generations="${EOPD_LOG_VAL_GENERATIONS:-10}" \
        trainer.project_name="${EOPD_PROJECT_NAME:-on-policy-distillation}" \
        trainer.experiment_name="$EOPD_EXPERIMENT_NAME" \
        trainer.n_gpus_per_node="${EOPD_N_GPUS_PER_NODE:-$EOPD_N_VISIBLE_GPUS}" \
        trainer.nnodes="${EOPD_NNODES:-1}" \
        trainer.save_freq="$EOPD_SAVE_FREQ" \
        trainer.resume_mode="$EOPD_RESUME_MODE" \
        trainer.resume_from_path="$EOPD_RESUME_FROM_PATH" \
        trainer.default_local_dir="$EOPD_CKPT_DIR" \
        trainer.max_actor_ckpt_to_keep=1 \
        trainer.max_critic_ckpt_to_keep=1 \
        trainer.test_freq="${EOPD_TEST_FREQ:-30}" \
        trainer.total_epochs="${EOPD_TOTAL_EPOCHS:-1}" \
        "$@"
