#!/usr/bin/env bash
set -euo pipefail

if [ "${TA_OPD_SHELL_DEBUG:-0}" = "1" ]; then
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

TA_OPD_N_VISIBLE_GPUS="$(count_visible_cuda_devices "$CUDA_VISIBLE_DEVICES")"

cd "$VERL_WORKSPACE"

TA_OPD_ENV_FILE="${TA_OPD_ENV_FILE:-$VERL_WORKSPACE/.env}"
if [ -f "$TA_OPD_ENV_FILE" ]; then
    case "$-" in
        *x*) TA_OPD_RESTORE_XTRACE=1; set +x ;;
        *) TA_OPD_RESTORE_XTRACE=0 ;;
    esac
    set -a
    # shellcheck disable=SC1090
    . "$TA_OPD_ENV_FILE"
    set +a
    if [ "$TA_OPD_RESTORE_XTRACE" = 1 ]; then
        set -x
    fi
    unset TA_OPD_RESTORE_XTRACE
fi

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

TA_OPD_TOPK="${TA_OPD_TOPK:-16}"
TA_OPD_METHOD="${TA_OPD_METHOD:-teachability}"
TA_OPD_RATIO="${TA_OPD_RATIO:-0.1}"
TA_OPD_SEED="${TA_OPD_SEED:-42}"
TA_OPD_EXPERIMENT_NAME="${TA_OPD_EXPERIMENT_NAME:-qwen3_4b_ta_opd_${TA_OPD_METHOD}_ratio${TA_OPD_RATIO}_k${TA_OPD_TOPK}_seed${TA_OPD_SEED}_teacher_qwen3_4b_non_thinking_rl_math}"
TA_OPD_SAVE_FREQ="${TA_OPD_SAVE_FREQ:-50}"
TA_OPD_DEFAULT_CKPT_DIR="/TA_OPD-checkpoints/${TA_OPD_EXPERIMENT_NAME}_save_step_${TA_OPD_SAVE_FREQ}"
TA_OPD_CKPT_DIR="${TA_OPD_CKPT_DIR:-$TA_OPD_DEFAULT_CKPT_DIR}"
TA_OPD_RESUME_MODE="${TA_OPD_RESUME_MODE:-disable}"
TA_OPD_RESUME_FROM_PATH="${TA_OPD_RESUME_FROM_PATH:-null}"

case "$TA_OPD_RESUME_MODE" in
    auto|disable|resume_path) ;;
    *)
        echo "ERROR: TA_OPD_RESUME_MODE must be one of: auto, disable, resume_path"
        exit 1
        ;;
esac

TA_OPD_LOGGER="${TA_OPD_LOGGER:-[\"console\",\"wandb\"]}"
if [[ "$TA_OPD_LOGGER" == *wandb* ]]; then
    python3 - <<'PY'
import os
import wandb

key = os.environ.get("WANDB_API_KEY")
if not key:
    raise RuntimeError("WANDB_API_KEY is empty. Set it in the environment or .env, or set TA_OPD_LOGGER='[\"console\"]'.")
wandb.login(key=key, relogin=True)
PY
fi

TA_OPD_RENORMALIZE_TOPK="${TA_OPD_RENORMALIZE_TOPK:-False}"
TA_OPD_EXACT_COVERAGE="${TA_OPD_EXACT_COVERAGE:-False}"
TA_OPD_Q_LOW="${TA_OPD_Q_LOW:-0.05}"
TA_OPD_Q_HIGH="${TA_OPD_Q_HIGH:-0.95}"
TA_OPD_MIN_KEEP_PER_SAMPLE="${TA_OPD_MIN_KEEP_PER_SAMPLE:-1}"
TA_OPD_RANDOM_SEED="${TA_OPD_RANDOM_SEED:-42}"
TA_OPD_ROLLOUT_GPU_MEMORY_UTILIZATION="${TA_OPD_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
TA_OPD_PROGRESS_DEBUG="${TA_OPD_PROGRESS_DEBUG:-True}"
TA_OPD_DEBUG_ENABLE="${TA_OPD_DEBUG_ENABLE:-True}"
TA_OPD_LOG_DIR="${TA_OPD_LOG_DIR:-/workspace/G-OPD-logs/${TA_OPD_EXPERIMENT_NAME}_save_step_${TA_OPD_SAVE_FREQ}}"
TA_OPD_DEBUG_ROOT="${TA_OPD_DEBUG_ROOT:-$TA_OPD_LOG_DIR}"
TA_OPD_DEBUG_DIR="${TA_OPD_DEBUG_DIR:-$TA_OPD_DEBUG_ROOT/token_debug}"
TA_OPD_DEBUG_MAX_SAMPLES="${TA_OPD_DEBUG_MAX_SAMPLES:-1}"
TA_OPD_DEBUG_MAX_TOKENS_PER_SAMPLE="${TA_OPD_DEBUG_MAX_TOKENS_PER_SAMPLE:-16}"
TA_OPD_ROLLOUT_DATA_DIR="${TA_OPD_ROLLOUT_DATA_DIR:-$TA_OPD_DEBUG_ROOT/rollout_data}"

echo "[TA_OPD] checkpoints: $TA_OPD_CKPT_DIR"
echo "[TA_OPD] debug root: $TA_OPD_DEBUG_ROOT"
echo "[TA_OPD] progress debug: $TA_OPD_PROGRESS_DEBUG"
echo "[TA_OPD] exact coverage: $TA_OPD_EXACT_COVERAGE"
echo "[TA_OPD] token debug: $TA_OPD_DEBUG_ENABLE -> $TA_OPD_DEBUG_DIR"
echo "[TA_OPD] rollout data: $TA_OPD_ROLLOUT_DATA_DIR"

python3 -m recipe.ta_opd_baseline.main_ta_opd \
        algorithm.adv_estimator=grpo \
        algorithm.rollout_correction.rollout_is=token \
        algorithm.rollout_correction.rollout_is_threshold=5.0 \
        algorithm.rollout_correction.rollout_rs=null \
        algorithm.rollout_correction.bypass_mode=false \
        actor_rollout_ref.rollout.calculate_log_probs=true \
        data.train_files="$TRAIN_FILE" \
        data.val_files="$AIME26_PARQUET" \
        data.train_batch_size="${TA_OPD_TRAIN_BATCH_SIZE:-128}" \
        data.max_prompt_length="${TA_OPD_MAX_PROMPT_LENGTH:-2048}" \
        data.max_response_length="${TA_OPD_MAX_RESPONSE_LENGTH:-2048}" \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.shuffle=True \
        data.seed="$TA_OPD_SEED" \
        data.return_raw_chat=True \
        +data.apply_chat_template_kwargs.enable_thinking=False \
        actor_rollout_ref.model.path="$STUDENT_MODEL" \
        +actor_rollout_ref.ref.model.path="$TEACHER_MODEL" \
        actor_rollout_ref.actor.optim.lr="${TA_OPD_LR:-2e-6}" \
        data.filter_overlong_prompts_workers=4 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.use_fused_kernels=False \
        actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True \
        actor_rollout_ref.actor.policy_loss.lambda_vals=1.0 \
        actor_rollout_ref.actor.policy_loss.multi_teacher_distill=False \
        actor_rollout_ref.actor.policy_loss.ta_opd.enable=True \
        actor_rollout_ref.actor.policy_loss.ta_opd.method="$TA_OPD_METHOD" \
        actor_rollout_ref.actor.policy_loss.ta_opd.ratio="$TA_OPD_RATIO" \
        actor_rollout_ref.actor.policy_loss.ta_opd.topk="$TA_OPD_TOPK" \
        actor_rollout_ref.actor.policy_loss.ta_opd.exact_coverage="$TA_OPD_EXACT_COVERAGE" \
        actor_rollout_ref.actor.policy_loss.ta_opd.renormalize_topk="$TA_OPD_RENORMALIZE_TOPK" \
        actor_rollout_ref.actor.policy_loss.ta_opd.q_low="$TA_OPD_Q_LOW" \
        actor_rollout_ref.actor.policy_loss.ta_opd.q_high="$TA_OPD_Q_HIGH" \
        actor_rollout_ref.actor.policy_loss.ta_opd.min_keep_per_sample="$TA_OPD_MIN_KEEP_PER_SAMPLE" \
        actor_rollout_ref.actor.policy_loss.ta_opd.random_seed="$TA_OPD_RANDOM_SEED" \
        actor_rollout_ref.actor.policy_loss.ta_opd.debug.enable="$TA_OPD_DEBUG_ENABLE" \
        actor_rollout_ref.actor.policy_loss.ta_opd.debug.dir="$TA_OPD_DEBUG_DIR" \
        actor_rollout_ref.actor.policy_loss.ta_opd.debug.max_samples="$TA_OPD_DEBUG_MAX_SAMPLES" \
        actor_rollout_ref.actor.policy_loss.ta_opd.debug.max_tokens_per_sample="$TA_OPD_DEBUG_MAX_TOKENS_PER_SAMPLE" \
        actor_rollout_ref.actor.ppo_mini_batch_size="${TA_OPD_PPO_MINI_BATCH_SIZE:-128}" \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${TA_OPD_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}" \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${TA_OPD_PPO_MAX_TOKEN_LEN_PER_GPU:-32768}" \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.actor.checkpoint.save_contents='["model"]' \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${TA_OPD_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}" \
        actor_rollout_ref.rollout.tensor_model_parallel_size="${TA_OPD_ROLLOUT_TP:-4}" \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization="$TA_OPD_ROLLOUT_GPU_MEMORY_UTILIZATION" \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.rollout.max_num_batched_tokens="${TA_OPD_MAX_NUM_BATCHED_TOKENS:-32768}" \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
        actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${TA_OPD_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}" \
        actor_rollout_ref.ref.topk_logits="$TA_OPD_TOPK" \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        reward_model.reward_manager=naive \
        trainer.gpt_rollout_score.enable=False \
        trainer.critic_warmup=0 \
        trainer.val_before_train=True \
        trainer.progress_debug="$TA_OPD_PROGRESS_DEBUG" \
        trainer.logger="$TA_OPD_LOGGER" \
        trainer.rollout_data_dir="$TA_OPD_ROLLOUT_DATA_DIR" \
        trainer.log_val_generations="${TA_OPD_LOG_VAL_GENERATIONS:-10}" \
        trainer.project_name="${TA_OPD_PROJECT_NAME:-on-policy-distillation}" \
        trainer.experiment_name="$TA_OPD_EXPERIMENT_NAME" \
        trainer.n_gpus_per_node="${TA_OPD_N_GPUS_PER_NODE:-$TA_OPD_N_VISIBLE_GPUS}" \
        trainer.nnodes="${TA_OPD_NNODES:-1}" \
        trainer.save_freq="$TA_OPD_SAVE_FREQ" \
        trainer.resume_mode="$TA_OPD_RESUME_MODE" \
        trainer.resume_from_path="$TA_OPD_RESUME_FROM_PATH" \
        trainer.default_local_dir="$TA_OPD_CKPT_DIR" \
        trainer.max_actor_ckpt_to_keep=1 \
        trainer.max_critic_ckpt_to_keep=1 \
        trainer.test_freq="${TA_OPD_TEST_FREQ:-30}" \
        trainer.total_epochs="${TA_OPD_TOTAL_EPOCHS:-1}" \
        "$@"
