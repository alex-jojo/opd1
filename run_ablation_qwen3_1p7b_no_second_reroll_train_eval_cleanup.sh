#!/usr/bin/env bash
set -euo pipefail

cd /workspace/opd1

STUDENT_MODEL="/workspace/models/Qwen3-1.7B"
TEACHER_MODEL="/workspace/models/Qwen3-4B-Non-Thinking-RL-Math-Step500"
EXP="146_qwen3_1.7b_teacher_qwen3_4b_ablation_no_second_reroll_lr2e-6"
CKPT="/G-OPD-checkpoints/${EXP}_save_step_50"
RUN_DIR="/workspace/opd1/eval_runs/${EXP}"

if [ ! -f "$STUDENT_MODEL/config.json" ]; then
    echo "ERROR: student model is missing or not a HF model dir: $STUDENT_MODEL"
    exit 1
fi

mkdir -p "$RUN_DIR"

sed -i 's/\r$//' /workspace/opd1/verl/examples/g_opd/run_qwen3-4b-g-opd.sh
sed -i 's/\r$//' /workspace/opd1/math_eval/run_eval_math.sh

STUDENT_MODEL="$STUDENT_MODEL" \
TEACHER_MODEL="$TEACHER_MODEL" \
G_OPD_EXPERIMENT_NAME="$EXP" \
G_OPD_LR=2e-6 \
G_OPD_SAVE_FREQ=50 \
G_OPD_CKPT_DIR="$CKPT" \
G_OPD_RESUME_MODE=disable \
GPT_ROLLOUT_SCORE_ENABLE=True \
GPT_ROLLOUT_SCORE_RANK_GAP_DROP_ENABLE=True \
GPT_ROLLOUT_SCORE_RANK_GAP_DROP_THRESHOLD=0.75 \
GPT_ROLLOUT_SCORE_RANK_GAP_DROP_SCOPE=all \
GPT_ROLLOUT_SCORE_MAX_REROLLOUT_ATTEMPTS=0 \
GPT_ROLLOUT_SCORE_HINT_RETRIES=0 \
GPT_ROLLOUT_SCORE_REROLL_NOHINT_ENABLE=False \
GPT_ROLLOUT_SCORE_REROLL_HINT_LOSS_WEIGHT=0.5 \
GPT_ROLLOUT_SCORE_RUBRIC_ADV_SHIFT_ENABLE=True \
GPT_ROLLOUT_SCORE_RUBRIC_ADV_SHIFT_MODE=history_zscore \
GPT_ROLLOUT_SCORE_RUBRIC_ADV_SHIFT_COEF=0.10 \
GPT_ROLLOUT_SCORE_RUBRIC_ADV_SHIFT_CLIP=0.20 \
GPT_ROLLOUT_SCORE_HISTORY_NUM_BINS=12 \
GPT_ROLLOUT_SCORE_HISTORY_BUCKET_MODE=label \
GPT_ROLLOUT_SCORE_HISTORY_STD_FLOOR=8.0 \
GPT_ROLLOUT_SCORE_HISTORY_NEGATIVE_COEF_SCALE=0.8 \
GPT_ROLLOUT_SCORE_REROLL_APPEND_REQUIRE_IMPROVEMENT=True \
GPT_ROLLOUT_SCORE_REROLL_APPEND_MIN_SCORE=45 \
GPT_ROLLOUT_SCORE_REROLL_APPEND_MIN_GAIN=0 \
bash /workspace/opd1/verl/examples/g_opd/run_qwen3-4b-g-opd.sh \
  trainer.total_training_steps=110 \
  actor_rollout_ref.actor.checkpoint.save_contents='["model"]' \
  2>&1 | tee "$RUN_DIR/train.log"

ray stop --force || true

STUDENT_MODEL="$STUDENT_MODEL" \
G_OPD_EXPERIMENT_NAME="$EXP" \
G_OPD_SAVE_FREQ=50 \
G_OPD_CKPT_DIR="$CKPT" \
OUTPUT_DIR="$RUN_DIR/eval_outputs" \
SKIP_DOWNLOAD=1 \
DATASETS="aime24 aime25 aime26 hmmt26 amc23 math500" \
bash /workspace/opd1/math_eval/run_eval_math.sh \
  2>&1 | tee "$RUN_DIR/eval.log"

cp "$RUN_DIR/eval_outputs/summary.csv" "$RUN_DIR/summary_ablation_no_second_reroll.csv"
cat "$RUN_DIR/summary_ablation_no_second_reroll.csv"

rm -rf -- "$CKPT"
