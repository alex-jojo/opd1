# OPD + Rubric 实验说明

本文档整理当前 `opd+rubric` 方法、关键训练流程、已有结果、ablation 结果，以及服务器上可直接运行的命令。

## 1. 基本设定

- 代码框架：`verl`
- 训练入口：`/workspace/opd1/verl/examples/g_opd/run_qwen3-4b-g-opd.sh`
- 评测入口：`/workspace/opd1/math_eval/run_eval_math.sh`
- Teacher：`/workspace/models/Qwen3-4B-Non-Thinking-RL-Math-Step500`
- 训练数据：DAPO Math 17k English split
- 评测集：`aime24 aime25 aime26 hmmt26 amc23 math500`
- 默认训练步数：`trainer.total_training_steps=110`
- 默认只保存模型，不保存优化器：

```bash
actor_rollout_ref.actor.checkpoint.save_contents='["model"]'
```

## 2. 当前方法流程

当前主线是 OPD + GPT rubric。核心目标是保留 teacher OPD 的稳定监督，同时用 GPT rubric 对样本质量做筛选、分桶归一化和 advantage 修正。

### 2.1 原始 rollout

每个 batch 中 student 先对原始题目生成一次回答。随后计算：

- `old_log_probs`：student rollout 策略下的 token log prob
- `ref_log_prob`：teacher 对同一 response 的 token log prob
- OPD advantage 基础项来自 teacher 与 student 的 log prob 差异

当前 teacher score 用于 rank-gap 对齐：

```text
teacher_score = mean(ref_log_prob - old_log_probs over response tokens)
```

### 2.2 GPT rubric 打分

如果开启：

```bash
GPT_ROLLOUT_SCORE_ENABLE=True
```

每条原始 response 会调用 GPT rubric 打分，输出：

- `score_100`
- rubric 维度分
- reason
- revision suggestion，低分样本用于 reroll hint
- problem domain / difficulty label，用于 history bucket

第一次 GPT 输出限制：

```bash
GPT_ROLLOUT_SCORE_MAX_OUTPUT_TOKENS=768
```

低分样本若缺 suggestion，会额外调用一次 GPT hint：

```bash
GPT_ROLLOUT_SCORE_HINT_RETRIES=1
GPT_ROLLOUT_SCORE_HINT_MAX_OUTPUT_TOKENS=256
```

reroll 后第二次 GPT 打分输出限制：

```bash
GPT_ROLLOUT_SCORE_REROLL_MAX_OUTPUT_TOKENS=512
```

### 2.3 Rank-gap drop

rank-gap 用于处理 teacher rank 与 GPT rubric rank 严重不一致的样本：

```text
rank_gap = abs(teacher_rank - rubric_rank) / (valid_count - 1)
```

如果：

```bash
GPT_ROLLOUT_SCORE_RANK_GAP_DROP_ENABLE=True
GPT_ROLLOUT_SCORE_RANK_GAP_DROP_THRESHOLD=0.75
GPT_ROLLOUT_SCORE_RANK_GAP_DROP_SCOPE=all
```

则 rank 差超过阈值的样本会被 drop。方向统计会写入 wandb 和 JSONL，包括：

- rubric 低、teacher 高
- rubric 高、teacher 低

### 2.4 Rubric advantage shift

当前主线使用 history z-score：

```bash
GPT_ROLLOUT_SCORE_RUBRIC_ADV_SHIFT_ENABLE=True
GPT_ROLLOUT_SCORE_RUBRIC_ADV_SHIFT_MODE=history_zscore
GPT_ROLLOUT_SCORE_HISTORY_BUCKET_MODE=label
GPT_ROLLOUT_SCORE_HISTORY_NUM_BINS=12
GPT_ROLLOUT_SCORE_HISTORY_STD_FLOOR=8.0
GPT_ROLLOUT_SCORE_HISTORY_NEGATIVE_COEF_SCALE=0.8
```

每条样本按 GPT 输出的 domain/difficulty label 进入历史桶。当前样本 rubric score 与该桶历史分布比较：

```text
z = (rubric_score - history_mean) / max(history_std, std_floor)
shift = clip(coef * z, -clip, +clip)
```

默认：

```bash
GPT_ROLLOUT_SCORE_RUBRIC_ADV_SHIFT_COEF=0.10
GPT_ROLLOUT_SCORE_RUBRIC_ADV_SHIFT_CLIP=0.20
```

最终 OPD advantage：

```text
advantage = OPD_advantage + rubric_shift
```

### 2.5 Reroll

低于阈值的原始样本：

```bash
GPT_ROLLOUT_SCORE_MIN_SCORE_100=50
```

会用 GPT suggestion 拼接成 hint prompt，让 student 再生成一次。reroll 输出再经过 GPT 打分。

历史上跑过三类 reroll 用法：

- `no_second_reroll`：完全不开第二次 rollout，只做 GPT rank-gap 和 rubric shift。
- `reroll_hint/nohint OPD`：把 reroll 样本继续并入 OPD 训练。
- `reroll_sft`：把通过筛选的 reroll response 作为单独 SFT 正样本。

最新新增的稳妥 reroll_sft 设置是更弱、更严格的一版：

```bash
GPT_ROLLOUT_SCORE_REROLL_SFT_ENABLE=True
GPT_ROLLOUT_SCORE_REROLL_SFT_KEEP_HINT_OPD=False
GPT_ROLLOUT_SCORE_REROLL_SFT_LAMBDA=0.05
GPT_ROLLOUT_SCORE_REROLL_SFT_ALPHA=0.5
GPT_ROLLOUT_SCORE_REROLL_SFT_SCORE_COEF=0.5
GPT_ROLLOUT_SCORE_REROLL_SFT_GAIN_COEF=0.5
GPT_ROLLOUT_SCORE_REROLL_APPEND_MIN_SCORE=75
GPT_ROLLOUT_SCORE_REROLL_APPEND_MIN_GAIN=20
```

含义：

- 只有第二次 GPT 分数至少 75，且比第一次高至少 20，才进入 reroll_sft。
- `lambda=0.05` 让 reroll_sft 只作为弱辅助信号。
- score 和 gain 各占一半，避免只看绝对分数。
- `exp(alpha * z)` 只产生正权重，不产生负梯度。

## 3. Loss 构成

原始样本走 OPD/PPO loss：

```text
L_opd = PPOLoss(log_prob, old_log_prob, OPD_advantage + rubric_shift)
```

rank-gap drop 会通过 `g_opd_loss_weight` 把被 drop 样本权重置零。

reroll_sft 样本不走 OPD advantage，单独走：

```text
L_reroll_sft = - weight * log pi_theta(y_reroll | x)
```

总 loss：

```text
L_total = L_opd + L_reroll_sft
```

对应代码：

- `verl/verl/trainer/ppo/ray_trainer.py`：GPT 打分、reroll、rank-gap、history shift、batch 拼接
- `verl/verl/workers/actor/dp_actor.py`：policy loss 和 reroll_sft loss
- `verl/verl/trainer/config/ppo_trainer.yaml`：Hydra 默认配置
- `verl/examples/g_opd/run_qwen3-4b-g-opd.sh`：环境变量到 Hydra 参数的映射

## 4. Mean@8 主结果

| Teacher Model | Student Model | method | AIME24 | AIME25 | AIME26 | HMMT26 | AMC23 | MATH500 | AVG |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B-RL | Qwen3-0.6B | base | 0.83 | 2.08 | 0.00 | 0.00 | 22.81 | 44.68 | 11.73 |
| Qwen3-4B-RL | Qwen3-0.6B | opd | 7.50 | 15.42 | 7.92 | 4.92 | 40.31 | 68.85 | 24.15 |
| Qwen3-4B-RL | Qwen3-0.6B | EXOPD | 5.83 | 10.42 | 7.92 | 4.55 | 41.25 | 66.92 | 22.82 |
| Qwen3-4B-RL | Qwen3-0.6B | eopd | 7.08 | 14.17 | 7.50 | 5.68 | 41.56 | 69.33 | 24.22 |
| Qwen3-4B-RL | Qwen3-0.6B | ta_opd | 3.33 | 12.50 | 8.75 | 3.79 | 39.06 | 66.83 | 22.38 |
| Qwen3-4B-RL | Qwen3-0.6B | opd+rubric | 7.08 | 13.75 | 9.58 | 7.58 | 43.12 | 69.15 | 25.04 |
| Qwen3-4B-RL | Qwen3-1.7B | base | 14.17 | 12.08 | 9.58 | 6.44 | 48.12 | 72.00 | 27.06 |
| Qwen3-4B-RL | Qwen3-1.7B | opd | 21.67 | 23.75 | 17.50 | 12.50 | 57.81 | 82.55 | 35.96 |
| Qwen3-4B-RL | Qwen3-1.7B | EXOPD | 18.33 | 17.92 | 13.33 | 12.50 | 54.69 | 80.35 | 32.85 |
| Qwen3-4B-RL | Qwen3-1.7B | eopd | 21.25 | 24.17 | 20.00 | 14.02 | 56.25 | 82.35 | 36.34 |
| Qwen3-4B-RL | Qwen3-1.7B | ta_opd | 20.42 | 23.33 | 19.58 | 12.12 | 56.88 | 83.00 | 35.89 |
| Qwen3-4B-RL | Qwen3-1.7B | opd+rubric v7 | 24.17 | 24.17 | 20.42 | 15.15 | 60.63 | 83.05 | 37.93 |
| Qwen3-4B-RL | Qwen3-4B | base | 23.33 | 22.50 | 19.58 | 15.91 | 64.38 | 82.78 | 38.08 |
| Qwen3-4B-RL | Qwen3-4B | opd | 40.00 | 34.17 | 34.17 | 19.70 | 78.44 | 90.25 | 49.46 |
| Qwen3-4B-RL | Qwen3-4B | EXOPD | 32.08 | 26.67 | 27.50 | 21.21 | 76.88 | 88.45 | 45.47 |
| Qwen3-4B-RL | Qwen3-4B | eopd | 36.67 | 32.92 | 32.92 | 21.97 | 81.88 | 89.83 | 49.37 |
| Qwen3-4B-RL | Qwen3-4B | ta_opd | 40.42 | 33.33 | 35.00 | 20.83 | 79.06 | 89.98 | 49.77 |
| Qwen3-4B-RL | Qwen3-4B | opd+rubric v7 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## 5. Qwen3-1.7B Ablation

| Model | method | AIME24 AVG@8 | AIME24 P@8 | AIME24 maj@8 | AIME25 AVG@8 | AIME25 P@8 | AIME25 maj@8 | AIME26 AVG@8 | AIME26 P@8 | AIME26 maj@8 | HMMT26 AVG@8 | HMMT26 P@8 | HMMT26 maj@8 | AMC23 AVG@8 | AMC23 P@8 | AMC23 maj@8 | MATH500 AVG@8 | MATH500 P@8 | MATH500 maj@8 | Overall AVG@8 | Overall P@8 | Overall maj@8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | no_second_rollout | 25.00 | 56.67 | 50.00 | 24.17 | 36.67 | 36.67 | 19.58 | 33.33 | 30.00 | 13.26 | 18.18 | 18.18 | 59.69 | 77.50 | 77.50 | 83.03 | 91.20 | 90.00 | 37.46 | 52.26 | 50.39 |
| Qwen3-1.7B | no_rank_gap | 22.50 | 40.00 | 40.00 | 22.92 | 33.33 | 33.33 | 15.00 | 23.33 | 20.00 | 14.77 | 21.21 | 21.21 | 58.44 | 80.00 | 80.00 | 82.85 | 91.40 | 89.40 | 36.08 | 48.21 | 47.32 |
| Qwen3-1.7B | no_rubric_shift | 24.58 | 43.33 | 43.33 | 22.92 | 36.67 | 36.67 | 15.83 | 30.00 | 30.00 | 14.02 | 24.24 | 24.24 | 59.69 | 77.50 | 77.50 | 82.98 | 92.00 | 89.80 | 36.67 | 50.29 | 50.26 |
| Qwen3-1.7B | random_bucket | 24.58 | 50.00 | 50.00 | 24.58 | 40.00 | 36.67 | 17.92 | 33.33 | 33.33 | 13.64 | 21.21 | 21.21 | 57.50 | 75.00 | 75.00 | 83.65 | 91.20 | 90.60 | 36.98 | 51.79 | 51.14 |
| Qwen3-1.7B | ours | 24.17 | 43.33 | 43.33 | 24.17 | 33.33 | 33.33 | 20.42 | 36.67 | 36.67 | 15.15 | 24.24 | 21.21 | 60.63 | 85.00 | 85.00 | 83.05 | 91.20 | 89.20 | 37.93 | 52.30 | 51.46 |

## 6. 命令行总览

这里分清楚三类命令：

- 一键脚本：训练 + 测评 + 保存 summary + 删除 ckpt。
- 单独训练：只训练，保留 ckpt，后面手动测评。
- 单独测评：从已有 ckpt 合并 HF 模型并跑 `n=8` math eval。

### 6.1 主实验一键脚本

这些脚本都在 `/workspace/opd1` 根目录下。除特别说明外，都是训练、测评、保存 summary、最后删除 ckpt。

#### Qwen3-0.6B OPD + Rubric

```bash
cd /workspace/opd1
bash /workspace/opd1/run_v8_qwen3_0p6b_lr2e-6_train_eval_cleanup.sh
```

student / lr：

```bash
STUDENT_MODEL=/workspace/models/Qwen3-0.6B
G_OPD_LR=2e-6
```

#### Qwen3-1.7B OPD + Rubric 当前最好主线

当前表里 `ours = 37.93 / 52.30 / 51.46` 对应这个脚本，也就是不开第二次 reroll，只保留 GPT rank-gap + rubric shift：

```bash
cd /workspace/opd1
bash /workspace/opd1/run_ablation_qwen3_1p7b_no_second_reroll_train_eval_cleanup.sh
```

关键配置：

```bash
STUDENT_MODEL=/workspace/models/Qwen3-1.7B
G_OPD_LR=2e-6
GPT_ROLLOUT_SCORE_ENABLE=True
GPT_ROLLOUT_SCORE_RANK_GAP_DROP_ENABLE=True
GPT_ROLLOUT_SCORE_MAX_REROLLOUT_ATTEMPTS=0
GPT_ROLLOUT_SCORE_RUBRIC_ADV_SHIFT_ENABLE=True
GPT_ROLLOUT_SCORE_RUBRIC_ADV_SHIFT_MODE=history_zscore
GPT_ROLLOUT_SCORE_HISTORY_BUCKET_MODE=label
```

#### Qwen3-1.7B 稳定 reroll_sft 下一轮

这是新加的更保守 reroll_sft 版本，只有 summary 文件存在且非空才会删除 ckpt：

```bash
cd /workspace/opd1
bash /workspace/opd1/run_v13_qwen3_1p7b_lr2e-6_reroll_sft_stable_train_eval_cleanup.sh
```

关键配置：

```bash
STUDENT_MODEL=/workspace/models/Qwen3-1.7B
G_OPD_LR=2e-6
GPT_ROLLOUT_SCORE_REROLL_SFT_ENABLE=True
GPT_ROLLOUT_SCORE_REROLL_SFT_KEEP_HINT_OPD=False
GPT_ROLLOUT_SCORE_REROLL_SFT_LAMBDA=0.05
GPT_ROLLOUT_SCORE_REROLL_SFT_ALPHA=0.5
GPT_ROLLOUT_SCORE_REROLL_SFT_SCORE_COEF=0.5
GPT_ROLLOUT_SCORE_REROLL_SFT_GAIN_COEF=0.5
GPT_ROLLOUT_SCORE_REROLL_APPEND_MIN_SCORE=75
GPT_ROLLOUT_SCORE_REROLL_APPEND_MIN_GAIN=20
GPT_ROLLOUT_SCORE_REROLL_NOHINT_ENABLE=False
```

#### Qwen3-4B OPD + Rubric

```bash
cd /workspace/opd1
bash /workspace/opd1/run_v8_qwen3_4b_lr1e-6_train_eval_cleanup.sh
```

student / lr：

```bash
STUDENT_MODEL=/workspace/models/Qwen3-4B
G_OPD_LR=1e-6
```

### 6.2 对比方法训练命令

这些是单独训练入口，不自动测评、不自动删除 ckpt。要测评请看 6.3。

#### Vanilla OPD

```bash
cd /workspace/opd1
STUDENT_MODEL=/workspace/models/Qwen3-1.7B \
TEACHER_MODEL=/workspace/models/Qwen3-4B-Non-Thinking-RL-Math-Step500 \
G_OPD_EXPERIMENT_NAME=146_qwen3_1.7b_teacher_qwen3_4b_vanilla_opd \
G_OPD_LR=2e-6 \
G_OPD_SAVE_FREQ=50 \
G_OPD_RESUME_MODE=disable \
GPT_ROLLOUT_SCORE_ENABLE=False \
bash /workspace/opd1/verl/examples/g_opd/run_qwen3-4b-g-opd.sh \
  trainer.total_training_steps=110 \
  actor_rollout_ref.actor.checkpoint.save_contents='["model"]'
```

换 0.6B / 4B 时只改：

```bash
STUDENT_MODEL=/workspace/models/Qwen3-0.6B
G_OPD_LR=2e-6
```

或：

```bash
STUDENT_MODEL=/workspace/models/Qwen3-4B
G_OPD_LR=1e-6
```

#### OPD + Rubric 单独训练

如果不想用一键脚本，可以直接这样训练 1.7B 当前主线：

```bash
cd /workspace/opd1
STUDENT_MODEL=/workspace/models/Qwen3-1.7B \
TEACHER_MODEL=/workspace/models/Qwen3-4B-Non-Thinking-RL-Math-Step500 \
G_OPD_EXPERIMENT_NAME=146_qwen3_1.7b_teacher_qwen3_4b_opd_rubric_no_second_reroll_lr2e-6 \
G_OPD_LR=2e-6 \
G_OPD_SAVE_FREQ=50 \
G_OPD_RESUME_MODE=disable \
GPT_ROLLOUT_SCORE_ENABLE=True \
GPT_ROLLOUT_SCORE_RANK_GAP_DROP_ENABLE=True \
GPT_ROLLOUT_SCORE_RANK_GAP_DROP_THRESHOLD=0.75 \
GPT_ROLLOUT_SCORE_RANK_GAP_DROP_SCOPE=all \
GPT_ROLLOUT_SCORE_MAX_REROLLOUT_ATTEMPTS=0 \
GPT_ROLLOUT_SCORE_HINT_RETRIES=0 \
GPT_ROLLOUT_SCORE_RUBRIC_ADV_SHIFT_ENABLE=True \
GPT_ROLLOUT_SCORE_RUBRIC_ADV_SHIFT_MODE=history_zscore \
GPT_ROLLOUT_SCORE_RUBRIC_ADV_SHIFT_COEF=0.10 \
GPT_ROLLOUT_SCORE_RUBRIC_ADV_SHIFT_CLIP=0.20 \
GPT_ROLLOUT_SCORE_HISTORY_NUM_BINS=12 \
GPT_ROLLOUT_SCORE_HISTORY_BUCKET_MODE=label \
GPT_ROLLOUT_SCORE_HISTORY_STD_FLOOR=8.0 \
GPT_ROLLOUT_SCORE_HISTORY_NEGATIVE_COEF_SCALE=0.8 \
bash /workspace/opd1/verl/examples/g_opd/run_qwen3-4b-g-opd.sh \
  trainer.total_training_steps=110 \
  actor_rollout_ref.actor.checkpoint.save_contents='["model"]'
```

#### EXOPD

训练入口：

```bash
/workspace/opd1/verl/examples/g_opd/run_qwen3-4b-single-teacher-exopd.sh
```

1.7B 示例：

```bash
cd /workspace/opd1
STUDENT_MODEL=/workspace/models/Qwen3-1.7B \
TEACHER_MODEL=/workspace/models/Qwen3-4B-Non-Thinking-RL-Math-Step500 \
EXOPD_LAMBDA=1.25 \
EXOPD_LR=2e-6 \
G_OPD_EXPERIMENT_NAME=146_qwen3_1.7b_teacher_qwen3_4b_single_teacher_exopd_lambda_1p25 \
G_OPD_SAVE_FREQ=50 \
G_OPD_CKPT_DIR=/G-OPD-checkpoints/146_qwen3_1.7b_teacher_qwen3_4b_single_teacher_exopd_lambda_1p25_save_step_50 \
G_OPD_RESUME_MODE=disable \
bash /workspace/opd1/verl/examples/g_opd/run_qwen3-4b-single-teacher-exopd.sh \
  trainer.total_training_steps=110 \
  actor_rollout_ref.actor.checkpoint.save_contents='["model"]'
```

4B 示例：

```bash
cd /workspace/opd1
STUDENT_MODEL=/workspace/models/Qwen3-4B \
TEACHER_MODEL=/workspace/models/Qwen3-4B-Non-Thinking-RL-Math-Step500 \
EXOPD_LAMBDA=1.25 \
EXOPD_LR=1e-6 \
G_OPD_EXPERIMENT_NAME=146_qwen3_4b_teacher_qwen3_4b_single_teacher_exopd_lambda_1p25 \
G_OPD_SAVE_FREQ=50 \
G_OPD_CKPT_DIR=/G-OPD-checkpoints/146_qwen3_4b_teacher_qwen3_4b_single_teacher_exopd_lambda_1p25_save_step_50 \
G_OPD_RESUME_MODE=disable \
bash /workspace/opd1/verl/examples/g_opd/run_qwen3-4b-single-teacher-exopd.sh \
  trainer.total_training_steps=110 \
  actor_rollout_ref.actor.checkpoint.save_contents='["model"]'
```

#### EOPD

训练入口：

```bash
/workspace/opd1/verl/examples/g_opd/run_qwen3-1.7b-eopd.sh
```

1.7B 示例：

```bash
cd /workspace/opd1
STUDENT_MODEL=/workspace/models/Qwen3-1.7B \
TEACHER_MODEL=/workspace/models/Qwen3-4B-Non-Thinking-RL-Math-Step500 \
EOPD_LR=2e-6 \
EOPD_EXPERIMENT_NAME=eopd_146_qwen3_1.7b_teacher_qwen3_4b_non_thinking_rl_math \
EOPD_SAVE_FREQ=50 \
EOPD_CKPT_DIR=/EOPD-checkpoints/eopd_146_qwen3_1.7b_teacher_qwen3_4b_non_thinking_rl_math_save_step_50 \
EOPD_RESUME_MODE=disable \
bash /workspace/opd1/verl/examples/g_opd/run_qwen3-1.7b-eopd.sh \
  trainer.total_training_steps=110 \
  actor_rollout_ref.actor.checkpoint.save_contents='["model"]'
```

#### TA-OPD

训练入口：

```bash
/workspace/opd1/verl/examples/g_opd/run_qwen3-1.7b-ta-opd.sh
```

1.7B 示例：

```bash
cd /workspace/opd1
STUDENT_MODEL=/workspace/models/Qwen3-1.7B \
TEACHER_MODEL=/workspace/models/Qwen3-4B-Non-Thinking-RL-Math-Step500 \
TA_OPD_LR=2e-6 \
TA_OPD_METHOD=teachability \
TA_OPD_RATIO=0.1 \
TA_OPD_TOPK=16 \
TA_OPD_SEED=42 \
TA_OPD_EXPERIMENT_NAME=qwen3_1.7b_ta_opd_teachability_ratio0.1_k16_seed42_teacher_qwen3_4b_non_thinking_rl_math \
TA_OPD_SAVE_FREQ=50 \
TA_OPD_CKPT_DIR=/TA_OPD-checkpoints/qwen3_1.7b_ta_opd_teachability_ratio0.1_k16_seed42_teacher_qwen3_4b_non_thinking_rl_math_save_step_50 \
TA_OPD_RESUME_MODE=disable \
bash /workspace/opd1/verl/examples/g_opd/run_qwen3-1.7b-ta-opd.sh \
  trainer.total_training_steps=110 \
  actor_rollout_ref.actor.checkpoint.save_contents='["model"]'
```

### 6.3 对比方法一键脚本

当前仓库有一个 4B baseline 顺序脚本，会依次跑 EOPD、TA-OPD、EXOPD，并自动评测和清理 ckpt：

```bash
cd /workspace/opd1
bash /workspace/opd1/run_three_4b_baselines_sequential.sh
```

结果目录默认：

```bash
/workspace/opd1/baseline_runs/<RUN_ID>/
```

合并 summary：

```bash
/workspace/opd1/baseline_runs/<RUN_ID>/all_results.csv
```

### 6.4 Ablation 一键脚本

这些 ablation 都是 Qwen3-1.7B，且都是训练 + 测评 + 保存 summary + 删除 ckpt。

#### 无第二次 rollout

```bash
cd /workspace/opd1
bash /workspace/opd1/run_ablation_qwen3_1p7b_no_second_reroll_train_eval_cleanup.sh
```

#### 无 rank-gap drop

```bash
cd /workspace/opd1
bash /workspace/opd1/run_ablation_qwen3_1p7b_no_rank_gap_drop_train_eval_cleanup.sh
```

#### 无 rubric shift

```bash
cd /workspace/opd1
bash /workspace/opd1/run_ablation_qwen3_1p7b_no_rubric_shift_train_eval_cleanup.sh
```

#### 随机分桶

```bash
cd /workspace/opd1
bash /workspace/opd1/run_ablation_qwen3_1p7b_random_bucket_train_eval_cleanup.sh
```

### 6.5 单独测评命令

通用测评入口：

```bash
/workspace/opd1/math_eval/run_eval_math.sh
```

它会做三件事：

1. 找到 `G_OPD_CKPT_DIR/global_step_<N>/actor` 下的 FSDP checkpoint。
2. 如果没有 `merged_hf`，先把 FSDP checkpoint merge 成 HF 模型。
3. 对 `aime24 aime25 aime26 hmmt26 amc23 math500` 跑 `n=8`，并生成 `summary.csv`。

#### OPD / OPD+Rubric / Vanilla OPD 测评

```bash
cd /workspace/opd1
EXP=146_qwen3_1.7b_teacher_qwen3_4b_opd_rubric_no_second_reroll_lr2e-6
CKPT=/G-OPD-checkpoints/${EXP}_save_step_50
RUN_DIR=/workspace/opd1/eval_runs/${EXP}
mkdir -p "$RUN_DIR"

STUDENT_MODEL=/workspace/models/Qwen3-1.7B \
G_OPD_EXPERIMENT_NAME="$EXP" \
G_OPD_SAVE_FREQ=50 \
G_OPD_CKPT_DIR="$CKPT" \
OUTPUT_DIR="$RUN_DIR/eval_outputs" \
SKIP_DOWNLOAD=1 \
DATASETS="aime24 aime25 aime26 hmmt26 amc23 math500" \
bash /workspace/opd1/math_eval/run_eval_math.sh \
  2>&1 | tee "$RUN_DIR/eval.log"

cp "$RUN_DIR/eval_outputs/summary.csv" "$RUN_DIR/summary.csv"
cat "$RUN_DIR/summary.csv"
```

如果明确知道 actor checkpoint 路径，也可以直接指定：

```bash
FSDP_CKPT_DIR=/G-OPD-checkpoints/${EXP}_save_step_50/global_step_110/actor
```

#### EXOPD 测评

```bash
cd /workspace/opd1
EXP=146_qwen3_1.7b_teacher_qwen3_4b_single_teacher_exopd_lambda_1p25
RUN_DIR=/workspace/opd1/eval_runs/${EXP}
mkdir -p "$RUN_DIR"

EXOPD_LAMBDA=1.25 \
EXOPD_STEP=110 \
G_OPD_EXPERIMENT_NAME="$EXP" \
G_OPD_SAVE_FREQ=50 \
G_OPD_CKPT_DIR=/G-OPD-checkpoints/${EXP}_save_step_50 \
OUTPUT_DIR="$RUN_DIR/eval_outputs" \
SKIP_DOWNLOAD=1 \
DATASETS="aime24 aime25 aime26 hmmt26 amc23 math500" \
bash /workspace/opd1/math_eval/run_eval_qwen3-4b-single-teacher-exopd.sh \
  2>&1 | tee "$RUN_DIR/eval.log"

cp "$RUN_DIR/eval_outputs/summary.csv" "$RUN_DIR/summary.csv"
cat "$RUN_DIR/summary.csv"
```

#### EOPD 测评

```bash
cd /workspace/opd1
EXP=eopd_146_qwen3_1.7b_teacher_qwen3_4b_non_thinking_rl_math
RUN_DIR=/workspace/opd1/eval_runs/${EXP}
mkdir -p "$RUN_DIR"

EOPD_EXPERIMENT_NAME="$EXP" \
EOPD_SAVE_FREQ=50 \
EOPD_CKPT_DIR=/EOPD-checkpoints/${EXP}_save_step_50 \
EOPD_STEP=110 \
OUTPUT_DIR="$RUN_DIR/eval_outputs" \
SKIP_DOWNLOAD=1 \
DATASETS="aime24 aime25 aime26 hmmt26 amc23 math500" \
bash /workspace/opd1/math_eval/run_eval_qwen3-1.7b-eopd.sh \
  2>&1 | tee "$RUN_DIR/eval.log"

cp "$RUN_DIR/eval_outputs/summary.csv" "$RUN_DIR/summary.csv"
cat "$RUN_DIR/summary.csv"
```

#### TA-OPD 测评

```bash
cd /workspace/opd1
EXP=qwen3_1.7b_ta_opd_teachability_ratio0.1_k16_seed42_teacher_qwen3_4b_non_thinking_rl_math
RUN_DIR=/workspace/opd1/eval_runs/${EXP}
mkdir -p "$RUN_DIR"

TA_OPD_EXPERIMENT_NAME="$EXP" \
TA_OPD_SAVE_FREQ=50 \
TA_OPD_CKPT_DIR=/TA_OPD-checkpoints/${EXP}_save_step_50 \
TA_OPD_STEP=110 \
OUTPUT_DIR="$RUN_DIR/eval_outputs" \
SKIP_DOWNLOAD=1 \
DATASETS="aime24 aime25 aime26 hmmt26 amc23 math500" \
bash /workspace/opd1/math_eval/run_eval_qwen3-1.7b-ta-opd.sh \
  2>&1 | tee "$RUN_DIR/eval.log"

cp "$RUN_DIR/eval_outputs/summary.csv" "$RUN_DIR/summary.csv"
cat "$RUN_DIR/summary.csv"
```

## 7. GPT Prompt

GPT rubric 代码在：

```bash
/workspace/opd1/verl/verl/trainer/ppo/gpt_rollout_scorer.py
```

核心模板：

```python
EVALUATION_PROMPT_TEMPLATE
REVISION_SUGGESTION_PROMPT_TEMPLATE
```

### 7.1 GPT rubric prompt

```text
You are a mathematical solution quality evaluation model. Your task is to score a given solution based on the provided [Problem], [Ground Truth Answer], and [Solution to Evaluate].
Please note: the [Ground Truth Answer] contains only the final correct answer and does not include a reference solution or intermediate reasoning. Therefore, you should not require the evaluated solution to match any specific solution method. Instead, you should independently judge whether the solution understands the problem, reasons in a mathematically meaningful way, explores useful directions when needed, reaches a correct or partially useful conclusion, and explains the process clearly.
Please score the solution strictly according to the following 7 rubrics. Each rubric should receive a numeric score from 1.0 to 4.0, allowing only 0.5-point increments:
4.0 points: Excellent performance, with almost no obvious issues.
3.0 points: Generally good, with only minor flaws.
2.0 points: Contains clear issues, but still has some reasonable content.
1.0 point: Poor performance with no effective mathematical content, content that is completely impossible to evaluate, or content unrelated to the problem.
Use 0.5-point scores, such as 2.5 or 3.5, when the quality falls between two adjacent anchor levels.

Scoring principles:
Do not automatically give a high score just because the final answer is correct; the reasoning process must also be reasonable.
Do not automatically give a low score just because the final answer is wrong; meaningful setup, useful intermediate results, or relevant exploration should receive credit.
Do not require a fully formal proof when the solution already shows correct and useful mathematical reasoning.
Do not penalize a solution simply because it differs from common methods, as long as the method is mathematically reasonable.
Do not reward long, random, or repetitive exploration if it does not use the problem structure.
Do not write a new complete solution; only evaluate the given solution itself.
If the final answer is correct but the reasoning is clearly wrong, guessed, or unsupported, lower the scores for Mathematical Rigor, Exploration and Exploitation, and Solution Reasonableness.
If the final answer is wrong but the solution contains correct and relevant intermediate work, give partial credit in the applicable rubrics.
If the solution contains multiple contradictory answers, lower the score for Answer Correctness and Verifiability.

Revision Suggestion Rules
After assigning the rubric scores, estimate the final weighted percentage. If it would be below 50, revision_suggestion is mandatory and must be non-empty. If it would be at least 50, set revision_suggestion to an empty string "".
The revision_suggestion should help the next attempt reason better, but it must not reveal the correct final answer or give a complete solution path.
Write the revision_suggestion in 1 to 2 short sentences and no more than 256 characters.
For a below-50 solution, never leave revision_suggestion blank; this field is used as supervision for a second rollout.
When the solution already has a useful setup, equation, case split, intermediate result, or promising idea, the suggestion should guide the solver to continue from that useful part, check the weak step, and complete the reasoning more carefully.
When the solution is mostly off-track, based on a wrong interpretation, or stuck in unhelpful computation, the suggestion should gently point toward a different broad direction that fits the problem structure, such as using constraints, trying cases, looking for an invariant, setting up an equation, drawing a diagram, bounding quantities, or checking special cases.
When the solution has the right general method but loses accuracy near the end, the suggestion should focus on verifying the final computation, sign, condition, format, or answer extraction.
The revision_suggestion should not mention rubric names, scores, weights, or evaluation policy.
The revision_suggestion should not say "the correct answer is..." or include the ground truth answer.
The revision_suggestion should not provide a hidden shortcut that directly determines the final answer.
The revision_suggestion should not merely say "try again" or "be more rigorous"; it should name a concrete reasoning action.

Problem Classification Rules
Classify the problem itself, ignoring the quality of the submitted solution. Use exactly one problem_domain:
- geometry_visual: synthetic geometry, coordinate geometry, areas, angles, circles, polygons, diagrams, grids, spatial/visual constructions.
- algebra_symbolic: equations, inequalities, functions, expressions, complex numbers, absolute values, symbolic casework, algebraic structure.
- discrete_counting_process: counting, probability, arrangements, finite state processes, recurrences, sequences, digit/time/card/grid counting.
- arithmetic_number_modeling: number theory, divisibility, modular/integer constraints, ratios, rates, units, prices, recipes, word-problem modeling.

Classify difficulty_3 from the shortest reasonable solution to the problem itself:
- easy: direct formula, direct count, direct substitution, or simple proportion; usually 1-2 core reasoning moves.
- medium: needs one non-obvious setup, transformation, recurrence, case split, finite enumeration, or standard theorem; usually 3-6 core reasoning moves.
- hard: needs multiple linked constraints, auxiliary construction, complex angle/area work, multi-case reasoning, or a nonlocal insight.

Rubric 1: Problem Understanding and Constraint Use
Weight: 15%

Rubric 2: Mathematical Rigor
Weight: 15%

Rubric 3: Answer Correctness and Verifiability
Weight: 20%

Rubric 4: Exploration and Exploitation
Weight: 20%

Rubric 5: Solution Reasonableness
Weight: 12.5%

Rubric 6: Expression Fluency
Weight: 10%

Rubric 7: Expression Conciseness
Weight: 7.5%

Output Format
Please strictly output in the requested JSON schema and do not output any extra text.
The JSON must contain only rubric_scores, revision_suggestion, problem_domain, and difficulty_3 at the top level. Do not include weighted_score_1_to_4, final_score_100, or overall_comment.
```

实际 prompt 中还会填入：

```text
[Problem]
{problem}

[Ground Truth Answer]
{ground_truth}

[Solution to Evaluate]
{solution}
```

### 7.2 GPT hint retry prompt

当第一次 rubric 输出低于 50 分但没有给出 usable hint 时，会额外调用一次 hint prompt：

```text
You are writing a concise hint for a second attempt at a math problem.
The previous evaluator scored the solution below 50/100 but did not provide a usable hint. Produce only one revision_suggestion.

Rules:
The revision_suggestion must be non-empty and no more than 256 characters.
Write 1 to 2 short sentences.
Do not reveal the correct final answer or include the ground truth answer.
Do not give a complete solution path.
Do not mention rubric names, scores, weights, or evaluation policy.
Do not merely say "try again" or "be more rigorous"; name a concrete reasoning action.
If the solution has useful partial work, guide the next attempt to continue from it and check the weak step.
If the solution is off-track, point toward a broad problem-appropriate direction such as using constraints, cases, equations, invariants, bounds, or a diagram.

[Problem]
{problem}

[Ground Truth Answer]
{ground_truth}

[Low-Scoring Solution]
{solution}

[Evaluator Notes]
{rubric_feedback}

Output Format
Please strictly output in the requested JSON schema and do not output any extra text.
```

## 8. 输出位置

一键脚本输出：

```bash
/workspace/opd1/eval_runs/${EXP}/train.log
/workspace/opd1/eval_runs/${EXP}/eval.log
/workspace/opd1/eval_runs/${EXP}/summary*.csv
```

OPD / OPD+Rubric / EXOPD checkpoint：

```bash
/G-OPD-checkpoints/${EXP}_save_step_50
```

EOPD checkpoint：

```bash
/EOPD-checkpoints/${EXP}_save_step_50
```

TA-OPD checkpoint：

```bash
/TA_OPD-checkpoints/${EXP}_save_step_50
```

GPT case study / rank-gap dump：

```bash
/workspace/G-OPD-logs/${EXP}_save_step_50/
```
