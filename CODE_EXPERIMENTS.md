# Qwen3 code-distillation experiments

This repository includes sequential pipelines for Qwen3 0.6B, 1.7B, and 4B
students distilled from `Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step300`.
Each pipeline runs EOPD, vanilla OPD, TA-OPD, and single-teacher ExOPD, then
evaluates the saved checkpoint with LiveCodeBench v5 and EvalPlus
(HumanEval+ and MBPP), using eight samples by default.

## Environment

Create the verl environment as described in [README.md](README.md), then install
the evaluation packages:

```bash
cd verl
pip install -e .
cd ../code_eval/coding/LiveCodeBench
pip install -e .
cd ../evalplus
pip install -e .
```

The scripts download missing Hugging Face models and the
`Skywork/Skywork-OR1-RL-Data` code split into `/workspace/models` and `data/`.
Override `TEACHER_MODEL`, `STUDENT_MODEL`, `TRAIN_SRC`, or `TRAIN_FILE` when
using different storage paths.

W&B credentials are read from the environment and are never stored in the
repository:

```bash
export WANDB_API_KEY="..."
export WANDB_MODE=online
```

## Run training and evaluation

From the repository root, choose one student size:

```bash
# Qwen3-0.6B student
bash code_sequential/run_three_0.6_code_baselines_sequential_qwen3_4b.sh

# Qwen3-1.7B student
bash code_sequential/run_three_1.7_code_baselines_sequential_qwen3_4b.sh

# Qwen3-4B student
bash code_sequential/run_three_4b_code_baselines_sequential_qwen3_4b.sh
```

Useful overrides:

```bash
TOTAL_STEPS=109 SAVE_FREQ=109 KEEP_CKPT=1 \
  bash code_sequential/run_three_4b_code_baselines_sequential_qwen3_4b.sh
```

- `KEEP_CKPT=0` (default) removes each checkpoint only after its evaluation
  succeeds; use `KEEP_CKPT=1` to retain checkpoints.
- `DRY_RUN=1` prints training/evaluation commands without executing them.
- `RUN_LCB`, `RUN_EVALPLUS`, `RUN_HUMANEVAL`, and `RUN_MBPP` independently
  enable evaluation components in the common evaluator. The full sequential
  pipeline validates that all nine metrics are present, so disabling a
  component there intentionally makes final validation fail.
- `LCB_N=8`, `LCB_RELEASE=v5`, and `LCB_MAX_TOKENS=2048` are the common
  pipeline defaults. The common evaluator defaults `EVALPLUS_MAX_TOKENS` to
  4096, while the 0.6B pipeline explicitly sets it to 2048; set it explicitly
  when matched generation lengths are required.
- Vanilla OPD uses the real code reward for logging. TA-OPD and ExOPD use the
  configured constant reward where required by their baseline definitions.

For a Chinese explanation of the evaluation call chain, datasets, metric
definitions, standalone checkpoint/base-model commands, and troubleshooting,
see [CODE_EVALUATION_GUIDE_ZH.md](CODE_EVALUATION_GUIDE_ZH.md).

Each run writes only local artifacts:

- `baseline_runs/<UTC run id>/status.tsv`: stage progress and failures.
- `baseline_runs/<UTC run id>/<baseline>/train.log`: training log.
- `baseline_runs/<UTC run id>/<baseline>/eval.log`: evaluation log.
- `baseline_runs/<UTC run id>/<baseline>/eval_outputs/summary.csv`: per-baseline
  metrics.
- `baseline_runs/<UTC run id>/all_results.csv`: combined metrics.

These generated files, model downloads, datasets, checkpoints, and W&B local
state are ignored by Git.

## Evaluate an untrained base model

The common evaluator can also run directly on a Hugging Face model directory:

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
MODEL_PATH=/workspace/models/Qwen3-4B \
MODEL_NAME="qwen3_4b_base_matched_lcbv5_n8_${RUN_ID}" \
OUTPUT_DIR="$PWD/code_eval/results/qwen3_4b_base_matched_${RUN_ID}" \
FSDP_CKPT_DIR=/workspace/models/Qwen3-4B \
EVALPLUS_FORCE_BASE_PROMPT=0 \
bash code_eval/run_eval_code.sh
```

For a base model, `FSDP_CKPT_DIR` is only used to satisfy the common entry-point
interface; `MODEL_PATH` points directly to the already merged Hugging Face
model.
