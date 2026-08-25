# Code 测评使用说明

本文说明当前仓库中代码生成测评的入口、执行逻辑、数据集、指标和常用命令。除非特别说明，所有命令都应在仓库根目录执行。

## 1. 应该运行哪个脚本

| 需求 | 推荐入口 | 说明 |
| --- | --- | --- |
| 从训练开始，完整比较所有蒸馏方法 | `code_sequential/run_three_<size>_code_baselines_sequential_qwen3_4b.sh` | 串行执行训练、测评、汇总和 checkpoint 清理 |
| 只测一个已有 FSDP checkpoint | `code_eval/run_eval_code.sh` | 必要时先把 FSDP 分片合并为 Hugging Face 模型 |
| 只测一个 Hugging Face 模型目录 | `code_eval/run_eval_code.sh` | 直接使用已有模型权重，不执行训练 |
| 按实验名称测 EOPD、OPD、TA-OPD 或 ExOPD checkpoint | `code_eval/run_eval_qwen3-*-code.sh` | 这些是参数包装器，最终仍调用 `run_eval_code.sh` |
| 手工单独运行某个原始 benchmark | `code_eval/scripts/run_evalplus.sh` 或 `run_lcb_gen.sh` | 更底层，不负责统一汇总；一般优先使用通用入口 |

虽然文件名中是 `run_three`，当前脚本实际依次运行四种方法：

1. EOPD
2. TA-OPD
3. single-teacher ExOPD
4. vanilla OPD

0.6B、1.7B 和 4B 学生模型分别使用：

```bash
bash code_sequential/run_three_0.6_code_baselines_sequential_qwen3_4b.sh
bash code_sequential/run_three_1.7_code_baselines_sequential_qwen3_4b.sh
bash code_sequential/run_three_4b_code_baselines_sequential_qwen3_4b.sh
```

## 2. 测评逻辑

完整调用链如下：

```text
FSDP checkpoint 或 Hugging Face 模型
                  |
                  v
       检查模型文件是否完整
                  |
                  v
   必要时将 FSDP 分片合并到 merged_hf/
                  |
          +-------+--------+
          |                |
          v                v
 LiveCodeBench v5      EvalPlus
   生成并判题       HumanEval + MBPP
          |                |
          +-------+--------+
                  |
                  v
       summarize_code_eval.py
                  |
                  v
              summary.csv
```

`code_eval/run_eval_code.sh` 具体执行以下步骤：

1. 确定待测模型。
   - 若设置了 `MODEL_PATH` 且该目录已经包含 Hugging Face 权重、`config.json` 和 `tokenizer_config.json`，直接使用。
   - 否则从 `FSDP_CKPT_DIR` 读取 `fsdp_config.json`，检查所有 rank 分片，然后调用 `verl.model_merger`，默认合并到 `FSDP_CKPT_DIR/merged_hf`。
   - 如果只提供 `G_OPD_CKPT_DIR`，脚本会查找其中数值最大的 `global_step_*` actor checkpoint。
2. 运行 LiveCodeBench。
   - 检查所需数据文件；默认缺失时从 `livecodebench/code_generation_lite` 自动下载。
   - 使用 vLLM 对每道题生成多份代码，并执行测试用例得到 `graded_list`。
3. 运行 EvalPlus。
   - 分别对 HumanEval 和 MBPP 生成代码。
   - 执行生成代码，记录每个样本的 `base_status` 和 `plus_status`。
4. 调用 `code_eval/summarize_code_eval.py`，将三个 benchmark 的结果写入统一 CSV。

完整 `run_three_*` 流水线会对每种方法重复“训练 → 检查 checkpoint → 测评 → 汇总”。只有该方法测评成功后，才会在默认配置下删除它的 checkpoint。

## 3. 会测什么

### LiveCodeBench v5

- 场景：`codegeneration`
- 默认 release：`v5`
- 当前脚本将 `v5` 对应的数据文件解析为 `test5.jsonl`
- 测试内容：较新的竞赛式编程题，模型需要根据题目生成完整解答，然后通过输入输出测试判题
- 默认每题生成 8 份答案
- 单份答案是否正确取自 LiveCodeBench 输出中的 `graded_list`

如果设置 `LCB_RELEASE=release_v5`，数据检查逻辑会使用从 `test.jsonl` 到 `test5.jsonl` 的累计文件；它与默认的 `v5` 不是同一个取值。

### HumanEval+

- 当前仓库数据：`code_eval/data/HumanEvalPlus.jsonl`，共 164 道题
- 任务形式：根据函数签名、文档字符串和示例补全 Python 函数
- 当前汇总规则：一份答案必须同时满足 `base_status=pass` 和 `plus_status=pass` 才算正确
- 因而 CSV 中的 `HumanEval+` 是增强测试口径，不只是原始 HumanEval 测试

### MBPP

- 当前仓库数据：`code_eval/data/MbppPlus.jsonl`，共 378 道题
- 任务形式：根据自然语言描述和断言编写 Python 函数
- EvalPlus 会生成 base 和 plus 两类测试状态
- 但当前 `summarize_code_eval.py` 对 MBPP 只检查 `base_status=pass`，没有要求 `plus_status=pass`
- 因而 CSV 列名是 `MBPP`，不应把该结果当作 `MBPP+`

这些 benchmark 都会实际执行模型生成的代码。建议在隔离容器或其他受控环境中运行，不要在保存敏感数据的宿主环境直接执行未知模型输出。

## 4. 指标含义

默认每题采样 `k=8`。设某道题 8 个样本中有 `c` 个通过测试：

| 指标 | 单题计分 | 含义 |
| --- | --- | --- |
| `AVG@8` | `c / 8` | 平均样本正确率；越高表示随机采一份答案越容易正确 |
| `P@8` | `1[c > 0]` | 8 份答案中至少一份正确的题目比例 |
| `maj@8` | `1[c > 4]` | 8 份答案中严格多数正确的题目比例，即至少 5 份正确 |

CSV 中的值已经乘以 100，保留两位小数，但不带 `%` 符号。例如 `37.50` 表示 37.50%。

需要注意：

- 当前汇总列名固定写成 `@8`。
- LiveCodeBench 实际会统计 `graded_list` 中存在的全部样本；如果把 `LCB_N` 改成其他值，列名仍显示 `@8`。
- LiveCodeBench 的汇总列名也固定写成 `LiveCodeBenchv5`；改变 `LCB_RELEASE` 不会自动改变列名。
- EvalPlus 汇总固定取每道题前 8 个样本；不足 8 个样本的题会被跳过。因此若使用通用入口生成汇总，不要把 `EVALPLUS_N_SAMPLES` 设为小于 8，也不要启用 `EVALPLUS_GREEDY=1`。
- 某个结果文件不存在或没有可汇总样本时，对应指标会显示 `NA`。
- 单独运行 `run_eval_code.sh` 时可以关闭任意 benchmark；但完整 `run_three_*` 流水线会校验九个指标均非 `NA`。在完整流水线中关闭任意组件，当前实现会在汇总校验阶段失败。

## 5. 环境准备

先按根目录 `README.md` 配置 verl，再安装两个测评包：

```bash
cd verl
pip install -e .
cd ../code_eval/coding/LiveCodeBench
pip install -e .
cd ../evalplus
pip install -e .
cd ../../..
```

通用评测脚本按 Linux/bash、Python 3、CUDA 和 vLLM 环境编写。默认使用 4 张 GPU：`0,1,2,3`，tensor parallel size 为 4。

## 6. 常用运行方式

### 6.1 完整训练并测评四种方法

建议第一次运行时保留 checkpoint：

```bash
KEEP_CKPT=1 \
bash code_sequential/run_three_1.7_code_baselines_sequential_qwen3_4b.sh
```

只打印即将执行的训练和测评命令：

```bash
DRY_RUN=1 \
bash code_sequential/run_three_1.7_code_baselines_sequential_qwen3_4b.sh
```

默认 `KEEP_CKPT=0`。每个方法测评成功后，该方法的 checkpoint 会被删除；训练或测评失败时不会执行该方法的清理。

### 6.2 测评一个指定 FSDP checkpoint

```bash
FSDP_CKPT_DIR=/G-OPD-checkpoints/my_experiment/global_step_109/actor \
MODEL_NAME=my_experiment_step109 \
OUTPUT_DIR="$PWD/code_eval/results/my_experiment_step109" \
bash code_eval/run_eval_code.sh
```

如果 `MODEL_PATH` 未设置，合并后的模型默认写到：

```text
/G-OPD-checkpoints/my_experiment/global_step_109/actor/merged_hf
```

checkpoint 目录至少应包含：

```text
fsdp_config.json
model_world_size_<N>_rank_0.pt
...
model_world_size_<N>_rank_<N-1>.pt
```

### 6.3 自动选择一个实验的最新 checkpoint

```bash
G_OPD_CKPT_DIR=/G-OPD-checkpoints/my_experiment \
MODEL_NAME=my_experiment_latest \
OUTPUT_DIR="$PWD/code_eval/results/my_experiment_latest" \
bash code_eval/run_eval_code.sh
```

脚本会在该实验目录下查找数值最大的 `global_step_*` actor checkpoint。

### 6.4 测评一个 Hugging Face 模型目录

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"

MODEL_PATH=/workspace/models/Qwen3-4B \
FSDP_CKPT_DIR=/workspace/models/Qwen3-4B \
MODEL_NAME="qwen3_4b_base_${RUN_ID}" \
OUTPUT_DIR="$PWD/code_eval/results/qwen3_4b_base_${RUN_ID}" \
EVALPLUS_FORCE_BASE_PROMPT=0 \
bash code_eval/run_eval_code.sh
```

这里 `MODEL_PATH` 指向已经可由 Transformers/vLLM 加载的 Hugging Face 模型。`FSDP_CKPT_DIR` 只是为了满足当前通用入口的参数检查，不会触发 FSDP 合并。

### 6.5 只跑 LiveCodeBench

```bash
RUN_LCB=1 \
RUN_EVALPLUS=0 \
FSDP_CKPT_DIR=/path/to/global_step_109/actor \
MODEL_NAME=my_model_lcb \
OUTPUT_DIR="$PWD/code_eval/results/my_model_lcb" \
bash code_eval/run_eval_code.sh
```

### 6.6 只跑 HumanEval+

```bash
RUN_LCB=0 \
RUN_EVALPLUS=1 \
RUN_HUMANEVAL=1 \
RUN_MBPP=0 \
FSDP_CKPT_DIR=/path/to/global_step_109/actor \
MODEL_NAME=my_model_humaneval \
OUTPUT_DIR="$PWD/code_eval/results/my_model_humaneval" \
bash code_eval/run_eval_code.sh
```

### 6.7 只跑 MBPP

```bash
RUN_LCB=0 \
RUN_EVALPLUS=1 \
RUN_HUMANEVAL=0 \
RUN_MBPP=1 \
FSDP_CKPT_DIR=/path/to/global_step_109/actor \
MODEL_NAME=my_model_mbpp \
OUTPUT_DIR="$PWD/code_eval/results/my_model_mbpp" \
bash code_eval/run_eval_code.sh
```

## 7. 常用参数

### 通用开关和路径

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `FSDP_CKPT_DIR` | 无 | 指定一个 actor FSDP checkpoint |
| `G_OPD_CKPT_DIR` | 无 | 指定实验根目录并自动选择最新 step |
| `MODEL_PATH` | `<FSDP_CKPT_DIR>/merged_hf` | 实际交给 vLLM 的 Hugging Face 模型目录 |
| `MODEL_NAME` | 根据 checkpoint 自动生成 | 输出文件和汇总中的模型名称 |
| `OUTPUT_DIR` | `code_eval/results` | EvalPlus 结果、模型软链接及 summary 的根目录 |
| `SUMMARY_FILE` | `<OUTPUT_DIR>/<MODEL_NAME>_code_summary.csv` | 汇总 CSV 路径 |
| `RUN_LCB` | `1` | 是否运行 LiveCodeBench |
| `RUN_EVALPLUS` | `1` | 是否运行 EvalPlus |
| `RUN_HUMANEVAL` | `1` | 是否运行 HumanEval |
| `RUN_MBPP` | `1` | 是否运行 MBPP |

### LiveCodeBench

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `LCB_RELEASE` | `v5` | 数据 release |
| `LCB_N` | `8` | 每题采样数 |
| `LCB_TEMPERATURE` | `1.0` | 采样温度 |
| `LCB_TOP_P` | `1.0` | top-p |
| `LCB_MAX_TOKENS` | `2048` | 单个答案最大生成 token 数 |
| `LCB_GPUS` | `0,1,2,3` | 可见 GPU |
| `LCB_TP` | `4` | tensor parallel size |
| `LCB_BATCH_SIZE` | `64` | cache batch size |
| `LCB_AUTO_DOWNLOAD_DATA` | `1` | 数据缺失时是否自动下载 |

### EvalPlus

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `EVALPLUS_BACKEND` | `vllm` | 推理后端 |
| `EVALPLUS_N_SAMPLES` | `8` | 每题采样数 |
| `EVALPLUS_TEMPERATURE` | `1.0` | 采样温度 |
| `EVALPLUS_MAX_TOKENS` | `4096` | 通用入口中单个答案最大生成 token 数 |
| `EVALPLUS_GPUS` | 与 `LCB_GPUS` 相同 | 可见 GPU |
| `EVALPLUS_TP` | 与 `LCB_TP` 相同 | tensor parallel size |
| `EVALPLUS_DTYPE` | `bfloat16` | 推理 dtype |
| `EVALPLUS_FORCE_BASE_PROMPT` | `0` | 是否强制使用 base prompt |
| `EVALPLUS_ALLOW_OVERWRITE` | `1` | 是否重新生成并备份已有 result 文件 |

0.6B 完整流水线会显式把 `EVALPLUS_MAX_TOKENS` 设为 `2048`；1.7B/4B 共享流水线当前未覆盖该变量，因此继承通用评测器的 `4096`。需要严格横向对比时，应在命令中显式设置同一个值，例如：

```bash
EVALPLUS_MAX_TOKENS=2048 \
bash code_sequential/run_three_1.7_code_baselines_sequential_qwen3_4b.sh
```

## 8. 输出文件

单模型通用测评的主要输出：

```text
<OUTPUT_DIR>/
├── <MODEL_NAME>_code_summary.csv
├── evalplus_results/
│   ├── humaneval/
│   └── mbpp/
└── lcb_model_paths/
    └── <MODEL_NAME> -> <MODEL_PATH>
```

LiveCodeBench 的原始输出默认位于：

```text
code_eval/coding/LiveCodeBench/lcb_outputs/<MODEL_NAME>/
```

完整 `run_three_*` 流水线的输出：

```text
baseline_runs/<UTC run id>/
├── status.tsv
├── all_results.csv
├── eopd/
│   ├── train.log
│   ├── eval.log
│   └── eval_outputs/summary.csv
├── ta_opd/
├── exopd/
└── opd/
```

`all_results.csv` 是四种方法的最终横向汇总。`status.tsv` 记录每个方法的训练、测评和清理状态。

## 9. 常见问题

### 结果全部是 `NA`

优先检查：

1. 对应 benchmark 是否被 `RUN_*` 开关关闭。
2. `eval.log` 中是否出现结果文件路径不存在的 warning。
3. EvalPlus 是否确实为每题生成了至少 8 份答案。
4. `MODEL_NAME`、`LCB_N` 或 temperature 改变后，脚本计算出的 LCB 文件名是否与实际输出一致。

### 提示 checkpoint 不完整

确认 `global_step_<N>/actor` 下存在 `fsdp_config.json` 和全部 rank 分片。训练进程可能还在保存 checkpoint，此时应等待保存完成后再测。

### GPU 数量不是 4

例如只使用两张卡：

```bash
LCB_GPUS=0,1 \
LCB_TP=2 \
EVALPLUS_GPUS=0,1 \
EVALPLUS_TP=2 \
bash code_eval/run_eval_code.sh
```

模型必须能在对应 GPU 数量和显存条件下被 vLLM 加载。

### 只想比较模型，不想删除 checkpoint

直接运行 `code_eval/run_eval_code.sh` 不会删除源 checkpoint。只有完整 `run_three_*` 流水线包含清理逻辑；运行该流水线时设置 `KEEP_CKPT=1` 即可保留。
