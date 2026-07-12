# G-OPD
[![arXiv](https://img.shields.io/badge/arXiv-2602.12125-red.svg)](https://arxiv.org/abs/2602.12125)


We propose **G-OPD**, a generalized on-policy distillation framework, in which we introduce a reward scaling factor and a flexible reference model. Building on G-OPD, we propose **ExOPD** (On-Policy Distillation with Reward Extrapolation), which outperforms standard OPD in both same-size and strong-to-weak distillation settings.

---------

## News
- [2026.3.18] We update the code logic for our self-implemented **on-policy self/context-distillation**. For usage, please refer to [On-Policy Context-Distillation](#on-policy-context-distillation).
- [2026.2.17] We release the training code and [data](https://huggingface.co/datasets/Keven16/G-OPD-Training-Data).
- [2026.2.16] We release the evaluation code.
- [2026.2.13] We release our paper on [arxiv](https://arxiv.org/abs/2602.12125).


## Installation
Our code is mainly based on [verl](https://github.com/volcengine/verl) (v0.6.1). To prepare the environment, please follow these steps:

```bash
conda create -n verl python==3.10
conda activate verl
cd verl/
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install math-verify
```


## Training
Our training data is provided in [here](https://huggingface.co/datasets/Keven16/G-OPD-Training-Data).

For the reproducible Qwen3 code-distillation pipelines used in this checkout
(0.6B, 1.7B, and 4B students; training plus LiveCodeBench/EvalPlus evaluation),
see [CODE_EXPERIMENTS.md](CODE_EXPERIMENTS.md). The pipelines download models
and the Skywork code split on demand; local datasets, checkpoints, and results
are intentionally excluded from Git.

### GRPO

We provide an example for GRPO training in the script `verl/examples/grpo_trainer/run_qwen3-4b-math.sh`.

### Single-Teacher Distillation

An example for running G-OPD in the single-teacher distillation setting is

```bash
export WANDB_API_KEY=""
export WANDB_MODE=online
export USED_MODEL="no_api"


aime24_test_path=../G-OPD-Training-Data/AIME2024/test.parquet
aime25_test_path=../G-OPD-Training-Data/AIME2025/test.parquet

test_files="['$aime24_test_path', '$aime25_test_path']"


python3 -m verl.trainer.main_ppo \
       algorithm.adv_estimator=grpo \
        algorithm.rollout_correction.rollout_is=token \
        algorithm.rollout_correction.rollout_is_threshold=5.0 \
        algorithm.rollout_correction.rollout_rs=null \
        algorithm.rollout_correction.bypass_mode=false \
        actor_rollout_ref.rollout.calculate_log_probs=true \
        data.train_files=../G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet \
        data.val_files="$test_files" \
        data.train_batch_size=1024 \
        data.max_prompt_length=2048 \
        data.max_response_length=16384 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.shuffle=True \
        data.seed=42 \
        data.return_raw_chat=True \ # must be specified
        +data.apply_chat_template_kwargs.enable_thinking=False \ # thinking/non-thinking mode
        actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \ # student model path
        +actor_rollout_ref.model.base_model_path=Qwen/Qwen3-1.7B \ # reference model path (either student's initial state (default mode) or teacher's pre-RL variant (reward correction))
        +actor_rollout_ref.ref.model.path=Qwen/Qwen3-4B-Non-Thinking-RL-Math \ # teacher model path
        +actor_rollout_ref.ref.model.base_model_path=Qwen/Qwen3-1.7B \ # useless in the single-teacher distillation setting, can be set to student model
        actor_rollout_ref.actor.optim.lr=1e-5 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True \ # turn on for on-policy distillation experiments
        actor_rollout_ref.actor.policy_loss.lambda_vals=1.25 \ # reward scaling factor
        actor_rollout_ref.actor.ppo_mini_batch_size=1024 \
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
        trainer.experiment_name='qwen3_1.7b_non_thinking_teacher_qwen3_4b_non_thinking_rl_math_exopd_deepmath' \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.save_freq=50 \
        trainer.default_local_dir=/G-OPD-checkpoints/Qwen3-1.7B-Non-Thinking-Teacher-Qwen3-4B-Non-Thinking-RL-Math-ExOPD-DeepMath \
        trainer.test_freq=10 \
        trainer.total_epochs=3 $@


```


We provide more examples in the script `verl/examples/g_opd/run_qwen3-4b-g-opd.sh`.

### Multi-Teacher Distillation
In multi-teacher distillation experiments, we currently only support the two-teacher setting. First, you need to write `math` or `code` into the `extra_info` field of each data sample to indicate which domain teacher should be used:
```python
data = {
  ...,
  extra_info: {
    "opd_teacher": "math",
    ...
  }
}
```

Then, with the current implementation, we manually set `actor_rollout_ref.ref.model.path` to the math teacher's model path and `actor_rollout_ref.ref.model.base_model_path` to the code teacher's model path.

An example should be

```bash
export WANDB_API_KEY=""
export WANDB_MODE=online
export USED_MODEL="no_api"


aime24_test_path=../G-OPD-Training-Data/AIME2024/test.parquet
aime25_test_path=../G-OPD-Training-Data/AIME2025/test.parquet
code_test_path=../G-OPD-Training-Data/Eurus/code_validation.parquet

test_files="['$aime24_test_path', '$aime25_test_path', '$code_test_path']"


python3 -m verl.trainer.main_ppo \
       algorithm.adv_estimator=grpo \
        algorithm.rollout_correction.rollout_is=token \
        algorithm.rollout_correction.rollout_is_threshold=5.0 \
        algorithm.rollout_correction.rollout_rs=null \
        algorithm.rollout_correction.bypass_mode=false \
        actor_rollout_ref.rollout.calculate_log_probs=true \
        data.train_files=../G-OPD-Training-Data/math_and_code/train.parquet \
        data.val_files="$test_files" \
        data.train_batch_size=1024 \
        data.max_prompt_length=2048 \
        data.max_response_length=16384 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.shuffle=True \
        data.seed=42 \
        data.return_raw_chat=True \
        +data.apply_chat_template_kwargs.enable_thinking=False \
        actor_rollout_ref.model.path=Qwen/Qwen3-4B \ # student model path
        +actor_rollout_ref.model.base_model_path=Qwen/Qwen3-4B \ # reference model path
        +actor_rollout_ref.ref.model.path=Qwen3-4B-Non-Thinking-RL-Math \ # manually set to math teacher
        +actor_rollout_ref.ref.model.base_model_path=Qwen3-4B-Non-Thinking-RL-Code \ # manually set to code teacher
        actor_rollout_ref.actor.optim.lr=1e-5 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True \ # turn on for on-policy distillation experiments
        actor_rollout_ref.actor.policy_loss.lambda_vals=1.25 \ # reward scaling factor
        actor_rollout_ref.actor.policy_loss.multi_teacher_distill=true \ # turn on for multi-teacher distillation
        actor_rollout_ref.actor.ppo_mini_batch_size=1024 \
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
        trainer.critic_warmup=0 \
        trainer.val_before_train=True \
        trainer.logger='["console","wandb"]' \
        trainer.log_val_generations=10 \
        trainer.project_name='on-policy-distillation' \
        trainer.experiment_name='Qwen3-4B-Non-Thinking-Multi-Teacher-Distill-ExOPD' \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.save_freq=50 \
        trainer.default_local_dir=/G-OPD-checkpoints/Qwen3-4B-Non-Thinking-Multi-Teacher-Distill-ExOPD \
        trainer.test_freq=10 \
        trainer.total_epochs=3 $@
```

### On-Policy Context-Distillation

We also provide our self-implemented code for on-policy self/context-distillation methods. In such cases, the teacher distribution is the in-context learning distribution produced by the student model itself (we do not perform weight EMA here as we find it leads to training collapse easily, so we fix the teacher's weights to be the student's initial state) when it is given additional context. The additional context can be [a ground-truth solution](https://arxiv.org/pdf/2601.18734), [external feedback](https://arxiv.org/pdf/2601.20802), or [high-level experience](https://arxiv.org/pdf/2602.12275). 


We provide an example to perform [OPSD](https://arxiv.org/pdf/2601.18734) in `./verl/examples/g_opd/run_qwen3-4b-context-distill.sh`, while our framework can also support other on-policy self-distillation methods. The corresponding data format can refer to this [format](https://huggingface.co/datasets/Keven16/OPSD-Example-Data). If you want to run on-policy context-distillation under other context settings, you can refer to and modify the relevant code in `./verl/verl/trainer/ppo/ref_input_utils.py`.

## Evaluation

### Math Reasoning Evaluation
Math evaluation data is in the ``data/`` folder. Math evaluation code and script are in the ``math_eval/`` folder. 

```bash
cd math_eval/
sh scripts/run_eval_math.sh
```

### Code Generation Evaluation
Our evaluation is mainly based on the code provided in [Absolute-Zero-Reasoner](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner). 

#### EvalPlus

```bash
CUDA_VISIBLE_DEVICES=0 bash code_eval/scripts/run_evalplus.sh humaneval <MODEL_PATH>  0 1.0 1.0 4
```

#### LiveCodeBench

Download data first

```bash
git clone https://hf-mirror.com/datasets/livecodebench/code_generation_lite code_eval/coding/LiveCodeBench/code_generation_lite
```

Evaluation
```bash
bash code_eval/scripts/run_lcb_gen.sh --model Qwen3-4B-NonThinking --local_model_path  <MODEL_PATH>
```


## Acknowledgments
Our training code is mainly based on [verl](https://github.com/volcengine/verl). Our evaluation code is mainly based on [Absolute-Zero-Reasoner](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner), which is built opon [EvalPlus](https://github.com/evalplus/evalplus) and [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench).



## Citation
If you find our work helpful, please kindly cite as
```bibtex
@article{yang2026learning,
  title={Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation},
  author={Yang, Wenkai and Liu, Weijie and Xie, Ruobing and Yang, Kai and Yang, Saiyong and Lin, Yankai},
  journal={arXiv preprint arXiv:2602.12125},
  year={2026}
}
```