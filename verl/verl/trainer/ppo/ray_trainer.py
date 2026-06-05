# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger


def _is_truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off", "none", "null"}
    return bool(value)


def _g_opd_truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
        ref_tokenizer=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
            ref_tokenizer: Optional tokenizer for reference model. If provided and different from tokenizer,
                re-tokenization will be performed before computing ref log probs.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        # Store ref_tokenizer for re-tokenization when ref model uses different tokenizer
        self.ref_tokenizer = ref_tokenizer
        self.use_ref_retokenization = ref_tokenizer is not None

        # Critique distillation configuration
        self.critique_vllm_url = config.algorithm.get("critique_vllm_url", None)
        self.use_context_distillation = self.critique_vllm_url is not None
        self.critique_model = config.algorithm.get("critique_model", None)
        self.max_critique_tokens = config.algorithm.get("max_critique_tokens", 2048)
        self.critique_temperature = config.algorithm.get("critique_temperature", 0.0)
        self.critique_top_p = config.algorithm.get("critique_top_p", 1.0)

        # Ref solution distillation configuration
        self.use_ref_solution_distillation = config.algorithm.get("use_ref_solution_distillation", False)

        if self.use_ref_retokenization or self.use_context_distillation or self.use_ref_solution_distillation:
            return_raw_chat = config.data.get("return_raw_chat", False)
            if not return_raw_chat:
                raise ValueError(
                    "When using a different tokenizer for ref model (ref_tokenizer is provided), "
                    "context distillation (critique_vllm_url is provided), "
                    "or ref solution distillation (use_ref_solution_distillation=True), "
                    "you must set data.return_raw_chat=True in config to enable re-tokenization. "
                    "This is needed to access the original messages for re-tokenizing with ref model's chat template."
                )
        
        if self.use_context_distillation:
            print(f"Context distillation enabled with vLLM URL: {self.critique_vllm_url}")
        if self.use_ref_solution_distillation:
            print("Ref solution distillation enabled")

        # Store base model paths for corrected reward computation
        self.base_model_path = config.actor_rollout_ref.model.get("base_model_path", None)
        self.ref_base_model_path = config.actor_rollout_ref.ref.get("model", None)
        if self.ref_base_model_path is not None:
            self.ref_base_model_path = self.ref_base_model_path.get("base_model_path", None)
        self.use_base_models = self.base_model_path is not None and self.ref_base_model_path is not None
        
        if self.use_base_models:
            print(f"Corrected reward enabled with base models:")
            print(f"  Actor base model: {self.base_model_path}")
            print(f"  Ref base model: {self.ref_base_model_path}")

                
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )
            for key in (
                "gpt_rollout_score",
                "gpt_rollout_score_100",
                "gpt_rollout_weighted_score_1_to_4",
                "gpt_rollout_rubric_scores",
                "gpt_rollout_reason",
                "gpt_rollout_revision_suggestion",
                "gpt_rollout_error",
                "gpt_rollout_model",
                "gpt_rollout_pass_score_threshold",
                "gpt_rollout_initial_score",
                "gpt_rollout_initial_score_100",
                "gpt_rollout_initial_weighted_score_1_to_4",
                "gpt_rollout_initial_rubric_scores",
                "gpt_rollout_initial_reason",
                "gpt_rollout_initial_revision_suggestion",
                "gpt_rollout_initial_error",
                "gpt_rollout_initial_model",
                "gpt_rollout_initial_pass_score_threshold",
                "gpt_rollout_reroll_count",
            ):
                if key in batch.non_tensor_batch:
                    value = batch.non_tensor_batch[key]
                    if hasattr(value, "tolist"):
                        value = value.tolist()
                    reward_extra_infos_to_dump.setdefault(key, value)

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _json_safe_value(self, value):
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {k: self._json_safe_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe_value(v) for v in value]
        return value

    def _get_non_tensor_row_value(self, batch: DataProto, key: str, idx: int, default=None):
        if key not in batch.non_tensor_batch:
            return default

        value = batch.non_tensor_batch[key]
        try:
            row_value = value[idx]
        except Exception:
            return default
        return self._json_safe_value(row_value)

    def _decode_rollout_prompt_response(self, batch: DataProto, idx: int) -> tuple[str, str]:
        prompt_ids = batch.batch["prompts"]
        response_ids = batch.batch["responses"]
        attention_mask = batch.batch.get("attention_mask", None)
        response_mask = batch.batch.get("response_mask", None)
        prompt_len = prompt_ids.shape[-1]

        valid_prompt_ids = prompt_ids[idx]
        if attention_mask is not None:
            valid_prompt_len = int(attention_mask[idx, :prompt_len].sum().item())
            valid_prompt_ids = prompt_ids[idx, -valid_prompt_len:] if valid_prompt_len > 0 else prompt_ids[idx, :0]

        valid_response_ids = response_ids[idx]
        if response_mask is not None:
            valid_response_len = int(response_mask[idx].sum().item())
            valid_response_ids = response_ids[idx, :valid_response_len]
        elif attention_mask is not None:
            valid_response_len = int(attention_mask[idx, prompt_len:].sum().item())
            valid_response_ids = response_ids[idx, :valid_response_len]

        return (
            self.tokenizer.decode(valid_prompt_ids.detach().cpu().tolist(), skip_special_tokens=True),
            self.tokenizer.decode(valid_response_ids.detach().cpu().tolist(), skip_special_tokens=True),
        )

    def _get_gpt_case_study_low_idxs(
        self,
        scores_100: list,
        threshold_100: float,
        include_errors: bool,
    ) -> list[int]:
        low_idxs = []
        for idx, score_100 in enumerate(scores_100):
            try:
                score_value = float(score_100)
            except (TypeError, ValueError):
                if include_errors:
                    low_idxs.append(idx)
                continue
            if np.isfinite(score_value) and score_value <= threshold_100:
                low_idxs.append(idx)
        return low_idxs

    def _maybe_log_gpt_rollout_case_studies(
        self,
        batch: DataProto,
        result: dict,
        scorer_config: dict,
        timing_raw: dict,
    ) -> None:
        case_study_dir = scorer_config.get("case_study_dir", None)
        if not case_study_dir or str(case_study_dir).strip().lower() in {"none", "null", "false", "0"}:
            return

        threshold_config = scorer_config.get("case_study_threshold_100", None)
        if threshold_config is None or str(threshold_config).strip().lower() in {"", "none", "null"}:
            threshold_config = scorer_config.get("min_score_100", 50.0)
        threshold_100 = float(threshold_config)
        max_cases_per_step = int(scorer_config.get("case_study_max_per_step", 16))
        if max_cases_per_step <= 0:
            return

        include_errors = _is_truthy(scorer_config.get("case_study_include_errors", False))
        low_idxs = self._get_gpt_case_study_low_idxs(
            result["scores_100"],
            threshold_100=threshold_100,
            include_errors=include_errors,
        )
        if not low_idxs:
            return
        selected_idxs = low_idxs[:max_cases_per_step]

        entries = []
        for idx in selected_idxs:
            prompt_text, response_text = self._decode_rollout_prompt_response(batch=batch, idx=idx)
            extra_info = self._get_non_tensor_row_value(batch, "extra_info", idx, {}) or {}
            reward_model = self._get_non_tensor_row_value(batch, "reward_model", idx, {}) or {}
            ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
            if ground_truth is None and isinstance(extra_info, dict):
                ground_truth = extra_info.get("answer")

            entries.append(
                {
                    "step": self.global_steps,
                    "row_idx": idx,
                    "threshold_100": threshold_100,
                    "input": prompt_text,
                    "output": response_text,
                    "student_first_input": prompt_text,
                    "student_first_output": response_text,
                    "student_second_input": None,
                    "student_second_output": None,
                    "student_second_attempted": False,
                    "student_second_attempt": None,
                    "student_second_accepted": None,
                    "student_second_rejected_reason": None,
                    "student_second_score_delta_100": None,
                    "student_second_gpt_score": None,
                    "student_second_gpt_score_100": None,
                    "student_second_gpt_weighted_score_1_to_4": None,
                    "student_second_gpt_rubric_scores": None,
                    "student_second_gpt_reason": "",
                    "student_second_gpt_revision_suggestion": "",
                    "student_second_gpt_error": "",
                    "student_second_gpt_model": None,
                    "student_reroll_count": 0,
                    "problem": extra_info.get("problem") if isinstance(extra_info, dict) else None,
                    "gts": ground_truth,
                    "data_source": self._get_non_tensor_row_value(batch, "data_source", idx, None),
                    "request_id": self._get_non_tensor_row_value(batch, "request_id", idx, None),
                    "extra_info": extra_info,
                    "reward_model": reward_model,
                    "gpt_score": result["scores"][idx],
                    "gpt_score_100": result["scores_100"][idx],
                    "gpt_weighted_score_1_to_4": result["weighted_scores_1_to_4"][idx],
                    "gpt_rubric_scores": result["rubric_scores"][idx],
                    "gpt_reason": result["reasons"][idx],
                    "gpt_revision_suggestion": result["revision_suggestions"][idx],
                    "gpt_error": result["errors"][idx],
                    "gpt_model": result["models"][idx],
                }
            )

        result["_case_study_dir"] = case_study_dir
        result["_case_study_entries"] = entries
        self._write_gpt_rollout_case_studies(
            case_study_dir=case_study_dir,
            entries=entries,
            timing_raw=timing_raw,
            selected_count=len(selected_idxs),
            low_count=len(low_idxs),
        )

    def _write_gpt_rollout_case_studies(
        self,
        case_study_dir: str,
        entries: list[dict],
        timing_raw: dict,
        selected_count: int,
        low_count: int,
    ) -> None:
        if not entries:
            return

        with marked_timer("dump_gpt_low_score_cases", timing_raw, color="green"):
            os.makedirs(case_study_dir, exist_ok=True)
            filename = os.path.join(case_study_dir, f"{self.global_steps}.jsonl")
            lines = [json.dumps(self._json_safe_value(entry), ensure_ascii=False) for entry in entries]

            with open(filename, "w") as f:
                f.write("\n".join(lines) + "\n")

            self._debug_progress(
                f"gpt_case_study:dumped selected={selected_count} low_count={low_count} file={filename}"
            )

    def _maybe_record_gpt_case_study_reroll_attempts(
        self,
        initial_score_result: dict,
        reroll_prompt_batch: DataProto,
        reroll_scoring_batch: DataProto,
        reroll_score_result: dict,
        row_idxs: list[int],
        accepted_batch_idxs: list[int],
        attempt: int,
    ) -> None:
        entries = initial_score_result.get("_case_study_entries")
        if not entries:
            return

        entry_by_idx = {int(entry["row_idx"]): entry for entry in entries}
        accepted_idx_set = set(int(idx) for idx in accepted_batch_idxs)
        prompt_input_ids = reroll_prompt_batch.batch["input_ids"]
        prompt_attention_mask = reroll_prompt_batch.batch["attention_mask"]
        for output_position, idx in enumerate(row_idxs):
            entry = entry_by_idx.get(int(idx))
            if entry is None:
                continue

            valid_prompt_ids = prompt_input_ids[output_position][
                prompt_attention_mask[output_position].bool()
            ]
            second_input = self.tokenizer.decode(
                valid_prompt_ids.detach().cpu().tolist(),
                skip_special_tokens=True,
            )
            _, second_output = self._decode_rollout_prompt_response(
                batch=reroll_scoring_batch,
                idx=output_position,
            )
            initial_score_100 = initial_score_result["scores_100"][idx]
            second_score_100 = reroll_score_result["scores_100"][output_position]
            initial_score_value = self._finite_gpt_score_value(initial_score_100)
            second_score_value = self._finite_gpt_score_value(second_score_100)
            if initial_score_value is None or second_score_value is None:
                score_delta_100 = None
            else:
                score_delta_100 = second_score_value - initial_score_value

            accepted = int(idx) in accepted_idx_set
            rejected_reason = None
            if not accepted:
                rejected_reason = "reroll_score_invalid" if second_score_value is None else "score_not_better"

            entry.update(
                {
                    "student_second_input": second_input,
                    "student_second_output": second_output,
                    "student_second_attempted": True,
                    "student_second_attempt": attempt + 1,
                    "student_second_accepted": accepted,
                    "student_second_rejected_reason": rejected_reason,
                    "student_second_score_delta_100": score_delta_100,
                    "student_second_gpt_score": reroll_score_result["scores"][output_position],
                    "student_second_gpt_score_100": second_score_100,
                    "student_second_gpt_weighted_score_1_to_4": reroll_score_result[
                        "weighted_scores_1_to_4"
                    ][output_position],
                    "student_second_gpt_rubric_scores": reroll_score_result["rubric_scores"][
                        output_position
                    ],
                    "student_second_gpt_reason": reroll_score_result["reasons"][output_position],
                    "student_second_gpt_revision_suggestion": reroll_score_result[
                        "revision_suggestions"
                    ][output_position],
                    "student_second_gpt_error": reroll_score_result["errors"][output_position],
                    "student_second_gpt_model": reroll_score_result["models"][output_position],
                }
            )

    def _maybe_update_gpt_case_studies_after_reroll(
        self,
        batch: DataProto,
        initial_score_result: dict,
        reroll_counts: np.ndarray,
        timing_raw: dict,
    ) -> None:
        case_study_dir = initial_score_result.get("_case_study_dir")
        entries = initial_score_result.get("_case_study_entries")
        if not case_study_dir or not entries:
            return

        selected_count = len(entries)
        updated_count = 0
        attempted_count = 0
        for entry in entries:
            idx = int(entry["row_idx"])
            reroll_count = int(reroll_counts[idx])
            entry["student_reroll_count"] = reroll_count
            if entry.get("student_second_attempted"):
                attempted_count += 1
            if reroll_count <= 0:
                continue

            _, second_output = self._decode_rollout_prompt_response(batch=batch, idx=idx)
            entry["student_second_output"] = second_output
            entry["student_second_accepted"] = True
            entry["student_second_rejected_reason"] = None
            updated_count += 1

        if updated_count == 0 and attempted_count == 0:
            return

        self._write_gpt_rollout_case_studies(
            case_study_dir=case_study_dir,
            entries=entries,
            timing_raw=timing_raw,
            selected_count=selected_count,
            low_count=selected_count,
        )
        self._debug_progress(
            f"gpt_case_study:updated_second_outputs accepted={updated_count} attempted={attempted_count}"
        )

    def _gpt_rollout_result_values(self, result: dict, prefix: str) -> dict:
        return {
            f"{prefix}_score": result["scores"],
            f"{prefix}_score_100": result["scores_100"],
            f"{prefix}_weighted_score_1_to_4": result["weighted_scores_1_to_4"],
            f"{prefix}_rubric_scores": result["rubric_scores"],
            f"{prefix}_reason": result["reasons"],
            f"{prefix}_revision_suggestion": result["revision_suggestions"],
            f"{prefix}_error": result["errors"],
            f"{prefix}_model": result["models"],
        }

    def _get_gpt_rollout_pass_flags(self, scores_100: list, threshold_100: float) -> list[bool]:
        pass_flags = []
        for score_100 in scores_100:
            try:
                score_value = float(score_100)
            except (TypeError, ValueError):
                pass_flags.append(False)
                continue
            pass_flags.append(np.isfinite(score_value) and score_value > threshold_100)
        return pass_flags

    def _set_gpt_rollout_result(
        self,
        batch: DataProto,
        result: dict,
        prefix: str = "gpt_rollout",
        threshold_100: Optional[float] = None,
        row_idxs: Optional[list[int]] = None,
    ) -> None:
        row_idxs_np = None if row_idxs is None else np.array(row_idxs, dtype=np.int64)

        for key, values in self._gpt_rollout_result_values(result=result, prefix=prefix).items():
            values_array = np.array(values, dtype=object)
            if row_idxs_np is None:
                batch.non_tensor_batch[key] = values_array
                continue

            if key not in batch.non_tensor_batch:
                batch.non_tensor_batch[key] = np.full(len(batch), None, dtype=object)
            batch.non_tensor_batch[key][row_idxs_np] = values_array

        if threshold_100 is not None:
            pass_flags = self._get_gpt_rollout_pass_flags(result["scores_100"], threshold_100)
            pass_key = f"{prefix}_pass_score_threshold"
            pass_array = np.array(pass_flags, dtype=object)
            if row_idxs_np is None:
                batch.non_tensor_batch[pass_key] = pass_array
            else:
                if pass_key not in batch.non_tensor_batch:
                    batch.non_tensor_batch[pass_key] = np.full(len(batch), None, dtype=object)
                batch.non_tensor_batch[pass_key][row_idxs_np] = pass_array

    def _log_gpt_rollout_score_metrics(
        self,
        metrics: dict,
        scores: list,
        scores_100: list,
        threshold_100: float,
        prefix: str = "gpt_rollout_score",
    ) -> None:
        valid_scores = []
        for score in scores:
            try:
                score_value = float(score)
            except (TypeError, ValueError):
                continue
            if np.isfinite(score_value):
                valid_scores.append(score_value)

        metrics[f"{prefix}/valid_count"] = len(valid_scores)
        metrics[f"{prefix}/error_count"] = len(scores) - len(valid_scores)
        if valid_scores:
            metrics[f"{prefix}/mean"] = float(np.mean(valid_scores))
            metrics[f"{prefix}/min"] = float(np.min(valid_scores))
            metrics[f"{prefix}/max"] = float(np.max(valid_scores))
            metrics[f"{prefix}_100/mean"] = float(np.mean(valid_scores) * 100.0)

        pass_flags = self._get_gpt_rollout_pass_flags(scores_100, threshold_100)
        metrics[f"{prefix}/threshold_100"] = threshold_100
        metrics[f"{prefix}/pass_count"] = int(sum(pass_flags))
        metrics[f"{prefix}/low_count"] = len(pass_flags) - int(sum(pass_flags))

    def _score_gpt_rollouts(self, batch: DataProto, scorer_config: dict, timing_raw: dict, timer_name: str) -> dict:
        with marked_timer(timer_name, timing_raw, color="green"):
            from verl.trainer.ppo.gpt_rollout_scorer import score_rollouts_with_gpt

            return score_rollouts_with_gpt(batch=batch, tokenizer=self.tokenizer, config=scorer_config)

    def _maybe_score_gpt_rollouts(self, batch: DataProto, metrics: dict, timing_raw: dict) -> Optional[dict]:
        scorer_config = self.config.trainer.get("gpt_rollout_score", None)
        if not scorer_config or not _is_truthy(scorer_config.get("enable", False)):
            return

        self._debug_progress(
            f"gpt_rollout_score:start batch_size={len(batch)} "
            f"model={scorer_config.get('model', '?')} max_workers={scorer_config.get('max_workers', '?')}"
        )
        result = self._score_gpt_rollouts(
            batch=batch, scorer_config=scorer_config, timing_raw=timing_raw, timer_name="gpt_rollout_score"
        )
        min_score_100 = float(scorer_config.get("min_score_100", 50.0))
        self._set_gpt_rollout_result(
            batch=batch, result=result, prefix="gpt_rollout", threshold_100=min_score_100
        )
        self._set_gpt_rollout_result(
            batch=batch, result=result, prefix="gpt_rollout_initial", threshold_100=min_score_100
        )
        self._log_gpt_rollout_score_metrics(
            metrics=metrics,
            scores=result["scores"],
            scores_100=result["scores_100"],
            threshold_100=min_score_100,
            prefix="gpt_rollout_score",
        )
        self._log_gpt_rollout_score_metrics(
            metrics=metrics,
            scores=result["scores"],
            scores_100=result["scores_100"],
            threshold_100=min_score_100,
            prefix="gpt_rollout_initial_score",
        )
        self._maybe_log_gpt_rollout_case_studies(
            batch=batch,
            result=result,
            scorer_config=scorer_config,
            timing_raw=timing_raw,
        )
        valid_count = sum(score is not None for score in result["scores_100"])
        error_count = len(result["scores_100"]) - valid_count
        self._debug_progress(
            f"gpt_rollout_score:done valid_count={valid_count} error_count={error_count} threshold_100={min_score_100}"
        )

        return result

    def _get_low_gpt_score_idxs(self, scores_100: list, threshold_100: float) -> list[int]:
        low_idxs = []
        for idx, score_100 in enumerate(scores_100):
            try:
                score_value = float(score_100)
            except (TypeError, ValueError):
                low_idxs.append(idx)
                continue
            if not np.isfinite(score_value) or score_value <= threshold_100:
                low_idxs.append(idx)
        return low_idxs

    def _is_gpt_timeout_error(self, error: object) -> bool:
        error_text = str(error or "").strip().lower()
        return any(marker in error_text for marker in ("timeout", "timed out", "time out", "time-out"))

    def _get_initial_gpt_timeout_idxs(self, result: dict) -> list[int]:
        timeout_idxs = []
        scores_100 = result.get("scores_100", [])
        errors = result.get("errors", [])
        for idx, error in enumerate(errors):
            score_100 = scores_100[idx] if idx < len(scores_100) else None
            if score_100 is None and self._is_gpt_timeout_error(error):
                timeout_idxs.append(idx)
        return timeout_idxs

    def _finite_gpt_score_value(self, score: object) -> Optional[float]:
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            return None
        return score_value if np.isfinite(score_value) else None

    def _is_reroll_score_better(self, reroll_score_100: object, initial_score_100: object) -> bool:
        reroll_value = self._finite_gpt_score_value(reroll_score_100)
        if reroll_value is None:
            return False

        initial_value = self._finite_gpt_score_value(initial_score_100)
        if initial_value is None:
            return True

        return reroll_value > initial_value

    def _select_gpt_rollout_result(self, result: dict, positions: list[int]) -> dict:
        return {
            key: [values[position] for position in positions] if isinstance(values, list) else values
            for key, values in result.items()
        }

    def _render_reroll_context(self, previous_solution: str, feedback: dict) -> str:
        feedback_json = json.dumps(feedback, ensure_ascii=False)
        return (
            "[Previous Solution]\n"
            f"{previous_solution}\n\n"
            "[GPT Feedback on Previous Solution]\n"
            f"{feedback_json}"
        )

    def _render_reroll_prompt_suffix(self, problem: str, reroll_context: str) -> str:
        return (
            "[Problem]\n"
            f"{problem}\n\n"
            f"{reroll_context}\n\n"
            "Please solve the problem above again.\n"
            "Do not discuss the feedback explicitly.\n"
            "Do not mention the previous solution.\n"
            "Produce a clean corrected solution only.\n"
            "Be concise.\n"
            "End with the final answer."
        )

    def _build_gpt_feedback_for_reroll_context(self, result: dict, idx: int, previous_solution: str = "") -> str:
        rubric_scores = result["rubric_scores"][idx]
        revision_suggestion = result["revision_suggestions"][idx]

        feedback = {}
        if isinstance(rubric_scores, dict):
            sanitized_rubric = {}
            for name, score_info in rubric_scores.items():
                if isinstance(score_info, dict):
                    sanitized_rubric[name] = {
                        key: value for key, value in score_info.items() if key != "weight"
                    }
                else:
                    sanitized_rubric[name] = score_info
            feedback["rubric_scores"] = sanitized_rubric

        if revision_suggestion:
            feedback["revision_suggestion"] = revision_suggestion

        if not feedback:
            return ""

        return self._render_reroll_context(previous_solution=previous_solution, feedback=feedback)

    def _get_reroll_summary_max_workers(self, scorer_config: dict, job_count: int) -> int:
        raw_value = scorer_config.get("reroll_summary_max_workers", None)
        if raw_value is None:
            raw_value = scorer_config.get("max_workers", 8)
        try:
            max_workers = int(raw_value)
        except (TypeError, ValueError):
            max_workers = 8
        return min(max(max_workers, 1), max(job_count, 1))

    def _format_gpt_feedback_for_reroll(
        self,
        result: dict,
        idx: int,
        problem: str = "",
        previous_solution: str = "",
        scorer_config: Optional[dict] = None,
    ) -> str:
        reroll_context = self._build_gpt_feedback_for_reroll_context(
            result=result,
            idx=idx,
            previous_solution=previous_solution,
        )
        if not reroll_context:
            return ""

        if scorer_config is None:
            return self._render_reroll_prompt_suffix(problem=problem, reroll_context=reroll_context)

        max_context_tokens = max(
            1,
            int(
                scorer_config.get(
                    "max_reroll_context_tokens",
                    scorer_config.get("max_reroll_feedback_tokens", 1024),
                )
            ),
        )
        context_ids = self.tokenizer.encode(reroll_context, add_special_tokens=False)
        if len(context_ids) <= max_context_tokens:
            return self._render_reroll_prompt_suffix(problem=problem, reroll_context=reroll_context)

        try:
            from verl.trainer.ppo.gpt_rollout_scorer import summarize_reroll_context_with_gpt

            summary = summarize_reroll_context_with_gpt(
                context=reroll_context,
                target_tokens=max_context_tokens,
                config=scorer_config,
                request_idx=idx + 1,
                verbose=_is_truthy(scorer_config.get("verbose", False)),
            )
        except Exception as exc:
            summary = {
                "reroll_context": reroll_context,
                "error": str(exc),
            }

        if summary.get("error"):
            self._debug_progress(f"gpt_reroll_summary:failed idx={idx} error={str(summary['error'])[:240]}")

        reroll_context = summary.get("reroll_context") or reroll_context
        context_ids = self.tokenizer.encode(reroll_context, add_special_tokens=False)
        if len(context_ids) > max_context_tokens:
            self._debug_progress(
                f"gpt_reroll_context:summary_still_too_long idx={idx} tokens={len(context_ids)} "
                f"limit={max_context_tokens}; applying context tail fallback"
            )
            reroll_context = self.tokenizer.decode(context_ids[-max_context_tokens:], skip_special_tokens=True)

        return self._render_reroll_prompt_suffix(problem=problem, reroll_context=reroll_context)

    def _append_gpt_feedback_to_reroll_prompts(
        self,
        batch: DataProto,
        gen_batch: DataProto,
        row_idxs: list[int],
        initial_score_result: dict,
        scorer_config: dict,
    ) -> int:
        if not row_idxs:
            return 0

        input_ids = gen_batch.batch["input_ids"]
        attention_mask = gen_batch.batch["attention_mask"]
        position_ids = gen_batch.batch["position_ids"]
        prompt_length = input_ids.shape[-1]
        max_context_tokens = max(
            0,
            int(
                scorer_config.get(
                    "max_reroll_context_tokens",
                    scorer_config.get("max_reroll_feedback_tokens", 1024),
                )
            ),
        )
        if max_context_tokens == 0:
            return 0
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

        feedback_requests = []
        for idx in row_idxs:
            valid_prompt_len = int(attention_mask[idx].sum().item())
            if valid_prompt_len > 0:
                valid_prompt_ids = input_ids[idx, -valid_prompt_len:].detach().cpu().tolist()
            else:
                valid_prompt_ids = []

            problem = ""
            extra_infos = batch.non_tensor_batch.get("extra_info", None)
            if extra_infos is not None:
                extra_info = extra_infos[idx]
                if isinstance(extra_info, dict):
                    problem = extra_info.get("problem") or extra_info.get("question") or ""
            if not problem:
                problem = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)

            previous_solution = ""
            if "responses" in batch.batch:
                response_ids = batch.batch["responses"][idx]
                response_mask = batch.batch.get("response_mask", None)
                if response_mask is not None:
                    valid_response_len = int(response_mask[idx].sum().item())
                    response_ids = response_ids[:valid_response_len]
                previous_solution = self.tokenizer.decode(
                    response_ids.detach().cpu().tolist(),
                    skip_special_tokens=True,
                )

            reroll_context = self._build_gpt_feedback_for_reroll_context(
                result=initial_score_result,
                idx=idx,
                previous_solution=previous_solution,
            )
            if not reroll_context:
                continue

            context_ids = self.tokenizer.encode(reroll_context, add_special_tokens=False)
            feedback_requests.append(
                {
                    "idx": idx,
                    "valid_prompt_ids": valid_prompt_ids,
                    "problem": problem,
                    "reroll_context": reroll_context,
                    "needs_summary": len(context_ids) > max_context_tokens,
                }
            )

        summary_results = {}
        summary_jobs = [request for request in feedback_requests if request["needs_summary"]]
        if summary_jobs:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from verl.trainer.ppo.gpt_rollout_scorer import summarize_reroll_context_with_gpt

            summary_max_workers = self._get_reroll_summary_max_workers(scorer_config, len(summary_jobs))
            self._debug_progress(
                f"gpt_reroll_summary:start count={len(summary_jobs)} max_workers={summary_max_workers}"
            )
            with ThreadPoolExecutor(max_workers=summary_max_workers) as executor:
                future_to_request = {
                    executor.submit(
                        summarize_reroll_context_with_gpt,
                        context=request["reroll_context"],
                        target_tokens=max_context_tokens,
                        config=scorer_config,
                        request_idx=request["idx"] + 1,
                        verbose=_is_truthy(scorer_config.get("verbose", False)),
                    ): request
                    for request in summary_jobs
                }
                for future in as_completed(future_to_request):
                    request = future_to_request[future]
                    try:
                        summary = future.result()
                    except Exception as exc:
                        summary = {
                            "reroll_context": "",
                            "error": str(exc),
                        }
                    summary_results[request["idx"]] = summary
            self._debug_progress(f"gpt_reroll_summary:done count={len(summary_jobs)}")

        appended_count = 0
        for request in feedback_requests:
            idx = request["idx"]
            reroll_context = request["reroll_context"]
            if request["needs_summary"]:
                summary = summary_results.get(idx, {})
                if summary.get("error"):
                    self._debug_progress(f"gpt_reroll_summary:failed idx={idx} error={str(summary['error'])[:240]}")
                reroll_context = summary.get("reroll_context") or reroll_context

            context_ids = self.tokenizer.encode(reroll_context, add_special_tokens=False)
            if len(context_ids) > max_context_tokens:
                self._debug_progress(
                    f"gpt_reroll_context:summary_still_too_long idx={idx} tokens={len(context_ids)} "
                    f"limit={max_context_tokens}; applying context tail fallback"
                )
                reroll_context = self.tokenizer.decode(context_ids[-max_context_tokens:], skip_special_tokens=True)

            feedback_text = self._render_reroll_prompt_suffix(
                problem=request["problem"],
                reroll_context=reroll_context,
            )
            apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
            reroll_prompt_text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": feedback_text}],
                add_generation_prompt=True,
                tokenize=False,
                **apply_chat_template_kwargs,
            )
            combined_ids = self.tokenizer.encode(reroll_prompt_text, add_special_tokens=False)
            if not combined_ids:
                continue

            if len(combined_ids) > prompt_length:
                combined_ids = combined_ids[-prompt_length:]

            new_input_ids = torch.full_like(input_ids[idx], int(pad_token_id))
            new_attention_mask = torch.zeros_like(attention_mask[idx])
            new_ids = torch.tensor(combined_ids, dtype=input_ids.dtype, device=input_ids.device)
            new_input_ids[-len(combined_ids) :] = new_ids
            new_attention_mask[-len(combined_ids) :] = 1

            input_ids[idx] = new_input_ids
            attention_mask[idx] = new_attention_mask
            if "raw_prompt_ids" in gen_batch.non_tensor_batch:
                gen_batch.non_tensor_batch["raw_prompt_ids"][idx] = list(combined_ids)

            new_position_ids = compute_position_id_with_mask(new_attention_mask.unsqueeze(0)).squeeze(0)
            if position_ids.dim() == 2:
                position_ids[idx] = new_position_ids.to(device=position_ids.device, dtype=position_ids.dtype)
            elif position_ids.dim() == 3:
                position_ids[idx] = (
                    new_position_ids.to(device=position_ids.device, dtype=position_ids.dtype)
                    .unsqueeze(0)
                    .expand_as(position_ids[idx])
                )

            appended_count += 1

        return appended_count

    def _mark_gpt_rollout_result_unscored_after_reroll(self, batch: DataProto, row_idxs: list[int]) -> None:
        if not row_idxs:
            return

        row_idxs_np = np.array(row_idxs, dtype=np.int64)
        null_fields = (
            "gpt_rollout_score",
            "gpt_rollout_score_100",
            "gpt_rollout_weighted_score_1_to_4",
            "gpt_rollout_rubric_scores",
            "gpt_rollout_model",
            "gpt_rollout_pass_score_threshold",
        )
        empty_string_fields = (
            "gpt_rollout_reason",
            "gpt_rollout_revision_suggestion",
        )
        for key in null_fields:
            if key in batch.non_tensor_batch:
                batch.non_tensor_batch[key][row_idxs_np] = None
        for key in empty_string_fields:
            if key in batch.non_tensor_batch:
                batch.non_tensor_batch[key][row_idxs_np] = ""
        if "gpt_rollout_error" in batch.non_tensor_batch:
            batch.non_tensor_batch["gpt_rollout_error"][row_idxs_np] = "skipped GPT rescore after reroll"

    def _replace_rollout_rows(self, batch: DataProto, rollout_output: DataProto, row_idxs: list[int]) -> None:
        if not row_idxs:
            return

        row_idxs_np = np.array(row_idxs, dtype=np.int64)

        for key, value in rollout_output.batch.items():
            if key in batch.batch:
                row_idxs_torch = torch.tensor(row_idxs, device=batch.batch[key].device, dtype=torch.long)
                batch.batch[key][row_idxs_torch] = value.to(batch.batch[key].device)

        for key, value in rollout_output.non_tensor_batch.items():
            if key in batch.non_tensor_batch:
                batch.non_tensor_batch[key][row_idxs_np] = value

    def _set_standard_opd_teacher_inputs(self, batch: DataProto) -> None:
        batch.batch["ref_input_ids"] = batch.batch["input_ids"].clone()
        batch.batch["ref_attention_mask"] = batch.batch["attention_mask"].clone()
        batch.batch["ref_position_ids"] = batch.batch["position_ids"].clone()

    def _maybe_reroll_low_gpt_rollouts(
        self,
        batch: DataProto,
        gen_batch: DataProto,
        initial_score_result: dict | None,
        metrics: dict,
        timing_raw: dict,
    ) -> None:
        scorer_config = self.config.trainer.get("gpt_rollout_score", None)
        if not scorer_config or not _is_truthy(scorer_config.get("enable", False)) or initial_score_result is None:
            return

        threshold_100 = float(scorer_config.get("min_score_100", 50.0))
        configured_max_attempts = int(scorer_config.get("max_rerollout_attempts", 1))
        max_attempts = min(max(configured_max_attempts, 0), 1)
        reroll_counts = np.zeros(len(batch), dtype=object)
        raw_low_idxs = self._get_low_gpt_score_idxs(initial_score_result["scores_100"], threshold_100)
        initial_timeout_idxs = self._get_initial_gpt_timeout_idxs(initial_score_result)
        initial_timeout_idx_set = set(initial_timeout_idxs)
        low_idxs = [idx for idx in raw_low_idxs if idx not in initial_timeout_idx_set]
        metrics["gpt_rollout_reroll/initial_low_count"] = len(low_idxs)
        metrics["gpt_rollout_reroll/initial_low_or_error_count"] = len(raw_low_idxs)
        metrics["gpt_rollout_reroll/initial_timeout_passthrough_count"] = len(initial_timeout_idxs)
        metrics["gpt_rollout_reroll/threshold_100"] = threshold_100
        metrics["gpt_rollout_reroll/max_attempts"] = max_attempts
        self._debug_progress(
            f"gpt_reroll:start low_count={len(low_idxs)} timeout_passthrough={len(initial_timeout_idxs)} "
            f"threshold_100={threshold_100} max_attempts={max_attempts}"
        )
        metrics["gpt_rollout_reroll/configured_max_attempts"] = configured_max_attempts

        gen_batch_for_reroll = None
        for attempt in range(max_attempts):
            if not low_idxs:
                break

            if gen_batch_for_reroll is None:
                gen_batch_for_reroll = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )
                feedback_count = self._append_gpt_feedback_to_reroll_prompts(
                    batch=batch,
                    gen_batch=gen_batch_for_reroll,
                    row_idxs=low_idxs,
                    initial_score_result=initial_score_result,
                    scorer_config=scorer_config,
                )
                metrics["gpt_rollout_reroll/feedback_prompt_count"] = feedback_count
            reroll_gen_batch = gen_batch_for_reroll.select_idxs(low_idxs)
            self._debug_progress(f"gpt_reroll_generate:start attempt={attempt + 1}/{max_attempts} count={len(low_idxs)}")
            with marked_timer(f"gpt_reroll_gen_{attempt + 1}", timing_raw, color="red"):
                if not self.async_rollout_mode:
                    reroll_output = self.actor_rollout_wg.generate_sequences(reroll_gen_batch)
                else:
                    reroll_output = self.async_rollout_manager.generate_sequences(reroll_gen_batch)

                for key, value in reroll_output.meta_info.get("timing", {}).items():
                    timing_raw[f"gpt_reroll_{attempt + 1}/{key}"] = value
                reroll_output.meta_info.pop("timing", None)

            reroll_scoring_batch = batch.select_idxs(low_idxs)
            self._replace_rollout_rows(
                batch=reroll_scoring_batch,
                rollout_output=reroll_output,
                row_idxs=list(range(len(low_idxs))),
            )
            reroll_scoring_batch.batch["response_mask"] = compute_response_mask(reroll_scoring_batch)

            self._debug_progress(
                f"gpt_reroll_rescore:start attempt={attempt + 1}/{max_attempts} count={len(low_idxs)}"
            )
            reroll_score_result = self._score_gpt_rollouts(
                batch=reroll_scoring_batch,
                scorer_config=scorer_config,
                timing_raw=timing_raw,
                timer_name=f"gpt_reroll_score_{attempt + 1}",
            )
            self._debug_progress(
                f"gpt_reroll_rescore:done attempt={attempt + 1}/{max_attempts} count={len(low_idxs)}"
            )

            accepted_output_positions = []
            accepted_batch_idxs = []
            rejected_batch_idxs = []
            for output_position, idx in enumerate(low_idxs):
                if self._is_reroll_score_better(
                    reroll_score_result["scores_100"][output_position],
                    initial_score_result["scores_100"][idx],
                ):
                    accepted_output_positions.append(output_position)
                    accepted_batch_idxs.append(idx)
                    continue
                rejected_batch_idxs.append(idx)

            self._maybe_record_gpt_case_study_reroll_attempts(
                initial_score_result=initial_score_result,
                reroll_prompt_batch=reroll_gen_batch,
                reroll_scoring_batch=reroll_scoring_batch,
                reroll_score_result=reroll_score_result,
                row_idxs=low_idxs,
                accepted_batch_idxs=accepted_batch_idxs,
                attempt=attempt,
            )

            if accepted_batch_idxs:
                accepted_reroll_output = reroll_output.select_idxs(accepted_output_positions)
                self._replace_rollout_rows(
                    batch=batch,
                    rollout_output=accepted_reroll_output,
                    row_idxs=accepted_batch_idxs,
                )
                accepted_score_result = self._select_gpt_rollout_result(
                    reroll_score_result,
                    accepted_output_positions,
                )
                self._set_gpt_rollout_result(
                    batch=batch,
                    result=accepted_score_result,
                    prefix="gpt_rollout",
                    threshold_100=threshold_100,
                    row_idxs=accepted_batch_idxs,
                )
                batch.batch["response_mask"] = compute_response_mask(batch)

            for idx in accepted_batch_idxs:
                reroll_counts[idx] = int(reroll_counts[idx]) + 1

            reroll_valid_count = sum(score is not None for score in reroll_score_result["scores_100"])
            metrics[f"gpt_rollout_reroll/attempt_{attempt + 1}_generated_count"] = len(low_idxs)
            metrics[f"gpt_rollout_reroll/attempt_{attempt + 1}_rescore_valid_count"] = reroll_valid_count
            metrics[f"gpt_rollout_reroll/attempt_{attempt + 1}_rerolled_count"] = len(accepted_batch_idxs)
            metrics[f"gpt_rollout_reroll/attempt_{attempt + 1}_rejected_count"] = len(rejected_batch_idxs)
            self._debug_progress(
                f"gpt_reroll_select:done attempt={attempt + 1}/{max_attempts} "
                f"accepted={len(accepted_batch_idxs)} rejected={len(rejected_batch_idxs)}"
            )
            low_idxs = []

        batch.non_tensor_batch["gpt_rollout_reroll_count"] = reroll_counts
        self._maybe_update_gpt_case_studies_after_reroll(
            batch=batch,
            initial_score_result=initial_score_result,
            reroll_counts=reroll_counts,
            timing_raw=timing_raw,
        )
        final_scores = batch.non_tensor_batch["gpt_rollout_score"].tolist()
        final_scores_100 = batch.non_tensor_batch["gpt_rollout_score_100"].tolist()
        final_low_idxs = [
            idx
            for idx in self._get_low_gpt_score_idxs(final_scores_100, threshold_100)
            if idx not in initial_timeout_idx_set
        ]
        self._log_gpt_rollout_score_metrics(
            metrics=metrics,
            scores=final_scores,
            scores_100=final_scores_100,
            threshold_100=threshold_100,
            prefix="gpt_rollout_score",
        )
        metrics["gpt_rollout_reroll/final_low_count"] = len(final_low_idxs)
        metrics["gpt_rollout_reroll/rerolled_count"] = int(sum(int(count) > 0 for count in reroll_counts))
        metrics["gpt_rollout_reroll/total_rerolls"] = int(sum(int(count) for count in reroll_counts))
        metrics["gpt_rollout_reroll/rescore_skipped_count"] = 0
        self._debug_progress(
            f"gpt_reroll:done final_low_count={len(final_low_idxs)} "
            f"accepted_rerolls={metrics['gpt_rollout_reroll/total_rerolls']} "
            f"rescore_skipped_count={metrics['gpt_rollout_reroll/rescore_skipped_count']}"
        )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _progress_debug_enabled(self) -> bool:
        trainer_value = self.config.trainer.get("progress_debug", None)
        if trainer_value is None:
            trainer_value = os.environ.get("G_OPD_PROGRESS_DEBUG", "0")
        return _g_opd_truthy(trainer_value)

    def _debug_progress(self, message: str) -> None:
        if not self._progress_debug_enabled():
            return
        total_steps = getattr(self, "total_training_steps", "?")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[g_opd_progress] {timestamp} step={self.global_steps}/{total_steps} {message}", flush=True)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role=str(Role.ActorRollout),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.ActorRollout)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            # NOTE: while there is no checkpoint to load, we still need to offload the model and optimizer to CPU
            self.actor_rollout_wg.load_checkpoint(None)
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                self.actor_rollout_wg.load_checkpoint(None)
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        global_seqlen_lst = calculate_workload(global_seqlen_lst)
        world_size = self.actor_rollout_wg.world_size
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(global_seqlen_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    global_seqlen_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                global_seqlen_lst, k_partitions=world_size, equal_size=True
            )
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (global_seqlen_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                self._debug_progress(
                    f"step:start epoch={epoch} train_batch_size={len(batch)} "
                    f"rollout_n={self.config.actor_rollout_ref.rollout.n}"
                )

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    self._debug_progress(
                        f"rollout_generate:start prompt_batch_size={len(gen_batch)} "
                        f"repeated_batch_size={len(gen_batch_output)} async={self.async_rollout_mode}"
                    )
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    self._debug_progress(f"rollout_generate:done output_batch_size={len(gen_batch_output)}")

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    self._debug_progress(f"batch_prepare:done rollout_batch_size={len(batch)}")

                    gpt_score_result = self._maybe_score_gpt_rollouts(batch=batch, metrics=metrics, timing_raw=timing_raw)
                    self._maybe_reroll_low_gpt_rollouts(
                        batch=batch,
                        gen_batch=gen_batch,
                        initial_score_result=gpt_score_result,
                        metrics=metrics,
                        timing_raw=timing_raw,
                    )

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    self._debug_progress(
                        f"reward:start use_rm={self.use_rm} launch_async={self.config.reward_model.launch_reward_fn_async}"
                    )
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                    self._debug_progress("reward:done")

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        self._debug_progress("old_log_prob:skip bypass_mode=true")
                        self._debug_progress("rollout_correction:start mode=bypass")
                        from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction

                        apply_rollout_correction(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                        self._debug_progress("rollout_correction:done mode=bypass")
                    else:  # Recompute old_log_probs
                        self._debug_progress("old_log_prob:start")
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(
                                loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                            )
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))
                        self._debug_progress("old_log_prob:done")

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'
                    self._set_standard_opd_teacher_inputs(batch)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        self._debug_progress("ref_log_prob:start")
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            # Get apply_chat_template_kwargs from config if available
                            apply_chat_template_kwargs = self.config.data.get(
                                "apply_chat_template_kwargs", {}
                            )

                            # Context distillation: generate critiques and re-tokenize with critique-augmented prompts
                            if self.use_context_distillation:
                                from verl.trainer.ppo.ref_input_utils import prepare_critique_distillation_inputs
                                
                                batch = prepare_critique_distillation_inputs(
                                    batch=batch,
                                    tokenizer=self.tokenizer,
                                    critique_vllm_url=self.critique_vllm_url,
                                    critique_model=self.critique_model,
                                    critique_prompt_template=None,
                                    ref_apply_chat_template_kwargs=apply_chat_template_kwargs,
                                    max_critique_tokens=self.max_critique_tokens,
                                    critique_temperature=self.critique_temperature,
                                    critique_top_p=self.critique_top_p,
                                )

                                if not self.ref_in_actor:
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                else:
                                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                            # Ref solution distillation: use ref_solution from dataset as teacher context
                            elif self.use_ref_solution_distillation:
                                from verl.trainer.ppo.ref_input_utils import prepare_ref_model_inputs_based_on_correct_solution

                                batch = prepare_ref_model_inputs_based_on_correct_solution(
                                    batch=batch,
                                    tokenizer=self.tokenizer,
                                    apply_chat_template_kwargs=apply_chat_template_kwargs,
                                )

                                if not self.ref_in_actor:
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                else:
                                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                            # If ref model uses different tokenizer/prompt template, re-tokenize inputs for ref model
                            elif self.use_ref_retokenization:
                                from verl.trainer.ppo.ref_input_utils import prepare_ref_model_inputs
                                
                                batch = prepare_ref_model_inputs(
                                    batch=batch,
                                    ref_tokenizer=self.ref_tokenizer,
                                    apply_chat_template_kwargs=apply_chat_template_kwargs,
                                )
                                
                                if not self.ref_in_actor:
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                else:
                                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)
                            
                            else:
                                # Standard ref model log prob computation
                                if not self.ref_in_actor:
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                else:
                                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                    if self.use_reference_policy:
                        self._debug_progress("ref_log_prob:done")

                    # Compute base model log probs for corrected reward computation
                    # This computes: base_log_prob from actor's base model (using input_ids)
                    # and base_ref_log_prob from ref's base model (using ref_input_ids)
                    if self.use_base_models:
                        self._debug_progress("base_log_probs:start")
                        with marked_timer("base_log_probs", timing_raw, color="green"):
                            # First compute base_ref_log_prob using ref's base model
                            # This uses ref_input_ids which may be present in batch
                            if not self.ref_in_actor:
                                base_ref_log_prob = self.ref_policy_wg.compute_base_ref_log_prob(batch)
                            else:
                                base_ref_log_prob = self.actor_rollout_wg.compute_base_ref_log_prob(batch)
                            batch = batch.union(base_ref_log_prob)
                            
                            # Now compute base_log_prob using actor's base model with input_ids
                            # We need to temporarily remove ref_input_ids to ensure compute_log_prob uses input_ids
                            ref_input_tensors = {}
                            if "ref_input_ids" in batch.batch:
                                ref_input_tensors["ref_input_ids"] = batch.batch.pop("ref_input_ids")
                            if "ref_attention_mask" in batch.batch:
                                ref_input_tensors["ref_attention_mask"] = batch.batch.pop("ref_attention_mask")
                            if "ref_position_ids" in batch.batch:
                                ref_input_tensors["ref_position_ids"] = batch.batch.pop("ref_position_ids")
                            
                            # Compute base_log_prob using actor's base model with input_ids
                            base_log_prob = self.actor_rollout_wg.compute_base_log_prob(batch)
                            batch = batch.union(base_log_prob)
                            
                            # Restore ref_input_ids tensors back to batch
                            for key, tensor in ref_input_tensors.items():
                                batch.batch[key] = tensor
                            
                            print(f"Computed base log probs for corrected reward: "
                                  f"base_log_prob shape={batch.batch['base_log_prob'].shape}, "
                                  f"base_ref_log_prob shape={batch.batch['base_ref_log_prob'].shape}") 
                    
                    if self.use_base_models:
                        self._debug_progress("base_log_probs:done")

                    # compute values
                    if self.use_critic:
                        self._debug_progress("values:start")
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)
                        self._debug_progress("values:done")

                    self._debug_progress("advantage:start")
                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
                    self._debug_progress("advantage:done")

                    # update critic
                    if self.use_critic:
                        self._debug_progress("update_critic:start")
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)
                        self._debug_progress("update_critic:done")

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        self._debug_progress("update_actor:start")
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                        self._debug_progress("update_actor:done")
                    else:
                        self._debug_progress(
                            f"update_actor:skip critic_warmup={self.config.trainer.critic_warmup}"
                        )

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._debug_progress(f"rollout_data_log:start dir={rollout_data_dir}")
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)
                        self._debug_progress("rollout_data_log:done")

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    self._debug_progress("validation:start")
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)
                    self._debug_progress("validation:done")

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                self._debug_progress("metrics_log:start")
                logger.log(data=metrics, step=self.global_steps)
                self._debug_progress("metrics_log:done")

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)


