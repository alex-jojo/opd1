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




# 1.鍒嗘。

# 2.rubric浣滃樊





import json
import math
import os
import hashlib
import time
import uuid
import zipfile
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional
from xml.sax.saxutils import escape as xml_escape

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


def _sanitize_metrics_for_logging(metrics: dict) -> tuple[dict, list[tuple[str, str]]]:
    sanitized = {}
    dropped = []
    for key, value in metrics.items():
        if torch.is_tensor(value):
            if value.numel() != 1:
                dropped.append((str(key), f"tensor_shape={tuple(value.shape)}"))
                continue
            value = value.detach().cpu().item()
        elif isinstance(value, np.ndarray):
            if value.size != 1:
                dropped.append((str(key), f"ndarray_shape={value.shape}"))
                continue
            value = value.item()
        elif isinstance(value, np.generic):
            value = value.item()

        if isinstance(value, bool):
            sanitized[key] = int(value)
        elif isinstance(value, (int, float)) and np.isfinite(value):
            sanitized[key] = value
        else:
            dropped.append((str(key), type(value).__name__))
    return sanitized, dropped


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

        self._gpt_rubric_history: deque[dict[str, object]] = deque()

                
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

        self.rubric_probe_config = self.config.trainer.get("rubric_probe_data", None)
        self.collect_rubric_probe_data = bool(
            self.rubric_probe_config and _is_truthy(self.rubric_probe_config.get("enable", False))
        )
        if self.collect_rubric_probe_data:
            scorer_config = self.config.trainer.get("gpt_rollout_score", None)
            if not scorer_config or not _is_truthy(scorer_config.get("enable", False)):
                raise ValueError("rubric_probe_data requires trainer.gpt_rollout_score.enable=True")
            if not self.use_reference_policy:
                raise ValueError("rubric_probe_data requires a reference/teacher policy")
            if str(self.config.actor_rollout_ref.actor.strategy).lower() not in {"fsdp", "fsdp2"}:
                raise ValueError("rubric_probe_data currently supports FSDP/FSDP2 actor and teacher policies")
            if self.use_ref_retokenization or self.use_context_distillation or self.use_ref_solution_distillation:
                raise ValueError(
                    "rubric_probe_data currently requires standard teacher inputs so the teacher encodes exactly "
                    "the same student response tokens (no ref re-tokenization or context/ref-solution distillation)"
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
        """Dump rollout/validation samples as Excel."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.xlsx")

        n = len(inputs)
        base_data = {
            "step": [self.global_steps] * n,
            "row_idx": list(range(n)),
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        entries = []
        for i in range(n):
            entries.append({k: v[i] for k, v in base_data.items()})

        self._write_rows_to_xlsx(entries, filename)
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
                reward_extra_infos_to_dump.setdefault(
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
                "gpt_rollout_initial_revision_suggestion_source",
                "gpt_rollout_initial_error",
                "gpt_rollout_initial_model",
                "gpt_rollout_initial_pass_score_threshold",
                "gpt_rollout_revision_suggestion_source",
                "gpt_rollout_reroll_count",
                "g_opd_sample_kind",
                "g_opd_loss_weight",
                "g_opd_reroll_sft_weight",
                "g_opd_source_row_idx",
                "g_opd_reroll_gain_100",
                "g_opd_teacher_rank",
                "g_opd_rubric_rank",
                "g_opd_teacher_rubric_rank_gap",
                "g_opd_padding_sample",
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

    def _excel_column_name(self, col_idx: int) -> str:
        name = ""
        col_idx += 1
        while col_idx:
            col_idx, remainder = divmod(col_idx - 1, 26)
            name = chr(65 + remainder) + name
        return name

    def _excel_cell_text(self, value) -> str:
        value = self._json_safe_value(value)
        if value is None:
            return ""
        if isinstance(value, (dict, list, tuple)):
            text = json.dumps(value, ensure_ascii=False)
        else:
            text = str(value)
        text = "".join(ch for ch in text if ch in "\t\n\r" or ord(ch) >= 32)
        return text[:32767]

    def _write_rows_to_xlsx(self, rows: list[dict], filename: str) -> None:
        headers = []
        for row in rows:
            for key in row.keys():
                if key not in headers:
                    headers.append(key)
        large_json_headers = {
            key
            for key in headers
            if key in {"extra_info", "reward_model"} or key.endswith("rubric_scores")
        }
        headers = [key for key in headers if key not in large_json_headers] + [
            key for key in headers if key in large_json_headers
        ]

        def cell_xml(row_idx: int, col_idx: int, value) -> str:
            cell_ref = f"{self._excel_column_name(col_idx)}{row_idx}"
            text = xml_escape(self._excel_cell_text(value))
            space = ' xml:space="preserve"' if text.strip() != text else ""
            return f'<c r="{cell_ref}" t="inlineStr"><is><t{space}>{text}</t></is></c>'

        sheet_rows = []
        header_cells = [cell_xml(1, col_idx, header) for col_idx, header in enumerate(headers)]
        sheet_rows.append(f'<row r="1">{"".join(header_cells)}</row>')
        for row_offset, row in enumerate(rows, start=2):
            cells = [cell_xml(row_offset, col_idx, row.get(header, "")) for col_idx, header in enumerate(headers)]
            sheet_rows.append(f'<row r="{row_offset}">{"".join(cells)}</row>')

        worksheet_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f'<sheetData>{"".join(sheet_rows)}</sheetData></worksheet>'
        )
        workbook_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="data" sheetId="1" r:id="rId1"/></sheets></workbook>'
        )
        workbook_rels_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            'Target="worksheets/sheet1.xml"/></Relationships>'
        )
        root_rels_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/></Relationships>'
        )
        content_types_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '</Types>'
        )

        with zipfile.ZipFile(filename, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", content_types_xml)
            zf.writestr("_rels/.rels", root_rels_xml)
            zf.writestr("xl/workbook.xml", workbook_xml)
            zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
            zf.writestr("xl/worksheets/sheet1.xml", worksheet_xml)

    def _write_rows_to_jsonl(self, rows: list[dict], filename: str) -> None:
        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(self._json_safe_value(row), ensure_ascii=False) + "\n")

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

    def _save_rubric_probe_data(self, batch: DataProto, metrics: dict, timing_raw: dict) -> None:
        if not self.collect_rubric_probe_data:
            return

        from verl.trainer.ppo.rubric_probe_data import pop_rubric_probe_hidden, save_rubric_probe_batch

        student_checkpoint = str(self.config.actor_rollout_ref.model.path)
        ref_model_config = self.config.actor_rollout_ref.ref.get("model", {}) or {}
        teacher_checkpoint = str(ref_model_config.get("path", student_checkpoint))
        with marked_timer("rubric_probe_data", timing_raw, color="green"):
            stats = save_rubric_probe_batch(
                batch=batch,
                tokenizer=self.tokenizer,
                config=self.rubric_probe_config,
                global_step=self.global_steps,
                student_checkpoint=student_checkpoint,
                teacher_checkpoint=teacher_checkpoint,
            )
        pop_rubric_probe_hidden(batch)
        batch.meta_info.pop("rubric_probe_return_hidden", None)
        for key, value in stats.items():
            metrics[f"rubric_probe_data/{key}"] = int(value)
        self._debug_progress(
            f"rubric_probe_data:saved count={stats.get('saved', 0)} "
            f"invalid_rubric={stats.get('invalid_rubric', 0)} "
            f"output_dir={self.rubric_probe_config.get('output_dir')}"
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
            if np.isfinite(score_value) and score_value < threshold_100:
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
        include_errors = _is_truthy(scorer_config.get("case_study_include_errors", False))
        low_idxs = self._get_gpt_case_study_low_idxs(
            result["scores_100"],
            threshold_100=threshold_100,
            include_errors=include_errors,
        )
        selected_idxs = list(range(len(result["scores_100"])))

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
                    "problem": extra_info.get("problem") if isinstance(extra_info, dict) else prompt_text,
                    "ground_truth": ground_truth,
                    "first_output": response_text,
                    "final_output": response_text,
                    "rule_score": None,
                    "acc": None,
                    "pred": None,
                    "first_gpt_score_100": result["scores_100"][idx],
                    "second_gpt_score_100": None,
                    "final_gpt_score_100": result["scores_100"][idx],
                    "is_low_score_case": idx in low_idxs,
                    "reroll_count": 0,
                    "second_output": None,
                    "second_accepted": None,
                    "second_score_delta_100": None,
                    "gpt_reason": result["reasons"][idx],
                    "gpt_revision_suggestion": result["revision_suggestions"][idx],
                    "gpt_problem_domain": result.get("problem_domains", [None for _ in result["scores_100"]])[idx],
                    "gpt_difficulty_3": result.get("difficulty_3", [None for _ in result["scores_100"]])[idx],
                    "gpt_error": result["errors"][idx],
                    "threshold_100": threshold_100,
                    "data_source": self._get_non_tensor_row_value(batch, "data_source", idx, None),
                    "request_id": self._get_non_tensor_row_value(batch, "request_id", idx, None),
                    "uid": self._get_non_tensor_row_value(batch, "uid", idx, None),
                    "first_gpt_weighted_score_1_to_4": result["weighted_scores_1_to_4"][idx],
                    "second_gpt_weighted_score_1_to_4": None,
                    "second_gpt_reason": "",
                    "second_gpt_revision_suggestion": "",
                    "second_gpt_error": "",
                    "extra_info": extra_info,
                    "reward_model": reward_model,
                    "first_gpt_rubric_scores": result["rubric_scores"][idx],
                    "second_gpt_rubric_scores": None,
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
            filename = os.path.join(case_study_dir, f"{self.global_steps}.xlsx")
            self._write_rows_to_xlsx([self._json_safe_value(entry) for entry in entries], filename)

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
        rejected_reasons: Optional[dict[int, str]],
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
            rejected_reason = None if accepted else (rejected_reasons or {}).get(
                int(idx),
                "reroll_score_invalid" if second_score_value is None else "reroll_selection_rejected",
            )

            entry.update(
                {
                    "second_input": second_input,
                    "second_output": second_output,
                    "second_attempted": True,
                    "second_attempt": attempt + 1,
                    "second_appended": accepted,
                    "second_accepted": accepted,
                    "second_rejected_reason": rejected_reason,
                    "second_score_delta_100": score_delta_100,
                    "second_gpt_score_100": second_score_100,
                    "second_gpt_weighted_score_1_to_4": reroll_score_result[
                        "weighted_scores_1_to_4"
                    ][output_position],
                    "second_gpt_rubric_scores": reroll_score_result["rubric_scores"][
                        output_position
                    ],
                    "second_gpt_reason": reroll_score_result["reasons"][output_position],
                    "second_gpt_revision_suggestion": reroll_score_result[
                        "revision_suggestions"
                    ][output_position],
                    "second_gpt_problem_domain": reroll_score_result.get(
                        "problem_domains", [None for _ in reroll_score_result["scores_100"]]
                    )[output_position],
                    "second_gpt_difficulty_3": reroll_score_result.get(
                        "difficulty_3", [None for _ in reroll_score_result["scores_100"]]
                    )[output_position],
                    "second_gpt_error": reroll_score_result["errors"][output_position],
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
            entry["reroll_count"] = reroll_count
            if entry.get("second_attempted"):
                attempted_count += 1
            if reroll_count <= 0:
                continue

            entry["second_appended"] = True
            entry["second_accepted"] = False
            entry["second_rejected_reason"] = "appended_not_replaced"
            entry["final_output"] = entry.get("first_output")
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

    def _maybe_update_gpt_case_study_rule_scores(
        self,
        batch: DataProto,
        initial_score_result: dict | None,
        reward_extra_infos_dict: dict,
        timing_raw: dict,
    ) -> None:
        if initial_score_result is None:
            return
        case_study_dir = initial_score_result.get("_case_study_dir")
        entries = initial_score_result.get("_case_study_entries")
        if not case_study_dir or not entries:
            return

        sequence_scores = batch.batch["token_level_scores"].sum(-1).detach().cpu().tolist()
        uid_to_pos = {}
        if "uid" in batch.non_tensor_batch:
            for pos, uid in enumerate(batch.non_tensor_batch["uid"].tolist()):
                uid_to_pos[str(uid)] = pos

        def aligned_value(key: str, pos: int):
            values = reward_extra_infos_dict.get(key)
            if values is None or pos >= len(values):
                return None
            return self._json_safe_value(values[pos])

        updated_count = 0
        for entry in entries:
            pos = None
            uid = entry.get("uid")
            if uid is not None:
                pos = uid_to_pos.get(str(uid))
            if pos is None:
                row_idx = int(entry["row_idx"])
                if 0 <= row_idx < len(sequence_scores):
                    pos = row_idx
            if pos is None or pos >= len(sequence_scores):
                continue

            entry["rule_score"] = self._json_safe_value(sequence_scores[pos])
            entry["acc"] = aligned_value("acc", pos)
            entry["pred"] = aligned_value("pred", pos)
            updated_count += 1

        if updated_count <= 0:
            return

        self._write_gpt_rollout_case_studies(
            case_study_dir=case_study_dir,
            entries=entries,
            timing_raw=timing_raw,
            selected_count=len(entries),
            low_count=sum(1 for entry in entries if entry.get("is_low_score_case")),
        )
        self._debug_progress(f"gpt_case_study:updated_rule_scores count={updated_count}")

    def _gpt_rollout_result_values(self, result: dict, prefix: str) -> dict:
        values = {
            f"{prefix}_score": result["scores"],
            f"{prefix}_score_100": result["scores_100"],
            f"{prefix}_weighted_score_1_to_4": result["weighted_scores_1_to_4"],
            f"{prefix}_rubric_scores": result["rubric_scores"],
            f"{prefix}_reason": result["reasons"],
            f"{prefix}_revision_suggestion": result["revision_suggestions"],
            f"{prefix}_problem_domain": result.get("problem_domains", [None for _ in result["scores"]]),
            f"{prefix}_difficulty_3": result.get("difficulty_3", [None for _ in result["scores"]]),
            f"{prefix}_error": result["errors"],
            f"{prefix}_model": result["models"],
        }
        if "revision_suggestion_sources" in result:
            values[f"{prefix}_revision_suggestion_source"] = result["revision_suggestion_sources"]
        return values

    def _inherit_gpt_problem_labels(
        self,
        *,
        target_result: dict,
        source_result: dict,
        source_row_idxs: list[int],
    ) -> None:
        source_domains = source_result.get("problem_domains", [])
        source_difficulties = source_result.get("difficulty_3", [])
        target_result["problem_domains"] = [
            source_domains[idx] if idx < len(source_domains) else None for idx in source_row_idxs
        ]
        target_result["difficulty_3"] = [
            source_difficulties[idx] if idx < len(source_difficulties) else None for idx in source_row_idxs
        ]

    def _get_gpt_rollout_pass_flags(self, scores_100: list, threshold_100: float) -> list[bool]:
        pass_flags = []
        for score_100 in scores_100:
            try:
                score_value = float(score_100)
            except (TypeError, ValueError):
                pass_flags.append(False)
                continue
            pass_flags.append(np.isfinite(score_value) and score_value >= threshold_100)
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

    def _fallback_reroll_revision_suggestion(self, score_100: float) -> str:
        if score_100 < 35.0:
            return (
                "Re-read the problem, identify the target and key constraints, then build a direct setup before computing."
            )
        if score_100 < 45.0:
            return (
                "Keep the useful setup, then check the first weak inference and use the constraints to finish the reasoning."
            )
        return (
            "Verify the final computation, conditions, and answer extraction against the original problem before concluding."
        )

    def _count_low_score_no_revision_suggestion(self, result: dict, threshold_100: float) -> int:
        count = 0
        scores_100 = result.get("scores_100", [])
        suggestions = result.get("revision_suggestions", [])
        for idx, score_100 in enumerate(scores_100):
            score_value = self._finite_gpt_score_value(score_100)
            if score_value is None or score_value >= threshold_100:
                continue
            suggestion = suggestions[idx] if idx < len(suggestions) else ""
            if not str(suggestion or "").strip():
                count += 1
        return count

    def _count_revision_suggestion_sources(self, result: dict, source_names: set[str]) -> int:
        sources = result.get("revision_suggestion_sources", [])
        return sum(1 for source in sources if str(source or "") in source_names)

    def _ensure_low_score_revision_suggestions(
        self,
        result: dict,
        threshold_100: float,
        *,
        fallback_source: str = "trainer_local_fallback",
    ) -> int:
        scores_100 = result.get("scores_100", [])
        if "revision_suggestions" not in result or not isinstance(result["revision_suggestions"], list):
            result["revision_suggestions"] = ["" for _ in scores_100]
        if len(result["revision_suggestions"]) < len(scores_100):
            result["revision_suggestions"].extend(["" for _ in range(len(scores_100) - len(result["revision_suggestions"]))])
        if "revision_suggestion_sources" not in result or not isinstance(result["revision_suggestion_sources"], list):
            result["revision_suggestion_sources"] = ["" for _ in scores_100]
        if len(result["revision_suggestion_sources"]) < len(scores_100):
            result["revision_suggestion_sources"].extend(
                ["" for _ in range(len(scores_100) - len(result["revision_suggestion_sources"]))]
            )

        fallback_count = 0
        for idx, score_100 in enumerate(scores_100):
            score_value = self._finite_gpt_score_value(score_100)
            if score_value is None or score_value >= threshold_100:
                continue
            if str(result["revision_suggestions"][idx] or "").strip():
                continue
            result["revision_suggestions"][idx] = self._fallback_reroll_revision_suggestion(score_value)
            result["revision_suggestion_sources"][idx] = fallback_source
            fallback_count += 1
        return fallback_count

    def _score_gpt_rollouts(self, batch: DataProto, scorer_config: dict, timing_raw: dict, timer_name: str) -> dict:
        with marked_timer(timer_name, timing_raw, color="green"):
            from verl.trainer.ppo.gpt_rollout_scorer import score_rollouts_with_gpt

            return score_rollouts_with_gpt(batch=batch, tokenizer=self.tokenizer, config=scorer_config)

    def _reroll_gpt_scorer_config(self, scorer_config: dict) -> dict:
        config = OmegaConf.to_container(scorer_config, resolve=True) if OmegaConf.is_config(scorer_config) else dict(scorer_config)
        config["max_output_tokens"] = int(config.get("reroll_max_output_tokens", 512))
        return config

    def _maybe_score_gpt_rollouts(self, batch: DataProto, metrics: dict, timing_raw: dict) -> Optional[dict]:
        scorer_config = self.config.trainer.get("gpt_rollout_score", None)
        if not scorer_config or not _is_truthy(scorer_config.get("enable", False)):
            return

        self._debug_progress(
            f"gpt_rollout_score:start batch_size={len(batch)} "
            f"model={scorer_config.get('model', '?')} max_workers={scorer_config.get('max_workers', '?')} "
            f"max_output_tokens={scorer_config.get('max_output_tokens', '?')}"
        )
        result = self._score_gpt_rollouts(
            batch=batch, scorer_config=scorer_config, timing_raw=timing_raw, timer_name="gpt_rollout_score"
        )
        min_score_100 = float(scorer_config.get("min_score_100", 50.0))
        low_no_hint_before = self._count_low_score_no_revision_suggestion(result, min_score_100)
        local_hint_fallback_count = self._ensure_low_score_revision_suggestions(result, min_score_100)
        scorer_hint_fallback_count = self._count_revision_suggestion_sources(
            result,
            {"local_fallback"},
        )
        gpt_hint_retry_count = self._count_revision_suggestion_sources(
            result,
            {"gpt_hint_retry"},
        )
        metrics["gpt_rollout_score/low_no_hint_count"] = low_no_hint_before + scorer_hint_fallback_count
        metrics["gpt_rollout_score/local_hint_fallback_count"] = local_hint_fallback_count + scorer_hint_fallback_count
        metrics["gpt_rollout_score/gpt_hint_retry_count"] = gpt_hint_retry_count
        metrics["gpt_rollout_score/initial_missing_hint_count"] = (
            gpt_hint_retry_count + scorer_hint_fallback_count + local_hint_fallback_count
        )
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
            if not np.isfinite(score_value) or score_value < threshold_100:
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

    def _select_gpt_rollout_result(self, result: dict, positions: list[int]) -> dict:
        return {
            key: [values[position] for position in positions] if isinstance(values, list) else values
            for key, values in result.items()
        }

    def _render_reroll_context(self, hint: str) -> str:
        hint = str(hint or "").strip()
        if not hint:
            return ""
        return (
            "[hint]\n"
            f"{hint}"
        )

    def _normalize_reroll_context(self, reroll_context: str) -> str:
        reroll_context = str(reroll_context or "").strip()
        if not reroll_context:
            return ""
        hint_header = "[hint]"
        if reroll_context.lower().startswith(hint_header):
            hint = reroll_context[len(hint_header) :].strip()
            return self._render_reroll_context(hint)
        return self._render_reroll_context(reroll_context)

    def _truncate_reroll_context(self, reroll_context: str, max_context_tokens: int) -> str:
        reroll_context = self._normalize_reroll_context(reroll_context)
        if not reroll_context:
            return ""
        context_ids = self.tokenizer.encode(reroll_context, add_special_tokens=False)
        if len(context_ids) <= max_context_tokens:
            return reroll_context

        hint_header = "[hint]\n"
        header_ids = self.tokenizer.encode(hint_header, add_special_tokens=False)
        hint_budget = max_context_tokens - len(header_ids)
        if hint_budget <= 0:
            return ""

        hint = reroll_context[len("[hint]") :].strip()
        hint_ids = self.tokenizer.encode(hint, add_special_tokens=False)
        truncated_hint = self.tokenizer.decode(hint_ids[:hint_budget], skip_special_tokens=True).strip()
        return self._render_reroll_context(truncated_hint)

    def _render_reroll_prompt_suffix(self, problem: str, reroll_context: str) -> str:
        return (
            "[Problem]\n"
            f"{problem}\n\n"
            f"{reroll_context}"
        )

    def _build_gpt_feedback_for_reroll_context(self, result: dict, idx: int) -> str:
        revision_suggestion = result["revision_suggestions"][idx]
        return self._render_reroll_context(hint=revision_suggestion)

    def _append_gpt_feedback_to_reroll_prompts(
        self,
        batch: DataProto,
        gen_batch: DataProto,
        row_idxs: list[int],
        initial_score_result: dict,
        scorer_config: dict,
    ) -> tuple[list[int], int]:
        if not row_idxs:
            return [], 0

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
            return [], 0
        threshold_100 = float(scorer_config.get("min_score_100", 50.0))
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

        feedback_requests = []
        local_hint_fallback_count = 0
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

            reroll_context = self._build_gpt_feedback_for_reroll_context(
                result=initial_score_result,
                idx=idx,
            )
            if not reroll_context:
                score_value = self._finite_gpt_score_value(initial_score_result["scores_100"][idx])
                if score_value is not None and score_value < threshold_100:
                    fallback_hint = self._fallback_reroll_revision_suggestion(score_value)
                    initial_score_result["revision_suggestions"][idx] = fallback_hint
                    if "revision_suggestion_sources" in initial_score_result:
                        initial_score_result["revision_suggestion_sources"][idx] = "prompt_local_fallback"
                    if "gpt_rollout_revision_suggestion" in batch.non_tensor_batch:
                        batch.non_tensor_batch["gpt_rollout_revision_suggestion"][idx] = fallback_hint
                    if "gpt_rollout_revision_suggestion_source" in batch.non_tensor_batch:
                        batch.non_tensor_batch["gpt_rollout_revision_suggestion_source"][idx] = "prompt_local_fallback"
                    reroll_context = self._render_reroll_context(hint=fallback_hint)
                    local_hint_fallback_count += 1
                if not reroll_context:
                    continue

            context_ids = self.tokenizer.encode(reroll_context, add_special_tokens=False)
            feedback_requests.append(
                {
                    "idx": idx,
                    "valid_prompt_ids": valid_prompt_ids,
                    "problem": problem,
                    "reroll_context": reroll_context,
                }
            )

        appended_idxs = []
        for request in feedback_requests:
            idx = request["idx"]
            reroll_context = self._normalize_reroll_context(request["reroll_context"])

            if len(self.tokenizer.encode(reroll_context, add_special_tokens=False)) > max_context_tokens:
                self._debug_progress(
                    f"gpt_reroll_context:too_long idx={idx} limit={max_context_tokens}; truncating hint"
                )
                reroll_context = self._truncate_reroll_context(reroll_context, max_context_tokens)
            if not reroll_context:
                continue

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
                self._debug_progress(
                    f"gpt_reroll_prompt:skip_too_long idx={idx} tokens={len(combined_ids)} limit={prompt_length}"
                )
                continue

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

            appended_idxs.append(idx)

        return appended_idxs, local_hint_fallback_count

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

    def _ensure_g_opd_sample_metadata(
        self,
        batch: DataProto,
        *,
        kind: str = "orig",
        loss_weight: float = 1.0,
    ) -> None:
        n = len(batch)
        defaults = {
            "g_opd_sample_kind": np.full(n, kind, dtype=object),
            "g_opd_loss_weight": np.full(n, float(loss_weight), dtype=np.float32),
            "g_opd_reroll_sft_weight": np.zeros(n, dtype=np.float32),
            "g_opd_source_row_idx": np.arange(n, dtype=object),
            "g_opd_reroll_gain_100": np.zeros(n, dtype=np.float32),
            "gpt_rollout_reroll_count": np.zeros(n, dtype=object),
        }
        for key, value in defaults.items():
            if key not in batch.non_tensor_batch or len(batch.non_tensor_batch[key]) != n:
                batch.non_tensor_batch[key] = value

    def _set_reroll_sample_metadata(
        self,
        batch: DataProto,
        *,
        source_row_idxs: list[int],
        initial_score_result: dict,
        reroll_score_result: dict,
        loss_weight: float,
    ) -> None:
        n = len(batch)
        gains = []
        for output_position, source_idx in enumerate(source_row_idxs):
            initial_score = initial_score_result["scores_100"][source_idx]
            reroll_score = reroll_score_result["scores_100"][output_position]
            initial_value = self._finite_gpt_score_value(initial_score)
            reroll_value = self._finite_gpt_score_value(reroll_score)
            gains.append(0.0 if initial_value is None or reroll_value is None else reroll_value - initial_value)

        batch.non_tensor_batch["g_opd_sample_kind"] = np.full(n, "reroll_hint", dtype=object)
        batch.non_tensor_batch["g_opd_loss_weight"] = np.full(n, float(loss_weight), dtype=np.float32)
        batch.non_tensor_batch["g_opd_reroll_sft_weight"] = np.zeros(n, dtype=np.float32)
        batch.non_tensor_batch["g_opd_source_row_idx"] = np.array(source_row_idxs, dtype=object)
        batch.non_tensor_batch["g_opd_reroll_gain_100"] = np.array(gains, dtype=np.float32)
        batch.non_tensor_batch["gpt_rollout_reroll_count"] = np.ones(n, dtype=object)
        if "uid" in batch.non_tensor_batch:
            batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(n)], dtype=object)

    def _build_reroll_nohint_batch(
        self,
        *,
        original_batch: DataProto,
        reroll_batch: DataProto,
        source_row_idxs: list[int],
        initial_score_result: dict,
        reroll_score_result: dict,
        scorer_config: dict,
        threshold_100: float,
    ) -> DataProto | None:
        if not _is_truthy(scorer_config.get("reroll_nohint_enable", False)):
            return None

        min_score = float(scorer_config.get("reroll_nohint_min_score", threshold_100))
        min_gain = float(scorer_config.get("reroll_nohint_min_gain", 10.0))
        max_weight = float(scorer_config.get("reroll_nohint_max_weight", 0.5))
        gain_norm = max(float(scorer_config.get("reroll_nohint_gain_norm", 50.0)), 1e-6)

        selected_positions = []
        selected_source_idxs = []
        selected_weights = []
        selected_gains = []
        for output_position, source_idx in enumerate(source_row_idxs):
            initial_value = self._finite_gpt_score_value(initial_score_result["scores_100"][source_idx])
            reroll_value = self._finite_gpt_score_value(reroll_score_result["scores_100"][output_position])
            if initial_value is None or reroll_value is None:
                continue
            gain = reroll_value - initial_value
            if reroll_value < min_score or gain < min_gain:
                continue
            selected_positions.append(output_position)
            selected_source_idxs.append(source_idx)
            selected_gains.append(gain)
            selected_weights.append(max_weight * float(np.clip(gain / gain_norm, 0.0, 1.0)))

        if not selected_positions:
            return None

        nohint_batch = original_batch.select_idxs(selected_source_idxs)
        reroll_rows = reroll_batch.select_idxs(selected_positions)
        prompt_len = nohint_batch.batch["prompts"].shape[-1]

        nohint_batch.batch["responses"] = reroll_rows.batch["responses"].clone()
        for stale_key in ("rollout_log_probs", "rollout_is_weights"):
            if stale_key in nohint_batch.batch:
                nohint_batch.batch.pop(stale_key)
        nohint_batch.batch["input_ids"] = torch.cat(
            [nohint_batch.batch["prompts"], nohint_batch.batch["responses"]],
            dim=-1,
        )
        prompt_attention_mask = nohint_batch.batch["attention_mask"][:, :prompt_len].clone()
        reroll_response_mask = reroll_rows.batch.get("response_mask", None)
        if reroll_response_mask is None:
            reroll_response_mask = reroll_rows.batch["attention_mask"][:, prompt_len:]
        nohint_batch.batch["attention_mask"] = torch.cat(
            [prompt_attention_mask, reroll_response_mask.to(prompt_attention_mask.device)],
            dim=-1,
        )
        new_position_ids = compute_position_id_with_mask(nohint_batch.batch["attention_mask"])
        if nohint_batch.batch["position_ids"].dim() == 2:
            nohint_batch.batch["position_ids"] = new_position_ids.to(
                device=nohint_batch.batch["position_ids"].device,
                dtype=nohint_batch.batch["position_ids"].dtype,
            )
        elif nohint_batch.batch["position_ids"].dim() == 3:
            nohint_batch.batch["position_ids"] = (
                new_position_ids.to(
                    device=nohint_batch.batch["position_ids"].device,
                    dtype=nohint_batch.batch["position_ids"].dtype,
                )
                .unsqueeze(1)
                .expand_as(nohint_batch.batch["position_ids"])
            )
        nohint_batch.batch["response_mask"] = compute_response_mask(nohint_batch)

        selected_score_result = self._select_gpt_rollout_result(reroll_score_result, selected_positions)
        self._set_gpt_rollout_result(
            batch=nohint_batch,
            result=selected_score_result,
            prefix="gpt_rollout",
            threshold_100=threshold_100,
        )
        nohint_batch.non_tensor_batch["g_opd_sample_kind"] = np.full(len(nohint_batch), "reroll_nohint", dtype=object)
        nohint_batch.non_tensor_batch["g_opd_loss_weight"] = np.array(selected_weights, dtype=np.float32)
        nohint_batch.non_tensor_batch["g_opd_reroll_sft_weight"] = np.zeros(len(nohint_batch), dtype=np.float32)
        nohint_batch.non_tensor_batch["g_opd_source_row_idx"] = np.array(selected_source_idxs, dtype=object)
        nohint_batch.non_tensor_batch["g_opd_reroll_gain_100"] = np.array(selected_gains, dtype=np.float32)
        nohint_batch.non_tensor_batch["gpt_rollout_reroll_count"] = np.ones(len(nohint_batch), dtype=object)
        if "uid" in nohint_batch.non_tensor_batch:
            nohint_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(nohint_batch))], dtype=object)
        return nohint_batch

    def _reroll_sft_weighted_scores(
        self,
        *,
        initial_score_result: dict,
        reroll_score_result: dict,
        source_row_idxs: list[int],
        scorer_config: dict,
    ) -> list[float]:
        score_coef = float(scorer_config.get("reroll_sft_score_coef", 1.0))
        gain_coef = float(scorer_config.get("reroll_sft_gain_coef", 0.0))
        rewards = []
        for output_position, source_idx in enumerate(source_row_idxs):
            reroll_value = self._finite_gpt_score_value(reroll_score_result["scores_100"][output_position])
            initial_value = self._finite_gpt_score_value(initial_score_result["scores_100"][source_idx])
            if reroll_value is None:
                rewards.append(float("nan"))
                continue
            gain_value = 0.0 if initial_value is None else reroll_value - initial_value
            reward = (score_coef * reroll_value + gain_coef * gain_value) / 100.0
            rewards.append(float(reward))
        return rewards

    def _compute_reroll_sft_weights(self, rewards: list[float], scorer_config: dict) -> np.ndarray:
        rewards_array = np.array(rewards, dtype=np.float32)
        finite_mask = np.isfinite(rewards_array)
        weights = np.zeros_like(rewards_array, dtype=np.float32)
        if not finite_mask.any():
            return weights

        alpha = float(scorer_config.get("reroll_sft_alpha", 1.0))
        z_clip = abs(float(scorer_config.get("reroll_sft_z_clip", 2.0)))
        weight_min = max(0.0, float(scorer_config.get("reroll_sft_weight_min", 0.1)))
        weight_max = max(weight_min, float(scorer_config.get("reroll_sft_weight_max", 4.0)))
        lambda_weight = max(0.0, float(scorer_config.get("reroll_sft_lambda", 0.3)))
        normalize = _is_truthy(scorer_config.get("reroll_sft_normalize_weights", True))
        std_floor = max(float(scorer_config.get("reroll_sft_std_floor", 1e-6)), 1e-6)

        valid_rewards = rewards_array[finite_mask]
        mean = float(np.mean(valid_rewards))
        std = max(float(np.std(valid_rewards)), std_floor)
        z_scores = np.clip((rewards_array - mean) / std, -z_clip, z_clip)
        raw_weights = np.exp(alpha * z_scores).astype(np.float32)
        raw_weights = np.clip(raw_weights, weight_min, weight_max)
        if normalize:
            mean_weight = float(np.mean(raw_weights[finite_mask]))
            if np.isfinite(mean_weight) and mean_weight > 1e-8:
                raw_weights = raw_weights / mean_weight
        weights[finite_mask] = raw_weights[finite_mask] * lambda_weight
        return weights.astype(np.float32)

    def _build_reroll_sft_batch(
        self,
        *,
        original_batch: DataProto,
        reroll_batch: DataProto,
        source_row_idxs: list[int],
        initial_score_result: dict,
        reroll_score_result: dict,
        scorer_config: dict,
        threshold_100: float,
    ) -> DataProto | None:
        if not _is_truthy(scorer_config.get("reroll_sft_enable", False)):
            return None

        rewards = self._reroll_sft_weighted_scores(
            initial_score_result=initial_score_result,
            reroll_score_result=reroll_score_result,
            source_row_idxs=source_row_idxs,
            scorer_config=scorer_config,
        )
        weights = self._compute_reroll_sft_weights(rewards, scorer_config)
        selected_positions = np.where(weights > 0.0)[0].astype(np.int64).tolist()
        if not selected_positions:
            return None

        selected_source_idxs = [source_row_idxs[position] for position in selected_positions]
        sft_batch = original_batch.select_idxs(selected_source_idxs)
        reroll_rows = reroll_batch.select_idxs(selected_positions)
        prompt_len = sft_batch.batch["prompts"].shape[-1]

        sft_batch.batch["responses"] = reroll_rows.batch["responses"].clone()
        for stale_key in ("rollout_log_probs", "rollout_is_weights"):
            if stale_key in sft_batch.batch:
                sft_batch.batch.pop(stale_key)
        sft_batch.batch["input_ids"] = torch.cat(
            [sft_batch.batch["prompts"], sft_batch.batch["responses"]],
            dim=-1,
        )
        prompt_attention_mask = sft_batch.batch["attention_mask"][:, :prompt_len].clone()
        reroll_response_mask = reroll_rows.batch.get("response_mask", None)
        if reroll_response_mask is None:
            reroll_response_mask = reroll_rows.batch["attention_mask"][:, prompt_len:]
        sft_batch.batch["attention_mask"] = torch.cat(
            [prompt_attention_mask, reroll_response_mask.to(prompt_attention_mask.device)],
            dim=-1,
        )
        new_position_ids = compute_position_id_with_mask(sft_batch.batch["attention_mask"])
        if sft_batch.batch["position_ids"].dim() == 2:
            sft_batch.batch["position_ids"] = new_position_ids.to(
                device=sft_batch.batch["position_ids"].device,
                dtype=sft_batch.batch["position_ids"].dtype,
            )
        elif sft_batch.batch["position_ids"].dim() == 3:
            sft_batch.batch["position_ids"] = (
                new_position_ids.to(
                    device=sft_batch.batch["position_ids"].device,
                    dtype=sft_batch.batch["position_ids"].dtype,
                )
                .unsqueeze(1)
                .expand_as(sft_batch.batch["position_ids"])
            )
        sft_batch.batch["response_mask"] = compute_response_mask(sft_batch)

        selected_score_result = self._select_gpt_rollout_result(reroll_score_result, selected_positions)
        self._set_gpt_rollout_result(
            batch=sft_batch,
            result=selected_score_result,
            prefix="gpt_rollout",
            threshold_100=threshold_100,
        )
        selected_gains = []
        for output_position, source_idx in zip(selected_positions, selected_source_idxs):
            initial_value = self._finite_gpt_score_value(initial_score_result["scores_100"][source_idx])
            reroll_value = self._finite_gpt_score_value(reroll_score_result["scores_100"][output_position])
            selected_gains.append(0.0 if initial_value is None or reroll_value is None else reroll_value - initial_value)
        sft_batch.non_tensor_batch["g_opd_sample_kind"] = np.full(len(sft_batch), "reroll_sft", dtype=object)
        sft_batch.non_tensor_batch["g_opd_loss_weight"] = np.zeros(len(sft_batch), dtype=np.float32)
        sft_batch.non_tensor_batch["g_opd_reroll_sft_weight"] = weights[selected_positions].astype(np.float32)
        sft_batch.non_tensor_batch["g_opd_source_row_idx"] = np.array(selected_source_idxs, dtype=object)
        sft_batch.non_tensor_batch["g_opd_reroll_gain_100"] = np.array(selected_gains, dtype=np.float32)
        sft_batch.non_tensor_batch["gpt_rollout_reroll_count"] = np.ones(len(sft_batch), dtype=object)
        if "uid" in sft_batch.non_tensor_batch:
            sft_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(sft_batch))], dtype=object)
        return sft_batch

    def _concat_training_batches(self, batches: list[DataProto]) -> DataProto:
        batches = [batch for batch in batches if batch is not None and len(batch) > 0]
        if len(batches) == 1:
            return batches[0]

        batch_key_sets = [set(batch.batch.keys()) for batch in batches if batch.batch is not None]
        if batch_key_sets:
            common_batch_keys = set.intersection(*batch_key_sets)
            if any(keys != common_batch_keys for keys in batch_key_sets):
                batches = [
                    batch.select(
                        batch_keys=list(common_batch_keys),
                        non_tensor_batch_keys=list(batch.non_tensor_batch.keys()),
                    )
                    for batch in batches
                ]

        all_non_tensor_keys = set()
        for batch in batches:
            all_non_tensor_keys.update(batch.non_tensor_batch.keys())
        for batch in batches:
            for key in all_non_tensor_keys:
                if key not in batch.non_tensor_batch:
                    if key == "g_opd_reroll_sft_weight":
                        batch.non_tensor_batch[key] = np.zeros(len(batch), dtype=np.float32)
                    else:
                        batch.non_tensor_batch[key] = np.full(len(batch), None, dtype=object)

        return DataProto.concat(batches)

    def _worker_group_dp_size(self, worker_group, mesh_name: str) -> int:
        if worker_group is None:
            return 1
        try:
            if mesh_name not in worker_group._dispatch_info:
                worker_group._dispatch_info[mesh_name] = worker_group._query_dispatch_info(mesh_name)
            dp_rank_mapping = worker_group._dispatch_info[mesh_name]
        except Exception as exc:
            self._debug_progress(f"update_batch_pad:failed_to_query_dp_size mesh={mesh_name} error={exc}")
            return 1
        if not dp_rank_mapping:
            return 1
        return max(dp_rank_mapping) + 1

    def _actor_critic_update_divisor(self) -> int:
        divisor = self._worker_group_dp_size(self.actor_rollout_wg, "actor")
        if self.use_critic:
            critic_divisor = self._worker_group_dp_size(self.critic_wg, "critic")
            divisor = math.lcm(max(divisor, 1), max(critic_divisor, 1))
        return max(divisor, 1)

    def _zero_padding_rows(self, batch: DataProto) -> None:
        padding_flags = batch.non_tensor_batch.get("g_opd_padding_sample", None)
        if padding_flags is None:
            return
        pad_idxs = np.where(np.array([bool(flag) for flag in padding_flags.tolist()], dtype=bool))[0]
        if len(pad_idxs) == 0:
            return
        for key in ("response_mask", "advantages", "returns", "token_level_scores", "token_level_rewards"):
            if key in batch.batch:
                batch.batch[key][pad_idxs] = 0
        if "values" in batch.batch:
            batch.batch["values"][pad_idxs] = 0
        if "loss_mask" in batch.batch:
            batch.batch["loss_mask"][pad_idxs] = 0
        if "g_opd_loss_weight" in batch.non_tensor_batch:
            batch.non_tensor_batch["g_opd_loss_weight"][pad_idxs] = 0.0
        if "g_opd_reroll_sft_weight" in batch.non_tensor_batch:
            batch.non_tensor_batch["g_opd_reroll_sft_weight"][pad_idxs] = 0.0

    def _pad_training_batch_to_update_divisor(
        self,
        batch: DataProto,
        metrics: dict,
        *,
        metric_prefix: str,
        zero_training_signals: bool,
    ) -> DataProto:
        divisor = self._actor_critic_update_divisor()
        remainder = len(batch) % divisor
        pad_size = 0 if remainder == 0 else divisor - remainder
        metrics[f"{metric_prefix}/original_size"] = len(batch)
        metrics[f"{metric_prefix}/divisor"] = divisor
        metrics[f"{metric_prefix}/pad_size"] = pad_size
        if pad_size <= 0:
            if zero_training_signals:
                self._zero_padding_rows(batch)
            return batch

        padded_batch, actual_pad_size = pad_dataproto_to_divisor(batch, divisor)
        pad_start = len(batch)
        pad_idxs = np.arange(pad_start, pad_start + actual_pad_size, dtype=np.int64)

        old_padding_flags = padded_batch.non_tensor_batch.get("g_opd_padding_sample", None)
        padded_batch.non_tensor_batch["g_opd_padding_sample"] = np.full(len(padded_batch), False, dtype=object)
        if old_padding_flags is not None:
            padded_batch.non_tensor_batch["g_opd_padding_sample"][: len(old_padding_flags)] = old_padding_flags
        padded_batch.non_tensor_batch["g_opd_padding_sample"][pad_idxs] = True
        if "g_opd_loss_weight" not in padded_batch.non_tensor_batch:
            padded_batch.non_tensor_batch["g_opd_loss_weight"] = np.full(len(padded_batch), 1.0, dtype=np.float32)
        padded_batch.non_tensor_batch["g_opd_loss_weight"][pad_idxs] = 0.0
        if "g_opd_reroll_sft_weight" in padded_batch.non_tensor_batch:
            padded_batch.non_tensor_batch["g_opd_reroll_sft_weight"][pad_idxs] = 0.0
        if "g_opd_sample_kind" in padded_batch.non_tensor_batch:
            padded_batch.non_tensor_batch["g_opd_sample_kind"][pad_idxs] = "padding"
        if "uid" in padded_batch.non_tensor_batch:
            padded_batch.non_tensor_batch["uid"][pad_idxs] = np.array(
                [f"padding-{self.global_steps}-{i}" for i in range(actual_pad_size)],
                dtype=object,
            )
        if "response_mask" in padded_batch.batch:
            padded_batch.batch["response_mask"][pad_idxs] = 0
        if zero_training_signals:
            self._zero_padding_rows(padded_batch)

        padded_batch.meta_info.update(batch.meta_info)
        metrics[f"{metric_prefix}/final_size"] = len(padded_batch)
        self._debug_progress(
            f"{metric_prefix}: padded size {len(batch)} -> {len(padded_batch)} "
            f"divisor={divisor} pad={actual_pad_size} zero_training_signals={zero_training_signals}"
        )
        return padded_batch

    def _rank_descending(self, values: list[float | None], eligible_mask: Optional[np.ndarray] = None) -> np.ndarray:
        ranks = np.full(len(values), -1, dtype=np.int64)
        if eligible_mask is None:
            eligible_mask = np.ones(len(values), dtype=bool)
        valid = [
            (idx, float(value))
            for idx, value in enumerate(values)
            if eligible_mask[idx] and value is not None and np.isfinite(value)
        ]
        valid.sort(key=lambda item: item[1], reverse=True)
        for rank, (idx, _) in enumerate(valid):
            ranks[idx] = rank
        return ranks

    def _rank_gap_direction(self, teacher_rank: int, rubric_rank: int) -> str:
        if teacher_rank < rubric_rank:
            return "rubric_low_teacher_high"
        if rubric_rank < teacher_rank:
            return "rubric_high_teacher_low"
        return "rank_tie"

    def _gpt_history_random_bucket(
        self,
        *,
        row_idx: int,
        uid: object | None,
        bucket_count: int,
        seed: int,
    ) -> str:
        bucket_count = max(1, int(bucket_count))
        key = f"{seed}:{uid if uid is not None else row_idx}"
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        bucket_idx = int.from_bytes(digest, byteorder="big", signed=False) % bucket_count
        return f"random_{bucket_idx}"

    def _history_zscore_shift(
        self,
        *,
        scorer_config,
        rubric_scores: list[float | None],
        valid_mask: np.ndarray,
        non_padding_mask: np.ndarray,
        batch: DataProto,
        coef: float,
        clip: float,
        metrics: dict,
    ) -> np.ndarray:
        shift = np.zeros(len(rubric_scores), dtype=np.float32)
        history_size = max(1, int(scorer_config.get("history_size", 2048)))
        warmup_steps = max(0, int(scorer_config.get("history_warmup_steps", 5)))
        min_bin_count = max(1, int(scorer_config.get("history_min_bin_count", 64)))
        global_min_count = max(1, int(scorer_config.get("history_global_min_count", 256)))
        std_floor = max(float(scorer_config.get("history_std_floor", 12.5)), 1e-6)
        z_clip = abs(float(scorer_config.get("history_z_clip", 2.0)))
        negative_coef_scale = max(0.0, float(scorer_config.get("history_negative_coef_scale", 0.5)))
        min_component_count = max(1, min_bin_count // 2)
        bucket_mode = str(scorer_config.get("history_bucket_mode", "label")).strip().lower()
        if bucket_mode not in {"label", "random"}:
            bucket_mode = "label"
        random_bucket_count = max(1, int(scorer_config.get("history_num_bins", 12)))
        random_bucket_seed = int(scorer_config.get("history_random_bucket_seed", 42))
        valid_domains = {
            "geometry_visual",
            "algebra_symbolic",
            "discrete_counting_process",
            "arithmetic_number_modeling",
        }
        valid_difficulties = {"easy", "medium", "hard"}

        while len(self._gpt_rubric_history) > history_size:
            self._gpt_rubric_history.popleft()

        history = list(self._gpt_rubric_history)
        history_count = int(len(history))
        metrics["g_opd_rubric_adv_shift/history_count"] = history_count
        metrics["g_opd_rubric_adv_shift/history_num_bins"] = 12 if bucket_mode == "label" else random_bucket_count
        metrics["g_opd_rubric_adv_shift/history_bucket_mode_label"] = 1 if bucket_mode == "label" else 0
        metrics["g_opd_rubric_adv_shift/history_bucket_mode_random"] = 1 if bucket_mode == "random" else 0
        metrics["g_opd_rubric_adv_shift/history_random_bucket_seed"] = random_bucket_seed
        metrics["g_opd_rubric_adv_shift/history_warmup_steps"] = warmup_steps
        metrics["g_opd_rubric_adv_shift/history_min_bin_count"] = min_bin_count
        metrics["g_opd_rubric_adv_shift/history_min_component_count"] = min_component_count
        metrics["g_opd_rubric_adv_shift/history_global_min_count"] = global_min_count
        metrics["g_opd_rubric_adv_shift/history_std_floor"] = std_floor
        metrics["g_opd_rubric_adv_shift/history_z_clip"] = z_clip
        metrics["g_opd_rubric_adv_shift/history_negative_coef_scale"] = negative_coef_scale

        use_history = self.global_steps >= warmup_steps and history_count >= global_min_count
        if use_history:
            bucket_scores: dict[tuple[str, str] | str, list[float]] = defaultdict(list)
            domain_scores: dict[str, list[float]] = defaultdict(list)
            difficulty_scores: dict[str, list[float]] = defaultdict(list)
            global_scores: list[float] = []
            for item in history:
                if not isinstance(item, dict):
                    continue
                domain = str(item.get("domain") or "").strip().lower()
                difficulty = str(item.get("difficulty") or "").strip().lower()
                try:
                    rubric_value = float(item.get("rubric"))
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(rubric_value):
                    continue
                if bucket_mode == "random":
                    bucket = str(item.get("bucket") or "").strip().lower()
                    if not bucket:
                        continue
                    bucket_scores[bucket].append(rubric_value)
                else:
                    if domain not in valid_domains or difficulty not in valid_difficulties:
                        continue
                    bucket_scores[(domain, difficulty)].append(rubric_value)
                    domain_scores[domain].append(rubric_value)
                    difficulty_scores[difficulty].append(rubric_value)
                global_scores.append(rubric_value)

            if len(global_scores) < global_min_count:
                use_history = False
            else:
                global_array = np.array(global_scores, dtype=np.float32)
                global_mean = float(np.mean(global_array))
                global_std = max(float(np.std(global_array)), std_floor)
                if bucket_mode == "label":
                    for domain in sorted(valid_domains):
                        metrics[f"g_opd_rubric_adv_shift/history_domain_count/{domain}"] = int(
                            len(domain_scores.get(domain, []))
                        )
                    for difficulty in ("easy", "medium", "hard"):
                        metrics[f"g_opd_rubric_adv_shift/history_difficulty_count/{difficulty}"] = int(
                            len(difficulty_scores.get(difficulty, []))
                        )
                else:
                    for bucket_idx in range(random_bucket_count):
                        bucket_key = f"random_{bucket_idx}"
                        metrics[f"g_opd_rubric_adv_shift/history_random_bucket_count/{bucket_key}"] = int(
                            len(bucket_scores.get(bucket_key, []))
                        )

            rubric_array = np.array(
                [np.nan if value is None else float(value) for value in rubric_scores],
                dtype=np.float32,
            )
            used_bin_count = 0
            used_blended_count = 0
            used_domain_count = 0
            used_difficulty_count = 0
            used_global_count = 0
            skipped_missing_label_count = 0
            z_values: list[float] = []
            if use_history:
                domains = batch.non_tensor_batch.get("gpt_rollout_problem_domain", None)
                difficulties = batch.non_tensor_batch.get("gpt_rollout_difficulty_3", None)
                uids = batch.non_tensor_batch.get("uid", None)
                domains_list = domains.tolist() if domains is not None and hasattr(domains, "tolist") else domains
                difficulties_list = (
                    difficulties.tolist() if difficulties is not None and hasattr(difficulties, "tolist") else difficulties
                )
                uids_list = uids.tolist() if uids is not None and hasattr(uids, "tolist") else uids
                for idx in np.where(valid_mask)[0]:
                    rubric_value = float(rubric_array[idx])
                    if not np.isfinite(rubric_value):
                        continue
                    if bucket_mode == "random":
                        bucket = self._gpt_history_random_bucket(
                            row_idx=int(idx),
                            uid=(
                                uids_list[idx]
                                if uids_list is not None and idx < len(uids_list)
                                else None
                            ),
                            bucket_count=random_bucket_count,
                            seed=random_bucket_seed,
                        )
                        exact = bucket_scores.get(bucket, [])
                        if len(exact) >= min_bin_count:
                            selected_scores = exact
                            used_bin_count += 1
                        elif len(exact) > 0:
                            selected_scores = list(global_scores) + list(exact) * max(
                                1,
                                min_bin_count // max(len(exact), 1),
                            )
                            used_blended_count += 1
                        else:
                            selected_scores = list(global_scores)
                            used_global_count += 1
                        score_array = np.array(selected_scores, dtype=np.float32)
                        baseline_mean = float(np.mean(score_array))
                        baseline_std = max(float(np.std(score_array)), std_floor)
                        z_value = float(np.clip((rubric_value - baseline_mean) / baseline_std, -z_clip, z_clip))
                        z_values.append(z_value)
                        local_coef = coef if z_value >= 0.0 else coef * negative_coef_scale
                        shift[idx] = float(np.clip(local_coef * z_value, -clip, clip))
                        continue
                    domain = (
                        str(domains_list[idx] or "").strip().lower()
                        if domains_list is not None and idx < len(domains_list)
                        else ""
                    )
                    difficulty = (
                        str(difficulties_list[idx] or "").strip().lower()
                        if difficulties_list is not None and idx < len(difficulties_list)
                        else ""
                    )
                    if domain not in valid_domains or difficulty not in valid_difficulties:
                        skipped_missing_label_count += 1
                        continue

                    exact = bucket_scores.get((domain, difficulty), [])
                    domain_values = domain_scores.get(domain, [])
                    difficulty_values = difficulty_scores.get(difficulty, [])
                    selected_scores: list[float]
                    if len(exact) >= min_bin_count:
                        selected_scores = exact
                        used_bin_count += 1
                    else:
                        components = [list(global_scores)]
                        if len(domain_values) >= min_component_count:
                            components.append(list(domain_values))
                        if len(difficulty_values) >= min_component_count:
                            components.append(list(difficulty_values))
                        if len(exact) > 0:
                            components.append(list(exact) * max(1, min_bin_count // max(len(exact), 1)))
                        selected_scores = [value for component in components for value in component]
                        if len(domain_values) >= min_component_count and len(difficulty_values) >= min_component_count:
                            used_blended_count += 1
                        elif len(domain_values) >= min_component_count:
                            used_domain_count += 1
                        elif len(difficulty_values) >= min_component_count:
                            used_difficulty_count += 1
                        else:
                            used_global_count += 1

                    score_array = np.array(selected_scores, dtype=np.float32)
                    baseline_mean = float(np.mean(score_array))
                    baseline_std = max(float(np.std(score_array)), std_floor)
                    z_value = float(np.clip((rubric_value - baseline_mean) / baseline_std, -z_clip, z_clip))
                    z_values.append(z_value)
                    local_coef = coef if z_value >= 0.0 else coef * negative_coef_scale
                    shift[idx] = float(np.clip(local_coef * z_value, -clip, clip))

                metrics["g_opd_rubric_adv_shift/history_active"] = 1
                metrics["g_opd_rubric_adv_shift/history_global_mean"] = global_mean
                metrics["g_opd_rubric_adv_shift/history_global_std"] = global_std
                metrics["g_opd_rubric_adv_shift/history_bucket_count"] = len(bucket_scores)
                metrics["g_opd_rubric_adv_shift/history_used_bin_count"] = used_bin_count
                metrics["g_opd_rubric_adv_shift/history_used_blended_count"] = used_blended_count
                metrics["g_opd_rubric_adv_shift/history_used_domain_count"] = used_domain_count
                metrics["g_opd_rubric_adv_shift/history_used_difficulty_count"] = used_difficulty_count
                metrics["g_opd_rubric_adv_shift/history_used_global_count"] = used_global_count
                metrics["g_opd_rubric_adv_shift/history_missing_label_count"] = skipped_missing_label_count
                if z_values:
                    z_array = np.array(z_values, dtype=np.float32)
                    metrics["g_opd_rubric_adv_shift/history_z_mean"] = float(np.mean(z_array))
                    metrics["g_opd_rubric_adv_shift/history_z_max_abs"] = float(np.max(np.abs(z_array)))
            else:
                metrics["g_opd_rubric_adv_shift/history_active"] = 0
        else:
            metrics["g_opd_rubric_adv_shift/history_active"] = 0

        update_mask = valid_mask & non_padding_mask
        sample_kinds = batch.non_tensor_batch.get("g_opd_sample_kind", None)
        if sample_kinds is not None:
            update_mask &= np.array([str(kind or "orig") == "orig" for kind in sample_kinds.tolist()], dtype=bool)
        domains = batch.non_tensor_batch.get("gpt_rollout_problem_domain", None)
        difficulties = batch.non_tensor_batch.get("gpt_rollout_difficulty_3", None)
        uids = batch.non_tensor_batch.get("uid", None)
        domains_list = domains.tolist() if domains is not None and hasattr(domains, "tolist") else domains
        difficulties_list = difficulties.tolist() if difficulties is not None and hasattr(difficulties, "tolist") else difficulties
        uids_list = uids.tolist() if uids is not None and hasattr(uids, "tolist") else uids
        added = 0
        skipped_label_update_count = 0
        for idx in np.where(update_mask)[0]:
            rubric_value = rubric_scores[idx]
            if rubric_value is None:
                continue
            rubric_value = float(rubric_value)
            if not np.isfinite(rubric_value):
                continue
            if bucket_mode == "random":
                bucket = self._gpt_history_random_bucket(
                    row_idx=int(idx),
                    uid=(
                        uids_list[idx]
                        if uids_list is not None and idx < len(uids_list)
                        else None
                    ),
                    bucket_count=random_bucket_count,
                    seed=random_bucket_seed,
                )
                self._gpt_rubric_history.append({"bucket": bucket, "rubric": rubric_value})
                added += 1
                continue
            domain = (
                str(domains_list[idx] or "").strip().lower()
                if domains_list is not None and idx < len(domains_list)
                else ""
            )
            difficulty = (
                str(difficulties_list[idx] or "").strip().lower()
                if difficulties_list is not None and idx < len(difficulties_list)
                else ""
            )
            if domain not in valid_domains or difficulty not in valid_difficulties:
                skipped_label_update_count += 1
                continue
            self._gpt_rubric_history.append({"domain": domain, "difficulty": difficulty, "rubric": rubric_value})
            added += 1
        while len(self._gpt_rubric_history) > history_size:
            self._gpt_rubric_history.popleft()
        metrics["g_opd_rubric_adv_shift/history_added_count"] = added
        metrics["g_opd_rubric_adv_shift/history_skipped_label_update_count"] = skipped_label_update_count
        metrics["g_opd_rubric_adv_shift/history_count_after_update"] = int(len(self._gpt_rubric_history))
        return shift

    def _rank_gap_entries(
        self,
        batch: DataProto,
        *,
        teacher_scores: list[float | None],
        rubric_scores: list[float | None],
        teacher_ranks: np.ndarray,
        rubric_ranks: np.ndarray,
        rank_gap: np.ndarray,
        high_gap_mask: np.ndarray,
        drop_mask: np.ndarray,
        drop_threshold: float,
    ) -> list[dict]:
        entries = []
        for idx in np.where(high_gap_mask)[0].tolist():
            prompt_text, response_text = self._decode_rollout_prompt_response(batch=batch, idx=idx)
            extra_info = self._get_non_tensor_row_value(batch, "extra_info", idx, {}) or {}
            reward_model = self._get_non_tensor_row_value(batch, "reward_model", idx, {}) or {}
            ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
            if ground_truth is None and isinstance(extra_info, dict):
                ground_truth = extra_info.get("answer")
            teacher_rank = int(teacher_ranks[idx])
            rubric_rank = int(rubric_ranks[idx])
            entries.append(
                {
                    "step": self.global_steps,
                    "row_idx": int(idx),
                    "direction": self._rank_gap_direction(teacher_rank, rubric_rank),
                    "dropped": bool(drop_mask[idx]),
                    "rank_gap": float(rank_gap[idx]),
                    "drop_threshold": float(drop_threshold),
                    "teacher_rank": teacher_rank,
                    "rubric_rank": rubric_rank,
                    "teacher_score": teacher_scores[idx],
                    "rubric_score_100": rubric_scores[idx],
                    "gpt_problem_domain": self._get_non_tensor_row_value(
                        batch,
                        "gpt_rollout_problem_domain",
                        idx,
                        None,
                    ),
                    "gpt_difficulty_3": self._get_non_tensor_row_value(
                        batch,
                        "gpt_rollout_difficulty_3",
                        idx,
                        None,
                    ),
                    "sample_kind": self._get_non_tensor_row_value(batch, "g_opd_sample_kind", idx, "orig"),
                    "source_row_idx": self._get_non_tensor_row_value(batch, "g_opd_source_row_idx", idx, None),
                    "reroll_count": self._get_non_tensor_row_value(batch, "gpt_rollout_reroll_count", idx, None),
                    "reroll_gain_100": self._get_non_tensor_row_value(batch, "g_opd_reroll_gain_100", idx, None),
                    "problem": extra_info.get("problem") if isinstance(extra_info, dict) else prompt_text,
                    "ground_truth": ground_truth,
                    "output": response_text,
                    "gpt_weighted_score_1_to_4": self._get_non_tensor_row_value(
                        batch,
                        "gpt_rollout_weighted_score_1_to_4",
                        idx,
                        None,
                    ),
                    "gpt_revision_suggestion": self._get_non_tensor_row_value(
                        batch,
                        "gpt_rollout_revision_suggestion",
                        idx,
                        None,
                    ),
                    "gpt_revision_suggestion_source": self._get_non_tensor_row_value(
                        batch,
                        "gpt_rollout_revision_suggestion_source",
                        idx,
                        None,
                    ),
                    "gpt_rubric_scores": self._get_non_tensor_row_value(
                        batch,
                        "gpt_rollout_rubric_scores",
                        idx,
                        None,
                    ),
                    "data_source": self._get_non_tensor_row_value(batch, "data_source", idx, None),
                    "request_id": self._get_non_tensor_row_value(batch, "request_id", idx, None),
                    "uid": self._get_non_tensor_row_value(batch, "uid", idx, None),
                    "extra_info": extra_info,
                    "reward_model": reward_model,
                }
            )
        return entries

    def _maybe_dump_rank_gap_examples(
        self,
        *,
        scorer_config: dict,
        entries: list[dict],
        timing_raw: Optional[dict],
    ) -> None:
        if not entries:
            return
        dump_dir = scorer_config.get("rank_gap_case_study_dir", None)
        if not dump_dir or str(dump_dir).strip().lower() in {"none", "null", "false", "0"}:
            dump_dir = scorer_config.get("case_study_dir", None)
        if not dump_dir or str(dump_dir).strip().lower() in {"none", "null", "false", "0"}:
            return

        filename = os.path.join(str(dump_dir), "rank_gap", f"{self.global_steps}.jsonl")
        if timing_raw is None:
            self._write_rows_to_jsonl(entries, filename)
        else:
            with marked_timer("dump_rank_gap_cases", timing_raw, color="green"):
                self._write_rows_to_jsonl(entries, filename)
        self._debug_progress(f"rank_gap_case_study:dumped count={len(entries)} file={filename}")

    def _rank_gap_drop_and_rubric_shift(
        self,
        batch: DataProto,
        metrics: dict,
        timing_raw: Optional[dict] = None,
    ) -> DataProto:
        scorer_config = self.config.trainer.get("gpt_rollout_score", None)
        if not scorer_config or not _is_truthy(scorer_config.get("enable", False)):
            return batch
        if "gpt_rollout_score_100" not in batch.non_tensor_batch:
            return batch
        if "old_log_probs" not in batch.batch or "ref_log_prob" not in batch.batch:
            return batch

        response_mask = batch.batch["response_mask"]
        token_counts = response_mask.sum(dim=-1).clamp_min(1.0)
        teacher_scores = (
            ((batch.batch["ref_log_prob"] - batch.batch["old_log_probs"]) * response_mask).sum(dim=-1)
            / token_counts
        ).detach().cpu().tolist()
        rubric_scores = [
            self._finite_gpt_score_value(score)
            for score in batch.non_tensor_batch["gpt_rollout_score_100"].tolist()
        ]
        padding_flags = batch.non_tensor_batch.get("g_opd_padding_sample", None)
        if padding_flags is None:
            non_padding_mask = np.ones(len(batch), dtype=bool)
        else:
            non_padding_mask = ~np.array([bool(flag) for flag in padding_flags.tolist()], dtype=bool)
        sample_kinds = batch.non_tensor_batch.get("g_opd_sample_kind", None)
        if sample_kinds is None:
            orig_mask = np.ones(len(batch), dtype=bool)
        else:
            orig_mask = np.array([str(kind or "orig") == "orig" for kind in sample_kinds.tolist()], dtype=bool)
        rank_eligible_mask = non_padding_mask & orig_mask
        metrics["g_opd_rank_gap_drop/rank_eligible_orig_count"] = int(rank_eligible_mask.sum())
        metrics["g_opd_rank_gap_drop/rank_excluded_non_orig_count"] = int((non_padding_mask & ~orig_mask).sum())
        teacher_ranks = self._rank_descending(teacher_scores, eligible_mask=rank_eligible_mask)
        rubric_ranks = self._rank_descending(rubric_scores, eligible_mask=rank_eligible_mask)
        valid_mask = (teacher_ranks >= 0) & (rubric_ranks >= 0) & non_padding_mask
        valid_count = int(valid_mask.sum())
        if valid_count <= 0:
            return batch

        rank_gap = np.zeros(len(batch), dtype=np.float32)
        denominator = max(valid_count - 1, 1)
        rank_gap[valid_mask] = np.abs(teacher_ranks[valid_mask] - rubric_ranks[valid_mask]).astype(np.float32) / denominator
        batch.non_tensor_batch["g_opd_teacher_rank"] = teacher_ranks.astype(object)
        batch.non_tensor_batch["g_opd_rubric_rank"] = rubric_ranks.astype(object)
        batch.non_tensor_batch["g_opd_teacher_rubric_rank_gap"] = rank_gap.astype(object)

        drop_enable = _is_truthy(scorer_config.get("rank_gap_drop_enable", False))
        drop_threshold = float(scorer_config.get("rank_gap_drop_threshold", 1.1))
        drop_scope = str(scorer_config.get("rank_gap_drop_scope", "all")).strip().lower()
        if drop_scope not in {"appended", "all"}:
            drop_scope = "all"
        keep_mask = np.ones(len(batch), dtype=bool)
        high_gap_mask = valid_mask & (rank_gap > drop_threshold)
        rubric_low_teacher_high_mask = high_gap_mask & (teacher_ranks < rubric_ranks)
        rubric_high_teacher_low_mask = high_gap_mask & (rubric_ranks < teacher_ranks)
        metrics["g_opd_rank_gap_drop/rubric_low_teacher_high_count"] = int(rubric_low_teacher_high_mask.sum())
        metrics["g_opd_rank_gap_drop/rubric_high_teacher_low_count"] = int(rubric_high_teacher_low_mask.sum())
        metrics["g_opd_rank_gap_drop/high_gap_count"] = int(high_gap_mask.sum())
        rank_gap_entries = []
        if drop_enable:
            drop_candidate_mask = np.ones(len(batch), dtype=bool)
            if drop_scope == "appended":
                if sample_kinds is None:
                    drop_candidate_mask = np.zeros(len(batch), dtype=bool)
                else:
                    drop_candidate_mask = np.array(
                        [str(kind or "orig") != "orig" for kind in sample_kinds.tolist()],
                        dtype=bool,
                    )
            drop_mask = high_gap_mask & drop_candidate_mask
            protected_orig_count = int((high_gap_mask & ~drop_candidate_mask).sum())
            keep_mask = ~drop_mask
            if not keep_mask.any():
                best_rubric_idx = int(np.argmin(np.where(rubric_ranks >= 0, rubric_ranks, len(batch) + 1)))
                keep_mask[best_rubric_idx] = True
                drop_mask[best_rubric_idx] = False
            dropped = int((~keep_mask).sum())
            metrics["g_opd_rank_gap_drop/dropped_count"] = dropped
            metrics["g_opd_rank_gap_drop/drop_threshold"] = drop_threshold
            metrics["g_opd_rank_gap_drop/scope_appended"] = 1 if drop_scope == "appended" else 0
            metrics["g_opd_rank_gap_drop/scope_all"] = 1 if drop_scope == "all" else 0
            metrics["g_opd_rank_gap_drop/protected_orig_count"] = protected_orig_count
            metrics["g_opd_rank_gap_drop/dropped_rubric_low_teacher_high_count"] = int(
                (drop_mask & rubric_low_teacher_high_mask).sum()
            )
            metrics["g_opd_rank_gap_drop/dropped_rubric_high_teacher_low_count"] = int(
                (drop_mask & rubric_high_teacher_low_mask).sum()
            )
            metrics["gpt_rollout_reroll/protected_orig_from_rank_drop_count"] = protected_orig_count
            rank_gap_entries = self._rank_gap_entries(
                batch,
                teacher_scores=teacher_scores,
                rubric_scores=rubric_scores,
                teacher_ranks=teacher_ranks,
                rubric_ranks=rubric_ranks,
                rank_gap=rank_gap,
                high_gap_mask=high_gap_mask,
                drop_mask=drop_mask,
                drop_threshold=drop_threshold,
            )
            self._maybe_dump_rank_gap_examples(
                scorer_config=scorer_config,
                entries=rank_gap_entries,
                timing_raw=timing_raw,
            )
            if dropped:
                batch = batch.select_idxs(keep_mask)
                teacher_scores = [value for keep, value in zip(keep_mask, teacher_scores) if keep]
                rubric_scores = [value for keep, value in zip(keep_mask, rubric_scores) if keep]
                padding_flags = batch.non_tensor_batch.get("g_opd_padding_sample", None)
                if padding_flags is None:
                    non_padding_mask = np.ones(len(batch), dtype=bool)
                else:
                    non_padding_mask = ~np.array([bool(flag) for flag in padding_flags.tolist()], dtype=bool)
                sample_kinds = batch.non_tensor_batch.get("g_opd_sample_kind", None)
                if sample_kinds is None:
                    orig_mask = np.ones(len(batch), dtype=bool)
                else:
                    orig_mask = np.array([str(kind or "orig") == "orig" for kind in sample_kinds.tolist()], dtype=bool)
                rank_eligible_mask = non_padding_mask & orig_mask
                teacher_ranks = self._rank_descending(teacher_scores, eligible_mask=rank_eligible_mask)
                rubric_ranks = self._rank_descending(rubric_scores, eligible_mask=rank_eligible_mask)
                valid_mask = (teacher_ranks >= 0) & (rubric_ranks >= 0) & non_padding_mask
                valid_count = int(valid_mask.sum())
                rank_gap = np.zeros(len(batch), dtype=np.float32)
                denominator = max(valid_count - 1, 1)
                rank_gap[valid_mask] = (
                    np.abs(teacher_ranks[valid_mask] - rubric_ranks[valid_mask]).astype(np.float32) / denominator
                )
                batch.non_tensor_batch["g_opd_teacher_rank"] = teacher_ranks.astype(object)
                batch.non_tensor_batch["g_opd_rubric_rank"] = rubric_ranks.astype(object)
                batch.non_tensor_batch["g_opd_teacher_rubric_rank_gap"] = rank_gap.astype(object)
        elif int(high_gap_mask.sum()) > 0:
            drop_mask = np.zeros(len(batch), dtype=bool)
            rank_gap_entries = self._rank_gap_entries(
                batch,
                teacher_scores=teacher_scores,
                rubric_scores=rubric_scores,
                teacher_ranks=teacher_ranks,
                rubric_ranks=rubric_ranks,
                rank_gap=rank_gap,
                high_gap_mask=high_gap_mask,
                drop_mask=drop_mask,
                drop_threshold=drop_threshold,
            )
            self._maybe_dump_rank_gap_examples(
                scorer_config=scorer_config,
                entries=rank_gap_entries,
                timing_raw=timing_raw,
            )

        shift_enable = _is_truthy(scorer_config.get("rubric_adv_shift_enable", False))
        shift = np.zeros(len(batch), dtype=np.float32)
        if shift_enable and valid_count > 0:
            mode = str(scorer_config.get("rubric_adv_shift_mode", "rank_residual")).strip().lower()
            if mode not in {"rank_residual", "rubric_rank", "history_zscore"}:
                mode = "rank_residual"
            coef = float(scorer_config.get("rubric_adv_shift_coef", 0.10))
            clip = abs(float(scorer_config.get("rubric_adv_shift_clip", 0.20)))
            denominator = max(valid_count - 1, 1)
            raw_shift = np.zeros(len(batch), dtype=np.float32)
            if mode == "rubric_rank":
                centered = np.zeros(len(batch), dtype=np.float32)
                centered[valid_mask] = 0.5 - (rubric_ranks[valid_mask].astype(np.float32) / denominator)
                raw_shift = 2.0 * coef * centered
            elif mode == "history_zscore":
                raw_shift = self._history_zscore_shift(
                    scorer_config=scorer_config,
                    rubric_scores=rubric_scores,
                    valid_mask=valid_mask,
                    non_padding_mask=non_padding_mask,
                    batch=batch,
                    coef=coef,
                    clip=clip,
                    metrics=metrics,
                )
            else:
                # Correct teacher preference only when rubric ranks the sample differently.
                raw_shift[valid_mask] = (
                    coef
                    * (teacher_ranks[valid_mask].astype(np.float32) - rubric_ranks[valid_mask].astype(np.float32))
                    / denominator
                )
            shift = np.clip(raw_shift, -clip, clip).astype(np.float32)
            metrics["g_opd_rubric_adv_shift/coef"] = coef
            metrics["g_opd_rubric_adv_shift/clip"] = clip
            metrics["g_opd_rubric_adv_shift/mode_rank_residual"] = 1 if mode == "rank_residual" else 0
            metrics["g_opd_rubric_adv_shift/mode_rubric_rank"] = 1 if mode == "rubric_rank" else 0
            metrics["g_opd_rubric_adv_shift/mode_history_zscore"] = 1 if mode == "history_zscore" else 0
            metrics["g_opd_rubric_adv_shift/mean"] = float(np.mean(shift[valid_mask]))
            metrics["g_opd_rubric_adv_shift/max_abs"] = float(np.max(np.abs(shift[valid_mask])))
            metrics["g_opd_rubric_adv_shift/positive_count"] = int((shift[valid_mask] > 0).sum())
            metrics["g_opd_rubric_adv_shift/negative_count"] = int((shift[valid_mask] < 0).sum())

        batch.non_tensor_batch["opd_rubric_adv_shift"] = shift.astype(np.float32)
        if len(rank_gap) > 0:
            metrics["g_opd_teacher_rubric_rank_gap/mean"] = float(np.mean(rank_gap))
            metrics["g_opd_teacher_rubric_rank_gap/max"] = float(np.max(rank_gap))
        return batch

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
    ) -> DataProto:
        scorer_config = self.config.trainer.get("gpt_rollout_score", None)
        if not scorer_config or not _is_truthy(scorer_config.get("enable", False)) or initial_score_result is None:
            self._ensure_g_opd_sample_metadata(batch, kind="orig", loss_weight=1.0)
            return batch

        threshold_100 = float(scorer_config.get("min_score_100", 50.0))
        configured_max_attempts = int(scorer_config.get("max_rerollout_attempts", 1))
        max_attempts = min(max(configured_max_attempts, 0), 1)
        orig_loss_weight = float(scorer_config.get("orig_loss_weight", 1.0))
        reroll_sft_enable = _is_truthy(scorer_config.get("reroll_sft_enable", False))
        append_reroll_hint_to_opd = _is_truthy(scorer_config.get("reroll_sft_keep_hint_opd", not reroll_sft_enable))
        reroll_hint_loss_weight = float(scorer_config.get("reroll_hint_loss_weight", 0.5))
        append_require_improvement = _is_truthy(
            scorer_config.get("reroll_append_require_improvement", True)
        )
        append_min_gain = float(scorer_config.get("reroll_append_min_gain", 0.0))
        append_min_score = float(scorer_config.get("reroll_append_min_score", 45.0))
        self._ensure_g_opd_sample_metadata(batch, kind="orig", loss_weight=orig_loss_weight)
        reroll_counts = np.zeros(len(batch), dtype=object)
        batch.non_tensor_batch["gpt_rollout_reroll_count"] = reroll_counts
        raw_low_idxs = self._get_low_gpt_score_idxs(initial_score_result["scores_100"], threshold_100)
        initial_timeout_idxs = self._get_initial_gpt_timeout_idxs(initial_score_result)
        initial_timeout_idx_set = set(initial_timeout_idxs)
        low_idxs = [idx for idx in raw_low_idxs if idx not in initial_timeout_idx_set]
        metrics["gpt_rollout_reroll/initial_low_count"] = len(low_idxs)
        metrics["gpt_rollout_reroll/initial_low_or_error_count"] = len(raw_low_idxs)
        metrics["gpt_rollout_reroll/initial_timeout_passthrough_count"] = len(initial_timeout_idxs)
        metrics["gpt_rollout_reroll/threshold_100"] = threshold_100
        metrics["gpt_rollout_reroll/max_attempts"] = max_attempts
        metrics["gpt_rollout_reroll/append_require_improvement"] = 1 if append_require_improvement else 0
        metrics["gpt_rollout_reroll/append_min_gain"] = append_min_gain
        metrics["gpt_rollout_reroll/append_min_score"] = append_min_score
        self._debug_progress(
            f"gpt_reroll:start low_count={len(low_idxs)} timeout_passthrough={len(initial_timeout_idxs)} "
            f"threshold_100={threshold_100} max_attempts={max_attempts} "
            f"append_min_score={append_min_score} append_min_gain={append_min_gain} "
            f"append_require_improvement={append_require_improvement}"
        )
        metrics["gpt_rollout_reroll/configured_max_attempts"] = configured_max_attempts
        metrics["gpt_rollout_reroll/reroll_sft_enable"] = 1 if reroll_sft_enable else 0
        metrics["gpt_rollout_reroll/append_hint_to_opd"] = 1 if append_reroll_hint_to_opd else 0

        gen_batch_for_reroll = None
        appended_batches: list[DataProto] = []
        for attempt in range(max_attempts):
            if not low_idxs:
                break

            if gen_batch_for_reroll is None:
                gen_batch_for_reroll = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )
                prompt_ready_idxs, prompt_hint_fallback_count = self._append_gpt_feedback_to_reroll_prompts(
                    batch=batch,
                    gen_batch=gen_batch_for_reroll,
                    row_idxs=low_idxs,
                    initial_score_result=initial_score_result,
                    scorer_config=scorer_config,
                )
                metrics["gpt_rollout_reroll/feedback_prompt_count"] = len(prompt_ready_idxs)
                metrics["gpt_rollout_reroll/prompt_hint_fallback_count"] = prompt_hint_fallback_count
                skipped_no_suggestion_count = len(low_idxs) - len(prompt_ready_idxs)
                metrics["gpt_rollout_reroll/skipped_no_revision_suggestion_count"] = skipped_no_suggestion_count
                if skipped_no_suggestion_count:
                    self._debug_progress(
                        f"gpt_reroll:skip_no_revision_suggestion count={skipped_no_suggestion_count}"
                    )
                low_idxs = prompt_ready_idxs
                if not low_idxs:
                    break
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

            reroll_scorer_config = self._reroll_gpt_scorer_config(scorer_config)
            self._debug_progress(
                f"gpt_reroll_rescore:start attempt={attempt + 1}/{max_attempts} count={len(low_idxs)} "
                f"max_output_tokens={reroll_scorer_config.get('max_output_tokens', '?')}"
            )
            reroll_score_result = self._score_gpt_rollouts(
                batch=reroll_scoring_batch,
                scorer_config=reroll_scorer_config,
                timing_raw=timing_raw,
                timer_name=f"gpt_reroll_score_{attempt + 1}",
            )
            self._debug_progress(
                f"gpt_reroll_rescore:done attempt={attempt + 1}/{max_attempts} count={len(low_idxs)}"
            )

            append_output_positions = []
            append_batch_idxs = []
            invalid_batch_idxs = []
            rejected_not_improved_idxs = []
            rejected_below_min_score_idxs = []
            rejected_reasons: dict[int, str] = {}
            for output_position, idx in enumerate(low_idxs):
                reroll_value = self._finite_gpt_score_value(
                    reroll_score_result["scores_100"][output_position]
                )
                initial_value = self._finite_gpt_score_value(initial_score_result["scores_100"][idx])
                if reroll_value is None:
                    invalid_batch_idxs.append(idx)
                    rejected_reasons[int(idx)] = "reroll_score_invalid"
                    continue
                if reroll_value < append_min_score:
                    rejected_below_min_score_idxs.append(idx)
                    rejected_reasons[int(idx)] = "reroll_score_below_min"
                    continue
                gain = None if initial_value is None else reroll_value - initial_value
                if append_require_improvement and (
                    initial_value is None or gain is None or gain <= append_min_gain
                ):
                    rejected_not_improved_idxs.append(idx)
                    rejected_reasons[int(idx)] = "reroll_score_not_improved"
                    continue
                append_output_positions.append(output_position)
                append_batch_idxs.append(idx)

            self._maybe_record_gpt_case_study_reroll_attempts(
                initial_score_result=initial_score_result,
                reroll_prompt_batch=reroll_gen_batch,
                reroll_scoring_batch=reroll_scoring_batch,
                reroll_score_result=reroll_score_result,
                row_idxs=low_idxs,
                accepted_batch_idxs=append_batch_idxs,
                rejected_reasons=rejected_reasons,
                attempt=attempt,
            )

            if append_batch_idxs:
                appended_batch = reroll_scoring_batch.select_idxs(append_output_positions)
                appended_score_result = self._select_gpt_rollout_result(
                    reroll_score_result,
                    append_output_positions,
                )
                self._inherit_gpt_problem_labels(
                    target_result=appended_score_result,
                    source_result=initial_score_result,
                    source_row_idxs=append_batch_idxs,
                )
                self._set_gpt_rollout_result(
                    batch=appended_batch,
                    result=appended_score_result,
                    prefix="gpt_rollout",
                    threshold_100=threshold_100,
                )
                if append_reroll_hint_to_opd:
                    self._set_reroll_sample_metadata(
                        batch=appended_batch,
                        source_row_idxs=append_batch_idxs,
                        initial_score_result=initial_score_result,
                        reroll_score_result=appended_score_result,
                        loss_weight=reroll_hint_loss_weight,
                    )
                    appended_batches.append(appended_batch)
                sft_batch = self._build_reroll_sft_batch(
                    original_batch=batch,
                    reroll_batch=appended_batch,
                    source_row_idxs=append_batch_idxs,
                    initial_score_result=initial_score_result,
                    reroll_score_result=appended_score_result,
                    scorer_config=scorer_config,
                    threshold_100=threshold_100,
                )
                if sft_batch is not None:
                    appended_batches.append(sft_batch)
                    metrics[f"gpt_rollout_reroll/attempt_{attempt + 1}_sft_count"] = len(sft_batch)
                nohint_batch = None if reroll_sft_enable else self._build_reroll_nohint_batch(
                    original_batch=batch,
                    reroll_batch=appended_batch,
                    source_row_idxs=append_batch_idxs,
                    initial_score_result=initial_score_result,
                    reroll_score_result=appended_score_result,
                    scorer_config=scorer_config,
                    threshold_100=threshold_100,
                )
                if nohint_batch is not None:
                    appended_batches.append(nohint_batch)
                    metrics[f"gpt_rollout_reroll/attempt_{attempt + 1}_nohint_count"] = len(nohint_batch)

            for idx in append_batch_idxs:
                reroll_counts[idx] = int(reroll_counts[idx]) + 1

            reroll_valid_count = sum(score is not None for score in reroll_score_result["scores_100"])
            second_rollout_count = len(low_idxs)
            second_append_rate = len(append_batch_idxs) / second_rollout_count if second_rollout_count > 0 else 0.0
            rejected_count = (
                len(invalid_batch_idxs) + len(rejected_not_improved_idxs) + len(rejected_below_min_score_idxs)
            )
            metrics[f"gpt_rollout_reroll/attempt_{attempt + 1}_generated_count"] = second_rollout_count
            metrics[f"gpt_rollout_reroll/attempt_{attempt + 1}_rescore_valid_count"] = reroll_valid_count
            metrics[f"gpt_rollout_reroll/attempt_{attempt + 1}_appended_count"] = len(append_batch_idxs)
            metrics[f"gpt_rollout_reroll/attempt_{attempt + 1}_append_rate"] = second_append_rate
            metrics[f"gpt_rollout_reroll/attempt_{attempt + 1}_invalid_score_count"] = len(invalid_batch_idxs)
            metrics[f"gpt_rollout_reroll/attempt_{attempt + 1}_rejected_count"] = rejected_count
            metrics[f"gpt_rollout_reroll/attempt_{attempt + 1}_rejected_not_improved_count"] = len(
                rejected_not_improved_idxs
            )
            metrics[f"gpt_rollout_reroll/attempt_{attempt + 1}_rejected_below_min_score_count"] = len(
                rejected_below_min_score_idxs
            )
            metrics["gpt_rollout_reroll/second_append_rate"] = second_append_rate
            metrics["gpt_rollout_reroll/second_rejected_count"] = rejected_count
            metrics["gpt_rollout_reroll/second_rejected_not_improved_count"] = len(rejected_not_improved_idxs)
            metrics["gpt_rollout_reroll/second_rejected_below_min_score_count"] = len(
                rejected_below_min_score_idxs
            )
            self._debug_progress(
                f"gpt_reroll_select:done attempt={attempt + 1}/{max_attempts} "
                f"appended={len(append_batch_idxs)} "
                f"invalid_score={len(invalid_batch_idxs)} "
                f"rejected_not_improved={len(rejected_not_improved_idxs)} "
                f"rejected_below_min_score={len(rejected_below_min_score_idxs)}"
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
        sample_kind_counts = defaultdict(int)
        for appended_batch in appended_batches:
            kinds = appended_batch.non_tensor_batch.get("g_opd_sample_kind")
            if kinds is None:
                continue
            for kind in kinds.tolist():
                sample_kind_counts[str(kind)] += 1
        metrics["gpt_rollout_reroll/appended_count"] = int(sum(sample_kind_counts.values()))
        metrics["gpt_rollout_reroll/appended_hint_count"] = int(sample_kind_counts.get("reroll_hint", 0))
        metrics["gpt_rollout_reroll/appended_nohint_count"] = int(sample_kind_counts.get("reroll_nohint", 0))
        metrics["gpt_rollout_reroll/appended_sft_count"] = int(sample_kind_counts.get("reroll_sft", 0))
        metrics["gpt_rollout_reroll/orig_count"] = len(batch)
        metrics["gpt_rollout_reroll/rescore_skipped_count"] = 0
        self._debug_progress(
            f"gpt_reroll:done final_low_count={len(final_low_idxs)} "
            f"appended={metrics['gpt_rollout_reroll/appended_count']} "
            f"rescore_skipped_count={metrics['gpt_rollout_reroll/rescore_skipped_count']}"
        )
        if appended_batches:
            batch = self._concat_training_batches([batch] + appended_batches)
            metrics["gpt_rollout_reroll/final_train_batch_size"] = len(batch)
        return batch

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
            metric_dict["val-aux/num_turns/min"] = float(sample_turns.min())
            metric_dict["val-aux/num_turns/max"] = float(sample_turns.max())
            metric_dict["val-aux/num_turns/mean"] = float(sample_turns.mean())

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
            val_log_metrics, dropped_val_metrics = _sanitize_metrics_for_logging(val_metrics)
            if dropped_val_metrics:
                preview = ", ".join(f"{key}:{reason}" for key, reason in dropped_val_metrics[:8])
                self._debug_progress(
                    f"val_metrics_log:sanitized kept={len(val_log_metrics)} "
                    f"dropped={len(dropped_val_metrics)} dropped_preview={preview}"
                )
            logger.log(data=val_log_metrics, step=self.global_steps)
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
                    batch = self._maybe_reroll_low_gpt_rollouts(
                        batch=batch,
                        gen_batch=gen_batch,
                        initial_score_result=gpt_score_result,
                        metrics=metrics,
                        timing_raw=timing_raw,
                    )
                    batch.batch["response_mask"] = compute_response_mask(batch)
                    batch = self._pad_training_batch_to_update_divisor(
                        batch=batch,
                        metrics=metrics,
                        metric_prefix="g_opd_pre_balance_batch",
                        zero_training_signals=False,
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
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: 蟺_rollout, 蟺_胃)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: 蟺_rollout, 蟺_old, 蟺_胃)
                    #   Note: 蟺_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if self.collect_rubric_probe_data:
                        if bypass_recomputing_logprobs:
                            raise ValueError(
                                "rubric_probe_data cannot collect student hidden states when "
                                "algorithm.rollout_correction.bypass_mode=True"
                            )
                        batch.meta_info["rubric_probe_return_hidden"] = True
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

                    # Save before rank-gap filtering so all original, non-padding
                    # rows with valid GPT labels remain in the probe dataset.
                    self._save_rubric_probe_data(batch=batch, metrics=metrics, timing_raw=timing_raw)

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

                        batch = self._rank_gap_drop_and_rubric_shift(
                            batch=batch,
                            metrics=metrics,
                            timing_raw=timing_raw,
                        )
                        batch = self._pad_training_batch_to_update_divisor(
                            batch=batch,
                            metrics=metrics,
                            metric_prefix="g_opd_update_batch",
                            zero_training_signals=True,
                        )
                        if reward_extra_infos_dict:
                            reward_extra_infos_dict = {
                                key: batch.non_tensor_batch[key].tolist()
                                for key in reward_extra_infos_dict
                                if key in batch.non_tensor_batch
                            }

                        self._maybe_update_gpt_case_study_rule_scores(
                            batch=batch,
                            initial_score_result=gpt_score_result,
                            reward_extra_infos_dict=reward_extra_infos_dict,
                            timing_raw=timing_raw,
                        )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable 蟺_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving 蟺_胃 vs 蟺_rollout
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
                        self._zero_padding_rows(batch)
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
                log_metrics, dropped_log_metrics = _sanitize_metrics_for_logging(metrics)
                if dropped_log_metrics:
                    preview = ", ".join(f"{key}:{reason}" for key, reason in dropped_log_metrics[:8])
                    self._debug_progress(
                        f"metrics_log:sanitized kept={len(log_metrics)} dropped={len(dropped_log_metrics)} "
                        f"dropped_preview={preview}"
                    )
                else:
                    self._debug_progress(f"metrics_log:sanitized kept={len(log_metrics)} dropped=0")
                logger.log(data=log_metrics, step=self.global_steps)
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
