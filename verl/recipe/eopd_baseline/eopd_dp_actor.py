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
Single Process Actor
"""

import json
import logging
import os
from pathlib import Path

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None
        self._eopd_debug_update_count = 0

    @staticmethod
    def _cfg_get(config, key, default=None):
        if config is None:
            return default
        if hasattr(config, "get"):
            return config.get(key, default)
        return getattr(config, key, default)

    def _eopd_config(self):
        policy_loss_config = self._cfg_get(self.config, "policy_loss", None)
        return self._cfg_get(policy_loss_config, "eopd", None)

    def _eopd_enabled(self):
        eopd_config = self._eopd_config()
        policy_loss_config = self._cfg_get(self.config, "policy_loss", None)
        return bool(self._cfg_get(eopd_config, "enable", self._cfg_get(policy_loss_config, "eopd_enable", False)))

    def _eopd_debug_config(self):
        return self._cfg_get(self._eopd_config(), "debug", None)

    def _eopd_debug_enabled(self):
        return bool(self._cfg_get(self._eopd_debug_config(), "enable", False))

    def _validate_eopd_teacher_topk(self, data: DataProto) -> None:
        if not self._eopd_enabled():
            return

        required = ("teacher_topk_logps", "teacher_topk_indices", "teacher_entropy")
        missing = [key for key in required if key not in data.batch]
        if missing:
            raise RuntimeError(
                "EOPD is enabled but internal ref teacher tensors are missing: "
                f"{missing}. Expected recipe.eopd_baseline.eopd_fsdp_workers.compute_ref_log_prob "
                "to return teacher_topk_logps, teacher_topk_indices, and teacher_entropy."
            )

    def _compute_eopd_forward_kl_loss(
        self,
        student_topk_logps: torch.Tensor,
        teacher_topk_logps: torch.Tensor,
        teacher_entropy: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        eopd_config = self._eopd_config()
        entropy_threshold = float(self._cfg_get(eopd_config, "entropy_threshold", 0.8))
        teacher_topk_logps = teacher_topk_logps.to(student_topk_logps.device)
        teacher_entropy = teacher_entropy.to(student_topk_logps.device)
        teacher_probs = teacher_topk_logps.exp()
        fkl = torch.sum(teacher_probs * (teacher_topk_logps - student_topk_logps), dim=-1)
        high_entropy_mask = response_mask.to(torch.bool) & (teacher_entropy > entropy_threshold)
        high_entropy_mask_float = high_entropy_mask.to(fkl.dtype)
        valid_mask_float = response_mask.to(fkl.dtype)
        fkl_loss = torch.sum(fkl * high_entropy_mask_float) / valid_mask_float.sum().clamp_min(1)

        valid_mask = response_mask.to(torch.bool)
        valid_count = valid_mask.sum().clamp_min(1)
        metrics = {
            "actor/eopd_fkl_loss": fkl_loss.detach().item(),
            "actor/eopd_high_entropy_ratio": (high_entropy_mask.sum() / valid_count).detach().item(),
            "actor/eopd_teacher_entropy": (
                (teacher_entropy * valid_mask.to(teacher_entropy.dtype)).sum() / valid_count
            )
            .detach()
            .item(),
            "actor/eopd_entropy_threshold": entropy_threshold,
        }
        return fkl_loss, metrics

    def _dump_eopd_debug(
        self,
        *,
        step: int,
        ppo_epoch: int,
        mini_batch_index: int,
        micro_batch_index: int,
        model_inputs: dict,
        old_log_prob: torch.Tensor,
        student_log_prob: torch.Tensor,
        student_topk_logps: torch.Tensor,
        pg_loss: torch.Tensor,
        policy_loss_before_eopd: torch.Tensor,
        eopd_fkl_loss: torch.Tensor,
        eopd_alpha: float,
        eopd_metrics: dict,
    ) -> None:
        debug_config = self._eopd_debug_config()
        if not self._eopd_debug_enabled():
            return

        rank = torch.distributed.get_rank()
        if bool(self._cfg_get(debug_config, "rank0_only", True)) and rank != 0:
            return

        debug_dir = self._cfg_get(debug_config, "dir", None)
        if not debug_dir:
            raise ValueError("EOPD debug is enabled but actor.policy_loss.eopd.debug.dir is empty.")

        max_samples = max(1, int(self._cfg_get(debug_config, "max_samples", 1)))
        max_tokens = max(1, int(self._cfg_get(debug_config, "max_tokens_per_sample", 16)))
        entropy_threshold = float(self._cfg_get(self._eopd_config(), "entropy_threshold", 0.8))

        response_mask = model_inputs["response_mask"].detach().to(torch.bool)
        responses = model_inputs["responses"].detach()
        teacher_entropy = model_inputs["teacher_entropy"].detach()
        teacher_topk_logps = model_inputs["teacher_topk_logps"].detach()
        teacher_topk_indices = model_inputs["teacher_topk_indices"].detach()
        ref_log_prob = model_inputs["ref_log_prob"].detach()
        old_log_prob = old_log_prob.detach()
        student_log_prob = student_log_prob.detach()
        student_topk_logps = student_topk_logps.detach()
        rollout_is_weights = model_inputs.get("rollout_is_weights", None)
        if rollout_is_weights is not None:
            rollout_is_weights = rollout_is_weights.detach()

        teacher_probs = teacher_topk_logps.exp()
        fkl_per_token = torch.sum(
            teacher_probs * (teacher_topk_logps - student_topk_logps),
            dim=-1,
        )
        high_entropy_mask = response_mask & (teacher_entropy > entropy_threshold)
        valid_count = int(response_mask.sum().item())
        high_entropy_count = int(high_entropy_mask.sum().item())

        sample_records = []
        sample_count = min(max_samples, response_mask.size(0))
        for sample_index in range(sample_count):
            valid_positions = torch.nonzero(response_mask[sample_index], as_tuple=False).flatten()
            high_positions = torch.nonzero(high_entropy_mask[sample_index], as_tuple=False).flatten()
            low_positions = torch.nonzero(
                response_mask[sample_index] & ~high_entropy_mask[sample_index],
                as_tuple=False,
            ).flatten()

            high_quota = min(high_positions.numel(), (max_tokens + 1) // 2)
            selected_positions = high_positions[:high_quota]
            remaining = max_tokens - selected_positions.numel()
            if remaining > 0:
                selected_positions = torch.cat((selected_positions, low_positions[:remaining]))
            remaining = max_tokens - selected_positions.numel()
            if remaining > 0:
                selected_set = set(selected_positions.cpu().tolist())
                fallback = [pos for pos in valid_positions.cpu().tolist() if pos not in selected_set][:remaining]
                if fallback:
                    selected_positions = torch.cat(
                        (
                            selected_positions,
                            torch.tensor(fallback, device=selected_positions.device, dtype=selected_positions.dtype),
                        )
                    )
            selected_positions = selected_positions.sort().values

            token_records = []
            for position in selected_positions.cpu().tolist():
                teacher_topk_probs = teacher_probs[sample_index, position]
                student_topk_probs = student_topk_logps[sample_index, position].exp()
                reverse_kl = old_log_prob[sample_index, position] - ref_log_prob[sample_index, position]
                is_high_entropy = bool(high_entropy_mask[sample_index, position].item())
                token_record = {
                    "position": int(position),
                    "token_id": int(responses[sample_index, position].item()),
                    "teacher_entropy": float(teacher_entropy[sample_index, position].float().item()),
                    "high_entropy": is_high_entropy,
                    "old_log_prob": float(old_log_prob[sample_index, position].float().item()),
                    "student_log_prob": float(student_log_prob[sample_index, position].float().item()),
                    "teacher_sampled_log_prob": float(ref_log_prob[sample_index, position].float().item()),
                    "reverse_kl": float(reverse_kl.float().item()),
                    "reverse_kl_advantage": float((-reverse_kl).float().item()),
                    "fkl": float(fkl_per_token[sample_index, position].float().item()),
                    "gated_fkl": float(
                        (fkl_per_token[sample_index, position] if is_high_entropy else 0.0)
                    ),
                    "teacher_topk_token_ids": teacher_topk_indices[sample_index, position].cpu().tolist(),
                    "teacher_topk_probs": teacher_topk_probs.float().cpu().tolist(),
                    "student_probs_at_teacher_topk": student_topk_probs.float().cpu().tolist(),
                    "teacher_topk_prob_sum": float(teacher_topk_probs.sum().float().item()),
                    "student_prob_mass_on_teacher_topk": float(student_topk_probs.sum().float().item()),
                }
                if rollout_is_weights is not None:
                    token_record["rollout_is_weight"] = float(
                        rollout_is_weights[sample_index, position].float().item()
                    )
                token_records.append(token_record)

            sample_records.append(
                {
                    "sample_index": sample_index,
                    "valid_tokens": int(response_mask[sample_index].sum().item()),
                    "high_entropy_tokens": int(high_entropy_mask[sample_index].sum().item()),
                    "response_token_ids": responses[sample_index, valid_positions].cpu().tolist(),
                    "tokens": token_records,
                }
            )

        valid_entropies = teacher_entropy[response_mask].float()
        record = {
            "step": step,
            "rank": rank,
            "scope": "first_micro_batch",
            "ppo_epoch": ppo_epoch,
            "mini_batch_index": mini_batch_index,
            "micro_batch_index": micro_batch_index,
            "entropy_threshold": entropy_threshold,
            "alpha": eopd_alpha,
            "topk": int(teacher_topk_indices.size(-1)),
            "valid_tokens": valid_count,
            "high_entropy_tokens": high_entropy_count,
            "high_entropy_ratio": high_entropy_count / max(valid_count, 1),
            "teacher_entropy_mean": float(valid_entropies.mean().item()) if valid_count else 0.0,
            "teacher_entropy_min": float(valid_entropies.min().item()) if valid_count else 0.0,
            "teacher_entropy_max": float(valid_entropies.max().item()) if valid_count else 0.0,
            "pg_loss": float(pg_loss.detach().float().item()),
            "policy_loss_before_eopd": float(policy_loss_before_eopd.detach().float().item()),
            "eopd_fkl_loss": float(eopd_fkl_loss.detach().float().item()),
            "eopd_weighted_fkl_loss": float((eopd_alpha * eopd_fkl_loss).detach().float().item()),
            "policy_loss_after_eopd": float(
                (policy_loss_before_eopd + eopd_alpha * eopd_fkl_loss).detach().float().item()
            ),
            "metrics": eopd_metrics,
            "samples": sample_records,
        }

        output_dir = Path(str(debug_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"step_{step:06d}_rank_{rank:03d}.jsonl"
        with output_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if bool(self._cfg_get(debug_config, "print_to_console", True)):
            print(
                "[eopd_debug] "
                f"step={step} rank={rank} valid={valid_count} high_entropy={high_entropy_count} "
                f"ratio={record['high_entropy_ratio']:.4f} pg_loss={record['pg_loss']:.6f} "
                f"fkl_loss={record['eopd_fkl_loss']:.6f} output={output_path}",
                flush=True,
            )

    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        calculate_entropy=False,
        teacher_topk_indices=None,
        topk=None,
        renormalize_topk=True,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            # reset input_ids, attention_mask, position_ids to ref model inputs if ref model input_ids is different from actor input_ids
            if "ref_input_ids" in micro_batch.keys():
                input_ids = micro_batch["ref_input_ids"]
                attention_mask = micro_batch["ref_attention_mask"]
                position_ids = micro_batch["ref_position_ids"]
                batch_size, seqlen = input_ids.shape

            entropy = None
            teacher_topk_logps = None
            own_topk_logps = None
            own_topk_indices = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
                teacher_topk_indices_rmpad = None
                if teacher_topk_indices is not None:
                    if self.use_fused_kernels:
                        raise ValueError("EOPD forward KL requires logits, but use_fused_kernels=True.")
                    if self.use_ulysses_sp:
                        raise NotImplementedError("EOPD top-k forward KL does not support Ulysses SP yet.")
                    full_teacher_topk_indices = torch.zeros(
                        batch_size,
                        seqlen,
                        teacher_topk_indices.size(-1),
                        dtype=torch.long,
                        device=input_ids.device,
                    )
                    full_teacher_topk_indices[:, -response_length - 1 : -1, :] = teacher_topk_indices.to(
                        input_ids.device, dtype=torch.long
                    )
                    teacher_topk_indices_rmpad = index_first_axis(
                        rearrange(full_teacher_topk_indices, "b s k -> (b s) k"), indices
                    )

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                    own_topk_logps_rmpad = None
                    own_topk_indices_rmpad = None

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)
                    own_topk_logps_rmpad = None
                    own_topk_indices_rmpad = None
                    if teacher_topk_indices_rmpad is not None:
                        teacher_topk_logps_rmpad = torch.gather(
                            logits_rmpad,
                            dim=-1,
                            index=teacher_topk_indices_rmpad.to(logits_rmpad.device),
                        )
                        teacher_topk_logps_rmpad = teacher_topk_logps_rmpad - torch.logsumexp(
                            logits_rmpad, dim=-1, keepdim=True
                        )
                    if topk is not None:
                        actual_topk = min(int(topk), logits_rmpad.size(-1))
                        own_topk_logits_rmpad, own_topk_indices_rmpad = torch.topk(
                            logits_rmpad,
                            k=actual_topk,
                            dim=-1,
                        )
                        if renormalize_topk:
                            own_topk_logps_rmpad = torch.nn.functional.log_softmax(
                                own_topk_logits_rmpad.float(),
                                dim=-1,
                            )
                        else:
                            own_topk_logps_rmpad = own_topk_logits_rmpad.float() - torch.logsumexp(
                                logits_rmpad.float(),
                                dim=-1,
                                keepdim=True,
                            )

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy or topk is not None:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if own_topk_logps_rmpad is not None:
                        own_topk_logps_rmpad = gather_outputs_and_unpad(
                            own_topk_logps_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                        own_topk_indices_rmpad = gather_outputs_and_unpad(
                            own_topk_indices_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                if teacher_topk_indices_rmpad is not None:
                    full_teacher_topk_logps = pad_input(
                        hidden_states=teacher_topk_logps_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if own_topk_logps_rmpad is not None:
                    full_own_topk_logps = pad_input(
                        hidden_states=own_topk_logps_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    full_own_topk_indices = pad_input(
                        hidden_states=own_topk_indices_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if teacher_topk_indices_rmpad is not None:
                    teacher_topk_logps = full_teacher_topk_logps[:, -response_length - 1 : -1, :]
                if own_topk_logps_rmpad is not None:
                    own_topk_logps = full_own_topk_logps[:, -response_length - 1 : -1, :]
                    own_topk_indices = full_own_topk_indices[:, -response_length - 1 : -1, :]

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    if topk is not None:
                        raise ValueError("EOPD teacher top-k requires logits, but use_fused_kernels=True.")
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    if teacher_topk_indices is not None:
                        teacher_topk_logps = torch.gather(
                            logits,
                            dim=-1,
                            index=teacher_topk_indices.to(logits.device, dtype=torch.long),
                        )
                        teacher_topk_logps = teacher_topk_logps - torch.logsumexp(logits, dim=-1, keepdim=True)
                    if topk is not None:
                        actual_topk = min(int(topk), logits.size(-1))
                        own_topk_logits, own_topk_indices = torch.topk(logits, k=actual_topk, dim=-1)
                        if renormalize_topk:
                            own_topk_logps = torch.nn.functional.log_softmax(own_topk_logits.float(), dim=-1)
                        else:
                            own_topk_logps = own_topk_logits.float() - torch.logsumexp(
                                logits.float(),
                                dim=-1,
                                keepdim=True,
                            )
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs, teacher_topk_logps, own_topk_logps, own_topk_indices

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(
        self,
        data: DataProto,
        calculate_entropy=False,
        return_topk=False,
        topk=None,
        renormalize_topk=True,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            tuple: ``(log_probs, entropys)`` by default, or
            ``(log_probs, entropys, topk_logps, topk_indices)`` when
            ``return_topk=True``.
        """
        # set to eval
        self.actor_module.eval()
        if return_topk and topk is None:
            raise ValueError("return_topk=True requires topk to be set.")
        if topk is not None and self.use_fused_kernels:
            raise ValueError("EOPD teacher top-k requires logits, but use_fused_kernels=True.")

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        has_ref_input_ids = "ref_input_ids" in data.batch.keys() # handle when ref input_ids is different from actor input_ids
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if has_ref_input_ids:
            select_keys.extend(["ref_input_ids", "ref_attention_mask", "ref_position_ids"])
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        topk_logps_lst = []
        topk_indices_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, _, topk_logps, topk_indices = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    topk=topk if return_topk else None,
                    renormalize_topk=renormalize_topk,
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)
            if return_topk:
                if topk_logps is None or topk_indices is None:
                    raise RuntimeError("EOPD ref top-k was requested but _forward_micro_batch returned None.")
                topk_logps_lst.append(topk_logps)
                topk_indices_lst.append(topk_indices)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        topk_logps = None
        topk_indices = None
        if return_topk:
            topk_logps = torch.concat(topk_logps_lst, dim=0)
            topk_indices = torch.concat(topk_indices_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if return_topk:
                topk_logps = restore_dynamic_batch(topk_logps, batch_idx_list)
                topk_indices = restore_dynamic_batch(topk_indices, batch_idx_list)

        if return_topk:
            return log_probs, entropys, topk_logps, topk_indices
        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()
        self._validate_eopd_teacher_topk(data)
        self._eopd_debug_update_count += 1
        eopd_debug_step = int(data.meta_info.get("global_steps", self._eopd_debug_update_count))
        eopd_debug_dumped = False

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")
         # Include base model log probs for corrected reward computation
        # These are computed when actor_rollout_ref.model.base_model_path and
        # actor_rollout_ref.ref.model.base_model_path are both specified
        if "base_log_prob" in data.batch.keys():
            select_keys.append("base_log_prob")
        if "base_ref_log_prob" in data.batch.keys():
            select_keys.append("base_ref_log_prob")
        # Include ref_log_prob for only_reverse_kl_advantages mode
        if self.config.policy_loss.only_reverse_kl_advantages and "ref_log_prob" in data.batch.keys():
            if "ref_log_prob" not in select_keys:
                select_keys.append("ref_log_prob")
        if self._eopd_enabled():
            select_keys.extend(["teacher_topk_logps", "teacher_topk_indices", "teacher_entropy"])
        
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        # Include opd_teacher for multi-teacher distillation
        if "opd_teacher" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("opd_teacher")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for ppo_epoch in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch_index, micro_batch in enumerate(micro_batches):
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    teacher_topk_indices = model_inputs.get("teacher_topk_indices", None)
                    entropy, log_prob, student_teacher_topk_logps, _, _ = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        teacher_topk_indices=teacher_topk_indices,
                    )

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # only use reverse KL for advantages if only_reverse_kl_advantages is True
                    if self.config.policy_loss.only_reverse_kl_advantages:
                        # Corrected reverse KL with base model normalization if base log probs are available
                        # Formula: (log_prob_actor - log_prob_ref) - (log_prob_actor_base - log_prob_ref_base)
                        # This removes the base model bias from both actor and ref models
                        if "base_log_prob" in model_inputs and "base_ref_log_prob" in model_inputs:
                            lambda_vals = self.config.policy_loss.lambda_vals

                            if self.config.policy_loss.multi_teacher_distill:
                                #### multi-teacher distillation ####
                                if "opd_teacher" in model_inputs:
                                    opd_teacher = model_inputs["opd_teacher"]
                                    batch_size = old_log_prob.shape[0]

                                    reverse_kl = torch.zeros_like(old_log_prob)

                                    for i in range(batch_size):
                                        teacher_type = opd_teacher[i] if isinstance(opd_teacher, (list, tuple)) else opd_teacher
                                        # TODO: need to improve the logic here
                                        if teacher_type == "math":
                                            if lambda_vals == 1.0:
                                                reverse_kl[i] = old_log_prob[i] - model_inputs["ref_log_prob"][i]
                                            else:
                                                reverse_kl[i] = old_log_prob[i] - model_inputs["base_log_prob"][i] - (model_inputs["ref_log_prob"][i] - model_inputs["base_log_prob"][i]) * lambda_vals
                                        elif teacher_type == "code":
                                            if lambda_vals == 1.0:
                                                reverse_kl[i] = old_log_prob[i] - model_inputs["base_ref_log_prob"][i]
                                            else:
                                                reverse_kl[i] = old_log_prob[i] - model_inputs["base_log_prob"][i] - (model_inputs["base_ref_log_prob"][i] - model_inputs["base_log_prob"][i]) * lambda_vals
                                        else:
                                            reverse_kl[i] = old_log_prob[i] - model_inputs["ref_log_prob"][i]
                                else:
                                    reverse_kl = old_log_prob - model_inputs["ref_log_prob"]
                                #### multi-teacher distillation ####
                            else:
                                #### single-teacher distillation ####
                                reverse_kl = old_log_prob - model_inputs["base_log_prob"]
                                reward_correction = model_inputs["ref_log_prob"] - model_inputs["base_log_prob"]

                                if lambda_vals == 1.0:
                                    reverse_kl = old_log_prob - model_inputs["ref_log_prob"]
                                else:
                                    reverse_kl = reverse_kl - reward_correction * lambda_vals
                                #### single-teacher distillation ####
                        else:
                            # Standard reverse KL: log(π_actor / π_ref) = log_prob_actor - log_prob_ref
                            reverse_kl = old_log_prob - model_inputs["ref_log_prob"]
                        advantages = (- (reverse_kl))
                   
                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using pure rollout correction mode (metrics already in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "rollout_correction" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self._eopd_enabled():
                        if student_teacher_topk_logps is None:
                            raise RuntimeError("EOPD is enabled but student top-k log-probabilities were not computed.")
                        eopd_fkl_loss, eopd_metrics = self._compute_eopd_forward_kl_loss(
                            student_topk_logps=student_teacher_topk_logps,
                            teacher_topk_logps=model_inputs["teacher_topk_logps"],
                            teacher_entropy=model_inputs["teacher_entropy"],
                            response_mask=response_mask,
                        )
                        eopd_config = self._eopd_config()
                        eopd_alpha = float(self._cfg_get(eopd_config, "alpha", 1.0))
                        policy_loss_before_eopd = policy_loss
                        policy_loss = policy_loss + eopd_alpha * eopd_fkl_loss
                        eopd_metrics["actor/eopd_alpha"] = eopd_alpha
                        micro_batch_metrics.update(eopd_metrics)
                        if self._eopd_debug_enabled() and not eopd_debug_dumped:
                            self._dump_eopd_debug(
                                step=eopd_debug_step,
                                ppo_epoch=ppo_epoch,
                                mini_batch_index=batch_idx,
                                micro_batch_index=micro_batch_index,
                                model_inputs=model_inputs,
                                old_log_prob=old_log_prob,
                                student_log_prob=log_prob,
                                student_topk_logps=student_teacher_topk_logps,
                                pg_loss=pg_loss,
                                policy_loss_before_eopd=policy_loss_before_eopd,
                                eopd_fkl_loss=eopd_fkl_loss,
                                eopd_alpha=eopd_alpha,
                                eopd_metrics=eopd_metrics,
                            )
                            eopd_debug_dumped = True

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
