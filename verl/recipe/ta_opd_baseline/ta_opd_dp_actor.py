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
import math
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
        self._ta_opd_debug_update_count = 0

    @staticmethod
    def _cfg_get(config, key, default=None):
        if config is None:
            return default
        if hasattr(config, "get"):
            return config.get(key, default)
        return getattr(config, key, default)

    def _ta_opd_config(self):
        policy_loss_config = self._cfg_get(self.config, "policy_loss", None)
        return self._cfg_get(policy_loss_config, "ta_opd", None)

    def _ta_opd_enabled(self):
        ta_opd_config = self._ta_opd_config()
        policy_loss_config = self._cfg_get(self.config, "policy_loss", None)
        return bool(self._cfg_get(ta_opd_config, "enable", self._cfg_get(policy_loss_config, "ta_opd_enable", False)))

    def _ta_opd_debug_config(self):
        return self._cfg_get(self._ta_opd_config(), "debug", None)

    def _ta_opd_debug_enabled(self):
        return bool(self._cfg_get(self._ta_opd_debug_config(), "enable", False))

    def _ta_opd_method(self) -> str:
        return str(self._cfg_get(self._ta_opd_config(), "method", "teachability")).lower()

    def _ta_opd_requires_entropy(self, method: str | None = None) -> bool:
        method = (method or self._ta_opd_method()).lower()
        return method in {
            "entropy",
            "tip",
            "teachability_entropy",
            "h_teach",
            "h+teach",
            "ca_softor",
            "teachability_entropy_split",
            "split_budget_ca",
        }

    def _ta_opd_requires_topk(self, method: str | None = None) -> bool:
        method = (method or self._ta_opd_method()).lower()
        return method not in {"none", "full", "random", "entropy"}

    def _ta_opd_exact_coverage_enabled(self) -> bool:
        return bool(self._cfg_get(self._ta_opd_config(), "exact_coverage", False))

    def _validate_ta_opd_teacher_topk(self, data: DataProto) -> None:
        if not self._ta_opd_enabled() or not self._ta_opd_requires_topk():
            return

        required = ["teacher_topk_logps", "teacher_topk_indices"]
        if self._ta_opd_exact_coverage_enabled():
            required.append("teacher_student_topk_logps")
        missing = [key for key in required if key not in data.batch]
        if missing:
            raise RuntimeError(
                "TA-OPD is enabled but internal ref teacher tensors are missing: "
                f"{missing}. Expected recipe.ta_opd_baseline.ta_opd_fsdp_workers.compute_ref_log_prob "
                "to return teacher_topk_logps, teacher_topk_indices, and exact student-support teacher scores."
            )

    def _robust_normalize(self, scores: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        ta_opd_config = self._ta_opd_config()
        q_low = float(self._cfg_get(ta_opd_config, "q_low", 0.05))
        q_high = float(self._cfg_get(ta_opd_config, "q_high", 0.95))
        eps = float(self._cfg_get(ta_opd_config, "eps", 1e-12))

        out = torch.zeros_like(scores, dtype=torch.float32)
        valid_mask = valid_mask.to(torch.bool)
        values = scores[valid_mask].float()
        if values.numel() == 0:
            return out

        lo = torch.quantile(values, min(max(q_low, 0.0), 1.0))
        hi = torch.quantile(values, min(max(q_high, 0.0), 1.0))
        out[valid_mask] = torch.clamp((scores[valid_mask].float() - lo) / (hi - lo + eps), 0.0, 1.0)
        return out

    def _compute_ta_opd_raw_scores(
        self,
        *,
        student_entropy: torch.Tensor | None,
        student_teacher_topk_logps: torch.Tensor | None,
        student_topk_logps: torch.Tensor | None,
        student_topk_indices: torch.Tensor | None,
        teacher_student_topk_logps: torch.Tensor | None,
        teacher_topk_logps: torch.Tensor,
        teacher_topk_indices: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        eps = float(self._cfg_get(self._ta_opd_config(), "eps", 1e-12))
        device = response_mask.device
        raw_h = torch.zeros_like(response_mask, dtype=torch.float32, device=device)
        raw_d = torch.zeros_like(response_mask, dtype=torch.float32, device=device)
        raw_c = torch.zeros_like(response_mask, dtype=torch.float32, device=device)

        if student_entropy is not None:
            raw_h = student_entropy.to(device=device, dtype=torch.float32)

        if (
            student_topk_logps is not None
            and student_topk_indices is not None
            and teacher_topk_logps is not None
            and teacher_topk_indices is not None
        ):
            teacher_topk_logps = teacher_topk_logps.to(device=device, dtype=torch.float32)
            teacher_topk_indices = teacher_topk_indices.to(device=device, dtype=torch.long)
            student_topk_logps = student_topk_logps.to(device=device, dtype=torch.float32)
            student_topk_indices = student_topk_indices.to(device=device, dtype=torch.long)
            if teacher_student_topk_logps is not None:
                teacher_student_topk_logps = teacher_student_topk_logps.to(device=device, dtype=torch.float32)

            teacher_probs = teacher_topk_logps.exp()
            student_probs = student_topk_logps.exp()

            overlap = student_topk_indices.unsqueeze(-1).eq(teacher_topk_indices.unsqueeze(-2))
            student_only = ~overlap.any(dim=-1)

            # Match the original TA-OPD online proxy: probabilities are only
            # available on each model's returned top-K support, so missing
            # union-side probabilities are clamped to eps.
            teacher_mass = teacher_probs.sum(dim=-1).clamp_min(eps)
            student_mass = student_probs.sum(dim=-1).clamp_min(eps)
            teacher_norm = (teacher_probs / teacher_mass.unsqueeze(-1)).clamp_min(eps)
            student_probs_on_teacher = (overlap.to(student_probs.dtype) * student_probs.unsqueeze(-1)).sum(dim=-2)
            student_norm_on_teacher = (student_probs_on_teacher / student_mass.unsqueeze(-1)).clamp_min(eps)
            raw_d = torch.sum(teacher_norm * (teacher_norm.log() - student_norm_on_teacher.log()), dim=-1)

            student_norm = (student_probs / student_mass.unsqueeze(-1)).clamp_min(eps)
            eps_like_student = torch.full_like(student_norm, eps)
            raw_d = raw_d + torch.sum(
                student_only.to(student_norm.dtype) * eps_like_student * (eps_like_student.log() - student_norm.log()),
                dim=-1,
            )

            # Eq.10: exact teacher mass on the student's top-K support.
            if teacher_student_topk_logps is not None:
                raw_c = teacher_student_topk_logps.exp().sum(dim=-1).clamp(0.0, 1.0)
            else:
                # Eq.17 lower bound when teacher scores on student top-K are unavailable.
                teacher_on_student = (overlap.to(teacher_probs.dtype) * teacher_probs.unsqueeze(-2)).sum(dim=-1)
                raw_c = teacher_on_student.sum(dim=-1).clamp(0.0, 1.0)

        return {"H": raw_h, "D": raw_d, "C": raw_c}

    def _compute_ta_opd_scores(
        self,
        raw_scores: dict[str, torch.Tensor],
        response_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        raw_h = raw_scores["H"].float()
        raw_d = raw_scores["D"].float()
        raw_c = raw_scores["C"].float()
        raw_dc = raw_d * raw_c
        h_norm = self._robust_normalize(raw_scores["H"], response_mask)
        d_norm = self._robust_normalize(raw_scores["D"], response_mask)
        c_norm = self._robust_normalize(raw_scores["C"], response_mask)
        dc_norm = self._robust_normalize(raw_dc, response_mask)
        teachability = d_norm * c_norm
        ca_softor = h_norm + dc_norm - h_norm * dc_norm
        return {
            "raw_H": raw_h,
            "raw_D": raw_d,
            "raw_C": raw_c,
            "raw_DC": raw_dc,
            "entropy": h_norm,
            "divergence": d_norm,
            "compatibility": c_norm,
            "dc_norm": dc_norm,
            "teachability": teachability,
            "dincompat": d_norm * (1.0 - c_norm),
            "tip": h_norm + d_norm - h_norm * d_norm,
            "teachability_entropy": ca_softor,
            "ca_softor": ca_softor,
        }

    def _top_flat_indices(
        self,
        score: torch.Tensor,
        candidates: torch.Tensor,
        n: int,
        exclude: set[int] | None = None,
    ) -> list[int]:
        if n <= 0 or candidates.numel() == 0:
            return []
        exclude = exclude or set()
        candidate_list = [int(i) for i in candidates.detach().cpu().tolist() if int(i) not in exclude]
        if not candidate_list:
            return []
        candidate_tensor = torch.tensor(candidate_list, device=score.device, dtype=torch.long)
        values = score.flatten()[candidate_tensor]
        k = min(n, candidate_tensor.numel())
        selected = candidate_tensor[torch.topk(values, k=k, largest=True).indices]
        return [int(i) for i in selected.detach().cpu().tolist()]

    def _select_ta_opd_mask(
        self,
        scores: dict[str, torch.Tensor],
        response_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        ta_opd_config = self._ta_opd_config()
        method = self._ta_opd_method()
        ratio = float(self._cfg_get(ta_opd_config, "ratio", 0.1))
        min_keep_per_sample = int(self._cfg_get(ta_opd_config, "min_keep_per_sample", 1) or 0)
        valid = response_mask.to(torch.bool)
        valid_count = int(valid.sum().item())
        selected = torch.zeros_like(valid)
        min_keep_added = torch.zeros_like(valid)
        budget_score = torch.zeros_like(response_mask, dtype=torch.float32)
        budget = valid_count

        if valid_count == 0:
            scores["budget_score"] = budget_score
            scores["min_keep_added"] = min_keep_added
            return selected.to(response_mask.dtype), {
                "actor/ta_opd_valid_tokens": 0,
                "actor/ta_opd_kept_tokens": 0,
                "actor/ta_opd_keep_ratio": 0.0,
                "actor/ta_opd_budget_ratio": ratio,
                "actor/ta_opd_budget_n": 0,
            }

        if method in {"none", "full"} or ratio >= 1.0:
            selected = valid.clone()
            budget_score = valid.to(torch.float32)
        elif ratio <= 0.0:
            raise ValueError(f"actor.policy_loss.ta_opd.ratio must be > 0, got {ratio}")
        else:
            budget = max(1, int(math.ceil(valid_count * ratio)))
            flat_valid = torch.nonzero(valid.flatten(), as_tuple=False).flatten()

            if method == "random":
                seed = int(self._cfg_get(ta_opd_config, "random_seed", 42))
                generator = torch.Generator(device=valid.device)
                generator.manual_seed(seed + self._ta_opd_debug_update_count)
                random_score = torch.rand(valid.numel(), device=valid.device, generator=generator).view_as(valid)
                budget_score = random_score.float()
                chosen = self._top_flat_indices(random_score, flat_valid, budget)
            elif method in {"teachability_entropy_split", "split_budget_ca"}:
                gamma = float(self._cfg_get(ta_opd_config, "budget_gamma", 0.5))
                n_entropy = max(0, min(budget, int(round(budget * gamma))))
                budget_score = torch.maximum(scores["entropy"], scores["ca_softor"]).float()
                chosen = self._top_flat_indices(scores["entropy"], flat_valid, n_entropy)
                chosen_set = set(chosen)
                chosen.extend(
                    self._top_flat_indices(scores["teachability"], flat_valid, budget - len(chosen), chosen_set)
                )
            else:
                score_key = {
                    "teachability": "teachability",
                    "dlearn_high": "teachability",
                    "teachability_high": "teachability",
                    "divergence": "divergence",
                    "entropy": "entropy",
                    "tip": "tip",
                    "compatibility": "compatibility",
                    "teachability_entropy": "teachability_entropy",
                    "h_teach": "teachability_entropy",
                    "h+teach": "teachability_entropy",
                    "ca_softor": "ca_softor",
                    "dincompat": "dincompat",
                    "dincompat_high": "dincompat",
                }.get(method)
                if score_key is None:
                    raise ValueError(f"Unknown actor.policy_loss.ta_opd.method={method!r}")
                budget_score = scores[score_key].float()
                chosen = self._top_flat_indices(scores[score_key], flat_valid, budget)

            selected.flatten()[torch.tensor(chosen, device=valid.device, dtype=torch.long)] = True

            if min_keep_per_sample > 0:
                fallback_score = scores.get(
                    "ca_softor",
                    scores.get("teachability_entropy", scores.get("teachability", scores.get("entropy"))),
                )
                for sample_index in range(valid.size(0)):
                    sample_valid = valid[sample_index]
                    if not bool(sample_valid.any()):
                        continue
                    kept = int((selected[sample_index] & sample_valid).sum().item())
                    if kept >= min_keep_per_sample:
                        continue
                    need = min(min_keep_per_sample - kept, int(sample_valid.sum().item()) - kept)
                    if need <= 0:
                        continue
                    local_candidates = torch.nonzero(sample_valid & ~selected[sample_index], as_tuple=False).flatten()
                    local_values = fallback_score[sample_index, local_candidates]
                    local_selected = local_candidates[torch.topk(local_values, k=need, largest=True).indices]
                    selected[sample_index, local_selected] = True
                    min_keep_added[sample_index, local_selected] = True

        selected = selected & valid
        min_keep_added = min_keep_added & selected
        scores["budget_score"] = budget_score
        scores["min_keep_added"] = min_keep_added
        kept_count = int(selected.sum().item())

        def masked_mean(name: str, mask: torch.Tensor) -> float:
            values = scores[name][mask]
            return float(values.float().mean().detach().item()) if values.numel() > 0 else 0.0

        metrics = {
            "actor/ta_opd_valid_tokens": valid_count,
            "actor/ta_opd_kept_tokens": kept_count,
            "actor/ta_opd_keep_ratio": kept_count / max(valid_count, 1),
            "actor/ta_opd_budget_ratio": ratio,
            "actor/ta_opd_budget_n": budget,
            "actor/ta_opd_mean_entropy": masked_mean("entropy", valid),
            "actor/ta_opd_mean_divergence": masked_mean("divergence", valid),
            "actor/ta_opd_mean_compatibility": masked_mean("compatibility", valid),
            "actor/ta_opd_mean_teachability": masked_mean("teachability", valid),
            "actor/ta_opd_selected_teachability": masked_mean("teachability", selected),
        }
        return selected.to(response_mask.dtype), metrics

    def _compute_ta_opd_mask_for_batch(
        self,
        mini_batch: DataProto,
        temperature: float,
    ) -> tuple[torch.Tensor, dict, dict[str, torch.Tensor]]:
        response_mask = mini_batch.batch["response_mask"]
        method = self._ta_opd_method()
        ta_opd_config = self._ta_opd_config()
        topk = int(self._cfg_get(ta_opd_config, "topk", 16))
        renormalize_topk = bool(self._cfg_get(ta_opd_config, "renormalize_topk", False))

        if method in {"none", "full"} or float(self._cfg_get(ta_opd_config, "ratio", 0.1)) >= 1.0:
            scores = self._compute_ta_opd_scores(
                {
                    "H": torch.zeros_like(response_mask, dtype=torch.float32),
                    "D": torch.zeros_like(response_mask, dtype=torch.float32),
                    "C": torch.zeros_like(response_mask, dtype=torch.float32),
                },
                response_mask,
            )
            mask, metrics = self._select_ta_opd_mask(scores, response_mask)
            return mask, metrics, scores

        if method == "random":
            scores = self._compute_ta_opd_scores(
                {
                    "H": torch.zeros_like(response_mask, dtype=torch.float32),
                    "D": torch.zeros_like(response_mask, dtype=torch.float32),
                    "C": torch.zeros_like(response_mask, dtype=torch.float32),
                },
                response_mask,
            )
            mask, metrics = self._select_ta_opd_mask(scores, response_mask)
            return mask, metrics, scores

        use_dynamic_bsz = self.config.use_dynamic_bsz
        if use_dynamic_bsz:
            max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
        else:
            micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

        raw_h_lst = []
        raw_d_lst = []
        raw_c_lst = []
        valid_lst = []
        student_topk_logps_lst = []
        student_topk_indices_lst = []
        was_training = self.actor_module.training
        self.actor_module.eval()
        try:
            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                teacher_topk_indices = model_inputs.get("teacher_topk_indices", None)
                with torch.no_grad():
                    entropy, _, student_teacher_topk_logps, student_topk_logps, student_topk_indices = (
                        self._forward_micro_batch(
                            model_inputs,
                            temperature=temperature,
                            calculate_entropy=self._ta_opd_requires_entropy(method),
                            teacher_topk_indices=None,
                            topk=topk if self._ta_opd_requires_topk(method) else None,
                            renormalize_topk=renormalize_topk,
                        )
                    )
                raw_scores = self._compute_ta_opd_raw_scores(
                    student_entropy=entropy,
                    student_teacher_topk_logps=student_teacher_topk_logps,
                    student_topk_logps=model_inputs.get("student_topk_logps", student_topk_logps),
                    student_topk_indices=model_inputs.get("student_topk_indices", student_topk_indices),
                    teacher_student_topk_logps=model_inputs.get("teacher_student_topk_logps", None),
                    teacher_topk_logps=model_inputs.get("teacher_topk_logps", None),
                    teacher_topk_indices=teacher_topk_indices,
                    response_mask=model_inputs["response_mask"],
                )
                raw_h_lst.append(raw_scores["H"].detach().cpu())
                raw_d_lst.append(raw_scores["D"].detach().cpu())
                raw_c_lst.append(raw_scores["C"].detach().cpu())
                valid_lst.append(model_inputs["response_mask"].detach().cpu())
                debug_student_topk_logps = model_inputs.get("student_topk_logps", student_topk_logps)
                debug_student_topk_indices = model_inputs.get("student_topk_indices", student_topk_indices)
                if debug_student_topk_logps is not None and debug_student_topk_indices is not None:
                    student_topk_logps_lst.append(debug_student_topk_logps.detach().cpu())
                    student_topk_indices_lst.append(debug_student_topk_indices.detach().cpu())
        finally:
            if was_training:
                self.actor_module.train()

        raw_h = torch.concat(raw_h_lst, dim=0)
        raw_d = torch.concat(raw_d_lst, dim=0)
        raw_c = torch.concat(raw_c_lst, dim=0)
        valid_mask = torch.concat(valid_lst, dim=0)
        student_topk_logps = torch.concat(student_topk_logps_lst, dim=0) if student_topk_logps_lst else None
        student_topk_indices = torch.concat(student_topk_indices_lst, dim=0) if student_topk_indices_lst else None
        if use_dynamic_bsz:
            raw_h = restore_dynamic_batch(raw_h, batch_idx_list)
            raw_d = restore_dynamic_batch(raw_d, batch_idx_list)
            raw_c = restore_dynamic_batch(raw_c, batch_idx_list)
            valid_mask = restore_dynamic_batch(valid_mask, batch_idx_list)
            if student_topk_logps is not None and student_topk_indices is not None:
                student_topk_logps = restore_dynamic_batch(student_topk_logps, batch_idx_list)
                student_topk_indices = restore_dynamic_batch(student_topk_indices, batch_idx_list)

        scores = self._compute_ta_opd_scores({"H": raw_h, "D": raw_d, "C": raw_c}, valid_mask)
        if student_topk_logps is not None and student_topk_indices is not None:
            scores["student_topk_logps"] = student_topk_logps
            scores["student_topk_indices"] = student_topk_indices
        mask, metrics = self._select_ta_opd_mask(scores, valid_mask)
        metrics["actor/ta_opd_topk"] = topk
        return mask.to(response_mask.dtype), metrics, scores

    def _dump_ta_opd_debug(
        self,
        *,
        step: int,
        ppo_epoch: int,
        mini_batch_index: int,
        mini_batch: DataProto,
        ta_opd_mask: torch.Tensor,
        scores: dict[str, torch.Tensor],
        metrics: dict,
    ) -> None:
        debug_config = self._ta_opd_debug_config()
        if not self._ta_opd_debug_enabled():
            return

        rank = torch.distributed.get_rank()
        if bool(self._cfg_get(debug_config, "rank0_only", True)) and rank != 0:
            return

        debug_dir = self._cfg_get(debug_config, "dir", None)
        if not debug_dir:
            raise ValueError("TA-OPD debug is enabled but actor.policy_loss.ta_opd.debug.dir is empty.")

        max_samples = max(1, int(self._cfg_get(debug_config, "max_samples", 1)))
        max_tokens = max(1, int(self._cfg_get(debug_config, "max_tokens_per_sample", 16)))

        response_mask = mini_batch.batch["response_mask"].detach().to(torch.bool).cpu()
        responses = mini_batch.batch["responses"].detach().cpu()
        ta_opd_mask = ta_opd_mask.detach().to(torch.bool).cpu()
        valid_count = int(response_mask.sum().item())
        kept_count = int(ta_opd_mask.sum().item())
        score_cpu = {key: value.detach().cpu() for key, value in scores.items()}
        batch_cpu = {
            key: value.detach().cpu()
            for key, value in mini_batch.batch.items()
            if torch.is_tensor(value)
        }

        def scalar_from(mapping: dict[str, torch.Tensor], key: str, sample_index: int, position: int):
            value = mapping.get(key)
            if value is None or value.dim() < 2:
                return None
            item = value[sample_index, position]
            if item.numel() != 1:
                return None
            if item.dtype == torch.bool:
                return bool(item.item())
            if item.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                return int(item.item())
            return float(item.float().item())

        def topk_from(score_key: str, batch_key: str, sample_index: int, position: int):
            value = score_cpu.get(score_key)
            if value is None:
                value = batch_cpu.get(batch_key)
            if value is None or value.dim() < 3:
                return []
            return value[sample_index, position].tolist()

        token_records = []
        sample_count = min(max_samples, response_mask.size(0))
        for sample_index in range(sample_count):
            valid_positions = torch.nonzero(response_mask[sample_index], as_tuple=False).flatten()
            kept_positions = torch.nonzero(ta_opd_mask[sample_index], as_tuple=False).flatten()
            shown_positions = []
            seen_positions = set()
            for position in kept_positions.cpu().tolist() + valid_positions.cpu().tolist():
                if int(position) in seen_positions:
                    continue
                seen_positions.add(int(position))
                shown_positions.append(int(position))
                if len(shown_positions) >= max_tokens:
                    break

            for position in shown_positions:
                old_log_prob = scalar_from(batch_cpu, "old_log_probs", sample_index, position)
                ref_log_prob = scalar_from(batch_cpu, "ref_log_prob", sample_index, position)
                reverse_kl = scalar_from(batch_cpu, "ta_opd_reverse_kl", sample_index, position)
                if reverse_kl is None and old_log_prob is not None and ref_log_prob is not None:
                    reverse_kl = old_log_prob - ref_log_prob
                record = {
                    "record_type": "token",
                    "global_step": step,
                    "rank": rank,
                    "ppo_epoch": ppo_epoch,
                    "mini_batch_index": mini_batch_index,
                    "method": self._ta_opd_method(),
                    "sample_index": int(sample_index),
                    "tok_pos": int(position),
                    "token_id": int(responses[sample_index, position].item()),
                    "response_mask": bool(response_mask[sample_index, position].item()),
                    "ta_opd_mask": bool(ta_opd_mask[sample_index, position].item()),
                    "selected": bool(ta_opd_mask[sample_index, position].item()),
                    "student_topk_indices": topk_from("student_topk_indices", "student_topk_indices", sample_index, position),
                    "student_topk_logps": topk_from("student_topk_logps", "student_topk_logps", sample_index, position),
                    "teacher_topk_indices": topk_from("teacher_topk_indices", "teacher_topk_indices", sample_index, position),
                    "teacher_topk_logps": topk_from("teacher_topk_logps", "teacher_topk_logps", sample_index, position),
                    "raw_H": scalar_from(score_cpu, "raw_H", sample_index, position),
                    "raw_D": scalar_from(score_cpu, "raw_D", sample_index, position),
                    "raw_C": scalar_from(score_cpu, "raw_C", sample_index, position),
                    "raw_DC": scalar_from(score_cpu, "raw_DC", sample_index, position),
                    "H_norm": scalar_from(score_cpu, "entropy", sample_index, position),
                    "D_norm": scalar_from(score_cpu, "divergence", sample_index, position),
                    "C_norm": scalar_from(score_cpu, "compatibility", sample_index, position),
                    "DC_norm": scalar_from(score_cpu, "dc_norm", sample_index, position),
                    "teachability": scalar_from(score_cpu, "teachability", sample_index, position),
                    "dincompat": scalar_from(score_cpu, "dincompat", sample_index, position),
                    "tip": scalar_from(score_cpu, "tip", sample_index, position),
                    "ca_softor": scalar_from(score_cpu, "ca_softor", sample_index, position),
                    "budget_score": scalar_from(score_cpu, "budget_score", sample_index, position),
                    "budget_ratio": float(self._cfg_get(self._ta_opd_config(), "ratio", 0.1)),
                    "budget_n": int(metrics.get("actor/ta_opd_budget_n", 0)),
                    "valid_count": valid_count,
                    "kept_count": kept_count,
                    "min_keep_added": bool(score_cpu.get("min_keep_added", torch.zeros_like(response_mask))[sample_index, position].item()),
                    "old_log_prob": old_log_prob,
                    "ref_log_prob": ref_log_prob,
                    "current_log_prob": scalar_from(batch_cpu, "ta_opd_current_log_prob", sample_index, position),
                    "reverse_kl": reverse_kl,
                    "advantage_after_reverse_kl": scalar_from(batch_cpu, "ta_opd_advantage_after_reverse_kl", sample_index, position),
                    "loss_mask_used": scalar_from(batch_cpu, "ta_opd_loss_mask_used", sample_index, position),
                }
                token_records.append(record)

        record = {
            "record_type": "summary",
            "step": step,
            "rank": rank,
            "scope": "mini_batch_selection",
            "ppo_epoch": ppo_epoch,
            "mini_batch_index": mini_batch_index,
            "method": self._ta_opd_method(),
            "valid_tokens": valid_count,
            "kept_tokens": kept_count,
            "keep_ratio": kept_count / max(valid_count, 1),
            "metrics": metrics,
            "debugged_token_rows": len(token_records),
        }

        output_dir = Path(str(debug_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / (
            f"token_bank_step_{step:06d}_rank_{rank:03d}_"
            f"epoch_{ppo_epoch:02d}_mb_{mini_batch_index:03d}.jsonl"
        )
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            for token_record in token_records:
                f.write(json.dumps(token_record, ensure_ascii=False) + "\n")

        if bool(self._cfg_get(debug_config, "print_to_console", True)):
            print(
                "[ta_opd_debug] "
                f"step={step} rank={rank} valid={valid_count} kept={kept_count} "
                f"ratio={record['keep_ratio']:.4f} method={record['method']} output={output_path}",
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
                        raise ValueError("TA-OPD top-k scoring requires logits, but use_fused_kernels=True.")
                    if self.use_ulysses_sp:
                        raise NotImplementedError("TA-OPD top-k scoring does not support Ulysses SP yet.")
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
                        raise ValueError("TA-OPD top-k scoring requires logits, but use_fused_kernels=True.")
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
        gather_token_indices_key=None,
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
            tuple: ``(log_probs, entropys)`` by default. Optional returns add
            gathered log-probs for ``gather_token_indices_key`` and/or
            ``(topk_logps, topk_indices)`` when ``return_topk=True``.
        """
        # set to eval
        self.actor_module.eval()
        if return_topk and topk is None:
            raise ValueError("return_topk=True requires topk to be set.")
        if topk is not None and self.use_fused_kernels:
            raise ValueError("TA-OPD top-k scoring requires logits, but use_fused_kernels=True.")

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        has_ref_input_ids = "ref_input_ids" in data.batch.keys() # handle when ref input_ids is different from actor input_ids
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if has_ref_input_ids:
            select_keys.extend(["ref_input_ids", "ref_attention_mask", "ref_position_ids"])
        if gather_token_indices_key is not None:
            select_keys.append(gather_token_indices_key)
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        gathered_logps_lst = []
        topk_logps_lst = []
        topk_indices_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            gather_token_indices = (
                model_inputs[gather_token_indices_key] if gather_token_indices_key is not None else None
            )
            with torch.no_grad():
                entropy, log_probs, gathered_logps, topk_logps, topk_indices = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    teacher_topk_indices=gather_token_indices,
                    topk=topk if return_topk else None,
                    renormalize_topk=renormalize_topk,
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)
            if gather_token_indices_key is not None:
                if gathered_logps is None:
                    raise RuntimeError("TA-OPD token-index scoring was requested but no gathered log-probs were returned.")
                gathered_logps_lst.append(gathered_logps)
            if return_topk:
                if topk_logps is None or topk_indices is None:
                    raise RuntimeError("TA-OPD ref top-k was requested but _forward_micro_batch returned None.")
                topk_logps_lst.append(topk_logps)
                topk_indices_lst.append(topk_indices)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        gathered_logps = None
        if gather_token_indices_key is not None:
            gathered_logps = torch.concat(gathered_logps_lst, dim=0)
        topk_logps = None
        topk_indices = None
        if return_topk:
            topk_logps = torch.concat(topk_logps_lst, dim=0)
            topk_indices = torch.concat(topk_indices_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if gather_token_indices_key is not None:
                gathered_logps = restore_dynamic_batch(gathered_logps, batch_idx_list)
            if return_topk:
                topk_logps = restore_dynamic_batch(topk_logps, batch_idx_list)
                topk_indices = restore_dynamic_batch(topk_indices, batch_idx_list)

        if return_topk and gather_token_indices_key is not None:
            return log_probs, entropys, gathered_logps, topk_logps, topk_indices
        if gather_token_indices_key is not None:
            return log_probs, entropys, gathered_logps
        if return_topk:
            return log_probs, entropys, topk_logps, topk_indices
        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()
        self._validate_ta_opd_teacher_topk(data)
        self._ta_opd_debug_update_count += 1
        ta_opd_debug_step = int(data.meta_info.get("global_steps", self._ta_opd_debug_update_count))
        ta_opd_debug_dumped = False

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
        if self._ta_opd_enabled():
            for key in (
                "teacher_topk_logps",
                "teacher_topk_indices",
                "teacher_entropy",
                "teacher_student_topk_logps",
                "student_topk_logps",
                "student_topk_indices",
            ):
                if key in data.batch.keys():
                    select_keys.append(key)
        
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
                ta_opd_metrics = {}
                if self._ta_opd_enabled():
                    ta_opd_mask, ta_opd_metrics, ta_opd_scores = self._compute_ta_opd_mask_for_batch(
                        mini_batch=mini_batch,
                        temperature=temperature,
                    )
                    mini_batch.batch["ta_opd_mask"] = ta_opd_mask
                    if self._ta_opd_debug_enabled():
                        for score_key, score_value in ta_opd_scores.items():
                            mini_batch.batch[f"ta_opd_debug_{score_key}"] = score_value.to(ta_opd_mask.device)
                    append_to_dict(metrics, ta_opd_metrics)

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
                    opd_response_mask = model_inputs.get("ta_opd_mask", response_mask).to(response_mask.dtype)
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
                    entropy, log_prob, _, _, _ = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
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
                    debug_reverse_kl = None
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
                        debug_reverse_kl = reverse_kl

                    if self._ta_opd_enabled() and self._ta_opd_debug_enabled() and not ta_opd_debug_dumped:
                        micro_batch.batch["ta_opd_current_log_prob"] = log_prob.detach()
                        if "ref_log_prob" in model_inputs:
                            micro_batch.batch["ta_opd_reverse_kl"] = (
                                debug_reverse_kl.detach()
                                if debug_reverse_kl is not None
                                else (old_log_prob - model_inputs["ref_log_prob"]).detach()
                            )
                        micro_batch.batch["ta_opd_advantage_after_reverse_kl"] = advantages.detach()
                        micro_batch.batch["ta_opd_loss_mask_used"] = opd_response_mask.detach()
                        debug_scores = {
                            key[len("ta_opd_debug_") :]: value
                            for key, value in micro_batch.batch.items()
                            if key.startswith("ta_opd_debug_")
                        }
                        self._dump_ta_opd_debug(
                            step=ta_opd_debug_step,
                            ppo_epoch=ppo_epoch,
                            mini_batch_index=batch_idx,
                            mini_batch=micro_batch,
                            ta_opd_mask=opd_response_mask,
                            scores=debug_scores,
                            metrics=ta_opd_metrics,
                        )
                        ta_opd_debug_dumped = True
                   
                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=opd_response_mask,
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
                        entropy_loss = agg_loss(
                            loss_mat=entropy,
                            loss_mask=opd_response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=opd_response_mask, loss_agg_mode=loss_agg_mode)

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
