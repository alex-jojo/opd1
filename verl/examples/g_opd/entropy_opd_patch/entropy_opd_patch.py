"""Runtime patch for the isolated Entropy OPD 20% experiment.

This module is intentionally loaded only by the entropy-opd launch script. It
keeps the normal OPD/G-OPD code path untouched unless the
ENTROPY_OPD_ENABLE environment variable is set.
"""

from __future__ import annotations

import os
from typing import Any


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _top_fraction_mask(entropys, response_mask, fraction: float):
    import torch

    valid_mask = response_mask.to(dtype=torch.bool)
    selected = torch.zeros_like(valid_mask)

    for row_idx in range(entropys.shape[0]):
        valid_idx = torch.nonzero(valid_mask[row_idx], as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0 or fraction <= 0:
            continue

        k = max(1, int(torch.ceil(torch.tensor(valid_idx.numel() * fraction)).item()))
        k = min(k, valid_idx.numel())
        valid_entropy = entropys[row_idx, valid_idx]
        top_local_idx = torch.topk(valid_entropy, k=k, largest=True, sorted=False).indices
        selected[row_idx, valid_idx[top_local_idx]] = True

    return selected


def _masked_mean(values, mask):
    mask = mask.to(dtype=values.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def _to_device_tensor(value: Any, *, device, dtype=None):
    import torch

    if isinstance(value, torch.Tensor):
        value = value.to(device=device)
        return value if dtype is None else value.to(dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def apply() -> None:
    if not _env_flag("ENTROPY_OPD_ENABLE"):
        return

    import torch

    from verl import DataProto
    from verl.trainer.ppo import core_algos
    from verl.utils.device import get_device_id
    from verl.utils.fsdp_utils import fsdp_version
    from verl.single_controller.ray.base import RayWorkerGroup
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    if getattr(ActorRolloutRefWorker.compute_ref_log_prob, "_entropy_opd_patched", False):
        return

    original_compute_ref_log_prob = ActorRolloutRefWorker.compute_ref_log_prob
    original_update_policy = DataParallelPPOActor.update_policy
    original_compute_policy_loss_vanilla = core_algos.compute_policy_loss_vanilla
    original_ray_worker_group_init = RayWorkerGroup.__init__

    def ray_worker_group_init_with_entropy_env(self, *args, **kwargs):
        worker_env = dict(kwargs.get("worker_env", {}) or {})
        worker_env.setdefault("ENTROPY_OPD_ENABLE", os.environ.get("ENTROPY_OPD_ENABLE", "1"))
        worker_env.setdefault("ENTROPY_OPD_FRACTION", os.environ.get("ENTROPY_OPD_FRACTION", "0.2"))
        worker_env.setdefault("ENTROPY_OPD_VERBOSE", os.environ.get("ENTROPY_OPD_VERBOSE", "1"))
        patch_dir = os.environ.get("ENTROPY_OPD_PATCH_DIR", "/workspace/opd1/verl/examples/g_opd/entropy_opd_patch")
        verl_dir = os.environ.get("ENTROPY_OPD_VERL_DIR", "/workspace/opd1/verl")
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        worker_env.setdefault("PYTHONPATH", f"{patch_dir}:{verl_dir}:{existing_pythonpath}")
        kwargs["worker_env"] = worker_env
        return original_ray_worker_group_init(self, *args, **kwargs)

    def compute_ref_log_prob_with_entropy(self, data: DataProto):
        if not _env_flag("ENTROPY_OPD_ENABLE"):
            return original_compute_ref_log_prob(self, data)

        if self._is_lora:
            data.meta_info["is_lora"] = True
            ref_data = self.compute_log_prob(data)
            tensors = {"ref_log_prob": ref_data.batch["old_log_probs"]}
            if "entropys" in ref_data.batch:
                tensors["ref_entropys"] = ref_data.batch["entropys"]
            return DataProto.from_dict(tensors=tensors)

        assert self._is_ref
        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz

        with self.ulysses_sharding_manager:
            data = data.to("cpu")
            ref_log_prob, ref_entropys = self.ref_policy.compute_log_prob(data=data, calculate_entropy=True)
            output = DataProto.from_dict(tensors={"ref_log_prob": ref_log_prob, "ref_entropys": ref_entropys})

        output = output.to("cpu")

        if self.world_size > 1:
            if fsdp_version(self.ref_policy.actor_module) == 1:
                self.ref_policy.actor_module._handle.reshard(True)
            elif fsdp_version(self.ref_policy.actor_module) == 2:
                self.ref_policy.actor_module.reshard()

        return output

    def update_policy_entropy_opd(self, data: DataProto):
        if not _env_flag("ENTROPY_OPD_ENABLE"):
            return original_update_policy(self, data)

        self.actor_module.train()

        temperature = data.meta_info["temperature"]
        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
            "ref_log_prob",
            "ref_entropys",
        ]
        optional_keys = [
            "rollout_is_weights",
            "rollout_log_probs",
            "base_log_prob",
            "base_ref_log_prob",
        ]
        select_keys.extend(key for key in optional_keys if key in data.batch.keys())

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        for key in ["opd_teacher", "opd_reroll_bonus_scale", "opd_reroll_bonus_mask"]:
            if key in data.non_tensor_batch.keys():
                non_tensor_select_keys.append(key)

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1
        entropy_fraction = _env_float("ENTROPY_OPD_FRACTION", 0.2)
        entropy_fraction = min(max(entropy_fraction, 0.0), 1.0)

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                if self.config.use_dynamic_bsz:
                    from verl.utils.seqlen_balancing import prepare_dynamic_batch

                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    calculate_entropy = entropy_coeff != 0
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    if not (hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs):
                        old_log_prob = log_prob.detach() if on_policy else model_inputs["old_log_probs"]

                    ref_entropys = model_inputs["ref_entropys"]
                    entropy_mask_bool = _top_fraction_mask(ref_entropys, response_mask, entropy_fraction)
                    entropy_mask = entropy_mask_bool.to(dtype=response_mask.dtype)

                    lambda_vals = self.config.policy_loss.get("lambda_vals", 1.0)
                    if "base_log_prob" in model_inputs and "base_ref_log_prob" in model_inputs:
                        reverse_kl = old_log_prob - model_inputs["base_log_prob"]
                        reward_correction = model_inputs["ref_log_prob"] - model_inputs["base_log_prob"]
                        reverse_kl = old_log_prob - model_inputs["ref_log_prob"] if lambda_vals == 1.0 else reverse_kl - reward_correction * lambda_vals
                    else:
                        reverse_kl = old_log_prob - model_inputs["ref_log_prob"]

                    advantages = -reverse_kl

                    if "opd_reroll_bonus_scale" in model_inputs:
                        with torch.no_grad():
                            opd_adv = advantages.detach()
                            reroll_bonus_scale = _to_device_tensor(
                                model_inputs["opd_reroll_bonus_scale"], device=opd_adv.device, dtype=opd_adv.dtype
                            )
                            reroll_bonus_scale = torch.clamp(reroll_bonus_scale, min=1.0, max=1.25)
                            while reroll_bonus_scale.dim() < opd_adv.dim():
                                reroll_bonus_scale = reroll_bonus_scale.unsqueeze(-1)
                            advantages = torch.where(opd_adv > 0, opd_adv * reroll_bonus_scale, opd_adv)
                    elif "opd_reroll_bonus_mask" in model_inputs:
                        with torch.no_grad():
                            opd_adv = advantages.detach()
                            reroll_bonus_mask = _to_device_tensor(
                                model_inputs["opd_reroll_bonus_mask"], device=opd_adv.device, dtype=torch.bool
                            )
                            while reroll_bonus_mask.dim() < opd_adv.dim():
                                reroll_bonus_mask = reroll_bonus_mask.unsqueeze(-1)
                            advantages = torch.where(reroll_bonus_mask & (opd_adv > 0), opd_adv * 1.25, opd_adv)

                    advantages = advantages * entropy_mask
                    effective_response_mask = response_mask * entropy_mask

                    policy_loss_fn = core_algos.get_policy_loss_fn(self.config.policy_loss.get("loss_mode", "vanilla"))
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=effective_response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=model_inputs.get("rollout_is_weights", None),
                    )
                    micro_batch_metrics.update(pg_metrics)
                    micro_batch_metrics["actor/entropy_opd_selected_frac"] = _masked_mean(
                        entropy_mask, response_mask
                    ).detach().item()
                    micro_batch_metrics["actor/ref_entropy_selected_mean"] = _masked_mean(
                        ref_entropys, effective_response_mask
                    ).detach().item()

                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if self.config.policy_loss.get("loss_mode", "vanilla") != "rollout_correction" and rollout_log_prob is not None:
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        micro_batch_metrics.update(
                            compute_rollout_corr_metrics_from_logprobs(
                                log_prob=log_prob,
                                rollout_log_prob=rollout_log_prob,
                                response_mask=response_mask,
                            )
                        )

                    if entropy_coeff != 0:
                        entropy_loss = core_algos.agg_loss(
                            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
                        )
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        kld = core_algos.kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = core_algos.agg_loss(
                            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
                        )
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    from verl.utils.py_functional import append_to_dict

                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                from verl.utils.py_functional import append_to_dict

                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics

    def compute_policy_loss_vanilla_entropy_safe(*args, **kwargs):
        response_mask = kwargs.get("response_mask")
        if response_mask is None and len(args) >= 4:
            response_mask = args[3]
        if _env_flag("ENTROPY_OPD_ENABLE") and response_mask is not None and response_mask.sum().item() == 0:
            log_prob = kwargs.get("log_prob") if "log_prob" in kwargs else args[1]
            zero = log_prob.sum() * 0.0
            return zero, {"actor/pg_clipfrac": 0.0, "actor/ppo_kl": 0.0, "actor/pg_clipfrac_lower": 0.0}
        return original_compute_policy_loss_vanilla(*args, **kwargs)

    compute_ref_log_prob_with_entropy._entropy_opd_patched = True
    update_policy_entropy_opd._entropy_opd_patched = True
    compute_policy_loss_vanilla_entropy_safe._entropy_opd_patched = True
    ray_worker_group_init_with_entropy_env._entropy_opd_patched = True

    RayWorkerGroup.__init__ = ray_worker_group_init_with_entropy_env
    ActorRolloutRefWorker.compute_ref_log_prob = compute_ref_log_prob_with_entropy
    DataParallelPPOActor.update_policy = update_policy_entropy_opd
    core_algos.compute_policy_loss_vanilla = compute_policy_loss_vanilla_entropy_safe
    core_algos.POLICY_LOSS_REGISTRY["vanilla"] = compute_policy_loss_vanilla_entropy_safe

    if _env_flag("ENTROPY_OPD_VERBOSE", True):
        print(f"[entropy-opd] enabled: top_fraction={_env_float('ENTROPY_OPD_FRACTION', 0.2)}")
