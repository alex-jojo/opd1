"""Main entrypoint for the isolated TA-OPD baseline recipe."""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from recipe.ta_opd_baseline.ta_opd_ray_trainer import RayTAOPDTrainer
from verl.trainer.main_ppo import (
    TaskRunner as PPOTaskRunner,
    _env_flag,
    create_rl_dataset,
    create_rl_sampler,
    run_ppo,
)
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config


class TAOPDTaskRunner(PPOTaskRunner):
    """TaskRunner that swaps in the TA-OPD FSDP worker and trainer."""

    def add_actor_rollout_worker(self, config):
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role

        if config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
            raise NotImplementedError("The isolated TA-OPD baseline currently supports FSDP/FSDP2 only.")

        from recipe.ta_opd_baseline.ta_opd_fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

        actor_rollout_cls = (
            AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
        )
        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
        return actor_rollout_cls, RayWorkerGroup

    def run(self, config):
        from pprint import pprint

        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local

        if _env_flag("VERL_PRINT_CONFIG"):
            print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
            pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        ref_tokenizer = None
        ref_model_config = config.actor_rollout_ref.ref.get("model", {})
        ref_model_path = ref_model_config.get("path", None) if ref_model_config else None
        if ref_model_path is not None:
            ref_local_path = copy_to_local(
                ref_model_path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
            )
            ref_tokenizer = hf_tokenizer(ref_local_path, trust_remote_code=trust_remote_code)
            print(f"Loaded ref_tokenizer from {ref_local_path} for re-tokenization")

        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        resource_pool_manager = self.init_resource_pool_mgr(config)
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = RayTAOPDTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            ref_tokenizer=ref_tokenizer,
        )
        trainer.init_workers()
        trainer.fit()


@hydra.main(config_path="config", config_name="ta_opd_trainer", version_base=None)
def main(config):
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(TAOPDTaskRunner))


if __name__ == "__main__":
    main()
