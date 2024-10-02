import json
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union

import torch
from accelerate.utils import release_memory
from datasets import load_from_disk, concatenate_datasets, Dataset
from deepspeed import DeepSpeedEngine
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel
from wandb.sdk.wandb_run import Run

from treetune import logging_utils
from treetune.analyzers.analyzer import Analyzer
from treetune.common import Lazy
from treetune.trainers.data_collator import (
    PPODataCollator,
    COLUMN_VALUES,
    COLUMN_REF_SHIFTED_LOGPS,
    COLUMN_ACTOR_SHIFTED_LOGPS,
)
from treetune.trainers.policy_trainer import PolicyTrainer
from treetune.trainers.ppo_trainer import PPOTrainer
from treetune.trainers.utils import masked_whiten

logger = logging_utils.get_logger(__name__)


@Analyzer.register("ppo_gradient_variance")
class PPOGradientVarianceAnalyzer(Analyzer):
    def __init__(
        self,
        actor_deepspeed_config: Dict[str, Any],
        cloud_logger: Run,
        runtime,
        num_bootstrap_samples: int = 32,
        num_bootstrap_runs: int = 10,
        gradient_clipping: Optional[float] = None,
        per_device_batch_size: Optional[int] = None,
        max_num_checkpoints: Optional[int] = None,
        store_rolling_aggregates_on_cpu: bool = False,
        **kwargs,
    ):
        from treetune.runtime.policy_iteration_runtime import PolicyIterationRuntime

        assert isinstance(runtime, PolicyIterationRuntime)
        self.runtime: PolicyIterationRuntime = runtime

        super().__init__(cloud_logger, runtime, **kwargs)

        trainer = getattr(self.runtime, "trainer", None)
        if trainer is None or isinstance(trainer, Lazy):
            # noinspection PyProtectedMember
            trainer = self.runtime._construct_trainer(init_model_only=False)
        assert isinstance(trainer, PPOTrainer)

        self.trainer: PPOTrainer = trainer
        self.trainer.cache_deepspeed_engines = False
        self.trainer.actor_deepspeed_config = actor_deepspeed_config

        self.num_bootstrap_samples = num_bootstrap_samples
        self.num_bootstrap_runs = num_bootstrap_runs

        self.per_device_batch_size = per_device_batch_size
        self.max_num_checkpoints = max_num_checkpoints
        self.gradient_clipping = gradient_clipping

        self.store_rolling_aggregates_on_cpu = store_rolling_aggregates_on_cpu

        per_device_batch_size = self.per_device_batch_size
        if self.per_device_batch_size is None:
            per_device_batch_size = self.trainer.args.per_device_train_batch_size
        assert self.num_bootstrap_samples % per_device_batch_size == 0
        self.grad_acc_steps = self.num_bootstrap_samples // per_device_batch_size
        if actor_deepspeed_config.get("gradient_accumulation_steps", None) == "auto":
            actor_deepspeed_config["gradient_accumulation_steps"] = self.grad_acc_steps
        if actor_deepspeed_config.get("train_batch_size", None) == "auto":
            actor_deepspeed_config["train_batch_size"] = (
                per_device_batch_size * self.grad_acc_steps
            )

    def _get_list_of_evaluation_checkpoints(
        self, every_n_checkpoints: int, ignore_worker_vars: bool = False
    ) -> List[Tuple[Path, Path]]:
        checkpoint_dir = self.runtime.exp_root / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True, parents=True)

        ckpts = self.runtime._get_list_of_evaluation_checkpoints(
            checkpoint_dir, every_n_checkpoints, ignore_worker_vars=True
        )

        available_episodes = [
            episode_dir
            for episode_dir in checkpoint_dir.glob("episodes__iter*")
            if (episode_dir / "w_actLogp_and_values").exists()
            or (episode_dir / "w_actLogp").exists()
        ]

        iter_to_episode_dir = {}
        for episode_dir in available_episodes:
            iter_idx = int(episode_dir.name.split("__iter")[-1])
            if (episode_dir / "w_actLogp_and_values").exists():
                iter_to_episode_dir[iter_idx] = episode_dir / "w_actLogp_and_values"
            else:
                iter_to_episode_dir[iter_idx] = episode_dir / "w_actLogp"

        def can_analyze(ckpt: Path) -> bool:
            if not (ckpt / "hf_pretrained" / "pytorch_model.bin").exists():
                return False
            ckpt_iter = int(PolicyTrainer.parse_checkpoint_name(ckpt.name)[0])
            episodes_iter_of_ckpt = ckpt_iter + 1
            return episodes_iter_of_ckpt in iter_to_episode_dir

        ckpts = [ckpt for ckpt in ckpts if can_analyze(ckpt)]

        # Limit the number of checkpoints to analyze
        if self.max_num_checkpoints is not None:
            num_ckpts = len(ckpts)
            if num_ckpts > self.max_num_checkpoints:
                step = num_ckpts / self.max_num_checkpoints
                ckpts = [ckpts[int(i * step)] for i in range(self.max_num_checkpoints)]

        # Add the initial model
        if 0 in iter_to_episode_dir:
            ckpts = [None] + ckpts

        if (
            not ignore_worker_vars
            and "APP_EVAL_NUM_WORKERS" in os.environ
            and "APP_EVAL_WORKER_ID" in os.environ
        ):
            # Distribute the checkpoints across workers, such that each worker
            # evaluates an equal number of checkpoints
            num_workers = int(os.environ["APP_EVAL_NUM_WORKERS"])
            worker_id = int(os.environ["APP_EVAL_WORKER_ID"])
            logger.info(
                f"Running evaluation on worker {worker_id+1} out of {num_workers}"
            )
            ckpts = ckpts[worker_id::num_workers]

        episode_dirs = []
        for ckpt in ckpts:
            if ckpt is not None:
                iter_idx = int(PolicyTrainer.parse_checkpoint_name(ckpt.name)[0]) + 1
            else:
                iter_idx = 0
            episode_dirs.append(iter_to_episode_dir[iter_idx])

        return list(zip(ckpts, episode_dirs))

    def analyze(
        self, every_n_checkpoints: int = 1, force_rerun: bool = False, **kwargs
    ):
        analysis_root = self.get_analysis_root_dir()
        analysis_root.mkdir(exist_ok=True, parents=True)

        if not force_rerun and (analysis_root / "done").exists():
            logger.warning(
                f"Analysis directory {self.get_analysis_root_dir()} already exists."
            )
            return

        ckpts = self._get_list_of_evaluation_checkpoints(every_n_checkpoints)
        logger.info(
            f"Evaluating PPO Gradient Variance on {len(ckpts)} checkpoints: "
            f"{[ckpt.name if ckpt is not None else 'initial_model' for ckpt, _ in ckpts]}"
        )

        for ckpt, episode_dir in tqdm(ckpts, desc="Analyzing checkpoints"):
            self._analyze_checkpoint(ckpt, episode_dir, force_rerun=force_rerun)

        ckpts = self._get_list_of_evaluation_checkpoints(
            every_n_checkpoints, ignore_worker_vars=True
        )
        all_ckpts_are_done = all(
            (
                analysis_root / ("initial_model" if c is None else c.name) / "done"
            ).exists()
            for c, _ in ckpts
        )
        if all_ckpts_are_done:
            (analysis_root / "done").touch()
            metrics = {}
            for ckpt, _ in ckpts:
                ckpt_name = "initial_model" if ckpt is None else ckpt.name
                if ckpt_name != "initial_model":
                    iter_idx = int(PolicyTrainer.parse_checkpoint_name(ckpt_name)[0])
                else:
                    iter_idx = 0

                with open(analysis_root / ckpt_name / "metrics.json", "r") as f:
                    metrics[iter_idx] = json.load(f)
            self.log(metrics)

            # Remove ckpt_*/metrics.json files
            for ckpt, _ in ckpts:
                ckpt_name = "initial_model" if ckpt is None else ckpt.name
                try:
                    (analysis_root / ckpt_name / "metrics.json").unlink()
                except FileNotFoundError:
                    pass

    def _analyze_checkpoint(
        self, ckpt: Optional[Path], episode_dir: Path, force_rerun: bool = False
    ):
        if ckpt is not None:
            global_step = int(PolicyTrainer.parse_checkpoint_name(ckpt.name)[-1])
            ckpt_name = ckpt.name
        else:
            global_step = 0
            ckpt_name = "initial_model"

        ckpt_eval_root_dir = self.get_analysis_root_dir() / ckpt_name
        ckpt_eval_root_dir.mkdir(exist_ok=True, parents=True)

        if (
            not force_rerun
            and (ckpt_eval_root_dir / "done").exists()
            and (ckpt_eval_root_dir / "metrics.json").exists()
        ):
            logger.info(f"Skipping {ckpt} as it has already been analyzed.")
            return
        logger.info(f"Analyzing checkpoint {ckpt_name}")

        episodes = load_from_disk(str(episode_dir))
        actor = self._init_actor(ckpt)

        if isinstance(actor, DeepSpeedEngine):
            assert actor.zero_optimization_stage() == 0, "Zero stage must be 0"

        rng = random.Random(42)

        ds_lst = []
        for _ in range(self.num_bootstrap_runs):
            indices = rng.sample(range(len(episodes)), self.num_bootstrap_samples)
            ds_lst.append(episodes.select(indices))

        ds = concatenate_datasets(ds_lst)
        variance, sample_variance = self._compute_gradient_variance(actor, ds)
        self.trainer._destroy_ds_engine(actor)
        del actor
        release_memory()

        self.cloud_logger.log(
            {
                f"{self.plot_prefix}/{self.__class__.__name__}__variance": variance,
                f"{self.plot_prefix}/{self.__class__.__name__}__sample_variance": sample_variance,
                "train/global_step": global_step,
            }
        )

        # Save to disk
        with open(ckpt_eval_root_dir / "metrics.json", "w") as f:
            json.dump(
                {"variance": variance, "sample_variance": sample_variance},
                f,
            )

        (ckpt_eval_root_dir / "done").touch()

    # noinspection DuplicatedCode
    def _compute_gradient_variance(
        self, actor: Union[DeepSpeedEngine, PreTrainedModel], dataset: Dataset
    ) -> Tuple[float, float]:
        per_device_batch_size = self.per_device_batch_size
        if per_device_batch_size is None:
            per_device_batch_size = self.trainer.args.per_device_train_batch_size

        data_loader = DataLoader(
            dataset,
            batch_size=per_device_batch_size,
            collate_fn=PPODataCollator(),
            num_workers=4,
            shuffle=False,
        )

        grad_acc_steps = self.grad_acc_steps

        # Implementing Welford's online algorithm for computing variance
        # Refer to https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        grad_rolling_mean = {}
        grad_rolling_m2 = {}
        count = 0

        def update_rolling_aggregates(
            param_name: str, param_grad: torch.Tensor, new_count: int
        ):
            if param_name not in grad_rolling_mean:
                grad_rolling_mean[param_name] = torch.zeros_like(param_grad)
                grad_rolling_m2[param_name] = torch.zeros_like(param_grad)

            mean = grad_rolling_mean[param_name]
            m2 = grad_rolling_m2[param_name]

            delta = param_grad - mean
            mean += delta / new_count
            delta2 = param_grad - mean
            m2 += delta * delta2

            grad_rolling_mean[param_name] = mean
            grad_rolling_m2[param_name] = m2

        for step, inputs in enumerate(
            tqdm(
                data_loader,
                desc=f"Computing gradient variance",
                dynamic_ncols=True,
            )
        ):
            release_memory()
            is_grad_acc_boundary = (step + 1) % grad_acc_steps == 0

            # noinspection PyTypeChecker
            self._run_forward_and_backward(inputs, actor)

            if is_grad_acc_boundary:
                has_finite_grads = True
                for name, param in actor.named_parameters():
                    if param.grad is not None:
                        if torch.any(torch.isnan(param.grad)) or torch.any(
                            torch.isinf(param.grad)
                        ):
                            logger.warning(
                                f"Gradient variance computation: NaN in gradients at step {step}"
                            )
                            has_finite_grads = False
                            break
                if not has_finite_grads:
                    actor.zero_grad()
                    continue

                if self.gradient_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(
                        actor.parameters(), self.gradient_clipping
                    )

                count += 1
                for name, param in actor.named_parameters():
                    if param.grad is not None:
                        param_grad = param.grad.detach().clone()
                        if self.store_rolling_aggregates_on_cpu:
                            param_grad = param_grad.cpu()

                        update_rolling_aggregates(name, param_grad, count)
                        del param_grad

                actor.zero_grad()

        # Compute the variance
        # Which is basically the trace of the covariance matrix
        total_variance = 0.0
        total_sample_variance = 0.0

        for name in sorted(grad_rolling_mean.keys()):
            m2 = grad_rolling_m2[name]

            variance = m2 / count
            sample_variance = m2 / (count - 1)

            variance = variance.float().sum().item()
            sample_variance = sample_variance.float().sum().item()

            total_variance += variance
            total_sample_variance += sample_variance

        return total_variance, total_sample_variance

    # noinspection DuplicatedCode
    def _run_forward_and_backward(
        self,
        inputs: Dict[str, torch.Tensor],
        actor: Union[DeepSpeedEngine, PreTrainedModel],
    ) -> None:
        # Copy-pasted from PPOTrainer._training_step
        inputs = {k: v.to(actor.device) for k, v in inputs.items()}

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        scores = inputs["scores"]

        shifted_labels = labels[..., 1:].contiguous()
        shifted_labels_mask = (shifted_labels != -100).to(attention_mask.dtype)

        shifted_actor_logprobs = inputs[COLUMN_ACTOR_SHIFTED_LOGPS]
        assert shifted_actor_logprobs.shape == shifted_labels_mask.shape

        with torch.no_grad():
            # noinspection PyProtectedMember
            if self.trainer._is_kl_penalty_enabled():
                shifted_ref_logprobs = inputs[COLUMN_REF_SHIFTED_LOGPS]
            else:
                shifted_ref_logprobs = None

            # noinspection PyProtectedMember
            rewards, non_score_rewards, kls = self.trainer._compute_rewards(
                scores, shifted_actor_logprobs, shifted_ref_logprobs, attention_mask
            )

            if "advantages" not in inputs:
                values = inputs[COLUMN_VALUES]
                valid_values = values[:, :-1]
                assert valid_values.shape == shifted_actor_logprobs.shape
                valid_values = valid_values * shifted_labels_mask
                # noinspection PyProtectedMember
                advantages, returns = self.trainer._compute_advantages(
                    valid_values, rewards, shifted_labels_mask
                )
            else:
                precomputed_advantages = inputs["advantages"]
                advantages = precomputed_advantages[:, 1:]
                if self.trainer.ppo_hparams.whiten_advantages:
                    advantages = masked_whiten(
                        advantages,
                        shifted_labels_mask,
                        distributed=True,
                        unbiased_variance=True,
                    )

            assert advantages.shape == shifted_actor_logprobs.shape
            assert rewards.shape == shifted_actor_logprobs.shape

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        # noinspection PyProtectedMember
        actor_loss, _, _, _ = self.trainer._compute_actor_loss(
            actor,
            model_inputs=model_inputs,
            shifted_labels_mask=shifted_labels_mask,
            old_logprobs=shifted_actor_logprobs,
            ref_logprobs=shifted_ref_logprobs,
            advantages=advantages,
        )
        actor_loss.backward()

    # noinspection DuplicatedCode
    # noinspection PyProtectedMember
    def _init_actor(
        self, ckpt: Optional[Path]
    ) -> Union[DeepSpeedEngine, PreTrainedModel]:
        if ckpt is not None:
            # Patch the trainer to construct the model from the checkpoint
            assert "hf_model_name" in self.trainer.actor_lazy._params
            self.trainer.actor_lazy._params["hf_model_name"] = str(
                ckpt / "hf_pretrained"
            )
        else:
            init_model_name = self.trainer.actor_lazy._params["hf_model_name"]
            logger.info(f"Initializing actor model from {init_model_name}")

        actor = self.trainer._init_actor_model()
        actor.train()

        return actor
