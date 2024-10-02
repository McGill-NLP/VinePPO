import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Union, List, Dict, Any, Tuple
from weakref import WeakValueDictionary

import numpy as np
import torch
from accelerate.checkpointing import save_custom_state, load_custom_state
from accelerate.utils import release_memory, gather, pad_across_processes
from datasets import Dataset
from deepspeed import DeepSpeedEngine
from deepspeed import comm as dist
from deepspeed.runtime.utils import see_memory_usage
from torch.nn import functional as F
from tqdm import tqdm
from transformers import PreTrainedModel
from transformers.integrations import HfTrainerDeepSpeedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.trainer_pt_utils import get_model_param_count

from treetune.common import JsonDict, Lazy
from treetune.common.deepspeed_utils import (
    prepare_data_loader_for_inference,
    prepare_data_loader_for_training,
)
from treetune.common.py_utils import need_to_minimize_stored_files
from treetune.common.wandb_utils import get_repo_dir
from treetune.logging_utils import get_logger
from treetune.models.base_model import Model
from treetune.trainers.arguments import TrainingArguments
from treetune.trainers.base_trainer import Trainer
from treetune.trainers.data_collator import (
    PPODataCollator,
    COLUMN_REF_SHIFTED_LOGPS,
    COLUMN_ACTOR_SHIFTED_LOGPS,
    COLUMN_VALUES,
)
from treetune.trainers.deepspeed_policy_trainer import DeepSpeedPolicyTrainer
from treetune.trainers.policy_trainer import Checkpoint
from treetune.trainers.utils import (
    masked_mean,
    entropy_from_logits,
    DeepSpeedRunningMoments,
    masked_whiten,
    masked_var,
    monitor_tensor_anomalies,
    close_to_zero_percentage,
    masked_rescale_by_std,
)

logger = get_logger(__name__)


@dataclass
class PPOHParams:
    """
    Configuration class for PPOTrainer.

    Parameters:
        adap_kl_ctrl (bool):
            Use adaptive KL control, otherwise linear.
        init_kl_coef (Optional[float]):
            Initial KL penalty coefficient (used for adaptive and linear control).
        kl_penalty (Literal["kl", "abs", "mse", "full"]):
            KL penalty options. 'kl': model_logp - ref_logp, 'abs': abs(kl),
            'mse': mean squared error mse(kl) and 'full': the actual kl for all tokens in the distribution.
        target (Optional[float]):
            Target KL value for adaptive KL control.
        gamma (float):
            Gamma parameter for advantage calculation.
        lam (float):
            Lambda parameter for advantage calculation.
        cliprange (float):
            Range for clipping in PPO policy gradient loss.
        cliprange_value (float):
            Range for clipping values in loss calculation.
        vf_coef (float):
            Scaling factor for value loss.
        early_stopping (bool):
            Whether to stop the PPO optimization loop early if the KL is too high.
        target_kl (float):
            Stop early if we exceed this value by over 50%.
        compare_steps (int):
            Number of steps between comparison of the current reward with the best seen so far.
        ratio_threshold (float):
            Skip mini-batches with high PPO ratios that can cause loss spikes.
        use_score_scaling (bool):
            Use score scaling.
        use_score_norm (bool):
            Use score normalization. Only applicable if use_score_scaling is True.
        score_clip (Optional[float]):
            Score clipping.
        whiten_advantages (bool):
            Whiten the advantages before computing the actor loss.
        grayen_advantages (bool):
            Only change the scale of the advantages to have a std of 1.
        whiten_rewards (bool):
            Whiten the rewards before compute advantages.
        temperature (float):
            The temperature used for sampling.
    """

    adap_kl_ctrl: bool = True
    init_kl_coef: Optional[float] = 0.2
    kl_penalty: Literal["kl", "abs", "mse", "full", "control_variate"] = "kl"
    kl_penalty_loss_type: Optional[Literal["kl", "abs", "mse", "control_variate"]] = (
        None
    )
    kl_penalty_loss_clip_max: float = 10000
    kl_penalty_loss_clip_min: float = 0
    force_disable_kl_penalty: bool = False
    target: Optional[float] = 6.0
    horizon: Optional[int] = 10000
    gamma: float = 1
    lam: float = 0.95
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    early_stopping: bool = False
    target_kl: float = 1
    compare_steps: int = 1
    ratio_threshold: float = 10.0
    use_score_scaling: bool = False
    use_score_norm: bool = False
    score_clip: Optional[float] = None
    whiten_advantages: bool = True
    grayen_advantages: bool = False
    whiten_rewards: bool = False
    temperature: float = 1.0

    def __post_init__(self):
        assert self.temperature > 0, "Temperature should be positive."
        assert not (
            self.whiten_advantages and self.grayen_advantages
        ), "Either whiten or grayen advantages, not both."


@Trainer.register("ppo")
class PPOTrainer(DeepSpeedPolicyTrainer):
    def __init__(
        self,
        num_episodes_per_iteration: int,
        params: JsonDict,
        actor_model: Lazy[Model],
        actor_deepspeed_config: JsonDict,
        general_training_args: JsonDict,
        critic_model: Optional[Lazy[Model]] = None,
        critic_deepspeed_config: Optional[JsonDict] = None,
        reference_model: Optional[Lazy[Model]] = None,
        reference_deepspeed_config: Optional[JsonDict] = None,
        num_iterations: int = 1,
        num_epochs_per_iteration: int = 1,
        disable_critic_training: bool = False,
        report_entropy: bool = True,
        align_skipping_on_overflow: bool = True,
        enable_exponential_moving_average_actor: bool = False,
        cache_reference_model_on_temp_storage: bool = False,
        temp_checkpoint_dir: Optional[str] = None,
        profile_torch_memory: bool = False,
        cache_deepspeed_engines: bool = False,
        move_reference_model_to_cpu: bool = False,
        save_hf_critic_checkpoint: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._set_process_log_level(logger)

        self.ppo_hparams = PPOHParams(**params)
        self.args = TrainingArguments(**general_training_args)
        self.align_skipping_on_overflow = align_skipping_on_overflow

        self.num_iterations = num_iterations
        self.num_epochs_per_iteration = num_epochs_per_iteration
        self.num_episodes_per_iteration = num_episodes_per_iteration
        self._compute_batch_size_and_steps()

        self.actor_lazy = actor_model
        self.actor_deepspeed_config = actor_deepspeed_config

        self.critic_lazy = critic_model
        self.critic_deepspeed_config = critic_deepspeed_config
        if self.critic_lazy is None:
            logger.info(
                "No critic model provided. We then assume values are provided in the episodes."
            )

        self.reference_lazy = reference_model
        self.reference_deepspeed_config = reference_deepspeed_config
        if self.reference_lazy is None:
            logger.info("No reference model provided. We then assume no KL penalty.")

        if enable_exponential_moving_average_actor:
            self.ema_actor_model = self.actor_lazy
        else:
            self.ema_actor_model = None

        # This operation is done on the same data across all processes
        # So, there is no need to synchronize the operation
        self.running_scores = DeepSpeedRunningMoments(force_no_sync=True)

        if self.ppo_hparams.adap_kl_ctrl:
            self.kl_ctl = AdaptiveKLController(
                self.ppo_hparams.init_kl_coef,
                self.ppo_hparams.target,
                horizon=(
                    self.total_num_training_steps * self.global_batch_size
                ),  # Total number of episodes
            )
        else:
            self.kl_ctl = FixedKLController(self.ppo_hparams.init_kl_coef)

        from deepspeed.utils import logger as ds_logger

        ds_logger.setLevel(logging.DEBUG)

        # We are very conservative with memory (both CPU and GPU) and we want to avoid OOM errors.
        # We load the models in the memory whenever needed. This is why this variable exists.
        self.checkpoint_path_to_load = None

        if temp_checkpoint_dir is not None:
            self.temp_checkpoint_dir = Path(temp_checkpoint_dir)
        else:
            self.temp_checkpoint_dir = get_repo_dir() / "temp_ppo_checkpoints"
            logger.info(
                f"No temporary checkpoint directory provided. Using {self.temp_checkpoint_dir}"
            )
        self.temp_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.cache_reference_model = cache_reference_model_on_temp_storage
        self.profile_torch_memory = profile_torch_memory
        self.cache_deepspeed_engines = cache_deepspeed_engines
        self.move_reference_model_to_cpu = move_reference_model_to_cpu
        self.disable_critic_training = disable_critic_training
        self.report_entropy = report_entropy
        self.save_hf_critic_checkpoint = save_hf_critic_checkpoint

        if self._has_critic_model() and disable_critic_training:
            logger.warning(
                "*************************\n"
                "Critic training is disabled. The critic model will not be trained."
                "\n*************************"
            )

        if self._is_main_process():
            if getattr(self.cloud_logger, "define_metric", None):
                self.cloud_logger.define_metric("train/global_iteration")
                self.cloud_logger.define_metric(
                    "episodes_metric/*",
                    step_metric="train/global_iteration",
                    step_sync=True,
                )

    def _is_kl_penalty_enabled(self):
        return (
            not self.ppo_hparams.force_disable_kl_penalty
            and self.reference_lazy is not None
        )

    def _has_critic_model(self):
        return self.critic_lazy is not None

    def _compute_batch_size_and_steps(self):
        if self.args.target_train_batch_size is not None:
            if (
                self.args.per_device_train_batch_size is None
                and self.args.gradient_accumulation_steps is None
            ):
                raise ValueError(
                    "Either per_device_train_batch_size or gradient_accumulation_steps "
                    "should be provided."
                )
            if (
                self.args.per_device_train_batch_size is not None
                and self.args.gradient_accumulation_steps is not None
            ):
                raise ValueError(
                    "Only one of per_device_train_batch_size or gradient_accumulation_steps "
                    "should be provided."
                )

            if self.args.per_device_train_batch_size is not None:
                self.args.gradient_accumulation_steps = (
                    self.args.target_train_batch_size
                    // self.args.per_device_train_batch_size
                    // self.distributed_state.num_processes
                )
            elif self.args.gradient_accumulation_steps is not None:
                self.args.per_device_train_batch_size = (
                    self.args.target_train_batch_size
                    // self.args.gradient_accumulation_steps
                    // self.distributed_state.num_processes
                )

        self.global_batch_size = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * self.distributed_state.num_processes
        )
        self.total_num_training_steps = (
            self.num_iterations
            * self.num_epochs_per_iteration
            * self.num_episodes_per_iteration
            // self.global_batch_size
        )
        logger.info(f"Per device batch size: {self.args.per_device_train_batch_size}")
        logger.info(
            f"Gradient accumulation steps: {self.args.gradient_accumulation_steps}"
        )
        logger.info(f"Num of total processes: {self.distributed_state.num_processes}")
        logger.info(
            f"Global batch size (w. parallel, distributed & accumulation): {self.global_batch_size}"
        )
        logger.info(
            f"Total number of training steps (Gradient Updates): {self.total_num_training_steps}"
        )

    def get_models(
        self,
    ) -> WeakValueDictionary[str, Union[PreTrainedModel, DeepSpeedEngine]]:
        weak_dict = WeakValueDictionary()
        if getattr(self, "_actor_engine", None) is not None:
            weak_dict["actor"] = self._actor_engine

        if getattr(self, "_critic_engine", None) is not None:
            weak_dict["critic"] = self._critic_engine

        if getattr(self, "_reference_engine", None) is not None:
            weak_dict["reference"] = self._reference_engine

        return weak_dict

    def step(self, episodes_dataset: Dataset) -> Optional[Path]:
        """
        Perform a single step of PPO training. A Single step of policy training amounts to possibly multi epochs
        of training on the episodes.

        Distributed Note:
            This function is called on each process. i.e., they all receive a full copy of the episodes_dataset.

        Considering our conservative memory approach, here is a general idea of each step:
        1. Initialize (load into CPU/GPU) the reference model if it exists.
        2. Compute the reference log probabilities if the reference model exists.
        3. Remove the reference model from the memory.
        4. Initialize (load into CPU/GPU ) the actor model and its optimizer.
        5. Initialize (load into CPU/GPU ) the critic model and its optimizer if it exists.
        6. Load the checkpoint if needed (i.e. self.checkpoint_path_to_load is not None).
        7. Train the actor & possibly the critic with PPO.
        8. Save the actor & critic state (i.e. save checkpoint).
        9. Remove the actor & critic from the memory. (Including the optimizer states)
        10. (Outside of this function) Generate new episodes by sampling from the actor.
        11. (Outside of this function) go back to step 1.

        Args:
            episodes_dataset (Dataset):
                A HuggingFace Dataset containing the episodes to train on.
                It should have the following columns:
                    - "query_token_ids": The token ids of the query.
                    - "response_token_ids": The token ids of the response.
                    - "score": The reward of the response (single scalar per response)
                    - "advantages": The advantages of the response. (Optional)

        Returns:
            Optional[Path]:
                The path to the latest policy (actor) checkpoint.
        """
        episodes_dataset = self._filter_episodes(episodes_dataset)
        if self._is_kl_penalty_enabled():
            # Compute or reload from disk the episodes with reference log probabilities
            # It takes care initializing and destroying the reference model
            episodes_dataset = self._get_episodes_w_ref_logps(episodes_dataset)

        # Initialize the actor and critic models along with their optimizers
        logger.info("Initializing the actor model.")
        actor_engine = self._init_actor_model()
        critic_engine = None
        if self._has_critic_model():
            logger.info("Initializing the critic model.")
            critic_engine = self._init_critic_model()

        # Load from checkpoint if specified
        need_to_save_temp_checkpoint = not self.cache_deepspeed_engines
        if self.checkpoint_path_to_load is not None:
            logger.info(f"Loading checkpoint from {self.checkpoint_path_to_load}...")
            self._load_checkpoint_to_ds_engines(
                self.checkpoint_path_to_load, actor_engine, critic_engine
            )
            self.checkpoint_path_to_load = None
        elif not need_to_save_temp_checkpoint:
            logger.info(f"Resuming from {self.state.state_dict()}")
        else:
            assert (
                self.state.state_dict() == self.state.INITIAL_STATE_DICT
            ), f"State should be INITIAL. Got: {self.state.state_dict()}"
            logger.info("No checkpoint to load. Training will start from scratch.")

        # Compute or reload from disk the episodes with current actor log probabilities and values
        episodes_dataset = self._get_episodes_w_curr_logps_and_values(
            episodes_dataset, actor_engine, critic_engine
        )

        # Train the actor and critic models using PPO
        self._train_actor_critic(episodes_dataset, actor_engine, critic_engine)

        # Save the models' state if needed
        should_save_full_ckpt = (
            self.args.save_steps != -1
            and self.state.iteration % self.args.save_steps == 0
        )
        temp_ckpt_path = (
            self.temp_checkpoint_dir / self._get_automatic_checkpoint_name()
        )
        if need_to_save_temp_checkpoint:
            self._save_checkpoint(temp_ckpt_path, actor_engine, critic_engine)
            self.checkpoint_path_to_load = temp_ckpt_path
            if should_save_full_ckpt:
                self._copy_to_permanent_storage(temp_ckpt_path)
        else:
            # Just save the actor for inference
            self._save_hf_pretrained(actor_engine, temp_ckpt_path / "hf_pretrained")
            if should_save_full_ckpt:
                self._save_automatic_checkpoint(
                    actor=actor_engine, critic=critic_engine
                )
        self._clean_old_temp_checkpoints(exclude=[temp_ckpt_path])
        self._clean_episodes(exclude=[temp_ckpt_path.name])

        # Clean up models and their optimizers from memory
        see_memory_usage("Before cleaning up deepspeed from memory", force=True)
        self._destroy_ds_engine(actor_engine)
        del actor_engine
        release_memory()
        if critic_engine is not None:
            self._destroy_ds_engine(critic_engine)
            del critic_engine
            release_memory()
        see_memory_usage("After cleaning up deepspeed from memory", force=True)
        if not self.cache_deepspeed_engines:
            self.log_tensors_on_gpu()

        path_to_latest_policy = temp_ckpt_path / "hf_pretrained"
        return path_to_latest_policy

    def _train_actor_critic(
        self,
        episodes: Dataset,
        actor: DeepSpeedEngine,
        critic: Optional[DeepSpeedEngine] = None,
    ):
        """
        Train the actor and critic models using PPO.

        Args:
            episodes (Dataset):
                The episodes to train on (possibly with reference log probabilities).
            actor (DeepSpeedEngine):
                The actor model to train.
            critic (Optional[DeepSpeedEngine]):
                The critic model to train.
        """
        # Step 1: Rescale and clip the scores if needed
        episodes = self._rescale_and_clip_scores(episodes)

        kls = self._log_episodes_metrics(episodes)

        # Step 2: The actual PPO training loop
        dataloader = prepare_data_loader_for_training(
            episodes,
            per_device_batch_size=self.args.per_device_train_batch_size,
            seed=self.args.seed,
            data_loader_kwargs={
                "collate_fn": PPODataCollator(),
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
            },
        )

        steps_in_epoch = len(dataloader)
        optim_steps_in_epoch = steps_in_epoch // self.args.gradient_accumulation_steps
        optim_steps_in_epoch = max(optim_steps_in_epoch, 1)
        num_optimization_steps_in_iteration = (
            self.num_epochs_per_iteration * optim_steps_in_epoch
        )
        total_num_optimization_steps = (
            self.num_iterations * num_optimization_steps_in_iteration
        )

        logger.info(f"***** Running a PPO training step: {self.state.iteration}  *****")

        logger.info(f"  Num Episodes = {len(episodes):,}")
        logger.info(f"  Num Epochs Per Iteration = {self.num_epochs_per_iteration:,}")
        logger.info(f"  Num Dataloader Steps in an Epoch = {steps_in_epoch:,}")
        logger.info(f"  Num Optim. steps in an Epoch = {optim_steps_in_epoch:,}")
        logger.info(
            f"  Num Optim. steps in an iteration "
            f"(#epoch x #optim_step_per_epoch) = {num_optimization_steps_in_iteration:,}"
        )
        logger.info(
            f"  Total Num Optim. steps (#iteration x #epoch x #optim_step_per_epoch) "
            f"= {total_num_optimization_steps:,}"
        )
        logger.info(f"  World Size = {actor.world_size}")
        logger.info(
            f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}"
        )
        logger.info(
            f"  Per device batch size = {self.args.per_device_train_batch_size}"
        )
        logger.info(
            f"  Global batch size (w. parallel, distributed & accumulation) = {self.global_batch_size}"
        )
        logger.info(f"  -------- Model Parameters --------")

        actor_num_trainable_params = get_model_param_count(actor, trainable_only=True)
        logger.info(
            f"  Number of trainable parameters (actor) = {actor_num_trainable_params:,}"
        )
        if self._can_log_to_cloud():
            self.cloud_logger.summary["actor/num_trainable_params"] = (
                actor_num_trainable_params
            )
        if critic is not None:
            critic_num_trainable_params = get_model_param_count(
                critic, trainable_only=True
            )
            logger.info(
                f"  Number of trainable parameters (critic) = {critic_num_trainable_params:,}"
            )
            if self._can_log_to_cloud():
                self.cloud_logger.summary["critic/num_trainable_params"] = (
                    critic_num_trainable_params
                )

        logger.info(f"  ---------------------------------")
        logger.info(f"  Current Global Step = {self.state.global_step}")

        # Create a new dataloader iterator
        dataloader_iter = iter(dataloader)

        # Check if we're resuming training in the middle of an iteration
        completed_optim_steps_in_this_iteration = (
            self.state.global_step % num_optimization_steps_in_iteration
        )
        assert (
            completed_optim_steps_in_this_iteration == 0
        ), "We don't support resuming training in the middle of an iteration. "

        progress_bar = tqdm(
            total=total_num_optimization_steps,
            disable=not self._is_main_process(),
            desc=f"Iteration {self.state.iteration}: Training",
            dynamic_ncols=True,
        )
        progress_bar.update(self.state.global_step)

        globalstep_last_logged = self.state.global_step

        actor.train()
        if critic is not None:
            critic.train()

        running_metrics = {}
        accumulated_metrics = {}

        dist.barrier()

        starting_epoch = 0
        for epoch in range(starting_epoch, self.num_epochs_per_iteration):
            for step, inputs in enumerate(dataloader_iter):
                # Store the grad_acc_boundary before engine.step() is called
                # as the engine.step() will reset the grad_acc_boundary
                is_grad_acc_boundary = actor.is_gradient_accumulation_boundary()
                if critic is not None and not self.disable_critic_training:
                    assert (
                        critic.is_gradient_accumulation_boundary()
                        == is_grad_acc_boundary
                    ), "Actor and critic should have synchronized optimization steps"

                # Perform the training step, LR scheduler step, zero_grad, and accumulation of gradients
                # noinspection PyTypeChecker
                metrics = self._training_step(inputs, actor, critic)

                self._update_metrics(running_metrics, accumulated_metrics, metrics)

                if is_grad_acc_boundary:
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    progress_bar.update(1)

                    should_log = self.state.global_step % self.args.logging_steps == 0
                    if should_log:
                        self._log_training_metrics(
                            globalstep_last_logged,
                            accumulated_metrics=accumulated_metrics,
                            progress_bar=progress_bar,
                            actor=actor,
                            critic=critic,
                        )
                        globalstep_last_logged = self.state.global_step

            # Recreate the dataloader iterator
            dataloader_iter = iter(dataloader)

        dist.barrier()

        for key, value in running_metrics.items():
            value = torch.tensor(value, device=actor.device)
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            value = value.cpu().item() / dist.get_world_size()
            running_metrics[key] = value

        if len(running_metrics) > 0:
            logger.info(f"Running metrics: {running_metrics}")

        if self._is_kl_penalty_enabled():
            assert isinstance(kls, float)
            self.kl_ctl.update(kls, self.num_episodes_per_iteration)

        self.state.iteration += 1

        progress_bar.close()

    def _training_step(
        self,
        inputs: Dict[str, torch.Tensor],
        actor: DeepSpeedEngine,
        critic: Optional[DeepSpeedEngine],
    ) -> Dict[str, Union[float, torch.Tensor]]:
        # To better understand the alignment of inputs, logits, logps, and labels,
        # we provide a detailed explanation below.
        # -----------------------------------------------------------------------------------------------------------
        # Consider the sequence: "<s> q1 q2 a b c </s>", where "<s> q1 q2" forms the prompt, and "a b c </s>"
        # the response, with `prompt_len = 3` and `response_len = 4`. Additionally, we include a padding token at
        # the end for the sake of generality.
        #
        # Here is the inputs dictionary setup:
        # Inputs:
        # [     <s>           q1           q2           a            b            c           </s>         <p>   ]
        # Attn Mask:
        # [       1            1            1           1            1            1              1           0   ]
        # Labels:
        # [    -100         -100         -100        ID_a         ID_b         ID_c        ID_</s>        -100   ]
        # >>> seq_len = torch.sum(attn_mask) = 7
        #
        # Feeding the Inputs+Attn Mask, the model outputs logits for next tokens in the sequence:
        # Logits:
        # [  p(.|<s>)      p(.|q1)      p(.|q2)      p(.|a)       p(.|b)       p(.|c)      p(.|</s>)    p(.|<p>)]
        #
        # We exclude the nonsensical last logit (predicting beyond </s>). Also, to obtain the logprobs of next
        # ground-truth token, we shift the labels by 1 to the left:
        #
        # Valid Logits:
        # [  p(.|<s>)      p(.|q1)      p(.|q2)      p(.|a)       p(.|b)       p(.|c)      p(.|</s>)  ]
        # Shifted Labels:
        # [     -100         -100         ID_a        ID_b         ID_c        ID_</s>         -100   ]
        # Shifted Labels Mask (aka Action Mask), i.e. shifted_labels != -100, highlighting valid token-preds/actions:
        # [        0            0            1           1            1            1              0   ]
        # Aligning them to obtain logprobs (lp) of predicting the next token (or lp of action):
        # [    lp(q1)       lp(q2)        lp(a)       lp(b)        lp(c)      lp(</s>)       lp(<p>)  ]
        #
        #
        # Applying the labels mask gives us logprobs of predicting the valid response tokens (aka actions):
        # [     -inf          inf         lp(a)       lp(b)        lp(c)      lp(</s>)         -inf   ]
        # Rewriting the original shifted inputs (for clarity). These are the states we care about:
        # [      <s>           q1           q2           a            b            c           </s>   ]
        # In this example, with S=[<s>;q1;q2] as the state and A=a, we compute lp(A|S) = log(p(a|<s>;q1;q2)) = lp(a)
        #
        # Note that the values are also computed for the entire sequence, but similar to inputs, we ignore
        # the last one since it nonsensical (i.e. V(</s>) is not used).
        # Valid Values:
        # [    V(<s>)        V(q1)        V(q2)        V(a)         V(b)         V(c)        V(</s>)  ]
        # Applying the action mask:
        # [     -inf         -inf         V(q2)        V(a)         V(b)         V(c)          -inf   ]
        #
        # >>> logits_seq_len = logps_seq_len = valid_values_len = seq_len - 1 = 6

        # noinspection DuplicatedCode
        inputs = {k: v.to(actor.device) for k, v in inputs.items()}

        input_ids = inputs["input_ids"]  # Shape: (batch_size, max_seq_len)
        attention_mask = inputs["attention_mask"]  # Shape: (batch_size, max_seq_len)
        labels = inputs["labels"]  # Shape: (batch_size, max_seq_len)
        scores = inputs["scores"]  # Shape: (batch_size,)

        shifted_labels = labels[
            ..., 1:
        ].contiguous()  # Shape: (batch_size, max_seq_len-1)
        shifted_labels_mask = (shifted_labels != -100).to(
            attention_mask.dtype
        )  # Shape: (batch_size, max_seq_len-1)

        # Note that this is the log probability of the actor model
        # in the beginning of this iteration (aka the old log probs)
        shifted_actor_logprobs = inputs[
            COLUMN_ACTOR_SHIFTED_LOGPS
        ]  # Shape: (batch_size, max_seq_len-1)
        assert shifted_actor_logprobs.shape == shifted_labels_mask.shape

        # Step 1: Compute the rewards, advantages, and returns
        with torch.no_grad():
            if self._is_kl_penalty_enabled():
                shifted_ref_logprobs = inputs[COLUMN_REF_SHIFTED_LOGPS]
            else:
                shifted_ref_logprobs = None

            # The following are computed for the actions. Thus, they are of shape (batch_size, max_seq_len-1)
            # Shape of `rewards`: (batch_size, max_seq_len-1)
            # Shape of `non_score_reward`: (batch_size, max_seq_len-1)
            # Shape of `kls`: (batch_size, max_seq_len-1)
            rewards, non_score_rewards, kls = self._compute_rewards(
                scores, shifted_actor_logprobs, shifted_ref_logprobs, attention_mask
            )

            # The `advantages` is computed for the actions. That's why they are of shape (batch_size, max_seq_len-1)
            # Shape of `advantages`: (batch_size, max_seq_len-1)

            # The following are computed for the valid states (everything but last state).
            # Shape of `valid_values`: (batch_size, max_seq_len-1)
            # Shape of `returns`: (batch_size, max_seq_len-1)
            if "advantages" not in inputs:
                # Advantages are not precomputed.
                # Compute them here using the values

                # Note that this is the value of the critic model in the beginning of
                # this iteration (aka the old values)
                values = inputs[COLUMN_VALUES]  # Shape: (batch_size, max_seq_len)
                valid_values = values[:, :-1]  # Shape: (batch_size, max_seq_len-1)
                assert valid_values.shape == shifted_actor_logprobs.shape
                valid_values = valid_values * shifted_labels_mask
                advantages, returns = self._compute_advantages(
                    valid_values, rewards, shifted_labels_mask
                )
            else:
                precomputed_advantages = inputs[
                    "advantages"
                ]  # Shape: (batch_size, max_seq_len)

                # Shift the advantages to left to match the actions
                advantages = precomputed_advantages[
                    :, 1:
                ]  # Shape: (batch_size, max_seq_len-1)
                if self.ppo_hparams.whiten_advantages:
                    advantages = masked_whiten(
                        advantages,
                        shifted_labels_mask,
                        distributed=True,
                        unbiased_variance=True,
                    )
                elif self.ppo_hparams.grayen_advantages:
                    advantages = masked_rescale_by_std(
                        advantages,
                        shifted_labels_mask,
                        distributed=True,
                        unbiased_variance=True,
                    )
                valid_values = None
                returns = None

            assert advantages.shape == shifted_actor_logprobs.shape
            assert rewards.shape == shifted_actor_logprobs.shape

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # Step 2: Compute the policy/actor loss
        actor_loss, is_skipped, actor_metrics, approx_ref_kl = self._compute_actor_loss(
            actor,
            model_inputs=model_inputs,
            shifted_labels_mask=shifted_labels_mask,
            old_logprobs=shifted_actor_logprobs,
            ref_logprobs=shifted_ref_logprobs,
            advantages=advantages,
        )
        actor.backward(actor_loss)
        self._check_overflow(actor)
        actor.step()
        # Get rid of actor's activations to free up memory
        actor_loss = actor_loss.detach().clone()
        release_memory()

        # Step 3: Compute the critic loss
        if critic is not None and not self.disable_critic_training:
            critic_loss, critic_metrics = self._compute_critics_loss(
                critic,
                model_inputs=model_inputs,
                shifted_labels_mask=shifted_labels_mask,
                old_valid_values=valid_values,
                returns=returns,
            )
            critic.backward(critic_loss)
            self._check_overflow(critic)
            critic.step()
            # Get rid of critic's activations to free up memory
            critic_loss = critic_loss.detach().clone()
            release_memory()
        else:
            critic_metrics = {}
            critic_loss = None

        metrics = {
            "advantages/mean": masked_mean(advantages, shifted_labels_mask).detach(),
            "advantages/std": (
                masked_var(advantages, shifted_labels_mask).detach().sqrt()
            ),
            "advantages/close_to_zero_perc": close_to_zero_percentage(
                advantages, shifted_labels_mask, threshold=1e-8
            ).detach(),
            "rewards/mean": masked_mean(rewards, shifted_labels_mask).detach(),
            "num_tokens": shifted_labels_mask.sum().detach(),
            "_num_participating_tokens": shifted_labels_mask.sum().detach(),
            **actor_metrics,
            **critic_metrics,
        }
        if returns is not None:
            metrics["returns"] = masked_mean(returns, shifted_labels_mask).detach()
        if non_score_rewards is not None:
            metrics["non_score_rewards"] = masked_mean(
                non_score_rewards, shifted_labels_mask
            ).detach()
        if kls is not None or approx_ref_kl is not None:
            if approx_ref_kl is not None:
                kls = approx_ref_kl

            metrics["kls"] = (kls * shifted_labels_mask).sum(dim=1).mean().detach()
            metrics["kl_coef"] = self.kl_ctl.value

        metrics["actor/loss"] = actor_loss
        metrics["actor/grad_norm"] = actor.get_global_grad_norm()
        if critic_loss is not None:
            metrics["critic/loss"] = critic_loss
            metrics["critic/grad_norm"] = critic.get_global_grad_norm()

        return metrics

    def _compute_actor_loss(
        self,
        actor: DeepSpeedEngine,
        model_inputs: Dict[str, torch.Tensor],
        shifted_labels_mask: torch.LongTensor,
        old_logprobs: torch.FloatTensor,
        ref_logprobs: Optional[torch.FloatTensor],
        advantages: torch.FloatTensor,
    ) -> Tuple[
        torch.FloatTensor, bool, Dict[str, torch.Tensor], Optional[torch.FloatTensor]
    ]:
        """
        Compute the actor loss using PPO.

        Args:
            actor (`DeepSpeedEngine`):
                The actor model.
            model_inputs (`Dict[str, torch.Tensor]`):
                The model inputs for the actor model. Contains the following keys:
                - "input_ids": The input token ids, shape (`batch_size`, `max_seq_len`).
                - "attention_mask": The attention mask, shape (`batch_size`, `max_seq_len`).
                - "labels": The labels, shape (`batch_size`, `max_seq_len`).
            shifted_labels_mask (`torch.LongTensor`):
                The shifted labels mask (aka action_mask), shape (`batch_size`, `max_seq_len-1`).
            old_logprobs (`torch.FloatTensor`):
                The log probabilities of the actor model for the previous iteration,
                shape (`batch_size`, `max_seq_len-1`).
            advantages (`torch.FloatTensor`):
                The advantages of the responses, shape (`batch_size`, `max_seq_len-1`).

        Returns:
            `torch.FloatTensor`: The actor loss.
            `bool`: Whether the batch was skipped.
            `Dict[str, torch.Tensor]`: Metrics from the training step.
            `Optional[torch.FloatTensor]`: The approx_kls tensor.
        """
        # Switch to RL terminology for more clarity
        action_mask = shifted_labels_mask  # Shape: (batch_size, max_seq_len-1)

        # Compute the log probabilities of the actor
        outputs = self._forward_pass_actor(
            actor,
            model_inputs,
            return_all_logps=True,
            return_entropy=self.report_entropy,
        )
        logprobs = outputs["all_logps"]  # Shape: (batch_size, seq_len-1)
        assert logprobs.shape == old_logprobs.shape
        assert action_mask.shape == logprobs.shape

        # Compute the PPO-clip loss
        log_ratio = (logprobs - old_logprobs) * action_mask
        ratio = torch.exp(log_ratio)

        pg_losses1 = -advantages * ratio
        with torch.no_grad():
            pg_losses1_anomalies = monitor_tensor_anomalies(
                pg_losses1.detach(), action_mask
            )
        pg_losses2 = -advantages * torch.clamp(
            ratio, 1.0 - self.ppo_hparams.cliprange, 1.0 + self.ppo_hparams.cliprange
        )
        pg_losses = torch.max(pg_losses1, pg_losses2)
        pg_loss = masked_mean(pg_losses, action_mask)

        if self.ppo_hparams.kl_penalty_loss_type is not None:
            assert ref_logprobs is not None
            ref_kl = self._compute_kl_penalty(
                logprobs,
                ref_logprobs,
                estimation_type=self.ppo_hparams.kl_penalty_loss_type,
            )
            ref_kl = torch.clamp(
                ref_kl * action_mask,
                min=self.ppo_hparams.kl_penalty_loss_clip_min,
                max=self.ppo_hparams.kl_penalty_loss_clip_max,
            )

            ref_kl_loss = self.kl_ctl.value * ref_kl.sum(dim=1).mean()
            pg_loss = pg_loss + ref_kl_loss
            ref_kl = ref_kl.detach()
        else:
            ref_kl = None
            ref_kl_loss = None

        is_skipped = False
        avg_ratio = masked_mean(ratio, action_mask)
        if avg_ratio.item() > self.ppo_hparams.ratio_threshold:
            logger.warning(
                f"High PPO ratio detected: {avg_ratio.item():.2f}. Skipping this batch."
            )
            pg_loss = pg_loss * 0.0
            is_skipped = True

        pg_clip_frac = masked_mean(
            torch.gt(pg_losses2, pg_losses1).float(), action_mask
        )
        approx_kl = 0.5 * masked_mean((logprobs - old_logprobs) ** 2, action_mask)
        policy_kl = masked_mean(old_logprobs - logprobs, action_mask)

        metrics = {
            "actor/approx_kl": approx_kl.detach(),
            "actor/policy_kl": policy_kl.detach(),
            "actor/clip_frac": pg_clip_frac.detach(),
            "actor/ratio": avg_ratio.detach(),
            **{
                f"actor/pg_losses1_anomalies__{i}": v
                for i, v in pg_losses1_anomalies.items()
            },
        }
        if "entropy" in outputs:
            metrics["actor/logit_entropy"] = outputs["entropy"].detach()
        if ref_kl_loss is not None:
            metrics["actor/ref_kl_loss"] = ref_kl_loss.detach()

        return pg_loss, is_skipped, metrics, ref_kl

    def _compute_critics_loss(
        self,
        critic: DeepSpeedEngine,
        model_inputs: Dict[str, torch.Tensor],
        shifted_labels_mask: torch.LongTensor,
        old_valid_values: torch.FloatTensor,
        returns: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.Tensor]]:
        """
        Compute the critic loss.

        Args:
            critic (`DeepSpeedEngine`):
                The critic model.
            model_inputs (`Dict[str, torch.Tensor]`):
                The model inputs for the critic model. Contains the following keys:
                - "input_ids": The input token ids, shape (`batch_size`, `max_seq_len`).
                - "attention_mask": The attention mask, shape (`batch_size`, `max_seq_len`).
            shifted_labels_mask (`torch.LongTensor`):
                The shifted labels mask (aka action_mask), shape (`batch_size`, `max_seq_len-1`).
            old_valid_values (`torch.FloatTensor`):
                The values of the responses from the previous iteration,
                shape (`batch_size`, `max_seq_len-1`).
            returns (`torch.FloatTensor`):
                The returns of the responses, shape (`batch_size`, `max_seq_len-1`).

        Returns:
            `torch.FloatTensor`: The critic loss.
            `Dict[str, torch.Tensor]`: Metrics from the training step.
        """
        # Switch to RL terminology for more clarity
        action_mask = shifted_labels_mask  # Shape: (batch_size, max_seq_len-1)

        # Compute the values
        if "labels" in model_inputs:
            del model_inputs["labels"]
        outputs = self._forward_pass_critic(critic, model_inputs)

        # Get the values of states up to last token (</s> token)
        valid_values = outputs["values"][:, :-1]

        assert valid_values.shape == old_valid_values.shape
        assert action_mask.shape == valid_values.shape

        # Compute the critic loss (MSE loss)
        values_clipped = torch.clamp(
            valid_values,
            old_valid_values - self.ppo_hparams.cliprange_value,
            old_valid_values + self.ppo_hparams.cliprange_value,
        )

        vf_losses1 = (valid_values - returns) ** 2
        with torch.no_grad():
            vf_losses1_anomalies = monitor_tensor_anomalies(
                vf_losses1.detach(), action_mask
            )
        vf_losses2 = (values_clipped - returns) ** 2
        vf_losses = torch.max(vf_losses1, vf_losses2)
        vf_loss = 0.5 * masked_mean(vf_losses, action_mask)

        vf_clip_frac = masked_mean(
            torch.gt(vf_losses2, vf_losses1).float(), action_mask
        )

        metrics = {
            "critic/value": masked_mean(valid_values, action_mask).detach(),
            "critic/mse": masked_mean(
                (valid_values - returns) ** 2, action_mask
            ).detach(),
            "critic/clip_frac": vf_clip_frac.detach(),
            **{
                f"critic/vf_losses1_anomalies__{i}": v
                for i, v in vf_losses1_anomalies.items()
            },
        }

        return vf_loss, metrics

    def _compute_rewards(
        self,
        scores: torch.FloatTensor,
        shifted_actor_logprobs: torch.FloatTensor,
        shifted_ref_logprobs: torch.FloatTensor,
        attention_mask: torch.LongTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute per token rewards from scores and KL-penalty.

        Args:
            scores (`torch.FloatTensor`):
                Scores from the episodes; one scalar per episode, shape (`batch_size`)
            shifted_actor_logprobs (`torch.FloatTensor`):
                Log probabilities of the actor, shape (`batch_size`, `max_seq_len-1`)
            shifted_ref_logprobs (`torch.FloatTensor`):
                Log probabilities of the reference model, shape (`batch_size`, `max_seq_len-1`)
            attention_mask (`torch.LongTensor`):
                Mask for the input, shape (`batch_size`, `max_seq_len`)

        Returns:
            `torch.FloatTensor`: Per token rewards, shape (`batch_size`, `max_seq_len-1`)
            `torch.FloatTensor`: Non-score rewards, shape (`batch_size`, `max_seq_len-1`)
            `torch.FloatTensor`: KL penalty, shape (`batch_size`, `max_seq_len-1`)
        """
        if (
            shifted_ref_logprobs is not None
            and self.ppo_hparams.kl_penalty_loss_type is None
        ):
            kl = self._compute_kl_penalty(shifted_actor_logprobs, shifted_ref_logprobs)
            non_score_rewards = -self.kl_ctl.value * kl
        else:
            # KL penalty is not part of the reward
            kl = None
            non_score_rewards = torch.zeros_like(shifted_actor_logprobs)

        # Initialize the rewards with non-score rewards
        rewards = non_score_rewards.clone()

        # Find the last non-masked index for each sample in the batch
        last_non_masked_indices = (
            torch.cumsum(attention_mask, dim=1)[:, -1] - 1
        )  # Shape: (batch_size)
        # Since the length of shifted_actor_log_probs is `max_seq_len - 1`, we need to
        # subtract 1 from the last non-masked index to get the corresponding index
        last_non_masked_label_indices = last_non_masked_indices - 1

        # Reward is score + KL penalty
        batch_size = rewards.size(0)
        rewards[torch.arange(batch_size), last_non_masked_label_indices] += scores

        if kl is not None:
            kl = kl.detach()

        return rewards.detach(), non_score_rewards.detach(), kl

    def _compute_advantages(
        self,
        valid_values: torch.FloatTensor,
        rewards: torch.FloatTensor,
        shifted_labels_mask: torch.LongTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Compute the advantages from the values and rewards.

        Args:
            valid_values (`torch.FloatTensor`):
                The values of the responses, shape (`batch_size`, `max_seq_len-1`)
            rewards (`torch.FloatTensor`):
                The rewards of the responses, shape (`batch_size`, `max_seq_len-1`)
            shifted_labels_mask (`torch.LongTensor`):
                Left Shifted by 1 Mask for the labels (i.e. actions), shape (`batch_size`, `max_seq_len-1`)

        Returns:
            `torch.FloatTensor`: The advantages of the responses, shape (`batch_size`, `max_seq_len-1`)
            `torch.FloatTensor`: The returns of the responses, shape (`batch_size`, `max_seq_len-1`)
        """
        lastgaelam = 0
        advantages_reversed = []
        actions_seq_len = rewards.shape[-1]

        # Make sure invalid rewards are masked
        rewards *= shifted_labels_mask

        if self.ppo_hparams.whiten_rewards:
            rewards = masked_whiten(
                rewards, shifted_labels_mask, shift_mean=False, distributed=True
            )

        for t in reversed(range(actions_seq_len)):
            next_state_values = (
                valid_values[:, t + 1] if t < (actions_seq_len - 1) else 0.0
            )
            delta = (
                rewards[:, t]
                + self.ppo_hparams.gamma * next_state_values
                - valid_values[:, t]
            )
            lastgaelam = (
                delta + self.ppo_hparams.gamma * self.ppo_hparams.lam * lastgaelam
            )
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)
        assert advantages.shape == rewards.shape

        returns = advantages + valid_values
        if self.ppo_hparams.whiten_advantages:
            advantages = masked_whiten(
                advantages,
                shifted_labels_mask,
                distributed=True,
                unbiased_variance=True,
            )
        elif self.ppo_hparams.grayen_advantages:
            advantages = masked_rescale_by_std(
                advantages,
                shifted_labels_mask,
                distributed=True,
                unbiased_variance=True,
            )
        return advantages.detach(), returns.detach()

    def _compute_kl_penalty(
        self,
        logprob: Union[torch.FloatTensor, np.ndarray],
        ref_logprob: Union[torch.FloatTensor, np.ndarray],
        estimation_type: Optional[str] = None,
    ) -> Union[torch.FloatTensor, np.ndarray]:
        """
        Compute the per-token KL penalty between the log probabilities of the actor and the reference model.

        Args:
            logprob (`Union[torch.FloatTensor, np.ndarray]`):
                Log probabilities of the actor, shape (`batch_size`, T)
            ref_logprob (`Union[torch.FloatTensor, np.ndarray]`):
                Log probabilities of the reference model, shape (`batch_size`, T)

        Returns:
            `Union[torch.FloatTensor, np.ndarray]`: KL penalty, shape (`batch_size`, `T`)
        """

        if estimation_type is None:
            estimation_type = self.ppo_hparams.kl_penalty

        if estimation_type == "kl":
            return logprob - ref_logprob

        if estimation_type == "abs":
            return (logprob - ref_logprob).abs()

        if estimation_type == "mse":
            return 0.5 * (logprob - ref_logprob).square()

        if estimation_type == "control_variate":
            # Compute the per-token approximate KL penalty between the log probabilities of the actor
            # and the reference model as suggested by Schulman in http://joschu.net/blog/kl-approx.html
            #
            # D_KL [π_θ || π_ref] =
            #    π_ref(y_t | x, y_<t) / π_θ(y_t | x, y_<t) - log(π_ref(y_t | x, y_<t) / π_θ(y_t | x, y_<t)) - 1
            #

            log_ratio = ref_logprob - logprob
            if isinstance(log_ratio, torch.Tensor):
                kl = torch.exp(log_ratio) - log_ratio - 1
            elif isinstance(log_ratio, np.ndarray):
                kl = np.exp(log_ratio) - log_ratio - 1
            else:
                raise ValueError("Unsupported type for log_ratio.")
            return kl

        if estimation_type == "seq_control_variate":
            log_ratio = ref_logprob - logprob
            if isinstance(log_ratio, torch.Tensor):
                prob_ratio = torch.exp(log_ratio.sum(dim=-1, keepdim=True))
                kl = prob_ratio - log_ratio - 1
            elif isinstance(log_ratio, np.ndarray):
                prob_ratio = np.exp(log_ratio.sum(axis=-1, keepdims=True))
                kl = prob_ratio - log_ratio - 1
            else:
                raise ValueError("Unsupported type for log_ratio.")
            return kl

        if estimation_type == "full":
            # Flip is required due to this issue? :https://github.com/pytorch/pytorch/issues/57459
            return F.kl_div(
                ref_logprob, logprob, log_target=True, reduction="none"
            ).sum(-1)

        raise NotImplementedError

    def _rescale_and_clip_scores(self, episodes: Dataset) -> Dataset:
        bias_correction = None
        scale_factor = None
        if self.ppo_hparams.use_score_scaling:
            assert "scores" in episodes.column_names, "Scores should be provided."
            scores = torch.tensor(episodes["scores"], dtype=torch.float32)
            scores_mean, scores_std = self.running_scores.update(scores)
            scale_factor = scores_std + torch.finfo(scores.dtype).eps
            if self.ppo_hparams.use_score_norm:  # todo: weird name, right?
                bias_correction = -scores_mean

        clip = self.ppo_hparams.score_clip

        def transform_reward(example: Dict[str, Any]) -> Dict[str, Any]:
            score = example["scores"]
            if bias_correction is not None:
                score = score + bias_correction
            if scale_factor is not None:
                score = score / scale_factor

            if clip is not None:
                score = torch.clip(torch.tensor(score).float(), -clip, clip)

            return {
                "scores": (
                    score.item() if isinstance(score, torch.Tensor) else float(score)
                )
            }

        if "scores" in episodes.column_names and any(
            val is not None for val in [bias_correction, scale_factor, clip]
        ):
            episodes = episodes.map(
                transform_reward,
                num_proc=self.distributed_state.num_processes,
                desc="Rescaling and clipping scores (if needed)",
            )

        return episodes

    log_keys_to_store_in_running_metrics = [
        "_num_participating_tokens",
    ]

    log_keys_weighed_by_num_participating_tokens = [
        "advantages/mean",
        "advantages/std",
        "rewards/mean",
        "returns",
        "non_score_rewards",
        "actor/loss",
        "actor/logit_entropy",
        "actor/approx_kl",
        "actor/policy_kl",
        "actor/clip_frac",
        "ratio",
        "critic/loss",
        "critic/value",
        "critic/mse",
        "critic/clip_frac",
    ]

    def _log_episodes_metrics(self, episodes: Dataset) -> Optional[float]:
        """
        Log scores, advantages, logprobs, and values of the episodes.

        Args:
            episodes (Dataset): The episodes dataset.

        Returns:
            Optional[float]: The KL from reference policy
        """
        if len(episodes) == 0:
            return

        def compute_seq_logp(
            episode: Dict[str, Any], logprobs_w_query: List[float]
        ) -> float:
            query_len = len(episode["query_token_ids"])
            logprobs = logprobs_w_query[query_len - 1 :]
            seq_logprob = sum(logprobs)
            return seq_logprob

        scores = []
        response_lengths = []
        advantages = []
        ref_logprobs = []
        actor_logprobs = []
        critic_values = []
        kls = []
        control_variate_kls = []
        for e in episodes:
            scores.append(e["scores"])
            response_lengths.append(len(e["response_token_ids"]))
            if "advantages" in e:
                advantages += e["advantages"]
            if COLUMN_REF_SHIFTED_LOGPS in e:
                ref_logprobs.append(compute_seq_logp(e, e[COLUMN_REF_SHIFTED_LOGPS]))
            if COLUMN_ACTOR_SHIFTED_LOGPS in e:
                actor_logprobs.append(
                    compute_seq_logp(e, e[COLUMN_ACTOR_SHIFTED_LOGPS])
                )
            if COLUMN_REF_SHIFTED_LOGPS in e and COLUMN_ACTOR_SHIFTED_LOGPS in e:
                actor_lp = np.array(e[COLUMN_ACTOR_SHIFTED_LOGPS])
                ref_lp = np.array(e[COLUMN_REF_SHIFTED_LOGPS])
                kl = self._compute_kl_penalty(actor_lp, ref_lp).sum()
                kls.append(kl)

                # This is unbiased & low variance
                control_variate_kl = self._compute_kl_penalty(
                    actor_lp, ref_lp, estimation_type="control_variate"
                ).sum()
                control_variate_kls.append(control_variate_kl)

            if COLUMN_VALUES in e:
                values = e[COLUMN_VALUES]
                values_without_query = values[
                    len(e["query_token_ids"]) - 1 : -1
                ]  # Skip the last token (</s>)
                if len(values_without_query) == 0:
                    logger.warning(
                        f"Empty values for episode: {json.dumps(e, indent=2)}"
                    )
                critic_values += values_without_query

        scores = np.array(scores)
        response_lengths = np.array(response_lengths)
        actor_logprobs = np.array(actor_logprobs)
        metrics = {
            "scores/mean": np.mean(scores),
            "scores/std": np.std(scores),
            "scores/dist": scores,
            "response_lengths/mean": np.mean(response_lengths),
            "response_lengths/std": np.std(response_lengths),
            "response_lengths/dist": response_lengths,
            "actor_logprobs/sum": np.mean(actor_logprobs),
            "actor_logprobs/normalized_by_response_len": np.mean(
                actor_logprobs / response_lengths
            ),
            "actor_logprobs/dist": actor_logprobs,
        }

        if len(kls) > 0:
            kls = np.array(kls)
            metrics["kls/mean"] = np.mean(kls)
            metrics["kls/dist"] = kls
            kls = float(metrics["kls/mean"])
        else:
            kls = None

        if len(control_variate_kls) > 0:
            control_variate_kls = np.array(control_variate_kls)
            metrics["kls/crtl_var__mean"] = np.mean(control_variate_kls)
            metrics["kls/crtl_var__dist"] = control_variate_kls

        if len(advantages) > 0:
            advantages = np.array(advantages)
            metrics["advantages/mean"] = np.mean(advantages)
            metrics["advantages/std"] = np.std(advantages)
            metrics["advantages/dist"] = advantages

        if len(ref_logprobs) > 0:
            ref_logprobs = np.array(ref_logprobs)
            metrics["ref_logprobs/sum"] = np.mean(ref_logprobs)
            metrics["ref_logprobs/normalized_by_response_len"] = np.mean(
                ref_logprobs / response_lengths
            )
            metrics["ref_logprobs/dist"] = ref_logprobs

        if len(critic_values) > 0:
            critic_values = np.array(critic_values)
            metrics["critic_values/mean"] = np.mean(critic_values)
            metrics["critic_values/std"] = np.std(critic_values)
            metrics["critic_values/dist"] = critic_values

        non_array_metrics = {
            k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)
        }
        logger.info(f"Episode Metrics: {non_array_metrics}")

        logs = {f"episodes_metric/{k}": v for k, v in metrics.items()}
        self._cloud_log(
            {
                **logs,
                "train/global_step": self.state.global_step,
                "train/global_iteration": self.state.iteration,
            }
        )

        return kls

    def _log_training_metrics(
        self,
        _globalstep_last_logged: int,
        accumulated_metrics: Dict[str, Union[float, torch.Tensor]],
        progress_bar: tqdm,
        actor: DeepSpeedEngine,
        critic: Optional[DeepSpeedEngine],
    ):
        # Wait for all processes to reach this point
        dist.barrier()

        logs: Dict[str, float] = {}

        # Compute the log values over all processes
        num_steps_since_last_log = (
            self.state.global_step - _globalstep_last_logged
        ) * self.args.gradient_accumulation_steps

        if "_num_participating_tokens" in accumulated_metrics:
            num_participating_tokens = accumulated_metrics["_num_participating_tokens"]
            dist.all_reduce(num_participating_tokens, op=dist.ReduceOp.SUM)
            num_participating_tokens = num_participating_tokens.item()
        else:
            num_participating_tokens = 1

        for metric_name, metric_value in accumulated_metrics.items():
            if metric_name.startswith("_"):
                continue
            if metric_value is None:
                continue

            is_weighed_by_num_actions = (
                metric_name in self.log_keys_weighed_by_num_participating_tokens
            )

            if isinstance(metric_value, torch.Tensor):
                metric_value = metric_value.to(actor.device)
                dist.all_reduce(metric_value, op=dist.ReduceOp.SUM)
                metric_value = metric_value.item()
                divisor = dist.get_world_size()
            else:
                assert not is_weighed_by_num_actions
                divisor = 1

            if is_weighed_by_num_actions:
                metric_value /= num_participating_tokens
            else:
                metric_value /= divisor * num_steps_since_last_log

            logs[metric_name] = round(metric_value, 8)

        logs["actor/lr"] = self._get_learning_rate(actor)
        if critic is not None:
            logs["critic/lr"] = self._get_learning_rate(critic)

        logs["epoch"] = round(self.state.epoch, 4)
        logs["step"] = self.state.global_step
        logs["actor/ds_step"] = actor.global_steps
        if critic is not None:
            logs["critic/ds_step"] = critic.global_steps

        # First log the metrics on the progress bar
        progress_bar.set_postfix(logs)

        # Add "train/" prefix for clarity.
        logs = {f"train/{k}": v for k, v in logs.items()}

        self._cloud_log({**logs, "train/global_step": self.state.global_step})

        # Reset the accumulated metrics
        for key in accumulated_metrics.keys():
            accumulated_metrics[key] -= accumulated_metrics[key]

    def _update_metrics(
        self,
        running_metrics: Dict[str, Union[torch.Tensor, float]],
        accumulated_metrics: Dict[str, Union[torch.Tensor, float]],
        step_metrics: Dict[str, Union[torch.Tensor, float]],
    ):
        dist.barrier()

        def get_initial_value(
            val: Union[float, torch.Tensor]
        ) -> Union[float, torch.Tensor]:
            if isinstance(val, torch.Tensor):
                return torch.tensor(0.0, dtype=val.dtype, device=val.device)
            return 0.0

        # Initialize running metrics if not already initialized
        for key in step_metrics.keys():
            if key in accumulated_metrics:
                continue
            accumulated_metrics[key] = get_initial_value(step_metrics[key])
            if key in self.log_keys_to_store_in_running_metrics:
                if key not in running_metrics:
                    running_metrics[key] = get_initial_value(step_metrics[key])

        num_tokens = step_metrics["_num_participating_tokens"].item()

        for key, value in step_metrics.items():
            if value is None:
                continue

            if key in self.log_keys_weighed_by_num_participating_tokens:
                weight = num_tokens
            else:
                weight = 1

            value = value * weight
            accumulated_metrics[key] += value

        # Update Running Metrics
        running_metrics["_num_participating_tokens"] += num_tokens

    def _get_episodes_w_ref_logps(self, episodes: Dataset) -> Dataset:
        logger.info(f"Computing the reference log probabilities.")

        ds_w_ref_logprobs_path = (
            self.checkpoints_dir
            / f"episodes__iter{self.state.iteration:04d}"
            / "w_refLogp"
        )

        # Initialize and use the reference model to compute log probabilities for the dataset
        ref_engine = self._init_reference_model()
        t0 = time.time()
        aug_ds = self._update_episodes_with_log_probs(
            ref_engine, episodes, COLUMN_REF_SHIFTED_LOGPS
        )
        self._cloud_log(
            {
                "timing/train/computing_ref_logprobs": time.time() - t0,
                "train/global_step": self.state.global_step,
            }
        )

        if self._is_main_process():
            # We reread the dataset from disk afterward. This is done to
            # ensure that the dataset is memory-mapped and that the reference
            # log probs are not actually loaded in CPU memory.
            aug_ds.save_to_disk(ds_w_ref_logprobs_path)
        self.distributed_state.wait_for_everyone()

        self._destroy_reference_engine(ref_engine)
        del ref_engine
        del aug_ds
        release_memory()
        if not self.cache_deepspeed_engines:
            self.log_tensors_on_gpu()

        episodes = Dataset.load_from_disk(str(ds_w_ref_logprobs_path))
        return episodes

    def _get_episodes_w_curr_logps_and_values(
        self,
        episodes: Dataset,
        actor: DeepSpeedEngine,
        critic: Optional[DeepSpeedEngine] = None,
    ) -> Dataset:
        metrics = {}

        logger.info(f"Computing the current actor logprobs.")
        ds_w_actor_logps_path = (
            self.checkpoints_dir
            / f"episodes__iter{self.state.iteration:04d}"
            / "w_actLogp"
        )
        t0 = time.time()
        aug_ds = self._update_episodes_with_log_probs(
            actor, episodes, COLUMN_ACTOR_SHIFTED_LOGPS
        )
        metrics["timing/train/computing_actor_logprobs"] = time.time() - t0
        if self._is_main_process():
            aug_ds.save_to_disk(ds_w_actor_logps_path)
        self.distributed_state.wait_for_everyone()

        del aug_ds
        release_memory()

        episodes = Dataset.load_from_disk(str(ds_w_actor_logps_path))

        if critic is not None:
            logger.info(f"Computing the current critic values.")
            ds_w_actor_logps_and_values_path = (
                self.checkpoints_dir
                / f"episodes__iter{self.state.iteration:04d}"
                / "w_actLogp_and_values"
            )
            t0 = time.time()
            aug_ds = self._update_episodes_with_values(critic, episodes, COLUMN_VALUES)
            metrics["timing/train/computing_critic_values"] = time.time() - t0
            if self._is_main_process():
                aug_ds.save_to_disk(ds_w_actor_logps_and_values_path)
            self.distributed_state.wait_for_everyone()

            del aug_ds
            release_memory()

            episodes = Dataset.load_from_disk(str(ds_w_actor_logps_and_values_path))

        if len(metrics) > 0:
            self._cloud_log({**metrics, "train/global_step": self.state.global_step})

        return episodes

    def _update_episodes_with_log_probs(
        self,
        model_engine: Union[DeepSpeedEngine, PreTrainedModel],
        dataset: Dataset,
        column_name: str,
    ) -> Dataset:
        # Create a distributed data loader such that the order of
        # episodes is preserved when batched are distributed across multiple processes.
        data_loader = prepare_data_loader_for_inference(
            dataset,
            per_device_batch_size=self.args.per_device_train_batch_size,
            data_loader_kwargs={
                "collate_fn": PPODataCollator(),
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
            },
        )

        model_engine.eval()

        dist.barrier()

        list_of_log_probs = []
        for inputs in tqdm(
            data_loader, desc="Computing log probs", disable=not self._is_main_process()
        ):
            with torch.no_grad():
                # Assume every sequence is padded from the right
                # noinspection DuplicatedCode
                assert torch.all(inputs["attention_mask"][:, 0] == 1)
                assert (
                    inputs["input_ids"].shape[0]
                    == self.args.per_device_train_batch_size
                ), (
                    f"We expect on all processes to have the same batch size of "
                    f"{self.args.per_device_train_batch_size}."
                )

                inputs = {k: v.to(model_engine.device) for k, v in inputs.items()}

                # Compute the sequence lengths as we need to extract
                # the log probs of the non-padded tokens
                seq_lengths = inputs["attention_mask"].sum(dim=1).detach().clone()
                seq_lengths = seq_lengths.unsqueeze(1)  # Shape: (batch_size, 1)

                # Compute the log probabilities for each token
                outputs = self._forward_pass_actor(
                    model_engine, inputs, return_all_logps=True
                )
                logps = outputs["all_logps"].detach()
                assert logps.shape[1] == inputs["input_ids"].shape[1] - 1

                # Gather across all distributed processes
                # Note that after all_gather, the order in which the batches were
                # distributed across processes is preserved. So, concatenating
                # them along the batch dimension will give us the original order.
                seq_lengths = gather(seq_lengths).cpu()
                logps = gather(
                    pad_across_processes(logps, dim=1, pad_index=0.0, pad_first=False)
                ).cpu()

                assert (
                    logps.shape[0]
                    == inputs["input_ids"].shape[0] * dist.get_world_size()
                )

                # Convert 2d tensors to a list of lists
                logps_seq_lengths = seq_lengths - 1
                for i, seq_len in enumerate(logps_seq_lengths.squeeze().tolist()):
                    assert seq_len <= logps.shape[1]
                    list_of_log_probs.append(logps[i, :seq_len].tolist())

        # Remove any extra log probs that were added due to padding
        list_of_log_probs = list_of_log_probs[: len(dataset)]

        with self.distributed_state.main_process_first():
            dataset = dataset.add_column(name=column_name, column=list_of_log_probs)
        return dataset

    def _update_episodes_with_values(
        self,
        model_engine: Union[DeepSpeedEngine, PreTrainedModel],
        dataset: Dataset,
        column_name: str,
    ) -> Dataset:
        # Create a distributed data loader such that the order of
        # episodes is preserved when distributed across multiple processes.
        data_loader = prepare_data_loader_for_inference(
            dataset,
            per_device_batch_size=self.args.per_device_train_batch_size,
            data_loader_kwargs={
                "collate_fn": PPODataCollator(),
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
            },
        )

        model_engine.eval()

        dist.barrier()

        list_of_values = []
        for inputs in tqdm(
            data_loader, desc="Computing values", disable=not self._is_main_process()
        ):
            with torch.no_grad():
                # Assume every sequence is padded from the right
                # noinspection DuplicatedCode
                assert torch.all(inputs["attention_mask"][:, 0] == 1)
                assert (
                    inputs["input_ids"].shape[0]
                    == self.args.per_device_train_batch_size
                ), (
                    f"We expect on all processes to have the same batch size of "
                    f"{self.args.per_device_train_batch_size}."
                )

                inputs = {k: v.to(model_engine.device) for k, v in inputs.items()}

                # Compute the sequence lengths as we need to extract
                # the values of the non-padded tokens
                seq_lengths = inputs["attention_mask"].sum(dim=1).detach().clone()
                seq_lengths = seq_lengths.unsqueeze(1)

                # Compute the values for each token
                outputs = self._forward_pass_critic(model_engine, inputs)
                values = outputs["values"].detach()
                assert values.shape[1] == inputs["input_ids"].shape[1]

                # Gather across all distributed processes
                seq_lengths = gather(seq_lengths).cpu()
                values = gather(
                    pad_across_processes(values, dim=1, pad_index=0.0, pad_first=False)
                ).cpu()

                assert (
                    values.shape[0]
                    == inputs["input_ids"].shape[0] * dist.get_world_size()
                )

                # Convert 2d tensors to a list of lists
                for i, seq_len in enumerate(seq_lengths.squeeze().tolist()):
                    assert seq_len <= values.shape[1]
                    list_of_values.append(values[i, :seq_len].tolist())

        # Remove any extra values that were added due to padding
        list_of_values = list_of_values[: len(dataset)]

        with self.distributed_state.main_process_first():
            dataset = dataset.add_column(name=column_name, column=list_of_values)
        return dataset

    def _forward_pass_actor(
        self,
        model_engine: Union[DeepSpeedEngine, PreTrainedModel],
        inputs: Dict[str, torch.Tensor],
        return_logits: bool = False,
        return_sequence_logp: bool = False,
        return_all_logps: bool = False,
        return_entropy: bool = False,
        sequence_logp_reduction: Optional[Literal["mean"]] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass for the model.

        Args:
            model_engine (Union[DeepSpeedEngine, PreTrainedModel]): The model to forward pass.
            inputs (Dict[str, torch.Tensor]): The inputs to the model, containing the following keys:
                - "input_ids": The input ids of the sequence.
                - "labels": The labels for the sequence.
                - "attention_mask": The attention mask of the sequence.
        Returns:
            Dict[str, Any]: The outputs containing the following keys:
                - "logits": The logits for the sequence.
                - "logps": The log probabilities of the sequence.
                - "all_logps": The log probabilities of all tokens in the sequence.
        """
        input_ids: torch.Tensor = inputs["input_ids"]
        labels: torch.Tensor = inputs["labels"]
        attention_mask: torch.Tensor = inputs["attention_mask"]

        outputs: CausalLMOutputWithPast = model_engine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        )

        logits = outputs.logits.float()  # Shape: (batch_size, max_seq_len, vocab_size)
        logits /= self.ppo_hparams.temperature

        # Shift so that tokens < n predict n
        # noinspection DuplicatedCode
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_label_mask = (shift_labels != -100).to(shift_logits.dtype)

        # Make sure all label indices are valid. i.e. convert -100 to 0
        shift_labels[shift_labels == -100] = 0

        log_probs = shift_logits.log_softmax(-1)
        per_token_log_probs = torch.gather(
            log_probs, dim=2, index=shift_labels.unsqueeze(2)
        )
        per_token_log_probs = per_token_log_probs.squeeze(2)

        # Multiply the log probs by the label mask to ignore the padding labels
        per_token_log_probs = per_token_log_probs * shift_label_mask

        output = {}
        if return_entropy:
            with torch.no_grad():
                mean_entropy = masked_mean(
                    entropy_from_logits(shift_logits.detach()), shift_label_mask
                )
                mean_entropy = mean_entropy.detach().clone()
                output["entropy"] = mean_entropy

        if return_logits:
            output["logits"] = logits

        if return_sequence_logp:
            sequence_log_probs = per_token_log_probs.sum(dim=-1)
            if sequence_logp_reduction == "mean":
                sequence_log_probs = sequence_log_probs / shift_label_mask.sum(dim=-1)
            output["sequence_logp"] = sequence_log_probs

        if return_all_logps:
            output["all_logps"] = per_token_log_probs

        return output

    def _forward_pass_critic(
        self,
        model_engine: Union[DeepSpeedEngine, PreTrainedModel],
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the critic model.

        Args:
            model_engine (Union[DeepSpeedEngine, PreTrainedModel]): The model to forward pass.
            inputs (Dict[str, torch.Tensor]): The inputs to the model, containing the following keys:
                - "input_ids": The input ids of the sequence.
                - "labels": The labels for the sequence.
                - "attention_mask": The attention mask of the sequence.
        Returns:
            Dict[str, torch.Tensor]: The outputs containing the following keys:
                - "values": The values of the sequence.
                - "value_mask": The mask for the values.
        """
        input_ids: torch.Tensor = inputs["input_ids"]
        attention_mask: torch.Tensor = inputs["attention_mask"]
        labels: Optional[torch.Tensor] = inputs.get("labels", None)

        outputs = model_engine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,  # We don't need the cache for training
        )

        predicted_values = outputs  # Shape: (batch_size, max_seq_len)

        # We convert the values to float32 to avoid any precision issues
        predicted_values = predicted_values.to(torch.float32)

        # Since we are only interested in the values of the response tokens,
        # we need to mask out the query tokens.
        if labels is not None:
            # noinspection PyUnresolvedReferences
            value_mask = (labels != -100).to(attention_mask.dtype)
        else:
            # Only mask the pad tokens if the labels are not provided
            value_mask = attention_mask

        return {"values": predicted_values, "value_mask": value_mask}

    def _init_reference_model(
        self, only_return_unwrapped_model: bool = False
    ) -> Union[DeepSpeedEngine, PreTrainedModel]:
        if hasattr(self, "_reference_engine"):
            reference_model = (
                self._reference_engine.module
                if isinstance(self._reference_engine, DeepSpeedEngine)
                else self._reference_engine
            )
            reference_model.to(self.distributed_state.device)
            return self._reference_engine

        this_process_device = self.distributed_state.device

        metrics = {}
        t0 = time.time()

        # Load the reference model into GPU
        cache_path = self.temp_checkpoint_dir / ".reference"
        if not cache_path.exists():
            cache_path = None
        # noinspection PyTypeChecker
        reference_model: PreTrainedModel = self.reference_lazy.construct(
            device=this_process_device,
            disable_dropout=True,
            runtime_hf_model_name=cache_path,
        )
        reference_model.eval()
        metrics["timing/reference/construct"] = time.time() - t0
        if (
            self.cache_reference_model
            and cache_path is None
            and self._is_main_process()
        ):
            # Since the reference model is used in every iteration, it makes
            # sense to cache it on the fast disk to avoid loading it from network storage
            reference_model.save_pretrained(
                self.temp_checkpoint_dir / ".reference", safe_serialization=False
            )

        if only_return_unwrapped_model:
            self._cloud_log({**metrics, "train/global_step": self.state.global_step})
            if self.cache_deepspeed_engines:
                self._reference_engine = reference_model

            return reference_model

        # Using a DeepSpeed engine for inference is useful for models that are
        # too large to fit in one GPU. DeepSpeed will automatically split
        # the model across multiple GPUs.
        ds_config = HfTrainerDeepSpeedConfig(self.reference_deepspeed_config)
        self._patch_ds_config_for_batch_size(
            ds_config, self.args, self.global_batch_size
        )
        self._patch_ds_config_for_dtype(ds_config, self.args)
        self._patch_ds_config_for_bucket_size(ds_config, reference_model.config)

        engine = self._initialize_deepspeed_engine_for_inference(
            model=reference_model, deepspeed_config=ds_config.config
        )
        engine.eval()

        if self.cache_deepspeed_engines:
            self._reference_engine = engine

        return engine

    def _init_actor_model(
        self, only_return_unwrapped_model: bool = False
    ) -> Union[DeepSpeedEngine, PreTrainedModel]:
        if hasattr(self, "_actor_engine"):
            return self._actor_engine

        logger.info(f"Creating the actor deepspeed engine...")

        this_process_device = self.distributed_state.device

        metrics = {}
        t0 = time.time()

        # Load the actor model into GPU
        # noinspection PyTypeChecker
        actor_model: PreTrainedModel = self.actor_lazy.construct(
            device=this_process_device,
            disable_dropout=True,
        )
        metrics["timing/actor/construct"] = time.time() - t0
        if only_return_unwrapped_model:
            self._cloud_log({**metrics, "train/global_step": self.state.global_step})
            return actor_model

        if self.args.gradient_checkpointing:
            actor_model.gradient_checkpointing_enable()

        t0 = time.time()

        ds_config = HfTrainerDeepSpeedConfig(self.actor_deepspeed_config)

        # Create the optimizer
        has_optimizer = ds_config.get_value("optimizer", None) is not None
        if has_optimizer:
            weight_decay = ds_config.get_value("optimizer.params.weight_decay", 0.0)
            if weight_decay == "auto":
                weight_decay = self.args.weight_decay

            optimizer = self.create_optimizer(actor_model, weight_decay)
        else:
            optimizer = None

        # Create the LR scheduler
        # noinspection DuplicatedCode
        has_deepspeed_scheduler = ds_config.get_value("scheduler", None) is not None
        warmup_steps = self.args.get_warmup_steps(self.total_num_training_steps)
        if has_deepspeed_scheduler:
            lr_scheduler = None
            self._patch_ds_config_for_lr_scheduler(
                ds_config,
                total_num_training_steps=self.total_num_training_steps,
                warmup_steps=warmup_steps,
                learning_rate=self.args.learning_rate,
            )
        elif self.args.lr_scheduler_type is not None:
            logger.info("Using non-DeepSpeed LR scheduler.")
            lr_scheduler = self.create_lr_scheduler(
                optimizer,
                name=self.args.lr_scheduler_type,
                warmup_steps=warmup_steps,
                num_training_steps=self.total_num_training_steps,
            )
        else:
            lr_scheduler = None

        self._patch_ds_config_for_optimizer(ds_config, self.args)
        self._patch_ds_config_for_batch_size(
            ds_config, self.args, self.global_batch_size
        )
        self._patch_ds_config_for_dtype(ds_config, self.args)
        self._patch_ds_config_for_bucket_size(ds_config, actor_model.config)

        engine = self._initialize_deepspeed_engine_for_training(
            actor_model,
            deepspeed_config=ds_config.config,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

        metrics["timing/actor/deepspeed_init"] = time.time() - t0
        self._cloud_log({**metrics, "train/global_step": self.state.global_step})

        if self.cache_deepspeed_engines:
            self._actor_engine = engine

        return engine

    def _init_critic_model(
        self,
        only_return_unwrapped_model: bool = False,
        hf_checkpoint_path: Optional[Path] = None,
    ) -> Union[DeepSpeedEngine, PreTrainedModel]:
        if hasattr(self, "_critic_engine"):
            return self._critic_engine

        logger.info(f"Creating the critic deepspeed engine...")

        this_process_device = self.distributed_state.device

        metrics = {}
        t0 = time.time()

        # noinspection PyTypeChecker
        critic_model: PreTrainedModel = self.critic_lazy.construct(
            device=this_process_device,
        )
        metrics["timing/critic/construct"] = time.time() - t0
        if hf_checkpoint_path is not None:
            assert (hf_checkpoint_path / "pytorch_model.bin").exists()
            critic_model.load_state_dict(
                torch.load(hf_checkpoint_path / "pytorch_model.bin")
            )
            critic_model.to(this_process_device)

        if only_return_unwrapped_model:
            critic_model.to(this_process_device)
            self._cloud_log({**metrics, "train/global_step": self.state.global_step})
            return critic_model

        # noinspection DuplicatedCode
        if self.args.gradient_checkpointing:
            critic_model.gradient_checkpointing_enable()

        t0 = time.time()

        ds_config = HfTrainerDeepSpeedConfig(self.critic_deepspeed_config)

        # Create the optimizer
        has_optimizer = ds_config.get_value("optimizer", None) is not None
        if has_optimizer:
            weight_decay = ds_config.get_value("optimizer.params.weight_decay", 0.0)
            if weight_decay == "auto":
                weight_decay = self.args.weight_decay

            optimizer = self.create_optimizer(critic_model, weight_decay)
        else:
            optimizer = None

        # Create the LR scheduler
        # noinspection DuplicatedCode
        has_deepspeed_scheduler = ds_config.get_value("scheduler", None) is not None
        warmup_steps = self.args.get_warmup_steps(self.total_num_training_steps)
        if has_deepspeed_scheduler:
            lr_scheduler = None
            self._patch_ds_config_for_lr_scheduler(
                ds_config,
                total_num_training_steps=self.total_num_training_steps,
                warmup_steps=warmup_steps,
                learning_rate=self.args.learning_rate,
            )
        elif self.args.lr_scheduler_type is not None:
            logger.info("Using non-DeepSpeed LR scheduler.")
            lr_scheduler = self.create_lr_scheduler(
                optimizer,
                name=self.args.lr_scheduler_type,
                warmup_steps=warmup_steps,
                num_training_steps=self.total_num_training_steps,
            )
        else:
            lr_scheduler = None

        self._patch_ds_config_for_optimizer(ds_config, self.args)
        self._patch_ds_config_for_batch_size(
            ds_config, self.args, self.global_batch_size
        )
        self._patch_ds_config_for_dtype(ds_config, self.args)
        self._patch_ds_config_for_bucket_size(ds_config, critic_model.config)

        engine = self._initialize_deepspeed_engine_for_training(
            critic_model,
            deepspeed_config=ds_config.config,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

        metrics["timing/critic/deepspeed_init"] = time.time() - t0

        if self.cache_deepspeed_engines:
            self._critic_engine = engine

        return engine

    def load_checkpoint(self, checkpoint: Union[Checkpoint, Path]) -> None:
        super().load_checkpoint(checkpoint)
        checkpoint_path = (
            checkpoint if isinstance(checkpoint, Path) else checkpoint.path
        )
        self._load_training_state(checkpoint_path)
        self.checkpoint_path_to_load = checkpoint_path

    def _load_checkpoint_to_ds_engines(
        self,
        checkpoint_path: Path,
        actor: Optional[DeepSpeedEngine] = None,
        critic: Optional[DeepSpeedEngine] = None,
    ) -> None:
        metrics = {}
        if actor is not None:
            t0 = time.time()
            actor.load_checkpoint(str(checkpoint_path / "actor"))
            metrics["timing/actor/load_checkpoint"] = time.time() - t0
        if critic is not None and not self.disable_critic_training:
            t0 = time.time()
            critic.load_checkpoint(str(checkpoint_path / "critic"))
            metrics["timing/critic/load_checkpoint"] = time.time() - t0

        if len(metrics) > 0:
            self._cloud_log({**metrics, "train/global_step": self.state.global_step})

    def _save_hf_pretrained(self, engine: DeepSpeedEngine, path: Path) -> None:
        if self._is_main_process():
            # Only save on the main process
            assert engine.zero_optimization_stage() < 3
            logger.info(f"Saving HF pretrained weights to {path}")
            unwrapped_model = engine.module
            unwrapped_model.save_pretrained(path, safe_serialization=False)
        dist.barrier()

    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        actor: Optional[DeepSpeedEngine] = None,
        critic: Optional[DeepSpeedEngine] = None,
    ) -> None:
        if self._is_main_process():
            if checkpoint_path.exists():
                logger.warning(
                    f"Checkpoint path {checkpoint_path} already exists. Overwriting."
                )
                shutil.rmtree(checkpoint_path)

            checkpoint_path.mkdir(parents=True, exist_ok=True)

        self._save_trainer_state(checkpoint_path)

        metrics = {}
        if actor is not None:
            t0 = time.time()
            self._save_hf_pretrained(actor, checkpoint_path / "hf_pretrained")
            metrics["timing/actor/save_hf_pretrained"] = time.time() - t0

            t0 = time.time()
            actor.save_checkpoint(str(checkpoint_path / "actor"))
            metrics["timing/actor/save_checkpoint"] = time.time() - t0

        if critic is not None and not self.disable_critic_training:
            if self.save_hf_critic_checkpoint:
                t0 = time.time()
                self._save_hf_pretrained(
                    critic, checkpoint_path / "critic" / "hf_pretrained"
                )
                metrics["timing/critic/save_hf_pretrained"] = time.time() - t0

            t0 = time.time()
            critic.save_checkpoint(str(checkpoint_path / "critic"))
            metrics["timing/critic/save_checkpoint"] = time.time() - t0

        if len(metrics) > 0:
            self._cloud_log({**metrics, "train/global_step": self.state.global_step})

    @staticmethod
    def is_checkpoint_resumable(checkpoint_path: Path) -> bool:
        if not checkpoint_path.exists():
            return False

        if not (checkpoint_path / "actor").exists():
            return False

        model_only_file_names = ["hf_pretrained", "pytorch_model.bin", "pytorch_model"]
        non_model_files = [
            file
            for file in (checkpoint_path / "actor").iterdir()
            if file.name not in model_only_file_names
        ]
        if len(non_model_files) == 0:
            return False

        if (checkpoint_path / "critic").exists():
            non_model_files = [
                file
                for file in (checkpoint_path / "critic").iterdir()
                if file.name not in model_only_file_names
            ]
            if len(non_model_files) == 0:
                return False

        return True

    def clean_checkpoints(self, exclude: Optional[List[Path]] = None) -> None:
        if exclude is None:
            exclude = []

        # Remove unnecessary checkpoints
        for checkpoint in self.checkpoints_dir.iterdir():
            if (
                checkpoint.is_dir()
                and checkpoint.name.startswith("ckpt--")
                and checkpoint not in exclude
            ):
                if self.args.checkpoint_keep_steps is not None:
                    checkpoint_iteration = self.parse_checkpoint_name(checkpoint.name)[
                        0
                    ]
                    if checkpoint_iteration % self.args.checkpoint_keep_steps == 0:
                        continue

                    logger.info(f"Removing checkpoint {checkpoint}")
                    shutil.rmtree(checkpoint)

        self.clean_non_model_checkpoints(exclude=exclude)

    def clean_non_model_checkpoints(self, exclude: Optional[List[Path]] = None) -> None:
        """
        Clean all optimizer and scheduler states which are not needed for evaluation.
        """
        if exclude is None:
            exclude = []

        def clean_deepspeed_checkpoint(path: Path) -> List[str]:
            removed_files_and_dirs = []
            for file in path.iterdir():
                if file.name not in ["hf_pretrained", "pytorch_model.bin"]:
                    removed_files_and_dirs.append(file.name)
                    if file.is_dir():
                        shutil.rmtree(file, ignore_errors=True)
                    else:
                        file.unlink()
            return removed_files_and_dirs

        for checkpoint in self.checkpoints_dir.iterdir():
            if (
                checkpoint.is_dir()
                and checkpoint.name.startswith("ckpt--")
                and checkpoint not in exclude
            ):
                logger.info(f"Removing non-model files from checkpoint {checkpoint}")

                # Remove everything except the `hf_pretrained` folder
                all_removed_files_and_dirs = []
                all_removed_files_and_dirs += clean_deepspeed_checkpoint(
                    checkpoint / "actor"
                )
                if (checkpoint / "critic").exists():
                    all_removed_files_and_dirs += clean_deepspeed_checkpoint(
                        checkpoint / "critic"
                    )

                logger.info(f"Removed files and dirs: {all_removed_files_and_dirs}")

    def _clean_episodes(self, exclude: List[str]) -> None:
        if not need_to_minimize_stored_files():
            return

        if self._is_main_process():
            keep_iterations = [
                int(self.parse_checkpoint_name(ckpt.name)[0])
                for ckpt in self.checkpoints_dir.iterdir()
                if ckpt.is_dir() and ckpt.name.startswith("ckpt--")
            ]
            keep_iterations += [0]  # Always keep the initial iteration
            keep_iterations += [
                int(self.parse_checkpoint_name(name)[0]) for name in exclude
            ]
            keep_iterations = set(keep_iterations)

            # Remove unnecessary episodes
            for episode in self.checkpoints_dir.glob("episodes__iter*"):
                if not episode.is_dir():
                    continue

                episode_iter = int(episode.name.split("__iter")[1])
                if episode_iter in keep_iterations:
                    continue

                logger.info(
                    f"Removing episode {episode.name}; "
                    f"excluding iterations: {keep_iterations}"
                )
                shutil.rmtree(episode, ignore_errors=True)

            # Remove unnecessary episodes insided experiment_root
            for episode in self.experiment_root.glob("episodes/episodes_*"):
                if not episode.is_dir():
                    continue

                episode_iter = int(episode.name.split("_")[1])
                if episode_iter in keep_iterations:
                    continue

                logger.info(
                    f"Removing exp_root/episode {episode.name}; "
                    f"excluding iterations: {keep_iterations}"
                )
                shutil.rmtree(episode, ignore_errors=True)

            # Remove unnecessary temp_episodes
            for episode in self.experiment_root.glob("temp_episodes/iteration__*"):
                if not episode.is_dir():
                    continue

                episode_iter = int(episode.name.split("__")[1])
                if episode_iter in keep_iterations:
                    continue

                logger.info(
                    f"Removing temp episode {episode.name}; "
                    f"excluding iterations: {keep_iterations}"
                )
                shutil.rmtree(episode, ignore_errors=True)

        dist.barrier()

    def _save_trainer_state(self, checkpoint_path: Path) -> None:
        super()._save_trainer_state(checkpoint_path)
        if self._is_main_process():
            save_custom_state(self.running_scores, checkpoint_path, index=10)
            save_custom_state(self.kl_ctl, checkpoint_path, index=11)

    def _load_training_state(self, checkpoint_path: Path) -> None:
        super()._load_training_state(checkpoint_path)
        load_custom_state(self.running_scores, checkpoint_path, index=10)
        load_custom_state(self.kl_ctl, checkpoint_path, index=11)

    def save_final_checkpoint(self) -> None:
        last_checkpoint_path = self.get_last_checkpoint().path
        final_checkpoint_path = self.checkpoints_dir / "final"
        final_checkpoint_path.write_text(last_checkpoint_path.name)

    def _copy_to_permanent_storage(self, checkpoint_path: Path) -> None:
        if not self._is_main_process():
            return

        permanent_storage_path = self.checkpoints_dir
        copy_cmd = f"cp -r {checkpoint_path} {permanent_storage_path}/"
        logger.info(f"Copying checkpoint to permanent storage: {copy_cmd}")

        # Start the copy in background
        subprocess.Popen(copy_cmd, shell=True)

    def _clean_old_temp_checkpoints(self, exclude: Optional[List[Path]] = None) -> None:
        if exclude is None:
            exclude = []

        if self._is_main_process():
            for checkpoint in self.temp_checkpoint_dir.iterdir():
                if (
                    checkpoint.is_dir()
                    and checkpoint.name.startswith("ckpt--")
                    and checkpoint not in exclude
                ):
                    logger.info(f"Removing old temp checkpoint {checkpoint}")
                    shutil.rmtree(checkpoint)

        dist.barrier()

    def _destroy_ds_engine(self, ds_engine: DeepSpeedEngine):
        if self.cache_deepspeed_engines:
            return
        super()._destroy_ds_engine(ds_engine)

    def _destroy_reference_engine(
        self, model_or_engine: Union[PreTrainedModel, DeepSpeedEngine]
    ):
        if self.cache_deepspeed_engines:
            if self.move_reference_model_to_cpu:
                model = (
                    model_or_engine.module
                    if isinstance(model_or_engine, DeepSpeedEngine)
                    else model_or_engine
                )
                model.to("cpu")
            return

        if isinstance(model_or_engine, DeepSpeedEngine):
            super()._destroy_ds_engine(model_or_engine)

    def _filter_episodes(self, episodes_dataset: Dataset) -> Dataset:
        """
        Filter out episodes that are too long.
        """
        if self.args.max_seq_len is not None:
            max_seq_len = self.args.max_seq_len
            orig_len = len(episodes_dataset)

            def filter_fn(example):
                return (
                    len(example["query_token_ids"]) + len(example["response_token_ids"])
                    <= max_seq_len
                )

            with self.distributed_state.main_process_first():
                episodes_dataset = episodes_dataset.filter(filter_fn, desc="Filtering")

            logger.error(
                f"Filtered out {orig_len - len(episodes_dataset)} episodes "
                f"that are too long. Remaining: {len(episodes_dataset)}"
            )

        return episodes_dataset

    def _check_overflow(self, actor: DeepSpeedEngine):
        assert actor.bfloat16_enabled()
        if hasattr(actor.optimizer, "check_overflow"):
            assert (
                not actor.optimizer.check_overflow()
            ), "We don't expect overflow in BF16 training"


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.value = state_dict["value"]


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.value = state_dict["value"]
