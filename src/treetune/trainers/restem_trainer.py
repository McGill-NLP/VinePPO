# REST(EM) Trainer, based on the paper "Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models" https://arxiv.org/abs/2312.06585
import json
import logging
import os
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

from treetune.common import JsonDict, Lazy, Params
from treetune.common.deepspeed_utils import (
    prepare_data_loader_for_inference,
    prepare_data_loader_for_training,
)
from treetune.common.vllm_server import VLLMServer
from treetune.common.wandb_utils import get_repo_dir
from treetune.inference_pipelines import InferencePipeline
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
)
from treetune.tokenization_utils.base_tokenizer import Tokenizer

logger = get_logger(__name__)

@Trainer.register("restem")
class RestEMTrainer(DeepSpeedPolicyTrainer):
    def __init__(
        self,
        num_epochs_per_iteration: int,
        sampling_temperature: float,
        actor_model: Lazy[Model],   # the model we generate samples from, but keep in mind we train from base model each iteration.
        actor_deepspeed_config: JsonDict,
        general_training_args: JsonDict,
        reference_model: Optional[Lazy[Model]] = None,
        reference_deepspeed_config: Optional[JsonDict] = None,
        align_skipping_on_overflow: bool = True,
        cache_reference_model_on_temp_storage: bool = False,
        temp_checkpoint_dir: Optional[str] = None,
        profile_torch_memory: bool = False,
        cache_deepspeed_engines: bool = False,
        move_reference_model_to_cpu: bool = False,
        kl_penalty_loss_type: Optional[Literal["kl", "abs", "mse", "control_variate"]] = None,
        early_stop_vllm_server: Lazy[VLLMServer] = None,  # we need this to choose the best checkpoint in each iteration
        early_stop_inference_pipeline_cfg: JsonDict = None,
        early_stop_tokenizer: Tokenizer = None,
        early_stop_method: Literal['restem_paper_original', 'choose_best'] = 'restem_paper_original',
         **kwargs,
    ):
        # pop num_episodes_per_iteration from kwargs
        kwargs.pop("num_episodes_per_iteration", None)
        super().__init__(**kwargs)
        self.kl_penalty_loss_type = kl_penalty_loss_type
        self._set_process_log_level(logger)

        assert sampling_temperature > 0, "Temperature should be positive."
        self.sampling_temperature = sampling_temperature
        self.args = TrainingArguments(**general_training_args)
        self.align_skipping_on_overflow = align_skipping_on_overflow

        self.num_epochs = num_epochs_per_iteration
        self._compute_batch_size_and_steps()

        self.actor_lazy = actor_model
        self.actor_deepspeed_config = actor_deepspeed_config

        self.reference_lazy = reference_model
        self.reference_deepspeed_config = reference_deepspeed_config
        if self.reference_lazy is None:
            logger.info("No reference model provided. We then provide no KL estimation.")

        # This operation is done on the same data across all processes
        # So, there is no need to synchronize the operation
        self.running_scores = DeepSpeedRunningMoments(force_no_sync=True)

        from deepspeed.utils import logger as ds_logger

        ds_logger.setLevel(logging.DEBUG)

        self.checkpoint_path_to_load = None

        if temp_checkpoint_dir is not None:
            self.temp_checkpoint_dir = Path(temp_checkpoint_dir)
        else:
            self.temp_checkpoint_dir = get_repo_dir() / "temp_restem_checkpoints"
            logger.info(
                f"No temporary checkpoint directory provided. Using {self.temp_checkpoint_dir}"
            )
        self.temp_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.cache_reference_model = cache_reference_model_on_temp_storage
        self.profile_torch_memory = profile_torch_memory
        self.cache_deepspeed_engines = cache_deepspeed_engines
        self.move_reference_model_to_cpu = move_reference_model_to_cpu
        self.early_stop_vllm_server = early_stop_vllm_server
        self.early_stop_inference_pipeline_cfg = early_stop_inference_pipeline_cfg
        self.early_stop_tokenizer = early_stop_tokenizer
        self.early_stop_method = early_stop_method

        if self._is_main_process():
            if getattr(self.cloud_logger, "define_metric", None):
                self.cloud_logger.define_metric("train/global_iteration")
                self.cloud_logger.define_metric(
                    "episodes_metric/*",
                    step_metric="train/global_iteration",
                    step_sync=True,
                )

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

        logger.info(f"Per device batch size: {self.args.per_device_train_batch_size}")
        logger.info(
            f"Gradient accumulation steps: {self.args.gradient_accumulation_steps}"
        )
        logger.info(f"Num of total processes: {self.distributed_state.num_processes}")
        logger.info(
            f"Global batch size (w. parallel, distributed & accumulation): {self.global_batch_size}"
        )

    def get_models(
        self,
    ) -> WeakValueDictionary[str, Union[PreTrainedModel, DeepSpeedEngine]]:
        weak_dict = WeakValueDictionary()
        if getattr(self, "_actor_engine", None) is not None:
            weak_dict["actor"] = self._actor_engine

        if getattr(self, "_reference_engine", None) is not None:
            weak_dict["reference"] = self._reference_engine

        return weak_dict

    def _is_kl_penalty_enabled(self) -> bool:
        return True

    def step(self, episodes_dataset: Dataset) -> Optional[Path]:
        """
        Perform a single step of RestEM training which means multiple epochs on the successful trajectories.
        Distributed Note:
            This function is called on each process. i.e., they all receive a full copy of the episodes_dataset.

        Considering our conservative memory approach, here is a general idea of each step:
        1. Initialize (load into CPU/GPU) the reference model if it exists.
        2. Compute the reference log probabilities if the reference model exists.
        3. Remove the reference model from the memory.
        4. Initialize (load into CPU/GPU ) the actor model (a fresh copy of the reference model) and its optimizer
        5. Train the actor with MLE on the episodes.
        6. Save the new actor for sampling in next step( which is the same as iteration here).
        7. Remove the actor from memory. (Including the optimizer states)
        8. Now, evaluate all checkpoints to check the best according to the evaluation metric (RestEM early stops, we emulate it this way)
        9. (Outside of this function) Generate new episodes by sampling from the actor.
        10. (Outside of this function) go back to step 1.

        Args:
            episodes_dataset (Dataset):
                A HuggingFace Dataset containing the episodes to train on.
                It should have the following columns:
                    - "query_token_ids": The token ids of the query.
                    - "response_token_ids": The token ids of the response.
                    - "score": The reward of the response (single scalar per response)

        Returns:
            Optional[Path]:
                The path to the latest policy (actor) checkpoint.
        """
        episodes_dataset = self._filter_episodes(episodes_dataset)


        self.total_num_training_steps = (
            self.num_epochs
            * len(episodes_dataset)
            // self.global_batch_size
        )
        logger.info(
            f"Total number of training steps (Gradient Updates): {self.total_num_training_steps}"
        )

        # Initialize the actor model with its optimizer
        logger.info("Initializing the actor model.")
        actor_engine = self._init_actor_model()

        # Load from checkpoint if specified
        if self.checkpoint_path_to_load is not None:
            logger.info(f"(@@-o)Loading checkpoint to continue from {self.checkpoint_path_to_load}...")
            self._load_checkpoint_to_ds_engines(
                self.checkpoint_path_to_load, actor_engine
            )
            self.checkpoint_path_to_load = None

        #
        # checkpointing when epoch is zero
        if self.state.epoch == 0:
            temp_ckpt_path = self.temp_checkpoint_dir / self._get_automatic_checkpoint_name()
            self._save_checkpoint(temp_ckpt_path, actor_engine)
            if self.args.save_steps != -1:
                permanent_ckpt_path = self._copy_to_permanent_storage(temp_ckpt_path) # this is a non-blocking copy
            self._clean_old_temp_checkpoints(exclude=[temp_ckpt_path])
        else:
            logger.info(f"(@@-o) The run was terminated in the middle of an iteration. we continue training.")

        if self.state.epoch == self.num_epochs:
            logger.info(f"(@@-o) The checkpoint is final, no further training should be done. The job should have been killed either in determining the best checkpoint or when generating new episodes.")
        else:
            # Compute or reload from disk the episodes with current actor log probabilities
            if self._is_kl_penalty_enabled():
                # Compute or reload from disk the episodes with reference log probabilities
                # It takes care initializing and destroying the reference model
                episodes_dataset = self._get_episodes_w_ref_logps(episodes_dataset)
            episodes_dataset = self._get_episodes_w_curr_logps(
                episodes_dataset, actor_engine
            )
            # Train the actor and critic models using restem trainer
            self._train_actor(episodes_dataset, actor_engine)

            # checkpointing for the last epoch
            # sometimes due to floating point errors, the epoch might not be exactly the last epoch, but it should be close enough, rounding it to 2 decimal points
            logger.info("(ROUND)before rounding the epoch: %s", self.state.epoch)
            self.state.epoch = round(self.state.epoch, 2)
            logger.info("(ROUND)after rounding the epoch: %s", self.state.epoch)
            assert self.state.epoch == self.num_epochs, f"The epoch should be the last epoch. but it is {self.state.epoch}"
            temp_ckpt_path = self.temp_checkpoint_dir / self._get_automatic_checkpoint_name()
            self._save_checkpoint(temp_ckpt_path, actor_engine)
            if self.args.save_steps != -1:
                permanent_ckpt_path = self._copy_to_permanent_storage(temp_ckpt_path)
            self._clean_old_temp_checkpoints(exclude=[temp_ckpt_path])

        # Clean up models and their optimizers from memory
        see_memory_usage("Before cleaning up deepspeed from memory", force=True)
        self._destroy_ds_engine(actor_engine)
        del actor_engine
        release_memory()
        see_memory_usage("After cleaning up deepspeed from memory", force=True)
        if not self.cache_deepspeed_engines:
            self.log_tensors_on_gpu()

        # evaluate the checkpoints, return the best one
        path_to_best_policy = self.evaluate_checkpoints() / "hf_pretrained"

        self.state.iteration += 1  # important to increment the iteration number after saving the last checkpoint, also after evaluation
        # as the evaluation uses this to filter the checkpoints of this iteration
        self.state.epoch = 0.0

        return path_to_best_policy

    def get_checkpoints_of_this_iteration(self):
        from treetune.runtime import PolicyIterationRuntime
        ckpts = PolicyIterationRuntime._get_list_of_evaluation_checkpoints(self.checkpoints_dir, every_n_checkpoints=1, ignore_worker_vars=True)
        logger.info("number of all checkpoints: %d", len(ckpts))
        # filer to get just current iteration
        filtered_ckpts = []
        for ckpt in ckpts:
            iteration = self.parse_checkpoint_name(ckpt.name)[0]  # (iteration, epoch, step)[0]
            if iteration == self.state.iteration:
                filtered_ckpts.append(ckpt)
        ckpts = filtered_ckpts
        logger.info("number of checkpoints in this iteration: %d", len(ckpts))
        logger.info("checkpoints in this iteration: %s", '\n'.join([ckpt.name for ckpt in ckpts]))

        return ckpts

    def evaluate_checkpoints(
        self,
    ):
        chosen_checkpoint = None # so the broadcast works
        if self._is_main_process():
            checkpoints = self.get_checkpoints_of_this_iteration()

            # (@@-o) check if this is already done and there is a chosen checkpoint
            chosen_checkpoints = []
            for ckpt in checkpoints:
                if (ckpt / "chosen_checkpoint").exists():
                    chosen_checkpoints.append(ckpt)

            if len(chosen_checkpoints) > 1:
                raise ValueError(f"More than one chosen checkpoint in this iteration: {chosen_checkpoints}")
            if len(chosen_checkpoints) == 1:
                chosen_checkpoint = chosen_checkpoints[0]
                logger.info(f"(@@-o) Chosen checkpoint from previous iteration: {chosen_checkpoint}. skipping evaluating again.")

            else:

                assert len(chosen_checkpoints) == 0, "There should be no chosen checkpoints after this line."

                es_accuracies = []
                for ckpt in checkpoints:
                    logger.info("Evaluating checkpoint for RestEM early stopping: %s", ckpt)
                    vllm_ckpt_dir = ckpt / "hf_pretrained"
                    assert vllm_ckpt_dir.exists(), f"Checkpoint directory {vllm_ckpt_dir} does not exist."
                    vllm_server = self.early_stop_vllm_server.construct(
                        seed=self.args.seed # todo: figure how to path the checkpoint dir
                    )

                    eval_dir = self.experiment_root / "restem_inside_evaluations"
                    eval_dir.mkdir(parents=True, exist_ok=True)

                    logs_dir = eval_dir / "logs"
                    logs_dir.mkdir(parents=True, exist_ok=True)

                    vllm_log_file = logs_dir / f"vllm_eval_{ckpt.name}.log"
                    vllm_log_file.touch()

                    logger.info(f"(EARLY-STOP)Starting VLLM server with log file {vllm_log_file}")
                    # save tokenizer there
                    self.early_stop_tokenizer.save_pretrained(vllm_ckpt_dir)

                    # wait for memory release, cause otherwise Cude Out of Memory =(
                    release_memory()
                    from treetune.common.gpu_utils import wait_for_memory_release
                    this_process_device = self.distributed_state.device
                    wait_for_memory_release(target_gpu_index=this_process_device.index, threshold_mb=4096)

                    server_url = vllm_server.start_server(
                        hf_ckpt_path_or_model=vllm_ckpt_dir,
                        wait_for_response=True,
                        log_path=vllm_log_file,
                        timeout=800,
                    )

                    os.environ["APP_OPENAI_VLLM_API_BASE"] = "none"

                    # Run the evaluation pipeline for early stopping
                    cfg = self.early_stop_inference_pipeline_cfg.copy()
                    inference_name = cfg["inference_name"]
                    logger.info(f"(EARLY-STOP)Running inference pipeline {inference_name}")
                    es_root_dir = eval_dir / ckpt.name / inference_name
                    es_root_dir.mkdir(parents=True, exist_ok=True)
                    pipeline = InferencePipeline.from_params(
                        Params(cfg),
                        tokenizer=self.early_stop_tokenizer,   # todo: find the tokenizer
                        seed=2746318213,
                        api_base_url=server_url,
                        model_name=str(vllm_ckpt_dir),
                        metrics_prefix=f"{ckpt.name}/",
                        enable_cloud_logging_during_inference=False,
                        use_cache=True,
                        cloud_logger=self.cloud_logger,
                        debug_mode=False,
                        exp_root=es_root_dir,
                        checkpoint_global_step=self.state.global_step,
                    )

                    results = pipeline.generate()
                    es_accuracy = pipeline.analyze(results)[0]['correct_frac']

                    es_accuracies.append(es_accuracy)

                    vllm_server.stop_server()
                    logger.info(f"(EARLY-STOP) Accuracy for {ckpt.name}: {es_accuracy:.3f}")

                # in RestEM we early stop based on the validation accuracy
                assert len(es_accuracies) == len(checkpoints) and len(checkpoints) > 0, "There should be a one-to-one correspondence between checkpoints and accuracies."
                if self.early_stop_method == 'restem_paper_original':
                    chosen_checkpoint = checkpoints[0]
                    for i in range(1, len(checkpoints)):
                        if es_accuracies[i] > es_accuracies[i-1]:
                            chosen_checkpoint = checkpoints[i]
                        else:
                            break
                elif self.early_stop_method == 'choose_best':
                    max_idx = 0
                    for i in range(1, len(checkpoints)):
                        if es_accuracies[i] > es_accuracies[max_idx]:
                            max_idx = i
                    chosen_checkpoint = checkpoints[max_idx]
                else:
                    raise ValueError(f"Invalid value for early_stop_method: {self.early_stop_method}")

                logger.info(f"(EARLY-STOP) Chosen checkpoint: {chosen_checkpoint}")
                (chosen_checkpoint / "chosen_checkpoint").touch()
                (self.experiment_root / "restem_inside_evaluations" / chosen_checkpoint.name / "chosen_checkpoint").touch()

        from accelerate.utils import broadcast_object_list
        chosen_checkpoint = broadcast_object_list([chosen_checkpoint], from_process=0)[0]

        return chosen_checkpoint

    def _train_actor(
        self,
        episodes: Dataset,
        actor: DeepSpeedEngine,
    ):
        """
        Train the actor model using MLE.

        Args:
            episodes (Dataset):
                The episodes to train on (possibly with reference log probabilities).
            actor (DeepSpeedEngine):
                The actor model to train.
        """
        kls = self._log_episodes_metrics(episodes)
        if self.state.epoch == 0:
            assert np.mean(kls) == 0.0, "The KLs should be zero in the first epoch as the actor and the reference are the same model"
        else:
            logger.info(f"(@@-o) KLs: mean kl with reference policy is: {np.mean(kls)}, as we're continuing it's ok that this is not zero.")

        # Step 2: The actual RestEM training loop
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
        num_optimization_steps_in_iteration = self.num_epochs * optim_steps_in_epoch
        total_num_optimization_steps = num_optimization_steps_in_iteration

        logger.info(f"***** Running a RestEM training step: {self.state.iteration}  *****")

        logger.info(f"  Num Episodes = {len(episodes):,}")
        logger.info(f"  Num Epochs Per Iteration = {self.num_epochs:,}")
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

        logger.info(f"  ---------------------------------")
        logger.info(f"  Current Global Step = {self.state.global_step}")

        # Create a new dataloader iterator
        dataloader_iter = iter(dataloader)

        progress_bar = tqdm(
            total=total_num_optimization_steps,
            disable=not self._is_main_process(),
            desc=f"Iteration {self.state.iteration}: Training",
            dynamic_ncols=True,
        )
        progress_bar.update(self.state.global_step)

        if self.state.epoch == 0:
            logger.info(f"Usual training loop. epoch has reset to zero.")
            num_batches_to_skip = 0
        elif 0 < self.state.epoch < self.num_epochs:
            # Skipping batches if we're in the middle of an iteration and continuing training now.
            logger.info("(@@-o) The run was terminated in the middle of an iteration. We compute how many batches to skip.")
            num_batches_to_skip = self.state.epoch * steps_in_epoch
            logger.info(f"(@@-o) num_batches_to_skip calculated from epoch: {num_batches_to_skip:.2f}") # this can have a fraction, as epoch is saved just to two decimal points
            num_batches_to_skip = int(num_batches_to_skip)
            logger.info(f"(@@-o) Plan is to skip {num_batches_to_skip} batches.")
        else:
            raise ValueError(f"Invalid value for self.state.epoch: {self.state.epoch}")

        progress_bar.update(num_batches_to_skip // self.args.gradient_accumulation_steps)

        globalstep_last_logged = self.state.global_step

        actor.train()

        running_metrics = {}
        accumulated_metrics = {}

        dist.barrier()

        starting_epoch = 0
        for epoch in range(starting_epoch, self.num_epochs):
            if num_batches_to_skip > 0:
                logger.info(f"(@@-o) Skipping {num_batches_to_skip} batches.")
            for step, inputs in enumerate(dataloader_iter):
                if num_batches_to_skip > 0:
                    num_batches_to_skip -= 1
                    if num_batches_to_skip == 0:
                        logger.info(f"(@@-o) Finished skipping batches at epoch {epoch} and step {step}.")
                        logger.info(f"(@@-o) current trainer state: {self.state}")
                    continue
                # Store the grad_acc_boundary before engine.step() is called
                # as the engine.step() will reset the grad_acc_boundary
                is_grad_acc_boundary = actor.is_gradient_accumulation_boundary()
                # Perform the training step, LR scheduler step, zero_grad, and accumulation of gradients
                # noinspection PyTypeChecker
                metrics = self._training_step(inputs, actor)
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
                        )
                        globalstep_last_logged = self.state.global_step

                    if self.args.save_steps != -1 and self.state.global_step % self.args.save_steps == 0:
                        temp_ckpt_path = self.temp_checkpoint_dir / self._get_automatic_checkpoint_name()
                        self._save_checkpoint(temp_ckpt_path, actor)
                        permanent_ckpt_path = self._copy_to_permanent_storage(temp_ckpt_path)
                        self._clean_old_temp_checkpoints(exclude=[temp_ckpt_path])
                        self._clean_deepspeed_checkpoints(exclude=[permanent_ckpt_path])

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

        progress_bar.close()

    def _training_step(
        self,
        inputs: Dict[str, torch.Tensor],
        actor: DeepSpeedEngine,
    ) -> Dict[str, Union[float, torch.Tensor]]:
        # refer to PPOTrainer._training_step to understand the alignment of inputs, logits, logps, and labels

        # noinspection DuplicatedCode
        inputs = {k: v.to(actor.device) for k, v in inputs.items()}

        input_ids = inputs["input_ids"]  # Shape: (batch_size, max_seq_len)
        attention_mask = inputs["attention_mask"]  # Shape: (batch_size, max_seq_len)
        labels = inputs["labels"]  # Shape: (batch_size, max_seq_len)

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

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        with torch.no_grad():
            shifted_ref_logprobs = inputs[COLUMN_REF_SHIFTED_LOGPS]

        # Step 2: Compute the policy/actor loss
        actor_loss, is_skipped, actor_metrics, approx_ref_kl = self._compute_actor_loss(
            actor,
            model_inputs=model_inputs,
            shifted_labels_mask=shifted_labels_mask,
            ref_logprobs=shifted_ref_logprobs,
        )
        actor.backward(actor_loss)
        self._check_overflow(actor)
        actor.step()
        # Get rid of actor's activations to free up memory
        actor_loss = actor_loss.detach().clone()
        release_memory()

        metrics = {
            "num_tokens": shifted_labels_mask.sum().detach(),
            "_num_participating_tokens": shifted_labels_mask.sum().detach(),
            **actor_metrics,
        }

        metrics["actor/loss"] = actor_loss
        metrics["actor/grad_norm"] = actor.get_global_grad_norm()
        metrics["jagged_kls"] = (approx_ref_kl * shifted_labels_mask).sum(dim=1).mean().detach()

        return metrics

    def _compute_actor_loss(
        self,
        actor: DeepSpeedEngine,
        model_inputs: Dict[str, torch.Tensor],
        shifted_labels_mask: torch.LongTensor,
        ref_logprobs: Optional[torch.FloatTensor],
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

        Returns:
            `torch.FloatTensor`: The actor loss.
            `bool`: Whether the batch was skipped.
            `Dict[str, torch.Tensor]`: Metrics from the training step.
            `Optional[torch.FloatTensor]`: The approx_kls tensor.
        """
        action_mask = shifted_labels_mask  # Shape: (batch_size, max_seq_len-1)

        # Compute the log probabilities of the actor
        outputs = self._forward_pass_actor(actor, model_inputs, return_all_logps=True,
                                           return_sequence_logp=True,
                                           sequence_logp_reduction="mean")

        seq_logprobs = outputs["sequence_logp"]  # Shape: (batch_size,)
        logprobs = outputs["all_logps"]  # Shape: (batch_size, seq_len-1)
        entropy = outputs["entropy"]
        assert logprobs.shape == ref_logprobs.shape
        assert action_mask.shape == logprobs.shape

        nll_loss = -1 * seq_logprobs.mean()
        assert ref_logprobs is not None
        ref_kl = self._compute_kl_penalty(
            logprobs,
            ref_logprobs,
            estimation_type=self.kl_penalty_loss_type,
        )
        ref_kl = ref_kl.detach()

        metrics = {
            "actor/logit_entropy": entropy.detach(),
        }

        is_skipped = False # We don't skip any batch in RestEM, just to keep the signature consistent with other Triners like PPO
        return nll_loss, is_skipped, metrics, ref_kl

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
            estimation_type = self.kl_penalty_loss_type

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
                    actor_lp, ref_lp, estimation_type="seq_control_variate"
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
            metrics["advantages/dist"] = advantages

        if len(ref_logprobs) > 0:
            ref_logprobs = np.array(ref_logprobs)
            metrics["ref_logprobs/sum"] = np.mean(ref_logprobs)
            metrics["ref_logprobs/normalized_by_response_len"] = np.mean(
                ref_logprobs / response_lengths
            )
            metrics["ref_logprobs/dist"] = ref_logprobs

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

        logs["epoch"] = round(self.state.epoch, 4)
        logs["step"] = self.state.global_step
        logs["actor/ds_step"] = actor.global_steps

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

        episodes = Dataset.load_from_disk(str(ds_w_ref_logprobs_path))
        return episodes

    def _get_episodes_w_curr_logps(
        self,
        episodes: Dataset,
        actor: DeepSpeedEngine,
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

    def _forward_pass_actor(
        self,
        model_engine: Union[DeepSpeedEngine, PreTrainedModel],
        inputs: Dict[str, torch.Tensor],
        return_logits: bool = False,
        return_sequence_logp: bool = False,
        return_all_logps: bool = False,
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
        logits /= self.sampling_temperature
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

        with torch.no_grad():
            mean_entropy = masked_mean(
                entropy_from_logits(shift_logits.detach()), shift_label_mask
            )
            mean_entropy = mean_entropy.detach().clone()
        output = {"entropy": mean_entropy}

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
        # we don't cache actor engine ever in RestEM, as the optimization in each
        # or iteration starts from scratch. Therefore, we don't want the engine which
        # contains the optimizer state to be cached from previous steps/iterations.

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
    ) -> None:
        metrics = {}
        if actor is not None:
            t0 = time.time()
            actor.load_checkpoint(str(checkpoint_path / "actor"))
            metrics["timing/actor/load_checkpoint"] = time.time() - t0

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

        if len(metrics) > 0:
            self._cloud_log({**metrics, "train/global_step": self.state.global_step})

    def is_checkpoint_resumable(self, checkpoint_path: Path) -> bool:
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

        epoch_str = checkpoint_path.name.split('--')[2]  # format ckpt--iter_xxxx--epoch_x.xx--step_xxxx
        assert epoch_str.startswith('epoch_')  # format iter_xxxx
        epoch = epoch_str.split('epoch_')[1]
        epoch = float(epoch)


        return True

    def clean_checkpoints(self, exclude: Optional[List[Path]] = None) -> None:
        if exclude is None:
            exclude = []

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
                        shutil.rmtree(file)
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

                logger.info(f"Removed files and dirs: {all_removed_files_and_dirs}")

    def _save_trainer_state(self, checkpoint_path: Path) -> None:
        super()._save_trainer_state(checkpoint_path)
        if self._is_main_process():
            save_custom_state(self.running_scores, checkpoint_path, index=10)

    def _load_training_state(self, checkpoint_path: Path) -> None:
        super()._load_training_state(checkpoint_path)
        load_custom_state(self.running_scores, checkpoint_path, index=10)

    def save_final_checkpoint(self) -> None:
        last_checkpoint_path = self.get_last_checkpoint().path
        final_checkpoint_path = self.checkpoints_dir / "final"
        final_checkpoint_path.write_text(last_checkpoint_path.name)

    def _copy_to_permanent_storage(self, checkpoint_path: Path) -> Path:
        permanent_storage_path = self.checkpoints_dir
        permanent_checkpoint_path = permanent_storage_path / checkpoint_path.name

        if not self._is_main_process():
            return permanent_checkpoint_path

        copy_cmd = f"cp -r {checkpoint_path} {permanent_storage_path}/"
        logger.info(f"Copying checkpoint to permanent storage: {copy_cmd}")

        # Copy and wait for it to finish, subprocess.run instead of subprocess.Popen
        subprocess.run(copy_cmd, shell=True, check=True)
        return permanent_checkpoint_path

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

    def _clean_deepspeed_checkpoints(self, exclude: List[Path]) -> None:

        if self._is_main_process():
            all_removed_files_and_dirs = []
            for checkpoint in self.checkpoints_dir.iterdir():
                if (
                    checkpoint.is_dir()
                    and checkpoint.name.startswith("ckpt--")
                    and checkpoint not in exclude
                ):
                    if (checkpoint / "actor").exists():
                        removed_files_and_dirs = []
                        for file in (checkpoint / "actor").iterdir():
                            assert file.name not in ["hf_pretrained", "pytorch_model.bin"]
                            removed_files_and_dirs.append(file.name)
                            if file.is_dir():
                                shutil.rmtree(file, ignore_errors=True)
                                logger.info(f"Removed directory: {file}")
                            else:
                                file.unlink()
                        all_removed_files_and_dirs += removed_files_and_dirs
        dist.barrier()

