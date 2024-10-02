import time
from typing import Dict, Union, Any, Mapping, Tuple, List

import torch
from datasets import Dataset
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.trainer_pt_utils import get_model_param_count
from transformers.trainer_utils import (
    seed_worker,
    speed_metrics,
)  # todo: what is seed worker?

from treetune.logging_utils import get_logger
from treetune.trainers import Trainer
from treetune.trainers.policy_trainer import PolicyTrainer
from treetune.trainers.utils import entropy_from_logits, masked_mean

try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss
    from transformers.utils.hub import cached_file

    is_flash_attn_xentropy_available = True
except ImportError:
    from torch.nn import CrossEntropyLoss

    is_flash_attn_xentropy_available = (
        True  # todo: why is this set to True? and why is it not used?
    )

logger = get_logger(__name__)


class MLEDataCollator:
    def __call__(self, data_instances: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collates the given data instances into a batch.
        Every data instance should have the following keys:
        - "query_token_ids": The token ids of the query.
        - "response_token_ids": The token ids of the response.
        - "score": The reward of the response (single scalar per response)
        - "advantages": The advantages of the response.

        Args:
            data_instances (List[Dict[str, Any]]):
                The data instances to collate.
        Returns:
            Dict[str, torch.Tensor]:
                The collated batch.
                It contains the following keys:
                - "input_ids": The token ids of the entire episode (query + responses).
                        Shape: (batch_size, max_seq_len)
                - "labels": The token ids of the entire episode (query + responses).
                - "attention_mask": The attention mask of the entire episode (query + responses).
                        Shape: (batch_size, max_seq_len)
                - "advantages": The advantages of the responses. It should be a 2D tensor of shape
                        (batch_size, episode_length-1)
        """

        # Get the maximum sequence length
        max_seq_len = max(
            len(instance["query_token_ids"]) + len(instance["response_token_ids"])
            for instance in data_instances
        )

        # Create the batch
        batch = {
            "input_ids": [],
            "labels": [],
            "attention_mask": [],
            "advantages": [],
        }

        # It doesn't matter what the pad token id is, since we will mask it out anyway
        pad_token_id = 0
        pad_label = -100

        for instance in data_instances:
            query_token_ids = instance["query_token_ids"]
            response_token_ids = instance["response_token_ids"]
            advantages = instance["advantages"]

            # Create the input ids and attention mask
            input_ids = query_token_ids + response_token_ids
            attention_mask = [1] * len(input_ids)
            num_pad_at_end = max_seq_len - len(input_ids)

            input_ids += [pad_token_id] * num_pad_at_end
            attention_mask += [0] * num_pad_at_end
            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)

            # Create the labels
            labels = (
                [pad_label] * len(query_token_ids)
                + response_token_ids
                + [pad_label] * num_pad_at_end
            )
            batch["labels"].append(labels)

            # Create the advantages
            advantages = (
                [0.0] * len(query_token_ids) + advantages + [0.0] * num_pad_at_end
            )
            batch["advantages"].append(advantages)

            assert len(labels) == len(advantages)

        # Convert the lists to tensors
        batch = {k: torch.tensor(v) for k, v in batch.items()}

        return batch


@Trainer.register("mle")
class MaximumLikelihoodTrainer(PolicyTrainer):
    def __init__(
        self,
        upscale_advantage: bool = False,
        loss_reduction_mode: str = "batch",
        **kwargs,
    ):
        """
        Initializes the MLE trainer.

        Args:
            upscale_advantage (bool, optional):
                Whether to upscale the advantages so that the mean advantage per example is the same.
                Defaults to False.
            loss_reduction_mode (str, optional):
                The loss reduction mode. Can be either "batch", "total_non_pad_tokens",
                and "per_instance_non_pad_tokens_then_batch".

        """
        super().__init__(data_collator=MLEDataCollator(), **kwargs)
        self.upscale_advantage = upscale_advantage
        self.loss_reduction_mode = loss_reduction_mode
        self._set_process_log_level(logger)
        self._total_train_loss = 0.0

    def step(self, episodes_dataset: Dataset) -> None:
        """
        Perform a single step of policy training.
        A Single step of policy training amounts to a single epoch of training on the
        episodes.

        Distributed Note:
            This function is called on each process. Thus, they all receive a full copy of the episodes_dataset.
            However, the dataloader is distributed, so each process will only receive a subset of the episodes.

        Args:
            episodes_dataset (Dataset):
                A HuggingFace Dataset containing the episodes to train on.
                It should have the following columns:
                    - "query_token_ids": The token ids of the query.
                    - "response_token_ids": The token ids of the response.
                    - "score": The reward of the response (single scalar per response)
                    - "advantages": The advantages of the response.
        """

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        start_time = time.time()

        # Create the dataloader
        dataloader = self._create_dataloader(episodes_dataset)

        self.accelerator.wait_for_everyone()

        logger.error(f"Synced all processes for iteration {self.state.iteration}")

        steps_in_epoch = len(dataloader)
        optim_steps_in_epoch = steps_in_epoch // self.args.gradient_accumulation_steps
        optim_steps_in_epoch = max(optim_steps_in_epoch, 1)
        num_optimization_steps = optim_steps_in_epoch * self.num_epochs_per_iteration

        logger.info(f"steps_in_epoch: {steps_in_epoch}")
        logger.info(f"optim_steps_in_epoch: {optim_steps_in_epoch}")
        logger.info(
            f"total_num_optimization_steps (#epoch x #optim_step_per_epoch): {num_optimization_steps}"
        )

        starting_epoch = 0
        completed_optim_steps_in_this_iteration = (
            self.state.global_step % num_optimization_steps
        )

        logger.info(
            f"completed_optim_steps_in_this_iteration: {completed_optim_steps_in_this_iteration}"
        )

        logger.info(
            f"***** Running a policy iteration step: {self.state.iteration}  *****"
        )

        logger.info(f"  Num Episodes = {len(episodes_dataset):,}")
        logger.info(f"  Num Epochs = {self.num_epochs_per_iteration:,}")
        logger.info(f"  Current Global Step = {self.state.global_step}")
        logger.info(
            f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}"
        )
        logger.info(
            f"  Num Optimization steps in this iteration = {num_optimization_steps:,}"
        )

        num_trainable_params = get_model_param_count(self.model, trainable_only=True)
        logger.info(f"  Number of trainable parameters = {num_trainable_params:,}")
        if self._can_log_to_cloud():
            self.cloud_logger.summary["num_trainable_params"] = num_trainable_params

        # Create a new dataloader iterator, so that we can skip the first batches
        dataloader_iter = iter(dataloader)

        # Check if we're resuming training in the middle of an iteration
        if completed_optim_steps_in_this_iteration > 0:
            starting_epoch = (
                completed_optim_steps_in_this_iteration // optim_steps_in_epoch
            )
            logger.info(f"**** Resuming training in the middle of an iteration. ****")
            logger.info(
                f"  Have completed {completed_optim_steps_in_this_iteration} steps and {starting_epoch} epochs. "
            )
            logger.info(
                f"  Will perform {num_optimization_steps - completed_optim_steps_in_this_iteration} more steps."
            )

            # Skip the first batches
            num_batches_to_skip = (
                completed_optim_steps_in_this_iteration
                * self.args.gradient_accumulation_steps
            )
            logger.info(
                f"  Skipping {num_batches_to_skip} batches from the dataloader "
            )

            num_skipped = 0
            while num_skipped < num_batches_to_skip:
                for _ in dataloader_iter:
                    num_skipped += 1
                    if num_skipped >= num_batches_to_skip:
                        break
                if num_skipped < num_batches_to_skip:
                    dataloader_iter = iter(dataloader)

            logger.info(f"  Finished skipping.")

        self.model.train()

        progress_bar = tqdm(
            total=num_optimization_steps,
            disable=not self.accelerator.is_local_main_process,
            desc=f"Iteration {self.state.iteration}: Training",
            dynamic_ncols=True,
        )
        progress_bar.update(completed_optim_steps_in_this_iteration)

        tr_loss = torch.tensor(0.0).to(self.accelerator.device)
        globalstep_last_logged = self.state.global_step

        for epoch in range(starting_epoch, self.num_epochs_per_iteration):
            if epoch == starting_epoch:
                completed_steps_in_this_epoch = (
                    completed_optim_steps_in_this_iteration
                    * self.args.gradient_accumulation_steps
                ) % steps_in_epoch
            else:
                completed_steps_in_this_epoch = 0

            for step, inputs in enumerate(dataloader_iter):
                step += completed_steps_in_this_epoch
                with self.accelerator.accumulate(self.model):
                    tr_loss_step, metrics, is_skipped = self.training_step(self.model, inputs)
                    tr_loss += tr_loss_step
                    if is_skipped:
                        self.accelerator.set_trigger()

                    if self.accelerator.sync_gradients:
                        if self.args.max_grad_norm is not None:
                            grad_norm = self.accelerator.clip_grad_norm_(
                                self.model.parameters(),
                                self.args.max_grad_norm,
                            )
                            if self.is_deepspeed_enabled:
                                # In deepspeed, model is actually a deepspeed engine instance
                                grad_norm = self.model.get_global_grad_norm()

                            metrics["grad_norm"] = grad_norm

                    if self.accelerator.check_trigger():
                        self.optimizer.zero_grad()
                        continue

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    progress_bar.update(1)

                    should_log = self.state.global_step % self.args.logging_steps == 0
                    if should_log:
                        tr_loss_scalar = self._log_training_metrics(
                            tr_loss, globalstep_last_logged, metrics, progress_bar
                        )
                        self._total_train_loss += tr_loss_scalar
                        globalstep_last_logged = self.state.global_step

                    should_save = (
                        self.args.save_steps != -1
                        and self.state.global_step % self.args.save_steps == 0
                    )
                    if should_save:
                        self._save_automatic_checkpoint()

            # Recreate the dataloader iterator
            dataloader_iter = iter(dataloader)

        train_loss = self._total_train_loss / self.state.global_step

        runtime_metrics = speed_metrics(
            "train",
            start_time,
            num_steps=num_optimization_steps,
            num_samples=len(episodes_dataset),
        )
        self._memory_tracker.stop_and_update_metrics(runtime_metrics)
        logs = {k.replace("train_", "train/"): v for k, v in runtime_metrics.items()}
        self._cloud_log({**logs, "train/global_step": self.state.global_step})

        if self.accelerator.is_local_main_process:
            logger.info(
                f"***** Iteration {self.state.iteration} Training finished. *****"
            )
            logger.info(
                f"  Average training loss: {train_loss:.4f} ({self._total_train_loss:.4f} total)"
            )
            logger.info(
                f"  Training epoch took: {time.time() - start_time:.4f} seconds"
            )
            for key, value in sorted(runtime_metrics.items()):
                logger.info(f"  {key} = {value}")

        self.state.iteration += 1

        # We create a data loader at each step, so we need to remove it from the accelerator
        self.accelerator._dataloaders.pop()

    def log_gradient_variance(
        self,
        episodes_dataset: Dataset,
        num_samples: int = 10000,
        store_rolling_aggregates_on_cpu: bool = False,
    ):
        # Originally the training was done using 4 GPUs. But for gradient variance computation, we only use 1 GPU.
        # So, we need to adjust the batch size accordingly.
        orig_gradient_accumulation_steps = self.args.gradient_accumulation_steps
        self.args.gradient_accumulation_steps *= 4 // self.accelerator.num_processes
        logger.info(
            f"Adjusting gradient accumulation steps "
            f"from {orig_gradient_accumulation_steps} to {self.args.gradient_accumulation_steps}"
        )

        if len(episodes_dataset) > num_samples:
            episodes_dataset = episodes_dataset.shuffle(seed=42).select(
                range(num_samples)
            )

        checkpoints = list(self.checkpoints_dir.iterdir())
        checkpoints = [
            checkpoint
            for checkpoint in checkpoints
            if checkpoint.is_dir()
            and checkpoint.name.startswith("ckpt--")
            and (checkpoint / "hf_pretrained").exists()
        ]

        if len(checkpoints) == 0:
            raise ValueError("No checkpoints found")

        # None is the initial model without any training
        checkpoints = [None] + checkpoints

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        for ckpt in checkpoints:
            if ckpt is None:
                step = 0
                model = self.model
            else:
                step = int(ckpt.name.split("--")[-1].split("step_")[-1])
                unwrapped_model.load_state_dict(
                    torch.load(ckpt / "hf_pretrained" / "pytorch_model.bin")
                )
                model = self.accelerator.prepare(unwrapped_model)

            model.eval()

            variance_metrics = self._compute_gradient_variance_metrics(
                model,
                episodes_dataset,
                store_rolling_aggregates_on_cpu=store_rolling_aggregates_on_cpu,
            )

            if self.accelerator.is_main_process:
                logs = {f"eval/gradient/{k}": v for k, v in variance_metrics.items()}
                logger.info(
                    f" *** Gradient variance metrics at step {step}: {logs} ***"
                )
                self._cloud_log({**logs, "train/global_step": step})

            self.accelerator.free_memory()

        # Reset the gradient accumulation steps
        self.args.gradient_accumulation_steps = orig_gradient_accumulation_steps

    def _compute_gradient_variance_metrics(
        self,
        model: PreTrainedModel,
        episodes_dataset: Dataset,
        store_rolling_aggregates_on_cpu: bool = False,
    ) -> Dict[str, float]:
        # Create the dataloader
        dataloader = self._create_dataloader(episodes_dataset)

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

        progress_bar = tqdm(
            total=len(dataloader),
            disable=not self.accelerator.is_local_main_process,
            desc=f"Computing gradient variance",
            dynamic_ncols=True,
        )

        for step, inputs in enumerate(dataloader):
            with self.accelerator.accumulate(model):
                with self.accelerator.autocast():
                    loss, _ = self.compute_loss(model, inputs)
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    # Skip this step if overflow/nan in model gradients
                    has_finite_grads = True
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if torch.any(torch.isnan(param.grad)):
                                logger.warning(
                                    f"Gradient variance computation: NaN in gradients at step {step}"
                                )
                                has_finite_grads = False
                                break
                    if not has_finite_grads:
                        model.zero_grad()
                        continue

                    count += 1
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param_grad = param.grad.detach().clone()
                            if store_rolling_aggregates_on_cpu:
                                param_grad = param_grad.cpu()

                            update_rolling_aggregates(name, param_grad, count)

                    model.zero_grad()

            progress_bar.update(1)

        progress_bar.close()

        # Compute the variance
        variance_mean = 0.0
        sample_variance_mean = 0.0
        total_num_params = 0

        for name in sorted(grad_rolling_mean.keys()):
            m2 = grad_rolling_m2[name]

            variance = m2 / count
            sample_variance = m2 / (count - 1)

            variance = variance.float().mean().cpu().numpy().item()
            sample_variance = sample_variance.float().mean().cpu().numpy().item()

            n_params = m2.numel()

            weight_existing = total_num_params / (total_num_params + n_params)
            weight_new = n_params / (total_num_params + n_params)

            variance_mean = weight_existing * variance_mean + weight_new * variance
            sample_variance_mean = (
                weight_existing * sample_variance_mean + weight_new * sample_variance
            )

            total_num_params += n_params

        return {
            "variance": variance_mean,
            "sample_variance": sample_variance_mean,
        }

    def training_step(
        self, model: PreTrainedModel, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], bool]:
        with self.accelerator.autocast():
            loss, metrics = self.compute_loss(model, inputs)

        if torch.isnan(loss):
            logger.error("NaN in loss")
            is_skipped = True
            return loss.detach().clone(), metrics, is_skipped

        self.accelerator.backward(loss)
        return loss.detach().clone(), metrics, False

    def compute_loss(
        self, model: PreTrainedModel, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes the loss for the given inputs. The loss is basically the negative log likelihood
        that weighted by the advantages of the responses.
        Args:
            model (PreTrainedModel): The model to compute the loss for.
            inputs (Dict[str, torch.Tensor]):
                The inputs to the model.
                It should contain the following keys:
                - "input_ids": The token ids of the entire episode (query + responses).
                        Shape: (batch_size, max_seq_len)
                - "labels": The token ids of the entire episode (query + responses).
                - "attention_mask": The attention mask of the entire episode (query + responses).
                        Shape: (batch_size, max_seq_len)
                - "advantages": The advantages of the responses. It should be a 2D tensor of shape
                        (batch_size, episode_length)
        Returns:
            torch.Tensor: The loss.
        """
        input_ids: torch.Tensor = inputs["input_ids"]
        labels: torch.Tensor = inputs["labels"]
        attention_mask: torch.Tensor = inputs["attention_mask"]
        advantages: torch.Tensor = inputs["advantages"]

        # Compute the logits
        if not self.is_flash_attention_model:
            outputs: CausalLMOutputWithPast = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,  # We don't need the cache for training
            )
        else:
            # Flash attention models do not support attention mask
            # But, since we're padding on right and the model is causal,
            # we're fine without the attention mask.
            assert torch.all(
                attention_mask[:, 0] == 1
            ), "Flash attention models do not support left padding"
            outputs: CausalLMOutputWithPast = model(input_ids=input_ids)
        logits = outputs.logits  # Shape: (batch_size, max_seq_len, vocab_size)
        orig_dtype = logits.dtype

        vocab_size = logits.shape[-1]
        batch_size = logits.shape[0]

        # Compute the loss in full precision
        logits = logits.to(torch.float32)

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_labels_unflattened = shift_labels
        shift_advantages = advantages[..., 1:].contiguous()

        if self.upscale_advantage:
            # Find number of non-pad labels
            num_non_pad_labels = torch.sum(
                (shift_labels_unflattened != -100).to(shift_advantages.dtype),
                dim=-1,
                keepdim=True,
            )

            # Scale the advantages so that the mean advantage per example is the same
            per_example_advantages = shift_advantages.sum(dim=-1, keepdim=True)
            advantage_scale = num_non_pad_labels / per_example_advantages
            shift_advantages *= advantage_scale

        # Flatten the tokens
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_advantages = shift_advantages.view(-1)
        shift_label_mask = (shift_labels != -100).to(shift_logits.dtype)

        mean_advantage = masked_mean(shift_advantages, shift_label_mask)
        mean_advantage = mean_advantage.detach().clone()

        mean_entropy = masked_mean(entropy_from_logits(shift_logits), shift_label_mask)
        mean_entropy = mean_entropy.detach().clone()

        loss_fct = CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits, shift_labels)

        # Multiply the loss by the advantages
        loss *= shift_advantages.to(loss.dtype)

        if self.loss_reduction_mode == "total_non_pad_tokens":
            num_non_pad_labels = torch.sum((shift_labels != -100).to(loss.dtype))
            loss = loss.sum() / num_non_pad_labels
        elif self.loss_reduction_mode == "per_instance_non_pad_tokens_then_batch":
            num_non_pad_labels = torch.sum(
                (shift_labels_unflattened != -100).to(loss.dtype), dim=-1, keepdim=True
            )  # [batch_size x seq_len]

            # Convert the loss back to [batch_size x seq_len]
            loss = loss.view(batch_size, -1)

            loss = loss.sum(dim=-1, keepdim=True) / num_non_pad_labels
            loss = loss.sum() / batch_size
        elif self.loss_reduction_mode == "batch":
            loss = loss.sum() / batch_size
        else:
            raise ValueError(f"Invalid loss reduction mode: {self.loss_reduction_mode}")

        loss = loss.to(orig_dtype)

        metrics = {"logit_entropy": mean_entropy, "advantage": mean_advantage}

        return loss, metrics

    def _log_training_metrics(
        self,
        tr_loss: torch.Tensor,
        _globalstep_last_logged: int,
        metrics: Dict[str, Union[float, torch.Tensor]],
        progress_bar: tqdm,
    ) -> float:
        logs: Dict[str, float] = {}

        # all_gather + mean() to get average loss over all processes
        tr_loss_scalar = self.accelerator.reduce(tr_loss, reduction="mean").item()

        # reset tr_loss to zero
        tr_loss -= tr_loss

        # Compute the log values over all processes
        num_steps_since_last_log = (
            self.state.global_step - _globalstep_last_logged
        ) * self.args.gradient_accumulation_steps
        logs["loss"] = round(tr_loss_scalar / num_steps_since_last_log, 4)
        logs["learning_rate"] = self._get_learning_rate()
        logs["epoch"] = round(self.state.epoch, 4)
        logs["step"] = self.state.global_step

        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, torch.Tensor):
                metric_value = self.accelerator.reduce(
                    metric_value, reduction="mean"
                ).item()
            if metric_value is not None:
                logs[f"metric/{metric_name}"] = round(metric_value, 6)

        # First log the metrics on the progress bar
        progress_bar.set_postfix(logs)

        # Add "train/" prefix for clarity.
        logs = {f"train/{k}": v for k, v in logs.items()}

        self._cloud_log({**logs, "train/global_step": self.state.global_step})

        return tr_loss_scalar

    def _prepare_input(
        self, data: Union[torch.Tensor, Any]
    ) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.args.device}
            if self.is_deepspeed_enabled and (
                torch.is_floating_point(data) or torch.is_complex(data)
            ):
                # NLP models inputs are int/uint and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update(
                    {
                        "dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()
                    }
                )
            return data.to(**kwargs)
        return data

    def _create_dataloader(self, episodes_dataset: Dataset) -> DataLoader:
        if episodes_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # Filter out the episodes that are too long
        if self.args.max_seq_len is not None:
            logger.error(f"Before filtering")
            max_seq_len = self.args.max_seq_len

            orig_len = len(episodes_dataset)
            episodes_dataset = episodes_dataset.filter(
                self._get_instance_length_filter_fn(max_seq_len)
            )
            logger.info(
                f"Filtered out {orig_len - len(episodes_dataset)} episodes "
                f"that are too long. Remaining: {len(episodes_dataset)}"
            )

        logger.error(f"Before creating data loader")

        data_collator = self.data_collator
        generator = torch.Generator()
        generator.manual_seed(self.args.seed)
        sampler = RandomSampler(episodes_dataset, generator=generator)

        data_loader = DataLoader(
            dataset=episodes_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            sampler=sampler,
            drop_last=self.args.dataloader_drop_last,
            worker_init_fn=seed_worker,
            persistent_workers=self.args.dataloader_num_workers > 0,
        )

        logger.error(f"After creating data loader")

        return self.accelerator.prepare(data_loader)

    def _get_instance_length_filter_fn(self, max_seq_len: int):
        def filter_fn(example):
            return (
                len(example["query_token_ids"]) + len(example["response_token_ids"])
                <= max_seq_len
            )

        return filter_fn
