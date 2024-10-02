import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
from typing import Union

import deepspeed
import torch
from accelerate import Accelerator, PartialState
from accelerate.utils import GradientAccumulationPlugin, DummyOptim, DummyScheduler
from datasets import Dataset
from peft import PeftModel
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    get_scheduler,
    DataCollator,
)
from transformers import Trainer as HfTrainer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import TrainerMemoryTracker, set_seed
from wandb.sdk.wandb_run import Run as WandbRun

from treetune.common import JsonDict
from treetune.common.py_utils import (
    is_flash_attention_model,
)
from treetune.logging_utils import get_logger
from treetune.models.base_model import Model
from treetune.trainers.arguments import TrainingArguments
from treetune.trainers.base_trainer import Trainer

logger = get_logger(__name__)


@dataclass
class Checkpoint:
    path: Path
    iteration: int


class TrainerState:
    global_step: int = 0
    epoch: float = 0.0
    iteration: int = 0

    INITIAL_STATE_DICT = {
        "global_step": 0,
        "epoch": 0.0,
        "iteration": 0,
    }

    def load_state_dict(self, state_dict):
        self.global_step = state_dict["global_step"]
        self.epoch = state_dict["epoch"]
        self.iteration = state_dict["iteration"]

    def state_dict(self):
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "iteration": self.iteration,
        }

    def __repr__(self):
        return f"TrainerState(global_step={self.global_step}, epoch={self.epoch}, iteration={self.iteration})"

    __qualname__ = "TrainerState"


class PolicyTrainer(Trainer):
    def __init__(
        self,
        training_args: JsonDict,
        model: Model,
        num_episodes_per_iteration: int,
        distributed_state: PartialState,
        experiment_root: Path,
        num_iterations: int = 1,
        num_epochs_per_iteration: int = 1,
        init_model_only: bool = False,
        data_collator: Optional[DataCollator] = None,
        deepspeed_config: Optional[JsonDict] = None,
        cloud_logger: Optional[WandbRun] = None,
    ):
        self.distributed_state = distributed_state
        self.args = TrainingArguments(**training_args)
        set_seed(self.args.seed)

        if isinstance(model, PeftModel):
            assert (
                distributed_state.num_processes == 1 and deepspeed_config is None
            ), "If PEFT just train on a single GPU with no deepspeed"

        self.state = TrainerState()

        self.cloud_logger = cloud_logger

        self.num_iterations = num_iterations
        self.num_epochs_per_iteration = num_epochs_per_iteration
        self.num_episodes_per_iteration = num_episodes_per_iteration
        self.data_collator = data_collator
        self.init_model_only = init_model_only

        self.experiment_root = experiment_root
        self.checkpoints_dir = self.experiment_root / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True, parents=True)
        checkpoint_format = self.get_checkpoint_format()
        assert checkpoint_format.startswith("ckpt--iter_")
        for state in checkpoint_format.replace("ckpt--", "").split("--"):
            assert len(state.split("_", 1)) == 2
            state_name, state_value = state.split("_", 1)
            assert state_value.startswith("{") and state_value.endswith("}")

        self.model: AutoModelForCausalLM = model

        # Flash attention models do not support gradient checkpointing for now
        self.is_flash_attention_model = is_flash_attention_model(self.model)
        if not self.is_flash_attention_model and self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if init_model_only:
            logger.warning(
                "Disabling DeepSpeed as currently it's not supported with `init_model_only=True`"
            )
            deepspeed_config = None

        self.deepspeed_plugin = None
        if deepspeed_config is not None:
            self.args.world_size = distributed_state.num_processes
            from transformers.deepspeed import HfTrainerDeepSpeedConfig

            self.hf_deepspeed_config = HfTrainerDeepSpeedConfig(deepspeed_config)
            self.hf_deepspeed_config.trainer_config_process(self.args)
            self.hf_deepspeed_config.fill_match(
                "optimizer.params.momentum", self.args.sgd_momentum, "sgd_momentum"
            )

            from accelerate.utils import DeepSpeedPlugin

            os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
            self.deepspeed_plugin = DeepSpeedPlugin(
                hf_ds_config=self.hf_deepspeed_config
            )
        self._create_accelerator_and_postprocess()
        self._set_process_log_level(logger)

        # memory metrics - must set up as early as possible
        self._memory_tracker = TrainerMemoryTracker(
            skip_memory_metrics=not self.accelerator.use_distributed
        )
        self._memory_tracker.stages = {
            "__init__": "init",
            "step": "step",
        }
        self._memory_tracker.start()

        global_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        self.total_num_training_steps = (
            num_iterations
            * num_epochs_per_iteration
            * num_episodes_per_iteration
            // global_batch_size
        )
        if self.accelerator.is_local_main_process:
            logger.info(
                f"Per device batch size: {self.args.per_device_train_batch_size}"
            )
            logger.info(
                f"Gradient accumulation steps: {self.args.gradient_accumulation_steps}"
            )
            logger.info(f"Num of total processes: {self.accelerator.num_processes}")
            logger.info(
                f"Global batch size (w. parallel, distributed & accumulation): {global_batch_size}"
            )
            logger.info(
                f"Total number of training steps: {self.total_num_training_steps}"
            )
        if self._can_log_to_cloud():
            self.cloud_logger.summary.update(
                {
                    "num_processes": self.accelerator.num_processes,
                    "global_batch_size": global_batch_size,
                    "total_num_training_steps": self.total_num_training_steps,
                }
            )
        if self.is_deepspeed_enabled:
            hf_deepspeed_config = self.accelerator.state.deepspeed_plugin.hf_ds_config

            # resume config update - some bits like `model` and `num_training_steps` only become available during train
            hf_deepspeed_config.trainer_config_finalize(
                self.args, self.model, self.total_num_training_steps
            )

        if init_model_only:
            self.model, self.data_collator = self.accelerator.prepare(
                self.model, self.data_collator
            )
        else:
            optimizer, lr_scheduler = self._create_optimizer_and_scheduler(
                self.total_num_training_steps
            )
            (
                self.model,
                self.optimizer,
                self.data_collator,
                self.lr_scheduler,
            ) = self.accelerator.prepare(
                self.model,
                optimizer,
                self.data_collator,
                lr_scheduler,
            )

        # very last
        self._memory_tracker.stop_and_update_metrics()

        # define default x-axis (for latest wandb versions)
        if self.accelerator.is_main_process:
            if getattr(self.cloud_logger, "define_metric", None):
                self.cloud_logger.define_metric("train/global_step")
                self.cloud_logger.define_metric(
                    "*", step_metric="train/global_step", step_sync=True
                )

    def get_checkpoint_format(self) -> str:
        return "ckpt--iter_{iteration}--epoch_{epoch}--step_{global_step}"

    @staticmethod
    def parse_checkpoint_name(checkpoint_name: str) -> Tuple[float, ...]:
        checkpoint_name = checkpoint_name.replace("ckpt--", "")
        parts = []
        for part in checkpoint_name.split("--"):
            part_id = part.split("_")[1]
            parts.append(float(part_id))
        assert len(parts) >= 1
        return tuple(parts)

    @staticmethod
    def is_checkpoint_resumable(checkpoint_path: Path) -> bool:
        non_model_files = [
            file
            for file in checkpoint_path.iterdir()
            if file.name not in ["hf_pretrained", "pytorch_model.bin", "pytorch_model"]
        ]
        return len(non_model_files) > 0

    def get_last_checkpoint(
        self, return_resumable_only: bool = False
    ) -> Optional[Checkpoint]:
        checkpoints = list(self.checkpoints_dir.iterdir())
        checkpoints = [
            checkpoint
            for checkpoint in checkpoints
            if checkpoint.is_dir() and checkpoint.name.startswith("ckpt--")
        ]
        if return_resumable_only:
            checkpoints = [
                checkpoint
                for checkpoint in checkpoints
                if self.is_checkpoint_resumable(checkpoint)
            ]
        if len(checkpoints) == 0:
            return None

        checkpoints = sorted(
            checkpoints, key=lambda x: self.parse_checkpoint_name(x.name)
        )
        last_checkpoint = checkpoints[-1]
        last_checkpoint_iteration = self.parse_checkpoint_name(last_checkpoint.name)[0]

        return Checkpoint(
            path=last_checkpoint, iteration=int(last_checkpoint_iteration)
        )

    def load_checkpoint(self, checkpoint: Union[Checkpoint, Path]) -> None:
        ckpt_path = (
            checkpoint.path if isinstance(checkpoint, Checkpoint) else checkpoint
        )
        assert ckpt_path.exists()

        logger.info(f"Loading checkpoint from {ckpt_path}")
        self.accelerator.load_state(str(ckpt_path))

    def save_checkpoint(self, checkpoint_path: Path) -> None:
        with self.accelerator.local_main_process_first():
            if checkpoint_path.exists():
                logger.warning(
                    f"Checkpoint path {checkpoint_path} already exists. Overwriting."
                )
                shutil.rmtree(checkpoint_path)

        # If the model is a flash attention model, save the HF pretrained weights as well
        if self.accelerator.is_local_main_process:
            if self.is_deepspeed_enabled:
                if self.accelerator.deepspeed_config["zero_optimization"]["stage"] == 3:
                    raise ValueError(
                        "Saving HF pretrained weights is not supported when using ZeRO-3"
                    )

            hf_checkpoint_path = checkpoint_path / "hf_pretrained"
            logger.info(f"Saving HF pretrained weights to {hf_checkpoint_path}")
            model = self.accelerator.unwrap_model(self.model)
            if self.is_flash_attention_model:
                model.save_hf_pretrained(hf_checkpoint_path)
            else:
                # For LoRA model we also use 'save_pretrained', which only saves the lora weights
                # During evaluation, we compute full model weights.
                model.save_pretrained(hf_checkpoint_path, safe_serialization=False)

        self.accelerator.save_state(
            str(checkpoint_path)
        )  # TODO: what happens to peft model here?

        self.accelerator.wait_for_everyone()

    def save_final_checkpoint(self) -> None:
        # Create a file called final that has the name of the last checkpoint
        self._save_automatic_checkpoint()
        last_checkpoint_path = self.get_last_checkpoint().path
        final_checkpoint_path = self.checkpoints_dir / "final"
        final_checkpoint_path.write_text(last_checkpoint_path.name)

    def clean_checkpoints(self, exclude: List[Path] = None) -> None:
        """
        Clean all optimizer and scheduler states which are not needed for evaluation.
        """
        if exclude is None:
            exclude = []

        for checkpoint in self.checkpoints_dir.iterdir():
            if (
                checkpoint.is_dir()
                and checkpoint.name.startswith("ckpt--")
                and checkpoint not in exclude
            ):
                if self.args.checkpoint_keep_steps is not None:
                    checkpoint_steps = self.parse_checkpoint_name(checkpoint.name)[-1]
                    if checkpoint_steps % self.args.checkpoint_keep_steps == 0:
                        continue

                logger.info(f"Removing checkpoint {checkpoint}")
                shutil.rmtree(checkpoint)

        self.clean_non_model_checkpoints(exclude=exclude)

    def clean_non_model_checkpoints(self, exclude: List[Path] = None) -> None:
        """
        Clean all optimizer and scheduler states which are not needed for evaluation.
        """
        if exclude is None:
            exclude = []

        for checkpoint in self.checkpoints_dir.iterdir():
            if (
                checkpoint.is_dir()
                and checkpoint.name.startswith("ckpt--")
                and checkpoint not in exclude
            ):
                logger.info(f"Removing non-model files from checkpoint {checkpoint}")

                # Remove everything except the `hf_pretrained` folder
                removed_files_and_dirs = []
                for file in checkpoint.iterdir():
                    if file.name not in ["hf_pretrained", "pytorch_model.bin"]:
                        removed_files_and_dirs.append(file.name)
                        if file.is_dir():
                            shutil.rmtree(file, ignore_errors=True)
                        else:
                            file.unlink()

                logger.info(f"Removed files and dirs: {removed_files_and_dirs}")

    def _save_automatic_checkpoint(self) -> Path:
        checkpoint_format = self.get_checkpoint_format()
        checkpoint_name = checkpoint_format.format(
            iteration=str(self.state.iteration).zfill(4),
            epoch=f"{self.state.epoch:.2f}",
            global_step=str(self.state.global_step).zfill(4),
        )
        checkpoint_path = self.checkpoints_dir / f"{checkpoint_name}"

        self.save_checkpoint(checkpoint_path)

        if self.accelerator.is_main_process:
            try:
                self.clean_checkpoints(exclude=[checkpoint_path])
            except Exception as e:
                logger.warning(f"Failed to clean non-model checkpoints: {e}")

        return checkpoint_path

    def step(self, episodes_dataset: Dataset) -> Optional[Path]:
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
                    - "advantages": The advantages of the response. (Optional)
        """
        raise NotImplementedError

    def _create_accelerator_and_postprocess(self):
        grad_acc_kwargs = {"num_steps": self.args.gradient_accumulation_steps}
        # grad_acc_kwargs["sync_with_dataloader"] = False
        gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)

        # Create accelerator object
        self.accelerator = Accelerator(
            dispatch_batches=False,
            deepspeed_plugin=self.deepspeed_plugin,
            gradient_accumulation_plugin=gradient_accumulation_plugin,
        )

        # Deepspeed and Accelerate flags covering both trainer args and accelerate launcher
        self.is_deepspeed_enabled = (
            getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        )

        if self.is_deepspeed_enabled:
            from deepspeed.utils import logger as ds_logger
            import logging

            ds_logger.setLevel(logging.DEBUG)

            if getattr(self.args, "hf_deepspeed_config", None) is None:
                from transformers.deepspeed import HfTrainerDeepSpeedConfig

                ds_plugin = self.accelerator.state.deepspeed_plugin

                ds_plugin.hf_ds_config = HfTrainerDeepSpeedConfig(
                    ds_plugin.hf_ds_config.config
                )
                ds_plugin.deepspeed_config = ds_plugin.hf_ds_config.config
                ds_plugin.hf_ds_config.trainer_config_process(self.args)

        if (
            self.args.gradient_checkpointing
            and not self.is_flash_attention_model
            and not self.is_deepspeed_enabled
        ):
            from accelerate import DistributedDataParallelKwargs

            self.accelerator.ddp_handler = DistributedDataParallelKwargs(
                find_unused_parameters=False
            )

        # Add state to accelerator for checkpointing
        self.accelerator.register_for_checkpointing(self.state)

    def _create_optimizer_and_scheduler(self, num_training_steps: int):
        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(num_training_steps, optimizer)
        return optimizer, scheduler

    def _create_optimizer(self) -> Union[torch.optim.Optimizer, DummyOptim]:
        opt_model = self.model

        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]

        use_deepspeed_optimizer = (
            self.accelerator.state.deepspeed_plugin is not None
            and "optimizer" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )

        if use_deepspeed_optimizer:
            optimizer_cls, optimizer_kwargs = DummyOptim, {}
        else:
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                self.args
            )

        optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        if optimizer_cls.__name__ == "Adam8bit":
            import bitsandbytes

            manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

            skipped = 0
            for module in opt_model.modules():
                if isinstance(module, nn.Embedding):
                    skipped += sum(
                        {p.data_ptr(): p.numel() for p in module.parameters()}.values()
                    )
                    logger.info(f"skipped {module}: {skipped/2**20}M params")
                    manager.register_module_override(
                        module, "weight", {"optim_bits": 32}
                    )
                    logger.debug(f"bitsandbytes: will optimize {module} in fp32")
            logger.info(f"skipped: {skipped/2**20}M params")

        return optimizer

    @staticmethod
    def get_optimizer_cls_and_kwargs(args):
        optimizer_cls, optimizer_kwargs = HfTrainer.get_optimizer_cls_and_kwargs(args)

        if optimizer_cls.__name__ == "SGD":
            logger.info(
                f"Using SGD optimizer -- finding the momentum parameter to {args.sgd_momentum}"
            )
            optimizer_kwargs.update({"momentum": args.sgd_momentum})

        return optimizer_cls, optimizer_kwargs

    def _create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer = None
    ) -> Union[torch.optim.lr_scheduler.LRScheduler, DummyScheduler]:
        use_deepspeed_scheduler = (
            self.accelerator.state.deepspeed_plugin is not None
            and "scheduler" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        if use_deepspeed_scheduler:
            lr_scheduler = DummyScheduler(
                optimizer,
                warmup_num_steps=self.args.get_warmup_steps(num_training_steps),
                total_num_steps=num_training_steps,
            )
        else:
            lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
        return lr_scheduler

    def _prepare_deepspeed(self, model: nn.Module):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepspeed_plugin.deepspeed_config
        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if (
                    hidden_size is not None
                    and config_kwargs["zero_optimization"]["stage"] == 3
                ):
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like:
                    # `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size
                            * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10
                            * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9
                            * hidden_size
                            * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and
        # is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def _can_log_to_cloud(self) -> bool:
        return self.accelerator.is_main_process and self.cloud_logger is not None

    def _cloud_log(self, *args, **kwargs):
        if self._can_log_to_cloud():
            self.cloud_logger.log(*args, **kwargs)

    def _get_learning_rate(self):
        if self.is_deepspeed_enabled:
            # with deepspeed's fp16 and dynamic loss scale enabled the optimizer/scheduler steps may
            # not run for the first few dozen steps while loss scale is too large, and thus during
            # that time `get_last_lr` will fail if called during that warm up stage, so work around it:
            try:
                last_lr = self.lr_scheduler.get_last_lr()[0]
            except AssertionError as e:
                if "need to call step" in str(e):
                    logger.warning(
                        "tried to get lr value before scheduler/optimizer started stepping, returning lr=0"
                    )
                    last_lr = 0
                else:
                    raise
        else:
            if isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                last_lr = self.optimizer.param_groups[0]["lr"]
            else:
                last_lr = self.lr_scheduler.get_last_lr()[0]
            if torch.is_tensor(last_lr):
                last_lr = last_lr.item()
        return last_lr

    def _set_process_log_level(self, logger_obj: logging.Logger):
        if not self.accelerator.is_local_main_process:
            logger_obj.setLevel(logging.WARNING)
