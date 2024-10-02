import logging
from pathlib import Path
from typing import Optional, Tuple, List
from typing import Union
from weakref import WeakValueDictionary

import deepspeed
from accelerate import PartialState
from accelerate.checkpointing import save_custom_state, load_custom_state
from accelerate.utils import DummyOptim, DummyScheduler
from datasets import Dataset
from deepspeed import DeepSpeedEngine
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import PreTrainedModel, get_scheduler, PretrainedConfig
from transformers.integrations import HfTrainerDeepSpeedConfig
from wandb.sdk.wandb_run import Run as WandbRun

from treetune.common import JsonDict
from treetune.common.py_utils import log_tensors_living_on_gpu
from treetune.logging_utils import get_logger
from treetune.trainers.arguments import TrainingArguments
from treetune.trainers.base_trainer import Trainer
from treetune.trainers.policy_trainer import TrainerState, Checkpoint

logger = get_logger(__name__)


class DeepSpeedPolicyTrainer(Trainer):
    """
    The difference between this and `PolicyTrainer` is that this class solely uses DeepSpeed for training and
    ditched the Accelerate library. This is because the Accelerate library does not support two models in a single
    training loop, which becomes problematic in actor-critic training.
    """

    def __init__(
        self,
        distributed_state: PartialState,
        experiment_root: Path,
        cloud_logger: Optional[WandbRun] = None,
    ):
        self.distributed_state = distributed_state

        self.state = TrainerState()

        self.cloud_logger = cloud_logger

        self.experiment_root = experiment_root
        self.checkpoints_dir = self.experiment_root / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True, parents=True)
        self._validate_checkpoint_format()

        # define default x-axis (for latest wandb versions)
        if self._is_main_process():
            if getattr(self.cloud_logger, "define_metric", None):
                self.cloud_logger.define_metric("train/global_step")
                self.cloud_logger.define_metric(
                    "*", step_metric="train/global_step", step_sync=True
                )

    def _validate_checkpoint_format(self, checkpoint_format: Optional[str] = None):
        if checkpoint_format is None:
            checkpoint_format = self.get_checkpoint_format()
        assert checkpoint_format.startswith("ckpt--iter_")
        for state in checkpoint_format.replace("ckpt--", "").split("--"):
            assert len(state.split("_", 1)) == 2
            state_name, state_value = state.split("_", 1)
            assert state_value.startswith("{") and state_value.endswith("}")

    def create_optimizer(
        self,
        model: PreTrainedModel,
        weight_decay: float = 0.0,
    ) -> Union[Optimizer, DummyOptim]:
        from accelerate.utils import DummyOptim

        optim_params = get_optimizer_grouped_parameters(model, weight_decay)
        optim = DummyOptim(optim_params)
        return optim

    def create_lr_scheduler(
        self,
        optim: Optimizer,
        name: str,
        warmup_steps: Optional[int] = None,
        num_training_steps: Optional[int] = None,
    ) -> LRScheduler:
        return get_scheduler(
            name=name,
            optimizer=optim,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

    def get_checkpoint_format(self) -> str:
        return "ckpt--iter_{iteration}--epoch_{epoch}--step_{global_step}"

    @staticmethod
    def parse_checkpoint_name(checkpoint_name: str) -> Tuple[float, ...]:
        # noinspection DuplicatedCode
        checkpoint_name = checkpoint_name.replace("ckpt--", "")
        parts = []
        for part in checkpoint_name.split("--"):
            part_id = part.split("_")[1]
            parts.append(float(part_id))
        assert len(parts) >= 1
        return tuple(parts)

    @staticmethod
    def is_checkpoint_resumable(checkpoint_path: Path) -> bool:
        raise NotImplementedError()

    def get_last_checkpoint(
        self, return_resumable_only: bool = False
    ) -> Optional[Checkpoint]:
        # noinspection DuplicatedCode
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
        if self._can_log_to_cloud():
            self.cloud_logger.summary["last_loaded_checkpoint"] = str(checkpoint.path.name)

    def save_checkpoint(self, checkpoint_path: Path) -> None:
        raise NotImplementedError()

    def save_final_checkpoint(self) -> None:
        # Create a file called final that has the name of the last checkpoint
        self._save_automatic_checkpoint()
        last_checkpoint_path = self.get_last_checkpoint().path
        final_checkpoint_path = self.checkpoints_dir / "final"
        final_checkpoint_path.write_text(last_checkpoint_path.name)

    def clean_checkpoints(self, exclude: List[Path] = None) -> None:
        raise NotImplementedError()

    def clean_non_model_checkpoints(self, exclude: List[Path] = None) -> None:
        raise NotImplementedError()

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
        raise NotImplementedError()

    def get_models(
        self,
    ) -> WeakValueDictionary[str, Union[PreTrainedModel, DeepSpeedEngine]]:
        """
        Get the models that are being trained/used by this trainer.

        Returns:
            WeakValueDictionary[str, Union[PreTrainedModel, DeepSpeedEngine]]:
                A dictionary of model names to the model objects.
                The reason for using a WeakValueDictionary is to avoid keeping a reference to the model objects.
        """
        raise NotImplementedError()

    def _get_learning_rate(self, engine: DeepSpeedEngine):
        # with deepspeed's fp16 and dynamic loss scale enabled the optimizer/scheduler steps may
        # not run for the first few dozen steps while loss scale is too large, and thus during
        # that time `get_last_lr` will fail if called during that warm up stage, so work around it:
        try:
            return engine.get_lr()[0]
        except AssertionError as e:
            if "need to call step" in str(e):
                logger.warning(
                    "tried to get lr value before scheduler/optimizer started stepping, returning lr=0"
                )
                last_lr = 0
            else:
                raise

        return last_lr

    def _patch_ds_config_for_optimizer(
        self,
        config: HfTrainerDeepSpeedConfig,
        args: TrainingArguments,
    ):
        config.fill_only("optimizer.params.lr", args.learning_rate, "learning_rate")
        config.fill_only(
            "optimizer.params.betas",
            [args.adam_beta1, args.adam_beta2],
            "adam_beta1+adam_beta2",
        )
        config.fill_only("optimizer.params.eps", args.adam_epsilon, "adam_epsilon")
        config.fill_only(
            "optimizer.params.weight_decay", args.weight_decay, "weight_decay"
        )

    def _patch_ds_config_for_lr_scheduler(
        self,
        config: HfTrainerDeepSpeedConfig,
        total_num_training_steps: int,
        warmup_steps: int,
        learning_rate: float,
    ) -> None:
        config.fill_only(
            "scheduler.params.total_num_steps",
            total_num_training_steps,
            "num_training_steps (calculated)",
        )
        config.fill_only(
            "scheduler.params.warmup_num_steps",
            warmup_steps,
            "warmup_steps",
        )
        config.fill_only(
            "scheduler.params.warmup_min_lr",
            0,
            "warmup_min_lr",
        )
        config.fill_only(
            "scheduler.params.warmup_max_lr",
            learning_rate,
            "warmup_max_lr",
        )

    def _patch_ds_config_for_batch_size(
        self,
        config: HfTrainerDeepSpeedConfig,
        args: TrainingArguments,
        global_batch_size: int,
    ) -> None:
        config.fill_only(
            "train_micro_batch_size_per_gpu",
            args.per_device_train_batch_size,
            "per_device_train_batch_size",
        )
        config.fill_only(
            "gradient_accumulation_steps",
            args.gradient_accumulation_steps,
            "gradient_accumulation_steps",
        )
        config.fill_only(
            "train_batch_size", global_batch_size, "train_batch_size (calculated)"
        )
        config.fill_only("gradient_clipping", args.max_grad_norm, "max_grad_norm")

    def _patch_ds_config_for_dtype(
        self, config: HfTrainerDeepSpeedConfig, args: TrainingArguments
    ) -> None:
        assert not args.fp16, "FP16 is not supported for now"
        config.fill_only(
            "bf16.enabled", (args.bf16 or args.bf16_full_eval), "bf16|bf16_full_eval"
        )

    def _patch_ds_config_for_bucket_size(
        self, config: HfTrainerDeepSpeedConfig, model_config: PretrainedConfig
    ) -> None:
        hidden_size_based_keys = [
            "zero_optimization.reduce_bucket_size",
            "zero_optimization.stage3_prefetch_bucket_size",
            "zero_optimization.stage3_param_persistence_threshold",
        ]
        hidden_size_auto_keys = [x for x in hidden_size_based_keys if config.is_auto(x)]

        if len(hidden_size_auto_keys) > 0:
            if hasattr(model_config, "hidden_size"):
                hidden_size = model_config.hidden_size
            elif hasattr(model_config, "hidden_sizes"):
                # if there are many hidden sizes pick the largest one
                hidden_size = max(model_config.hidden_sizes)
            else:
                logger.warning(
                    "The model's config file has neither `hidden_size` nor `hidden_sizes` entry, "
                    "therefore it's not possible to automatically fill out the following `auto` entries "
                    f"in the DeepSpeed config file: {hidden_size_auto_keys}. We will set them to default values."
                )

                # if hidden size is not available, set the default values
                default_values = {
                    "zero_optimization.reduce_bucket_size": 5e8,
                    "zero_optimization.stage3_prefetch_bucket_size": 5e8,
                    "zero_optimization.stage3_param_persistence_threshold": 1e6,
                }
                for key in hidden_size_auto_keys:
                    if config.is_auto(key):
                        config.fill_only(key, default_values[key])
                return

            config.fill_only(
                "zero_optimization.reduce_bucket_size", hidden_size * hidden_size
            )
            if config.is_zero3():
                # automatically assign the optimal config values based on model config
                config.fill_only(
                    "zero_optimization.stage3_prefetch_bucket_size",
                    0.9 * hidden_size * hidden_size,
                )
                config.fill_only(
                    "zero_optimization.stage3_param_persistence_threshold",
                    10 * hidden_size,
                )

    def _initialize_deepspeed_engine_for_training(
        self,
        model: PreTrainedModel,
        deepspeed_config: JsonDict,
        optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[Union[LRScheduler, DummyScheduler]] = None,
    ) -> DeepSpeedEngine:
        kwargs = {
            "model": model,
            "lr_scheduler": lr_scheduler,
            "config": deepspeed_config,
        }
        if isinstance(optimizer, DummyOptim):
            kwargs["model_parameters"] = optimizer.params
        else:
            kwargs["optimizer"] = optimizer
        engine, *_ = deepspeed.initialize(**kwargs)
        return engine

    def _initialize_deepspeed_engine_for_inference(
        self,
        model: PreTrainedModel,
        deepspeed_config: JsonDict,
    ) -> DeepSpeedEngine:
        engine, *_ = deepspeed.initialize(
            model=model,
            config=deepspeed_config,
        )
        return engine

    def _destroy_ds_engine(self, ds_engine: DeepSpeedEngine):
        def delete_attr(obj, a_name):
            if hasattr(obj, a_name):
                delattr(obj, a_name)

        # This is a workaround to avoid a memory leak in DeepSpeed
        # This bug exists in the DeepSpeed version 0.14.1
        for name, param in ds_engine.named_parameters():
            delete_attr(param, "get_full_hp_grad")
            delete_attr(param, "set_full_hp_grad")
            delete_attr(param, "load_hp_checkpoint_state")
            delete_attr(param, "_hp_mapping")

        ds_engine.empty_partition_cache()
        ds_engine.destroy()     # todo(milad): why deeospeed has these globals

    def log_tensors_on_gpu(self):
        if not self._is_main_process():
            return
        log_tensors_living_on_gpu(logger)

    def _save_checkpoint(self, checkpoint_path: Path, **kwargs):
        raise NotImplementedError()

    def _get_automatic_checkpoint_name(self) -> str:
        checkpoint_format = self.get_checkpoint_format()
        checkpoint_name = checkpoint_format.format(
            iteration=str(self.state.iteration).zfill(4),
            epoch=f"{self.state.epoch:.2f}",
            global_step=str(self.state.global_step).zfill(4),
        )
        return checkpoint_name

    def _save_automatic_checkpoint(self, **kwargs) -> Path:
        checkpoint_name = self._get_automatic_checkpoint_name()
        checkpoint_path = self.checkpoints_dir / f"{checkpoint_name}"

        self._save_checkpoint(checkpoint_path, **kwargs)

        if self._can_log_to_cloud():
            self.cloud_logger.summary["last_checkpoint"] = str(checkpoint_name)

        if self._is_main_process():
            try:
                self.clean_checkpoints(exclude=[checkpoint_path])
            except Exception as e:
                logger.warning(f"Failed to clean non-model checkpoints: {e}")

        return checkpoint_path

    def _can_log_to_cloud(self) -> bool:
        return self._is_main_process() and self.cloud_logger is not None

    def _is_main_process(self) -> bool:
        return self.distributed_state.is_main_process

    def _cloud_log(self, *args, **kwargs):
        if self._can_log_to_cloud():
            self.cloud_logger.log(*args, **kwargs)

    def _set_process_log_level(self, logger_obj: logging.Logger):
        if not self.distributed_state.is_local_main_process:
            logger_obj.setLevel(logging.WARNING)

    def _save_trainer_state(self, checkpoint_path: Path) -> None:
        if self._is_main_process():
            save_custom_state(self.state, checkpoint_path, index=0)

    def _load_training_state(self, checkpoint_path: Path) -> None:
        load_custom_state(self.state, checkpoint_path, index=0)


def get_optimizer_grouped_parameters(
    model: PreTrainedModel,
    weight_decay: float,
    lora_lr: Optional[float] = None,
    no_decay_name_list: Optional[List[str]] = None,
    lora_name_list: Optional[List[str]] = None,
):
    # Taken from
    # https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/utils/utils.py#L209
    if lora_name_list is None:
        lora_name_list = ["lora_right_weight", "lora_left_weight"]
    if no_decay_name_list is None:
        no_decay_name_list = [
            "bias",
            "layer_norm.weight",
            "layernorm.weight",
            "norm.weight",
            "ln_f.weight",
        ]

    optimizer_grouped_parameters = [
        # Weight decay, non-lora parameters
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (
                    not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad
                    and not any(nd in n.lower() for nd in lora_name_list)
                )
            ],
            "weight_decay": weight_decay,
        },
        # Weight decay, lora parameters
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (
                    not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad
                    and any(nd in n.lower() for nd in lora_name_list)
                )
            ],
            "weight_decay": weight_decay,
            "lr": lora_lr,
        },
        # No weight decay, irrespective of lora
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (
                    any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad
                )
            ],
            "weight_decay": 0.0,
        },
    ]

    non_empty_groups = []
    for group in optimizer_grouped_parameters:
        if group["params"]:
            non_empty_groups.append(group)
    return non_empty_groups
