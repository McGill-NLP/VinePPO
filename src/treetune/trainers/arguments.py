import json
import math
import os
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Union, Optional

import torch
from transformers import SchedulerType, is_torch_available
from transformers import TrainingArguments as HfTrainingArguments
from transformers.training_args import OptimizerNames
from transformers.utils import is_torch_tf32_available

assert HfTrainingArguments is not None

from treetune import logging_utils

logger = logging_utils.get_logger(__name__)


@dataclass
class TrainingArguments:
    per_device_train_batch_size: int = field(
        default=None,
        metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."},
    )
    per_device_eval_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for evaluation."},
    )
    target_train_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Target batch size for training. Overrides "
                "per_device_train_batch_size based on the number of devices."
            )
        },
    )

    gradient_accumulation_steps: int = field(
        default=None,
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass."
        },
    )
    optim: Union[OptimizerNames, str] = field(
        default="adamw_torch",
        metadata={"help": "The optimizer to use."},
    )
    optim_args: str = field(
        default=None,
        metadata={"help": "Optimizer arguments."},
    )
    learning_rate: float = field(
        default=5e-5, metadata={"help": "The initial learning rate for AdamW."}
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."}
    )
    adam_beta1: float = field(
        default=0.9, metadata={"help": "Beta1 for AdamW optimizer"}
    )
    adam_beta2: float = field(
        default=0.999, metadata={"help": "Beta2 for AdamW optimizer"}
    )
    adam_epsilon: float = field(
        default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."}
    )

    sgd_momentum: float = field(
        default=0.0, metadata={"help": "Momentum for SGD optimizer"}
    )

    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    num_train_epochs: float = field(
        default=3.0, metadata={"help": "Total number of training epochs to perform."}
    )
    lr_scheduler_type: Optional[Union[SchedulerType, str]] = field(
        default=None,
        metadata={"help": "The scheduler type to use."},
    )
    warmup_ratio: float = field(
        default=0.0,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."},
    )
    warmup_steps: int = field(
        default=0, metadata={"help": "Linear warmup over warmup_steps."}
    )

    seed: int = field(
        default=42,
        metadata={"help": "Random seed that will be set at the beginning of training."},
    )
    data_seed: Optional[int] = field(
        default=None, metadata={"help": "Random seed to be used with data samplers."}
    )

    bf16: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
                " architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change."
            )
        },
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"},
    )
    fp16_opt_level: str = field(
        default="O1",
        metadata={
            "help": (
                "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. "
                "See details at https://nvidia.github.io/apex/amp.html"
            )
        },
    )
    half_precision_backend: str = field(
        default="auto",
        metadata={
            "help": "The backend to be used for half precision.",
            "choices": ["auto", "cuda_amp", "apex", "cpu_amp"],
        },
    )
    bf16_full_eval: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use full bfloat16 evaluation instead of 32-bit. This is an experimental API and it may"
                " change."
            )
        },
    )
    fp16_full_eval: bool = field(
        default=False,
        metadata={"help": "Whether to use full float16 evaluation instead of 32-bit"},
    )
    tf32: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to enable tf32 mode, available in Ampere and newer GPU architectures. This is an experimental"
                " API and it may change."
            )
        },
    )
    dataloader_drop_last: bool = field(
        default=False,
        metadata={
            "help": "Drop the last incomplete batch if it is not divisible by the batch size."
        },
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": (
                "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded"
                " in the main process."
            )
        },
    )

    dataloader_pin_memory: bool = field(
        default=True, metadata={"help": "Whether or not to pin memory for DataLoader."}
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )

    full_determinism: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to call enable_full_determinism instead of set_seed for reproducibility in distributed"
                " training. Important: this will negatively impact the performance, so only use it for debugging."
            )
        },
    )

    torch_compile: bool = field(
        default=False,
        metadata={
            "help": "If set to `True`, the model will be wrapped in `torch.compile`."
        },
    )
    torch_compile_backend: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which backend to use with `torch.compile`, passing one will trigger a model compilation.",
        },
    )
    torch_compile_mode: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which mode to use with `torch.compile`, passing one will trigger a model compilation.",
        },
    )

    skip_memory_metrics: bool = field(
        default=False,
        metadata={
            "help": "Whether to skip adding of memory profiler reports to metrics."
        },
    )

    logging_steps: int = field(
        default=500,
        metadata={"help": "Log every X updates steps."},
    )

    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint every X updates steps."},
    )
    checkpoint_keep_steps: int = field(
        default=None,
        metadata={"help": "Maximum number of checkpoints to keep."},
    )

    save_on_each_node: bool = field(
        default=False,
        metadata={"help": "Whether to save checkpoint on each node or only master."},
    )

    max_seq_len: int = field(
        default=None,
        metadata={"help": "Max sequence length."},
    )

    def __post_init__(self):
        if self.lr_scheduler_type is not None:
            self.lr_scheduler_type = SchedulerType(self.lr_scheduler_type)
        self.optim = OptimizerNames(self.optim)

        if (
            self.torch_compile_mode is not None
            or self.torch_compile_backend is not None
        ) and not self.torch_compile:
            self.torch_compile = True
        if self.torch_compile and self.torch_compile_backend is None:
            self.torch_compile_backend = "inductor"

        # accelerate integration for torch compile
        if self.torch_compile:
            # set env vars for accelerate
            prefix = "ACCELERATE_DYNAMO_"
            os.environ[prefix + "BACKEND"] = self.torch_compile_backend
            if self.torch_compile_mode is not None:
                os.environ[prefix + "MODE"] = self.torch_compile_mode

        if is_torch_available() and self.torch_compile:
            if is_torch_tf32_available():
                if self.tf32 is None and not self.fp16 or self.bf16:
                    logger.info(
                        "Setting TF32 in CUDA backends to speedup torch compile, you won't see any improvement"
                        " otherwise."
                    )
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
            else:
                logger.warning(
                    "The speedups for torchdynamo mostly come wih GPU Ampere or higher and which is not detected here."
                )
        if is_torch_available() and self.tf32 is not None:
            if self.tf32:
                if is_torch_tf32_available():
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                else:
                    raise ValueError(
                        "--tf32 requires Ampere or a newer GPU arch, cuda>=11 and torch>=1.7"
                    )
            else:
                if is_torch_tf32_available():
                    torch.backends.cuda.matmul.allow_tf32 = False
                    torch.backends.cudnn.allow_tf32 = False
                # no need to assert on else

        # if training args is specified, it will override the one specified in the accelerate config
        if self.half_precision_backend != "apex":
            mixed_precision_dtype = os.environ.get("ACCELERATE_MIXED_PRECISION", "no")
            if self.fp16:
                mixed_precision_dtype = "fp16"
            elif self.bf16:
                mixed_precision_dtype = "bf16"
            os.environ["ACCELERATE_MIXED_PRECISION"] = mixed_precision_dtype

        if self.warmup_ratio < 0 or self.warmup_ratio > 1:
            raise ValueError("warmup_ratio must lie in range [0,1]")
        elif self.warmup_ratio > 0 and self.warmup_steps > 0:
            logger.info(
                "Both warmup_ratio and warmup_steps given, warmup_steps will override any effect of warmup_ratio"
                " during training"
            )

    def __str__(self):
        self_as_dict = asdict(self)

        # Remove deprecated arguments. That code should be removed once
        # those deprecated arguments are removed from TrainingArguments. (TODO: v5)
        if "per_gpu_train_batch_size" in self_as_dict:
            del self_as_dict["per_gpu_train_batch_size"]
        if "per_gpu_eval_batch_size" in self_as_dict:
            del self_as_dict["per_gpu_eval_batch_size"]

        self_as_dict = {
            k: f"<{k.upper()}>" if k.endswith("_token") else v
            for k, v in self_as_dict.items()
        }

        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__

    def get_warmup_steps(self, num_training_steps: int):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            self.warmup_steps
            if self.warmup_steps > 0
            else math.ceil(num_training_steps * self.warmup_ratio)
        )
        return warmup_steps

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.init
        }

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)
