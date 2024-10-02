from pathlib import Path
from typing import Optional, Union, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    PreTrainedModel,
    LlamaForCausalLM,
    PretrainedConfig,
    AutoConfig,
    AutoModel,
)

from treetune.common.py_utils import is_flash_attention_available
from treetune.logging_utils import get_logger
from treetune.models.base_model import Model
from treetune.models.pretrained import configure_dropout

logger = get_logger(__name__)


class PreTrainedModelForValueNetwork(Model, nn.Module):
    def __init__(
        self,
        pretrained_backbone_model: PreTrainedModel,
        value_head_dropout: Optional[float] = None,
    ):
        super().__init__()
        self.pretrained_model = pretrained_backbone_model
        self.config = self.pretrained_model.config

        hidden_size = self.pretrained_model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1, bias=True)
        self.dropout = (
            nn.Dropout(value_head_dropout)
            if value_head_dropout is not None
            else nn.Identity()
        )

        self._init_value_head()

    def _init_value_head(self):
        hidden_size = self.pretrained_model.config.hidden_size
        nn.init.normal_(self.value_head.weight, std=1 / np.sqrt(hidden_size + 1))
        nn.init.constant_(self.value_head.bias, val=0.0)

    @classmethod
    def from_di(
        cls,
        pretrained_backbone_model: Model,
        value_head_dropout: Optional[float] = None,
    ) -> nn.Module:
        return cls(pretrained_backbone_model, value_head_dropout=value_head_dropout)

    @property
    def device(self):
        return self.pretrained_model.device

    def gradient_checkpointing_enable(self):
        self.pretrained_model.gradient_checkpointing_enable()

    def forward(self, *args, **kwargs):
        """
        Forward pass of the model.
        Arg:
            *args: Variable length input arguments passed to the model.
            **kwargs: Variable length keyword arguments passed to the model.
        Returns:
            value: The value of the input sequence. Shape: (batch_size, sequence_length)
        """
        kwargs["output_hidden_states"] = True

        base_model_output = self.pretrained_model(*args, **kwargs)

        last_hidden_state = base_model_output.hidden_states[-1]

        output = self.dropout(last_hidden_state)

        # For now force upcast in fp32 if needed. Let's keep the
        # output in fp32 for numerical stability.
        if output.dtype != self.value_head.weight.dtype:
            output = output.to(self.value_head.weight.dtype)

        value = self.value_head(output)
        value = value.squeeze(-1)

        return value

    def save_pretrained(
        self,
        checkpoint_path: Union[str, Path],
        safe_serialization: Optional[bool] = None,
    ):
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            checkpoint_path.mkdir(parents=True)
        else:
            logger.warning(
                f"Checkpoint path {checkpoint_path} already exists and will be overwritten."
            )

        torch.save(self.state_dict(), checkpoint_path / "pytorch_model.bin")


class AlwaysConstantValueModelConfig(PretrainedConfig):
    def __init__(
        self,
        constant_value: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.constant_value = constant_value


class AlwaysConstantValueModel(Model, nn.Module):
    def __init__(
        self,
        config: AlwaysConstantValueModelConfig,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.config = config
        self.constant_value = config.constant_value
        self.device = device

        # Create a dummy parameter
        self.dummy_param = nn.Parameter(torch.tensor([0.0] * 16, device=device))

    def forward(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids", None)
        if input_ids is None:
            output = self.constant_value
        else:
            output = (
                torch.ones_like(input_ids, dtype=torch.float32) * self.constant_value
            )

        output += 0.0 * self.dummy_param.sum()

        return output

    def gradient_checkpointing_enable(self):
        pass

    @classmethod
    def from_di(
        cls,
        constant_value: float,
        device: Optional[torch.device] = None,
    ) -> nn.Module:
        config = AlwaysConstantValueModelConfig(constant_value=constant_value)
        return cls(config, device=device)

    def save_pretrained(
        self,
        checkpoint_path: Union[str, Path],
        safe_serialization: Optional[bool] = None,
    ):
        if not checkpoint_path.exists():
            checkpoint_path.mkdir(parents=True)
        else:
            logger.warning(
                f"Checkpoint path {checkpoint_path} already exists and will be overwritten."
            )

        torch.save(self.state_dict(), checkpoint_path / "pytorch_model.bin")


class LlamaRewardModel(Model, LlamaForCausalLM):
    def __init__(self, config):
        """
        We add this class to support pretrained reward model from:
        https://github.com/OpenLMLab/MOSS-RLHF/tree/main
        """
        super().__init__(config)
        self.reward_head = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, *args, **kwargs):
        kwargs.update(
            {
                "output_hidden_states": True,
                "return_dict": True,
                "use_cache": False,
            }
        )
        output = self.model.forward(*args, **kwargs)
        output = output.hidden_states[-1]

        # For now force upcast in fp32 if needed. Let's keep the
        # output in fp32 for numerical stability.
        if output.dtype != self.reward_head.weight.dtype:
            output = output.to(self.reward_head.weight.dtype)

        value = self.reward_head(output)
        value = value.squeeze(-1)

        return value

    @classmethod
    def from_di(
        cls,
        hf_model_name: str,
        pretrained_args: Optional[Dict[str, Any]] = None,
        device: Optional[Union[torch.device, str]] = None,
        disable_dropout: bool = False,
        runtime_hf_model_name: Optional[str] = None,
    ) -> "LlamaRewardModel":
        from accelerate import PartialState

        is_main_process = PartialState().is_main_process

        kwargs = {
            "use_flash_attention_2": pretrained_args.pop(
                "use_flash_attention_2", is_flash_attention_available()
            ),
            "torch_dtype": pretrained_args.pop("torch_dtype", torch.bfloat16),
        }

        if runtime_hf_model_name is not None:
            if is_main_process:
                logger.warning(
                    f"Overriding the `hf_model_name` with '{runtime_hf_model_name}' "
                    f"(original='{hf_model_name})'."
                )
            hf_model_name = runtime_hf_model_name

        if device is not None:
            kwargs["device_map"] = (
                device if isinstance(device, torch.device) else torch.device(device)
            )

        if disable_dropout:
            dropout_config = configure_dropout(hf_model_name, 0.0)
            if is_main_process:
                logger.info(f"Disabling dropout for keys: {dropout_config.keys()}")
            kwargs.update(dropout_config)

        model = cls.from_pretrained(hf_model_name, **kwargs)

        if disable_dropout and is_main_process:
            logger.info(f"Model config after disabling dropout: {model.config}")

        return model


class ScalarModelConfig(PretrainedConfig):
    def __init__(
        self,
        base_model: str = "EleutherAI/pythia-160m",
        base_config: PretrainedConfig = None,
        hidden_size: int = 768,
        bias: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if base_config is None:
            base_config = AutoConfig.from_pretrained("EleutherAI/pythia-160m")
        self.base_model = base_model
        self.base_config = base_config
        self.hidden_size = hidden_size
        self.bias = bias


class ScalarModel(Model, PreTrainedModel):
    config_class = ScalarModelConfig

    def __init__(self, config: ScalarModelConfig):
        super().__init__(config)
        self.config = config
        self.lm_backbone = AutoModel.from_pretrained(
            config.base_model,
            config=self.config.base_config,
            classifier_dropout=0.0,
            attn_implementation="flash_attention_2",
        )
        self.scalar_head = self.layer_init(
            nn.Linear(self.config.hidden_size, 1),
            std=1 / np.sqrt(self.config.hidden_size + 1),
        )

    def forward(self, **kwargs):
        kwargs["return_dict"] = True
        kwargs["output_hidden_states"] = True
        output = self.lm_backbone(**kwargs)
        reward = self.scalar_head(output.hidden_states[-1]) - self.config.bias
        reward = reward.squeeze(-1)
        return reward

    def gradient_checkpointing_enable(self):
        self.lm_backbone.gradient_checkpointing_enable()

    @classmethod
    def layer_init(cls, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.normal_(layer.weight, std=std)
        torch.nn.init.constant_(layer.bias, val=bias_const)
        return layer

    @classmethod
    def from_di(
        cls,
        hf_model_name: str,
        pretrained_args: Optional[Dict[str, Any]] = None,
        device: Optional[Union[torch.device, str]] = None,
        disable_dropout: bool = False,
        runtime_hf_model_name: Optional[str] = None,
    ) -> "LlamaRewardModel":
        from accelerate import PartialState

        is_main_process = PartialState().is_main_process

        pretrained_args = pretrained_args or {}

        kwargs = {
            "torch_dtype": pretrained_args.pop("torch_dtype", torch.bfloat16),
        }

        if runtime_hf_model_name is not None:
            if is_main_process:
                logger.warning(
                    f"Overriding the `hf_model_name` with '{runtime_hf_model_name}' "
                    f"(original='{hf_model_name})'."
                )
            hf_model_name = runtime_hf_model_name

        # if device is not None:
        #     kwargs["device_map"] = (
        #         device if isinstance(device, torch.device) else torch.device(device)
        #     )

        model = cls.from_pretrained(hf_model_name, **kwargs)

        return model


Model.register(
    "pretrained_causal_lm_with_value_head", constructor="from_di", exist_ok=True
)(PreTrainedModelForValueNetwork)

Model.register("always_constant_value_model", constructor="from_di", exist_ok=True)(
    AlwaysConstantValueModel
)

Model.register("moss_llama_reward_model", constructor="from_di", exist_ok=True)(
    LlamaRewardModel
)
Model.register("hf_tldr_reward_model", constructor="from_di", exist_ok=True)(
    ScalarModel
)
