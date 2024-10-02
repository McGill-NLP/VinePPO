from typing import List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from treetune.logging_utils import get_logger
from treetune.trainers.base_trainer import Trainer
from treetune.trainers.mle_trainer import MaximumLikelihoodTrainer
from treetune.trainers.utils import masked_mean

logger = get_logger(__name__)

class ValueDataCollator:
    def __call__(self, data_instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collates the given data instances into a batch.
        Every data instance should have the following keys:
        - "query_token_ids" (List[int]): The token ids of the query.
        - "response_token_ids" (List[int]): The token ids of the response.
        - "value_targets" (List[int]): The targets for value network to predict.

        Args:
            data_instances (List[Dict[str, Any]]):
                The data instances to collate.
        Returns:
            Dict[str, Any]:
                The collated batch.
                It contains the following keys:
                - "responses" (Dict[str, torch.Tensor]): The collated batch for the accepted responses.
                    - "input_ids": The input ids of the accepted responses.
                    - "attention_mask": The attention mask of the accepted responses.
                    - "value_targets": The targets for the value network to predict.
                    - "value_loss_mask": The mask for the value loss.
        """

        # Get the maximum sequence length
        max_accept_seq_len = max(
            len(instance["query_token_ids"])
            + len(instance["response_token_ids"])
            for instance in data_instances
        )

        # Create the batch
        batch = {
            "responses": {
                "input_ids": [],
                "value_targets": [],
                "attention_mask": [],
                "value_loss_mask": [],
            },
        }

        # It doesn't matter what the pad token id is, since we will mask it out anyway
        pad_token_id = 0
        pad_value = -int(1e9) # be aware, pytorch does not automatically avoid it, we need to mask it out manually,
        # set it to crazy number to increase the change of it showing up in case of a bug

        def get_padded_input_ids_attention_mask_value_targets_and_value_loss_mask(
            query_tok_ids, response_tok_ids, max_seq_len, value_targets
        ):
            assert len(response_tok_ids) == len(value_targets), f"Response tokens and value targets should have the same length. Got {len(response_tok_ids)} and {len(value_targets)} respectively."
            # Create the input ids and attention mask
            input_ids = query_tok_ids + response_tok_ids
            attention_mask = [1] * len(input_ids)
            num_pad_at_end = max_seq_len - len(input_ids)

            # Pad the input ids and attention mask at the end to the maximum sequence length
            input_ids += [pad_token_id] * num_pad_at_end
            attention_mask += [0] * num_pad_at_end

            # Create the values
            # set the query tokens at the start and also mask out the padding tokens at the end to crazy values
            value_targets = (
                [pad_value] * len(query_tok_ids)
                + value_targets
                + [pad_value] * num_pad_at_end
            )

            value_loss_mask = (
                [0.] * len(query_tok_ids)
                + [1.] * len(response_tok_ids)
                + [0.] * num_pad_at_end
            )


            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "value_targets": value_targets,
                "value_loss_mask": value_loss_mask,
            }

        for instance in data_instances:
            query_token_ids = instance["query_token_ids"]
            response_token_ids = instance["response_token_ids"]
            value_targets = instance["value_network_targets"]

            # Create the input ids, attention mask and labels for the accepted response
            response_batch = get_padded_input_ids_attention_mask_value_targets_and_value_loss_mask(
                query_token_ids, response_token_ids, max_accept_seq_len, value_targets
            )
            for k, v in response_batch.items():
                batch["responses"][k].append(v)

        # Convert the lists to tensors
        for k in batch["responses"]:
            batch["responses"][k] = torch.tensor(batch["responses"][k])

        return batch

@Trainer.register("value_network", exist_ok=True)
class ValueNetworkTrainer(MaximumLikelihoodTrainer): # really, MLETrainer is not a good name for the father, it just has useful methods
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_collator = ValueDataCollator()

        if getattr(self.args, "gradient_checkpointing", False):
            # For models that use gradient_checkpointing, we need to attach a hook that enables input
            # to explicitly have `requires_grad=True`, otherwise training will either silently
            # fail or completely fail.
            # See: https://github.com/huggingface/trl/blob/88685f2cd476b335918fad6104cd78e4199003c3/trl/trainer/dpo_trainer.py#L250
            self.accelerator.unwrap_model(self.model).enable_input_require_grads()

        self._set_process_log_level(logger)

    def compute_loss(
        self, model: PreTrainedModel, inputs: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Computes the loss for the given inputs.

        Args:
            model (PreTrainedModel): The model to compute the loss for.
            inputs (Dict[str, Any]):
                The inputs to the model.
                It should contain the following keys:
                - "responses" (Dict[str, torch.Tensor]): The inputs for the accepted responses.
                    - "input_ids": The input ids of the accepted responses.
                    - "value_targets": The targets for the value network to predict.
                    - "attention_mask": The attention mask of the responses.
                    - "value_loss_mask": The mask for the value loss.

        Returns:
            torch.Tensor: The loss.
        """
        responses = inputs["responses"]

        output = self._forward_pass(model, responses)

        loss = self._compute_value_loss(output["predicted_values"], responses["value_targets"], responses["value_loss_mask"])

        value_loss_mask = responses["value_loss_mask"]
        predicted_value_mean = masked_mean(output["predicted_values"], value_loss_mask)
        ground_truth_value_mean = masked_mean(responses["value_targets"], value_loss_mask)
        metrics = {
            "pred_values": predicted_value_mean.detach(),
            "ground_truth_values": ground_truth_value_mean.detach(),
        }

        return loss, metrics

    def _compute_value_loss(
        self,
        value_predictions: torch.Tensor,
        value_targets: torch.Tensor,
        value_loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the value network MSE loss.

        Args:
            - "value_predictions"
            - "value_targets"
            - "value_loss_mask"
        Returns:
            torch.Tensor:
                The mse loss
        """

        # check lengths of value_targets and value_predictions
        assert value_targets.size() == value_predictions.size() == value_loss_mask.size(), f"Value targets, predictions and loss mask should have the same size. Got {value_targets.size()}, {value_predictions.size()} and {value_loss_mask.size()} respectively."

        # make sure value targets are float
        value_targets = value_targets.to(torch.float32)

        loss = F.mse_loss(
            value_predictions,
            value_targets,
            reduction="none",
        )

        # zero out the loss for the masked tokens
        loss = masked_mean(loss, value_loss_mask)

        return loss

    def _forward_pass(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Args:
            model (PreTrainedModel):
            inputs (Dict[str, torch.Tensor]):
                - "input_ids"
                - "attention_mask"
        Returns:
            Dict[str, Any]:
                - "predicted_values"
        """
        input_ids: torch.Tensor = inputs["input_ids"]
        attention_mask: torch.Tensor = inputs["attention_mask"]

        if not self.is_flash_attention_model:
            outputs: CausalLMOutputWithPast = model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True,
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

        predicted_values = outputs  # Shape: (batch_size, max_seq_len)

        # Compute the loss in full precision
        predicted_values = predicted_values.to(torch.float32)

        return {
            "predicted_values": predicted_values,
        }

    def _get_instance_length_filter_fn(self, max_seq_len: int):
        def filter_fn(example):
            return (len(example["query_token_ids"]) + len(example["response_token_ids"])) <= max_seq_len

        return filter_fn

    # todo: I hate the fact that upscale_advantage is a part of trainer just because we inherited things from MLETrainer


