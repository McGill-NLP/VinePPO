from typing import List, Dict, Any, Literal, Tuple

import torch
import torch.nn.functional as F
from accelerate.utils import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from treetune.logging_utils import get_logger
from treetune.trainers.base_trainer import Trainer
from treetune.trainers.mle_trainer import MaximumLikelihoodTrainer
from treetune.trainers.utils import masked_mean, entropy_from_logits

logger = get_logger(__name__)


class DPODataCollator:
    def __call__(self, data_instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collates the given data instances into a batch.
        Every data instance should have the following keys:
        - "query_token_ids" (List[int]): The token ids of the query.
        - "accept_response_token_ids" (List[int]): The token ids of the accepted response.
        - "list_of_reject_response_token_ids" (List[List[int]]): The token ids of the rejected responses.

        Args:
            data_instances (List[Dict[str, Any]]):
                The data instances to collate.
        Returns:
            Dict[str, Any]:
                The collated batch.
                It contains the following keys:
                - "accept" (Dict[str, torch.Tensor]): The collated batch for the accepted responses.
                    - "input_ids": The input ids of the accepted responses.
                    - "labels": The labels of the accepted responses.
                    - "attention_mask": The attention mask of the accepted responses.
                - "rejects" (List[Dict[str, torch.Tensor]]): The collated batch for the rejected responses.
        """

        # Get the maximum sequence length
        max_accept_seq_len = max(
            len(instance["query_token_ids"])
            + len(instance["accept_response_token_ids"])
            for instance in data_instances
        )

        num_reject_seqs = len(data_instances[0]["list_of_reject_response_token_ids"])
        max_reject_seq_len = [-1] * num_reject_seqs
        for i in range(num_reject_seqs):
            max_reject_seq_len[i] = max(
                len(instance["query_token_ids"])
                + len(instance["list_of_reject_response_token_ids"][i])
                for instance in data_instances
            )

        # Check if the reference model logps are present
        has_ref_model_logps = "reference_accept_logps" in data_instances[0] and all(
            f"reference_reject_logps_{i}" in data_instances[0]
            for i in range(num_reject_seqs)
        )

        # Create the batch
        batch = {
            "accept": {
                "input_ids": [],
                "labels": [],
                "attention_mask": [],
            },
            "rejects": [
                {
                    "input_ids": [],
                    "labels": [],
                    "attention_mask": [],
                }
                for _ in range(num_reject_seqs)
            ],
        }
        if has_ref_model_logps:
            batch["reference_accept_logps"] = [
                d["reference_accept_logps"] for d in data_instances
            ]
            for i in range(num_reject_seqs):
                batch[f"reference_reject_logps_{i}"] = [
                    d[f"reference_reject_logps_{i}"] for d in data_instances
                ]

        # It doesn't matter what the pad token id is, since we will mask it out anyway
        pad_token_id = 0
        pad_label = -100

        def get_padded_input_ids_attention_mask_and_labels(
            query_tok_ids, response_tok_ids, max_seq_len
        ):
            # Create the input ids and attention mask
            input_ids = query_tok_ids + response_tok_ids
            attention_mask = [1] * len(input_ids)
            num_pad_at_end = max_seq_len - len(input_ids)

            # Pad the input ids and attention mask at the end to the maximum sequence length
            input_ids += [pad_token_id] * num_pad_at_end
            attention_mask += [0] * num_pad_at_end

            # Create the labels
            # Mask out the query tokens at the start and also mask out the padding tokens at the end
            labels = (
                [pad_label] * len(query_tok_ids)
                + response_tok_ids
                + [pad_label] * num_pad_at_end
            )
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        for instance in data_instances:
            query_token_ids = instance["query_token_ids"]
            accept_response_token_ids = instance["accept_response_token_ids"]

            # Create the input ids, attention mask and labels for the accepted response
            accept_batch = get_padded_input_ids_attention_mask_and_labels(
                query_token_ids, accept_response_token_ids, max_accept_seq_len
            )
            for k, v in accept_batch.items():
                batch["accept"][k].append(v)

            # Create the input ids, attention mask and labels for the rejected responses
            for i in range(num_reject_seqs):
                reject_sequences = instance["list_of_reject_response_token_ids"]
                reject_response_token_ids = reject_sequences[i]
                reject_batch = get_padded_input_ids_attention_mask_and_labels(
                    query_token_ids, reject_response_token_ids, max_reject_seq_len[i]
                )
                for k, v in reject_batch.items():
                    batch["rejects"][i][k].append(v)

        # Convert the lists to tensors
        for k in batch["accept"]:
            batch["accept"][k] = torch.tensor(batch["accept"][k])
        for i in range(num_reject_seqs):
            for k in batch["rejects"][i]:
                batch["rejects"][i][k] = torch.tensor(batch["rejects"][i][k])
        if has_ref_model_logps:
            batch["reference_accept_logps"] = torch.tensor(
                batch["reference_accept_logps"]
            )
            for i in range(num_reject_seqs):
                batch[f"reference_reject_logps_{i}"] = torch.tensor(
                    batch[f"reference_reject_logps_{i}"]
                )

        return batch


@Trainer.register("dpo")
class DPOTrainer(MaximumLikelihoodTrainer):
    def __init__(
        self,
        dpo_beta: float = 0.1,
        dpo_loss_type: Literal["sigmoid", "ipo"] = "sigmoid",
        dpo_sequence_logp_reduction: Literal["sum", "mean"] = "sum",
        dpo_label_smoothing: float = 0.0,
        dpo_use_reference_model: bool = False,
        dpo_stop_grad_reject_logps: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_collator = DPODataCollator()
        self.dpo_beta = dpo_beta
        self.dpo_loss_type = dpo_loss_type
        self.dpo_sequence_logp_reduction = dpo_sequence_logp_reduction
        self.dpo_label_smoothing = dpo_label_smoothing
        self.dpo_use_reference_model = dpo_use_reference_model
        self.dpo_stop_grad_reject_logps = dpo_stop_grad_reject_logps

        if getattr(self.args, "gradient_checkpointing", False):
            # For models that use gradient_checkpoiting, we need to attach a hook that enables input
            # to explicitly have `requires_grad=True`, otherwise training will either silently
            # fail or completely fail.
            # See: https://github.com/huggingface/trl/blob/88685f2cd476b335918fad6104cd78e4199003c3/trl/trainer/dpo_trainer.py#L250
            self.accelerator.unwrap_model(self.model).enable_input_require_grads()

        self._set_process_log_level(logger)

    def step(self, episodes_dataset: Dataset) -> None:
        if self.dpo_use_reference_model:
            assert (
                self.state.iteration == 0
            ), "Only the first iteration is supported for now."

            # Try to load the reference model logps from the cache
            cache_path = (
                self.checkpoints_dir
                / f"dpo_episodes_{self.state.iteration}_w_ref_logps"
            )
            if cache_path.exists():
                logger.info(
                    f"Loading episodes dataset with reference model logps from {cache_path}"
                )
                episodes_dataset = Dataset.load_from_disk(cache_path)
            else:
                # Make sure we're at the beginning of the training as we need an untouched model
                assert (
                    self.state.global_step == 0
                ), "Reference model logps can only be computed at the start of training"

                logger.info(
                    "Precomputing reference model logps for the episodes dataset"
                )
                episodes_dataset = self._precompute_reference_model_logps(
                    self.model, episodes_dataset
                )
                episodes_dataset.save_to_disk(cache_path)

        super().step(episodes_dataset)

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
                - "accept" (Dict[str, torch.Tensor]): The inputs for the accepted responses.
                    - "input_ids": The input ids of the accepted responses.
                    - "labels": The labels of the accepted responses.
                    - "attention_mask": The attention mask of the accepted responses.
                - "rejects" (List[Dict[str, torch.Tensor]]): The inputs for the rejected responses.
        Returns:
            torch.Tensor: The loss.
        """
        accept_seq = inputs["accept"]
        reject_seqs = inputs["rejects"]
        assert (
            len(reject_seqs) == 1
        ), "At present, only one reject sequence is supported"

        accept_output = self._forward_pass(model, accept_seq)
        if "reference_accept_logps" in inputs:
            accept_output["ref_logps"] = inputs["reference_accept_logps"]

        reject_outputs = [
            self._forward_pass(model, reject_seq) for reject_seq in reject_seqs
        ]
        for i, reject_output in enumerate(reject_outputs):
            if f"reference_reject_logps_{i}" in inputs:
                reject_output["ref_logps"] = inputs[f"reference_reject_logps_{i}"]

        loss, accept_rewards, reject_rewards = self._compute_dpo_loss(
            accept_output, reject_outputs
        )

        metrics = {
            "reward/accuracy": (accept_rewards > reject_rewards).float().mean(),
            "reward/accept": accept_rewards.mean(),
            "reward/reject": reject_rewards.mean(),
            "reward/margin": (accept_rewards - reject_rewards).mean(),
            "logps/accept": accept_output["logps"].detach().clone().mean(),
            "entropy/accept": accept_output["entropy"].mean(),
            **{
                f"logps/reject_{i}": reject_output["logps"].detach().clone().mean()
                for i, reject_output in enumerate(reject_outputs)
            },
            **{
                f"entropy/reject_{i}": reject_output["entropy"].mean()
                for i, reject_output in enumerate(reject_outputs)
            },
        }

        return loss, metrics

    def _compute_dpo_loss(
        self, accept_output: Dict[str, Any], reject_outputs: List[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the DPO loss.

        Args:
            accept_output (Dict[str, Any]):
                The output for the accepted response. It contains the following keys:
                - "logps": The log probabilities of the accepted response.
                - "ref_logps": The log probabilities of the accepted response under the reference model.
            reject_outputs (List[Dict[str, Any]]):
                The outputs for the rejected responses. Each output contains the following keys:
                - "logps": The log probabilities of the rejected response.
                - "ref_logps": The log probabilities of the rejected response under the reference model.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                The loss, the rewards for the accepted response and the rewards for the rejected responses.
        """
        assert (
            len(reject_outputs) == 1
        ), "At present, only one reject sequence is supported"

        accept_logps = accept_output["logps"]
        reject_logps = reject_outputs[0]["logps"]

        if self.dpo_stop_grad_reject_logps:
            reject_logps = reject_logps.detach()

        policy_log_ratios = accept_logps - reject_logps

        if self.dpo_use_reference_model:
            ref_accept_logps = accept_output["ref_logps"]
            ref_reject_logps = reject_outputs[0]["ref_logps"]

            ref_log_ratios = ref_accept_logps - ref_reject_logps
        else:
            ref_log_ratios = 0.0

        logits = policy_log_ratios - ref_log_ratios

        if self.dpo_loss_type == "sigmoid":
            loss = (
                -F.logsigmoid(self.dpo_beta * logits) * (1 - self.dpo_label_smoothing)
                - F.logsigmoid(-self.dpo_beta * logits) * self.dpo_label_smoothing
            )
        elif self.dpo_loss_type == "ipo":
            loss = (logits - 1 / (2 * self.dpo_beta)) ** 2
        else:
            raise ValueError(f"Invalid DPO loss type: {self.dpo_loss_type}")

        if self.dpo_use_reference_model:
            accept_rewards = self.dpo_beta * (accept_logps - ref_accept_logps)
            reject_rewards = self.dpo_beta * (reject_logps - ref_reject_logps)
        else:
            accept_rewards = self.dpo_beta * accept_logps
            reject_rewards = self.dpo_beta * reject_logps

        return (
            loss.mean(),
            accept_rewards.detach().clone(),
            reject_rewards.detach().clone(),
        )

    def _forward_pass(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        return_logits: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass for the model.

        Args:
            model (PreTrainedModel): The model to forward pass.
            inputs (Dict[str, torch.Tensor]): The inputs to the model, containing the following keys:
                - "input_ids": The input ids of the sequence.
                - "labels": The labels for the sequence.
                - "attention_mask": The attention mask of the sequence.
        Returns:
            Dict[str, Any]: The outputs containing the following keys:
                - "logits": The logits for the sequence.
                - "logps": The log probabilities of the sequence.
        """
        input_ids: torch.Tensor = inputs["input_ids"]
        labels: torch.Tensor = inputs["labels"]
        attention_mask: torch.Tensor = inputs["attention_mask"]

        if not self.is_flash_attention_model:
            outputs: CausalLMOutputWithPast = model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
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

        # Compute the loss in full precision
        logits = logits.to(torch.float32)

        # Shift so that tokens < n predict n
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

        sequence_log_probs = per_token_log_probs.sum(dim=-1)
        if self.dpo_sequence_logp_reduction == "mean":
            sequence_log_probs = sequence_log_probs / shift_label_mask.sum(dim=-1)

        mean_entropy = masked_mean(entropy_from_logits(shift_logits), shift_label_mask)
        mean_entropy = mean_entropy.detach().clone()

        output = {"logps": sequence_log_probs, "entropy": mean_entropy}
        if return_logits:
            output["logits"] = logits

        return output

    def _precompute_reference_model_logps(
        self, ref_model: PreTrainedModel, dataset: Dataset
    ):
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_num_workers > 0,
            shuffle=False,
        )
        data_loader = self.accelerator.prepare(data_loader)

        num_reject_seqs = len(dataset[0]["list_of_reject_response_token_ids"])

        ref_model.eval()

        reference_accept_logps = []
        reference_list_of_reject_logps = [[] for _ in range(num_reject_seqs)]
        for inputs in tqdm(
            iterable=data_loader, desc="Computing reference model log probs"
        ):
            for seq_input_dict, output_lst in zip(
                [inputs["accept"], *inputs["rejects"]],
                [reference_accept_logps, *reference_list_of_reject_logps],
            ):
                with torch.no_grad(), self.accelerator.autocast():
                    outputs = self._forward_pass(ref_model, seq_input_dict)
                    logps = outputs["logps"].float().detach().clone()
                    logps = self.accelerator.gather_for_metrics(logps)
                    output_lst.append(logps.cpu())

        reference_accept_logps = torch.cat(reference_accept_logps).float().numpy()
        reference_list_of_reject_logps = [
            torch.cat(logps).numpy() for logps in reference_list_of_reject_logps
        ]

        # Remove the data loader from accelerator
        self.accelerator._dataloaders.pop()

        dataset = dataset.add_column(
            name="reference_accept_logps", column=reference_accept_logps
        )

        for i in range(num_reject_seqs):
            dataset = dataset.add_column(
                name=f"reference_reject_logps_{i}",
                column=reference_list_of_reject_logps[i],
            )

        return dataset

    def _get_instance_length_filter_fn(self, max_seq_len: int):
        def filter_fn(example):
            return all(
                (len(example["query_token_ids"]) + len(response_tokens)) <= max_seq_len
                for response_tokens in [
                    example["accept_response_token_ids"],
                    *example["list_of_reject_response_token_ids"],
                ]
            )

        return filter_fn
