from typing import Dict, Any, List

import torch

from treetune.common import Registrable

COLUMN_REF_SHIFTED_LOGPS = "ref_shifted_log_probs"  # We mean shifted to left by 1
COLUMN_ACTOR_SHIFTED_LOGPS = "actor_shifted_log_probs"  # We mean shifted to left by 1
COLUMN_VALUES = "critic_values"


class DataCollator(Registrable):
    def __call__(self, instances: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class PPODataCollator:
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
                        Shape: (batch_size, max_seq_len)
                - "attention_mask": The attention mask of the entire episode (query + responses).
                        Shape: (batch_size, max_seq_len)
                - "advantages": The advantages of the responses.
                        (batch_size, max_seq_len)
                - "scores": The scores of the responses. It should be a 1D scalar tensor.
                        Shape: (batch_size,)
                - "values": The values of the response states.
                        (batch_size, max_seq_len)
                - "ref_shifted_log_probs": The reference log probabilities of the responses.
                        Shape: (batch_size, max_seq_len-1)
                - "actor_shifted_log_probs": The actor log probabilities of the responses.
                        Shape: (batch_size, max_seq_len-1)
        """

        # Get the maximum sequence length
        max_seq_len = max(
            len(instance["query_token_ids"]) + len(instance["response_token_ids"])
            for instance in data_instances
        )

        # Create the batch
        batch = {"input_ids": [], "labels": [], "attention_mask": []}

        has_advantages = "advantages" in data_instances[0]
        if has_advantages:
            batch["advantages"] = []

        has_scores = "scores" in data_instances[0]
        if has_scores:
            batch["scores"] = []

        has_ref_shifted_logps = COLUMN_REF_SHIFTED_LOGPS in data_instances[0]
        if has_ref_shifted_logps:
            batch[COLUMN_REF_SHIFTED_LOGPS] = []

        has_actor_logps = COLUMN_ACTOR_SHIFTED_LOGPS in data_instances[0]
        if has_actor_logps:
            batch[COLUMN_ACTOR_SHIFTED_LOGPS] = []

        has_values = COLUMN_VALUES in data_instances[0]
        if has_values:
            batch[COLUMN_VALUES] = []

        pad_token_id = 0  # It doesn't matter what the pad token id is, since we will mask it out anyway
        pad_label = (
            -100
        )  # -100 is the default value for the padding token in the loss function
        pad_logp = -float(1e9)  # Crazy value to show up it in case of a bug
        pad_value = -float(1e9)  # Crazy value to show up it in case of a bug

        def prepare_shifted_logps(shifted_logps_with_query, query_len, response_len):
            assert len(shifted_logps_with_query) == (
                (query_len + response_len - 1)
            ), f"We assume the ref. log probs are provided for the entire sequence"
            shifted_logps_without_query = shifted_logps_with_query[query_len - 1 :]
            assert len(shifted_logps_without_query) == response_len

            n_pads_at_end = (max_seq_len - 1) - len(shifted_logps_with_query)
            shifted_logs = (
                [pad_logp] * (query_len - 1)
                + shifted_logps_without_query
                + [pad_logp] * n_pads_at_end
            )

            return shifted_logs

        for instance in data_instances:
            query_token_ids = instance["query_token_ids"]
            response_token_ids = instance["response_token_ids"]

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

            if has_advantages:
                advantages = instance["advantages"]
                advantages = (
                    [0.0] * len(query_token_ids) + advantages + [0.0] * num_pad_at_end
                )
                assert len(labels) == len(advantages)
                batch["advantages"].append(advantages)

            if has_scores:
                assert isinstance(instance["scores"], float)
                batch["scores"].append(instance["scores"])

            if has_ref_shifted_logps:
                shifted_ref_logps = prepare_shifted_logps(
                    instance[COLUMN_REF_SHIFTED_LOGPS],
                    len(query_token_ids),
                    len(response_token_ids),
                )
                assert len(shifted_ref_logps) == max_seq_len - 1
                batch[COLUMN_REF_SHIFTED_LOGPS].append(shifted_ref_logps)

            if has_actor_logps:
                shifted_actor_logps = prepare_shifted_logps(
                    instance[COLUMN_ACTOR_SHIFTED_LOGPS],
                    len(query_token_ids),
                    len(response_token_ids),
                )
                assert len(shifted_actor_logps) == (max_seq_len - 1)
                batch[COLUMN_ACTOR_SHIFTED_LOGPS].append(shifted_actor_logps)

            if has_values:
                values_with_query = instance[COLUMN_VALUES]
                # We also include the values for the last query token
                # since it's where we start generating the response
                values_without_query = values_with_query[len(query_token_ids) - 1 :]
                assert (len(values_without_query) - 1) == len(response_token_ids)
                values = (
                    [pad_value] * (len(query_token_ids) - 1)
                    + values_without_query
                    + [pad_value] * num_pad_at_end
                )
                assert len(values) == max_seq_len
                batch[COLUMN_VALUES].append(values)

        # Convert the lists to tensors
        batch = {k: torch.tensor(v) for k, v in batch.items()}

        return batch


COLUMN_ACCEPT_REF_SHIFTED_LOGPS = "accept_ref_shifted_log_probs"  # We mean shifted to left by 1
COLUMN_ACCEPT_ACTOR_SHIFTED_LOGPS = "accept_actor_shifted_log_probs"  # We mean shifted to left by 1
COLUMN_REJECT_REF_SHIFTED_LOGPS = "reject_ref_shifted_log_probs"  # We mean shifted to left by 1
COLUMN_REJECT_ACTOR_SHIFTED_LOGPS = "reject_actor_shifted_log_probs"  # We mean shifted to left by 1

class DPODataCollator:
    def __call__(self, data_instances: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collates the given data instances into a batch.
        Every data instance should have the following keys:
        - "query_token_ids": The token ids of the query.
        - "accept_response_token_ids": The token ids of the response.
        - "reject_response_token_ids": The token ids of the response.

        Args:
            data_instances (List[Dict[str, Any]]):
                The data instances to collate.
        Returns:
            Dict[str, torch.Tensor]:
                The collated batch.
                It contains the following keys:
                - "accept_input_ids": The token ids of the entire episode (query + responses).
                        Shape: (batch_size, max_seq_len)
                - "reject_input_ids": The token ids of the entire episode (query + responses).
                        Shape: (batch_size, max_seq_len)
                - "accept_labels": The labels of the entire episode (query + responses).
                        Shape: (batch_size, max_seq_len)
                - "reject_labels": The labels of the entire episode (query + responses).
                        Shape: (batch_size, max_seq_len)
                - "accept_attention_mask": The attention mask of the entire episode (query + responses).
                        Shape: (batch_size, max_seq_len)
                - "reject_attention_mask": The attention mask of the entire episode (query + responses).
                        Shape: (batch_size, max_seq_len)
                - "accept_ref_shifted_log_probs": The reference log probabilities of the responses.
                        Shape: (batch_size, max_seq_len-1)
                - "reject_ref_shifted_log_probs": The reference log probabilities of the responses.
                        Shape: (batch_size, max_seq_len-1)
                - "accept_actor_shifted_log_probs": The actor log probabilities of the responses.
                        Shape: (batch_size, max_seq_len-1)
                - "reject_actor_shifted_log_probs": The actor log probabilities of the responses.
                        Shape: (batch_size, max_seq_len-1)
        """

        # Get the maximum sequence length
        max_accept_seq_length = max(
            len(instance["query_token_ids"]) + len(instance["accept_response_token_ids"]) for instance in data_instances
        )
        max_reject_seq_length = max(
            len(instance["query_token_ids"]) + len(instance['reject_response_token_ids']) for instance in data_instances
        )
        max_seq_len = max(max_accept_seq_length, max_reject_seq_length)

        # Create the batch
        batch = {"accept_input_ids": [],
                 "accept_labels": [],
                 "accept_attention_mask": [],
                 COLUMN_ACCEPT_REF_SHIFTED_LOGPS: [],
                 COLUMN_ACCEPT_ACTOR_SHIFTED_LOGPS: [],
                 "reject_input_ids": [],
                 "reject_labels": [],
                 "reject_attention_mask": [],
                 COLUMN_REJECT_REF_SHIFTED_LOGPS: [],
                 COLUMN_REJECT_ACTOR_SHIFTED_LOGPS: [],
                 }

        pad_token_id = 0  # It doesn't matter what the pad token id is, since we will mask it out anyway
        pad_label = (
            -100
        )  # -100 is the default value for the padding token in the loss function

        pad_logp = -float(1e9)  # Crazy value to show up in case of a bug

        def prepare_shifted_logps(shifted_logps_with_query, query_len, response_len):
            assert len(shifted_logps_with_query) == (
                (query_len + response_len - 1)
            ), f"We assume the ref. log probs are provided for the entire sequence"
            assert query_len > 0
            shifted_logps_without_query = shifted_logps_with_query[query_len - 1:]
            assert len(shifted_logps_without_query) == response_len

            n_pads_at_end = (max_seq_len - 1) - len(shifted_logps_with_query)
            shifted_logs = (
                [pad_logp] * (query_len - 1)
                + shifted_logps_without_query
                + [pad_logp] * n_pads_at_end
            )

            return shifted_logs

        for instance in data_instances:
            query_token_ids = instance["query_token_ids"]
            accept_response_token_ids = instance["accept_response_token_ids"]
            reject_response_token_ids = instance["reject_response_token_ids"]

            # Create the input ids and attention mask
            accept_input_ids = query_token_ids + accept_response_token_ids
            accept_attention_mask = [1] * len(accept_input_ids)
            accept_num_pad_at_end = max_seq_len - len(accept_input_ids)
            accept_input_ids += [pad_token_id] * accept_num_pad_at_end
            accept_attention_mask += [0] * accept_num_pad_at_end

            reject_input_ids = query_token_ids + reject_response_token_ids
            reject_attention_mask = [1] * len(reject_input_ids)
            reject_num_pad_at_end = max_seq_len - len(reject_input_ids)
            reject_input_ids += [pad_token_id] * reject_num_pad_at_end
            reject_attention_mask += [0] * reject_num_pad_at_end

            batch["accept_input_ids"].append(accept_input_ids)
            batch["accept_attention_mask"].append(accept_attention_mask)
            batch["reject_input_ids"].append(reject_input_ids)
            batch["reject_attention_mask"].append(reject_attention_mask)

            # Create the labels
            accept_labels = (
                [pad_label] * len(query_token_ids)
                + accept_response_token_ids
                + [pad_label] * accept_num_pad_at_end
            )
            batch["accept_labels"].append(accept_labels)

            reject_labels = (
                [pad_label] * len(query_token_ids)
                + reject_response_token_ids
                + [pad_label] * reject_num_pad_at_end
            )
            batch["reject_labels"].append(reject_labels)

            if COLUMN_ACCEPT_REF_SHIFTED_LOGPS in instance:
                accept_shifted_ref_logps = prepare_shifted_logps(
                    instance[COLUMN_ACCEPT_REF_SHIFTED_LOGPS],
                    len(query_token_ids),
                    len(accept_response_token_ids),
                )
                assert len(accept_shifted_ref_logps) == max_seq_len - 1
                batch[COLUMN_ACCEPT_REF_SHIFTED_LOGPS].append(accept_shifted_ref_logps)

            if COLUMN_ACCEPT_ACTOR_SHIFTED_LOGPS in instance:
                accept_shifted_actor_logps = prepare_shifted_logps(
                    instance[COLUMN_ACCEPT_ACTOR_SHIFTED_LOGPS],
                    len(query_token_ids),
                    len(accept_response_token_ids),
                )
                assert len(accept_shifted_actor_logps) == max_seq_len - 1
                batch[COLUMN_ACCEPT_ACTOR_SHIFTED_LOGPS].append(accept_shifted_actor_logps)

            if COLUMN_REJECT_REF_SHIFTED_LOGPS in instance:
                reject_shifted_ref_logps = prepare_shifted_logps(
                    instance[COLUMN_REJECT_REF_SHIFTED_LOGPS],
                    len(query_token_ids),
                    len(reject_response_token_ids),
                )
                assert len(reject_shifted_ref_logps) == max_seq_len - 1
                batch[COLUMN_REJECT_REF_SHIFTED_LOGPS].append(reject_shifted_ref_logps)

            if COLUMN_REJECT_ACTOR_SHIFTED_LOGPS in instance:
                reject_shifted_actor_logps = prepare_shifted_logps(
                    instance[COLUMN_REJECT_ACTOR_SHIFTED_LOGPS],
                    len(query_token_ids),
                    len(reject_response_token_ids),
                )
                assert len(reject_shifted_actor_logps) == max_seq_len - 1
                batch[COLUMN_REJECT_ACTOR_SHIFTED_LOGPS].append(reject_shifted_actor_logps)

        # Convert the lists to tensors
        batch = {k: torch.tensor(v) for k, v in batch.items()}

        return batch
