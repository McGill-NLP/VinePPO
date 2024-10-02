# REST(EM) Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models https://arxiv.org/abs/2312.06585
import json
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, Union, List, Optional

import numpy as np
import wandb
from datasets import Dataset

from treetune.episode_generators import EpisodeGenerator, MathEpisodeGenerator
from treetune.episode_generators.base_episode_generator import Episode
from treetune.logging_utils import get_logger

from tqdm import tqdm

logger = get_logger(__name__)


# so I could use DPOEpisode from  base_episode_generator.py, but I didn't like it as it has a list of reject responses, which is not needed here.
@dataclass
class DPOPositiveEpisode:
    query_token_ids: List[int]
    accept_response_token_ids: List[int]
    reject_response_token_ids: List[int]

    def __post_init__(self):
        assert len(self.query_token_ids) > 0
        assert len(self.accept_response_token_ids) > 0
        assert len(self.reject_response_token_ids) > 0

@EpisodeGenerator.register("math_dpo_positive_episode_generator")
class MATHDPOPositiveEpisodeGenerator(MathEpisodeGenerator):
    def __init__(
        self,
        reward_threshold: float = 0.5,  # equal and higher than reward_threshold is considered correct, lower is incorrect
        max_pairs_per_question: int = 8,  # -1: no limit, 10 is taken from theSELF-EXPLORE to Avoid the PIT: Improving the Reasoning Capabilities of Language Models with Fine-grained Rewards paper https://arxiv.org/pdf/2404.10346
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reward_threshold = reward_threshold
        self.max_pairs_per_question = max_pairs_per_question
        self._logger = logger

    def _generate_episodes(
        self,
        inference_results: Dataset,
        iteration: int,
    ) -> List[Union[Dict[str, Any], Episode]]:
        # copy-pasted and modified from math_episode_generator.py,
        episodes_dict = {}  # from idx to list of episodes
        metrics = {}
        encountered_question_indices = []
        question_reached_max_response = []
        skipped_questions = 0

        for instance in tqdm(inference_results, desc="Generating episodes"):
            all_rewards = []
            this_question_episodes = []

            tree = json.loads(instance["_treetune__reasoning_tree"])

            idx = instance["_treetune__idx"]
            assert idx not in encountered_question_indices, f"Question {idx} is encountered more than once in inference_result."
            encountered_question_indices.append(idx)

            paths = self.extract_paths_from_tree(tree)
            for path in paths:
                assert len(path["node_chain"]) == 2, "Does not support multi-hop paths. just query and response."

                finish_reason = path["node_chain"][-1]["finish_reason"]
                query_text = path["node_chain"][0]["text"]
                full_text = path["node_chain"][-1]["full_text"]
                response_text = full_text[len(query_text):]

                try:
                    num_reasoning_steps = self.compute_number_of_reasoning_steps(
                        response_text
                    )
                    metrics.setdefault("num_reasoning_steps", []).append(
                        num_reasoning_steps
                    )
                    metrics.setdefault("parse_failed", []).append(False)
                except Exception as e:
                    logger.error(f"Parsing reasoning steps failed {e}")
                    logger.error(f"Response: `{response_text}`")
                    metrics.setdefault("parse_failed", []).append(True)

                if finish_reason != "length":
                    # Generation stopped because the model hit <eos>
                    reward, is_unfinished_response = self.reward_function(
                        query_text, response_text, instance
                    )
                else:
                    # Generation stopped because the model hit the `max_tokens` limit
                    reward = self.reward_function.get_unfinished_response_penalty()
                    is_unfinished_response = True

                try:
                    query_token_ids, response_token_ids = (
                        self._tokenize_query_and_response(
                            query_text,
                            response_text,
                            # Only append EOS token if the response is complete
                            allow_append_eos=not is_unfinished_response,
                        )
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to tokenize query and response for instance {instance['_treetune__idx']}: {e}"
                    )
                    logger.error(f"Query: {query_text}")
                    logger.error(f"Response: {response_text}")
                    metrics.setdefault("empty_response", []).append(True)
                    continue

                if self.max_sequence_length is not None:
                    seq_len = len(query_token_ids) + len(response_token_ids)
                    if seq_len > self.max_sequence_length:
                        logger.warning(
                            f"Sequence length {seq_len} is greater than "
                            f"max sequence length {self.max_sequence_length}."
                        )

                        # Truncate the response
                        response_token_ids = response_token_ids[
                                             : self.max_sequence_length - len(query_token_ids)
                                             ]
                        reward = self.reward_function.get_unfinished_response_penalty()
                        is_unfinished_response = True

                episode = {'query_token_ids': query_token_ids, 'response_token_ids': response_token_ids, 'reward': reward}
                this_question_episodes.append(episode)
                all_rewards.append(float(reward))

                if len(response_token_ids) == 0:
                    logger.warning(
                        f"Response token ids are empty for instance {instance['_treetune__idx']}"
                    )
                    metrics.setdefault("empty_response", []).append(True)
                    continue

                metrics.setdefault("empty_response", []).append(False)
                metrics.setdefault("is_unfinished_response", []).append(
                    is_unfinished_response
                )

            # now that we have all the episodes for this question, we craft the DPOPositiveEpisodes
            acc_episodes = []
            rej_episodes = []
            for episode in this_question_episodes:
                if episode['reward'] >= self.reward_threshold:
                    acc_episodes.append(episode)
                else:
                    rej_episodes.append(episode)

            if len(acc_episodes) == 0 or len(rej_episodes) == 0:
                logger.warning(f"Question {idx} has {len(acc_episodes)} accept responses and {len(rej_episodes)} reject responses. Skipping this question.")
                skipped_questions += 1
            else:
                candidate_episodes = []
                for acc_episode in acc_episodes:
                    for rej_episode in rej_episodes:
                        assert acc_episode['query_token_ids'] == rej_episode['query_token_ids'], "Query token ids are different for accept and reject episodes."
                        dpo_episode = DPOPositiveEpisode(query_token_ids=acc_episode['query_token_ids'], accept_response_token_ids=acc_episode['response_token_ids'], reject_response_token_ids=rej_episode['response_token_ids'])
                        candidate_episodes.append(dpo_episode)

                # now we have all the candidate DPOPositiveEpisodes for this question, we sample max_pairs_per_question of them
                if self.max_pairs_per_question != -1 and len(candidate_episodes) > self.max_pairs_per_question:
                    chosen_episodes = np.random.choice(candidate_episodes, min(self.max_pairs_per_question, len(candidate_episodes)), replace=False)
                else:
                    chosen_episodes = candidate_episodes

                for episode in chosen_episodes:
                    episodes_dict.setdefault(idx, []).append(episode)

            if len(all_rewards) > 0:
                once_hit = any([r > self.reward_threshold for r in all_rewards])
                metrics.setdefault("once_hit", []).append(float(once_hit))

        # unravel episodes
        episodes = []
        for idx, episode_list in episodes_dict.items():
            for episode in episode_list:
                episodes.append(episode)

        if "is_unfinished_response" in metrics:
            metrics["is_unfinished_response"] = sum(
                metrics["is_unfinished_response"]
            ) / len(metrics["is_unfinished_response"])

        if "empty_response" in metrics:
            metrics["empty_response"] = sum(metrics["empty_response"]) / len(
                metrics["empty_response"]
            )

        if "num_reasoning_steps" in metrics:
            num_reasoning_steps = np.array(metrics.pop("num_reasoning_steps"))
            metrics["num_reasoning_steps/dist"] = num_reasoning_steps
            metrics["num_reasoning_steps/mean"] = np.mean(num_reasoning_steps)

        if "parse_failed" in metrics:
            metrics["parse_failed"] = sum(metrics["parse_failed"]) / len(
                metrics["parse_failed"]
            )

        if "once_hit" in metrics:
            metrics["once_hit"] = sum(metrics["once_hit"]) / len(metrics["once_hit"])

        metrics["question_reached_max_response"] = len(question_reached_max_response)
        metrics["total_final_episodes"] = len(episodes)

        if len(metrics) > 0:
            logs = {f"episodes_metric/{k}": v for k, v in metrics.items()}
            self._cloud_log({**logs, "train/global_iteration": iteration})

        return episodes

    def log_episodes(
        self,
        episodes: Union[List[Episode], Dataset],
        iteration_idx: int,
        num_examples: int = 100,
        num_examples_for_wandb: int = 128,
        seed: int = 42,
        log_to_cloud: bool = True,
    ):
        if not self.is_main_process():
            return

        table = wandb.Table(
            columns=[
                "idx",
                "query",
                "accept_response",
                "reject_response",
                "query_tokens",
                "accept_response_tokens",
                "reject_response_tokens",
                "accept_instance_length",
                "reject_instance_length"
            ]
        )

        logger.info(f"Logging {num_examples} examples:")
        rng = random.Random(seed)

        num_console_logs = min(num_examples, len(episodes))
        num_wandb_logs = min(num_examples_for_wandb, len(episodes))
        indices = rng.sample(range(len(episodes)), num_wandb_logs)

        for idx in indices:
            episode = episodes[idx]
            if not isinstance(episode, dict):
                episode = asdict(episode)

            query_token_ids = episode["query_token_ids"]
            accept_response_token_ids = episode["accept_response_token_ids"]
            reject_response_token_ids = episode["reject_response_token_ids"]

            query_tokens = [
                (
                    self.tokenizer.convert_ids_to_tokens(tok_id)
                    if tok_id >= 0
                    else str(tok_id)
                )
                for tok_id in query_token_ids
            ]
            query = self.tokenizer.decode(query_token_ids)

            accept_response_tokens = [
                (
                    self.tokenizer.convert_ids_to_tokens(tok_id)
                    if tok_id >= 0
                    else str(tok_id)
                )
                for tok_id in accept_response_token_ids
            ]
            accept_response = self.tokenizer.decode(accept_response_token_ids)

            reject_response_tokens = [
                (
                    self.tokenizer.convert_ids_to_tokens(tok_id)
                    if tok_id >= 0
                    else str(tok_id)
                )
                for tok_id in reject_response_token_ids
            ]
            reject_response = self.tokenizer.decode(reject_response_token_ids)

            accept_instance_length = (
                len(query_token_ids)
                + len(accept_response_token_ids)
            )
            reject_instance_length = (
                len(query_token_ids)
                + len(reject_response_token_ids)
            )

            table.add_data(
                idx,
                query,
                accept_response,
                reject_response,
                ", ".join(query_tokens),
                ", ".join(accept_response_tokens),
                ", ".join(reject_response_tokens),
                accept_instance_length,
                reject_instance_length
            )

            if len(table.data) >= num_console_logs:
                continue

            logger.info(f"Example {idx}")
            for k, v in episode.items():
                logger.info(f"{k}: `{v}`")
            logger.info(f"Query: `{query}`")
            logger.info(f"Accept Response: `{accept_response}`")
            logger.info(f"Reject Response: `{reject_response}`")
            logger.info(f"Accept Instance Length: {accept_instance_length}")
            logger.info(f"Reject Instance Length: {reject_instance_length}")

            logger.info("-" * 100)

        if log_to_cloud and self.cloud_logger is not None:
            self.cloud_logger.log({f"episodes/iteration_{iteration_idx:04}": table})

