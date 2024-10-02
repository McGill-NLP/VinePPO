# REST(EM) Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models https://arxiv.org/abs/2312.06585
import json
from typing import Any, Dict, Union, List

import numpy as np
from datasets import Dataset

from treetune.episode_generators import EpisodeGenerator, MathEpisodeGenerator
from treetune.episode_generators.base_episode_generator import Episode
from treetune.logging_utils import get_logger

from tqdm import tqdm

logger = get_logger(__name__)


@EpisodeGenerator.register("math_restem_episode_generator")
class MATHRestEMEpisodeGenerator(MathEpisodeGenerator):
    def __init__(
        self,
        max_response_per_question: int = 10,  # -1: no limit, 10 is taken from the original paper
        reward_threshold: float = 1.0,  # what reward is considered correct
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._logger = logger
        self.max_response_per_question = max_response_per_question
        self.reward_threshold = reward_threshold
        #   todo(milad): hmm, currently we are focusing on a SFT model, but we should also support base pretrained models that accept fewshot

    def _generate_episodes(  # we did this weird separation of _generate_episodes and _generate_episodes_functional, so the second one can be used in analyzers when generating episodes
        # from inference results. For example, in KLWithReferenceAnalyzer, we need to generate episodes from inference results, but the filtering based on reward and max_response_per_question should not be done there.
        self,
        inference_results: Dataset,
        iteration: int,
    ) -> List[Union[Dict[str, Any], Episode]]:

        episodes, metrics = self._generate_episodes_customizable(
            inference_results,
            filter_by_reward_threshold=True,
            reward_threshold=self.reward_threshold,
            max_response_per_question=self.max_response_per_question,
        )

        if len(metrics) > 0:
            logs = {f"episodes_metric/{k}": v for k, v in metrics.items()}
            self._cloud_log({**logs, "train/global_iteration": iteration})

        return episodes

    def _generate_episodes_customizable(
        self,
        inference_results: Dataset,
        filter_by_reward_threshold: bool,
        reward_threshold: float,
        max_response_per_question: int,  # -1: no limit, 10 is taken from the original paper
    ) -> List[Union[Dict[str, Any], Episode]]:
        # copy-pasted and modified from math_episode_generator.py,
        # needed to enforce just one response per query (was not possible to just call the father as this info was lost in its output),
        # and also just keep the correct responses
        episodes_dict = {}  # from idx to list of episodes
        metrics = {}
        encountered_question_indices = []
        question_reached_max_response = []

        for instance in tqdm(inference_results, desc="Generating episodes"):

            tree = json.loads(instance["_treetune__reasoning_tree"])

            idx = instance["_treetune__idx"]
            assert idx not in encountered_question_indices, f"Question {idx} is encountered more than once in inference_result."
            encountered_question_indices.append(idx)

            all_rewards = []
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

                episode = Episode(
                    query_token_ids=query_token_ids,
                    response_token_ids=response_token_ids,
                    scores=float(reward),
                )
                if not filter_by_reward_threshold or reward >= reward_threshold:
                    if idx not in episodes_dict or max_response_per_question == -1 or len(episodes_dict[idx]) < max_response_per_question:
                        episodes_dict.setdefault(idx, []).append(episode)
                    else:
                        logger.warning(f"Question {idx} has more than {max_response_per_question} responses. It is fine, just know it.")
                        question_reached_max_response.append(idx)

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

            if len(all_rewards) > 0:
                once_hit = any([r == 1.0 for r in all_rewards])
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

        return episodes, metrics
