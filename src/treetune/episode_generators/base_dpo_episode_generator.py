import random
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import wandb
from datasets import Dataset

from treetune.common import Lazy
from treetune.episode_generators.base_episode_generator import EpisodeGenerator
from treetune.inference_strategies import InferenceStrategy
from treetune.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class DPOEpisode:
    query_token_ids: List[int]
    accept_response_token_ids: List[int]
    list_of_reject_response_token_ids: List[List[int]]

    inference_data_instance: Optional[Dict[str, Any]] = None

    query_score: Optional[float] = None
    accept_response_advantage: Optional[List[float]] = None
    list_of_reject_response_advantages: Optional[List[List[float]]] = None

    def __post_init__(self):
        assert len(self.query_token_ids) > 0
        assert len(self.accept_response_token_ids) > 0
        assert len(self.list_of_reject_response_token_ids) > 0
        for reject_response_token_ids in self.list_of_reject_response_token_ids:
            assert len(reject_response_token_ids) > 0


class DPOEpisodeGenerator(EpisodeGenerator):
    def log_episodes(
        self,
        episodes: List[DPOEpisode],
        iteration_idx: int,
        num_examples: int = 10,
        seed: int = 42,
    ):
        # Determine the maximum number of reject responses among all episodes
        max_reject_responses = max(
            len(episode.list_of_reject_response_token_ids) for episode in episodes
        )

        # Dynamic column names based on the number of reject responses
        columns = [
            "idx",
            "query",
            "accept_response",
            "avg_instance_length",
            "query_tokens",
            "accept_response_tokens",
            *[f"reject_response_{i+1}" for i in range(max_reject_responses)],
            *[f"reject_response_tokens_{i+1}" for i in range(max_reject_responses)],
        ]

        table = wandb.Table(columns=columns)

        logger.info(f"Logging {num_examples} examples:")
        rng = random.Random(seed)
        indices = rng.sample(range(len(episodes)), min(num_examples, len(episodes)))

        for idx in indices:
            episode = episodes[idx]

            query_tokens = [
                self.tokenizer.convert_ids_to_tokens(tok_id)
                if tok_id >= 0
                else str(tok_id)
                for tok_id in episode.query_token_ids
            ]
            query = self.tokenizer.decode(episode.query_token_ids)

            accept_response_tokens = [
                self.tokenizer.convert_ids_to_tokens(tok_id)
                if tok_id >= 0
                else str(tok_id)
                for tok_id in episode.accept_response_token_ids
            ]
            accept_response = self.tokenizer.decode(episode.accept_response_token_ids)

            logger.info(f"*** Example {idx}:")
            logger.info(f"*** Query:\n`{query}`")
            logger.info(f"*** Accept Response:\n`{accept_response}`")

            reject_responses_data = []
            for i, reject_response_token_ids in enumerate(
                episode.list_of_reject_response_token_ids
            ):
                reject_response = self.tokenizer.decode(reject_response_token_ids)
                reject_response_tokens = [
                    self.tokenizer.convert_ids_to_tokens(tok_id)
                    if tok_id >= 0
                    else str(tok_id)
                    for tok_id in reject_response_token_ids
                ]
                logger.info(f"*** Reject Response {i+1}:\n`{reject_response}`")
                reject_responses_data.extend(
                    [reject_response, ", ".join(reject_response_tokens)]
                )

            avg_instance_length = sum(
                (len(episode.query_token_ids) + len(tokens_lst))
                for tokens_lst in [
                    episode.accept_response_token_ids,
                    *episode.list_of_reject_response_token_ids,
                ]
            ) / (1 + len(episode.list_of_reject_response_token_ids))

            row_data = [
                idx,
                query,
                accept_response,
                avg_instance_length,
                ", ".join(query_tokens),
                ", ".join(accept_response_tokens),
                *reject_responses_data,
                *([""] * (2 * max_reject_responses - len(reject_responses_data))),
            ]

            logger.info("-" * 100)
            table.add_data(*row_data)

        if self.cloud_logger is not None and self.is_main_process():
            self.cloud_logger.log({f"iterations/{iteration_idx}/episodes": table})


@EpisodeGenerator.register("dpo_empty")
class DPOEmptyEpisodeGenerator(DPOEpisodeGenerator):
    def __init__(self, inference_strategy: Lazy[InferenceStrategy], **kwargs):
        super().__init__(**kwargs)
        self.inference_strategy = inference_strategy.construct(result_dir=None)
        assert self.tokenizer is not None

    def generate(self) -> List[DPOEpisode]:
        out = self.inference_strategy.generate(None)
        assert isinstance(out, Dataset)

        dpo_dataset_dict = {
            "prompt": ["hello", "how are you"],
            "chosen": ["hi nice to meet you", "I am fine"],
            "rejected": ["leave me alone", "I am not fine"],
            "chosen_token_ids": [
                [12758, 1219, 72, 3621, 284, 1826, 345, 50256],
                [40, 716, 3734, 50256],
            ],
            "reject_token_ids": [
                [12758, 2305, 1015, 502, 3436, 50256],
                [40, 716, 407, 3734, 50256],
            ],
            "prompt_token_ids": [[50256], [50256, 4919, 389, 345]],
        }

        # For now randomly create dpo episodes

        episodes = []

        dpo_dataset = Dataset.from_dict(dpo_dataset_dict)
        dpo_dataset = dpo_dataset.select(range(2))

        for i in range(len(dpo_dataset)):
            query_token_ids = dpo_dataset["prompt_token_ids"][i]
            accept_response_token_ids = dpo_dataset["chosen_token_ids"][i]
            list_of_reject_response_token_ids = [dpo_dataset["reject_token_ids"][i]]
            episodes.append(
                DPOEpisode(
                    query_token_ids,
                    accept_response_token_ids,
                    list_of_reject_response_token_ids,
                )
            )

        return episodes
