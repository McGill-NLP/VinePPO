import random
from typing import List, Dict, Any, Tuple, Union

from datasets import Dataset
from tqdm import tqdm

from treetune.common.py_utils import format_string
from treetune.episode_generators.base_episode_generator import (
    EpisodeGenerator,
    Episode,
)
from treetune.logging_utils import get_logger
from treetune.tasks.base_task import Task

logger = get_logger(__name__)


@EpisodeGenerator.register("sft")
class SFTEpisodeGenerator(EpisodeGenerator):
    """
    A static episode generator that just converts task examples into episodes using
    a query and response template.
    """

    can_precompute_episodes = True

    def __init__(
        self,
        query_template: str,
        response_template: str,
        task: Task,
        append_bos_to_query: Union[str, bool] = "auto",
        append_eos_to_response: Union[str, bool] = "auto",
        task_split: str = "train",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.query_template = query_template
        self.response_template = response_template
        self.append_bos_to_query = append_bos_to_query
        self.append_eos_to_response = append_eos_to_response
        self.task = task
        self.episode_cache = None

        if not self.is_main_process():
            return

        self._ds = self.task.get_datasets(split=task_split)

        # Get the dataset fields that are used in the templates
        query_format_keys = []
        for column in self._ds.column_names:
            if f"{{{column}}}" in self.query_template:
                query_format_keys.append(column)

        response_format_keys = []
        for column in self._ds.column_names:
            if f"{{{column}}}" in self.response_template:
                response_format_keys.append(column)

        self.query_format_keys = query_format_keys
        self.response_format_keys = response_format_keys

        logger.info(f"Number of examples in dataset: {len(self._ds)}")
        logger.info(f"query_format_keys: {self.query_format_keys}.")
        logger.info(f"response_format_keys: {self.response_format_keys}")

        if append_bos_to_query == "auto":
            logger.info(
                f"append_bos_to_query is set to 'auto', which is {self._should_append_bos_to_query()}"
            )
        if append_eos_to_response == "auto":
            logger.info(
                f"append_eos_to_response is set to 'auto', which is {self._should_append_eos_to_response()}"
            )

        self.has_warned_about_decoding_mismatch = False

    def generate(self) -> List[Episode]:
        assert (
            self.is_main_process()
        ), "This method should only be called on the main process"

        if self.episode_cache is None:
            logger.warning(
                "`generate` is called before populating the cache. "
                "This isn't probably the intended use case of this class. "
                "But we will precompute the episodes now."
            )
            self.precompute_episodes()

        episodes = [] + self.episode_cache
        logger.info(f"Number of episodes in cache before generation: {len(episodes)}")

        random.shuffle(episodes)
        # update cache with left over episodes
        if len(episodes) > self.num_episodes_per_iteration:
            extra_episodes = episodes[self.num_episodes_per_iteration :]
            episodes = episodes[: self.num_episodes_per_iteration]
            self.episode_cache = extra_episodes
            logger.info(
                f"Number of episodes in cache after generation: {len(self.episode_cache)}"
            )

        # Won't need the dataset anymore
        del self._ds
        del self.episode_cache

        episodes = Dataset.from_dict(
            {k: [getattr(e, k) for e in episodes] for k in episodes[0].__dict__.keys()}
        )

        import gc

        gc.collect()

        return episodes

    def precompute_episodes(self):
        assert (
            self.is_main_process()
        ), "This method should only be called on the main process"

        self.episode_cache = []
        for example in tqdm(
            self._ds, desc="Precomputing episodes", total=len(self._ds)
        ):
            episode = self._convert_example_to_episode(example)
            self.episode_cache.append(episode)

    def _convert_example_to_episode(self, example: Dict[str, Any]) -> Episode:
        query_format_kwargs = {
            key: example[key] for key in self.query_format_keys if key in example
        }
        response_format_kwargs = {
            key: example[key] for key in self.response_format_keys if key in example
        }

        query = format_string(self.query_template, **query_format_kwargs)
        response = format_string(self.response_template, **response_format_kwargs)

        query_token_ids, response_token_ids = self._tokenize_query_and_response(
            query, response
        )

        advantages = self._compute_advantages(
            example, query_token_ids, response_token_ids
        )

        return Episode(
            query_token_ids=query_token_ids,
            response_token_ids=response_token_ids,
            reward=1.0,
            advantages=advantages,
        )

    def _compute_advantages(
        self,
        example: Dict[str, Any],
        query_token_ids: List[int],
        response_token_ids: List[int],
    ) -> List[float]:
        """
        Compute the advantages for each token in the response.

        Args:
            example: The example from the dataset
            query_token_ids: The token IDs of the query
            response_token_ids: The token IDs of the response

        Returns:
            A list of advantages for each token in the response
        """
        # For now, we just return a list of 1s
        return [1.0] * len(response_token_ids)

    def _tokenize_query_and_response(
        self, query: str, response: str
    ) -> Tuple[List[int], List[int]]:
        instance_text = f"{query}{response}"
        instance_encoding = self.tokenizer(
            instance_text,
            add_special_tokens=False,  # We already added BOS and EOS tokens at the end
            return_offsets_mapping=True,
        )

        token_ids = instance_encoding["input_ids"]
        offsets = instance_encoding["offset_mapping"]

        # Find the index where the response starts.
        response_start_index = next(
            i for i, (start, end) in enumerate(offsets) if start >= len(query)
        )

        # Split the token IDs into query and response parts
        query_token_ids = token_ids[:response_start_index]
        response_token_ids = token_ids[response_start_index:]

        # Check that the decoded text matches the original text
        if not self.has_warned_about_decoding_mismatch:
            decoded_instance = self.tokenizer.decode(
                query_token_ids + response_token_ids,
                clean_up_tokenization_spaces=False,
                skip_special_tokens=False,
            )
            if decoded_instance != instance_text:
                logger.warning(
                    f"Decoded instance does not match original instance.\n"
                    f"Original instance: {instance_text}\n"
                    f"Decoded instance: {decoded_instance}"
                )

            decoded_query = self.tokenizer.decode(
                query_token_ids,
                clean_up_tokenization_spaces=False,
                skip_special_tokens=False,
            )
            if decoded_query != query:
                logger.warning(
                    f"Decoded query does not match original query.\n"
                    f"Original query: {query}\n"
                    f"Decoded query: {decoded_query}"
                )

            decoded_response = self.tokenizer.decode(
                response_token_ids,
                clean_up_tokenization_spaces=False,
                skip_special_tokens=False,
            )
            if decoded_response != response:
                logger.warning(
                    f"Decoded response does not match original response.\n"
                    f"Original response: {response}\n"
                    f"Decoded response: {decoded_response}"
                )

            self.has_warned_about_decoding_mismatch = True

        # We manually add BOS and EOS tokens to the query and response
        # just to be very explicit about them. `add_special_tokens=True` may not
        # always add BOS and EOS tokens.
        if self._should_append_bos_to_query():
            query_token_ids = [self.tokenizer.bos_token_id] + query_token_ids
        if self._should_append_eos_to_response():
            response_token_ids = response_token_ids + [self.tokenizer.eos_token_id]

        return query_token_ids, response_token_ids

    def _should_append_bos_to_query(self) -> bool:
        """
        Determine whether to append BOS to the query based on the tokenizer
        """
        if self.append_bos_to_query != "auto":
            return self.append_bos_to_query

        if "llama" in self.tokenizer.name_or_path.lower():
            assert self.tokenizer.bos_token_id is not None
            return True
        else:
            raise ValueError(
                f"Cannot automatically determine whether to append BOS for tokenizer {self.tokenizer.name_or_path}"
            )

    def _should_append_eos_to_response(self) -> bool:
        """
        Determine whether to append EOS to the response based on the tokenizer
        """
        if self.append_eos_to_response != "auto":
            return self.append_eos_to_response

        if "llama" in self.tokenizer.name_or_path.lower():
            assert self.tokenizer.eos_token_id is not None
            return True
        else:
            raise ValueError(
                f"Cannot automatically determine whether to append EOS for tokenizer {self.tokenizer.name_or_path}"
            )
