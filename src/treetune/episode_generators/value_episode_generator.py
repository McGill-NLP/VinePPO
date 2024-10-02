import json
import random
from random import shuffle
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import torch
import wandb
from accelerate import PartialState
from datasets import Dataset
import numpy as np
from tqdm import tqdm
from wandb.apis.public import Run
import hashlib


from treetune.common import Lazy
from treetune.episode_generators.base_episode_generator import EpisodeGenerator
from treetune.inference_strategies import InferenceStrategy
from treetune.logging_utils import get_logger
from treetune.tasks import Task, GSM8K, MATH
from treetune.tokenization_utils import Tokenizer

import sys  # as our inference data is large we need to increase the recursion limit for the json reading
sys.setrecursionlimit(4000)

logger = get_logger(__name__)

def hash_text(text: str) -> int:
    hash_object = hashlib.sha256(text.encode())
    hash_hex = hash_object.hexdigest()
    seed_value = int(hash_hex, 16) % (2 ** 32)  # Reduce the seed value to a 32-bit integer for compatibility
    return seed_value

@dataclass
class ValueNetworkEpisode:
    query_token_ids: List[int]
    response_token_ids: List[int]
    value_network_targets: List[float]
    inference_data_instance: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        assert len(self.query_token_ids) > 0
        assert len(self.response_token_ids) > 0
        assert len(self.value_network_targets) > 0
        assert len(self.value_network_targets) == len(self.response_token_ids)


@EpisodeGenerator.register("value_network")
class ValueNetworkEpisodeGenerator(EpisodeGenerator):
    def __init__(
        self,
        tokenizer: Tokenizer,
        distributed_state: PartialState,
        inference_strategy: Lazy[InferenceStrategy],
        task: Task,
        num_episodes_per_iteration: int,
        num_responses_per_query_to_use: int,
        balancing_strategy: str = 'none',
        cloud_logger: Optional[Run] = None,
        debug: bool = False,  # TODO: milad, log when this is true
        attach_source_to_episodes: bool = False,  # for debugging purposes
    ):
        """
        :param balancing_strategy: our dataset natually has much more false samples than the correct samples,
        value networks trained on many false samples become pessimitic, there is a hope that balancing the dataset
        mitigates this.
        """
        self.inference_strategy = inference_strategy.construct(result_dir=None)
        self.task = task
        self.num_responses_per_query_to_use = num_responses_per_query_to_use
        self.balancing_strategy = balancing_strategy

        self.episode_cache = []

        super().__init__(
            tokenizer,
            distributed_state,
            num_episodes_per_iteration=num_episodes_per_iteration,
            cloud_logger=cloud_logger,
        )

        self.debug = debug
        if debug:
            self.num_episodes_per_iteration = 10
        self.can_precompute_episodes = not debug
        self.attach_source_to_episodes = attach_source_to_episodes

    def convert_to_episode(
        self,
        task: Task,
        tokenizer: Tokenizer,
        inference_result_instance: Dict[str, Any], # essentially a single question and many answers all sampled independently (fork structure, not general tree)
        ) -> Dict[str, Any]:
        answer_provided_counter = 0
        answer_not_provided_counter = 0
        is_correct_counter = 0
        is_not_correct_counter = 0
        episodes = []
        # I know, I know, I could have used functions from TreeEpisodeUtils, but I wanted to keep the code simple
        if isinstance(task, GSM8K):
            gold_answer = task.extract_gold_answer_from_text(inference_result_instance['answer'])
        elif isinstance(task, MATH):
            gold_answer = inference_result_instance['answer']
        else:
            raise ValueError(f"Implement how to extract gold answer for task {task}")

        # print(f'gold answer: {gold_answer}') # TODO: maybe log this in debug mode
        fork = json.loads(inference_result_instance['_treetune__reasoning_tree'])  # it is counterintuitive. The reasoning tree is not so much of a tree, just the root has more than one child => (fork)
        query_text = fork['text']

        if isinstance(task, MATH):
            query_text = query_text[:-1]

        # cut children to num_responses_per_query_to_use randomly, use a fixed seed for reproducibility based on query text
        assert len(fork['children']) >= self.num_responses_per_query_to_use, "num_responses_per_query_to_use is greater than the number of responses"

        # Create a local instance of a random number generator
        local_random = random.Random(hash_text(query_text))

        if self.balancing_strategy == 'none':
            selected_children = local_random.sample(fork['children'], self.num_responses_per_query_to_use)
        elif self.balancing_strategy == 'best_effort_equal':
            indices = list(range(len(fork['children'])))
            shuffled_indices = local_random.sample(indices, len(indices))

            def does_child_lead_to_correct_answer(child):
                cur_node = child
                while 'children' in cur_node:
                    assert len(cur_node['children']) == 1, "fork assumption violated"
                    cur_node = cur_node['children'][0]
                return 'answer' in cur_node and task.extract_predicted_answer_from_text(cur_node['answer']) == gold_answer

            ideal_num_of_correct_samples = (self.num_responses_per_query_to_use + 1) // 2 # we always round up as we want to have at least one correct sample preferably
            selected_children = []
            for idx in shuffled_indices:
                if len(selected_children) == ideal_num_of_correct_samples:
                    break
                if does_child_lead_to_correct_answer(fork['children'][idx]):
                    selected_children.append(fork['children'][idx])

            # sample the rest from the incorrect ones
            for idx in shuffled_indices:
                if len(selected_children) == self.num_responses_per_query_to_use:
                    break
                if not does_child_lead_to_correct_answer(fork['children'][idx]):
                    selected_children.append(fork['children'][idx])
        else:
            raise ValueError(f"Unknown balancing strategy: {self.balancing_strategy}")

        # loop over the forks
        for node_chain in selected_children:  # a single node_chain contains a single instance of LLM trying to answer the question
            current_node = node_chain
            step_texts = [current_node['text']]
            while 'children' in current_node:
                assert len(current_node['children']) == 1, "fork assumption violated"
                current_node = current_node['children'][0]
                step_texts.append(current_node['text'])

            response_text = '\n' + '\n'.join(step_texts)

            full_text = current_node['full_text']  # take the full text from the last node
            assert query_text + response_text == full_text, "full text = query text + response text assumption violated"

            out_full = tokenizer(full_text,
                                 return_tensors="pt",
                                 return_offsets_mapping=True)
            out_query = tokenizer(query_text,
                                  return_tensors="pt",
                                  return_offsets_mapping=True)
            out_response = tokenizer(response_text,
                                     return_tensors="pt",
                                     return_offsets_mapping=True,
                                     add_special_tokens=False)

            # --- check if the tokenization is correct ---
            assert tokenizer.decode(out_full['input_ids'][0], skip_special_tokens=True) == full_text  # mostly for educational purposes
            # ----- cleaning up the response because of weird behaviour of tokenizer  ----- #
            assert (
                out_response.tokens()[0] == "▁"
            )  # just repeating the assertion to explain the code below
            clean_response_input_ids = out_response.input_ids.squeeze()[
                                       1:
                                       ]  # remove the first token which is '▁'

            # ----- assertions ----- #
            input_ids_glued = np.concatenate(
                [out_query.input_ids.squeeze(), clean_response_input_ids]
            )
            assert np.allclose(input_ids_glued, out_full.input_ids.reshape(-1))

            # --- compute the reward for this tine ---
            if 'answer' not in current_node:
                is_response_chopped = True
                answer_not_provided_counter += 1
                reward = 0

            else:
                is_response_chopped = False
                answer_provided_counter += 1
                predicted_answer = task.extract_predicted_answer_from_text(current_node['answer'])
                if predicted_answer == gold_answer:
                    reward = 1
                    is_correct_counter += 1
                else:
                    reward = 0
                    is_not_correct_counter += 1

            # --- create the episode ---
            response_token_ids = clean_response_input_ids.tolist()
            if not is_response_chopped:
                response_token_ids.append(tokenizer.eos_token_id)  # TODO: milad, why does llama2 tokenizer did not add eos to full_text inputs ids in our assertion and so the assertion should have failed?
            response_targets = (torch.ones(len(response_token_ids)) * reward).tolist()  # targets for the value network prediction
            query_token_ids = out_query.input_ids.squeeze().tolist()

            episode = ValueNetworkEpisode(
                query_token_ids=query_token_ids,
                response_token_ids=response_token_ids,
                value_network_targets=response_targets,
                inference_data_instance=inference_result_instance if self.attach_source_to_episodes else None,
            )

            episodes.append(episode)

        return {
            'episodes': episodes,
            'answer_provided_counter': answer_provided_counter,
            'answer_not_provided_counter': answer_not_provided_counter,
            'is_correct_counter': is_correct_counter,
            'is_not_correct_counter': is_not_correct_counter,
        }

    def precompute_episodes(self):
        # merely a copy paster from other episode generators, but just changed the passed arguments to convert_to_episode
        assert (
            len(self.episode_cache) == 0
        ), "`precompute_episodes` can only be called once"

        results: Dataset = self.inference_strategy.generate(None)
        results.remove_columns('_treetune__candidate_answers')
        results_lst = results.to_list()

        if self.debug:
            results_lst = random.sample(results_lst, 1000)

        logger.info("**** Precomputing training episodes from inference results ****")
        logger.info(f"\tNumber of inference results: {len(results_lst)}")

        from multiprocessing import Pool
        from functools import partial

        with Pool(8) as p:
            results_lst = list(
                tqdm(
                    p.imap(
                        partial(
                            self.convert_to_episode,
                            self.task,
                            self.tokenizer,
                        ),
                        results_lst,
                        chunksize=20,
                    ),
                    total=len(results_lst),
                    desc="Converting inference results to episodes",
                )
            )

        answer_provided_counter = 0
        answer_not_provided_counter = 0
        is_correct_counter = 0
        is_not_correct_counter = 0
        for results in results_lst:
            answer_provided_counter += results['answer_provided_counter']
            answer_not_provided_counter += results['answer_not_provided_counter']
            is_correct_counter += results['is_correct_counter']
            is_not_correct_counter += results['is_not_correct_counter']
            self.episode_cache.extend(results['episodes'])

        logger.info(f"answer provided: {answer_provided_counter}, answer not provided: {answer_not_provided_counter}")
        logger.info(f"is correct: {is_correct_counter}, is not correct: {is_not_correct_counter}")
        logger.info(f"is correct ratio: {is_correct_counter / (is_correct_counter + is_not_correct_counter)}")
        if self.cloud_logger is not None:
            self.cloud_logger.log({
                "precomputed_episodes/answer_provided": answer_provided_counter,
                "precomputed_episodes/answer_not_provided": answer_not_provided_counter,
                "precomputed_episodes/is_correct": is_correct_counter,
                "precomputed_episodes/is_not_correct": is_not_correct_counter,
                "precomputed_episodes/is_correct_ratio": is_correct_counter / (is_correct_counter + is_not_correct_counter),
            })



    def generate(self) -> List[ValueNetworkEpisode]:
        # merely a copy paster from other episode generators, but just changed the passed arguments to convert_to_episode
        assert (
            self.is_main_process()
        ), "This method should only be called on the main process"
        episodes = (
            [] + self.episode_cache
        )  # it is important to copy the list here so that we don't modify the cache
        logger.info(f"Number of episodes in cache before generation: {len(episodes)}")
        # generate until we have enough episodes
        answer_provided_counter = 0
        answer_not_provided_counter = 0
        is_correct_counter = 0
        is_not_correct_counter = 0

        while len(episodes) < self.num_episodes_per_iteration:
            results = self.inference_strategy.generate(None)
            if self.debug:
                results = results.shuffle(seed=42).select(range(10))
                logger.warning("Debug mode: only using 100 inference results")

            for result in results:
                results_ctp = self.convert_to_episode(
                    self.task,
                    self.tokenizer,
                    result,
                )
                answer_provided_counter += results_ctp['answer_provided_counter']
                answer_not_provided_counter += results_ctp['answer_not_provided_counter']
                is_correct_counter += results_ctp['is_correct_counter']
                is_not_correct_counter += results_ctp['is_not_correct_counter']

                episodes.extend(results_ctp['episodes'])

        logger.info(f"answer provided: {answer_provided_counter}, answer not provided: {answer_not_provided_counter}")
        logger.info(f"is correct: {is_correct_counter}, is not correct: {is_not_correct_counter}")
        logger.info(f"is correct ratio: {is_correct_counter / (is_correct_counter + is_not_correct_counter) if is_correct_counter + is_not_correct_counter > 0 else 0.0}")
        if self.cloud_logger is not None:
            self.cloud_logger.log({
                "trained_on_episodes/answer_provided": answer_provided_counter,
                "trained_on_episodes/answer_not_provided": answer_not_provided_counter,
                "trained_on_episodes/is_correct": is_correct_counter,
                "trained_on_episodes/is_not_correct": is_not_correct_counter,
                "trained_on_episodes/is_correct_ratio": is_correct_counter / (is_correct_counter + is_not_correct_counter) if is_correct_counter + is_not_correct_counter > 0 else 0.0  ,
            })

        shuffle(episodes)

        # update cache with left over episodes
        if len(episodes) > self.num_episodes_per_iteration:
            extra_episodes = episodes[self.num_episodes_per_iteration:]
            episodes = episodes[: self.num_episodes_per_iteration]
            self.episode_cache = extra_episodes
            logger.info(
                f"Number of episodes in cache after generation: {len(self.episode_cache)}"
            )
        return episodes

    def log_episodes(
        self,
        episodes: List[ValueNetworkEpisode],
        iteration_idx: int,
        num_examples: int = 100,
        seed: int = 42,
    ):
        if not self.is_main_process():
            return

        table = wandb.Table(
            columns=[
                "idx",
                "query",
                "response",
                "query_tokens",
                "response_tokens",
                "instance_length",
            ]
        )

        logger.info(f"Logging {num_examples} examples:")
        rng = random.Random(seed)
        indices = rng.sample(range(len(episodes)), num_examples)
        for idx in indices:
            episode = episodes[idx]

            query_tokens = [
                self.tokenizer.convert_ids_to_tokens(tok_id)
                if tok_id >= 0
                else str(tok_id)
                for tok_id in episode.query_token_ids
            ]
            query = self.tokenizer.decode(episode.query_token_ids)

            response_tokens = [
                self.tokenizer.convert_ids_to_tokens(tok_id)
                if tok_id >= 0
                else str(tok_id)
                for tok_id in episode.response_token_ids
            ]
            response = self.tokenizer.decode(episode.response_token_ids)

            instance_length = len(episode.query_token_ids) + len(
                episode.response_token_ids
            )

            logger.info(f"Example {idx}")
            for k in episode.__dict__.keys():
                logger.info(f"{k}: {getattr(episode, k)}")
            logger.info(f"Query: `{query}`")
            logger.info(f"Response: `{response}`")
            logger.info(f"Instance Length: {instance_length}")

            logger.info("-" * 100)

            table.add_data(
                idx,
                query,
                response,
                ", ".join(query_tokens),
                ", ".join(response_tokens),
                instance_length,
            )

        if self.cloud_logger is not None:
            self.cloud_logger.log({f"iterations/{iteration_idx}/episodes": table})




