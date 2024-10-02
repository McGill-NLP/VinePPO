import copy
import json
import random
from random import shuffle
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
from accelerate import PartialState
from datasets import Dataset
from tqdm import tqdm
from wandb.apis.public import Run

from treetune import logging_utils
from treetune.common import Lazy
from treetune.episode_generators import EpisodeGenerator
from treetune.episode_generators.base_episode_generator import (
    Episode,
    EpisodeGeneratorStrategy,
)
from treetune.episode_generators.path_aggregators import PathAggregator
from treetune.episode_generators.path_filters import PathFilter, SuccessfulPathFilter
from treetune.episode_generators.path_post_processors import PathPostProcessor
from treetune.inference_strategies import InferenceStrategy
from treetune.tasks import Task
from treetune.tokenization_utils import Tokenizer

logger = logging_utils.get_logger(__name__)


@EpisodeGeneratorStrategy.register("tree")
class TreeEpisodeGeneratorStrategy(EpisodeGeneratorStrategy):
    def __init__(
        self,
        path_filters: List[PathFilter],
        path_aggregators: List[PathAggregator],
        path_post_processors: List[PathPostProcessor],
    ):
        self.path_filters = path_filters
        self.path_aggregators = path_aggregators
        self.path_post_processors = path_post_processors
        super().__init__()

    def __call__(self, paths):
        # logger.info(f"Number of paths before filtering: {len(paths)}")
        for path_filter in self.path_filters:
            paths = [path for path in paths if path_filter(path)]
        # logger.info(f"Number of paths after filtering: {len(paths)}")
        for path_post_processor in self.path_post_processors:
            paths = [path_post_processor(path) for path in paths]
        # logger.info(f"Number of paths after post processing: {len(paths)}")
        for path_aggregator in self.path_aggregators:
            paths = path_aggregator(paths)

        # logger.info(f"Number of paths after aggregation: {len(paths)}")

        def is_path_just_query(path):
            return len(path["node_chain"]) == 1

        paths = [path for path in paths if not is_path_just_query(path)]
        # logger.info(
        #    f"Number of paths after removing paths that are just query: {len(paths)}"
        # )
        return paths


class TreeEpisodeUtils:
    def extract_pred_answer(self, task, node):
        text = node["answer"]
        answer = task.extract_predicted_answer_from_text(text)
        return answer

    def is_answer_correct(self, task, node, gold_answer):
        pred_answer = self.extract_pred_answer(task, node)
        if pred_answer is None:
            return 0
        else:
            return int(pred_answer == gold_answer)

    def compute_score(self, task, root, gold_answer):
        def dfs(node):
            if "answer" in node:
                node["score"] = self.is_answer_correct(task, node, gold_answer)
                node["is_correct_answer"] = node["score"]
                return node["score"]

            child_scores = []
            for child in node.get("children", []):
                child_scores.append(dfs(child))

            if len(child_scores) > 0:
                node["score"] = sum(child_scores) / len(child_scores)
            else:
                node["score"] = 0

            return node["score"]

        dfs(root)
        return root

    def compute_advantage(self, root):
        def dfs(node):
            if "answer" in node:
                return

            for child in node.get("children", []):
                child["advantage"] = child["score"] - node["score"]
                dfs(child)

        dfs(root)
        return root

    def extract_paths_from_tree(
        self,
        tree,
        include_importance_weights: bool = False,
        repeat_early_stopped_paths: bool = False,
        branch_factor: Optional[int] = None,
        max_depth: Optional[int] = None,
    ):
        paths = []

        def extract_row(node_chain):
            return {"node_chain": node_chain}

        def is_leaf(node):
            return "children" not in node

        def dfs(node, depth, parent_node_chain):
            node_chain = parent_node_chain + [node]
            if branch_factor is not None and max_depth is not None:
                importance_weight = branch_factor ** (max_depth - depth)
            else:
                importance_weight = 1

            if include_importance_weights:
                node["importance_weight"] = (
                    importance_weight if max_depth > depth else 1
                )

            if is_leaf(node):
                row = extract_row(node_chain)
                if repeat_early_stopped_paths and max_depth > depth:
                    paths.extend([row] * importance_weight)
                else:
                    paths.append(row)

            for child in node.get("children", []):
                dfs(child, depth + 1, node_chain)
            # Since serialization that we do later on is slow,
            # we remove children from the tree as we don't need them anymore
            node.pop("children", None)

        # because we want to modify this tree for faster serialization
        tree = copy.deepcopy(tree)
        dfs(tree, 0, [])
        paths = json.loads(json.dumps(paths))
        return paths


@EpisodeGenerator.register("tree")
class TreeEpisodeGenerator(EpisodeGenerator, TreeEpisodeUtils):
    def __init__(
        self,
        tokenizer: Tokenizer,
        distributed_state: PartialState,
        inference_strategy: Lazy[InferenceStrategy],
        task: Task,
        episode_strategy: EpisodeGeneratorStrategy,
        num_episodes_per_iteration: int,
        # During tree generation, we early stop expanding a node if the model reaches the final answer.
        # Thus, in order to make sure such early stopped nodes have equal weight as a node that its children
        # are all expanded, we repeat them branch_factor ^ (max_depth - curr_depth) times.
        repeat_early_stopped_paths: bool = False,
        branch_factor: Optional[int] = None,
        max_depth: Optional[int] = None,
        include_importance_weights: bool = False,
        cloud_logger: Optional[Run] = None,
        debug: bool = False,
    ):
        self.inference_strategy = inference_strategy.construct(result_dir=None)
        self.task = task
        self.episode_strategy = episode_strategy

        self.repeat_early_stopped_paths = repeat_early_stopped_paths
        self.branch_factor = branch_factor
        self.max_depth = max_depth
        self.include_importance_weights = include_importance_weights

        if self.include_importance_weights:
            assert (
                max_depth is not None and branch_factor is not None
            ), "When `include_importance_weights` is True, `max_depth` and `branch_factor` must be provided"

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

    def precompute_episodes(self):
        assert (
            len(self.episode_cache) == 0
        ), "`precompute_episodes` can only be called once"

        results: Dataset = self.inference_strategy.generate(None)
        results_lst = results.to_list()

        if self.debug:
            results_lst = random.sample(results_lst, 100)

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
                            self.episode_strategy,
                            self.task,
                            self.tokenizer,
                            repeat_early_stopped_paths=self.repeat_early_stopped_paths,
                            branch_factor=self.branch_factor,
                            max_depth=self.max_depth,
                        ),
                        results_lst,
                        chunksize=20,
                    ),
                    total=len(results_lst),
                    desc="Converting inference results to episodes",
                )
            )

        for results in results_lst:
            self.episode_cache.extend(results)

    def generate(self) -> List[Episode]:
        assert (
            self.is_main_process()
        ), "This method should only be called on the main process"
        episodes = [] + self.episode_cache
        logger.info(f"Number of episodes in cache before generation: {len(episodes)}")
        # generate until we have enough episodes
        while len(episodes) < self.num_episodes_per_iteration:
            results = self.inference_strategy.generate(None)
            if self.debug:
                results = results.shuffle(seed=42).select(range(100))
                logger.warning("Debug mode: only using 10 inference results")

            for result in results:
                episodes.extend(
                    self.convert_to_episode(
                        self.episode_strategy,
                        self.task,
                        self.tokenizer,
                        result,
                        repeat_early_stopped_paths=self.repeat_early_stopped_paths,
                        branch_factor=self.branch_factor,
                        max_depth=self.max_depth,
                    )
                )

        shuffle(episodes)
        # update cache with left over episodes
        if len(episodes) > self.num_episodes_per_iteration:
            extra_episodes = episodes[self.num_episodes_per_iteration :]
            episodes = episodes[: self.num_episodes_per_iteration]
            self.episode_cache = extra_episodes
            logger.info(
                f"Number of episodes in cache after generation: {len(self.episode_cache)}"
            )
        return episodes

    def convert_to_episode(
        self,
        episode_strategy: TreeEpisodeGeneratorStrategy,
        task: Task,
        tokenizer: Tokenizer,
        tree_inference_result_instance: Dict[str, Any],
        repeat_early_stopped_paths: bool = False,
        branch_factor: Optional[int] = None,
        max_depth: Optional[int] = None,
    ):
        # assert tokenize is the llama2, because in llama2 tokens("\nStep") = ['▁', '\n', Step]
        # therefore we need to remove the first token and in other tokenizers I don't know if this is the case
        assert tokenizer.name_or_path == "meta-llama/Llama-2-7b-chat-hf"
        instance = tree_inference_result_instance

        # load tree, compute score and advantage
        tree = json.loads(instance["_treetune__reasoning_tree"])
        gold_answer = task.extract_gold_answer_from_text(instance["answer"])
        self.compute_score(task, tree, gold_answer)
        self.compute_advantage(tree)

        paths = self.extract_paths_from_tree(
            tree,
            include_importance_weights=self.include_importance_weights,
            repeat_early_stopped_paths=repeat_early_stopped_paths,
            branch_factor=branch_factor,
            max_depth=max_depth,
        )
        paths = episode_strategy(paths)

        def convert_path_to_episode(path):
            query_text = path["node_chain"][0]["text"]
            response_text = "\n" + "\n".join(
                [node["text"] for node in path["node_chain"][1:]]
            )
            out_response = tokenizer(
                response_text,
                return_tensors="pt",
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
            full_text = path["node_chain"][-1]["full_text"]
            out_full = tokenizer(
                full_text, return_tensors="pt", return_offsets_mapping=True
            )
            out_query = tokenizer(
                query_text, return_tensors="pt", return_offsets_mapping=True
            )

            last_node = path["node_chain"][-1]
            is_response_chopped = "answer" not in last_node

            # ----- assertions ----- #
            assert (
                "\n".join([node["text"] for node in path["node_chain"]])
                == path["node_chain"][-1]["full_text"]
            )  # making sure that the full text is the concatenation of the text of all nodes
            assert (
                query_text + response_text == full_text
            )  # making sure that the full text is the concatenation of the query and response

            # ----- computing advantages ----- #
            # compute advantages of characters
            advantage_char = np.ones(len(response_text)) * -777
            last_index = 0
            response_node_chain = path["node_chain"][1:]
            for i, node in enumerate(response_node_chain):
                if i == 0:
                    if len(response_node_chain) == 1:
                        text = "\n" + node["text"]
                    else:
                        text = "\n" + node["text"] + "\n"
                elif i != len(response_node_chain) - 1:
                    text = node["text"] + "\n"
                else:
                    text = node["text"]
                advantage_char[last_index : last_index + len(text)] = node["advantage"]
                last_index += len(text)

            # compute advantages of tokens
            input_ids = out_response["input_ids"][0]
            offset_mapping = out_response["offset_mapping"][0]
            advantage_token = np.ones(len(input_ids)) * -999
            for i in range(len(advantage_token)):
                start_char_of_token = offset_mapping[i][0]
                advantage_token[i] = advantage_char[start_char_of_token]

            # ----- cleaning up the response ----- #
            assert (
                out_response.tokens()[0] == "▁"
            )  # just repeating the assertion to explain the code below
            clean_response_input_ids = out_response.input_ids.reshape(-1)[
                1:
            ]  # remove the first token which is '▁'
            clean_advantage_token = advantage_token[
                1:
            ]  # remove the first token advantage which is '▁'

            # ----- assertions ----- #
            input_ids_glued = np.concatenate(
                [out_query.input_ids.reshape(-1), clean_response_input_ids]
            )
            assert np.allclose(input_ids_glued, out_full.input_ids.reshape(-1))

            # ----- episode generation ----- #
            reward = 1.0 if SuccessfulPathFilter()(path) else 0.0
            query_token_ids = out_query.input_ids.reshape(-1).numpy().tolist()
            response_token_ids = clean_response_input_ids.numpy().tolist()
            advantages = clean_advantage_token.tolist()

            if not is_response_chopped:
                # Add EOS token to the end of the response since we skipped it earlier
                response_token_ids.append(tokenizer.eos_token_id)
                advantages.append(advantages[-1])

            episode = Episode(
                reward=reward,
                query_token_ids=query_token_ids,
                response_token_ids=response_token_ids,
                advantages=advantages,
            )

            return episode

        episodes = [convert_path_to_episode(path) for path in paths]

        return episodes


@EpisodeGenerator.register("tree_state_action")
class TreeStateActionGenerator(TreeEpisodeGenerator):
    def __init__(self, num_first_level_children: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        assert not self.repeat_early_stopped_paths, (
            "TreeStateActionGenerator does not support `repeat_early_stopped_paths`"
            " as it is not compatible with state-action representation"
        )
        self.num_first_level_children = num_first_level_children

    def extract_paths_from_tree(
        self,
        tree,
        include_importance_weights: bool = False,
        repeat_early_stopped_paths: bool = False,
        branch_factor: Optional[int] = None,
        max_depth: Optional[int] = None,
    ):
        paths = []

        def extract_row(node_chain):
            return {"node_chain": node_chain}

        def dfs(node, depth, parent_node_chain):
            node_chain = parent_node_chain + [node]

            importance_weight = branch_factor ** (max_depth - depth)
            importance_weight = max(importance_weight, 1)
            if include_importance_weights:
                node["importance_weight"] = importance_weight

            is_root = depth == 0
            if not is_root:
                # Do not include the root node as it's just the query
                row = extract_row(node_chain)
                paths.append(row)

            for child in node.get("children", []):
                dfs(child, depth + 1, node_chain)

            # Since serialization that we do later on is slow,
            # we remove children from the tree as we don't need them anymore
            node.pop("children", None)

        # because we want to modify this tree for faster serialization
        tree = copy.deepcopy(tree)
        if self.num_first_level_children is not None:
            tree["children"] = random.sample(
                tree["children"],
                min(self.num_first_level_children, len(tree["children"])),
            )

        dfs(tree, 0, [])
        paths = json.loads(json.dumps(paths))
        return paths


@EpisodeGenerator.register("tree_for_MATH")
class TreeEpisodeGeneratorForMath(TreeEpisodeGenerator):
    def __init__(
        self,
        append_bos_to_query: bool = True,
        append_eos_response: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.append_bos_to_query = append_bos_to_query
        self.append_eos_to_response = append_eos_response

    def is_answer_correct(self, task, node, gold_answer):
        pred_answer = self.extract_pred_answer(task, node)
        if pred_answer is None:
            return 0
        else:
            return task.grade_answer(given_answer=pred_answer, ground_truth=gold_answer)

    def convert_to_episode(
        self,
        episode_strategy: TreeEpisodeGeneratorStrategy,
        task: Task,
        tokenizer: Tokenizer,
        tree_inference_result_instance: Dict[str, Any],
        repeat_early_stopped_paths: bool = False,
        branch_factor: Optional[int] = None,
        max_depth: Optional[int] = None,
    ):
        instance = tree_inference_result_instance

        # load tree, compute score and advantage
        tree = json.loads(instance["_treetune__reasoning_tree"])
        gold_answer = instance["answer"]
        self.compute_score(task, tree, gold_answer)
        self.compute_advantage(tree)

        paths = self.extract_paths_from_tree(
            tree,
            include_importance_weights=self.include_importance_weights,
            repeat_early_stopped_paths=repeat_early_stopped_paths,
            branch_factor=branch_factor,
            max_depth=max_depth,
        )
        paths = episode_strategy(paths)

        def convert_path_to_episode(path):
            assert len(path["node_chain"]) == 2, "Does not support multi-hop paths."

            query_text = path["node_chain"][0]["text"]
            full_text = path["node_chain"][-1]["full_text"]
            response_text = full_text[len(query_text) :]

            is_response_chopped = "# Answer" not in response_text

            try:
                query_token_ids, response_token_ids = self._tokenize_query_and_response(
                    query_text,
                    response_text,
                    allow_append_eos=not is_response_chopped,
                )
            except Exception as e:
                raise e

            # ----- episode generation ----- #
            reward = 1.0 if SuccessfulPathFilter()(path) else 0.0

            episode = Episode(
                reward=reward,
                query_token_ids=query_token_ids,
                response_token_ids=response_token_ids,
                advantages=[1] * len(response_token_ids),
            )

            return episode

        episodes = [convert_path_to_episode(path) for path in paths]

        return episodes

    def _tokenize_query_and_response(
        self, query: str, response: str, allow_append_eos: bool = True
    ) -> Tuple[List[int], List[int]]:
        instance_text = f"{query}{response}"
        instance_encoding = self.tokenizer(
            instance_text,
            add_special_tokens=False,  # We will add BOS and EOS tokens at the end
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
        if not getattr(self, "has_warned_about_decoding_mismatch", False):
            decoded_instance = self.tokenizer.decode(
                query_token_ids + response_token_ids,
                clean_up_tokenization_spaces=False,
                skip_special_tokens=False,
            )
            has_warned = False
            if decoded_instance != instance_text:
                logger.warning(
                    f"Decoded instance does not match original instance.\n"
                    f"Original instance: {instance_text}\n"
                    f"Decoded instance: {decoded_instance}"
                )
                has_warned = True

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
                has_warned = True

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
                has_warned = True

            self.has_warned_about_decoding_mismatch = has_warned

        # We manually add BOS and EOS tokens to the query and response
        # just to be very explicit about them. `add_special_tokens=True` may not
        # always add BOS and EOS tokens.
        if self._should_append_bos_to_query():
            query_token_ids = [self.tokenizer.bos_token_id] + query_token_ids

        if allow_append_eos and self._should_append_eos_to_response():
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
