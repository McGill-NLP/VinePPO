import asyncio
import os
import random
import re
import uuid
from typing import List, Optional, Callable

import evaluate
import numpy as np

from treetune import logging_utils
from treetune.common import Registrable, JsonDict
from treetune.common import guidance_utils as gu
from treetune.common.py_utils import format_string
from treetune.inference_strategies.tree_inference import Node
from treetune.inference_strategies.tree_inference.branch_factor_strategy import (
    BranchFactorStrategy,
)
from treetune.tokenization_utils import Tokenizer

logger = logging_utils.get_logger(__name__)


def replace_gen_pattern(s, my_repl):
    start_pattern = "{{gen"
    end_pattern = "}}"

    # Find the start index
    start_index = s.find(start_pattern)
    if start_index == -1:
        return s  # Start pattern not found, return original string

    # Find the end index starting from the end of the string
    end_index = s.rfind(end_pattern)
    if end_index == -1:
        return s  # End pattern not found, return original string

    # Adjust end index to include the end pattern
    end_index += len(end_pattern)

    # Replace the pattern with `my_repl`
    return s[:start_index] + my_repl + s[end_index:]


class NodeExpander(Registrable):
    def __init__(
        self, branch_factor_strategy: BranchFactorStrategy, seed: Optional[int] = None
    ):
        self.branch_factor_strategy = branch_factor_strategy
        self.seed = seed
        self._run_program = gu.run_program

    def set_run_program(self, run_program):
        self._run_program = run_program

    def set_seed(self, seed):
        self.seed = seed

    async def expand(self, current_node: Node, prefix: str, depth: int) -> List[Node]:
        raise NotImplementedError()


@NodeExpander.register("iid")
class IIDExpander(NodeExpander):
    def __init__(
        self, program: str, node_text_template: str, program_kwargs: JsonDict, **kwargs
    ):
        super().__init__(**kwargs)

        if "logprobs" not in program_kwargs:
            program_kwargs["logprobs"] = 0
        else:
            assert program_kwargs["logprobs"] in [0, 1], "logprobs must be 0 or 1"

        if "num_samples" not in program_kwargs and "={num_samples}" in program:
            program_kwargs["num_samples"] = 1
        else:
            assert (
                program_kwargs.get("num_samples", 1) == 1
            ), "only 1 sample is supported"

        self.program_kwargs = program_kwargs
        self.program_template = format_string(program, **program_kwargs)
        assert (
            '"chain_of_thought"' in self.program_template
        ), "Program_template must contain 'chain_of_thought'"

        self.node_text_template = node_text_template
        assert (
            "{chain_of_thought}" in self.node_text_template
        ), "Node_text_template must contain '{chain_of_thought}'"

    async def _sample_node(self, prefix: str, depth: int) -> Node:
        result = await self._run_program(self.program_template, prefix=prefix)
        variables = result.variables()
        chain_of_thought = variables["chain_of_thought"]
        stop_text = variables["stop_text"]

        node_text = self.node_text_template.format(chain_of_thought=chain_of_thought)

        node = {
            "text": node_text,
            "depth": depth,
            "full_text": result.text,
            "stop_text": stop_text,
        }

        if (
            "chain_of_thought_logprobs" in variables
            and self.program_kwargs["logprobs"] > 0
        ):
            logprobs: List[float] = variables["chain_of_thought_logprobs"]
            assert isinstance(logprobs, list)
            node_num_tokens = len(logprobs)
            node_logprobs = sum(logprobs)

            node["sum_logprobs"] = node_logprobs
            node["num_tokens"] = node_num_tokens

        return node

    async def expand(self, current_node: Node, prefix: str, depth: int) -> List[Node]:
        tasks = []
        branch_factor = self.branch_factor_strategy(current_node)
        for i in range(branch_factor):
            task = asyncio.create_task(self._sample_node(prefix, depth + 1))
            tasks.append(task)

        nodes = []
        for task in tasks:
            node = await task
            nodes.append(node)

        return nodes


@NodeExpander.register("iid_with_max_branching_depth")
class IIDExpanderWithMaxBranchingDepth(NodeExpander):
    def __init__(
        self,
        program: str,
        node_text_template: str,
        program_kwargs: JsonDict,
        program_after_branching: str,
        program_kwargs_after_branching: JsonDict,
        max_branching_depth: int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if "logprobs" not in program_kwargs:
            program_kwargs["logprobs"] = 0
        else:
            assert program_kwargs["logprobs"] in [0, 1], "logprobs must be 0 or 1"

        if "num_samples" not in program_kwargs and "={num_samples}" in program:
            program_kwargs["num_samples"] = 1
        else:
            assert (
                program_kwargs.get("num_samples", 1) == 1
            ), "only 1 sample is supported"

        self.program_kwargs = program_kwargs
        self.program_template = format_string(program, **program_kwargs)
        assert (
            '"chain_of_thought"' in self.program_template
        ), "Program_template must contain 'chain_of_thought'"

        # --------------------------
        # After branching
        # --------------------------

        if "logprobs" not in program_kwargs_after_branching:
            program_kwargs_after_branching["logprobs"] = 0
        else:
            assert program_kwargs_after_branching["logprobs"] in [
                0,
                1,
            ], "logprobs must be 0 or 1"

        if (
            "num_samples" not in program_kwargs_after_branching
            and "={num_samples}" in program_after_branching
        ):
            program_kwargs_after_branching["num_samples"] = 1
        else:
            assert (
                program_kwargs_after_branching.get("num_samples", 1) == 1
            ), "only 1 sample is supported"

        self.program_kwargs_after_branching = program_kwargs_after_branching
        self.program_after_branching = format_string(
            program_after_branching, **program_kwargs_after_branching
        )

        self.max_branching_depth = max_branching_depth

        self.node_text_template = node_text_template
        assert (
            "{chain_of_thought}" in self.node_text_template
        ), "Node_text_template must contain '{chain_of_thought}'"

    async def _sample_node(
        self, prefix: str, depth: int, is_branching_done: bool = False
    ) -> Optional[Node]:
        program = (
            self.program_template
            if not is_branching_done
            else self.program_after_branching
        )
        result = await self._run_program(program, prefix=prefix)
        variables = result.variables()
        if "chain_of_thought" not in variables:
            logger.error("chain_of_thought not found in variables")
            logger.error(f"variables: {variables}")
            return None  # We will sample another node

        chain_of_thought = variables["chain_of_thought"]
        stop_text = variables.get("stop_text", None)

        node_text = self.node_text_template.format(chain_of_thought=chain_of_thought)

        node = {
            "text": node_text,
            "depth": depth,
            "full_text": result.text,
            "stop_text": stop_text,
        }

        return node

    async def expand(self, current_node: Node, prefix: str, depth: int) -> List[Node]:
        branch_factor = self.branch_factor_strategy(current_node)
        if depth > self.max_branching_depth:
            assert branch_factor == 1, (
                f"branch_factor must be 1, but "
                f"got {branch_factor}, depth: {depth}, max_branching_depth: {self.max_branching_depth}"
            )

        is_branching_done = depth > self.max_branching_depth

        nodes = []
        while len(nodes) < branch_factor:
            tasks = []
            for i in range(branch_factor):
                task = asyncio.create_task(
                    self._sample_node(prefix, depth + 1, is_branching_done)
                )
                tasks.append(task)

            for task in tasks:
                node = await task
                if node is not None:
                    nodes.append(node)

            if len(nodes) < branch_factor:
                logger.warning(
                    f"branch_factor: {branch_factor}, but got {len(nodes)} nodes. Will sample again."
                )

        return nodes


@NodeExpander.register("efficient_iid")
class EfficientIIDExpander(NodeExpander):
    def __init__(
        self,
        program: str,
        node_text_template: str,
        program_kwargs: JsonDict,
        num_expansion_rounds: int = 1,
        model_context_size: Optional[int] = None,
        tokenizer: Optional[Tokenizer] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if "logprobs" in program_kwargs:
            program_kwargs["logprobs"] = 0
            logger.warning(
                "logprobs will be set to 0 as it's not supported. Please remove logprobs from program_kwargs."
            )
        else:
            program_kwargs["logprobs"] = 0

        if "num_samples" in program_kwargs:
            program_kwargs.pop("num_samples")
            logger.warning(
                "num_samples will be set by the branch_factor_strategy. "
                "Please remove num_samples from program_kwargs."
            )

        if "stop_regex" in program_kwargs:
            program_kwargs.pop("stop_regex")
            logger.warning("stop_regex is not supported. Please use `stop`")

        self.num_expansion_rounds = num_expansion_rounds
        self.program_kwargs = program_kwargs
        self.program_template = program
        assert (
            '"chain_of_thought"' in self.program_template
        ), "Program_template must contain 'chain_of_thought'"

        self.node_text_template = node_text_template
        assert (
            "{chain_of_thought}" in self.node_text_template
        ), "Node_text_template must contain '{chain_of_thought}'"

        self.model_context_size = model_context_size
        self.tokenizer = tokenizer
        if self.model_context_size is not None:
            assert self.tokenizer is not None, "tokenizer must be provided"

    async def _sample_node(
        self, prefix: str, depth: int, branch_factor: int, seed: Optional[int] = None
    ) -> List[Node]:
        program_kwargs = self.program_kwargs.copy()

        need_to_compute_max_tokens = (
            "max_tokens" in self.program_template
            and self.model_context_size is not None
        )
        if need_to_compute_max_tokens:
            new_max_tokens = self._compute_max_tokens(
                prefix, program_kwargs.get("max_tokens")
            )
            if new_max_tokens != program_kwargs.get("max_tokens"):
                logger.warning(
                    f"Overriding max_tokens: {program_kwargs.get('max_tokens')} -> {new_max_tokens}"
                )
            assert new_max_tokens > 0, f"new_max_tokens: {new_max_tokens}"
            program_kwargs["max_tokens"] = new_max_tokens
        program_kwargs["num_samples"] = branch_factor
        if seed is not None:
            program_kwargs["seed"] = seed

        program = format_string(self.program_template, **program_kwargs)
        result = await self._run_program(program, prefix=prefix)

        variables = result.variables()
        generated_chain_of_thoughts = variables["chain_of_thought"]
        finish_reasons = variables["chain_of_thought_finish_reason"]

        if branch_factor > 1:
            assert len(generated_chain_of_thoughts) == branch_factor
            assert len(finish_reasons) == branch_factor
        else:
            generated_chain_of_thoughts = [generated_chain_of_thoughts]
            finish_reasons = [finish_reasons]

        nodes = []
        for chain_of_thought, finish_reason in zip(
            generated_chain_of_thoughts, finish_reasons
        ):
            node_text = self.node_text_template.format(
                chain_of_thought=chain_of_thought
            )
            full_text = program.replace("{{prefix}}", prefix)
            full_text = replace_gen_pattern(full_text, node_text)
            # full_text = re.sub(r"\{\{gen(.|\s)*}}", node_text, full_text)

            node = {
                "text": node_text,
                "depth": depth,
                "full_text": full_text,
                "stop_text": None,
                "finish_reason": finish_reason,
            }
            nodes.append(node)

        return nodes

    def _compute_max_tokens(
        self, prefix: str, prompt_max_token: Optional[int] = None
    ) -> int:
        num_prefix_tokens = len(self.tokenizer.tokenize(prefix))
        return min(
            self.model_context_size - num_prefix_tokens,
            prompt_max_token if prompt_max_token is not None else float("inf"),
        )

    async def expand(self, current_node: Node, prefix: str, depth: int) -> List[Node]:
        branch_factor = self.branch_factor_strategy(current_node)
        tasks = []
        for i in range(self.num_expansion_rounds):
            seed = self.seed
            if seed is not None:
                seed += i
            task = asyncio.create_task(
                self._sample_node(prefix, depth + 1, branch_factor, seed=seed)
            )
            tasks.append(task)

        all_nodes = []
        for task in tasks:
            nodes = await task
            all_nodes.extend(nodes)

        return all_nodes


@NodeExpander.register("confidence_interval_aware_efficient_iid")
class ConfidenceIntervalAwareEfficientIIDExpander(EfficientIIDExpander):
    def __init__(
        self,
        acceptable_ci_length_threshold: float,
        max_num_rollouts: int,
        num_new_rollouts: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.acceptable_ci_length_threshold = acceptable_ci_length_threshold
        self.max_num_rollouts = max_num_rollouts
        self.num_new_rollouts = num_new_rollouts
        self.rollout_eval_callback: Optional[Callable[..., float]] = None

        from treetune.inference_strategies.tree_inference.answer_extraction import (
            AnswerExtractor,
        )

        self.answer_extractor: Optional[AnswerExtractor] = None

    def set_rollout_eval_callback(self, callback: Callable[..., float]) -> None:
        self.rollout_eval_callback = callback

    def set_answer_extractor(self, answer_extractor) -> None:
        self.answer_extractor = answer_extractor

    async def expand(self, current_node: Node, prefix: str, depth: int) -> List[Node]:
        assert "_request_object" in current_node
        request_obj = current_node["_request_object"]

        branch_factor = self.branch_factor_strategy(current_node)

        num_nodes_per_round = branch_factor
        initial_num_nodes = branch_factor * self.num_expansion_rounds

        async def _sample_new_nodes(num_nodes: int) -> List[Node]:
            i = 0
            sampled = 0
            tasks = []
            while sampled < num_nodes:
                to_sample = min(num_nodes_per_round, num_nodes - sampled)

                task = asyncio.create_task(
                    self._sample_node(prefix, depth + 1, to_sample)
                )
                tasks.append(task)

                sampled += to_sample
                i += 1

            nodes_lst = []
            for task in tasks:
                nodes = await task
                nodes_lst.extend(nodes)

            return nodes_lst

        async def _compute_rewards(nodes_lst: List[Node]) -> List[float]:
            rewards = []
            for node in nodes_lst:
                answer = await self.answer_extractor.extract_from_node(node)
                reward = self.rollout_eval_callback(
                    query=prefix,
                    rollout=answer,
                    finish_reason=node["finish_reason"],
                    request_object=request_obj,
                )
                rewards.append(reward)
            return rewards

        all_nodes = await _sample_new_nodes(initial_num_nodes)
        all_rewards = await _compute_rewards(all_nodes)

        ci_length = self._compute_confidence_interval_length(all_rewards)

        while ci_length > self.acceptable_ci_length_threshold:
            if len(all_nodes) >= self.max_num_rollouts:
                break
            logger.info(
                f"Resampling {self.num_new_rollouts} more rollouts (curr #rolls: {len(all_nodes)}): "
                f"ci_length: {ci_length}, threshold: {self.acceptable_ci_length_threshold}"
            )
            new_nodes = await _sample_new_nodes(self.num_new_rollouts)
            new_rewards = await _compute_rewards(new_nodes)

            all_nodes += new_nodes
            all_rewards += new_rewards

            ci_length = self._compute_confidence_interval_length(all_rewards)

        current_node["ci_length"] = ci_length

        return all_nodes

    def _compute_confidence_interval_length(
        self,
        rewards: List[float],
        t_score: float = 1.96,
    ) -> float:
        estimate = np.mean(rewards)
        ci = t_score * np.sqrt(estimate * (1 - estimate) / len(rewards))
        return 2 * ci


@NodeExpander.register("efficient_iid_for_tree")
class EfficientIIDExpanderForTree(EfficientIIDExpander):
    def __init__(
        self,
        max_branching_depth: int,
        intermediate_stop_sequence: str,
        full_response_stop_sequence: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_branching_depth = max_branching_depth

        if "stop" in self.program_kwargs:
            program_kwarg_stop = self.program_kwargs.pop("stop")
            logger.warning(
                f"stop will be overridden by intermediate_stop_sequence and final_stop_sequence. "
                f"Provided stop: {program_kwarg_stop}"
            )

        self.intermediate_stop_sequence = intermediate_stop_sequence
        self.final_stop_sequence = full_response_stop_sequence

    async def _sample_node(
        self,
        prefix: str,
        depth: int,
        branch_factor: int,
        use_intermediate_stop: bool = False,
    ) -> List[Node]:
        program_kwargs = self.program_kwargs.copy()
        program_kwargs["num_samples"] = branch_factor
        program_kwargs["stop"] = (
            self.intermediate_stop_sequence
            if use_intermediate_stop
            else self.final_stop_sequence
        )
        program = format_string(self.program_template, **program_kwargs)
        result = await self._run_program(program, prefix=prefix)
        variables = result.variables()
        generated_chain_of_thoughts = variables["chain_of_thought"]
        finish_reasons = variables["chain_of_thought_finish_reason"]

        if branch_factor > 1:
            assert len(generated_chain_of_thoughts) == branch_factor
            assert len(finish_reasons) == branch_factor
        else:
            generated_chain_of_thoughts = [generated_chain_of_thoughts]
            finish_reasons = [finish_reasons]

        nodes = []
        for chain_of_thought, finish_reason in zip(
            generated_chain_of_thoughts, finish_reasons
        ):
            node_text = self.node_text_template.format(
                chain_of_thought=chain_of_thought
            )
            full_text = program.replace("{{prefix}}", prefix)
            full_text = replace_gen_pattern(full_text, node_text)
            # full_text = re.sub(r"\{\{gen(.|\s)*}}", node_text, full_text)

            node = {
                "text": node_text,
                "depth": depth,
                "full_text": full_text,
                "stop_text": (
                    None
                    if not use_intermediate_stop
                    else self.intermediate_stop_sequence
                ),
                "finish_reason": finish_reason,
            }
            nodes.append(node)

        return nodes

    async def expand(self, current_node: Node, prefix: str, depth: int) -> List[Node]:
        branch_factor = self.branch_factor_strategy(current_node)
        if depth > self.max_branching_depth:
            assert branch_factor == 1, (
                f"branch_factor must be 1, but "
                f"got {branch_factor}, depth: {depth}, max_branching_depth: {self.max_branching_depth}"
            )

        use_intermediate_stop = depth <= self.max_branching_depth

        tasks = []
        for _ in range(self.num_expansion_rounds):
            task = asyncio.create_task(
                self._sample_node(
                    prefix,
                    depth + 1,
                    branch_factor,
                    use_intermediate_stop=use_intermediate_stop,
                )
            )
            tasks.append(task)

        all_nodes = []
        for task in tasks:
            nodes = await task
            all_nodes.extend(nodes)

        return all_nodes


@NodeExpander.register("high_low_temperature_iid")
class HighLowTemperatureIIDExpander(IIDExpander):
    def __init__(
        self, program: str, node_text_template: str, program_kwargs: JsonDict, **kwargs
    ):
        super(IIDExpander, self).__init__(**kwargs)
        self.program_template = format_string(program, **program_kwargs)
        assert (
            '"chain_of_thought_1"' in self.program_template
        ), "Program_template must contain 'chain_of_thought_1'"
        assert (
            '"chain_of_thought_2"' in self.program_template
        ), "Program_template must contain 'chain_of_thought_2'"

        self.node_text_template = node_text_template
        assert (
            "{chain_of_thought_1}" in self.node_text_template
        ), "Node_text_template must contain '{chain_of_thought}'"
        assert (
            "{chain_of_thought_2}" in self.node_text_template
        ), "Node_text_template must contain '{chain_of_thought}'"

    async def _sample_node(self, prefix: str, depth: int) -> Node:
        result = await self._run_program(self.program_template, prefix=prefix)
        variables = result.variables()
        chain_of_thought_1 = variables["chain_of_thought_1"]
        chain_of_thought_2 = variables["chain_of_thought_2"]
        stop_text = variables["stop_text"]

        node_text = self.node_text_template.format(
            chain_of_thought_1=chain_of_thought_1, chain_of_thought_2=chain_of_thought_2
        )

        node = {
            "text": node_text,
            "depth": depth,
            "full_text": result.text,
            "stop_text": stop_text,
        }

        return node


@NodeExpander.register("bleu_rejection_sampling_iid")
class BleuRejectionSamplingIIDExpander(IIDExpander):
    def __init__(
        self,
        program: str,
        node_text_template: str,
        program_kwargs: JsonDict,
        bleu_acc_threshold: float,
        max_try: int,
        **kwargs,
    ):
        super().__init__(program, node_text_template, program_kwargs, **kwargs)
        # Use a random experiment id
        self.bleu = evaluate.load("bleu", experiment_id=str(uuid.uuid4()))
        self.bleu_acc_threshold = bleu_acc_threshold
        self.max_try = max_try

    async def expand(self, current_node: Node, prefix: str, depth: int) -> List[Node]:
        acc_nodes = []
        try_counter = 0
        branch_factor = self.branch_factor_strategy(current_node)
        while len(acc_nodes) < branch_factor and try_counter < self.max_try:
            nodes = await super().expand(current_node, prefix, depth)
            try_counter += 1

            for node in nodes:
                if len(acc_nodes) == branch_factor:
                    break

                if len(acc_nodes) == 0:
                    acc_nodes.append(node)
                    continue

                if try_counter == self.max_try:
                    logger.info(
                        f"max_try reached, achieved {len(acc_nodes)} nodes, will append remaining nodes"
                    )
                    acc_nodes.append(node)
                    continue

                acc_texts = [acc_node["text"] for acc_node in acc_nodes]
                avg_bleu = self.avg_bleu_of_those_with_this(
                    those_texts=acc_texts, this_text=node["text"]
                )
                if avg_bleu <= self.bleu_acc_threshold:
                    acc_nodes.append(node)

        return acc_nodes

    def avg_bleu_of_those_with_this(self, *, those_texts, this_text):
        preds = []
        refs = []
        for i in range(len(those_texts)):
            that_text = those_texts[i]
            preds.append(that_text)
            refs.append(this_text)
        bleu_full_stats = self.bleu.compute(predictions=preds, references=refs)
        bleu = bleu_full_stats["bleu"]
        return bleu


@NodeExpander.register("iid_with_different_system_message")
class IIDWithDifferentSystemMessageExpander(IIDExpander):
    def __init__(self, system_messages: List[str], sys_msg_regex: str, **kwargs):
        super().__init__(**kwargs)
        self.system_messages = system_messages
        self.sys_msg_regex = re.compile(sys_msg_regex)
        seed = int(os.environ.get("APP_SEED", "42"))
        self.rng = random.Random(seed)

    async def expand(self, current_node: Node, prefix: str, depth: int) -> List[Node]:
        tasks = []
        branch_factor = self.branch_factor_strategy(current_node)
        assert len(self.system_messages) >= branch_factor

        # Sample system messages
        system_messages = self.rng.sample(self.system_messages, branch_factor)
        for new_sys_msg in system_messages:
            # Replace system message in the prefix
            old_sys_msg = self.sys_msg_regex.search(prefix).group(1)
            prefix = prefix.replace(old_sys_msg, new_sys_msg)

            # Sample node with the new prefix
            task = asyncio.create_task(self._sample_node(prefix, depth + 1))
            tasks.append(task)

        nodes = []
        for task in tasks:
            node = await task
            nodes.append(node)

        return nodes
