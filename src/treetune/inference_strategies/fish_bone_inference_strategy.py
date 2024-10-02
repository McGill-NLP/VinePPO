import asyncio
import json
from typing import List, Optional, Dict, Any

from datasets import Dataset

import guidance
from guidance.llms import OpenAI, OpenAIVLLM
from treetune import logging_utils
from treetune.common import guidance_utils as gu, Registrable, Lazy
from treetune.inference_strategies.base_inference_strategy import InferenceStrategy
from treetune.inference_strategies.tree_inference import Node
from treetune.inference_strategies.tree_inference.answer_extraction import (
    AnswerExtractor,
)
from treetune.inference_strategies.tree_inference.expansion import NodeExpander
from treetune.inference_strategies.tree_inference_strategy import GuidanceLLM
from treetune.tasks import Task
from treetune.tokenization_utils.base_tokenizer import Tokenizer

logger = logging_utils.get_logger(__name__)

@InferenceStrategy.register("fish_bone", exist_ok=True)
class FishBoneInferenceStrategy(InferenceStrategy):
    def __init__(
        self,
        node_expander: NodeExpander,
        answer_extractor: AnswerExtractor,
        guidance_llm: Lazy[GuidanceLLM],
        task: Task,  # for knowing how to split the response
        question_template: str,  # to know how to make the actual query to the model
        chosen_response_for_fish_bone_field: str,  # probably 'chosen_answer' in our code base
        fish_bone_n_samples: int = 3,
        max_concurrent_programs: int = 128,
        max_concurrent_generations: int = 2048,
        seed: Optional[int] = None,
        tokenizer: Optional[Tokenizer] = None,
        no_cache: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.node_expander = node_expander
        self.answer_extractor = answer_extractor
        self.seed = seed
        self.chosen_response_for_fish_bone_field = chosen_response_for_fish_bone_field

        self.task = task
        self.question_template = question_template

        self.max_concurrent_programs = max_concurrent_programs
        self.max_concurrent_generations = max_concurrent_generations

        self.guidance_llm_lazy = guidance_llm
        self.no_cache = no_cache

        self.tokenizer = tokenizer

        self.node_expander.set_seed(seed)
        self.answer_extractor.set_seed(seed)

        self.fish_bone_n_samples = fish_bone_n_samples

        if self.log_level is not None:
            logger.setLevel(self.log_level)

    def generate(self, dataset: Dataset) -> Dataset:
        return asyncio.run(self._concurrent_generate(dataset))

    async def _concurrent_generate(self, dataset: Dataset) -> Dataset:
        # Create a semaphore to limit the number of concurrent programs
        sem_program = asyncio.Semaphore(self.max_concurrent_programs)

        # Create a semaphore to limit the number of concurrent generations
        sem_generation = asyncio.Semaphore(self.max_concurrent_generations)

        async def sem_run_program(*args, **kwargs):
            async with sem_program:
                return await gu.run_program(*args, **kwargs)

        async def wrapper_construct_fish_bone(fish_bone_idx, *args, **kwargs):
            async with sem_generation:
                try:
                    fb = await self._construct_fish_bone(*args, **kwargs)
                    return fish_bone_idx, fb
                except:
                    # If there is an error, we just exit the program
                    # as soon as possible, otherwise the program will continue
                    # blocking the semaphore and thus blocking the entire process
                    exit(1)

        # Set the guidance LLM
        guidance.llm = self.guidance_llm_lazy.construct()

        self.node_expander.set_run_program(sem_run_program)
        self.answer_extractor.set_run_program(sem_run_program)

        assert self.chosen_response_for_fish_bone_field in dataset.column_names, (
            f"Chosen response for fish bone field '{self.chosen_response_for_fish_bone_field}' "
            f"must be in the dataset. Available columns: {dataset.column_names}"
        )

        tasks = []
        fish_bones = {}
        from tqdm import tqdm as tqdm_iter

        for data_instance in tqdm_iter(
            dataset,
            desc="Creating concurrent asyncio tasks for fish bone construction...",
        ):
            instance_idx = data_instance["_treetune__idx"]

            if not self.no_cache:
                fish_bone_file_path = self.get_fish_bone_instance_path(instance_idx)
                try:
                    with fish_bone_file_path.open("r") as f:
                        fish_bone = json.load(f)
                        logger.info(f"Loaded fish bone from {fish_bone_file_path}")
                    assert len(fish_bone) > 0
                    fish_bones[instance_idx] = fish_bone
                    # Skip if the tree is already constructed
                    continue
                except FileNotFoundError:
                    pass
                except Exception as e:
                    # If the file exists but is corrupted, we log the error and re-construct the tree
                    logger.error(f"Error loading fish bone from {fish_bone_file_path}, possibly corrupted, reconstructing: {e}")

            tasks.append(
                asyncio.create_task(
                    wrapper_construct_fish_bone(instance_idx, data_instance)
                )
            )

        # Report the current progress to cloud logger
        if self.cloud_logger is not None:
            self.cloud_logger.log({"construction_progress": len(fish_bones) / len(dataset)})

        # Create a progress bar for the fish bone construction tasks
        from tqdm.asyncio import tqdm as tqdm_asyncio

        # Maintain a progress bar for the fish bone construction tasks.
        # It updates whenever any of the tasks finishes.
        for task in tqdm_asyncio.as_completed(tasks, desc="Constructing fish bones"):
            instance_idx, fish_bone = await task
            fish_bones[instance_idx] = fish_bone
            fish_bone_file_path = self.get_fish_bone_instance_path(instance_idx)
            with fish_bone_file_path.open("w") as f:  # so we can resume later on
                json.dump(fish_bone, f)

            if self.cloud_logger is not None:
                self.cloud_logger.log(
                    {"construction_progress": len(fish_bones) / len(dataset)}
                )

        fish_bones = [
            fish_bones[idx] for idx in dataset["_treetune__idx"]
        ]  # change order back to original
        assert len(fish_bones) == len(
            dataset
        ), f"len(fish_bones)={len(fish_bones)}, len(dataset)={len(dataset)}"

        # Utility function to create a dataset
        def create_column(column_name, extraction_method):
            return Dataset.from_dict(
                {column_name: [extraction_method(fish_bone) for fish_bone in fish_bones]}
            )[column_name]

        # Add new columns to the dataset
        dataset = dataset.add_column(
            'fish_bone', create_column("fish_bone", self._convert_fish_bone_to_string)
        )

        return dataset

    def make_initial_query(self, instance: Dict[str, Any]) -> str:
        return self.question_template.format(query=instance["problem"])

    async def _construct_fish_bone(self, instance):
        # first we split the solution to its steps
        initial_query_to_model = self.make_initial_query(instance)
        solution = instance[self.chosen_response_for_fish_bone_field]
        step_indices = self.task.split_solution_into_intermediate_steps(solution)
        steps = [solution[step_indices[i]:step_indices[i+1]] for i in range(len(step_indices) - 1)]
        # Now, we create a tree of breadth 1 as a chain
        # first, the root
        initial_prompt = initial_query_to_model + steps[0]
        tree = {
            "text": initial_prompt,
            "depth": 0,
            "full_text": initial_prompt,
            "children": [],
            "fish_bone_children": [],
            "is_in_spine": True, # this is the spine of the fish bone
        }
        # then the rest of the nodes
        last_node = tree
        for i in range(1, len(steps)):
            node = {
                "text": steps[i],
                "depth": i,
                "full_text": last_node['full_text']+steps[i],
                "children": [],
                "fish_bone_children": [],
                "is_in_spine": True, # this is the spine of the fish bone
            }
            last_node['children'].append(node)
            last_node = node

        # Now, we do the fish bone expansion using DFS, DFS is probably overkill, but we're familiar with it because
        # of the previous implementations of the tree inference strategies
        async def dfs(node: Node, prefix: str, depth: int) -> None:
            # TODO(Milad): make it so for out of spine, expands like 3 times, but for in spine, only once
            if len(node['children']) == 0:
                return  # we're at the end of the spine

            fish_bone_children = await self.node_expander.expand(node, prefix, depth)
            node["fish_bone_children"] = fish_bone_children
            for child in fish_bone_children:
                child['is_in_spine'] = False

            # Either the child has finished (and we need to extract the answer)
            # or we need to expand the child further.
            # Both tasks can be done concurrently.
            answer_extraction_tasks = []
            children_expansion_tasks = []
            for child in fish_bone_children:
                # Check if the child can be produce an answer
                if child["stop_text"] is None:
                    # This means we have reached the end of the reasoning chain
                    answer_extraction_tasks.append(
                        asyncio.create_task(
                            self.answer_extractor.extract_from_node(child)
                        )
                    )
                else:
                    # If the child cannot produce an answer, we continue the search
                    # by expanding the child
                    answer_extraction_tasks.append(None)
                    children_expansion_tasks.append(
                        asyncio.create_task(dfs(child, child["full_text"], depth + 1))
                    )

            # Wait for the answer extraction tasks to finish
            for child, answer in zip(fish_bone_children, answer_extraction_tasks):
                if answer is not None:
                    child["answer"] = await answer

            # continue the fish bone construction down the spine
            if 'children' in node and len(node['children']) > 0:
                spine_children = node['children']
                assert len(spine_children) == 1, "Fish bone has a single spine"
                spine_child = spine_children[0]
                spine_child_task = asyncio.create_task(dfs(spine_child, spine_child["full_text"], depth + 1))
                children_expansion_tasks.append(spine_child_task)

            # Wait for the children expansion tasks to finish
            await asyncio.gather(*children_expansion_tasks)

        await dfs(tree, initial_prompt, 0)

        return tree

    def _convert_fish_bone_to_string(self, fish_bone: Node) -> str:
        fish_bone_str = json.dumps(fish_bone, indent=4, sort_keys=True)
        return fish_bone_str

    def get_temp_fish_bone_dir(self):
        temp_fish_bone_dirs = self.result_dir / "fish_bones"
        temp_fish_bone_dirs.mkdir(parents=True, exist_ok=True)
        return temp_fish_bone_dirs

    def get_fish_bone_instance_path(self, instance_idx):
        return self.get_temp_fish_bone_dir() / f"{instance_idx}.json"