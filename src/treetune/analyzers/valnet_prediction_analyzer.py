import copy
import gc
import json
import os
import random
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Callable, Any

import wandb
from datasets import Dataset, load_from_disk
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from treetune import logging_utils
from treetune.analyzers.analyzer import Analyzer
from treetune.common import Lazy
from treetune.common.py_utils import need_to_minimize_stored_files
from treetune.common.vllm_server import VLLMServer
from treetune.common.wandb_utils import save_inference_result_to_cloud
from treetune.episode_generators.episode_generator_with_reward_function import (
    RewardFunction,
)
from treetune.inference_strategies import InferenceStrategy
from treetune.tasks import Task
from treetune.tokenization_utils import Tokenizer
from treetune.trainers.policy_trainer import PolicyTrainer

logger = logging_utils.get_logger(__name__)


@Analyzer.register("valnet_prediction")
class ValNetPredictionAnalyzer(Analyzer):
    def __init__(
        self,
        task: Task,
        tokenizer: Tokenizer,
        inference_strategy: Lazy[InferenceStrategy],
        vllm_server: Lazy[VLLMServer],
        cloud_logger: Run,
        runtime,
        reward_function: Lazy[RewardFunction],
        solution_delimiter: str = "\nSolution:",
        dataset_split: str = "train",
        max_num_checkpoints: Optional[int] = None,
        max_num_requests: Optional[int] = None,
        problem_field: str = "problem",
        **kwargs,
    ):
        from treetune.runtime.policy_iteration_runtime import PolicyIterationRuntime

        assert isinstance(runtime, PolicyIterationRuntime)
        self.runtime: PolicyIterationRuntime = runtime
        super().__init__(cloud_logger, runtime, **kwargs)
        self.max_num_checkpoints = max_num_checkpoints
        self.dataset_split = dataset_split
        self.tokenizer = tokenizer
        self.solution_delimiter = solution_delimiter
        self.max_num_requests = max_num_requests
        self.inference_strategy_lazy = inference_strategy
        self.vllm_server_lazy = vllm_server
        self.reward_function = reward_function.construct(tokenizer=tokenizer)
        self.problem_field = problem_field

        self.task = task
        assert hasattr(self.task, "split_solution_into_intermediate_steps"), (
            f"Task {self.task} does not have a method "
            f"to split the solution into reasoning steps."
        )

    def _get_list_of_evaluation_checkpoints(
        self, every_n_checkpoints: int, ignore_worker_vars: bool = False
    ) -> List[Tuple[Path, Path]]:
        checkpoint_dir = self.runtime.exp_root / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True, parents=True)

        ckpts = self.runtime._get_list_of_evaluation_checkpoints(
            checkpoint_dir, every_n_checkpoints, ignore_worker_vars=True
        )

        available_episodes = [
            episode_dir
            for episode_dir in checkpoint_dir.glob("episodes__iter*")
            if episode_dir.is_dir() and (episode_dir / "w_actLogp_and_values").exists()
        ]

        iter_to_episode_dir = {}
        for episode_dir in available_episodes:
            iter_idx = int(episode_dir.name.split("__iter")[-1])
            iter_to_episode_dir[iter_idx] = episode_dir / "w_actLogp_and_values"

        def can_analyze(ckpt: Path) -> bool:
            if not (ckpt / "hf_pretrained" / "pytorch_model.bin").exists():
                return False
            ckpt_iter = int(PolicyTrainer.parse_checkpoint_name(ckpt.name)[0])
            return ckpt_iter in iter_to_episode_dir

        ckpts = [ckpt for ckpt in ckpts if can_analyze(ckpt)]

        # Limit the number of checkpoints to analyze
        if self.max_num_checkpoints is not None:
            num_ckpts = len(ckpts)
            if num_ckpts > self.max_num_checkpoints:
                step = num_ckpts / self.max_num_checkpoints
                ckpts = [ckpts[int(i * step)] for i in range(self.max_num_checkpoints)]

        if (
            not ignore_worker_vars
            and "APP_EVAL_NUM_WORKERS" in os.environ
            and "APP_EVAL_WORKER_ID" in os.environ
        ):
            # Distribute the checkpoints across workers, such that each worker
            # evaluates an equal number of checkpoints
            num_workers = int(os.environ["APP_EVAL_NUM_WORKERS"])
            worker_id = int(os.environ["APP_EVAL_WORKER_ID"])
            logger.info(
                f"Running evaluation on worker {worker_id+1} out of {num_workers}"
            )
            ckpts = ckpts[worker_id::num_workers]

        episode_dirs = [
            iter_to_episode_dir[int(PolicyTrainer.parse_checkpoint_name(ckpt.name)[0])]
            for ckpt in ckpts
        ]

        return list(zip(ckpts, episode_dirs))

    def analyze(
        self, every_n_checkpoints: int = 1, force_rerun: bool = False, **kwargs
    ):
        analysis_root = self.get_analysis_root_dir()
        analysis_root.mkdir(exist_ok=True, parents=True)

        if not force_rerun and (analysis_root / "done").exists():
            logger.warning(
                f"Analysis directory {self.get_analysis_root_dir()} already exists."
            )
            return

        ckpts = self._get_list_of_evaluation_checkpoints(every_n_checkpoints)
        logger.info(
            f"Evaluating {len(ckpts)} checkpoints:"
            f"{json.dumps([ckpt.name for ckpt,_ in ckpts], indent=2)}"
        )

        dataset = self.task.get_datasets(self.dataset_split)
        for ckpt, episode_dir in tqdm(ckpts, desc="Analyzing checkpoints"):
            self._analyze_checkpoint(
                ckpt, episode_dir, dataset, force_rerun=force_rerun
            )

        all_ckpts_are_done = all(
            (analysis_root / ckpt.name / "done").exists()
            for ckpt, _ in self._get_list_of_evaluation_checkpoints(
                every_n_checkpoints, ignore_worker_vars=True
            )
        )
        if all_ckpts_are_done:
            (analysis_root / "done").touch()

    def _analyze_checkpoint(
        self, ckpt: Path, episode_dir: Path, dataset: Dataset, force_rerun: bool = False
    ):
        ckpt_eval_root_dir = self.get_analysis_root_dir() / ckpt.name
        ckpt_eval_root_dir.mkdir(exist_ok=True, parents=True)

        if not force_rerun and (ckpt_eval_root_dir / "done").exists():
            logger.info(f"Skipping {ckpt} as it has already been analyzed.")
            return

        logger.info(f"Analyzing checkpoint {ckpt}")
        episodes = load_from_disk(str(episode_dir))
        episodes = self._attach_dataset_instance_idx(episodes, dataset)
        gc.collect()

        requests, full_trajectory_requests = self._create_mc_value_requests(episodes)
        logger.info(f"Created {len(requests)} MC value requests")
        logger.info(f"Created {len(full_trajectory_requests)} full trajectory requests")

        if self.max_num_requests is not None and len(requests) > self.max_num_requests:
            requests = random.Random(42).sample(requests, self.max_num_requests)
        requests = Dataset.from_list(requests)
        full_trajectory_requests = Dataset.from_list(full_trajectory_requests)

        if need_to_minimize_stored_files():
            results_path = Path(tempfile.mkdtemp()) / "inference_results"
        else:
            results_path = ckpt_eval_root_dir / "inference_results"
        results = self._obtain_inference_results(
            requests,
            ckpt,
            results_path,
            seed=42,
        )
        gc.collect()

        # Update episodes with the value estimates
        avg_num_unique_continuations = []
        gt_values = []
        returns_lst = []
        for res in tqdm(results, desc="Computing ground truth mc values"):
            data_instance = dataset[res["ds_instance_idx"]]
            assert res["query"] == json.loads(res["_treetune__reasoning_tree"])["text"]
            assert data_instance[self.problem_field] in res["query"]

            continuations = res["_treetune__candidate_answers"]
            avg_num_unique_continuations.append(len(set(continuations)))

            gt_value, returns = self._compute_mc_value(
                query=res["query"],
                value_estimation_result=res,
                data_instance=data_instance,
            )
            gt_values.append(gt_value)
            returns_lst.append(returns)

        # Remove unnecessary columns to save space
        unnecessary_columns = [
            "_treetune__candidate_answers",
            "_treetune__reasoning_tree",
        ]
        for col in unnecessary_columns:
            if col in results.column_names:
                results = results.remove_columns(col)

        # Add gt_values as a new column to the results
        results = results.add_column("gt_value", gt_values)
        results = results.add_column(
            "avg_num_unique_continuations", avg_num_unique_continuations
        )
        results = results.add_column("gt_mc_returns", returns_lst)

        global_step = int(PolicyTrainer.parse_checkpoint_name(ckpt.name)[-1])
        iter_idx = int(PolicyTrainer.parse_checkpoint_name(ckpt.name)[0])
        if self.plot_prefix is not None:
            self._plot_pred_vs_ground_truth(
                results,
                f"{self.plot_prefix}/{self.__class__.__name__}/iter_{iter_idx:04d}",
            )

            # Compute MSE for the full trajectory requests
            if len(full_trajectory_requests) > 0:
                mses = [
                    (d["predicted_value"] - d["gt_value"]) ** 2
                    for d in full_trajectory_requests
                ]
                mse = sum(mses) / len(mses)
                self.cloud_logger.log(
                    {
                        f"{self.plot_prefix}/{self.__class__.__name__}_fullTraj": mse,
                        "train/global_step": global_step,
                    }
                )

        # Upload the results to the cloud logger
        save_inference_result_to_cloud(
            results,
            f"{self.__class__.__name__}___iter_{iter_idx:04d}",
            cloud_logger=self.cloud_logger,
            policy="now",
        )
        save_inference_result_to_cloud(
            full_trajectory_requests,
            f"{self.__class__.__name__}_fullTraj___iter_{iter_idx:04d}",
            cloud_logger=self.cloud_logger,
            policy="now",
        )

        (ckpt_eval_root_dir / "done").touch()

    def _plot_pred_vs_ground_truth(self, results: Dataset, plot_id: str) -> None:
        if self.cloud_logger is None:
            return

        gt_values = results["gt_value"]
        predicted_values = results["predicted_value"]

        data = [[x, y] for (x, y) in zip(gt_values, predicted_values)]
        table = wandb.Table(data=data, columns=["Ground Truth Value", "Pred. Value"])
        plot = wandb.plot.scatter(
            table, "Ground Truth Value", "Pred. Value", title=plot_id.split("/")[-1]
        )
        self.cloud_logger.log({plot_id: plot})

    # noinspection DuplicatedCode
    def _compute_mc_value(
        self,
        *,
        query: str = None,
        value_estimation_result: Dict[str, Any] = None,
        data_instance: Dict[str, Any] = None,
    ) -> Tuple[float, List[float]]:
        tree = json.loads(value_estimation_result["_treetune__reasoning_tree"])
        rollouts = [(c["answer"], c["finish_reason"]) for c in tree["children"]]

        rewards = [
            (
                self.reward_function(query, rol, data_instance)[0]
                if finish_reason != "length"
                else self.reward_function.get_unfinished_response_penalty()
            )
            for rol, finish_reason in rollouts
        ]

        if len(rewards) == 0:
            mc_value = 0.0
        else:
            mc_value = sum(rewards) / len(rewards)

        return mc_value, rewards

    def _obtain_inference_results(
        self,
        requests_ds: Dataset,
        checkpoint: Path,
        results_path: Path,
        seed: int,
    ) -> Dataset:
        request_ids = requests_ds["_treetune__idx"]
        assert len(request_ids) == len(set(request_ids)), "Duplicate request ids found."

        vllm_server = self.vllm_server_lazy.construct(seed=42)
        hf_ckpt_path_or_model = checkpoint / "hf_pretrained"

        # Make sure the vLLM server can seamlessly load the model
        self.tokenizer.save_pretrained(hf_ckpt_path_or_model)

        server_url = vllm_server.start_server(
            hf_ckpt_path_or_model=str(hf_ckpt_path_or_model),
            wait_for_response=True,
            log_path=results_path.parent / f"{results_path.stem}.vllm_log",
            timeout=800,
        )
        guidance_llm_kwargs = {
            "api_base": server_url,
            "model": str(hf_ckpt_path_or_model),
        }

        # Initialize the inference strategy with the vLLM server URL
        inference_strategy_lazy = copy.deepcopy(self.inference_strategy_lazy)
        # noinspection PyProtectedMember
        inference_strategy_lazy._params["guidance_llm"].update(guidance_llm_kwargs)
        infer_strategy = inference_strategy_lazy.construct(
            result_dir=results_path.parent / f"{results_path.stem}.infer_strategy",
            seed=seed,
            cloud_logger=None,
        )

        results = infer_strategy.generate(requests_ds)
        results.save_to_disk(str(results_path))
        vllm_server.stop_server()

        return Dataset.load_from_disk(str(results_path))

    def _create_mc_value_requests(
        self, episodes: Dataset
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        full_trajectory_requests = []
        requests = []
        request_idx = 0
        for ep_idx, ep in enumerate(episodes):
            query_token_ids = ep["query_token_ids"]
            response_token_ids = ep["response_token_ids"]
            values = ep["critic_values"]
            assert len(values) == (len(query_token_ids) + len(response_token_ids))

            is_complete_response = response_token_ids[-1] == self.tokenizer.eos_token_id

            try:
                step_end_indices = self._get_intermediate_step_end_indices(
                    query_token_ids, response_token_ids
                )
            except Exception as e:
                logger.warning(
                    f"Failed to extract intermediate steps for episode {ep_idx}: {e}"
                )
                continue

            # Add intermediate steps
            for step_idx, end_idx in enumerate(step_end_indices):
                if is_complete_response and step_idx == len(step_end_indices) - 1:
                    continue

                request_token_ids = query_token_ids + response_token_ids[: end_idx + 1]
                val_net_prediction = values[len(request_token_ids) - 1]

                # Remove <bos>
                if request_token_ids[0] == self.tokenizer.bos_token_id:
                    request_token_ids = request_token_ids[1:]

                request_text = self.tokenizer.decode(
                    request_token_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )

                requests.append(
                    {
                        "query": request_text,
                        "predicted_value": val_net_prediction,
                        "ds_instance_idx": ep["ds_instance_idx"],
                        "episode_idx": ep_idx,
                        "step_idx": step_idx,
                        "is_last_step": False,
                        "total_steps": len(step_end_indices),
                        "is_complete_response": is_complete_response,
                        "_treetune__idx": request_idx,
                    }
                )

                request_idx += 1

            # Add the last step
            if is_complete_response:
                request_token_ids = query_token_ids + response_token_ids

                # We take the value of the second last token since
                # we don't train value net on the last token
                val_net_prediction = values[-2]

                # Remove <bos>
                if request_token_ids[0] == self.tokenizer.bos_token_id:
                    request_token_ids = request_token_ids[1:]

                request_text = self.tokenizer.decode(
                    request_token_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )

                full_trajectory_requests.append(
                    {
                        "query": request_text,
                        "predicted_value": val_net_prediction,
                        "ds_instance_idx": ep["ds_instance_idx"],
                        "step_idx": len(step_end_indices) - 1,
                        "total_steps": len(step_end_indices),
                        "is_last_step": True,
                        "is_complete_response": True,
                        "gt_value": ep["scores"],
                        "episode_idx": ep_idx,
                        "_treetune__idx": ep_idx,
                    }
                )

        return requests, full_trajectory_requests

    def _get_intermediate_step_end_indices(
        self, query_token_ids: List[int], response_token_ids: List[int]
    ) -> List[int]:
        trajectory_text = self.tokenizer.decode(
            query_token_ids + response_token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        response_parts = trajectory_text.split("\nSolution:")
        assert len(response_parts) == 2
        response = response_parts[1]

        # noinspection PyUnresolvedReferences
        indices = self.task.split_solution_into_intermediate_steps(response)

        # Create a map from character index to step index
        char_to_step_idx = [-1] * len(response)
        for i, (start, end) in enumerate(zip(indices[:-1], indices[1:])):
            for j in range(start, end):
                char_to_step_idx[j] = i
        assert all(i != -1 for i in char_to_step_idx)

        offsets = self._get_response_tokenization_offsets(response, response_token_ids)

        # Create a map from token index to step index
        token_to_step_idx = [-1] * len(response_token_ids)
        for i in range(len(response_token_ids)):
            start, _ = offsets[i]
            token_to_step_idx[i] = char_to_step_idx[start]

        # Find step boundaries
        step_end_indices = []
        i = 0
        while i < len(token_to_step_idx):
            step_idx = token_to_step_idx[i]
            while i < len(token_to_step_idx) and token_to_step_idx[i] == step_idx:
                i += 1
            step_end_indices.append(i - 1)

        return step_end_indices

    def _get_response_tokenization_offsets(
        self, response: str, response_token_ids: List[int]
    ) -> List[Tuple[int, int]]:
        encoding = self.tokenizer(
            response, add_special_tokens=False, return_offsets_mapping=True
        )
        tokens_ids = encoding["input_ids"]
        offsets = encoding["offset_mapping"]

        first_token = encoding.tokens()[0]
        if first_token == "▁":
            offsets = offsets[1:]
            tokens_ids = tokens_ids[1:]

        assert len(offsets) == len(tokens_ids)
        assert tokens_ids == response_token_ids, f"{tokens_ids} != {response_token_ids}"

        return offsets

    def _attach_dataset_instance_idx(
        self, episodes: Dataset, dataset: Dataset
    ) -> Dataset:
        dataset = dataset.to_list()

        query_tok_ids = [tuple(episode["query_token_ids"]) for episode in episodes]
        unique_query_tok_ids = sorted(set(query_tok_ids))
        unique_query_tok_ids = Dataset.from_dict(
            {"query_token_ids": unique_query_tok_ids}
        )
        unique_query_tok_ids = unique_query_tok_ids.map(
            self._get_data_instance_search_fn(dataset),
            num_proc=4,
            desc="Attaching dataset instance idx",
        )

        query_tok_id_to_ds_idx = {
            tuple(instance["query_token_ids"]): instance["ds_instance_idx"]
            for instance in unique_query_tok_ids
        }

        episodes = episodes.map(
            lambda example: {
                "ds_instance_idx": query_tok_id_to_ds_idx[
                    tuple(example["query_token_ids"])
                ],
            },
            num_proc=4,
            desc="Attaching dataset instance idx",
        )

        return episodes

    def _get_data_instance_search_fn(
        self, dataset: List[Dict[str, Any]]
    ) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        tokenizer = self.tokenizer
        problem_field = self.problem_field

        def search_fn(example: Dict[str, Any]) -> Dict[str, Any]:
            query_text = tokenizer.decode(
                example["query_token_ids"],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            instance_idx = next(
                idx
                for idx, instance in enumerate(dataset)
                if instance[problem_field] in query_text
            )

            return {"ds_instance_idx": instance_idx}

        return search_fn
