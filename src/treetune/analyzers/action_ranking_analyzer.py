import copy
import gc
import gzip
import itertools
import json
import pickle
import random
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

from accelerate.utils import release_memory
from datasets import Dataset, load_from_disk
from tqdm import tqdm

from treetune import logging_utils
from treetune.analyzers import ValNetPredictionAnalyzer
from treetune.common import Lazy
from treetune.common.gpu_utils import get_gpu_memory, wait_for_memory_release
from treetune.common.py_utils import need_to_minimize_stored_files
from treetune.inference_strategies import InferenceStrategy
from treetune.trainers.policy_trainer import PolicyTrainer

logger = logging_utils.get_logger(__name__)


class ActionRankingAnalyzer(ValNetPredictionAnalyzer):
    def __init__(
        self,
        min_num_alternative_actions: int,
        alternative_continuation_inference_strategy: Lazy[InferenceStrategy],
        max_num_states: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alt_cont_inference_strategy_lazy = (
            alternative_continuation_inference_strategy
        )
        self.min_num_alternative_actions = min_num_alternative_actions
        self.max_num_states = max_num_states

    def _analyze_checkpoint(
        self, ckpt: Path, episode_dir: Path, dataset: Dataset, force_rerun: bool = False
    ):
        # noinspection DuplicatedCode
        ckpt_eval_root_dir = self.get_analysis_root_dir() / ckpt.name
        ckpt_eval_root_dir.mkdir(exist_ok=True, parents=True)

        if not force_rerun and (ckpt_eval_root_dir / "done").exists():
            logger.info(f"Skipping {ckpt} as it has already been analyzed.")
            return

        logger.info(f"Analyzing checkpoint {ckpt}")
        episodes = load_from_disk(str(episode_dir))
        episodes = self._attach_dataset_instance_idx(episodes, dataset)
        gc.collect()

        # Create inference requests to sample alternative continuations per each reasoning step
        alt_cont_reqs = self._create_alternative_continuations_requests(episodes)
        logger.info(f"Created {len(alt_cont_reqs)} alternative continuation requests")

        if (
            self.max_num_requests is not None
            and len(alt_cont_reqs) > self.max_num_requests
        ):
            alt_cont_reqs = random.Random(42).sample(
                alt_cont_reqs, min(self.max_num_requests, len(alt_cont_reqs))
            )
        alt_cont_reqs = Dataset.from_list(alt_cont_reqs)

        # Start the vLLM server
        vllm_log_path = ckpt_eval_root_dir / "vllm_server.log"
        llm_kwargs = self._start_vllm_server(ckpt, vllm_log_path)

        if need_to_minimize_stored_files():
            infer_result_root = Path(tempfile.mkdtemp())
        else:
            infer_result_root = ckpt_eval_root_dir

        # 1. Sample alternative continuations
        alt_cont_results = self._perform_inference(
            requests_ds=alt_cont_reqs,
            results_path=infer_result_root / "alt_cont_inference_results",
            inference_strategy=self.alt_cont_inference_strategy_lazy,
            llm_kwargs=llm_kwargs,
            seed=42,
        )

        # 2.1. Convert the alternative continuation results to list of states with alternative actions
        states = self._convert_alt_cont_results_to_states(alt_cont_results)

        # 2.2. Filter states with less than `min_num_alternative_actions` alternative actions
        states = [
            state
            for state in states
            if len(state["next_actions"]) >= self.min_num_alternative_actions
        ]
        logger.info(f"Filtered states to {len(states)}")
        if self.max_num_states is not None:
            states = random.Random(42).sample(
                states, min(self.max_num_states, len(states))
            )

        # 3. Compute ground truth value of Q(s,a') for every state-action pair
        state_action_val_requests = self._create_mc_value_requests(states)
        state_action_val_requests = Dataset.from_list(state_action_val_requests)
        logger.info(
            f"Created {len(state_action_val_requests)} state-action mc value requests"
        )
        state_action_val_gt_results = self._perform_inference(
            requests_ds=state_action_val_requests,
            results_path=infer_result_root / "state_action_val_inference_results",
            inference_strategy=self.inference_strategy_lazy,
            llm_kwargs=llm_kwargs,
            seed=42,
        )

        # Kill the vLLM server as it is no longer needed
        self._kill_vllm_server()

        # Update states with the ground truth values
        for res in tqdm(
            state_action_val_gt_results, desc="Computing ground truth mc values"
        ):
            state = states[res["state_idx"]]
            ds_instance_dx = state["orig__ds_instance_idx"]
            data_instance = dataset[ds_instance_dx]

            assert res["query"] == json.loads(res["_treetune__reasoning_tree"])["text"]
            assert data_instance[self.problem_field] in res["query"]

            gt_value, returns = self._compute_mc_value(
                query=res["query"],
                value_estimation_result=res,
                data_instance=data_instance,
            )

            action_idx = res["action_idx"]
            assert state["next_action_gt_values"][action_idx] is None
            assert state["next_action_gt_mc_returns"][action_idx] is None
            state["next_action_gt_values"][action_idx] = gt_value
            state["next_action_gt_mc_returns"][action_idx] = returns
        del state_action_val_gt_results

        # Make sure all next action gt values are filled
        for state in states:
            assert all(gt_val is not None for gt_val in state["next_action_gt_values"])
            assert all(
                gt_ret is not None for gt_ret in state["next_action_gt_mc_returns"]
            )

        release_memory()

        # 4. Compute Q(s,a) for every state-action pair using the critic
        state_action_val_pred_results = self._compute_pred_values(
            state_action_val_requests, ckpt, states
        )

        # Update states with the value estimates
        for res in state_action_val_pred_results:
            state = states[res["state_idx"]]
            action_idx = res["action_idx"]
            assert state["next_action_pred_values"][action_idx] is None
            state["next_action_pred_values"][action_idx] = res["pred_value"]

        for state in states:
            assert all(
                pred_val is not None for pred_val in state["next_action_pred_values"]
            )
            # Attach the data instance to the state
            state["data_instance"] = dataset[state["orig__ds_instance_idx"]]

        self._plot_ranking_performance(states, ckpt)

        # 5. Save the results as compressed pickle
        if need_to_minimize_stored_files():
            save_dir = Path(tempfile.mkdtemp())
        else:
            save_dir = ckpt_eval_root_dir

        file_name_prefix = "__".join(
            [
                "analysis",
                self.__class__.__name__,
                self.get_analysis_id(),
                ckpt.name,
            ]
        )
        save_path = save_dir / f"{file_name_prefix}__states.pkl.gz"
        with gzip.open(save_path, "wb") as f:
            # noinspection PyTypeChecker
            pickle.dump(states, f)

        if self.cloud_logger is not None:
            self.cloud_logger.save(str(save_path.absolute()), policy="now")

        (ckpt_eval_root_dir / "done").touch()

    def _plot_ranking_performance(
        self, states: List[Dict[str, Any]], ckpt: Path
    ) -> None:
        # Convert all possible comparisons between actions
        # and compute the performance as binary classification
        all_comparisons = []
        for state in states:
            next_actions = state["next_actions"]
            for i, j in itertools.combinations(list(range(len(next_actions))), 2):
                gt_i = state["next_action_gt_values"][i]
                gt_j = state["next_action_gt_values"][j]

                pred_i = state["next_action_pred_values"][i]
                pred_j = state["next_action_pred_values"][j]

                is_correct = (gt_i - gt_j) * (pred_i - pred_j) > 0

                all_comparisons.append(is_correct)

        ranking_acc = sum(all_comparisons) / len(all_comparisons)

        global_step = int(PolicyTrainer.parse_checkpoint_name(ckpt.name)[-1])
        self.cloud_logger.log(
            {
                f"{self.plot_prefix}/{self.__class__.__name__}": ranking_acc,
                "train/global_step": global_step,
            }
        )

    # noinspection DuplicatedCode
    def _create_alternative_continuations_requests(
        self, episodes: Dataset
    ) -> List[Dict[str, Any]]:
        requests = []
        request_idx = 0
        for ep_idx, ep in enumerate(episodes):
            # noinspection PyTypeChecker
            query_token_ids = ep["query_token_ids"]
            # noinspection PyTypeChecker
            response_token_ids = ep["response_token_ids"]
            if "critic_values" in ep:
                # noinspection PyTypeChecker
                values = ep["critic_values"]
                assert len(values) == (len(query_token_ids) + len(response_token_ids))
            else:
                values = [None] * (len(query_token_ids) + len(response_token_ids))

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

                # noinspection PyTypeChecker
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
                        "step_end_indices": step_end_indices,
                        "_treetune__idx": request_idx,
                    }
                )

                request_idx += 1

        return requests

    # noinspection DuplicatedCode

    def _convert_alt_cont_results_to_states(
        self, alt_cont_results: Dataset
    ) -> List[Dict[str, Any]]:
        states = []
        for res in alt_cont_results:
            query = res["query"]

            tree = json.loads(res["_treetune__reasoning_tree"])
            full_continuations = [c["full_text"] for c in tree["children"]]

            next_actions = []
            for cont in full_continuations:
                action = self._convert_continuation_to_next_action(cont, query)
                if action is None:
                    continue
                next_actions.append(action)

            next_actions = sorted(list(set(next_actions)))
            if len(next_actions) <= 1:
                continue

            states.append(
                {
                    "state": query,
                    "next_actions": next_actions,
                    "next_action_gt_values": [None] * len(next_actions),
                    "next_action_gt_mc_returns": [None] * len(next_actions),
                    "next_action_pred_values": [None] * len(next_actions),
                    **{f"orig__{k}": v for k, v in res.items() if k != "query"},
                }
            )

        return states

    def _convert_continuation_to_next_action(
        self, full_continuation: str, query: str
    ) -> Optional[str]:
        assert full_continuation.startswith(query)

        solution_parts = full_continuation.split(self.solution_delimiter)
        if len(solution_parts) != 2:
            logger.warning(
                f"Failed to split full_continuation into query,solution: "
                f"`{full_continuation}`\n"
                "#" * 80
            )
            return None
        continued_solution = solution_parts[1]

        solution = query.split(self.solution_delimiter)[1]
        assert continued_solution.startswith(solution)

        # noinspection PyUnresolvedReferences
        try:
            indices = self.task.split_solution_into_intermediate_steps(
                continued_solution
            )
        except Exception as e:
            logger.warning(f"Failed to split solution into steps: {e}")
            return None

        steps = [
            continued_solution[indices[i] : indices[i + 1]]
            for i in range(len(indices) - 1)
        ]

        # Iterate over the steps and find the next action by
        # comparing the steps with the query
        next_action = None
        for step_idx in range(len(steps) - 1):
            continued_sol_up_to_curr_step = "".join(steps[: step_idx + 1])
            if solution == continued_sol_up_to_curr_step:
                next_action = steps[step_idx + 1]
                break

        return next_action

    def _create_mc_value_requests(
        self, states: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        requests = []
        request_idx = 0
        for state_idx, state in enumerate(states):
            state_text = state["state"]
            for action_idx, next_action in enumerate(state["next_actions"]):
                request_text = state_text + next_action

                requests.append(
                    {
                        "query": request_text,
                        "state_idx": state_idx,
                        "action_idx": action_idx,
                        "_treetune__idx": request_idx,
                    }
                )

                request_idx += 1

        return requests

    def _perform_inference(
        self,
        requests_ds: Dataset,
        results_path: Path,
        inference_strategy: Lazy[InferenceStrategy],
        llm_kwargs: Dict[str, str],
        seed: int,
    ) -> Dataset:
        request_ids = requests_ds["_treetune__idx"]
        assert len(request_ids) == len(set(request_ids)), "Duplicate request ids found."

        # Initialize the inference strategy with the vLLM server URL
        inference_strategy_lazy = copy.deepcopy(inference_strategy)
        # noinspection PyProtectedMember
        inference_strategy_lazy._params["guidance_llm"].update(llm_kwargs)
        infer_strategy = inference_strategy_lazy.construct(
            result_dir=results_path.parent / f"{results_path.stem}.infer_strategy",
            seed=seed,
            cloud_logger=None,
        )

        results = infer_strategy.generate(requests_ds)
        results.save_to_disk(str(results_path))

        return Dataset.load_from_disk(str(results_path))

    def _compute_pred_values(
        self,
        state_action_val_requests: Union[List[Dict[str, Any]], Dataset],
        ckpt: Path,
        states: List[Dict[str, Any]],
    ) -> Union[List[Dict[str, Any]], Dataset]:
        raise NotImplementedError

    def _start_vllm_server(self, checkpoint: Path, log_path: Path) -> Dict[str, str]:
        assert (
            getattr(self, "vllm_server", None) is None
        ), "vLLM server is already running."

        # Save gpu memory usage before starting the vLLM server so that
        # we can check if the memory has been released after the server is stopped
        if self.distributed_state is not None:
            self._gpu_idx = self.distributed_state.device.index
        else:
            self._gpu_idx = 0
        self._gpu_memory_usage_before_mb = get_gpu_memory()[self._gpu_idx]

        vllm_server = self.vllm_server_lazy.construct(seed=42)
        self.vllm_server = vllm_server

        hf_ckpt_path_or_model = checkpoint / "hf_pretrained"
        self.tokenizer.save_pretrained(hf_ckpt_path_or_model)
        server_url = vllm_server.start_server(
            hf_ckpt_path_or_model=str(hf_ckpt_path_or_model),
            wait_for_response=True,
            log_path=log_path,
            timeout=800,
        )

        llm_kwargs = {
            "api_base": server_url,
            "model": str(hf_ckpt_path_or_model),
        }

        return llm_kwargs

    def _kill_vllm_server(self) -> None:
        self.vllm_server.stop_server()

        threshold_mb = self._gpu_memory_usage_before_mb * 1.1  # Allow for 10% tolerance
        wait_for_memory_release(
            self._gpu_idx,
            threshold_mb=threshold_mb,
        )

        self.vllm_server = None
