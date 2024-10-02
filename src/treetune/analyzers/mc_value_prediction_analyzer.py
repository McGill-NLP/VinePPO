import gc
import json
import os
import pickle
import random
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Any

from datasets import Dataset, load_from_disk
from tqdm import tqdm

from treetune import logging_utils
from treetune.analyzers.analyzer import Analyzer
from treetune.analyzers.valnet_prediction_analyzer import ValNetPredictionAnalyzer
from treetune.common.py_utils import need_to_minimize_stored_files
from treetune.common.wandb_utils import save_inference_result_to_cloud
from treetune.trainers.policy_trainer import PolicyTrainer

logger = logging_utils.get_logger(__name__)


@Analyzer.register("mc_value_prediction")
class MCValuePredictionAnalyzer(ValNetPredictionAnalyzer):

    def _get_list_of_evaluation_checkpoints(
        self, every_n_checkpoints: int, ignore_worker_vars: bool = False
    ) -> List[Tuple[Path, Path]]:
        checkpoint_dir = self.runtime.exp_root / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True, parents=True)

        trajectory_dir = self.runtime.exp_root / "temp_episodes"
        trajectory_dir.mkdir(exist_ok=True, parents=True)

        # noinspection PyProtectedMember
        ckpts = self.runtime._get_list_of_evaluation_checkpoints(
            checkpoint_dir, every_n_checkpoints, ignore_worker_vars=True
        )

        def can_analyze(ckpt: Path) -> bool:
            if not (ckpt / "hf_pretrained" / "pytorch_model.bin").exists():
                return False
            ckpt_iter = int(PolicyTrainer.parse_checkpoint_name(ckpt.name)[0])
            traj_path = trajectory_dir / f"iteration__{ckpt_iter:04d}"
            return all(
                (p / "trajectories.pkl").exists()
                for p in traj_path.glob("infer_results/process_*")
            )

        # noinspection DuplicatedCode
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

        traj_dirs = [
            (
                trajectory_dir
                / f"iteration__{int(PolicyTrainer.parse_checkpoint_name(ckpt.name)[0]):04d}"
            )
            for ckpt in ckpts
        ]

        return list(zip(ckpts, traj_dirs))

    def analyze(
        self, every_n_checkpoints: int = 1, force_rerun: bool = False, **kwargs
    ):
        # noinspection DuplicatedCode
        analysis_root = self.get_analysis_root_dir()
        analysis_root.mkdir(exist_ok=True, parents=True)

        if not force_rerun and (analysis_root / "done").exists():
            logger.warning(
                f"Analysis directory {self.get_analysis_root_dir()} already exists."
            )
            return

        ckpts = self._get_list_of_evaluation_checkpoints(every_n_checkpoints)
        logger.info(
            f"Evaluating MC Prediction Accuracy on {len(ckpts)} "
            f"checkpoints: {[ckpt.name for ckpt,_ in ckpts]}"
        )

        for ckpt, episode_dir in tqdm(ckpts, desc="Analyzing checkpoints"):
            self._analyze_checkpoint(ckpt, episode_dir, force_rerun=force_rerun)

        all_ckpts_are_done = all(
            (analysis_root / ckpt.name / "done").exists()
            for ckpt, _ in self._get_list_of_evaluation_checkpoints(
                every_n_checkpoints, ignore_worker_vars=True
            )
        )
        if all_ckpts_are_done:
            (analysis_root / "done").touch()

    # noinspection PyMethodOverriding
    def _analyze_checkpoint(
        self, ckpt: Path, episode_dir: Path, force_rerun: bool = False
    ):
        # noinspection DuplicatedCode
        ckpt_eval_root_dir = self.get_analysis_root_dir() / ckpt.name
        ckpt_eval_root_dir.mkdir(exist_ok=True, parents=True)

        if not force_rerun and (ckpt_eval_root_dir / "done").exists():
            logger.info(f"Skipping {ckpt} as it has already been analyzed.")
            return

        logger.info(f"Analyzing checkpoint {ckpt}")

        trajectories = []
        process_dirs = list(episode_dir.glob("infer_results/process_*"))
        process_dirs.sort(key=lambda p: int(p.name.split("_")[-1]))
        for process_dir in process_dirs:
            trajectories.extend(self._load_trajectories(process_dir))
        logger.info(f"Loaded {len(trajectories)} trajectories")
        gc.collect()

        requests = self._create_mc_value_requests(trajectories)
        logger.info(f"Created {len(requests)} MC value requests")

        if self.max_num_requests is not None and len(requests) > self.max_num_requests:
            requests = random.Random(42).sample(requests, self.max_num_requests)
        requests = Dataset.from_list(requests)

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
        gt_values = []
        gt_returns = []
        avg_num_unique_continuations = []
        for res in tqdm(results, desc="Computing ground truth mc values"):
            traj_idx = res["traj_idx"]
            traj = trajectories[traj_idx]

            data_instance = traj["ds_instance"]
            value_idx = res["value_idx"]

            # Perform sanity checks
            # noinspection DuplicatedCode
            reconst_query = "".join(
                ([traj["query_text"]] + traj["steps"])[: value_idx + 1]
            )
            assert reconst_query == res["query"]
            assert reconst_query == json.loads(res["_treetune__reasoning_tree"])["text"]
            assert data_instance["problem"] in res["query"]

            continuations = res["_treetune__candidate_answers"]
            avg_num_unique_continuations.append(len(set(continuations)))

            gt_value, returns = self._compute_mc_value(
                query=res["query"],
                value_estimation_result=res,
                data_instance=data_instance,
            )
            gt_values.append(gt_value)
            gt_returns.append(returns)

        results = results.add_column("gt_value", gt_values)
        results = results.add_column(
            "avg_num_unique_continuations", avg_num_unique_continuations
        )
        results = results.add_column("gt_mc_returns", gt_returns)

        # Remove unnecessary columns to save space
        unnecessary_columns = [
            "_treetune__candidate_answers",
            "_treetune__reasoning_tree",
        ]
        for col in unnecessary_columns:
            if col in results.column_names:
                results = results.remove_columns(col)

        query_state_results = results.filter(lambda x: x["is_query"])
        non_query_state_results = results.filter(lambda x: not x["is_query"])

        iter_idx = int(PolicyTrainer.parse_checkpoint_name(ckpt.name)[0])
        if self.plot_prefix is not None:
            self._plot_pred_vs_ground_truth(
                non_query_state_results,
                f"{self.plot_prefix}/{self.__class__.__name__}/iter_{iter_idx:04d}",
            )
            self._plot_pred_vs_ground_truth(
                query_state_results,
                f"{self.plot_prefix}/{self.__class__.__name__}__query_state/iter_{iter_idx:04d}",
            )

        save_inference_result_to_cloud(
            results,
            f"{self.__class__.__name__}___iter_{iter_idx:04d}",
            cloud_logger=self.cloud_logger,
            policy="now",
        )
        (ckpt_eval_root_dir / "done").touch()

    def _create_mc_value_requests(
        self, trajectories: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        all_requests = []
        request_idx = 0
        for traj_idx, traj in enumerate(trajectories):
            reasoning_steps = traj["steps"]
            pred_values = traj["values"]

            # Add the query state
            all_requests.append(
                {
                    "gt_solution": traj["ds_instance"]["solution"],
                    "query": traj["query_text"],
                    "traj_idx": traj_idx,
                    "value_idx": 0,
                    "step_idx": -1,
                    "total_steps": len(reasoning_steps),
                    "is_last_step": False,
                    "is_query": True,
                    "predicted_value": pred_values[0],
                    "is_complete_response": not traj["is_unfinished_response"],
                    "_treetune__idx": request_idx,
                }
            )

            request_idx += 1

            # We don't need to estimate the value of the final step
            # Since we can't take any action after that
            max_step_idx = len(reasoning_steps) - 2

            # Add the reasoning steps
            for step_idx, step in enumerate(reasoning_steps):
                if step_idx > max_step_idx:
                    break

                query = self._create_step_query(traj, step_idx)
                all_requests.append(
                    {
                        "gt_solution": traj["ds_instance"]["solution"],
                        "query": query,
                        "traj_idx": traj_idx,
                        "value_idx": step_idx + 1,
                        "step_idx": step_idx,
                        "total_steps": len(reasoning_steps),
                        "is_last_step": False,
                        "is_query": False,
                        "predicted_value": pred_values[step_idx + 1],
                        "is_complete_response": not traj["is_unfinished_response"],
                        "_treetune__idx": request_idx,
                    }
                )

                request_idx += 1

        # Make sure the there's no duplicate request ids
        assert len(all_requests) == len(set(r["_treetune__idx"] for r in all_requests))

        return all_requests

    # noinspection DuplicatedCode
    def _create_step_query(self, traj: Dict[str, Any], step_idx: int) -> str:
        query_text = traj["query_text"]
        response_text = traj["response_text"]
        full_text = traj["full_text"]
        indices = traj["step_indices"]
        response_up_to_step = response_text[: indices[step_idx + 1]]

        request_query = query_text + response_up_to_step
        assert full_text.startswith(request_query), f"{full_text} != {request_query}"

        return request_query

    def _load_trajectories(self, process_dir: Path) -> List[Dict[str, Any]]:
        traj_infer_result = load_from_disk(str(process_dir / "traj_results_ds"))
        with open(process_dir / "trajectories.pkl", "rb") as f:
            trajectories = pickle.load(f)

        for traj in trajectories:
            ds_instance = traj_infer_result[traj["instance_idx"]]
            assert ds_instance[self.problem_field] in traj["query_text"]
            traj["ds_instance"] = ds_instance

        return trajectories
