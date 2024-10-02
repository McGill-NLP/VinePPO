import copy
import os
import pickle
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from treetune import logging_utils
from treetune.analyzers.analyzer import Analyzer
from treetune.common.py_utils import need_to_minimize_stored_files

logger = logging_utils.get_logger(__name__)


@Analyzer.register("mc_advantage_distribution")
class MCAdvantageDistributionAnalyzer(Analyzer):
    def __init__(
        self,
        cloud_logger: Run,
        runtime,
        max_num_iterations: Optional[int] = None,
        **kwargs,
    ):
        from treetune.runtime.policy_iteration_runtime import PolicyIterationRuntime

        assert isinstance(runtime, PolicyIterationRuntime)
        self.runtime: PolicyIterationRuntime = runtime

        super().__init__(cloud_logger, runtime, **kwargs)

        self.max_num_iterations = max_num_iterations

    def _get_list_of_evaluation_iterations(
        self, ignore_worker_vars: bool = False
    ) -> List[Path]:
        episodes_dir = self.runtime.exp_root / "temp_episodes"
        episodes_dir.mkdir(exist_ok=True, parents=True)

        def can_analyze(iter_path: Path) -> bool:
            return all(
                (p / "trajectories.pkl").exists()
                for p in iter_path.glob("infer_results/process_*")
            )

        available_iterations = [
            iter_dir
            for iter_dir in episodes_dir.glob("iteration__*")
            if can_analyze(iter_dir)
        ]
        available_iterations.sort(key=lambda x: int(x.name.replace("iteration__", "")))

        # Limit the number of checkpoints to analyze
        # noinspection DuplicatedCode
        if self.max_num_iterations is not None:
            num_iterations = len(available_iterations)
            if num_iterations > self.max_num_iterations:
                step = num_iterations / self.max_num_iterations
                available_iterations = [
                    available_iterations[int(i * step)]
                    for i in range(self.max_num_iterations)
                ]

        if (
            not ignore_worker_vars
            and "APP_EVAL_NUM_WORKERS" in os.environ
            and "APP_EVAL_WORKER_ID" in os.environ
        ):
            # Distribute the iterations across workers, such that each worker
            # evaluates an equal number of iterations
            num_workers = int(os.environ["APP_EVAL_NUM_WORKERS"])
            worker_id = int(os.environ["APP_EVAL_WORKER_ID"])
            logger.info(
                f"Running evaluation on worker {worker_id+1} out of {num_workers}"
            )
            available_iterations = available_iterations[worker_id::num_workers]

        return available_iterations

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

        iterations = self._get_list_of_evaluation_iterations()
        logger.info(
            f"Evaluating Advantage distribution on {len(iterations)} checkpoints: "
            f"{[iter_.name for iter_ in iterations]}"
        )

        for iter_ in tqdm(iterations, desc="Analyzing checkpoints"):
            self._analyze_iteration(iter_, force_rerun=force_rerun)

        iterations = self._get_list_of_evaluation_iterations(ignore_worker_vars=True)
        all_iters_are_done = all(
            (analysis_root / iter_.name / "done").exists() for iter_ in iterations
        )
        if all_iters_are_done:
            (analysis_root / "done").touch()

    def _analyze_iteration(self, iteration: Path, force_rerun: bool = False):
        iteration_name = iteration.name
        iter_eval_root_dir = self.get_analysis_root_dir() / iteration_name
        iter_eval_root_dir.mkdir(exist_ok=True, parents=True)

        if not force_rerun and (iter_eval_root_dir / "done").exists():
            logger.info(f"Skipping {iteration} as it has already been analyzed.")
            return
        logger.info(f"Analyzing iteration {iteration_name}")

        trajectories = []
        for process_dir in iteration.glob("infer_results/process_*"):
            with open(process_dir / "trajectories.pkl", "rb") as f:
                trajectories.extend(pickle.load(f))
        logger.info(f"Loaded {len(trajectories)} trajectories")

        all_advantages = []
        all_values = []
        for i, traj in enumerate(trajectories):
            num_steps = len(traj["steps"])
            values = copy.deepcopy(traj["values"])
            advantages = self._compute_step_advantages(traj)
            for j, adv in enumerate(advantages):
                all_advantages.append(
                    {
                        "adv": adv,
                        "step_idx": j,
                        "trajectory": i,
                        "num_steps": num_steps,
                    }
                )

            for j, val in enumerate(values[:-1]):
                if val is None:
                    continue
                all_values.append(
                    {
                        "value": val,
                        "step_idx": j,
                        "trajectory": i,
                        "num_steps": num_steps,
                    }
                )

        df = pd.DataFrame(all_advantages)
        df_values = pd.DataFrame(all_values)

        file_name_prefix = "__".join(
            [
                "analysis",
                self.__class__.__name__,
                self.get_analysis_id(),
                iteration_name,
            ]
        )

        if need_to_minimize_stored_files():
            save_dir = Path(tempfile.mkdtemp())
        else:
            save_dir = iter_eval_root_dir

        df_path = save_dir / f"{file_name_prefix}__advantages.csv.gz"
        df.to_csv(df_path, index=False, compression="gzip")

        df_values_path = save_dir / f"{file_name_prefix}__values.csv.gz"
        df_values.to_csv(df_values_path, index=False, compression="gzip")

        # Upload to wandb
        if self.cloud_logger is not None:
            self.cloud_logger.save(str(df_path.absolute()), policy="now")
            self.cloud_logger.save(str(df_values_path.absolute()), policy="now")

        if self.plot_prefix is not None:
            self._plot_adv_histogram(
                df,
                plot_id=f"{self.plot_prefix}/{self.__class__.__name__}/{iteration_name}",
            )
            self._plot_value_histogram(
                df_values,
                plot_id=f"{self.plot_prefix}/{self.__class__.__name__}_val/{iteration_name}",
            )
        (iter_eval_root_dir / "done").touch()

    # noinspection DuplicatedCode
    def _compute_step_advantages(self, trajectory: Dict[str, Any]):
        step_rewards = trajectory["step_rewards"]
        values = trajectory["values"]

        # The value of the final/terminating state is by definition 0
        values[-1] = 0.0

        # Fill in the missing values from the end
        assert all(v is not None for v in values)

        advantages = [None] * len(step_rewards)
        assert len(advantages) == len(values) - 1
        assert len(advantages) == len(step_rewards)
        for i in range(len(advantages)):
            advantages[i] = step_rewards[i] + values[i + 1] - values[i]

        return advantages

    def _plot_adv_histogram(self, df: pd.DataFrame, plot_id: str) -> None:
        if self.cloud_logger is None:
            return
        df = df.copy()
        df = df[df["num_steps"] > 2]
        df["rel_step"] = (df["step_idx"] + 1) / df["num_steps"]

        # Add a small normal noise to avoid overlapping points
        df["adv"] += np.random.normal(loc=0, scale=0.01, size=len(df))
        df["rel_step"] += np.random.normal(loc=0, scale=0.01, size=len(df))

        data = df[["adv", "rel_step"]].values.tolist()
        table = wandb.Table(data=data, columns=["Advantage", "Relative Reasoning Step"])
        plot = wandb.plot.scatter(
            table,
            x="Relative Reasoning Step",
            y="Advantage",
            title=plot_id.split("/")[-1],
        )
        self.cloud_logger.log({plot_id: plot})

    def _plot_value_histogram(self, df: pd.DataFrame, plot_id: str) -> None:
        if self.cloud_logger is None:
            return
        df = df.copy()
        df = df[df["num_steps"] > 2]
        df["rel_step"] = (df["step_idx"] + 1) / df["num_steps"]

        # Add a small normal noise to avoid overlapping points
        df["value"] += np.random.normal(loc=0, scale=0.01, size=len(df))
        df["rel_step"] += np.random.normal(loc=0, scale=0.01, size=len(df))

        data = df[["value", "rel_step"]].values.tolist()
        table = wandb.Table(data=data, columns=["Value", "Relative Reasoning Step"])
        plot = wandb.plot.scatter(
            table,
            x="Relative Reasoning Step",
            y="Value",
            title=plot_id.split("/")[-1],
        )
        self.cloud_logger.log({plot_id: plot})
