import copy
import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List

from datasets import Dataset

from treetune import logging_utils
from treetune.analyzers.analyzer import Analyzer
from treetune.common import JsonDict, Params, Lazy
from treetune.inference_strategies import InferenceStrategy
from treetune.runtime.base_runtime import Runtime, BaseRuntime
from treetune.tasks import Task

logger = logging_utils.get_logger(__name__)


@Runtime.register("inference")
class InferenceRuntime(BaseRuntime):
    def __init__(
        self,
        inference_strategy: Lazy[InferenceStrategy],
        task: Task,
        dataset_split: str,
        dataset_portion: float = 1.0,
        dataset_shuffle_before_portion: bool = False,
        dataset_num_shards: int = 1,
        dataset_shard_index: int = 0,
        save_inference_result: bool = True,
        save_inference_result_to_cloud: bool = True,
        analyzers: Optional[List[JsonDict]] = None,
        prompt_library: Optional[JsonDict] = None,
        **kwargs,
    ):
        """
        Params:
            inference_strategy (InferenceStrategy):
                The inference strategy to use
            task (Task):
                The task to use
            dataset_split (str):
                The dataset split from task which will be used for inference.
                It should match the splits provided by the task.
            dataset_portion (float):
                The portion of the dataset to use for inference.
                Useful for debugging.
            dataset_shuffle_before_portion (bool):
                Whether to shuffle the dataset before taking the portion.
            dataset_num_shards (int):
                The number of shards to split the dataset into.
                This is useful for parallelizing inference across multiple jobs.
            dataset_shard_index (int):
                The index of the shard to use for this job.
                This is useful for parallelizing inference across multiple jobs.
            save_inference_result (bool):
                Whether to save the inference result to a file.
            save_inference_result_to_cloud (bool):
                Whether to save the inference result to the cloud (Wandb).
            prompt_library (JsonDict):
                This is not used in the code. It is only here to make the
                dependency injection library happy.
        """
        super().__init__(**kwargs)

        self.task = task

        self.dataset_split = dataset_split
        self.dataset_portion = dataset_portion
        self.dataset_shuffle_before_portion = dataset_shuffle_before_portion
        self.dataset_num_shards = dataset_num_shards
        self.dataset_shard_index = dataset_shard_index

        self.save_inference_result = save_inference_result
        self.save_inference_result_to_cloud = save_inference_result_to_cloud

        if self.debug_mode:
            logger.info(
                "Debug mode is on. Setting save_inference_result_to_cloud to False."
            )
            # self.save_inference_result = False
            self.save_inference_result_to_cloud = False

            logger.info("Debug mode is on. Using 10 examples from the dataset.")
            dataset_len = len(self.task.get_datasets(self.dataset_split))
            self.dataset_portion = 10 / dataset_len

        # Unique identifier for this inference job.
        self.inference_job_id = f"{self.dataset_split}__{self.dataset_shard_index}__{self.dataset_num_shards}"
        logger.info(
            f"Inference Job ID (split__shard__num_shards): {self.inference_job_id}"
        )
        self.inference_strategy = inference_strategy.construct(
            result_dir=self._get_result_dir(), cloud_logger=self.cloud_logger
        )
        self.analyzers = analyzers

    def generate(self):
        self._log_method_call_to_console(logger, "generate")

        from treetune.common.py_utils import is_world_process_zero

        if not is_world_process_zero():
            logger.warning("Skipping generate() because this is not the main process.")
            return

        dataset = self.task.get_datasets(self.dataset_split)
        logger.info(f"Original Dataset size: {len(dataset)}")

        if self.dataset_portion < 1.0:
            if self.dataset_shuffle_before_portion:
                dataset = dataset.shuffle(seed=self.global_vars["seed"])
            dataset = dataset.select(range(int(len(dataset) * self.dataset_portion)))
            logger.info(
                f"Portion {self.dataset_portion} of Dataset size: {len(dataset)}"
            )

        logger.info(
            f"Dataset Examples: "
            f"{json.dumps([dataset[i] for i in range(min(5, len(dataset)))], indent=2, sort_keys=True)}"
        )

        dataset = dataset.shard(
            self.dataset_num_shards, self.dataset_shard_index, contiguous=True
        )
        logger.info(
            f"Sharded Dataset size: {len(dataset)} "
            f"(shard {self.dataset_shard_index+1} / {self.dataset_num_shards})"
        )

        results = self.inference_strategy.generate(dataset)

        if self.save_inference_result:
            self._save_inference_result(results)

        if self.save_inference_result_to_cloud:
            self._save_inference_result_to_cloud(results)

    def analyze(self):
        self._log_method_call_to_console(logger, "analyze")

        from treetune.common.py_utils import is_world_process_zero

        if not is_world_process_zero():
            logger.warning("Skipping analyze() because this is not the main process.")
            return

        if self.analyzers is None:
            logger.warning("self.analyzers is None. Exiting...")
            return

        analyzers = copy.deepcopy(self.analyzers)
        for config_obj in analyzers:
            analyzer = Analyzer.from_params(
                Params(config_obj),
                cloud_logger=self.cloud_logger,
                runtime=self,
            )

            logger.info(f"Using {analyzer.__class__.__name__}...")
            analyzer.analyze()
            analyzer.flush_local_log()

    def generate_and_analyze(self):
        self.generate()
        self.analyze()

    def _save_inference_result(self, results: Dataset):
        output_dir = self._get_result_dir()
        logger.info(f"Saving inference results to {output_dir}")
        results.save_to_disk(str(output_dir))

    def _save_inference_result_to_cloud(self, results: Dataset):
        if self.cloud_logger is None:
            logger.warning(
                "Cloud logger is not set. "
                "Cannot save inference results to the cloud."
                "This command has no effect."
            )
            return

        output_dir = self._get_result_dir()
        if not output_dir.exists():
            logger.warning(
                f"Output directory {output_dir} does not exist. "
                "First saving inference results to local disk."
            )
            self._save_inference_result(results)

        # First, create a zip file of the inference results into output_dir/inference_results.zip
        # This is because the cloud logger only accepts files.
        temp_dir = Path(self.exp_root) / next(tempfile._get_candidate_names())
        temp_dir.mkdir(parents=True, exist_ok=True)

        inference_results_zip = temp_dir / "inference_results.zip"
        logger.info(f"Creating zip file {inference_results_zip}")
        shutil.make_archive(
            str(inference_results_zip.with_suffix("")), "zip", output_dir
        )

        # Then, upload the zip file to the cloud.
        self.cloud_logger.save(str(inference_results_zip.absolute()))

    def _get_result_dir(self) -> Path:
        result_dir = self.exp_root / "inference_results" / self.inference_job_id
        result_dir.mkdir(parents=True, exist_ok=True)
        return result_dir
