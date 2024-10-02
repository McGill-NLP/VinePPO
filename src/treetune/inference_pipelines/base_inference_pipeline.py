import copy
import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List

from datasets import Dataset
from wandb.sdk.wandb_run import Run

from treetune.analyzers import Analyzer
from treetune.common import Registrable, Lazy, JsonDict, Params
from treetune.common.py_utils import need_to_minimize_stored_files
from treetune.inference_strategies.base_inference_strategy import InferenceStrategy
from treetune.tasks.base_task import Task

from treetune.logging_utils import get_logger


logger = get_logger(__name__)


class InferencePipeline(Registrable):
    pass


@InferencePipeline.register("vllm")
class VLLMInferencePipeline(InferencePipeline):
    def __init__(
        self,
        inference_strategy: Lazy[InferenceStrategy],
        task: Task,
        dataset_split: str,
        inference_name: Optional[str] = None,
        exp_root: Optional[Path] = None,
        dataset_portion: float = 1.0,
        dataset_shuffle_before_portion: bool = False,
        dataset_num_shards: int = 1,
        dataset_shard_index: int = 0,
        analyzers: Optional[List[JsonDict]] = None,
        debug_mode: bool = False,
        cloud_logger: Optional[Run] = None,
        use_cache: Optional[bool] = None,
        enable_cloud_logging_during_inference: bool = True,
        seed: int = 42,
        metrics_prefix: str = "",
        api_base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        prompt_library: Optional[JsonDict] = None,
        checkpoint_global_step: Optional[int] = None,
    ):
        """
        Params:
            exp_root (Path):
                The root directory of the experiment.
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
            analyzers (List[JsonDict]):
                A list of analyzers to use.
            debug_mode (bool):
                Whether to run in debug mode.
            use_cache (bool):
                Whether to use cache.
            enable_cloud_logging_during_inference (bool):
                Whether to enable cloud logging during inference.
            seed (int):
                The seed to use.
            metrics_prefix (str):
                The prefix to use for metrics.
        """
        self.task = task
        self.exp_root = exp_root
        self.dataset_split = dataset_split
        self.dataset_portion = dataset_portion
        self.dataset_shuffle_before_portion = dataset_shuffle_before_portion
        self.dataset_num_shards = dataset_num_shards
        self.dataset_shard_index = dataset_shard_index
        self.debug_mode = debug_mode
        self.cloud_logger = cloud_logger
        self.seed = seed
        self.metrics_prefix = metrics_prefix
        self.inference_name = inference_name
        self.checkpoint_global_step = checkpoint_global_step

        if self.inference_name is not None:
            self.metrics_prefix = f"{self.inference_name}/{self.metrics_prefix}"

        if self.exp_root is None:
            # Create a tmp directory
            self.exp_root = Path("/tmp") / next(tempfile._get_candidate_names())
            self.exp_root.mkdir(parents=True, exist_ok=True)

        if self.debug_mode:
            logger.info("Debug mode is on. Using 10 examples from the dataset.")
            dataset_len = len(self.task.get_datasets(self.dataset_split))
            self.dataset_portion = 10 / dataset_len

        # Unique identifier for this inference job.
        self.inference_job_id = f"{self.dataset_split}__{self.dataset_shard_index}__{self.dataset_num_shards}"
        logger.info(
            f"Inference on {task.name} (split__shard__num_shards): {self.inference_job_id}"
        )

        if need_to_minimize_stored_files():
            self._cached_results_dir = (
                Path(tempfile.mkdtemp()) / "inference_results" / self.inference_job_id
            )
            self._cached_results_dir.mkdir(parents=True, exist_ok=True)

        inference_strategy_kwargs = {"result_dir": self.get_result_dir()}
        if enable_cloud_logging_during_inference:
            inference_strategy_kwargs["cloud_logger"] = self.cloud_logger
        if use_cache is not None:
            inference_strategy._params.pop("no_cache", None)
            inference_strategy._constructor_extras.pop("no_cache", None)
            # P.S. no_cache was a stupid name. We should refactor it some time.
            inference_strategy_kwargs["no_cache"] = not use_cache
        if api_base_url is not None:
            if "guidance_llm" not in inference_strategy._params:
                logger.warning("api_base_url is not used in the inference strategy.")
            else:
                inference_strategy._params["guidance_llm"]["api_base"] = api_base_url
        if model_name is not None:
            if "guidance_llm" not in inference_strategy._params:
                logger.warning("model_name is not used in the inference strategy.")
            else:
                inference_strategy._params["guidance_llm"]["model"] = model_name

        logger.info(
            f"Guidance LLM params: {inference_strategy._params.get('guidance_llm', {}).as_dict()}"
        )
        self.inference_strategy = inference_strategy.construct(
            **inference_strategy_kwargs
        )
        self.analyzers = analyzers

    def get_result_dir(self) -> Path:
        if hasattr(self, "_cached_results_dir"):
            return self._cached_results_dir

        result_dir = self.exp_root / "inference_results" / self.inference_job_id
        result_dir.mkdir(parents=True, exist_ok=True)
        return result_dir

    def _get_result_dir(self) -> Path:
        return self.get_result_dir()

    def generate(self):
        dataset = self.task.get_datasets(self.dataset_split)
        logger.info(f"Original Dataset size: {len(dataset)}")

        if self.dataset_portion < 1.0:
            if self.dataset_shuffle_before_portion:
                dataset = dataset.shuffle(seed=42)
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

        return results

    def analyze(self, results: Dataset):
        if self.analyzers is None:
            logger.warning("self.analyzers is None. Exiting...")
            return

        # First save the results on disk
        results.save_to_disk(self.get_result_dir())

        analyzer_outputs = []
        analyzers = copy.deepcopy(self.analyzers)
        for config_obj in analyzers:
            analyzer = Analyzer.from_params(
                Params(config_obj),
                cloud_logger=self.cloud_logger,
                runtime=self,
                metrics_prefix=self.metrics_prefix,
                global_step=self.checkpoint_global_step,
                plot_prefix=f"eval/{self.inference_name}",
            )
            output = analyzer.analyze()
            analyzer_outputs.append(output)
            analyzer.flush_local_log()

        return analyzer_outputs

    def save_results_to_cloud(self, results: Dataset):
        output_dir = self._get_result_dir()
        results.save_to_disk(output_dir)

        # First, create a zip file of the inference results into output_dir/inference_results.zip
        # This is because the cloud logger only accepts files.
        temp_dir = Path(tempfile.mkdtemp())
        inference_results_zip = (
            temp_dir / f"{self.metrics_prefix.replace('/', '__')}inference_results.zip"
        )
        logger.info(f"Creating zip file {inference_results_zip}")
        shutil.make_archive(
            str(inference_results_zip.with_suffix("")), "zip", output_dir
        )

        # Then, upload the zip file to the cloud.
        self.cloud_logger.save(str(inference_results_zip.absolute()), policy="now")


InferencePipeline.default_implementation = "vllm"
