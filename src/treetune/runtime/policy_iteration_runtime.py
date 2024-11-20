import copy
import hashlib
import json
import os
import pickle
import shlex
import shutil
import subprocess
import time
import weakref
from pathlib import Path
from typing import Optional, List, Any, Union

import torch
from accelerate import DistributedType
from accelerate.utils import broadcast
from datasets import Dataset
from transformers import AutoConfig

from treetune.analyzers import Analyzer
from treetune.common import Lazy, JsonDict, Params
from treetune.common.notebook_utils import get_repo_dir
from treetune.common.py_utils import need_to_minimize_stored_files
from treetune.common.vllm_server import VLLMServer
from treetune.episode_generators.base_episode_generator import EpisodeGenerator
from treetune.inference_pipelines.base_inference_pipeline import InferencePipeline
from treetune.logging_utils import get_logger
from treetune.models.base_model import Model
from treetune.runtime.base_runtime import DistributedRuntime, Runtime
from treetune.tokenization_utils.base_tokenizer import Tokenizer
from treetune.trainers.base_trainer import Trainer
from treetune.trainers.deepspeed_policy_trainer import DeepSpeedPolicyTrainer
from treetune.trainers.policy_trainer import PolicyTrainer

logger = get_logger(__name__)


def get_zero_to_fp32_script_path() -> Path:
    """Get the path to the `zero_to_fp32.py` script."""
    return get_repo_dir() / "scripts" / "zero_to_fp32.py"


@Runtime.register("policy_iteration")
class PolicyIterationRuntime(DistributedRuntime):
    def __init__(
        self,
        trainer: Lazy[Trainer],
        episode_generator: Lazy[EpisodeGenerator],
        tokenizer: Tokenizer,
        num_iterations: int,
        num_episodes_per_iteration: Optional[int],
        model: Optional[Lazy[Model]] = None,
        evaluation_vllm_server: Lazy[VLLMServer] = None,
        inference_pipelines: Optional[List[JsonDict]] = None,
        clean_non_model_checkpoints: bool = True,
        evaluate_every_n_checkpoints: int = 1,
        episodes_cloud_log_steps: int = 1,
        early_stop_iteration: Optional[int] = None,
        analyzers: Optional[List[JsonDict]] = None,
        prompt_library: Optional[JsonDict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_iterations = num_iterations
        self.num_episodes_per_iteration = num_episodes_per_iteration

        self.episodes_checkpoint_dir = self.exp_root / "episodes"
        self.episodes_checkpoint_dir.mkdir(exist_ok=True, parents=True)

        self.model = model
        self.model_lazy = model
        self.tokenizer = tokenizer
        self.episode_generator = episode_generator
        self.trainer = trainer

        self.evaluation_vllm_server = evaluation_vllm_server
        self.inference_pipeline_configs = inference_pipelines

        self.clean_non_model_checkpoints = clean_non_model_checkpoints
        self.evaluate_every_n_checkpoints = evaluate_every_n_checkpoints
        self.episodes_cloud_log_steps = episodes_cloud_log_steps
        self.early_stop_iteration = early_stop_iteration

        self.analyzers = analyzers

    def _init_policy_iteration(self, init_model_only: bool = False):
        self._init_episode_generator()

        # We need to specify the device since flash attention models
        # takes a lot of time to load to CPU for some reason.
        # So, we load them on GPU directly.

        if self.model_lazy is not None:
            if (
                self.model._params.get("hf_model_name") == "facebook/opt-125m"
            ):  # todo: milad, this is kind of ugly, but why it is necessary?
                self.model = self.model_lazy.construct(pretrained_args={})
            else:
                self.model = self.model_lazy.construct(
                    device=self.distributed_state.device
                )

        self.trainer = self._construct_trainer(init_model_only)
        assert isinstance(self.trainer, (PolicyTrainer, DeepSpeedPolicyTrainer))

    def _init_episode_generator(self):
        self.episode_generator = self.episode_generator.construct(
            tokenizer=self.tokenizer,
            distributed_state=self.distributed_state,
            num_episodes_per_iteration=self.num_episodes_per_iteration,
            debug=self.debug_mode,
            cloud_logger=self.cloud_logger,
            exp_root=self.exp_root,
            seed=self.global_vars["seed"],
        )
        # Handle the case where we are precomputing episodes from an offline inference result
        if self.episode_generator.can_precompute_episodes:
            episode_cache_len = self._precompute_episodes()
            if self.num_episodes_per_iteration is None:
                logger.info(
                    "Setting num_episodes_per_iteration to episode_cache_len since it's None"
                )
                self.num_episodes_per_iteration = episode_cache_len
            else:
                logger.info(
                    f"Skipping setting num_episodes_per_iteration since "
                    f"it's already set to {self.num_episodes_per_iteration}"
                )

            if self.distributed_state.is_main_process and self.cloud_logger is not None:
                self.cloud_logger.summary["num_episodes_per_iteration"] = (
                    self.num_episodes_per_iteration
                )

    def _construct_trainer(self, init_model_only):
        return self.trainer.construct(
            model=self.model,
            cloud_logger=self.cloud_logger,
            distributed_state=self.distributed_state,
            experiment_root=self.exp_root,
            num_iterations=self.num_iterations,
            num_episodes_per_iteration=self.num_episodes_per_iteration,
            init_model_only=init_model_only,
        )

    def only_generate_episodes(self):
        assert (
            self.distributed_state.is_main_process
        ), "This method should only be called from the main process. "

        self._init_episode_generator()

        starting_iteration = 0
        for iteration in range(starting_iteration, self.num_iterations):
            logger.info(f"Running iteration {iteration}")
            episodes_datasets = self._generate_episodes(iteration)
            logger.info(f"Generated {len(episodes_datasets)} episodes")

            logger.info(f"Finished iteration {iteration}")

    def run_iteration_loop(self, force_rerun: bool = False):
        # Check if final checkpoint exists 存在就直接不用训练了
        final_checkpoint = self.exp_root / "checkpoints" / "final"
        if final_checkpoint.exists() and not force_rerun:
            logger.info("Final checkpoint already exists. Skipping iteration loop.")
            return

        t0 = time.time()
        # Initialize the model, optimizer, etc.
        self._init_policy_iteration()
        self._cloud_log({"timing/total/init_policy_iteration": time.time() - t0})

        # noinspection PyTypeChecker 检查类型
        trainer: Union[PolicyTrainer, DeepSpeedPolicyTrainer] = self.trainer
        if isinstance(self.episode_generator, EpisodeGenerator):
            self.episode_generator.set_trainer(weakref.ref(trainer))

        is_local_main_process = self.distributed_state.is_local_main_process

        latest_policy_path = None
        starting_iteration = 0
        last_checkpoint = trainer.get_last_checkpoint(return_resumable_only=True)
        if not force_rerun and last_checkpoint is not None:
            trainer.load_checkpoint(last_checkpoint)
            starting_iteration = trainer.state.iteration
            latest_policy_path = last_checkpoint.path / "hf_pretrained"
            if self.tokenizer is not None and is_local_main_process:
                # Save the tokenizer to enable seamless loading
                # of the model into vLLM
                self.tokenizer.save_pretrained(latest_policy_path)
            if is_local_main_process:
                logger.info(f"**** Resuming from iteration {starting_iteration} ****")
        # 从保存的checkpoint开始训练
        for iteration in range(starting_iteration, self.num_iterations):
            if (
                self.early_stop_iteration is not None
                and iteration >= self.early_stop_iteration
            ):
                logger.info(
                    f"Early stopping at iteration {iteration} since it's >= {self.early_stop_iteration}"
                )
                break

            if is_local_main_process:
                logger.info("*" * 80)
                logger.info(f"Running iteration {iteration}")
                logger.info("*" * 80)

            t0 = time.time()
            # 生成episode数据
            episodes = self._generate_episodes(
                iteration,
                latest_policy_path=latest_policy_path,
                allow_loading_from_cache=iteration == starting_iteration,
            )
            logger.info(f"Num. Episodes={len(episodes)}")
            self._cloud_log({"timing/total/episode_generation": time.time() - t0})

            assert (
                iteration == trainer.state.iteration
            ), f"{iteration} != {trainer.state.iteration}"

            self.distributed_state.wait_for_everyone()

            t0 = time.time()
            # 训练
            latest_policy_path = trainer.step(episodes)
            self._cloud_log({"timing/total/training_step": time.time() - t0})

            assert (
                iteration + 1 == trainer.state.iteration
            ), f"{iteration+1} != {trainer.state.iteration}"

            if (
                latest_policy_path is not None
                and self.tokenizer is not None
                and is_local_main_process
            ):
                # Save the tokenizer to enable seamless loading
                # of the model into vLLM
                self.tokenizer.save_pretrained(latest_policy_path)

            if is_local_main_process:
                logger.info(f"Finished iteration {iteration}")

        trainer.save_final_checkpoint()

        if (
            need_to_minimize_stored_files()
            and hasattr(trainer, "clean_non_model_checkpoints")
            and is_local_main_process
        ):
            try:
                trainer.clean_non_model_checkpoints()
            except Exception as e:
                logger.error(f"Failed to clean non-model checkpoints due to {e}")

        logger.info("Finished policy iteration loop")

    def run_evaluation_of_gradient_variance(self):
        # Check if it's already done
        if (self.exp_root / "gradient_variance_estimation_done").exists():
            logger.info("Gradient variance estimation is already done. Skipping...")
            return

        trainer_params = getattr(self.trainer, "_params", None)
        if trainer_params is not None:
            training_args = trainer_params.setdefault("training_args", {})
            torch_compile = training_args.get("torch_compile", True)
            if torch_compile:
                logger.warning(
                    f"torch_compile was enabled. Disabling it for gradient variance estimation."
                )
            # This is a hack to disable torch compile for gradient variance estimation
            # since currently enabling it actually slows down the computation.
            # Need to investigate why this is the case.
            training_args["torch_compile"] = False

        self._init_policy_iteration(init_model_only=True)

        from treetune.trainers.mle_trainer import MaximumLikelihoodTrainer

        assert isinstance(
            self.trainer, MaximumLikelihoodTrainer
        ), "Only supported for MLE trainer"

        logger.info(f"Estimating gradient variance for iteration 0")
        episodes_datasets = self._generate_episodes(0)
        logger.info(f"Generated {len(episodes_datasets)} episodes")

        self.distributed_state.wait_for_everyone()
        self.trainer.log_gradient_variance(
            episodes_datasets,
            num_samples=128 * 100,
            store_rolling_aggregates_on_cpu=False,
        )

        (self.exp_root / "gradient_variance_estimation_done").touch()

    def run_evaluation(
        self, force_rerun: bool = False, every_n_checkpoints: Optional[int] = None
    ):
        if isinstance(self, DistributedRuntime):
            assert (
                self.distributed_state.num_processes == 1
            ), "Distributed evaluation is not supported "

        if every_n_checkpoints is None:
            every_n_checkpoints = self.evaluate_every_n_checkpoints

        evaluation_root_dir = self.exp_root / "evaluation"
        evaluation_root_dir.mkdir(exist_ok=True, parents=True)

        checkpoint_dir = self.exp_root / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True, parents=True)

        ckpts = self._get_list_of_evaluation_checkpoints(
            checkpoint_dir, every_n_checkpoints
        )
        logger.info(
            f"Running evaluation on {len(ckpts)} checkpoints "
            f"(since every_n_checkpoints = {every_n_checkpoints})."
        )

        # Iterate over all checkpoints and run all inference pipelines
        for ckpt in ckpts:
            ckpt_global_step = PolicyTrainer.parse_checkpoint_name(ckpt.name)[-1]

            eval_dir = evaluation_root_dir / ckpt.name
            # Check if we have file named "done" in the directory
            if not force_rerun and (eval_dir / "done").exists():
                logger.info(f"Skipping checkpoint {ckpt} because it's already done.")
                continue

            logger.info(f"Running inference on checkpoint {ckpt.name} at \n{ckpt}")
            vllm_ckpt_dir = self._prepare_ckpt_for_vllm(ckpt)
            vllm_server = self.evaluation_vllm_server.construct(
                seed=self.global_vars["seed"]
            )

            logs_dir = evaluation_root_dir / "logs"
            logs_dir.mkdir(exist_ok=True, parents=True)

            vllm_log_file = logs_dir / f"vllm__{vllm_ckpt_dir.name}.log"
            if vllm_log_file.exists():
                vllm_log_file.unlink()
            vllm_log_file.touch()

            logger.info(f"Starting VLLM server with log file {vllm_log_file}")
            server_url = vllm_server.start_server(
                hf_ckpt_path_or_model=vllm_ckpt_dir,
                wait_for_response=True,
                log_path=vllm_log_file,
                timeout=800,
            )
            os.environ["APP_OPENAI_VLLM_API_BASE"] = "none"

            # Run inference on all pipelines
            for pipeline_cfg in self.inference_pipeline_configs:
                pipeline_cfg = copy.deepcopy(pipeline_cfg)
                inference_name = pipeline_cfg["inference_name"]
                logger.info(f"Running inference pipeline `{inference_name}`")

                infer_pipeline_root_dir = eval_dir / inference_name
                infer_pipeline_root_dir.mkdir(exist_ok=True, parents=True)

                infer_pipeline = InferencePipeline.from_params(
                    Params(pipeline_cfg),
                    tokenizer=self.tokenizer,
                    seed=2746318213,
                    api_base_url=server_url,
                    model_name=str(vllm_ckpt_dir),
                    metrics_prefix=f"{ckpt.name}/",
                    enable_cloud_logging_during_inference=False,
                    use_cache=True,
                    cloud_logger=self.cloud_logger,
                    debug_mode=self.debug_mode,
                    exp_root=infer_pipeline_root_dir,
                    checkpoint_global_step=ckpt_global_step,
                )
                results = infer_pipeline.generate()
                infer_pipeline.save_results_to_cloud(results)
                infer_pipeline.analyze(results)

            # Mark the checkpoint as done
            (eval_dir / "done").touch()

            vllm_server.stop_server()

            # Remove vllm checkpoint directory
            shutil.rmtree(vllm_ckpt_dir)

        # Also, run the analyzers if any
        self._run_analyzers(every_n_checkpoints, force_rerun)

        # Mark the evaluation as done only if all checkpoints are done
        # The launcher infrastructure uses this to determine if evaluation is needed to be launched
        is_training_finished = (checkpoint_dir / "final").exists()
        if not is_training_finished:
            logger.info(
                "Skipping marking evaluation as done because training is not finished"
            )
            return

        all_eval_ckpts = self._get_list_of_evaluation_checkpoints(
            checkpoint_dir, every_n_checkpoints, ignore_worker_vars=True
        )
        logger.info(f"All evaluation checkpoints: {all_eval_ckpts}")
        all_ckpts_are_done = all(
            (evaluation_root_dir / ckpt.name / "done").exists()
            for ckpt in all_eval_ckpts
        )
        all_analyzers_are_done = (
            self.exp_root / "evaluation" / "analyzers_done"
        ).exists()
        if all_ckpts_are_done and all_analyzers_are_done:
            (evaluation_root_dir / "done").touch()

    def _run_analyzers(self, every_n_checkpoints: int, force_rerun: bool):
        if self.analyzers is None:
            (self.exp_root / "evaluation" / "analyzers_done").touch()
            return

        analysis_base_dir = self.exp_root / "evaluation" / "analysis"
        analysis_base_dir.mkdir(exist_ok=True, parents=True)
        all_analyzer_dirs = []

        analyzers = copy.deepcopy(self.analyzers)
        for config_obj in analyzers:
            analyzer = Analyzer.from_params(
                Params(config_obj),
                cloud_logger=self.cloud_logger,
                runtime=self,
                plot_prefix="eval/analyzers",
                analysis_base_dir=analysis_base_dir,
                distributed_state=self.distributed_state,
            )
            logger.info(f"Using {analyzer.__class__.__name__}...")

            analyzer.analyze(
                every_n_checkpoints=every_n_checkpoints, force_rerun=force_rerun
            )
            analyzer.flush_local_log()

            all_analyzer_dirs.append(analyzer.get_analysis_root_dir())

        all_analyzers_are_done = all(
            (analyzer_dir / "done").exists() for analyzer_dir in all_analyzer_dirs
        )
        if all_analyzers_are_done:
            (self.exp_root / "evaluation" / "analyzers_done").touch()

    def clean_up_checkpoints(self, keep_every_n_checkpoints: Optional[int] = None):
        """
        Remove all model weights except for every n checkpoints, which we use for evaluation.

        Args:
            keep_every_n_checkpoints:
                Keep every n checkpoints. For example, if this is 5, then we will keep
        """
        if isinstance(self, DistributedRuntime):
            assert self.distributed_state.num_processes == 1

        if keep_every_n_checkpoints is None:
            keep_every_n_checkpoints = self.evaluate_every_n_checkpoints

        if keep_every_n_checkpoints <= 1:
            logger.info(
                "Skipping checkpoint cleaning since keep_every_n_checkpoints == 1"
            )
            return

        checkpoint_dir = self.exp_root / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True, parents=True)

        checkpoints_to_keep = self._get_list_of_evaluation_checkpoints(
            checkpoint_dir, keep_every_n_checkpoints
        )

        logger.info(
            f"Only keeping {len(checkpoints_to_keep)} checkpoints "
            f"(since keep_every_n_checkpoints={keep_every_n_checkpoints})"
        )
        checkpoints_to_keep = checkpoints_to_keep[::keep_every_n_checkpoints]

        for checkpoint in checkpoint_dir.iterdir():
            if checkpoint.is_dir() and checkpoint.name.startswith("ckpt--"):
                if checkpoint in checkpoints_to_keep:
                    continue

                try:
                    logger.info(f"Cleaning checkpoint {checkpoint}")
                    # Remove everything except the `hf_pretrained` folder
                    removed_files_and_dirs = []
                    for file in checkpoint.iterdir():
                        if file.name not in ["hf_pretrained", "pytorch_model.bin"]:
                            removed_files_and_dirs.append(file.name)
                            if file.is_dir():
                                shutil.rmtree(file)
                            else:
                                file.unlink()

                    logger.info(f"Removed files and dirs: {removed_files_and_dirs}")
                except Exception as e:
                    logger.warning(
                        f"Failed to clean checkpoint {checkpoint} due to {e}"
                    )
                    logger.info("Continuing...")

    @staticmethod
    def _get_list_of_evaluation_checkpoints(
        checkpoint_dir: Path, every_n_checkpoints: int, ignore_worker_vars: bool = False
    ) -> List[Path]:
        # Get all items in the directory that are directories
        ckpts = [
            file
            for file in checkpoint_dir.iterdir()
            if file.is_dir() and file.name.startswith("ckpt--")
        ]
        ckpts = sorted(ckpts, key=lambda x: PolicyTrainer.parse_checkpoint_name(x.name))
        if len(ckpts) == 0:
            return []

        last_ckpt = ckpts[-1]

        logger.info(f"Found {len(ckpts)} checkpoints")
        if every_n_checkpoints > 1:
            ckpts = ckpts[::every_n_checkpoints]

        # Make sure last checkpoint is included
        if last_ckpt not in ckpts:
            ckpts.append(last_ckpt)

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

        return ckpts

    def run_evaluation_on_baseline(self, force_rerun: bool = False):
        if isinstance(self, DistributedRuntime):
            assert (
                self.distributed_state.num_processes == 1
            ), "Distributed evaluation is not supported "

        evaluation_root_dir = self.exp_root / "evaluation"
        evaluation_root_dir.mkdir(exist_ok=True, parents=True)

        eval_dir = evaluation_root_dir / "baseline"
        # Check if we have file named "done" in the directory
        if not force_rerun and (eval_dir / "done").exists():
            logger.info(f"Skipping baseline because it's already done.")
            return

        logger.info(f"Running inference on checkpoint baseline")
        vllm_ckpt = self.tokenizer.name_or_path
        vllm_server = self.evaluation_vllm_server.construct(
            seed=self.global_vars["seed"]
        )

        logs_dir = evaluation_root_dir / "logs"
        logs_dir.mkdir(exist_ok=True, parents=True)

        server_url = vllm_server.start_server(
            hf_ckpt_path_or_model=vllm_ckpt,
            wait_for_response=True,
            log_path=logs_dir / f"vllm__baseline.log",
            timeout=800,
        )
        os.environ["APP_OPENAI_VLLM_API_BASE"] = "none"

        # Run inference on all pipelines
        for pipeline_cfg in self.inference_pipeline_configs:
            pipeline_cfg = copy.deepcopy(pipeline_cfg)
            inference_name = pipeline_cfg["inference_name"]
            logger.info(f"Running inference pipeline `{inference_name}`")

            infer_pipeline_root_dir = eval_dir / inference_name
            infer_pipeline_root_dir.mkdir(exist_ok=True, parents=True)

            infer_pipeline = InferencePipeline.from_params(
                Params(pipeline_cfg),
                tokenizer=self.tokenizer,
                seed=self.global_vars["seed"],
                api_base_url=server_url,
                model_name=vllm_ckpt,
                metrics_prefix="baseline/",
                enable_cloud_logging_during_inference=False,
                use_cache=True,
                cloud_logger=self.cloud_logger,
                debug_mode=self.debug_mode,
                exp_root=infer_pipeline_root_dir,
            )
            results = infer_pipeline.generate()
            infer_pipeline.save_results_to_cloud(results)
            infer_pipeline.analyze(results)

        # Mark the checkpoint as done
        (eval_dir / "done").touch()

        vllm_server.stop_server()

    def _precompute_episodes(self) -> int:
        """
        Precomputes episodes and its number and returns it.

        Returns:
            num_episodes_per_iteration:
                Number of episodes per iteration (synchronized across processes)
        """
        # Create a unique cache name for the episode generator
        episode_gen_config = self.config_dict["episode_generator"]
        episode_gen_config_str = json.dumps(episode_gen_config, sort_keys=True)
        cache_name = hashlib.md5(episode_gen_config_str.encode()).hexdigest()
        precomputed_episodes_path = self.episodes_checkpoint_dir / f"{cache_name}.pkl"

        # We only want to precompute/load episodes on the main process
        # as the episode generator is not distributed. Additionally, we
        # need to synchronize the number of episodes across all processes.
        # which we do by broadcasting the number of episodes to all processes.
        with self.distributed_state.main_process_first():
            if self.distributed_state.is_main_process:
                need_to_precompute_episodes = False
                try:
                    # First check if we have already cached the episodes
                    with open(precomputed_episodes_path, "rb") as f:
                        episodes_cache = pickle.load(f)
                    logger.info(
                        f"Loaded precomputed episodes from {precomputed_episodes_path}"
                    )
                    self.episode_generator.episode_cache = episodes_cache
                except FileNotFoundError:
                    need_to_precompute_episodes = True
                except Exception as e:
                    need_to_precompute_episodes = True
                    logger.warning(
                        f"Failed to load precomputed episodes from {precomputed_episodes_path} due to {e}. "
                        "Precomputing episodes again."
                    )

                if need_to_precompute_episodes:
                    self.episode_generator.precompute_episodes()
                    with open(precomputed_episodes_path, "wb") as f:
                        pickle.dump(self.episode_generator.episode_cache, f)

                num_episodes_per_iteration = torch.tensor(
                    len(self.episode_generator.episode_cache),
                    device=self.distributed_state.device,
                )
            else:
                num_episodes_per_iteration = torch.tensor(
                    0, device=self.distributed_state.device
                )
        # Broadcasting the number of episodes to all processes in the distributed environment
        if self.distributed_state.num_processes > 1:
            broadcast(
                num_episodes_per_iteration,
                from_process=self._main_process_index(),
            )
        num_episodes_per_iteration = num_episodes_per_iteration.item()
        logger.info(
            f"Newly computed number of episodes per iteration: {num_episodes_per_iteration}"
        )

        # Updating the episode generator with the number of episodes per iteration
        if self.episode_generator.num_episodes_per_iteration is None:
            self.episode_generator.num_episodes_per_iteration = (
                num_episodes_per_iteration
            )
        else:
            logger.warning(
                f"Skipping setting num_episodes_per_iteration since "
                f"it's already set to {self.episode_generator.num_episodes_per_iteration}"
            )

        return num_episodes_per_iteration

    def _generate_episodes(
        self,
        iteration_id: int,
        latest_policy_path: Optional[Path] = None,
        allow_loading_from_cache: bool = True,
    ) -> Dataset:
        if self.distributed_state.use_distributed:
            self.distributed_state.wait_for_everyone()

        episodes_path = (
            self.episodes_checkpoint_dir / f"episodes_{str(iteration_id).zfill(4)}"
        )

        if allow_loading_from_cache and episodes_path.exists():
            logger.warning(
                f"Episode at {iteration_id} already exist. Loading from {episodes_path}"
            )
            episodes_dataset = Dataset.load_from_disk(str(episodes_path))
            self._log_some_examples(episodes_dataset, iteration_id)
            return episodes_dataset

        is_main_process = self.distributed_state.is_main_process

        if is_main_process:
            logger.info("-" * 80)
            logger.info(
                f"Episode at {iteration_id} does not exist. Generating episodes..."
            )
            logger.info("-" * 80)

        def remove_null_columns(ds: Dataset):
            null_columns = []
            for k, v in ds.features.items():
                if v.dtype == "null":
                    null_columns.append(k)
            return ds.remove_columns(null_columns)

        # If episode_generator supports distributed, generate in all processes
        if self.episode_generator.support_distributed:
            episodes = self.episode_generator.generate(
                iteration=iteration_id, latest_policy_path=latest_policy_path
            )
            assert isinstance(episodes, Dataset)
            if is_main_process:
                remove_null_columns(episodes).save_to_disk(episodes_path)

        # If it does not support distributed, only generate in the main process
        elif is_main_process:
            episodes = self.episode_generator.generate()
            if not isinstance(episodes, Dataset):
                episodes = Dataset.from_dict(
                    {
                        k: [getattr(e, k) for e in episodes]
                        for k in episodes[0].__dict__.keys()
                    }
                )
            remove_null_columns(episodes).save_to_disk(episodes_path)

        # Wait for episodes to be generated
        self.distributed_state.wait_for_everyone()

        assert episodes_path.exists(), (
            f"Episodes path {episodes_path} does "
            f"not exist from {self.distributed_state.process_index} perspective"
        )
        episodes_dataset = Dataset.load_from_disk(str(episodes_path))

        self._log_some_examples(episodes_dataset, iteration_id)
        # from treetune.common.wandb_utils import save_inference_result_to_cloud

        # save_inference_result_to_cloud(
        #     episodes_dataset,
        #     f"episodes_{iteration_id:04d}",
        #     self.cloud_logger if is_main_process else None,
        # )
        # self.distributed_state.wait_for_everyone()

        return episodes_dataset

    def _main_process_index(self) -> int:
        return (
            0
            if self.distributed_state.distributed_type != DistributedType.MEGATRON_LM
            else (self.distributed_state.num_processes - 1)
        )

    def _log_some_examples(
        self,
        episodes: Union[List[Any], Dataset],
        iteration_idx: int,
        num_examples: int = 3,
    ):
        if not self.distributed_state.is_main_process:
            return

        self.episode_generator.log_episodes(
            episodes,
            iteration_idx,
            num_examples=num_examples,
            seed=self.global_vars["seed"],
            log_to_cloud=iteration_idx % self.episodes_cloud_log_steps == 0,
        )

    def _prepare_ckpt_for_vllm(self, ckpt_dir: Path) -> Path:
        """Prepare the checkpoint directory for evaluation."""

        # Use current working directory to create temporary ckpt path
        output_dir = Path.cwd() / f"tmp_ckpt__{ckpt_dir.name}"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Save tokenizer
        if not (ckpt_dir / "hf_pretrained").exists():
            self.tokenizer.save_pretrained(output_dir)

        # Special case of handling LoRA checkpoints
        if (
            self.model is not None
            and len(self.model._params.get("lora_config", {})) > 0
        ):
            # We first instantiate the model, and load the state dict into it
            # This is more of a hack. The proper way to do this is to use
            # from_pretrained, but that doesn't seem to work for some reason.
            lora_model = self.model_lazy.construct()
            ckpt_state_dict = torch.load(
                ckpt_dir / "pytorch_model.bin", map_location="cpu"
            )
            lora_model.load_state_dict(ckpt_state_dict)
            model = lora_model.merge_and_unload()
            model.save_pretrained(output_dir, safe_serialization=False)
            return output_dir

        # Check if it already has hf_pretrained directory
        hf_pretrained_dir = ckpt_dir / "hf_pretrained"
        if hf_pretrained_dir.exists() and hf_pretrained_dir.is_dir():
            # Check lora model
            lora_model_dir = hf_pretrained_dir / "lora"
            if lora_model_dir.exists() and lora_model_dir.is_dir():
                raise NotImplementedError("Should not happen")
                # from peft import AutoPeftModelForCausalLM
                #
                # model = AutoPeftModelForCausalLM.from_pretrained(lora_model_dir)
                # model = model.merge_and_unload()
                # model.save_pretrained(output_dir)
            else:
                # Link the files in the hf_pretrained directory to the output directory
                for file in hf_pretrained_dir.iterdir():
                    (output_dir / file.name).symlink_to(file.absolute())

            if not (output_dir / "config.json").exists():
                config = AutoConfig.from_pretrained(self.model._params["hf_model_name"])
                config.save_pretrained(output_dir)

            hf_tokenizer_files = [f for f in output_dir.glob("tokenizer*")]
            if len(hf_tokenizer_files) == 0:
                self.tokenizer.save_pretrained(output_dir)

            return output_dir

        # Check if it's a deespseed checkpoint
        pytorch_model_dir = ckpt_dir / "pytorch_model"
        if pytorch_model_dir.exists() and pytorch_model_dir.is_dir():
            # We need to use `zero_to_fp32.py` to convert the checkpoint
            # to a format that can be loaded by vLLM
            logger.info("Converting DeepSpeed checkpoint to vLLM checkpoint")
            command = f"python {get_zero_to_fp32_script_path()} {ckpt_dir} {output_dir}/pytorch_model.bin"

            # Run the command using subprocess
            try:
                subprocess.check_call(shlex.split(command))
            except subprocess.CalledProcessError as e:
                logger.error(f"Error converting checkpoint: {e}")
                logger.error(f"stdout: {e.stdout}")
                logger.error(f"stderr: {e.stderr}")
                raise e

            config = AutoConfig.from_pretrained(self.model._params["hf_model_name"])
            config.torch_dtype = torch.float32
            config.save_pretrained(output_dir)

            # Return the path to the converted checkpoint
            return output_dir

        # Otherwise, it's a normal checkpoint
        pytorch_model_files = ckpt_dir.glob("pytorch_model*.bin")
        pytorch_model_files = list(pytorch_model_files)
        assert (
            len(pytorch_model_files) >= 1
        ), "No pytorch_model.bin found in checkpoint directory"

        # Copy the pytorch_model.bin file to the output directory
        output_dir.mkdir(exist_ok=True, parents=True)
        for file in pytorch_model_files:
            shutil.copy(file, output_dir)

        config = AutoConfig.from_pretrained(self.model._params["hf_model_name"])
        config.save_pretrained(output_dir)

        # Return the path to the converted checkpoint
        return output_dir
