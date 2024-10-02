import json

import torch
from tqdm import tqdm
from transformers import PreTrainedModel
from wandb.sdk.wandb_run import Run

from treetune import logging_utils
from treetune.analyzers.analyzer import Analyzer
from treetune.trainers.policy_trainer import PolicyTrainer

logger = logging_utils.get_logger(__name__)


@Analyzer.register("weight_l2_distance")
class WeightL2DistanceAnalyzer(Analyzer):
    def __init__(self, cloud_logger: Run, runtime, **kwargs):
        from treetune.runtime.policy_iteration_runtime import PolicyIterationRuntime

        assert isinstance(runtime, PolicyIterationRuntime)
        self.runtime: PolicyIterationRuntime = runtime
        super().__init__(cloud_logger, runtime, **kwargs)

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

        checkpoint_dir = self.runtime.exp_root / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True, parents=True)

        ckpts = self.runtime._get_list_of_evaluation_checkpoints(
            checkpoint_dir, every_n_checkpoints
        )
        ckpts = [
            ckpt
            for ckpt in ckpts
            if (ckpt / "hf_pretrained" / "pytorch_model.bin").exists()
        ]
        logger.info(f"Computing L2 distance on {len(ckpts)} checkpoints: {ckpts}")

        from transformers import PreTrainedModel

        # Lazy loading of the model. This is to avoid loading the model multiple times.
        # and skip the loading if the checkpoint has already been analyzed.
        _initial_model = None
        _loadable_model = None

        def get_model(loadable: bool = False) -> PreTrainedModel:
            nonlocal _initial_model
            nonlocal _loadable_model
            if loadable:
                if _loadable_model is None:
                    _loadable_model = self.runtime.model_lazy.construct(
                        pretrained_args={"device": torch.device("cpu")}
                    )
                return _loadable_model
            else:
                if _initial_model is None:
                    _initial_model = self.runtime.model_lazy.construct(
                        pretrained_args={"device": torch.device("cpu")}
                    )
                return _initial_model

        metrics = {}
        for ckpt in tqdm(ckpts, desc="Computing L2 distance from the initial models"):
            result_file = analysis_root / f"{ckpt.name}.json"
            if result_file.exists():
                l2_distance = json.loads(result_file.read_text())["l2_distance"]
                metrics[ckpt.name] = l2_distance
                logger.info(f"Skipping {ckpt} as it has already been analyzed.")
                continue

            initial_model = get_model()
            loadable_model = get_model(loadable=True)
            loadable_model.load_state_dict(
                torch.load(ckpt / "hf_pretrained" / "pytorch_model.bin")
            )
            loadable_model.eval()

            l2_distance = self._compute_l2_distance(initial_model, loadable_model)
            metrics[ckpt.name] = l2_distance

            with result_file.open("w") as f:
                json.dump({"l2_distance": l2_distance}, f)

        for ckpt_name, l2_dist in metrics.items():
            global_step = PolicyTrainer.parse_checkpoint_name(ckpt_name)[-1]
            if self.cloud_logger is not None and self.plot_prefix is not None:
                self.cloud_logger.log(
                    {
                        f"{self.plot_prefix}/{self.__class__.__name__}": l2_dist,
                        "train/global_step": global_step,
                    }
                )

        self.log_metrics(metrics)

        del _initial_model
        del _loadable_model

        import gc
        gc.collect()
        torch.cuda.empty_cache()

        all_ckpts = self.runtime._get_list_of_evaluation_checkpoints(
            checkpoint_dir, every_n_checkpoints, ignore_worker_vars=True
        )
        all_ckpts = [ckpt for ckpt in all_ckpts if (ckpt / "hf_pretrained").exists()]
        all_ckpts_are_done = all(
            (analysis_root / f"{ckpt.name}.json").exists() for ckpt in all_ckpts
        )
        if all_ckpts_are_done:
            (analysis_root / "done").touch()

    def _compute_l2_distance(
        self, model1: PreTrainedModel, model2: PreTrainedModel
    ) -> float:
        l2_distance = torch.tensor(0.0, device=model1.device, dtype=torch.float32)
        with torch.no_grad():
            for (name1, param1), (name2, param2) in zip(
                model1.named_parameters(), model2.named_parameters()
            ):
                assert name1 == name2
                assert param1.shape == param2.shape
                diff = param1.float() - param2.float()
                l2_distance += torch.sum(diff * diff)

        l2_distance = torch.sqrt(l2_distance).detach().clone().cpu().item()
        return l2_distance
