import copy
import json
from pathlib import Path
from typing import Optional

from accelerate import PartialState
from wandb.sdk.wandb_run import Run

from treetune import logging_utils
from treetune.common import Registrable

logger = logging_utils.get_logger(__name__)


class Analyzer(Registrable):
    def __init__(
        self,
        cloud_logger: Run,
        runtime,
        metrics_prefix: str = "",
        global_step: Optional[int] = None,
        plot_prefix: Optional[str] = None,
        analysis_base_dir: Optional[Path] = None,
        distributed_state: Optional[PartialState] = None,
    ):
        from treetune.runtime.base_runtime import BaseRuntime

        self.cloud_logger = cloud_logger
        self.runtime: BaseRuntime = runtime
        self.metrics_prefix = metrics_prefix
        self.global_step = global_step
        self.plot_prefix = plot_prefix
        self.analysis_base_dir = analysis_base_dir
        self.distributed_state = distributed_state
        self._local_log_obj = {}

        self.task = getattr(self.runtime, "task", None)

    def get_analysis_id(self) -> str:
        return self.metrics_prefix

    def analyze(self, *args, **kwargs):
        if (self.get_analysis_root_dir()).exists():
            logger.warning(
                f"Analysis directory {self.get_analysis_root_dir()} already exists."
            )

    def log(self, obj):
        self._local_log_obj.update(obj)

    def get_analysis_root_dir(self) -> Path:
        analysis_id = self.get_analysis_id()
        base_dir = self.analysis_base_dir or self.runtime.exp_root / "analysis"
        analysis_root = base_dir / self.__class__.__name__
        if analysis_id is not None and len(analysis_id) > 0:
            analysis_root /= analysis_id
        return analysis_root

    def log_metrics(self, metrics):
        # Log the metrics to the console
        logger.info(f"Metrics: {metrics}")

        # Log the metrics to the local file
        self.log(metrics)

        # Log the metrics to the cloud
        if hasattr(self, "cloud_logger") and self.cloud_logger is not None:
            if self.global_step is not None and self.plot_prefix is not None:
                plot_metrics = copy.deepcopy(metrics)
                plot_metrics = {
                    f"{self.plot_prefix}/{k}": v for k, v in plot_metrics.items()
                }
                self.cloud_logger.log(
                    {**plot_metrics, "train/global_step": self.global_step}
                )

            # Append the analysis ID to the metrics
            prefix = f"analysis/{self.__class__.__name__}/"
            analysis_id = self.get_analysis_id()
            if analysis_id is not None and len(analysis_id) > 0:
                prefix += analysis_id + "/"
            metrics = {prefix + k: v for k, v in metrics.items()}
            self.cloud_logger.summary.update(metrics)

    def flush_local_log(self):
        analysis_root = self.get_analysis_root_dir()
        analysis_root.mkdir(parents=True, exist_ok=True)

        _local_log_obj = self._local_log_obj
        if (analysis_root / "log.json").exists():
            logger.warning(
                f"Analysis file {analysis_root}/log.json already exists. "
                "Updating the existing log file."
            )

            # Update the existing log file
            with (analysis_root / "log.json").open("r") as f:
                existing_log_obj = json.load(f)
            existing_log_obj.update(_local_log_obj)
            _local_log_obj = existing_log_obj

        if len(_local_log_obj) == 0:
            # Nothing to log
            return

        with (analysis_root / "log.json").open("w") as f:
            json.dump(_local_log_obj, f, indent=4, sort_keys=True)

        # Save the log file to the cloud
        name = "__".join(["analysis", self.__class__.__name__, self.get_analysis_id()])
        name = name.replace("/", "__")
        log_file_copy = analysis_root / f"log_{name}.json"
        with log_file_copy.open("w") as f:
            json.dump(_local_log_obj, f, indent=4, sort_keys=True)

        if self.cloud_logger is not None:
            self.cloud_logger.save(str(log_file_copy.absolute()), policy="now")
