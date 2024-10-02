import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Dict, Optional, Any, List

import torch.cuda

from treetune.common import Registrable
from treetune.common import (
    gpu_utils,
    py_utils,
)
from treetune import logging_utils
from treetune.common.py_utils import (
    is_world_process_zero,
    can_upload_files_to_cloud,
)

logger = logging_utils.get_logger(__name__)


class Runtime(Registrable):
    pass


@Runtime.register("base")
class BaseRuntime(Runtime):
    def __init__(
        self,
        exp_name: str,
        directory: str = "experiments",
        project_name: Optional[str] = None,
        global_vars: Optional[Dict[str, Any]] = None,
        config_dict: Optional[Dict[str, Any]] = None,
    ):
        self.exp_name = exp_name
        self.project_name = project_name

        # Make sure that the experiment directory exists.
        exp_root = Path(directory) / self.exp_name
        exp_root.mkdir(parents=True, exist_ok=True)
        self.exp_root = exp_root

        # Make sure that the logs directory exists.
        logs_dir = self.exp_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = logs_dir

        self.global_vars = global_vars or {"seed": 42}
        self.debug_mode = self.global_vars.get("debug_mode", False)
        self.config_dict = config_dict

        seed = self.global_vars["seed"]
        logger.info(f"Setting seed = {seed}")
        py_utils.set_seed_for_all(seed)

        if self.debug_mode:
            logger.info(">>>>>>>>>>>>>>>>> DEBUG MODE <<<<<<<<<<<<<<<<<<<")

        if is_world_process_zero():
            cloud_logger = self._create_cloud_logger()
            if cloud_logger is not None:
                from wandb.sdk.wandb_run import Run

                self.cloud_logger: Run = self._create_cloud_logger()
        else:
            self.cloud_logger = None

        self._write_meta_data()

    def _write_meta_data(self):
        gpu_info = gpu_utils.get_cuda_info()
        if len(gpu_info) != 0:
            # log_obj = {f"gpus_info/#{i}/": gi for i, gi in enumerate(gpu_info)}
            # self.logger.summary.update(log_obj)

            logger.info(f"GPUs Info: \n{json.dumps(gpu_info, indent=4)}")

        metadata = {"exp_name": self.exp_name, "gpus_info": gpu_info}
        with open(self.exp_root / "metadata.json", "w") as f:
            f.write(json.dumps(metadata, indent=4, sort_keys=True))

        conf_path = self.exp_root / "config.json"
        with conf_path.open("w") as f:
            f.write(json.dumps(self.config_dict, indent=4, sort_keys=True))

        dotenv_path = self.exp_root / "dotenv.txt"
        with dotenv_path.open("w") as f:
            for k, v in os.environ.items():
                if k.startswith("APP_"):
                    f.write(f"{k}={v}\n")
                    logger.info(f"Application ENV: {k}={v}")

        if not can_upload_files_to_cloud():
            return

        if conf_path.exists():
            self.cloud_logger.save(str(conf_path.absolute()), policy="now")
        self.cloud_logger.save(str(dotenv_path.absolute()), policy="now")

    def _create_cloud_logger(self):
        try:
            import wandb
        except ImportError:
            logger.warning(
                "Wandb is not installed. Please install it using `pip install wandb`"
            )
            return None

        if wandb.run is None:
            if self.debug_mode:
                mode = "disabled"
            else:
                mode = None

            wandb_entity = self.global_vars.get("wandb_entity", None)
            # Check if the entity is set in the environment variable.
            if "WANDB_ENTITY" in os.environ and wandb_entity is not None:
                logger.warning(
                    f"WANDB_ENTITY is set in the environment variable ({os.environ['WANDB_ENTITY']}), "
                    f"but it is also set in the config file ({wandb_entity}). "
                    "The value in the config file will be used."
                )

            settings = wandb.Settings()
            settings.update(
                _save_requirements=True,
                _disable_meta=False,
            )
            wandb.init(
                config=self.config_dict,
                project=self.project_name,
                name=self.exp_name,
                resume="allow",
                mode=mode,
                force=True,
                entity=wandb_entity,
            )

        return wandb.run

    def _log_method_call_to_console(
        self,
        target_logger,
        method_name: str,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
    ):
        if target_logger is None:
            target_logger = logger

        target_logger.info("*" * 80)
        target_logger.info(f"* {method_name} called")
        if args is not None:
            target_logger.info(f"* \targs: {args}")
        if kwargs is not None:
            target_logger.info(f"* \tkwargs: {kwargs}")
        target_logger.info("*" * 80)

    def hello_world(self):
        logger.info("Hello World!")
        logger.info(f"Debug Mode: {self.debug_mode}")
        logger.info(f"Cloud Logger enabled: {self.cloud_logger is not None}")
        logger.info(f"Experiment Directory: {self.exp_root}")


class DistributedRuntime(BaseRuntime):
    def __init__(
        self,
        exp_name: str,
        use_deepspeed: bool = False,
        directory: str = "experiments",
        project_name: Optional[str] = None,
        global_vars: Optional[Dict[str, Any]] = None,
        config_dict: Optional[Dict[str, Any]] = None,
    ):
        super(BaseRuntime, self).__init__()
        self.use_deepspeed = use_deepspeed
        self._initialize_distributed_setup()

        self.exp_name = exp_name
        self.project_name = project_name

        # Make sure that the experiment directory exists.
        exp_root = Path(directory) / self.exp_name
        if self.distributed_state.is_local_main_process:
            exp_root.mkdir(parents=True, exist_ok=True)
        self.exp_root = exp_root

        # Make sure that the logs directory exists.
        logs_dir = self.exp_root / "logs"
        if self.distributed_state.is_local_main_process:
            logs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = logs_dir

        self.global_vars = global_vars or {"seed": 42}
        self.debug_mode = self.global_vars.get("debug_mode", False)
        self.config_dict = config_dict

        seed = self.global_vars["seed"]
        logger.info(f"Setting seed = {seed}")
        py_utils.set_seed_for_all(seed)

        if self.debug_mode:
            logger.info(">>>>>>>>>>>>>>>>> DEBUG MODE <<<<<<<<<<<<<<<<<<<")

        if self.distributed_state.is_main_process:
            cloud_logger = self._create_cloud_logger()
            if cloud_logger is not None:
                from wandb.sdk.wandb_run import Run

                self.cloud_logger: Run = self._create_cloud_logger()

            self._write_meta_data()
        else:
            self.cloud_logger = None

        if self.distributed_state.use_distributed:
            self.distributed_state.wait_for_everyone()

    def _initialize_distributed_setup(self):
        from accelerate import PartialState

        use_cpu = not torch.cuda.is_available()
        ddp_timeout = 10000

        if self.use_deepspeed:
            os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
            self.distributed_state = PartialState(
                timeout=timedelta(seconds=ddp_timeout)
            )
            del os.environ["ACCELERATE_USE_DEEPSPEED"]
        else:
            kwargs = {"timeout": timedelta(seconds=ddp_timeout)}
            if not use_cpu:
                kwargs["backend"] = "nccl"
            self.distributed_state = PartialState(use_cpu, **kwargs)

    def _create_cloud_logger(self):
        if not self.distributed_state.is_main_process:
            return None
        return super()._create_cloud_logger()

    def _cloud_log(self, *args, **kwargs):
        if self.distributed_state.is_main_process and self.cloud_logger is not None:
            self.cloud_logger.log(*args, **kwargs)
