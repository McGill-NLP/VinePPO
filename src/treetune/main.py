import json
import os
from typing import Dict, Any, List
from pathlib import Path
import sys

import _jsonnet
import fire


# Set PYTHONPATH to src/ directory.
# This is needed to make sure that the imports below work correctly.
source_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(source_dir))

INITIAL_SEED = 42


class EntryPoint(object):
    _runtime = None
    _config = None

    def __init__(self, configs: str, debug_mode: bool = None):
        # Lazy import to avoid long startup time.
        from treetune import logging_utils

        logger = logging_utils.get_logger()

        from treetune.common import py_utils
        from treetune.common import Params

        # configs format: "path/to/config1.jsonnet,path/to/config2.jsonnet"
        config_paths = [f.strip() for f in configs.split(",")]

        # Build the config object from the jsonnet files.
        config = self._load_config_obj(config_paths)

        # If the config doesn't specify a seed, use the default.
        if "global_vars" not in config or "seed" not in config["global_vars"]:
            if "global_vars" not in config:
                config["global_vars"] = dict()
            config["global_vars"]["seed"] = INITIAL_SEED
            logger.info(
                f"Seed was not specified in the config. Setting to {INITIAL_SEED}."
            )

        # If debug_mode is set, we make sure that it is set in the config object as well.
        if debug_mode is not None:
            if "global_vars" not in config:
                config["global_vars"] = dict()
            config["global_vars"]["debug_mode"] = debug_mode

        # Set a unique run name if not provided in the environment variables.
        config["exp_name"] = os.environ.get(
            "APP_EXPERIMENT_NAME", py_utils.unique_run_name_from_configs(config_paths)
        )

        config_str = json.dumps(config, indent=4, sort_keys=True)
        logger.info(f"Config files: {config_paths}")
        logger.info(f"----Config----\n{config_str}\n--------------")

        # Create the runtime object.
        from treetune.runtime import Runtime

        config = self._patch_config_obj_for_di(config)
        self._config = config
        self._runtime = Runtime.from_params(Params({"config_dict": config, **config}))

    def _patch_config_obj_for_di(self, config):
        if "runtime_type" in config:
            config["type"] = config["runtime_type"]
            del config["runtime_type"]
        return config

    def _load_config_obj(self, config_paths: List[str]) -> Dict[str, Any]:
        # If there are any environment variables that start with APP_, we will pass them
        # to jsonnet as external variables.  This allows overriding variables in the
        # config files with shell environment variables.
        ext_vars = {k: v for k, v in os.environ.items() if k.startswith("APP_")}

        # Make sure the random seed is set, even if it's not in the config file.
        seed = os.environ.get("APP_SEED", str(INITIAL_SEED))
        if not seed.isnumeric():
            seed = str(INITIAL_SEED)
        ext_vars["APP_SEED"] = seed

        # Construct the root jsonnet file, which imports all of the config files
        # specified on the command line.
        jsonnet_str = "+".join([f'(import "{f}")' for f in config_paths])
        json_str = _jsonnet.evaluate_snippet("snippet", jsonnet_str, ext_vars=ext_vars)
        config: Dict[str, Any] = json.loads(json_str)

        # Override the root directory, if an environment variable is set.
        orig_directory = config.get("directory", "experiments")
        config["directory"] = os.environ.get("APP_DIRECTORY", orig_directory)

        return config

    def __getattr__(self, attr):
        if attr in self.__class__.__dict__:
            return getattr(self, attr)
        else:
            return getattr(self._runtime, attr)

    def __dir__(self):
        return sorted(set(super().__dir__() + self._runtime.__dir__()))


if __name__ == "__main__":
    fire.Fire(EntryPoint)
