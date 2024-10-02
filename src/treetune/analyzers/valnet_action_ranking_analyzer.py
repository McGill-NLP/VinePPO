import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple

from datasets import Dataset
from deepspeed import DeepSpeedEngine
from transformers import PreTrainedModel

from treetune import logging_utils
from treetune.analyzers.action_ranking_analyzer import ActionRankingAnalyzer
from treetune.analyzers.analyzer import Analyzer
from treetune.common import Lazy
from treetune.trainers.policy_trainer import PolicyTrainer
from treetune.trainers.ppo_trainer import PPOTrainer

logger = logging_utils.get_logger(__name__)


@Analyzer.register("valnet_action_ranking")
class ValNetActionRankingAnalyzer(ActionRankingAnalyzer):
    def __init__(
        self,
        critic_deepspeed_config: Dict[str, Any],
        per_device_batch_size: Optional[int] = None,
        append_bos_to_query: Union[str, bool] = "auto",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.append_bos_to_query = append_bos_to_query
        trainer = getattr(self.runtime, "trainer", None)
        if trainer is None or isinstance(trainer, Lazy):
            # noinspection PyProtectedMember
            trainer = self.runtime._construct_trainer(init_model_only=False)
        assert isinstance(trainer, PPOTrainer)

        self.trainer: PPOTrainer = trainer
        self.trainer.cache_deepspeed_engines = False
        self.trainer.critic_deepspeed_config = critic_deepspeed_config

        self.per_device_batch_size = per_device_batch_size

        per_device_batch_size = self.per_device_batch_size
        if self.per_device_batch_size is None:
            per_device_batch_size = self.trainer.args.per_device_train_batch_size
        if critic_deepspeed_config.get("train_batch_size", None) == "auto":
            critic_deepspeed_config["train_batch_size"] = per_device_batch_size
        self.trainer.args.per_device_train_batch_size = per_device_batch_size

    # noinspection DuplicatedCode
    def _get_list_of_evaluation_checkpoints(
        self, every_n_checkpoints: int, ignore_worker_vars: bool = False
    ) -> List[Tuple[Path, Path]]:
        checkpoint_dir = self.runtime.exp_root / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True, parents=True)

        # noinspection PyProtectedMember
        ckpts = self.runtime._get_list_of_evaluation_checkpoints(
            checkpoint_dir, every_n_checkpoints, ignore_worker_vars=True
        )

        available_episodes = [
            episode_dir
            for episode_dir in checkpoint_dir.glob("episodes__iter*")
            if episode_dir.is_dir() and (episode_dir / "w_actLogp_and_values").exists()
        ]

        iter_to_episode_dir = {}
        for episode_dir in available_episodes:
            iter_idx = int(episode_dir.name.split("__iter")[-1])
            iter_to_episode_dir[iter_idx] = episode_dir / "w_actLogp_and_values"

        def can_analyze(ckpt: Path) -> bool:
            if not (ckpt / "hf_pretrained" / "pytorch_model.bin").exists():
                return False
            if not (ckpt / "critic" / "hf_pretrained" / "pytorch_model.bin").exists():
                return False
            ckpt_iter = int(PolicyTrainer.parse_checkpoint_name(ckpt.name)[0])
            return ckpt_iter in iter_to_episode_dir

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

        episode_dirs = [
            iter_to_episode_dir[int(PolicyTrainer.parse_checkpoint_name(ckpt.name)[0])]
            for ckpt in ckpts
        ]

        return list(zip(ckpts, episode_dirs))

    def _compute_pred_values(
        self,
        state_action_val_requests: Union[List[Dict[str, Any]], Dataset],
        ckpt: Path,
        states: List[Dict[str, Any]],
    ) -> Union[List[Dict[str, Any]], Dataset]:
        del states  # Make sure we don't accidentally use it

        tokenizer = self.tokenizer
        append_bos = self._should_append_bos_to_query()

        dataset = []
        for req in state_action_val_requests:
            query = req["query"]
            query_token_ids = tokenizer(query, add_special_tokens=False)["input_ids"]
            if append_bos:
                query_token_ids = [tokenizer.bos_token_id] + query_token_ids
            query_token_ids, response_token_ids = (
                # The critic computes the values for all tokens.
                # So, it doesn't matter how we split the query and response.
                query_token_ids[:-1],
                query_token_ids[-1:],
            )
            dataset.append(
                {
                    "query_token_ids": query_token_ids,
                    "response_token_ids": response_token_ids,
                }
            )
        dataset = Dataset.from_list(dataset)

        critic = self._init_critic(ckpt)

        # noinspection PyProtectedMember
        dataset_with_values = self.trainer._update_episodes_with_values(
            critic, dataset, "critic_values"
        )

        # noinspection PyProtectedMember
        self.trainer._destroy_ds_engine(critic)

        # `_update_episodes_with_values` returns a values for all tokens.
        # We only need the value of the last token.
        def get_last_value(example: Dict["str", Any]) -> Dict["str", Any]:
            return {
                "pred_value": example["critic_values"][-1],
            }

        dataset_with_values = dataset_with_values.map(get_last_value)
        dataset_with_values = dataset_with_values.remove_columns(["critic_values"])

        # Add "pred_value" to the original requests
        critic_values = dataset_with_values["pred_value"]
        if isinstance(state_action_val_requests, list):
            assert len(state_action_val_requests) == len(critic_values)
            for req, value in zip(state_action_val_requests, critic_values):
                req["pred_value"] = value
        else:
            state_action_val_requests = state_action_val_requests.add_column(
                "pred_value", critic_values
            )

        return state_action_val_requests

    # noinspection PyProtectedMember
    def _init_critic(self, ckpt: Path) -> Union[DeepSpeedEngine, PreTrainedModel]:
        # Patch the trainer to construct the model from the checkpoint
        hf_ckpt_path = ckpt / "critic" / "hf_pretrained"
        critic = self.trainer._init_critic_model(hf_checkpoint_path=hf_ckpt_path)
        critic.eval()

        return critic

    def _should_append_bos_to_query(self) -> bool:
        """
        Determine whether to append BOS to the query based on the tokenizer
        """
        if self.append_bos_to_query != "auto":
            return self.append_bos_to_query

        if "llama" in self.tokenizer.name_or_path.lower():
            assert self.tokenizer.bos_token_id is not None
            return True
        else:
            raise ValueError(
                f"Cannot automatically determine whether to append BOS for tokenizer {self.tokenizer.name_or_path}"
            )
