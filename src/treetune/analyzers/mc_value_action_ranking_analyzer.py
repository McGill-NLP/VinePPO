import copy
import os
import random
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple

from datasets import Dataset

from treetune import logging_utils
from treetune.analyzers.action_ranking_analyzer import ActionRankingAnalyzer
from treetune.analyzers.analyzer import Analyzer
from treetune.trainers.policy_trainer import PolicyTrainer

logger = logging_utils.get_logger(__name__)


@Analyzer.register("mc_value_action_ranking")
class MCValueActionRankingAnalyzer(ActionRankingAnalyzer):
    # noinspection DuplicatedCode
    def __init__(self, num_mc_rollouts: int, **kwargs):
        super().__init__(**kwargs)
        self.num_mc_rollouts = num_mc_rollouts

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
            if episode_dir.is_dir() and (episode_dir / "w_actLogp").exists()
        ]

        iter_to_episode_dir = {}
        for episode_dir in available_episodes:
            iter_idx = int(episode_dir.name.split("__iter")[-1])
            iter_to_episode_dir[iter_idx] = episode_dir / "w_actLogp"

        def can_analyze(ckpt: Path) -> bool:
            if not (ckpt / "hf_pretrained" / "pytorch_model.bin").exists():
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
        rng = random.Random(42)

        out = []
        for req in state_action_val_requests:
            req = copy.deepcopy(req)
            state = states[req["state_idx"]]
            action_idx = req["action_idx"]
            gt_mc_returns = state["next_action_gt_mc_returns"][action_idx]
            assert gt_mc_returns is not None
            mc_values = rng.sample(gt_mc_returns, self.num_mc_rollouts)
            pred_value = sum(mc_values) / len(mc_values)
            req["pred_value"] = pred_value
            out.append(req)

        return out
