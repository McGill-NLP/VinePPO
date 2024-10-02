# PolicyIterationRuntime does everything we need to train value networks
# it is just that the name is not appropriate as we're not training a policy
# hence, this empty class
from typing import Optional

from treetune.runtime import Runtime, PolicyIterationRuntime
from treetune.logging_utils import get_logger
from treetune.runtime.base_runtime import DistributedRuntime

logger = get_logger(__name__)


@Runtime.register("value_iteration")
class ValueIterationRuntime(PolicyIterationRuntime):
    pass

    def run_evaluation(
        self, force_rerun: bool = False, every_n_checkpoints: Optional[int] = None
    ):
        if isinstance(self, DistributedRuntime):
            assert (
                self.distributed_state.num_processes == 1
            ), "Distributed evaluation is not supported "

        if every_n_checkpoints is None:
            every_n_checkpoints = self.evaluate_every_n_checkpoints

        # Run the analyzers
        self._run_analyzers(every_n_checkpoints, force_rerun)

        checkpoint_dir = self.exp_root / "checkpoints"
        is_training_finished = (checkpoint_dir / "final").exists()
        if not is_training_finished:
            logger.info(
                "Skipping marking evaluation as done because training is not finished"
            )
            return

        evaluation_root_dir = self.exp_root / "evaluation"
        evaluation_root_dir.mkdir(exist_ok=True, parents=True)

        all_analyzers_are_done = (
            self.exp_root / "evaluation" / "analyzers_done"
        ).exists()
        if all_analyzers_are_done:
            (evaluation_root_dir / "done").touch()
