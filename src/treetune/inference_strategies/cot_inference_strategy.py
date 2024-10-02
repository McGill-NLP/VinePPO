import json

from treetune import logging_utils
from treetune.common import Lazy
from treetune.inference_strategies.base_inference_strategy import InferenceStrategy
from treetune.inference_strategies.tree_inference.branch_factor_strategy import (
    ListBranchFactor,
)
from treetune.inference_strategies.tree_inference.expansion import NodeExpander
from treetune.inference_strategies.tree_inference_strategy import TreeInferenceStrategy

logger = logging_utils.get_logger(__name__)


@InferenceStrategy.register("cot", exist_ok=True)
class COTInferenceStrategy(TreeInferenceStrategy):
    def __init__(
        self,
        samples: int,
        node_expander: Lazy[NodeExpander],
        **kwargs,
    ):
        if "branch_factor_strategy" in node_expander._params:
            old_strategy = node_expander._params["branch_factor_strategy"]
            logger.warning(
                "The old branch_factor_strategy is overwritten by COTInferenceStrategy"
            )
            logger.warning(f"Old branch_factor_strategy: {old_strategy}")
            node_expander._params.pop("branch_factor_strategy")

        node_expander = node_expander.construct(
            branch_factor_strategy=ListBranchFactor(
                branch_factors=[
                    {"depth": 0, "branch_factor": samples},
                    {"depth": 1, "branch_factor": 1},
                ],
            ),
        )
        super().__init__(
            node_expander=node_expander,
            **kwargs,
        )

        if self.cloud_logger is not None:
            self.cloud_logger.summary["branch_factor_strategy"] = json.dumps(
                self.node_expander.branch_factor_strategy.branch_factors
            )
