from typing import Dict, Tuple, List

from treetune.common import Registrable
from treetune.inference_strategies.tree_inference import Node


class BranchFactorStrategy(Registrable):
    def decide_branch_factor(self, node: Node) -> int:
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.decide_branch_factor(*args, **kwargs)


@BranchFactorStrategy.register("constant", exist_ok=True)
class ConstantBranchFactor(BranchFactorStrategy):
    def __init__(self, constant: int):
        super().__init__()
        self.constant = constant

    def decide_branch_factor(self, node: Node) -> int:
        return self.constant


@BranchFactorStrategy.register("list", exist_ok=True)
class ListBranchFactor(BranchFactorStrategy):
    def __init__(self, branch_factors: List[Dict[str, int]]):
        """
        :param branch_factors: a list of dictionaries, which has two keys: "depth" and "branch_factor".
        The list should be sorted by "depth" in ascending order.
        which means if depth is in  [branch_factors[i]['depth'], branch_factors[i+1]['depth']), then
        the branch factor is branch_factors[i]['branch_factor'].
        """
        super().__init__()
        assert len(branch_factors) > 0, "branch_factors should not be empty, use at least {'depth': 0, 'branch_factor': x}}"
        self.branch_factors = sorted(branch_factors, key=lambda x: x["depth"])
        if self.branch_factors[0]['depth'] != 0:
            raise ValueError("The first depth must be 0")

    def decide_branch_factor(self, node: Node) -> int:
        depth = node["depth"]
        if len(self.branch_factors) == 1:
            return self.branch_factors[0]['branch_factor']

        for i in range(len(self.branch_factors)):
            if depth < self.branch_factors[i]['depth']:
                return self.branch_factors[i-1]['branch_factor']

        return self.branch_factors[-1]['branch_factor']


@BranchFactorStrategy.register("fish_bone", exist_ok=True)
class FishBoneBranchFactor(BranchFactorStrategy):
    def __init__(self, fish_bone_samples: int):
        super().__init__()
        self.fish_bone_samples = fish_bone_samples

    def decide_branch_factor(self, node: Node) -> int:
        if 'is_in_spine' in node and node['is_in_spine'] is True:
            return self.fish_bone_samples
        else:
            return 1






