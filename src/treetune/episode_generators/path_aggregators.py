import json
import random

from treetune.common import Registrable


class PathAggregator(Registrable):
    def __call__(self, paths):
        raise NotImplementedError


@PathAggregator.register("none")
class NonePathAggregator(PathAggregator):
    def __call__(self, paths):
        return paths


@PathAggregator.register("unique")
class UniquePathAggregator(PathAggregator):
    def __call__(self, paths):
        # we transform the paths to json to make them hashable, then we return them to their original form
        unique_paths = {json.dumps(p, sort_keys=True) for p in paths}
        return [json.loads(p) for p in unique_paths]


@PathAggregator.register("top_k_percent")
class TopKPercentPathAggregator(PathAggregator):
    def __init__(
        self,
        top_k_percent: float,
        keep_at_least_one: bool = True,
        reduction: str = "mean",
    ):
        assert 0 < top_k_percent <= 1
        self.top_k_percent = top_k_percent
        self.keep_at_least_one = keep_at_least_one
        self.reduction = reduction

    def __call__(self, paths):
        if len(paths) == 0:
            return paths

        # Compute the advantage sum of each path
        path_advantage_sums = []
        for path in paths:
            all_advantages = [node.get("advantage", 0) for node in path["node_chain"]]
            if self.reduction == "mean":
                score = sum(all_advantages) / len(all_advantages)
            elif self.reduction == "sum":
                score = sum(all_advantages)
            else:
                raise ValueError(f"Unknown reduction: {self.reduction}")

            path_advantage_sums.append((path, score))

        # Sort the paths by their advantage sum
        path_advantage_sums = sorted(
            path_advantage_sums, key=lambda x: x[1], reverse=True
        )

        # Keep the top k percent of paths
        num_paths_to_keep = int(len(path_advantage_sums) * self.top_k_percent)
        if self.keep_at_least_one:
            num_paths_to_keep = max(num_paths_to_keep, 1)

        paths = [path for path, _ in path_advantage_sums[:num_paths_to_keep]]

        return paths


@PathAggregator.register("max_k_paths")
class MaxKPathAggregator(PathAggregator):
    def __init__(self, k: int):
        self.k = k
        self.rng = random.Random(42)

    def __call__(self, paths):
        if len(paths) <= self.k:
            return paths

        return self.rng.sample(paths, self.k)
