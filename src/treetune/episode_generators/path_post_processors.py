import numpy as np

from treetune.common import Registrable


class PathPostProcessor(Registrable):
    def __call__(self, path):
        raise NotImplementedError


@PathPostProcessor.register("chop_zero_advantage_tail")
class ChopZeroAdvantageTail(PathPostProcessor):
    def __call__(self, path):
        assert (
            len(path.keys()) == 1 and "node_chain" in path.keys()
        )  # making sure that the path is just a node_chain
        node_chain = path["node_chain"]
        index = 0
        for i in range(len(node_chain) - 1, 0, -1):
            if np.abs(node_chain[i]["advantage"]) >= 1e-6:
                index = i
                break
        chopped_chain = node_chain[: index + 1]
        return {"node_chain": chopped_chain}


@PathPostProcessor.register("keep_first_step_only")
class KeepFirstStepOnly(PathPostProcessor):
    def __call__(self, path):
        assert len(path.keys()) == 1 and "node_chain" in path.keys()
        node_chain = path["node_chain"]
        # Node 1 is the query, Node 2 is the first reasoning step
        chopped_chain = node_chain[:2]
        return {"node_chain": chopped_chain}


@PathPostProcessor.register("chop_up_to_first_negative_advantage")
class ChopUpToFirstNegativeAdvantage(PathPostProcessor):
    def __call__(self, path):
        assert len(path.keys()) == 1 and "node_chain" in path.keys()
        node_chain = path["node_chain"]
        if len(node_chain) == 1:
            # This is a query only path, so we can't chop it.
            assert (
                "advantage" not in node_chain[0]
            ), "This node should not have advantage"
            return path

        last_non_negative_index = 0
        for i in range(1, len(node_chain)):
            if node_chain[i]["advantage"] < 0:
                break
            last_non_negative_index = i

        chopped_chain = node_chain[: last_non_negative_index + 1]
        return {"node_chain": chopped_chain}


@PathPostProcessor.register("all_advantages_to_one")
class AllAdvantagesToOne(PathPostProcessor):
    def __call__(self, path):
        for node in path["node_chain"]:
            if "advantage" in node:
                node["advantage"] = 1
        return path


@PathPostProcessor.register("zero_advantages_to_epsilon")
class ZeroAdvantagesToEpsilon(PathPostProcessor):
    def __init__(self, epsilon: float = 0.05):
        self.epsilon = epsilon

    def __call__(self, path):
        for node in path["node_chain"]:
            if "advantage" in node and node["advantage"] == 0:
                node["advantage"] = self.epsilon
        return path


@PathPostProcessor.register("negative_advantages_to_zero")
class NegativeAdvantagesToZero(PathPostProcessor):
    def __call__(self, path):
        for node in path["node_chain"]:
            if "advantage" in node and node["advantage"] < 0:
                node["advantage"] = 0
        return path


@PathPostProcessor.register("non_last_nodes_score_to_zero")
class AllNonLastNodesScoreToZero(PathPostProcessor):
    def __call__(self, path):
        for node in path["node_chain"][:-1]:
            if "score" in node:
                node["score"] = 0
        return path


@PathPostProcessor.register("non_last_nodes_advantage_to_zero")
class AllNonLastNodesAdvantageToZero(PathPostProcessor):
    def __call__(self, path):
        for node in path["node_chain"][:-1]:
            if "advantage" in node:
                node["advantage"] = 0
        return path


@PathPostProcessor.register("fill_advantage_from_score")
class FillAdvantageFromScore(PathPostProcessor):
    def __call__(self, path):
        # We start from the second node, as the first node is the query
        for node in path["node_chain"][1:]:
            node["advantage"] = node["score"]
        return path


@PathPostProcessor.register("apply_importance_weights")
class ApplyImportanceWeights(PathPostProcessor):
    def __call__(self, path):
        for node in path["node_chain"]:
            if "advantage" in node:
                node["_orig_advantage"] = node["advantage"]
                node["advantage"] = node["advantage"] * node["importance_weight"]
        return path


@PathPostProcessor.register("last_step_advantage_to_one")
class LastStepAdvantageToOne(PathPostProcessor):
    def __call__(self, path):
        last_node = path["node_chain"][-1]
        last_node["advantage"] = 1
        return path
