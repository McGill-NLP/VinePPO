from treetune.common import Registrable


class PathFilter(Registrable):
    def __call__(self, path) -> bool:
        raise NotImplementedError


@PathFilter.register("none")
class NonePathFilter(PathFilter):
    def __call__(self, path):
        return True


@PathFilter.register("successful")
class SuccessfulPathFilter(PathFilter):
    def __call__(self, path):
        last_node = path["node_chain"][-1]
        if "answer" not in last_node:
            return False
        return last_node["is_correct_answer"]


@PathFilter.register("non_zero_score")
class NonZeroScorePathFilter(PathFilter):
    def __call__(self, path):
        first_step_node = path["node_chain"][1]
        return first_step_node["score"] > 0


@PathFilter.register("non_zero_last_step_score")
class NonZeroLastStepScorePathFilter(PathFilter):
    def __call__(self, path):
        last_node = path["node_chain"][-1]
        return last_node["score"] > 0


@PathFilter.register("zero_last_step_score")
class ZeroLastStepScorePathFilter(PathFilter):
    def __call__(self, path):
        last_node = path["node_chain"][-1]
        return last_node["score"] == 0


@PathFilter.register("non_zero_last_step_advantage")
class NonZeroLastStepAdvantagePathFilter(PathFilter):
    def __call__(self, path):
        last_node = path["node_chain"][-1]
        return last_node["advantage"] > 0
