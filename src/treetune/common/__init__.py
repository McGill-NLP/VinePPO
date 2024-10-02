from enum import Enum
from typing import Dict, Any

from .from_params import FromParams, ConfigurationError
from .lazy import Lazy
from .params import Params
from .registrable import Registrable

assert FromParams
assert Lazy
assert Params
assert Registrable
assert ConfigurationError

JsonDict = Dict[str, Any]

class ExperimentStage(Enum):
    TRAINING = 0
    VALIDATION = 1
    TEST = 2
    PREDICTION = 3

    @staticmethod
    def from_split(split: str) -> "ExperimentStage":
        stage = {
            "valid": ExperimentStage.VALIDATION,
            "validation": ExperimentStage.VALIDATION,
            "test": ExperimentStage.TEST,
            "train": ExperimentStage.TRAINING,
            "predict": ExperimentStage.PREDICTION,
        }[split]
        return stage