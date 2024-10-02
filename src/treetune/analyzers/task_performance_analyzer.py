from datasets import Dataset

from treetune import logging_utils
from treetune.analyzers import Analyzer

logger = logging_utils.get_logger(__name__)


@Analyzer.register("task_performance")
class TaskPerformanceAnalyzer(Analyzer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.result_dir = self.runtime._get_result_dir()

        assert self.result_dir is not None, "Result directory is not set."

        if not self.result_dir.exists():
            raise ValueError(f"Result directory {self.result_dir} does not exist.")

    def get_analysis_id(self) -> str:
        return super().get_analysis_id() + self.result_dir.name

    def analyze(self):
        super().analyze()

        logger.info(f"Analyzing {self.result_dir}...")

        # Load the mutated dataset
        output_dataset = Dataset.load_from_disk(str(self.result_dir))

        assert (
            "_treetune__candidate_answers" in output_dataset.features
        ), "The dataset does not contain the candidate answers."

        predictions = output_dataset["_treetune__candidate_answers"]
        references = output_dataset

        metrics = self.task.evaluate_predictions(
            predictions=predictions, references=references
        )
        self.log_metrics(metrics)
        return metrics
