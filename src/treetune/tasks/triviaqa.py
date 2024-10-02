import re
import string
from collections import Counter
from typing import Any, Dict, List

from datasets import (
    Dataset,
)
from tqdm import tqdm

from treetune import logging_utils
from treetune.tasks import Task

logger = logging_utils.get_logger(__name__)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    return white_space_fix(
        remove_articles(handle_punc(lower(replace_underscore(s))))
    ).strip()


def f1_score(prediction, ground_truth):
    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return prediction == ground_truth


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def get_normalized_ground_truths(data_instance: Dict[str, Any]):
    return data_instance["answer"]["normalized_aliases"] + [
        data_instance["answer"]["normalized_value"]
    ]


@Task.register("triviaqa", exist_ok=True)
class TriviaQA(Task):
    def extract_predicted_answer_from_text(self, text: str) -> str:
        return text

    def evaluate_predictions(
        self,
        *,
        predictions: List[List[str]] = None,
        references: Dataset = None,
    ) -> Dict[str, float]:
        assert len(predictions) == len(references)

        num_skipped = 0
        f1 = exact_match = 0

        for answer_candidates, ref in tqdm(
            zip(predictions, references), total=len(references)
        ):
            if len(answer_candidates) == 0:
                logger.warning(
                    f"Found no predictions for {ref['_treetune__idx']}. Skipping this sample."
                )
                num_skipped += 1
                continue

            assert (
                len(answer_candidates) == 1
            ), f"Only one prediction is supported. Got {len(answer_candidates)}"

            try:
                prediction = self.extract_predicted_answer_from_text(
                    answer_candidates[0]
                )
                normalized_prediction = normalize_answer(prediction)
                normalized_ground_truths = get_normalized_ground_truths(ref)

                em_for_this_question = metric_max_over_ground_truths(
                    exact_match_score, normalized_prediction, normalized_ground_truths
                )
                exact_match += em_for_this_question
                f1_for_this_question = metric_max_over_ground_truths(
                    f1_score, normalized_prediction, normalized_ground_truths
                )
                f1 += f1_for_this_question
            except Exception as e:
                logger.warning(
                    f"Error while evaluating {ref['_treetune__idx']}. Skipping this sample. Error: {e}"
                )
                num_skipped += 1
                continue

        exact_match = exact_match / len(references)
        f1 = f1 / len(references)

        return {
            "exact_match": exact_match,
            "f1": f1,
            "num_skipped": num_skipped,
        }
