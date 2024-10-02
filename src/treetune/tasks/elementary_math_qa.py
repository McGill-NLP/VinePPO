import re
from typing import Any, Dict, List

from datasets import (
    Dataset,
)

from treetune import logging_utils
from treetune.tasks import Task

logger = logging_utils.get_logger(__name__)

FIND_NUMBERS_REGEX = re.compile(
    r"(?:[+-]?\d+\.\d*|[+-]?\.\d+|[+-]?\d+e[-+]?\d+|[+-]?\d+)"
)


@Task.register("elementary_math_qa_one_number", exist_ok=True)
class ElementaryMathQA(Task):
    def extract_predicted_answer_from_text(self, text: str) -> str:
        text = text.replace(",", "")
        pred_answer = FIND_NUMBERS_REGEX.findall(text)
        if len(pred_answer) == 0:
            return None
        else:
            # Pick the last number
            pred_answer = pred_answer[-1].strip()
            return pred_answer

    def extract_gold_answer_from_text(self, text: str) -> str:
        text = text.replace(",", "")
        pred_answer = FIND_NUMBERS_REGEX.findall(text)
        if len(pred_answer) == 0:
            raise ValueError(f"Found no numbers in prediction: {text}.")
        elif len(pred_answer) > 1:
            raise ValueError(f"Found more than one number in prediction: {text}.")
        else:
            # Pick the last number
            pred_answer = pred_answer[-1].strip()
            return pred_answer

    def evaluate_predictions(
        self,
        *,
        predictions: List[List[Dict[str, Any]]] = None,
        references: Dataset = None,
    ) -> Dict[str, float]:
        num_skipped = 0
        exact_match_acc = []
        exact_match_frac = []
        majority_vote_acc = []
        unique_answer_count = []
        for answer_candidates, ref in zip(predictions, references):
            if len(answer_candidates) == 0:
                logger.warning(
                    f"Found no predictions for {ref['_treetune__idx']}. Skipping this sample."
                )
                exact_match_acc.append(0.0)
                num_skipped += 1
                continue

            gold_answer = self.extract_gold_answer_from_text(ref["answer"])

            pred_answers = []
            for candidate in answer_candidates:
                # Find all numbers in the prediction using FIND_NUMBERS_REGEX
                pred_answer = FIND_NUMBERS_REGEX.findall(candidate)
                if len(pred_answer) == 0:
                    logger.warning(
                        f"Found no numbers in prediction: {candidate}. Skipping this candidate."
                    )
                    continue

                # Pick the last number
                pred_answer = pred_answer[-1].strip()
                pred_answers.append(pred_answer)

            if len(pred_answers) == 0:
                logger.warning(f"len(set(pred_answers) = 0. Skipping this sample.")
                exact_match_acc.append(0.0)
                num_skipped += 1
                continue

            num_exact_match = 0
            for pred_answer in pred_answers:
                if gold_answer == pred_answer:
                    num_exact_match += 1

            exact_match_acc.append(float(num_exact_match > 0))
            exact_match_frac.append(num_exact_match / len(answer_candidates))
            majority_vote = max(set(pred_answers), key=pred_answers.count)
            majority_vote_acc.append(gold_answer == majority_vote)
            unique_answer_count.append(len(set(pred_answers)))

        exact_match = sum(exact_match_acc) / len(exact_match_acc)

        return {
            "exact_match": exact_match,
            "num_skipped": num_skipped,
            "exact_match_frac": sum(exact_match_frac) / len(exact_match_frac),
            "majority_vote_acc": sum(majority_vote_acc) / len(majority_vote_acc),
            "unique_answer_count": sum(unique_answer_count) / len(unique_answer_count),
        }
