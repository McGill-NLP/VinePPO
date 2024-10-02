import regex
from typing import Any, Dict, List, Optional

from datasets import (
    Dataset,
    DatasetDict,
)
from tqdm import tqdm

from treetune import logging_utils
from treetune.common.execute import check_correctness
from treetune.tasks import Task

logger = logging_utils.get_logger(__name__)

# FIND_CODE_REGEX = re.compile(r"```\n(def(\n|.)*)```")
# FIND_CODE_REGEX = re.compile(r"```\n(def\s+\w+\([^)]*\):(?:\n\s+.+)+?)\n```")
# FIND_CODE_REGEX = re.compile(r"```\n((?:.*\n)*?def\s+\w+\([^)]*\):(?:\n\s+.+)+?)\n```")
FIND_CODE_REGEX = regex.compile(
    r"```(python|Python|python2|python3)?\n((?:.*\n)*?def\s+\w+\([^)]*\)(\s*->\s*.+)?:(?:\n\s+.+)+?)\n```"
)


def compute_code_generation_metrics(
    predicted_code: str, test_cases: List[str], task_id: Optional[str] = None
):
    code_to_run = predicted_code + "\n\n" + "\n".join(test_cases)
    output = check_correctness(code_to_run, timeout=10, task_id=task_id)

    metrics = {
        "passed": float(output.passed),
        "syntax_error": float(output.has_syntax_error),
    }
    return metrics


@Task.register("mbpp", exist_ok=True)
class MBPP(Task):
    def extract_predicted_answer_from_text(self, text: str) -> str:
        # Find the first code block
        try:
            code_block = FIND_CODE_REGEX.search(text, timeout=10)
        except TimeoutError as e:
            raise ValueError(f"Failed to find code block in text: {text}") from e

        if code_block is None:
            raise ValueError(f"Could not find code block in text: {text}")

        # Extract the code block
        code_block = code_block.group(2)

        return code_block

    def evaluate_predictions(
        self,
        *,
        predictions: List[List[str]] = None,
        references: Dataset = None,
    ) -> Dict[str, float]:
        assert len(predictions) == len(references)

        num_skipped = 0

        passed = []
        passed_fraction = []
        syntax_error_fraction = []

        for answer_candidates, ref in tqdm(
            zip(predictions, references), total=len(references)
        ):
            if len(answer_candidates) == 0:
                logger.warning(
                    f"Found no predictions for {ref['_treetune__idx']}. Skipping this sample."
                )
                passed.append(0.0)
                num_skipped += 1
                continue

            train_test_case = ref["_prompt_test_case"]
            reference_test_cases = ref["_reference_test_cases"]
            all_test_cases = [train_test_case] + reference_test_cases

            test_imports: Optional[List[str]] = ref["test_imports"]
            if test_imports is not None and len(test_imports) > 0:
                all_test_cases = test_imports + ["\n\n"] + all_test_cases

            extracted_codes = []
            for candidate in answer_candidates:
                try:
                    extracted_code = self.extract_predicted_answer_from_text(candidate)
                    extracted_codes.append(extracted_code)
                except Exception as e:
                    logger.warning(
                        f"Failed to extract code from prediction: "
                        f"\n-----------------------\n{candidate}\n-----------------------\n"
                        f"with error: {e}"
                        "Skipping this candidate."
                    )
                    continue

            if len(extracted_codes) == 0:
                logger.warning(f"len(extracted_codes) = 0. Skipping this sample.")
                passed.append(0.0)
                num_skipped += 1
                continue

            num_passed = 0
            num_syntax_error = 0
            for extracted_code in extracted_codes:
                metrics = compute_code_generation_metrics(
                    extracted_code, all_test_cases
                )
                num_passed += metrics["passed"]
                num_syntax_error += metrics["syntax_error"]

            passed.append(float(num_passed > 0))
            passed_fraction.append(num_passed / len(extracted_codes))
            syntax_error_fraction.append(num_syntax_error / len(extracted_codes))

        return {
            "passed": (sum(passed) / len(passed)) if len(passed) > 0 else 0.0,
            "passed_fraction": (sum(passed_fraction) / len(passed_fraction))
            if len(passed_fraction) > 0
            else 0.0,
            "syntax_error_fraction": (
                sum(syntax_error_fraction) / len(syntax_error_fraction)
            )
            if len(syntax_error_fraction) > 0
            else 0.0,
            "num_skipped": num_skipped,
        }

    def build_dataset(
        self,
    ) -> DatasetDict:
        datasets = super().build_dataset()
        datasets = datasets.map(
            self._preprocess_example, num_proc=4, desc="Preprocessing examples"
        )
        return datasets

    def _preprocess_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        prompt = example["prompt"].strip()
        test_cases = example["test_list"]
        test_cases = [test_case.strip() for test_case in test_cases]

        prompt_test_case = test_cases[0]
        reference_test_cases = test_cases.copy()[1:]

        question = f"{prompt}\n[TESTS]\n{prompt_test_case}\n[/TESTS]"

        final_answer = example["code"]
        return {
            "question": question,
            "_final_answer": final_answer,
            "_prompt_test_case": prompt_test_case,
            "_reference_test_cases": reference_test_cases,
        }
