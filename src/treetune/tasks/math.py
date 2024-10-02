import copy
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import Dataset, DatasetDict

from treetune import logging_utils
from treetune.tasks import Task
from treetune.tasks.math_answer_exctraction import (
    extract_math_minerva_few_shot_cot_answer,
    extract_math_answer,
)
from treetune.tasks.math_grader import grade_answer
from treetune.tasks.math_grader_minerva import eval_math
from treetune.tokenization_utils import Tokenizer

logger = logging_utils.get_logger(__name__)


@Task.register("math", exist_ok=True)
class MATH(Task):
    def __init__(
        self,
        prepend_in_context_few_shot: bool,
        few_shot_dataset_path: Optional[str] = None,
        use_minerva_few_shot_prompt: bool = False,
        use_gold_steps_for_few_shot: bool = False,
        num_few_shot_examples: Optional[int] = None,
        max_few_shot_problem_length: Optional[int] = None,
        max_few_shot_solution_length: Optional[int] = None,
        max_few_shot_num_steps: Optional[int] = None,
        tokenizer: Optional[Tokenizer] = None,
        ensure_fit_in_context_size: bool = False,
        max_few_shot_dataset_size: Optional[int] = None,
        context_size: Optional[int] = None,
        max_generation_tokens: Optional[int] = None,
        intermediate_step_delimiter: str = "\n",
        inplace_split_solution: bool = False,
        answer_prefix: Optional[str] = "\n\n# Answer\n",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prepend_in_context_few_shot = prepend_in_context_few_shot
        self.num_few_shot_examples = num_few_shot_examples
        self.ensure_fit_in_context_size = ensure_fit_in_context_size
        self.tokenizer = tokenizer
        self.use_minerva_few_shot_prompt = use_minerva_few_shot_prompt
        self.use_gold_steps_for_few_shot = use_gold_steps_for_few_shot
        self.few_shot_dataset = None
        self.context_size = context_size
        self.max_generation_tokens = max_generation_tokens
        self.max_few_shot_problem_length = max_few_shot_problem_length
        self.max_few_shot_solution_length = max_few_shot_solution_length
        self.max_few_shot_num_steps = max_few_shot_num_steps
        self.max_few_shot_dataset_size = max_few_shot_dataset_size
        self.intermediate_step_delimiter = intermediate_step_delimiter
        self.answer_prefix = answer_prefix
        self.inplace_split_solution = inplace_split_solution

        # If few-shot examples are to be included, load the dataset from the provided path.
        if self.prepend_in_context_few_shot:

            if self.ensure_fit_in_context_size:
                assert self.context_size is not None, "Context size must be provided."

            if not self.use_minerva_few_shot_prompt:
                self.few_shot_dataset = Dataset.load_from_disk(few_shot_dataset_path)

            if self.max_few_shot_dataset_size is not None:
                self.few_shot_dataset = self.few_shot_dataset.shuffle(seed=42).select(
                    range(self.max_few_shot_dataset_size)
                )

            if (
                self.few_shot_dataset is not None
                and "gold_solution_steps" in self.few_shot_dataset.column_names
            ):

                def append_gold_solution_steps_str(
                    example: Dict[str, Any]
                ) -> Dict[str, Any]:
                    sol_steps = example["gold_solution_steps"]
                    sol_steps_str = "\n".join(sol_steps)
                    return {"gold_solution_steps_str": sol_steps_str}

                self.few_shot_dataset = self.few_shot_dataset.map(
                    append_gold_solution_steps_str,
                    num_proc=4,
                    desc="Appending gold solution steps",
                )

            if self.few_shot_dataset is not None and any(
                [
                    self.max_few_shot_problem_length,
                    self.max_few_shot_solution_length,
                    self.max_few_shot_num_steps,
                ]
            ):
                assert tokenizer is not None, "Tokenizer must be provided."
                self._preprocess_few_shot_dataset()

    def _preprocess_few_shot_dataset(self):
        tokenizer = self.tokenizer
        use_gold_steps_for_few_shot = self.use_gold_steps_for_few_shot

        def keep_shorter_than_max_length(example: Dict[str, Any]) -> bool:
            problem_length = len(tokenizer.encode(example["problem"]))
            if "gold_solution_steps" in example and use_gold_steps_for_few_shot:
                solution = example["gold_solution_steps"]
                num_steps = len(solution)
            elif isinstance(example["solution"], list):
                solution = "\n".join(example["solution"])
                num_steps = len(example["solution"])
            else:
                solution = example["solution"]
                num_steps = None

            solution_length = len(tokenizer.encode(solution))

            is_short_enough = True
            if self.max_few_shot_problem_length is not None:
                is_short_enough &= problem_length <= self.max_few_shot_problem_length

            if self.max_few_shot_solution_length is not None:
                is_short_enough &= solution_length <= self.max_few_shot_solution_length

            if self.max_few_shot_num_steps is not None and num_steps is not None:
                is_short_enough &= num_steps <= self.max_few_shot_num_steps

            return is_short_enough

        ds_len_before = len(self.few_shot_dataset)
        self.few_shot_dataset = self.few_shot_dataset.filter(
            keep_shorter_than_max_length,
            num_proc=4,
            desc="Filtering few-shot examples",
        )
        logger.info(
            f"Filtered few-shot examples from {ds_len_before} to {len(self.few_shot_dataset)} examples"
        )

    def build_dataset(self) -> DatasetDict:
        datasets = super().build_dataset()

        def append_gold_solution_steps_str(example: Dict[str, Any]) -> Dict[str, Any]:
            sol_steps = example["gold_solution_steps"]
            sol_steps_str = "\n".join(sol_steps)
            return {"gold_solution_steps_str": sol_steps_str}

        if 'train' in datasets and "gold_solution_steps" in datasets["train"].column_names:
            datasets = datasets.map(
                append_gold_solution_steps_str,
                num_proc=4,
                desc="Appending gold solution steps",
            )

        if self.use_minerva_few_shot_prompt:
            map_fn = self._get_preprocess_example_for_minerva_prompt()
        else:
            map_fn = self._get_preprocess_example()
        datasets = datasets.map(
            map_fn,
            num_proc=4,
            desc="Preprocessing examples",
        )
        return datasets

    def _get_preprocess_example(self):
        tokenizer = self.tokenizer
        few_shot_dataset = self.few_shot_dataset
        num_few_shot_examples = self.num_few_shot_examples
        prepend_in_context_few_shot = self.prepend_in_context_few_shot
        use_gold_steps_for_few_shot = self.use_gold_steps_for_few_shot
        ensure_fit_in_context_size = self.ensure_fit_in_context_size
        context_size = self.context_size
        max_generation_tokens = self.max_generation_tokens
        answer_prefix = self.answer_prefix

        def get_solution_text(example, answer=None):
            solution = example["solution"]
            if "gold_solution_steps_str" in example and use_gold_steps_for_few_shot:
                # MATH solutions split into steps using a heuristic.
                solution = example["gold_solution_steps_str"]
            elif isinstance(solution, list):
                # OpenAI PRM format
                return "\n".join(solution)  # Already contains final answer.

            if answer is not None and answer_prefix is not None:
                # Append the answer to the solution text.
                return solution + answer_prefix + answer
            return solution

        def generate_fewshot_context(rng, problem_text, delimiter):
            random_indices = rng.choice(
                len(few_shot_dataset),
                num_few_shot_examples + 1,  # Extra to avoid self-inclusion.
                replace=False,
            )

            # Filter out the current problem from the few-shot examples.
            few_shot_examples = [
                few_shot_dataset[i]
                for i in random_indices.tolist()
                if few_shot_dataset[i]["problem"] != problem_text
            ][:num_few_shot_examples]

            # Format few-shot examples as strings.
            few_shot_examples_strs = []
            for ex in few_shot_examples:
                fs_problem = ex["problem"]
                fs_solution_str = get_solution_text(ex, answer=ex.get("answer"))
                fs_str = f"Problem:\n{fs_problem}\n\nSolution:\n{fs_solution_str}"
                few_shot_examples_strs.append(fs_str)

            few_shot_context = delimiter.join(few_shot_examples_strs)
            return few_shot_context

        def _preprocess_example(example: Dict[str, Any]) -> Dict[str, Any]:
            problem_text = example["problem"]

            few_shot_delimiter = "\n\n\n"

            max_retries = 10 if ensure_fit_in_context_size else 1

            output = {}
            if prepend_in_context_few_shot:
                # Generate a seed based on the example's index for reproducibility.
                init_seed = example["_treetune__idx"]

                num_tries = 0
                while num_tries < max_retries:
                    rng = np.random.RandomState(init_seed + num_tries)
                    few_shot_ctx = generate_fewshot_context(
                        rng, problem_text, few_shot_delimiter
                    )
                    query = (
                        few_shot_ctx
                        + few_shot_delimiter
                        + f"Problem:\n{problem_text}\n\nSolution:\n"
                    )
                    prompt_tokens = tokenizer.encode(query)
                    if (len(prompt_tokens) + max_generation_tokens) <= context_size:
                        break
                    logger.warning(
                        f"Could not fit the prompt in the context size. Retrying..."
                    )
                    num_tries += 1

                if ensure_fit_in_context_size and num_tries == max_retries:
                    logger.warning(
                        f"Could not fit the few-shot context in the context size for problem: {problem_text}"
                    )
                    # Just discard the first few tokens
                    extra_tokens_length = (
                        len(prompt_tokens)
                        + self.max_generation_tokens
                        - self.context_size
                    )
                    prompt_tokens = prompt_tokens[extra_tokens_length:]
                    query = tokenizer.decode(prompt_tokens)

                output["_few_shot_context"] = few_shot_ctx
            else:
                query = problem_text

            output["query"] = query

            return output

        return _preprocess_example

    def _get_preprocess_example_for_minerva_prompt(self):
        tokenizer = self.tokenizer
        prepend_in_context_few_shot = self.prepend_in_context_few_shot
        ensure_fit_in_context_size = self.ensure_fit_in_context_size
        context_size = self.context_size
        max_generation_tokens = self.max_generation_tokens

        def map_fn(example: Dict[str, Any]) -> Dict[str, Any]:
            problem_text = example["problem"]

            output = {}
            if prepend_in_context_few_shot:
                query = (
                    MINERVA_FEWSHOT_MATH_PROMPT
                    + "\n\n"
                    + f"Problem:\n{problem_text}\n\nSolution:\n"
                )
                prompt_tokens = tokenizer.encode(query)

                if (
                    ensure_fit_in_context_size
                    and (len(prompt_tokens) + max_generation_tokens) > context_size
                ):
                    logger.warning(
                        f"Could not fit the few-shot context in the context size for problem: {problem_text}"
                    )
                    # Just discard the first few tokens
                    extra_tokens_length = (
                        len(prompt_tokens) + max_generation_tokens - context_size
                    )
                    prompt_tokens = prompt_tokens[extra_tokens_length:]
                    query = tokenizer.decode(prompt_tokens)

                assert (
                    len(prompt_tokens) + max_generation_tokens
                ) <= context_size, f"Could not fit the prompt in the context size for problem: {problem_text}"
                output["_few_shot_context"] = MINERVA_FEWSHOT_MATH_PROMPT
            else:
                query = problem_text

            output["query"] = query

            return output

        return map_fn

    def extract_predicted_answer_from_text(
        self, text: str, problem: Optional[str] = None
    ) -> Optional[str]:
        """
        We assume that the solution is in the format:
        Solution:
        <reasoning_step_1>
        <reasoning_step_2>
        ...
        <reasoning_step_n>
        # Answer

        <answer>
        """
        if self.use_minerva_few_shot_prompt or self.answer_prefix is None:
            return self.extract_predicted_answer_from_text_minerva(text, problem)

        splits = text.split("# Answer\n")

        # Be conservative and return None if the format is not as expected.
        if len(splits) != 2:
            return None

        return splits[1].strip()

    def extract_predicted_answer_from_text_minerva(
        self, text: str, problem: str
    ) -> Optional[str]:
        return extract_math_minerva_few_shot_cot_answer(problem, text)

    def split_solution_into_intermediate_steps(self, solution: str) -> List[int]:
        """
        Split the solution into reasoning steps.

        Args:
            solution: The solution text.

        Returns:
            A list of indices where each index corresponds to the start of a reasoning step.
            Example:
            >>> solution = '...'
            >>> indices = split_solution_into_reasoning_steps(solution)
            >>> steps = [solution[indices[i]:indices[i+1]] for i in range(len(indices) - 1)]
        """
        if self.use_minerva_few_shot_prompt or self.inplace_split_solution:
            return self._split_solution_into_intermediate_steps_minerva(solution)

        delimiter = self.intermediate_step_delimiter
        answer_prefix = self.answer_prefix

        solution_parts = solution.split(answer_prefix)

        if len(solution_parts) != 2:
            sol_without_answer, answer = solution, None
        else:
            sol_without_answer, answer = solution_parts

        steps = sol_without_answer.split(delimiter)

        # Merge first empty steps to the first non-empty step
        first_non_empty_step_idx = None
        for i, step in enumerate(steps):
            if step.strip() != "":
                first_non_empty_step_idx = i
                break

        if first_non_empty_step_idx is not None and first_non_empty_step_idx > 0:
            new_first_step = delimiter.join(steps[: first_non_empty_step_idx + 1])

            steps = [new_first_step] + steps[first_non_empty_step_idx + 1 :]

        if answer is not None:
            # We want to merge the last step with the answer

            # Find last non-empty step index
            last_non_empty_step_idx = None
            for i in range(len(steps) - 1, -1, -1):
                if steps[i].strip() != "":
                    last_non_empty_step_idx = i
                    break

            if last_non_empty_step_idx is None:
                # Then it means the entire solution is a single step
                last_non_empty_step_idx = 0

            new_last_step = delimiter.join(steps[last_non_empty_step_idx:])
            # Also merge the last step with the answer
            new_last_step = f"{new_last_step}{answer_prefix}{answer}"
            steps = steps[:last_non_empty_step_idx] + [new_last_step]

        reconstructed_solution = delimiter.join(steps)
        assert (
            reconstructed_solution == solution
        ), f"{reconstructed_solution} != {solution}"

        # Find the indices of the reasoning steps
        indices = [0]
        for i, step in enumerate(steps):
            if i == 0:
                indices.append(indices[-1] + len(step))
            else:
                indices.append(indices[-1] + len(step) + len(delimiter))

        assert indices[-1] == len(solution), f"{indices[-1]} != {len(solution)}"

        return indices

    def _split_solution_into_intermediate_steps_minerva(
        self, solution: str
    ) -> List[int]:
        from treetune.tasks.math_extract_steps_inplace import split_solution_inplace

        if self.answer_prefix is None:
            return split_solution_inplace(solution)

        solution_parts = solution.split(self.answer_prefix)

        if len(solution_parts) != 2:
            sol_without_answer, answer = solution, None
        else:
            sol_without_answer, answer = solution_parts

        indices_without_answer = split_solution_inplace(sol_without_answer)

        # Merge last step with the answer if it exists
        if answer is not None:
            indices_without_answer[-1] = len(solution)

        return indices_without_answer

    def grade_answer(
        self,
        *,
        given_answer: str = None,
        ground_truth: str = None,
        item: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> bool:
        """
        The grading function is provided by OpenAI PRM paper
        https://github.com/openai/prm800k/tree/main?tab=readme-ov-file#answer-grading
        """
        use_minerva_few_shot_prompt = self.use_minerva_few_shot_prompt
        answer_prefix = self.answer_prefix

        def grade_fn():
            if use_minerva_few_shot_prompt or answer_prefix is None:
                return self.grade_answer_minerva_format(
                    given_answer=given_answer, item=item
                )
            return grade_answer(given_answer=given_answer, ground_truth=ground_truth)

        if timeout is None:
            return grade_fn()

        from call_function_with_timeout import SetTimeout

        func = SetTimeout(grade_fn, timeout=timeout)
        is_done, is_timeout, error_message, results = func()
        if is_timeout:
            logger.warning(
                f"Grading function timed out for problem:\n{item['problem']}\n and solution:\n{given_answer}"
            )
            return False

        return results

    def grade_answer_minerva_format(
        self,
        *,
        given_answer: str = None,
        item: Optional[Dict[str, Any]] = None,
    ) -> bool:
        item = copy.deepcopy(item)
        answer = extract_math_answer(item["problem"], item["solution"])
        item["answer"] = answer
        item["prediction"] = given_answer
        return eval_math(item)

    def evaluate_predictions(
        self,
        *,
        predictions: List[List[str]] = None,
        references: Dataset = None,
    ) -> Dict[str, float]:
        assert len(predictions) == len(references)
        assert len(predictions) > 0, "No predictions provided."

        once_hit_acc = []
        correct_frac = []
        majority_vote_acc = []
        unique_answer_count = []
        none_answer_extracted = []
        for solution_candidates, ref in zip(predictions, references):
            gold_answer = ref["answer"]
            problem = ref["problem"]

            assert len(solution_candidates) > 0
            answer_candidates = [
                self.extract_predicted_answer_from_text(sol, problem=problem)
                for sol in solution_candidates
            ]
            none_answer_extracted.append(
                sum([1 for ans in answer_candidates if ans is None])
                / len(answer_candidates)
            )

            grading_results = [
                self.grade_answer(given_answer=ans, ground_truth=gold_answer, item=ref)
                for ans in answer_candidates
            ]
            once_hit_acc.append(float(any(grading_results)))
            correct_frac.append(sum(grading_results) / len(grading_results))

            answer_candidates = [
                tuple(ans) if isinstance(ans, list) else ans
                for ans in answer_candidates
            ]

            majority_answer, _ = Counter(answer_candidates).most_common(n=1)[0]
            assert len(answer_candidates) == len(grading_results)
            majority_answer_index = answer_candidates.index(majority_answer)
            majority_answer_is_correct = grading_results[majority_answer_index]
            majority_vote_acc.append(majority_answer_is_correct)

            unique_answer_count.append(len(set(answer_candidates)))

        once_hit = sum(once_hit_acc) / len(once_hit_acc)
        correct_frac = sum(correct_frac) / len(correct_frac)

        return {
            "once_hit": once_hit,
            "exact_match": once_hit,  # for backwards compatibility
            "correct_frac": correct_frac,
            "exact_match_frac": correct_frac,  # for backwards compatibility
            "majority_vote_acc": sum(majority_vote_acc) / len(majority_vote_acc),
            "unique_answer_count": sum(unique_answer_count) / len(unique_answer_count),
            "none_answer_extracted_frac_per_problem": (
                sum(none_answer_extracted) / len(none_answer_extracted)
            ),
        }


MINERVA_FEWSHOT_MATH_PROMPT = """Problem:
Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}

Solution:
The expressions inside each square root must be non-negative.
Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$.
Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$.
Therefore, the domain of the expression is $\\boxed{[2,5)}$.
Final Answer: The final answer is $[2,5)$. I hope it is correct.

Problem:
If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$

Solution:
We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$
Final Answer: The final answer is $24$. I hope it is correct.

Problem:
Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?

Solution:
If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$: \\begin{align*}
30n&=480\\\\
\\Rightarrow\\qquad n&=480/30=\\boxed{16}
\\end{align*}
Final Answer: The final answer is $16$. I hope it is correct.

Problem:
If the system of equations

\\begin{align*}
6x-4y&=a,\\\\
6y-9x &=b.
\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ is nonzero.

Solution:
If we multiply the first equation by $-\\frac{3}{2}$, we obtain

$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have

$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$
Final Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct."""

KONKUR_PROMPT = """Problem:
Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}

Solution:
The expressions inside each square root must be non-negative.
Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$.
Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$.
Therefore, the domain of the expression is $\\boxed{[2,5)}$.
Final Answer: The final answer is $\\boxed{[2,5)}$. I hope it is correct.

Problem:
What is the difference between the average rate of change of the function $f(x) = \pi \cos 2x \sin x$ in the interval $[0, \frac{\pi}{6}]$ and its instantaneous rate of change in the interval $[0, \frac{\pi}{2}]$?

Solution:
First, we find the average rate of change of the function in the interval $[0, \frac{\pi}{6}]$.
$f(0) = \pi \cos 0 \cdot \sin 0 = 0$
$f\left( \frac{\pi}{6} \right) = \pi \cos \left( \frac{\pi}{3} \right) \sin \left( \frac{\pi}{6} \right) = \pi \cdot \frac{1}{2} \cdot \frac{1}{2} = \frac{\pi}{4}$
$m_1 = \frac{f\left( \frac{\pi}{6} \right) - f(0)}{\frac{\pi}{6} - 0} = \frac{\frac{\pi}{4} - 0}{\frac{\pi}{6}} = \frac{\frac{\pi}{4}}{\frac{\pi}{6}} = \frac{6}{4} = \frac{3}{2}$
Second, For the interval $[0, \frac{\pi}{2}]$, we have:
$f(0) = 0$
$f\left( \frac{\pi}{2} \right) = \pi \cos \left( \pi \right) \sin \left( \frac{\pi}{2} \right) = \pi \cdot (-1) \cdot 1 = -\pi$
$m_2 = \frac{f\left( \frac{\pi}{2} \right) - f(0)}{\frac{\pi}{2} - 0} = \frac{-\pi - 0}{\frac{\pi}{2}} = -2$
Finally, we find the difference:
$|m_1 - m_2| = \left| \frac{3}{2} - (-2) \right| = \frac{7}{2} = \\boxed{\frac{3}{5}}$
Final Answer: The final answer is $\\boxed{\frac{3}{5}}$. I hope it is correct.
"""
