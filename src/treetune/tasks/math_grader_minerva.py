import multiprocessing
from copy import deepcopy
from math import isclose
from multiprocessing import Pool
from typing import Union

import regex
from sympy import simplify, N
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr

from treetune.tasks.math_grader import (
    _sympy_parse,
    should_allow_eval,
    _parse_latex,
    _inject_implicit_mixed_number,
)

from treetune.logging_utils import get_logger

logger = get_logger(__name__)


def math_equal_process(param):
    return math_equal(param[-2], param[-1])


def parse_digits(num):
    # format: 234.23 || 23%
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except:
                pass
    return None


def is_digit(num):
    # paired with parse_digits
    return parse_digits(num) is not None


def _sympy_simplifies_to_zero(expr: str) -> bool:
    try:
        sympy_diff = _sympy_parse(expr)
        if simplify(sympy_diff) == 0:
            return True
    except Exception as e:
        return False


class SympySimplifier:
    """
    A class to safely simplify SymPy expressions, preventing indefinite hangs on complex inputs.

    This class uses a single-worker process pool to handle SymPy simplifications. This approach serves two purposes:
    1. It allows termination of potentially hanging operations caused by expressions with extremely large numbers
       (e.g., 37452^18374514).
    2. It maintains efficiency by reusing the same process for multiple calls, avoiding the overhead of creating
       a new process for each simplification attempt.

    The class manages a pool with a single worker, which can be terminated and recreated if a simplification
    operation exceeds the specified timeout. This ensures that the program can continue execution even if
    SymPy encounters an expression it cannot simplify within a reasonable time frame.

    Credits: @miladInk
    """

    @classmethod
    def _set_pool(cls, pool: Pool):
        cls._pool = pool

    @classmethod
    def _get_process_pool(cls) -> Pool:
        if not hasattr(cls, "_pool"):
            cls._set_pool(Pool(1))
        return cls._pool

    @classmethod
    def simplifies_to_zero(cls, expr: str, timeout: int = 15) -> bool:
        pool = cls._get_process_pool()
        try:
            res = pool.apply_async(_sympy_simplifies_to_zero, (expr,))
            return res.get(timeout=timeout)
        except multiprocessing.context.TimeoutError:
            pool.terminate()
            cls._set_pool(Pool(1))
            logger.warning(f"Timeout for sympy simplification: {expr}")
            return False


def symbolic_equal(a_latex, b_latex):
    def _parse(s, fast_latex=False):
        latex_parser = _parse_latex if fast_latex else parse_latex
        for f in [latex_parser, parse_expr]:
            try:
                return f(s)
            except:
                pass
        return s

    if not should_allow_eval(_parse(a_latex, fast_latex=True)) or not should_allow_eval(
        _parse(b_latex, fast_latex=True)
    ):
        return False

    a = _parse(a_latex, fast_latex=True)
    b = _parse(b_latex, fast_latex=True)

    try:
        expr = f"({a})-({b})"
        expr = _inject_implicit_mixed_number(expr)
        expr = expr.replace(" ", "")
        if SympySimplifier.simplifies_to_zero(expr):
            return True
    except Exception as e:
        pass

    try:
        if isclose(N(a), N(b), abs_tol=1e-3):
            return True
    except:
        pass
    return False


def symbolic_equal_process(a, b, output_queue):
    result = symbolic_equal(a, b)
    output_queue.put(result)


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = False,
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    if str(prediction) == str(reference):
        return True

    try:  # 1. numerical equal
        if is_digit(prediction) and is_digit(reference):
            prediction = parse_digits(prediction)
            reference = parse_digits(reference)
            # number questions
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if isclose(item, prediction, abs_tol=1e-3):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    if (
        regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
        and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                [
                    math_equal(
                        pred_parts[i], ref_parts[i], include_percentage, is_close
                    )
                    for i in range(len(pred_parts))
                ]
            ):
                return True

    if (
        (
            prediction.startswith("\\begin{pmatrix}")
            or prediction.startswith("\\begin{bmatrix}")
        )
        and (
            prediction.endswith("\\end{pmatrix}")
            or prediction.endswith("\\end{bmatrix}")
        )
        and (
            reference.startswith("\\begin{pmatrix}")
            or reference.startswith("\\begin{bmatrix}")
        )
        and (
            reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}")
        )
    ):
        pred_lines = [
            line.strip()
            for line in prediction[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            if line.strip()
        ]
        ref_lines = [
            line.strip()
            for line in reference[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            if line.strip()
        ]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) == len(ref_parts):
                    if not all(
                        [
                            math_equal(
                                pred_parts[i],
                                ref_parts[i],
                                include_percentage,
                                is_close,
                            )
                            for i in range(len(pred_parts))
                        ]
                    ):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True

    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
            return True
    elif (
        prediction.count("=") == 1
        and len(prediction.split("=")[0].strip()) <= 2
        and "=" not in reference
    ):
        if math_equal(
            prediction.split("=")[1], reference, include_percentage, is_close
        ):
            return True
    elif (
        reference.count("=") == 1
        and len(reference.split("=")[0].strip()) <= 2
        and "=" not in prediction
    ):
        if math_equal(
            prediction, reference.split("=")[1], include_percentage, is_close
        ):
            return True

    # symbolic equal with sympy
    if timeout:
        if call_with_timeout(symbolic_equal_process, prediction, reference):
            return True
    else:
        if symbolic_equal(prediction, reference):
            return True

    return False


def is_correct(item, pred_key="prediction", prec=1e-3):
    pred = item[pred_key]
    ans = item["answer"]
    if isinstance(pred, list) and isinstance(ans, list):
        pred_matched = set()
        ans_matched = set()
        for i in range(len(pred)):
            for j in range(len(ans)):
                item_cpy = deepcopy(item)
                item_cpy.update({pred_key: pred[i], "answer": ans[j]})
                if is_correct(item_cpy, pred_key=pred_key, prec=prec):
                    pred_matched.add(i)
                    ans_matched.add(j)
                    if item_cpy[pred_key] == "2,3,4":
                        print(item, flush=True)
                        print("wtf", flush=True)
        return len(pred_matched) == len(pred) and len(ans_matched) == len(ans)
    elif isinstance(pred, str) and isinstance(ans, str):
        if "\\cup" in pred and "\\cup" in ans:
            item = deepcopy(item)
            item.update(
                {
                    pred_key: pred.split("\\cup"),
                    "answer": ans.split("\\cup"),
                }
            )
            return is_correct(item, pred_key=pred_key, prec=prec)
        else:
            label = False
            try:
                label = (
                    abs(
                        float(regex.sub(r",", "", str(pred)))
                        - float(regex.sub(r",", "", str(ans)))
                    )
                    < prec
                )
            except:
                pass
            label = label or (ans and pred == ans) or math_equal(pred, ans)
            return label
    else:
        print(item, flush=True)
        raise NotImplementedError()


def eval_math(item, pred_key="prediction", prec=1e-3):
    pred = item[pred_key]
    if pred_key == "program_output" and isinstance(pred, str):
        pred = [pred]
    ans = item["answer"]
    if isinstance(pred, list) and isinstance(ans, list):
        # for some questions in MATH, `reference` repeats answers
        _ans = []
        for a in ans:
            if a not in _ans:
                _ans.append(a)
        ans = _ans
        # some predictions for MATH questions also repeats answers
        _pred = []
        for a in pred:
            if a not in _pred:
                _pred.append(a)
        # some predictions mistakenly box non-answer strings
        pred = _pred[-len(ans) :]

    item.update({pred_key: pred, "answer": ans})
    return is_correct(item, pred_key=pred_key, prec=prec)
