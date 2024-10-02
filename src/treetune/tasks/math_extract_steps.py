import copy
import hashlib
import re
from typing import List


def md5_hash(text):
    """Return the MD5 hash of the given text."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def replace_math_with_placeholders(latex_content):
    math_patterns = [
        r"\[asy\].*?\[/asy\]",  # Matches Asymptote environments
        r"\$\\begin\{tabular\}.*?\\end\{tabular\}\$",  # Specific pattern for tabular inside inline math
        r"\\begin\{tabular\}.*?\\end\{tabular\}",  # Matches tabular environments
        r"\\\((.*?)\\\)",  # Matches display math mode with \( \)
        r"\\\[.*?\\\]",  # Matches display math mode with \[ \]
        r"\$\$(.*?)\$\$",  # Matches display math mode with $$
        r"(?<!\\)\$(.*?[^\\])\$(?!\$)",  # Updated to require at least one character inside
        r"\\begin\{([^}]*)\}.*?\\end\{\1\}",  # Matches multiline environments like \begin{equation} ... \end{equation}
    ]
    patterns_type = [
        "asymptote",
        "tabular_inline",
        "tabular",
        "display(",
        "display_[",
        "display_$$",
        "display_$",
        "environment",
    ]

    assert len(math_patterns) == len(
        patterns_type
    ), "Length of patterns and patterns_type should be the same"

    placeholders = {}
    placeholders_type = {}

    def replace_math(match, pattern_type):
        hash_key = md5_hash(match.group(0))
        placeholder = f"<|PLACEHOLDER_{hash_key}_PLACEHOLDER|>"
        placeholders[placeholder] = match.group(0)
        placeholders_type[placeholder] = pattern_type
        return placeholder

    for pattern, pattern_type in zip(math_patterns, patterns_type):
        latex_content = re.sub(
            pattern,
            lambda match: replace_math(match, pattern_type),
            latex_content,
            flags=re.DOTALL,
        )

    return latex_content, placeholders, placeholders_type


def recover_math_from_placeholders(text_with_placeholders, placeholders):
    for placeholder, math in placeholders.items():
        text_with_placeholders = text_with_placeholders.replace(placeholder, math)
    return text_with_placeholders


def naive_split_by_newline_and_period(disguised_latex: str):
    period_pattern = r"(?<!\b[A-Za-z]\.)(?<!\s[A-Za-z]\.)(?<!\s[A-Za-z][A-Za-z]\.)(?<=\.)(?=\s+[A-Z])"
    parts = re.split(period_pattern, disguised_latex)
    parts = [part.strip() for part in parts if len(part.strip()) > 0]

    final_parts = []
    for part in parts:
        parts_new = part.split("\n")
        parts_new = [p.strip() for p in parts_new if len(p.strip()) > 0]
        final_parts.extend(parts_new)

    final_parts = merge_short_parts_forward(final_parts)

    # Make sure we didn't cut into PLACEHOLDER's
    for p in final_parts:
        if "<|PLACEHOLDER_" in p:
            assert "PLACEHOLDER|>" in p, "Cut into a placeholder"

    return final_parts


def merge_short_parts_forward(parts):
    # Merge short parts with the next part
    parts = copy.deepcopy(parts)
    i = 0
    while i < len(parts) - 1:
        if not parts[i].endswith(".") and len(parts[i]) < 20:
            parts[i] = parts[i].rstrip() + " " + parts.pop(i + 1).lstrip()
        else:
            i += 1

    return parts


def has_placeholders(text):
    return "<|PLACEHOLDER_" in text


PLACEHOLDER_LENGTH = 60


def find_index_of_all_placeholders(text) -> list[int]:
    return [
        m.start()
        for m in re.finditer(r"<\|PLACEHOLDER_[0-9a-f]{32}_PLACEHOLDER\|>", text)
    ]


def split_tailing_periods_in_placeholders(parts, placeholders):
    # Define the points where we want to split
    tail_patterns = [r".$", r".$$", r".\)", r".\]"]

    resulting_parts = []
    for part in parts:
        split_points = []
        for start_idx in find_index_of_all_placeholders(part):
            ph_text = part[start_idx : start_idx + PLACEHOLDER_LENGTH]
            actual_text = placeholders[ph_text]
            for pattern in tail_patterns:
                if actual_text.endswith(pattern):
                    split_points.append(start_idx + PLACEHOLDER_LENGTH)

        split_points = [0] + split_points + [len(part)]
        for i in range(1, len(split_points)):
            resulting_parts.append(part[(split_points[i - 1]) : (split_points[i])])

    for part in resulting_parts:
        if "<|PLACEHOLDER_" in part:
            assert "_PLACEHOLDER|>" in part, "Cut into a placeholder"

    return resulting_parts


def split_newline_in_placeholders(parts, placeholders):
    for key, value in placeholders.items():
        # Replace multiple newlines with a single newline
        placeholders[key] = re.sub(r"\n+", "\n", value)

    resulting_parts = []
    for part in parts:
        part = recover_math_from_placeholders(part, placeholders)
        new_parts = [p.strip() for p in part.split("\n") if len(p.strip()) > 0]
        resulting_parts.extend(new_parts)

    return resulting_parts


def tidy_latex_environment(text):
    # Make sure every \begin{...} is followed by \n
    text = re.sub(r"\\begin\{([^}]*)\}", r"\\begin{\1}\n", text)
    # Make sure every \end{...} is preceded by \n
    text = re.sub(r"\\end\{([^}]*)\}", r"\n\\end{\1}", text)
    return text


def tidy_asymptote(text):
    # Make sure every [asy] is followed by \n
    text = re.sub(r"\[asy\]", r"[asy]\n", text)
    # Make sure every [/asy] is preceded by \n
    text = re.sub(r"\[/asy\]", r"\n[/asy]", text)
    return text


def merge_single_opening_to_previous_parts(parts):
    """Make sure if the entire part is a single opening, i.e. $, $$, \(, \[, \begin{...}, [asy],
    it is merged with the previous part (if exists)."""

    single_openings = [
        r"\$",
        r"\$\$",
        r"\\\(",
        r"\\\[",
        r"\\begin\{[^}]*\}",
        r"\[asy\]",
    ]

    parts = copy.deepcopy(parts)

    i = 0
    while i < len(parts):
        has_merged = False
        for opening in single_openings:
            if i >= len(parts):
                break
            if re.fullmatch(opening, parts[i]):
                if i > 0:
                    parts[i - 1] = parts[i - 1].rstrip() + " " + parts.pop(i).lstrip()
                    has_merged = True
                    break
                else:
                    # Merge with the next part
                    if opening in [r"\\begin\{[^}]*\}", r"\[asy\]"]:
                        continue
                    if i < len(parts) - 1:
                        parts[i + 1] = parts[i] + parts[i + 1]
                        parts.pop(i)
                        has_merged = True
                        break
        if not has_merged:
            i += 1

    return parts


def merge_single_closing_to_next_parts(parts):
    """Make sure if the entire part is a single closing, i.e. $, $$, \(, \], \end{...}, [/asy],
    it is merged with the next part (if exists)."""

    single_closings = [r"\$", r"\$\$", r"\\\)", r"\\\]", r"\\end\{[^}]*\}", r"\[/asy\]"]

    parts = copy.deepcopy(parts)

    i = 0
    while i < len(parts):
        has_merged = False
        for closing in single_closings:
            if i >= len(parts):
                break
            if re.fullmatch(closing, parts[i]):
                if i < len(parts) - 1:
                    parts[i + 1] = parts[i] + parts[i + 1]
                    parts.pop(i)
                    has_merged = True
                    break
                else:
                    # Merge with the previous part
                    if closing in [r"\\end\{[^}]*\}", r"\[/asy\]"]:
                        continue
                    if i > 0:
                        parts[i - 1] += parts.pop(i)
                        has_merged = True
                        break
        if not has_merged:
            i += 1

    return parts


def best_effort_break_long_part_in_language(part):
    # Try to split the part by punctuation

    # Disguise the math content
    # print(part)
    part, placeholders, placeholder_types = replace_math_with_placeholders(part)

    punctuation_patterns = [
        # Comma
        r",",
        r"\sand\s",
    ]
    pattern_lengths = [1, 4]

    new_parts = []
    for punc_pattern, pat_len in zip(punctuation_patterns, pattern_lengths):
        # Find all possible positions of the punctuation
        positions = [m.start() + pat_len for m in re.finditer(punc_pattern, part)]
        positions.sort()

        if len(positions) == 0:
            # Continue to the next punctuation pattern
            continue

        # Find the optimal division point
        min_diff = float("inf")
        optimal_pos = None

        for pos in positions:
            first_part_length = pos
            second_part_length = len(part) - pos
            diff = abs(first_part_length - second_part_length)

            if diff < min_diff:
                min_diff = diff
                optimal_pos = pos

        min_optimal_length = min(optimal_pos, len(part) - optimal_pos)
        if min_optimal_length < 20:
            # We don't want to split on short parts
            # Continue to the next punctuation pattern
            continue

        if optimal_pos is not None:
            new_parts.append(part[:optimal_pos].strip())
            new_parts.append(part[optimal_pos:].strip())
            break

    if len(new_parts) == 0:
        # If no split was successful, return the original part
        new_parts.append(part)

    # Recover the math content from the placeholders
    new_parts = [recover_math_from_placeholders(p, placeholders) for p in new_parts]

    return new_parts


def best_effort_break_long_part_in_math(part):
    split_patterns = [
        r"=",
        r"\\equiv",
        r"\+",
    ]
    split_patterns_len = [1, 6, 1]

    new_parts = []
    for split_pattern, pat_len in zip(split_patterns, split_patterns_len):
        # Find all possible positions of the split pattern
        positions = [m.start() + pat_len for m in re.finditer(split_pattern, part)]
        positions.sort()

        if len(positions) == 0:
            # Continue to the next split pattern
            continue

        # Find the optimal division point
        min_diff = float("inf")
        optimal_pos = None

        for pos in positions:
            first_part_length = pos
            second_part_length = len(part) - pos
            diff = abs(first_part_length - second_part_length)

            if diff < min_diff:
                min_diff = diff
                optimal_pos = pos

        min_optimal_length = min(optimal_pos, len(part) - optimal_pos)
        if min_optimal_length < 20:
            # We don't want to split on short parts
            # Continue to the next split pattern
            continue

        if optimal_pos is not None:
            new_parts.append(part[:optimal_pos])
            new_parts.append(part[optimal_pos:].lstrip())
            break

    if len(new_parts) == 0:
        # If no split was successful, return the original part
        new_parts.append(part)

    return new_parts


def try_to_break_very_long_parts(parts):
    resulting_parts = []
    for part in parts:
        if len(part) > 100:
            resulting_parts.extend(best_effort_break_long_part_in_language(part))
        else:
            resulting_parts.append(part)

    final_parts = []
    for part in resulting_parts:
        if len(part) > 100:
            final_parts.extend(best_effort_break_long_part_in_math(part))
        else:
            final_parts.append(part)

    return final_parts


def split_solution(solution_latex: str) -> List[str]:
    # First disguise the math content
    disguised_text, placeholders, placeholder_types = replace_math_with_placeholders(
        solution_latex
    )

    # Tidy the LaTeX placeholders
    for key, value in placeholders.items():
        if placeholder_types[key] == "environment":
            placeholders[key] = tidy_latex_environment(value)
        elif placeholder_types[key] == "asymptote":
            placeholders[key] = tidy_asymptote(value)

    # Split the disguised text into parts
    parts = naive_split_by_newline_and_period(disguised_text)

    # Split the parts that end with certain patterns
    parts = split_tailing_periods_in_placeholders(parts, placeholders)

    # Recover the math content from the placeholders
    parts = split_newline_in_placeholders(parts, placeholders)

    # Merge single openings to the previous parts
    parts = merge_single_opening_to_previous_parts(parts)

    # Merge single closings to the next parts
    parts = merge_single_closing_to_next_parts(parts)

    # Merge short parts with the next part
    # parts = merge_short_parts_forward(parts)

    # Try to break very long parts
    parts = try_to_break_very_long_parts(parts)

    return parts
