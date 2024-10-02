import re
from typing import List, Tuple, Dict

from treetune.tasks.math_extract_steps import md5_hash

MAX_PART_LENGTH = 100
PLACEHOLDER_LENGTH = 60


def has_placeholders(text):
    return "<|PLACEHOLDER_" in text


def find_index_of_all_placeholders(text) -> list[int]:
    return [
        m.start()
        for m in re.finditer(r"<\|PLACEHOLDER_[0-9a-f]{32}_PLACEHOLDER\|>", text)
    ]


def replace_math_with_placeholders_corrected(
    latex_content: str,
) -> Tuple[str, Dict[str, str], Dict[str, str]]:
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


def naive_split_by_newline_and_period(disguised_latex: str) -> List[int]:
    # Further refined regex pattern for splitting text into sentences
    period_pattern = r"(?<!\b[A-Za-z]\.)(?<!\s[A-Za-z]\.)(?<!\s[A-Za-z][A-Za-z]\.)(?<=\.)(?=\s+[A-Z])"
    parts = re.split(period_pattern, disguised_latex)
    # parts = [part.strip() for part in parts if len(part.strip()) > 0]

    final_parts = []
    for part in parts:
        parts_new = part.split("\n")
        # Create new parts where "\n" is added to beginning of the parts except the first part
        for i, part_new in enumerate(parts_new):
            if i == 0:
                final_parts.append(part_new)
            else:
                final_parts.append("\n" + part_new)

    # Merge single \n with the next part
    i = 0
    while i < len(final_parts) - 1:
        if final_parts[i] == "\n":
            final_parts[i + 1] = final_parts[i] + final_parts[i + 1]
            final_parts.pop(i)
        elif not final_parts[i].endswith(".") and len(final_parts[i]) < 20:
            # if i != 0:
            # assert final_parts[i] == "" or final_parts[i].startswith("\n"), "Short part should start with \n"
            assert final_parts[i + 1].startswith("\n"), "Next part should start with \n"
            final_parts[i] = final_parts[i] + final_parts.pop(i + 1)
        else:
            i += 1

    # Make sure we didn't cut into PLACEHOLDER's
    for p in final_parts:
        if "<|PLACEHOLDER_" in p:
            assert "PLACEHOLDER|>" in p, "Cut into a placeholder"

    indices = [0]  # Store the indices of the end of each part
    for part in final_parts:
        indices.append(indices[-1] + len(part))
    return indices


def split_tailing_periods_in_placeholders(
    text: str, indices: List[int], placeholders: Dict[str, str]
) -> List[int]:
    assert indices[-1] == len(text), "Last index should be the length of the text"

    # Define the points where we want to split
    tail_patterns = [r".$", r".$$", r".\)", r".\]"]

    parts = [text[indices[i] : indices[i + 1]] for i in range(len(indices) - 1)]

    new_indices = [0]
    for part in parts:
        split_points = []
        for start_idx in find_index_of_all_placeholders(part):
            ph_text = part[start_idx : start_idx + PLACEHOLDER_LENGTH]
            actual_text = placeholders[ph_text]
            for pattern in tail_patterns:
                if actual_text.endswith(pattern):
                    split_points.append(start_idx + PLACEHOLDER_LENGTH)
                    break

        split_points = split_points + [len(part)]
        new_indices += [new_indices[-1] + i for i in split_points]

    # Merge short parts with the next part
    i = 0
    while i < len(new_indices) - 1:
        part = text[new_indices[i] : new_indices[i + 1]]
        part_no_ph = recover_math_from_placeholders(part, placeholders)
        has_period = any([part_no_ph.endswith(pattern) for pattern in tail_patterns])
        if not has_period and len(part_no_ph) < 20:
            if i == len(new_indices) - 2:
                new_indices.pop(i)
            else:
                new_indices.pop(i + 1)
        else:
            i += 1

    resulting_parts = [
        text[new_indices[i] : new_indices[i + 1]] for i in range(len(new_indices) - 1)
    ]
    for part in resulting_parts:
        if "<|PLACEHOLDER_" in part:
            assert "PLACEHOLDER|>" in part, "Cut into a placeholder"

    return new_indices


def split_newline_in_placeholders(
    text: str, indices: List[int], placeholders: Dict[str, str]
) -> Tuple[List[int], str]:
    assert indices[-1] == len(text), "Last index should be the length of the text"

    parts = [text[indices[i] : indices[i + 1]] for i in range(len(indices) - 1)]

    new_indices = [0]
    for part in parts:
        split_points = []

        while True:
            ph_positions = find_index_of_all_placeholders(part)
            if len(ph_positions) == 0:
                break

            for start_idx in ph_positions:
                ph_text = part[start_idx : start_idx + PLACEHOLDER_LENGTH]
                actual_text = placeholders[ph_text]
                # indices of all newlines in the actual text
                newline_indices = [
                    start_idx + m.start() for m in re.finditer(r"\n", actual_text)
                ]
                split_points += newline_indices
                part = part.replace(ph_text, actual_text)
                break

        split_points += [len(part)]
        new_indices += [new_indices[-1] + i for i in split_points]

    recovered_text = recover_math_from_placeholders(text, placeholders)

    # Merge short parts with the next part and merge environment opening with the next part
    single_openings = [
        r"\$",
        r"\$\$",
        r"\\\(",
        r"\\\[",
        r"\\begin\{[^}]*\}",
        r"\[asy\]",
    ]
    i = 0
    while i < len(new_indices) - 1:
        part = recovered_text[new_indices[i] : new_indices[i + 1]]
        is_single_opening = any(
            [re.fullmatch(opening, part.strip()) for opening in single_openings]
        )
        if is_single_opening or len(part) < 3 or part == "\n":
            if i == len(new_indices) - 2:
                new_indices.pop(i)
            else:
                new_indices.pop(i + 1)
        else:
            i += 1

    return new_indices, recovered_text


def best_effort_break_long_part_in_language(part: str) -> List[int]:
    # Try to split the part by punctuation

    # Disguise the math content
    orig_part = part
    part, placeholders, placeholder_types = replace_math_with_placeholders_corrected(
        part
    )

    punctuation_patterns = [
        r",",
        r"\sand\s",
        r":",
        r"\sto\s",
    ]
    pattern_lengths = [1, 4, 1, 3]

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

        min_optimal_length = min(
            len(recover_math_from_placeholders(part[:optimal_pos], placeholders)),
            len(recover_math_from_placeholders(part[optimal_pos:], placeholders)),
        )
        if min_optimal_length < 40:
            # We don't want to split on short parts
            # Continue to the next punctuation pattern
            continue

        if optimal_pos is not None:
            new_parts.append(part[:optimal_pos])
            new_parts.append(part[optimal_pos:])
            break

    if len(new_parts) == 0:
        # If no split was successful, return the original part
        new_parts.append(part)

    # Recover the math content from the placeholders
    new_parts = [recover_math_from_placeholders(p, placeholders) for p in new_parts]

    assert "".join(new_parts) == orig_part, "Parts should be equal to the original part"

    indices = [0]
    for new_part in new_parts:
        indices.append(indices[-1] + len(new_part))

    return indices


def best_effort_break_long_part_in_math(part: str) -> List[int]:
    split_patterns = [
        r",\$",
        r",\\\)",
        r",\\\]",
        r"=",
        r"\\equiv",
        r"\+",
        r"\\cancelto",
    ]
    split_patterns_len = [
        2,
        3,
        3,
        1,
        6,
        1,
        9,
    ]

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
            new_parts.append(part[optimal_pos:])
            break

    if len(new_parts) == 0:
        # If no split was successful, return the original part
        new_parts.append(part)

    assert "".join(new_parts) == part, "Parts should be equal to the original part"

    indices = [0]
    for new_part in new_parts:
        indices.append(indices[-1] + len(new_part))

    return indices


def break_part_as_much_as_possible(part: str) -> List[int]:
    indices = [0, len(part)]

    split_with_math = [-1, 0]

    def is_in_math_range(idx):
        return split_with_math[0] <= idx < split_with_math[1]

    def update_math_range(begin, end):
        if split_with_math[0] == -1:
            split_with_math[0] = begin
        else:
            split_with_math[0] = min(split_with_math[0], begin)
        split_with_math[1] = max(split_with_math[1], end)

    i = 0
    while i < len(indices) - 1:
        sub_part = part[indices[i] : indices[i + 1]]
        if len(sub_part) > MAX_PART_LENGTH:
            # Try to break the part
            if not is_in_math_range(indices[i]):
                new_indices = best_effort_break_long_part_in_language(sub_part)
                assert new_indices[0] == 0 and new_indices[-1] == len(sub_part)
                if len(new_indices) > 2:
                    new_indices = [indices[i] + idx for idx in new_indices]
                    indices = indices[: i + 1] + new_indices[1:-1] + indices[i + 1 :]
                    continue

            new_indices = best_effort_break_long_part_in_math(sub_part)
            assert new_indices[0] == 0 and new_indices[-1] == len(sub_part)
            if len(new_indices) > 2:
                update_math_range(new_indices[1], new_indices[-1])
                new_indices = [indices[i] + idx for idx in new_indices]
                indices = indices[: i + 1] + new_indices[1:-1] + indices[i + 1 :]
                continue

            # Couldn't break the part, move to the next part
            i += 1
        else:
            i += 1

    return indices


def try_to_break_very_long_parts(text: str, indices: List[int]) -> List[int]:
    assert indices[-1] == len(text), "Last index should be the length of the text"

    parts = [text[indices[i] : indices[i + 1]] for i in range(len(indices) - 1)]

    new_indices = [0]
    for part in parts:
        if len(part) > 100:
            part_indices = break_part_as_much_as_possible(part)
            new_indices.extend([new_indices[-1] + idx for idx in part_indices[1:]])
        else:
            new_indices.append(new_indices[-1] + len(part))

    return new_indices


def split_solution_inplace(reasoning_latex: str) -> List[int]:
    # First disguise the math content
    disguised_text, placeholders, placeholder_types = (
        replace_math_with_placeholders_corrected(reasoning_latex)
    )

    # Split the disguised text into parts
    indices = naive_split_by_newline_and_period(disguised_text)

    # Split the parts that end with certain patterns
    indices = split_tailing_periods_in_placeholders(
        disguised_text, indices, placeholders
    )

    # Recover the math content from the placeholders
    indices, text = split_newline_in_placeholders(disguised_text, indices, placeholders)

    # Try to break very long parts
    indices = try_to_break_very_long_parts(text, indices)
    assert text == reasoning_latex
    assert indices[-1] == len(text)

    return indices
