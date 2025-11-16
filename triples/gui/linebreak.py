import re

# --- Helper functions for parsing LaTeX (mostly unchanged from previous) ---
DELIM_PAIRS = {
    "\\left(": "\\right)", "\\left[": "\\right]", "\\left{": "\\right}",
    "\\left|": "\\right|", "\\left.": "\\right.",
    "(": ")",
    "[": "]",
}
OPENER_REGEX_PATTERNS = [
    r"\\left\(", r"\\left\[", r"\\left\{", r"\\left\|", r"\\left\.",
    r"\\frac",
    r"\(", r"\[",
]
OPENING_BRACKETS_FOR_DEPTH = ['(', '[', '{']
CLOSING_BRACKETS_FOR_DEPTH = [')', ']', '}']
ADDITIVE_TERMS_LEFT_DELIMS = {
    "\\left(": "\\right)", "\\left[": "\\right]", "\\left{": "\\right}",
    "\\left|": "\\right|", "\\left.": "\\right."
}

def _get_latex_argument_content(text, start_pos):
    if start_pos >= len(text): return "", start_pos
    char = text[start_pos]
    if char == '{':
        brace_depth = 0
        for i in range(start_pos, len(text)):
            if text[i] == '{' and (i == 0 or text[i-1] != '\\'): brace_depth += 1
            elif text[i] == '}' and (i == 0 or text[i-1] != '\\'):
                brace_depth -= 1
                if brace_depth == 0: return text[start_pos + 1 : i], i + 1
        raise ValueError(f"Unmatched brace starting at {start_pos} in '{text}'")
    elif char == '\\':
        match = re.match(r"\\[a-zA-Z]+(?:\s*[*!]?)?", text[start_pos:]) or \
                re.match(r"\\.", text[start_pos:])
        if match:
            token = match.group(0)
            return token, start_pos + len(token)
        return text[start_pos], start_pos + 1
    return text[start_pos], start_pos + 1

# --- Part 1: Identify Additive Expressions (unchanged from previous correct version) ---
def get_additive_terms(latex_str):
    terms = []
    buffer = ""
    depth_stack = []
    n_str = len(latex_str)
    idx = 0
    while idx < n_str:
        found_left_delim = False
        for l_delim_str, r_delim_str in ADDITIVE_TERMS_LEFT_DELIMS.items():
            if latex_str.startswith(l_delim_str, idx):
                depth_stack.append(r_delim_str)
                buffer += latex_str[idx : idx + len(l_delim_str)]
                idx += len(l_delim_str)
                found_left_delim = True
                break
        if found_left_delim: continue

        found_right_delim = False
        if depth_stack:
            expected_closer = depth_stack[-1]
            if expected_closer.startswith("\\right") and latex_str.startswith(expected_closer, idx):
                depth_stack.pop()
                buffer += latex_str[idx : idx + len(expected_closer)]
                idx += len(expected_closer)
                found_right_delim = True
        if found_right_delim: continue

        char = latex_str[idx]

        if char in OPENING_BRACKETS_FOR_DEPTH:
            if char == '(': depth_stack.append(')')
            elif char == '[': depth_stack.append(']')
            elif char == '{': depth_stack.append('}')
            buffer += char
            idx += 1
        elif char in CLOSING_BRACKETS_FOR_DEPTH:
            if depth_stack and depth_stack[-1] == char: depth_stack.pop()
            buffer += char
            idx += 1
        elif latex_str.startswith("\\frac", idx) and not depth_stack:
            buffer += char
            idx += 1
        elif char in ('+', '-') and not depth_stack:
            if buffer: terms.append(buffer.strip())
            buffer = char
            idx += 1
        else:
            buffer += char
            idx += 1

    if buffer: terms.append(buffer.strip())

    final_terms = []
    if not terms: return None

    if terms[0] == "" and len(terms) > 1:
        first_term_candidate = terms[1]
        if first_term_candidate.startswith('+') or first_term_candidate.startswith('-'):
            final_terms.append(first_term_candidate)
            start_idx_for_loop = 2
        else:
            final_terms.append(terms[0])
            final_terms.append(terms[1])
            start_idx_for_loop = 2
    elif terms[0] != "" and (terms[0].startswith('+') or terms[0].startswith('-')):
        final_terms.append(terms[0])
        start_idx_for_loop = 1
    else:
        final_terms.append(terms[0])
        start_idx_for_loop = 1

    for i in range(start_idx_for_loop, len(terms)):
        final_terms.append(terms[i])

    final_terms = [t for t in final_terms if t or t == terms[0]]
    if final_terms and final_terms[0] == "" and len(final_terms) > 1:
        final_terms.pop(0)

    return final_terms if len(final_terms) >= 2 else None


# --- Part 2: Wrap with aligned if needed (MODIFIED) ---
DEFAULT_MAX_TERMS_FOR_ALIGN = 2
DEFAULT_MAX_LEN_FOR_ALIGN = 160
DEFAULT_MAX_LINE_LEN_IN_ALIGNED = 100 # New default constant

def wrap_with_aligned_if_needed(latex_str,
                                 max_terms_trigger=DEFAULT_MAX_TERMS_FOR_ALIGN,
                                 max_len_trigger=DEFAULT_MAX_LEN_FOR_ALIGN,
                                 max_line_len=DEFAULT_MAX_LINE_LEN_IN_ALIGNED): # New parameter
    terms = get_additive_terms(latex_str)

    apply_aligned_wrapper = False
    if terms:
        if len(terms) > max_terms_trigger or len(latex_str) > max_len_trigger:
            apply_aligned_wrapper = True

    if not apply_aligned_wrapper:
        return latex_str

    if not terms: # Should be caught by apply_aligned_wrapper logic, but safeguard
        return latex_str

    output_lines_as_term_lists = [] # List of lists of terms
    current_line_term_group = []

    # Add first term to the first group
    current_line_term_group.append(terms[0].strip())

    for i in range(1, len(terms)):
        term_with_sign = terms[i].strip()

        # Construct the string for the current line *if this term were added*
        cur_line_length = sum(len(term) for term in current_line_term_group) if current_line_term_group else 0
        new_line_length = cur_line_length + len(term_with_sign) + (1 if current_line_term_group else 0)

        ratio_of_current = (cur_line_length + 1) / (max_line_len + 1) + (max_line_len + 1) / (cur_line_length + 1)
        ratio_of_extended = (new_line_length + 1) / (max_line_len + 1) + (max_line_len + 1) / (new_line_length + 1)
        # ratio_of_new_line = (len(term_with_sign) + 1) / (max_line_len + 1) + (max_line_len + 1) / (len(term_with_sign) + 1)

        if (cur_line_length > max_line_len or ratio_of_extended > ratio_of_current) and len(current_line_term_group) > 0:
            # The new term makes the line too long. Finalize the previous line.
            output_lines_as_term_lists.append(list(current_line_term_group))
            current_line_term_group = [term_with_sign] # Start new line with the current term
        else:
            # Term fits, add it to the current line group
            current_line_term_group.append(term_with_sign)

    # Add the last accumulated line group
    if current_line_term_group:
        output_lines_as_term_lists.append(list(current_line_term_group))

    # Format with \begin{aligned} and &
    aligned_body_parts = []
    for i, term_group in enumerate(output_lines_as_term_lists):
        # Construct line string: "term1" + " " + "+term2" ...
        line_str = term_group[0]
        for k in range(1, len(term_group)):
            line_str += " " + term_group[k]
        # if i == 0 and line_str.count("=") == 1:
        #     # TODO: Handle cases where there are multiple "="
        #     line_str = line_str.replace("=", "= & ")
        # else:
        #     # Each line starts with "& "
        line_str = "& " + line_str
        aligned_body_parts.append(line_str)

    if len(aligned_body_parts) == 1:
        return latex_str

    aligned_body = "\\\\\n".join(aligned_body_parts)

    return f"\\begin{{aligned}}{aligned_body}\\end{{aligned}}"


# --- Part 3: Recursive Formatting (MODIFIED to pass new param) ---
def _find_next_opener_info(latex_str, start_pos, opener_regex_patterns):
    first_match_pos = -1
    found_actual_opener, found_opener_regex = None, None
    for opener_pattern in opener_regex_patterns:
        try:
            match_iter = re.finditer(opener_pattern, latex_str[start_pos:])
            for match in match_iter:
                current_match_pos = start_pos + match.start()
                if first_match_pos == -1 or current_match_pos < first_match_pos:
                    first_match_pos, found_actual_opener, found_opener_regex = \
                        current_match_pos, match.group(0), opener_pattern
                break
        except re.error: pass
    return (first_match_pos, found_actual_opener, found_opener_regex) if first_match_pos != -1 else (-1, None, None)

def recursive_latex_auto_linebreak(latex_str,
                           max_terms_aligned=DEFAULT_MAX_TERMS_FOR_ALIGN,
                           max_len_aligned=DEFAULT_MAX_LEN_FOR_ALIGN,
                           max_line_len_in_aligned=DEFAULT_MAX_LINE_LEN_IN_ALIGNED): # New parameter
    processed_parts = []
    current_pos = 0
    n = len(latex_str)

    while current_pos < n:
        match_pos, actual_opener_text, opener_regex = _find_next_opener_info(latex_str, current_pos, OPENER_REGEX_PATTERNS)

        if match_pos == -1:
            processed_parts.append(latex_str[current_pos:])
            break

        if match_pos > current_pos:
            processed_parts.append(latex_str[current_pos:match_pos])

        processed_parts.append(actual_opener_text)
        current_pos = match_pos + len(actual_opener_text)

        if opener_regex == r"\\frac":
            num_content, end_pos_num = _get_latex_argument_content(latex_str, current_pos)
            # Pass all formatting parameters recursively
            formatted_num = recursive_latex_auto_linebreak(num_content, max_terms_aligned, max_len_aligned, max_line_len_in_aligned)
            processed_parts.append("{" + formatted_num + "}")
            current_pos = end_pos_num

            den_content, end_pos_den = _get_latex_argument_content(latex_str, current_pos)
            formatted_den = recursive_latex_auto_linebreak(den_content, max_terms_aligned, max_len_aligned, max_line_len_in_aligned)
            processed_parts.append("{" + formatted_den + "}")
            current_pos = end_pos_den
        else:
            literal_closer = DELIM_PAIRS.get(actual_opener_text)
            if not literal_closer:
                continue

            content_start_idx = current_pos
            level, search_idx, end_delim_start_idx = 0, content_start_idx, -1

            while search_idx < n:
                if latex_str.startswith(actual_opener_text, search_idx):
                    level += 1
                    search_idx += len(actual_opener_text)
                elif latex_str.startswith(literal_closer, search_idx):
                    if level == 0:
                        end_delim_start_idx = search_idx
                        break
                    level -= 1
                    search_idx += len(literal_closer)
                elif latex_str.startswith("\\" + actual_opener_text, search_idx):
                    search_idx += len("\\" + actual_opener_text)
                elif latex_str.startswith("\\" + literal_closer, search_idx):
                    search_idx += len("\\" + literal_closer)
                else:
                    search_idx += 1

            if end_delim_start_idx != -1:
                content = latex_str[content_start_idx : end_delim_start_idx]
                # Pass all formatting parameters recursively
                formatted_content = recursive_latex_auto_linebreak(content, max_terms_aligned, max_len_aligned, max_line_len_in_aligned)
                processed_parts.append(formatted_content)
                processed_parts.append(literal_closer)
                current_pos = end_delim_start_idx + len(literal_closer)
            else:
                pass

    final_reconstructed_str = "".join(processed_parts)
    # Pass all formatting parameters to the final wrap
    return wrap_with_aligned_if_needed(final_reconstructed_str, max_terms_aligned, max_len_aligned, max_line_len_in_aligned)


# --- Example Usage ---
if __name__ == '__main__':
    print("--- Part 1: get_additive_terms (sanity check) ---")
    s1 = "4a+2b-3c/5+(c-a)^2/2"
    print(f"'{s1}' -> {get_additive_terms(s1)}")
    s_long_sum = "a+b+c+d+e+f+g+h+i"
    print(f"'{s_long_sum}' -> {get_additive_terms(s_long_sum)}")


    print("\n--- Part 2: wrap_with_aligned_if_needed (with new line length logic) ---")
    expr_to_wrap = "termA+termB+termC+termD+termE+termF+termG"
    print(f"\nInput: '{expr_to_wrap}'")
    print(f"Wrapped (max_terms=2, max_len=20, max_line_len=15):") # max_terms=2 means 3+ terms trigger align
    print(wrap_with_aligned_if_needed(expr_to_wrap, max_terms_trigger=2, max_len_trigger=20, max_line_len=15))
    # termA +termB (len approx 6+1+6=13) -> fits
    # termA +termB +termC (len approx 13+1+6=20) -> exceeds 15. Break.
    # Expected:
    # & termA +termB \\
    # & +termC +termD \\
    # & +termE +termF \\
    # & +termG

    print(f"\nInput: '{s_long_sum}' (a+b+c+d+e+f+g+h+i)")
    print(f"Wrapped (max_terms=2, max_len=5, max_line_len=10):") # triggers align, line length 10
    print(wrap_with_aligned_if_needed(s_long_sum, max_terms_trigger=2, max_len_trigger=5, max_line_len=10))
    # a (1)
    # a +b (1+1+2=4)
    # a +b +c (4+1+2=7)
    # a +b +c +d (7+1+2=10)
    # a +b +c +d +e (10+1+2=13) > 10. Break.
    # Line 1: a +b +c +d
    # New line starts with +e
    # +e (2)
    # +e +f (2+1+2=5)
    # +e +f +g (5+1+2=8)
    # +e +f +g +h (8+1+2=11) > 10. Break.
    # Line 2: +e +f +g
    # New line starts with +h
    # +h (2)
    # +h +i (2+1+2=5)
    # Line 3: +h +i
    # Expected:
    # & a +b +c +d\\
    # & +e +f +g\\
    # & +h +i

    print("\n--- Part 3: recursive_latex_auto_linebreak (passing new param) ---")
    # Test with a long expression inside parentheses, requiring internal alignment with line length
    long_paren_expr = "f(x) = (a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q+r) + (s+t+u+v+w) - 2(x+y+z)"
    print(f"\nInput: '{long_paren_expr}'")
    print(f"Recursively formatted (max_terms_aligned=2, max_len_aligned=30, max_line_len_in_aligned=20):")
    # First parenthesis content: "a+b+...+r" (18 terms) -> will be wrapped by aligned
    #   Inside that aligned: lines should be <= 20 chars.
    #   a +b +c +d +e +f +g +h +i (9 terms, length ~ 9*2 + 8*1 = 26) -> should break
    #   Example: a +b +c +d +e +f +g +h (8 terms, 8*2+7 = 23)
    #            a +b +c +d +e +f +g (7 terms, 7*2+6 = 20) -> fits 20
    # Outer expression: 3 terms: "(...)" + "(...)" - "2(...)" -> may be wrapped by aligned if total > 30 or terms > 2
    print(recursive_latex_auto_linebreak(long_paren_expr, max_terms_aligned=2, max_len_aligned=30, max_line_len_in_aligned=20))

    print("-" * 20)
    nested_frac_long_num = "E = \\frac{X_1+X_2+X_3+X_4+X_5+X_6+X_7+X_8+X_9+X_{10}}{k+l+m} + Y"
    print(f"\nInput: '{nested_frac_long_num}'")
    print(f"Recursively formatted (max_terms_aligned=1, max_len_aligned=10, max_line_len_in_aligned=18):")
    # Numerator has 10 terms. max_terms_aligned=1 means >=2 terms will trigger aligned.
    # max_line_len_in_aligned=18 for numerator's lines.
    # X_1 +X_2 +X_3 (len ~3+1+3+1+3 = 11)
    # X_1 +X_2 +X_3 +X_4 (len ~11+1+3 = 15)
    # X_1 +X_2 +X_3 +X_4 +X_5 (len ~15+1+3 = 19) > 18. Break.
    # Line 1: X_1 +X_2 +X_3 +X_4
    print(recursive_latex_auto_linebreak(nested_frac_long_num, max_terms_aligned=1, max_len_aligned=10, max_line_len_in_aligned=18))
