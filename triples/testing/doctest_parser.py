from typing import List, Tuple, Any, Dict, Union, Callable, Optional
import ast
import os
import types
import importlib
import pkgutil
import inspect

BRACKET_PAIRS = {"(": ")", "[": "]", "{": "}"}

FuncLocator = Tuple[str, str, Callable]

def discover_functions_from_module(
    module: types.ModuleType,
    selector: Optional[Callable[[types.ModuleType, str, Callable], bool]] = None
) -> List[FuncLocator]:
    """
    Return a list of (module.__name__, func_name) for functions in `module`
    that satisfy selector(module, func_name). The function only inspects
    function names and docstrings (no parsing of examples).
    """
    sel = selector or (lambda mod, fname, obj: "=>" in (obj.__doc__ or ""))
    results: List[FuncLocator] = []
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if sel(module, name, obj) and module.__name__ == getattr(obj, "__module__", None):
            results.append((module.__name__, name, obj))
    return results


def discover_functions_from_scope(
    scope: str,
    selector: Optional[Callable[[types.ModuleType, str], bool]] = None,
    *,
    recursive_pkg: bool = True
) -> List[FuncLocator]:
    """
    Discover functions matching selector under a scope.

    - scope may be:
      * a dotted package/module string (e.g. 'triples.core'), OR
      * a filesystem directory path (it will import .py files under it), OR
      * a filesystem file path to a single .py file.

    The function only imports modules and collects function names (cheap relative to parsing examples).
    """
    results: List[FuncLocator] = []

    # If scope is a path to a .py file
    if os.path.isfile(scope) and scope.endswith(".py"):
        spec_name = f"_docloc_{os.path.splitext(os.path.basename(scope))[0]}"
        spec = importlib.util.spec_from_file_location(spec_name, scope)
        module = importlib.util.module_from_spec(spec)
        loader = spec.loader
        assert loader is not None
        loader.exec_module(module)
        results.extend(discover_functions_from_module(module, selector=selector))
        return results

    # If scope is directory: import each .py file under it
    if os.path.isdir(scope):
        for dirpath, _, filenames in os.walk(scope):
            for fname in filenames:
                if not fname.endswith(".py"):
                    continue
                fpath = os.path.join(dirpath, fname)
                # import by path (use stable name derived from path)
                try:
                    rel = os.path.relpath(fpath, scope)
                    mod_hint = rel.replace(os.path.sep, ".").rsplit(".py", 1)[0]
                    spec_name = f"_doccol_{mod_hint}"
                    spec = importlib.util.spec_from_file_location(spec_name, fpath)
                    module = importlib.util.module_from_spec(spec)
                    loader = spec.loader
                    assert loader is not None
                    loader.exec_module(module)
                    results.extend(discover_functions_from_module(module, selector=selector))
                except Exception:
                    # ignore import errors here (user can run specific modules manually if needed)
                    continue
        return results

    # Otherwise treat as dotted package/module name
    try:
        pkg = importlib.import_module(scope)
    except Exception:
        # if import fails, return empty list instead of raising
        return results

    # include the package/module itself
    results.extend(discover_functions_from_module(pkg, selector=selector))

    # Walk submodules if it's a package
    if hasattr(pkg, "__path__") and recursive_pkg:
        for finder, name, ispkg in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            try:
                m = importlib.import_module(name)
                results.extend(discover_functions_from_module(m, selector=selector))
            except Exception:
                # skip submodule import failures
                continue

    # remove duplicates by key[2]
    # results = list({k[2]: k for k in results}.values())

    return results


def _split_top_level(s: str, sep: str = ",") -> List[str]:
    """
    Split string `s` by `sep` occurrences that are at top-level,
    i.e. not nested inside any brackets or string literals.
    Uses a single unified stack-driven algorithm for all bracket types.
    """
    parts = []
    cur = []
    stack = []  # stack of expected closing brackets
    in_single = False
    in_double = False
    esc = False

    for ch in s:
        if esc:
            cur.append(ch)
            esc = False
            continue
        if ch == "\\":
            cur.append(ch)
            esc = True
            continue
        if in_single:
            cur.append(ch)
            if ch == "'" and not esc:
                in_single = False
            continue
        if in_double:
            cur.append(ch)
            if ch == '"' and not esc:
                in_double = False
            continue

        # not in string
        if ch == "'":
            in_single = True
            cur.append(ch)
            continue
        if ch == '"':
            in_double = True
            cur.append(ch)
            continue

        # opening bracket -> push expected closing
        if ch in BRACKET_PAIRS:
            stack.append(BRACKET_PAIRS[ch])
            cur.append(ch)
            continue
        # closing bracket -> pop if matches top, otherwise keep (tolerant)
        if stack and ch == stack[-1]:
            stack.pop()
            cur.append(ch)
            continue
        if ch == sep and not stack and not in_single and not in_double:
            parts.append("".join(cur))
            cur = []
            continue
        cur.append(ch)

    if cur:
        parts.append("".join(cur))
    return parts


def _find_top_level_char(s: str, char: str) -> int:
    """
    Return the index of the first occurrence of `char` that is at top-level
    (i.e., not inside brackets or string literals). Return -1 if not found.
    """
    stack = []
    in_single = False
    in_double = False
    esc = False

    for i, ch in enumerate(s):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if in_single:
            if ch == "'" and not esc:
                in_single = False
            continue
        if in_double:
            if ch == '"' and not esc:
                in_double = False
            continue
        if ch == "'":
            in_single = True
            continue
        if ch == '"':
            in_double = True
            continue
        if ch in BRACKET_PAIRS:
            stack.append(BRACKET_PAIRS[ch])
            continue
        if stack and ch == stack[-1]:
            stack.pop()
            continue
        if ch == char and not stack and not in_single and not in_double:
            return i
    return -1


def _strip_outer_parens(s: str) -> str:
    """
    If the entire string `s` is enclosed by a single matching pair of parentheses/brackets,
    strip that outermost pair and return the inner substring (stripped). Otherwise return s.strip().
    """
    s = s.strip()
    if not s:
        return s
    open_ch = s[0]
    if open_ch not in BRACKET_PAIRS:
        return s
    expected_close = BRACKET_PAIRS[open_ch]
    if s[-1] != expected_close:
        return s
    # Check whether the opening bracket at position 0 closes only at the last character.
    depth = 0
    for i, ch in enumerate(s):
        if ch == open_ch:
            depth += 1
        elif ch == expected_close:
            depth -= 1
            # If depth becomes zero before the last index, then outer bracket does not wrap entire string
            if depth == 0 and i != len(s) - 1:
                return s
    # If we reach here, outer bracket encloses whole string
    return s[1:-1].strip()


def _try_literal_eval(token: str) -> Any:
    """
    Try to evaluate token as a Python literal using ast.literal_eval.
    Raise the exception to the caller if parsing fails.
    """
    return ast.literal_eval(token)


def parse_ident_list_like(s: str) -> Union[List[str], Tuple[str, ...], set]:
    """
    Parse a bracketed identifier collection like "[a,b,c]", "(a,b)", "{a,b}" into:
      - list for square brackets,
      - tuple for parentheses,
      - set for curly braces (unless it looks like a dict: contains ':' at top-level).
    The function does NOT support dict parsing and will raise ValueError for dict-like inputs.
    """
    s = s.strip()
    if len(s) < 2:
        raise ValueError("not a bracketed sequence")
    open_ch = s[0]
    if open_ch not in BRACKET_PAIRS:
        raise ValueError("not a bracketed sequence")
    close_ch = BRACKET_PAIRS[open_ch]
    if s[-1] != close_ch:
        raise ValueError("mismatched brackets")

    inner = s[1:-1].strip()
    if inner == "":
        if open_ch == "[":
            return []
        if open_ch == "(":
            return tuple()
        if open_ch == "{":
            return set()

    # Quick detect for dict-like content in curly braces: top-level ':' present => dict
    if open_ch == "{":
        if _find_top_level_char(inner, ":") != -1:
            raise ValueError("dict-like content not supported by parse_ident_list_like")

    items = _split_top_level(inner, sep=",")
    parsed_items = [it.strip() for it in items if it.strip() != ""]

    if open_ch == "[":
        return parsed_items
    if open_ch == "(":
        return tuple(parsed_items)
    # curly braces -> set
    return set(parsed_items)


def parse_example_line(line: str, parse_ident_list: bool = True
                       ) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Parse a single example line.
    Returns (args, kwargs) where args[0] is the expression string (outer parens stripped).
    Positional args and keyword values are attempted to be parsed with ast.literal_eval;
    if parsing fails, the raw string is preserved unless parse_ident_list is True and
    the token looks like a bracketed identifier collection (e.g., [a,b,c] or (a,b) or {a,b}).
    """
    rest = line.strip()
    parts = _split_top_level(rest, sep=",")
    if not parts:
        raise ValueError("no content")

    # First part is the expression; add it as args[0]
    expr_raw = parts[0].strip()
    expr = expr_raw
    args = [expr]
    kwargs = {}

    for token in parts[1:]:
        tok = token.strip()
        if tok == "":
            continue
        eq_pos = _find_top_level_char(tok, "=")
        if eq_pos != -1:
            key = tok[:eq_pos].strip()
            val_raw = tok[eq_pos + 1 :].strip()
            parsed_val: Any
            try:
                parsed_val = _try_literal_eval(val_raw)
            except Exception:
                if parse_ident_list:
                    try:
                        parsed_val = parse_ident_list_like(val_raw)
                    except Exception:
                        parsed_val = val_raw
                else:
                    parsed_val = val_raw
            kwargs[key] = parsed_val
        else:
            # positional argument
            try:
                parsed_val = _try_literal_eval(tok)
            except Exception:
                if parse_ident_list:
                    try:
                        parsed_val = parse_ident_list_like(tok)
                    except Exception:
                        parsed_val = tok
                else:
                    parsed_val = tok
            args.append(parsed_val)

    return args, kwargs


def parse_directive(line: str):
    import re
    parts = re.split(r",(?![^\[\]]*\])", line)
    result = {}
    for part in parts:
        if "=" not in part:
            raise ValueError(f"Invalid directive: {part}")
        key, val_raw = map(str.strip, part.split("=", 1))
        try:
            parsed_val = _try_literal_eval(val_raw)
        except Exception:
            try:
                parsed_val = parse_ident_list_like(val_raw)
            except Exception:
                parsed_val = val_raw
        result[key] = parsed_val
    return result


def _collect_doctest_examples_raw(doc: str,
        set_environ=lambda x: None, parser=lambda *a,**k: (a,k)):
    lines = [_.strip() for _ in doc.splitlines()]
    lines = [_ for _ in lines if _.startswith('=>') or _.startswith('::')]
    cases = []
    for line in lines:
        if not (line.startswith('=>') or line.startswith('::')):
            continue

        # Although it can be implemented more carefully,
        # the current is enough
        idx = line.find('#')
        if idx >= 0:
            if 'doctest:+SKIP' in line[idx:].replace(' ',''):
                continue
            line = line[:idx]

        if line.startswith('=>'):
            args, kwargs = parse_example_line(line[3:])
            cases.append((line, parser(*args, **kwargs)))
        elif line.startswith('::'):
            directive = parse_directive(line[2:])
            set_environ(directive)
    return cases


def collect_doctest_examples(doc: str, configs: dict = {}) -> List[Tuple[str, Tuple[tuple, dict]]]:
    from sympy import Symbol
    from sympy.combinatorics.permutations import Permutation
    from sympy.combinatorics.perm_groups import PermutationGroup

    from ..utils.text_process import preprocess_text
    _configs = {
        "ineqs": [],
        "eqs": [],
        "gens": [Symbol(_) for _ in "abc"],
        "sym": "cyc",
        "return_type": "expr",
        "parser": preprocess_text,
    }
    _configs.update(configs)
    configs = _configs

    def parse_gens(gens):
        if gens is None:
            return configs["gens"]
        return [Symbol(_) for _ in gens]

    def parse_sym(sym):
        if sym is None:
            return configs["sym"]
        if isinstance(sym, list):
            return PermutationGroup(*[Permutation(_) for _ in sym])
        return sym

    def parse_expr(expr, gens, symmetry, return_type="expr"):
        if not isinstance(expr, str):
            return expr
        return configs["parser"](expr, gens, symmetry, return_type=return_type)

    def parse_ineqs(ineqs, gens, sym, return_type="expr"):
        if ineqs is None:
            return configs["ineqs"]
        return [parse_expr(_, gens, sym, return_type=return_type) for _ in ineqs]

    def parse_eqs(eqs, gens, sym, return_type="expr"):
        if eqs is None:
            return configs["eqs"]
        return [parse_expr(_, gens, sym, return_type=return_type) for _ in eqs]

    def set_environ(kwargs):
        if "gens" in kwargs:
            configs["gens"] = parse_gens(kwargs["gens"])
        if "sym" in kwargs:
            configs["sym"] = parse_sym(kwargs["sym"])
        if "parser" in kwargs:
            configs["parser"] = kwargs["parser"]
        if "ineqs" in kwargs:
            configs["ineqs"] = parse_ineqs(kwargs["ineqs"], configs["gens"], configs["sym"])
        if "eqs" in kwargs:
            configs["eqs"] = parse_eqs(kwargs["eqs"], configs["gens"], configs["sym"])

    def single_parser(expr, ineqs=None, eqs=None, gens=None, sym=None, **kwargs):
        gens = parse_gens(gens)
        sym = parse_sym(sym)
        return_type = kwargs.get("return_type", configs["return_type"])
        expr = parse_expr(expr, gens, sym, return_type=return_type)
        ineqs = parse_ineqs(ineqs, gens, sym, return_type=return_type)
        eqs = parse_eqs(eqs, gens, sym, return_type=return_type)
        return (expr, ineqs, eqs), {}

    return _collect_doctest_examples_raw(doc, set_environ, single_parser)


def solution_checker(solution, expr, ineq_constraints, eq_constraints,
        *args, **kwargs) -> bool:
    if solution is None:
        return False

    from sympy import sympify, Eq
    expr = sympify(expr).as_expr()
    sol = solution.solution

    fs1 = expr.free_symbols
    fs2 = sol.free_symbols

    assert len(fs2 - fs1) == 0, f"Unexpected variables in solution: {fs2 - fs1}"

    symbols = fs1.union(fs2)

    if len(symbols) > 10:
        from sympy import prime
        samples = [
            [prime(7*n+1) for n in range(1, len(symbols) + 1)],
            [prime(11*n+1) for n in range(1, len(symbols) + 1)]
        ]
    else:
        samples = (
            (19, 47, 79, 109, 151, 191, 229, 269, 311, 353)[:len(symbols)],
            (37, 83, 139, 197, 263, 331, 397, 461, 541, 607)[:len(symbols)]
        )
    for sample in samples:
        vals = dict(zip(symbols, sample))
        if Eq((sol.xreplace(vals) - expr.xreplace(vals)).simplify(), 0).simplify() != True:
            return False
    return True


def run_doctest_examples(func, solver, checker=solution_checker, configs = {}):
    doc = (func.__doc__ or "") if not isinstance(func, str) else func

    examples = collect_doctest_examples(doc, configs)

    for line, (args, kwargs) in examples:
        sol = None
        try:
            sol = solver(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Solver raised {type(e)} at line {line}\n"\
                    + f"args = {args}\nkwargs = {kwargs}") from e
        try:
            check = checker(sol, *args, **kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Checker raised {type(e)} at line {line}\n"\
                    + f"args = {args}\nkwargs = {kwargs}") from e
        if not check:
            raise AssertionError(
                f"Checker returned False at line {line}\n"\
                    + f"args = {args}\nkwargs = {kwargs}\nsolution = {sol}"
            )
