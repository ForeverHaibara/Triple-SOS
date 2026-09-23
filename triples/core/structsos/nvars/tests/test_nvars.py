from functools import partial

import pytest

from .....testing.doctest_parser import (
    discover_functions_from_scope,
    run_doctest_examples,
)
from .....utils import preprocess_text

nvars_funcs = discover_functions_from_scope("triples.core.structsos.nvars")

@pytest.mark.slow
@pytest.mark.parametrize(
    "func",
    [_[2] for _ in nvars_funcs],
    ids = [f"{_[0]}:{_[1]}" for _ in nvars_funcs]
)
def test_doc_structsos_nvars(func):
    from ...structsos import StructuralSOS
    solver = lambda *args, **kwargs: \
        StructuralSOS(*args, **kwargs, raise_exception=True)

    run_doctest_examples(
        func,
        solver=solver,
        configs = {
            "return_type": "poly",
            "parser": partial(
                preprocess_text, preserve_patterns=("sqrt", "sin", "cos", "pi"))
        }
    )
