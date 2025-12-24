from .....testing.doctest_parser import run_doctest_examples, discover_functions_from_scope

from sympy.abc import a, b, c

import pytest

constrained_funcs = discover_functions_from_scope("triples.core.structsos.constrained")

@pytest.mark.slow
@pytest.mark.parametrize(
    "func",
    [_[2] for _ in constrained_funcs],
    ids = [f"{_[0]}:{_[1]}" for _ in constrained_funcs]
)
def test_doc_structsos_constrained(func):
    from ...structsos import StructuralSOS
    solver = lambda *args, **kwargs: \
        StructuralSOS(*args, **kwargs, raise_exception=True)

    run_doctest_examples(
        func,
        solver=solver,
        configs = {
            "ineqs": [],
            "gens": [a, b, c],
            "return_type": "poly",
        }
    )
