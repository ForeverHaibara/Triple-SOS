import pytest
from sympy.abc import a, b, c, d

from .....testing.doctest_parser import (
    discover_functions_from_scope,
    run_doctest_examples,
)

quaternary_funcs = discover_functions_from_scope("triples.core.structsos.quaternary")

@pytest.mark.slow
@pytest.mark.parametrize(
    "func",
    [_[2] for _ in quaternary_funcs],
    ids = [f"{_[0]}:{_[1]}" for _ in quaternary_funcs]
)
def test_doc_structsos_quaternary(func):
    from ...structsos import StructuralSOS
    solver = lambda *args, **kwargs: \
        StructuralSOS(*args, **kwargs, raise_exception=True)

    run_doctest_examples(
        func,
        solver=solver,
        configs = {
            "ineqs": [a, b, c, d],
            "gens": [a, b, c, d],
            "return_type": "poly",
        }
    )
