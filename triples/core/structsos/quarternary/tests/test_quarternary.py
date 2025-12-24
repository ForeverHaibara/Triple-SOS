from .....testing.doctest_parser import run_doctest_examples, discover_functions_from_scope

from sympy.abc import a, b, c, d

import pytest

quarternary_funcs = discover_functions_from_scope("triples.core.structsos.quarternary")

@pytest.mark.slow
@pytest.mark.parametrize(
    "func",
    [_[2] for _ in quarternary_funcs],
    ids = [f"{_[0]}:{_[1]}" for _ in quarternary_funcs]
)
def test_doc_structsos_quarternary(func):
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
