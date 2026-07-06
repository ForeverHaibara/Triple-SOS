from .....testing.doctest_parser import run_doctest_examples, discover_functions_from_scope

from sympy import Function
from sympy.abc import a, b, c, u, v, w

from ...structsos import StructuralSOS
from .....utils import pl, CyclicSum
from .....testing.doctest_parser import solution_checker

import pytest

ternary_funcs = discover_functions_from_scope("triples.core.structsos.ternary")

@pytest.mark.slow
@pytest.mark.parametrize(
    "func",
    [_[2] for _ in ternary_funcs],
    ids = [f"{_[0]}:{_[1]}" for _ in ternary_funcs]
)
def test_doc_structsos_ternary(func):
    solver = lambda *args, **kwargs: \
        StructuralSOS(*args, **kwargs, raise_exception=True)

    run_doctest_examples(
        func,
        solver=solver,
        configs = {
            "ineqs": [a, b, c],
            "return_type": "poly",
        }
    )


def test_structsos_ternary_linear_substitution():
    F = Function("F")

    sol = StructuralSOS(pl("s(a5(a-b)(a-c))"), [b+c-a,c+a-b,a+b-c])
    assert solution_checker(sol) and sol.solution.has(CyclicSum)

    sol = StructuralSOS(pl("s(a(b+c)(a-b)(a-c))"), {b+c-a: u, c+a-b: v, a+b-c: w})
    assert solution_checker(sol)

    sol = StructuralSOS(pl("s(a(b+c)(a-b)(a-c))"), {b+c-a: F(a), c+a-b: F(b), a+b-c: F(c)})
    assert solution_checker(sol) and sol.solution.has(CyclicSum)

    sol = StructuralSOS(pl("s(a(b+c)(a-b)(a-c))"), [b+c-a,c+a-b,a+b-c])
    assert solution_checker(sol) and sol.solution.has(CyclicSum)

    sol = StructuralSOS(pl("s((b+c-3a)3(a-b)(a-c))"), {b+c-3*a: u, c+a-3*b: v, a+b-3*c: w})
    assert solution_checker(sol)
