import sympy as sp

from ..cauchy import norm_cone_prog_problem


def test_norm_cone_prog_problem_solve():
    """Solve two nontrivial weighted 2-norm cone constraints."""
    A1 = sp.eye(2)
    A2 = sp.diag(1, 2)
    weight = sp.Matrix([1, 1])
    affine = sp.Matrix([1, 1])

    sdp = norm_cone_prog_problem(
        [A1, A2], [weight, weight], [affine, affine], [2, 2]
    )
    x0, x1 = sdp.gens[:2]
    sdp.solve_obj(-x0 - x1, constraints=[x0 + x1 <= 1, x0 >= 0, x1 >= 0])

    assert abs((-x0 - x1).xreplace(sdp.as_params()) + 1) < 1e-5


def test_norm_cone_prog_problem_empty():
    assert norm_cone_prog_problem([], [], [], []).dof == 0


def test_norm_cone_prog_problem_linear_and_power_tree():
    A = sp.Matrix([[1, 0], [0, 1]])
    weight = sp.Matrix([1, 1])
    affine = sp.Matrix([1, 1])

    linear = norm_cone_prog_problem([A], [weight], [affine], [1])
    cubic = norm_cone_prog_problem([A], [weight], [affine], [3])

    assert linear.dof == 2
    assert cubic.dof > linear.dof
    assert cubic.gens[:2] == linear.gens
