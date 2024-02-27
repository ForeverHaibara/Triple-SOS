import sympy as sp

from .solver import SDPProblem
from .utils import (
    degree_of_monomial, solve_undetermined_linear
)
from ...utils.basis_generator import generate_expr
from ...utils import deg, arraylize_sp


class SOSProblem():
    """
    Helper class for SDPSOS. See details at SOSProblem.solve.

    Assume that a polynomial can be written in the form v^T @ M @ v.
    Sometimes there are implicit constraints that M = Q @ S @ Q.T where Q is a rational matrix.
    So we can solve the problem on S first and then restore it back to M.

    To summarize, it is about solving for S >> 0 such that
    eq @ vec(S) = vec(P) where P is determined by the target polynomial.
    """
    def __init__(
        self,
        poly: sp.Poly,
    ):
        """
        
        """
        self.poly = poly
        self._degree = deg(poly)
        self.sdp = None

    def _construct_primal_sdp(self, monomials):
        """
        Translate the current SOS problem to SDP problem.
        The problem is to find M1, M2, ..., Mn such that

            Poly = p1*(v1'M1v1) + p2*(v2'M2v2) + ...

        where vi are vectors of monomials like [a^2, b^2, c^2, ab, bc, ca]
        while pi are extra monomials like 1, a, ab or abc.
        M1, M2, ... are symmetric matrices and we want them to be positive semidefinite.

        # The first step is to find M1, M2, ... that satisfy the equation,
        # but PSD property is not required at this stage.
        """
        degree = self._degree
        eq_list = []
        splits = []
        for monomial in monomials:
            m = sum(monomial)
            if degree - m <= 1:
                continue

            # monomial vectors
            vec  = generate_expr((degree - m)//2, cyc = False)[1]
            vec2 = generate_expr((degree - m), cyc = False)[0]
            l = len(vec)

            def mapping(i, j):
                d = tuple(d1 + d2 + d3 for d1, d2, d3 in zip(monomial, vec[i], vec[j]))
                return vec2[d]

            eq = sp.zeros(len(vec2), l**2)

            cnt = 0
            for i in range(l):
                for j in range(l):
                    eq[mapping(i, j), cnt] = 1
                    cnt += 1
            eq_list.append(eq)
            splits.append(l)

        eq_mat = sp.Matrix.vstack(*eq_list)
        rhs = arraylize_sp(self.poly, cyc = False)
        return SDPProblem.from_equations(eq_mat, rhs, splits)