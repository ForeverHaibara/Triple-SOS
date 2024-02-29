from typing import Union, Optional, List, Tuple

import sympy as sp

from .solver import SDPProblem
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
        self._is_cyc = True
        self.sdp = None

    def _construct_sdp(
        self,
        monomials: List[Tuple[int, ...]],
        nullspace: Optional[List[sp.Matrix]] = None,
        cyc: bool = False
    ) -> SDPProblem:
        """
        Translate the current SOS problem to SDP problem.
        The problem is to find M1, M2, ..., Mn such that

            Poly = p1*(v1'M1v1) + p2*(v2'M2v2) + ...

        where vi are vectors of monomials like [a^2, b^2, c^2, ab, bc, ca]
        while pi are extra monomials like 1, a, ab or abc.
        M1, M2, ... are symmetric matrices and we want them to be positive semidefinite.

        Parameters
        ----------
        monomials : List[Tuple[int, ...]]
            A list of monomials. Each monomial is a tuple of integers.
            For example, (2, 0, 0) represents a^2 for a three-variable polynomial.
        nullspace : Optional[List[sp.Matrix]]
            The nullspace for each matrix. If nullspace is None, it is skipped.
        cyc : bool
            Whether the polynomial is cyclic or not. If cyc is True, each term of the
            SOS decomposition is wrapped by a cyclic sum.

        Returns
        ----------
        SDPProblem
            The SDP problem to solve.
        """
        degree = self._degree
        eq_list, splits = [], {}

        vec2 = generate_expr(degree, cyc = cyc)[0]

        for monomial in monomials:
            m = sum(monomial)
            if (degree - m) % 2 != 0:
                continue

            # monomial vectors
            vec = generate_expr((degree - m)//2, cyc = False)[1]
            l = len(vec)
            eq = sp.zeros(len(vec2), l**2)

            if not cyc:
                def mapping(i, j):
                    d = tuple(d1 + d2 + d3 for d1, d2, d3 in zip(monomial, vec[i], vec[j]))
                    return vec2[d]
            else:
                def mapping(i, j):
                    d = tuple(d1 + d2 + d3 for d1, d2, d3 in zip(monomial, vec[i], vec[j]))
                    for _ in range(len(d)):
                        v = vec2.get(d)
                        if v is not None:
                            return v
                        d = d[1:] + (d[0],)

            cnt = 0
            for i in range(l):
                for j in range(l):
                    eq[mapping(i, j), cnt] = 1
                    cnt += 1

            eq_list.append(eq)
            splits[str(monomial)] = l

        eq_mat = sp.Matrix.hstack(*eq_list)
        rhs = arraylize_sp(self.poly, cyc = cyc)
        sdp = SDPProblem.from_equations(eq_mat, rhs, splits)
        return sdp