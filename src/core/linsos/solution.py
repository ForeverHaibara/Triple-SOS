from typing import List, Tuple

import numpy as np
import sympy as sp
from sympy.core.singleton import S

from .basis import LinearBasis, LinearBasisTangentCyclic, LinearBasisTangent, a, b, c
from .updegree import LinearBasisMultiplier
from ...utils.expression.cyclic import CyclicSum, is_cyclic_expr
from ...utils.expression.solution import SolutionSimple
from ...utils.roots.rationalize import cancel_denominator


def _get_common_base_and_extraction(tangent: sp.Expr, y: List[sp.Expr], base_info: List[Tuple]) -> Tuple[sp.Expr, sp.Expr, sp.Expr]:
    """
    Get the common base and the extraction of the expression.
    """
    base_info = np.array(base_info, dtype = 'int32')
    common_base = base_info.min(axis = 0)
    base_info -= common_base

    gcd = cancel_denominator(y)

    extracted = []
    for v, (i, j, k, m, n, p) in zip(y, base_info):
        extracted.append(
            (v / gcd) * (a-b)**(2*i) * (b-c)**(2*j) * (c-a)**(2*k) * a**m * b**n * c**p
        )
    extracted = sp.Add(*extracted)

    i, j, k, m, n, p = common_base
    common_base = (a-b)**(2*i) * (b-c)**(2*j) * (c-a)**(2*k) * a**m * b**n * c**p

    return gcd, common_base, extracted


class SolutionLinear(SolutionSimple):
    """
    Solution of Linear SOS. It takes the form of 
    f(a,b,c) * multiplier = sum(y_i * basis_i)
    """
    method = 'LinearSOS'
    def __init__(self, 
            problem = None,
            y: List[sp.Rational] = [],
            basis: List[LinearBasis] = [],
            multiplier: sp.Expr = S.One,
            is_equal: bool = True,
            collect: bool = True,
        ):
        """
        Parameters
        ----------
        problem: sp.polys.Poly
            The target polynomial.
        y: List[sp.Rational]
            The coefficients of the basis.
        basis: List[LinearBasis]
            The collection of basis.
        multiplier: sp.Expr
            The multiplier such that poly * multiplier = sum(y_i * basis_i).
        is_equal: bool
            Whether the problem is an equality.
            For linear sos, this should be checked in function
            `linear_correction`.
        collect: bool
            Whether to collect the solution by analogue terms. 
            This keeps the solution neat.
            See `collect` method for more details.
        """
        self.problem = problem
        self.y = y
        self.basis = basis
        self.multiplier = multiplier
        self.is_equal_ = is_equal

        self.collect_multipliers()

        if collect:
            self.collect()
        else:
            self.numerator = sum(v * b.expr for v, b in zip(self.y, self.basis) if v != 0)
            self.solution = self.numerator / self.multiplier

    def collect_multipliers(self):
        r"""
        Collect multipliers. For example, if we have 
        \sum (a^2-ab) * f(a,b,c) = g(a,b,c) + \sum (-ab) * f(a,b,c), then we should combine them.
        """
        multipliers = [self.multiplier]
        non_mul_y = []
        nom_mul_basis = []
        for v, base in zip(self.y, self.basis):
            if isinstance(base, LinearBasisMultiplier):
                multipliers.append(v * base.multiplier)
            else:
                non_mul_y.append(v)
                nom_mul_basis.append(base)

        def _is_cyclic_sum(m):
            if isinstance(m, CyclicSum):
                return True
            if isinstance(m, sp.Mul) and len(m.args) == 2 and m.args[0].is_constant() and isinstance(m.args[1], CyclicSum):
                return True
            return False

        def _get_cyclic_sum_core(m):
            if isinstance(m, CyclicSum):
                return m.args[0]
            if isinstance(m, sp.Mul) and len(m.args) == 2 and m.args[0].is_constant() and isinstance(m.args[1], CyclicSum):
                return m.args[1].args[0] * m.args[0]
            return None

        if all(_is_cyclic_sum(m) for m in multipliers):
            multipliers = [_get_cyclic_sum_core(m) for m in multipliers]
            self.multiplier = CyclicSum(sp.Add(*multipliers), (a, b, c))
            r, self.multiplier = self.multiplier.as_content_primitive()
        else:
            self.multiplier = sp.Add(*multipliers)
            r, self.multiplier = self.multiplier.as_content_primitive()
            # r = S.One

        self.y = [v / r for v in non_mul_y]
        self.basis = nom_mul_basis

    def collect(self):
        """
        Collect terms with same tangents. For example, if we have
        a*b*(a^2-b^2-ab-ac+2bc)^2 + a*c*(a^2-b^2-ab-ac+2bc)^2, then we should combine them.
        """
        basis_by_tangents = {}
        exprs = []
        for v, base in zip(self.y, self.basis):
            cls = base.__class__
            if cls not in (LinearBasisTangentCyclic, LinearBasisTangent):
                expr = v * base.expr
                exprs.append(expr)
                continue

            tangent = base.tangent
            info = base.info_
            if cls is LinearBasisTangentCyclic and is_cyclic_expr(tangent, (a,b,c)):
                # we shall rotate the expression so that j is maximum
                i, j, k, m, n, p = info
                if i != j and i != k and j != k:
                    if i > j and i > k:
                        info = j, k, i, n, p, m
                    elif j > i and j > k:
                        info = k, i, j, p, m, n
                elif i > k:
                    if i == j: # i == j > k
                        info = k, i, j, p, m, n
                    else: # i > j == k
                        info = j, k, i, n, p, m
                elif j > k: # j > k == i
                    info = k, i, j, p, m, n
                i, j, k, m, n, p = info
                info = j, k, i, n, p, m

            collection = basis_by_tangents.get((cls, tangent))
            if collection is None:
                basis_by_tangents[(cls, tangent)] = ([v], [info])
            else:
                collection[0].append(v)
                collection[1].append(info)

        for (cls, tangent), (y, base_info) in basis_by_tangents.items():
            gcd, common_base, extracted = _get_common_base_and_extraction(tangent, y, base_info)

            f = (lambda x: CyclicSum(x)) if cls is LinearBasisTangentCyclic else (lambda x: x)
            collected = gcd * f(common_base * extracted * tangent.together())
            exprs.append(collected)

        self.numerator = sp.Add(*exprs)
        self.solution = self.numerator / self.multiplier
        return self