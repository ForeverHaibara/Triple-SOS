from typing import List

import numpy as np
import sympy as sp
from sympy.core.singleton import S

from .basis import LinearBasis, LinearBasisTangent, a, b, c
from .updegree import LinearBasisMultiplier
from ...utils.polytools import deg
from ...utils.expression.cyclic import CyclicSum, is_cyclic_expr
from ...utils.expression.solution import SolutionSimple, congruence_as_sos
from ...utils.basis_generator import arraylize_sp, generate_expr
from ...utils.roots.rationalize import cancel_denominator

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
        $\sum (a^2-ab) * f(a,b,c) = g(a,b,c) + \sum (-ab) * f(a,b,c)$, then we should combine them.
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
            r = S.One

        self.y = [v / r for v in non_mul_y]
        self.basis = nom_mul_basis

    def collect(self):
        r"""
        Collect terms with same tangents. For example, if we have
        $a*b*(a^2-b^2-ab-ac+2bc)^2 + a*c*(a^2-b^2-ab-ac+2bc)^2$, then we should combine them.
        """
        basis_by_tangents = {}
        exprs = []
        for v, base in zip(self.y, self.basis):
            if not isinstance(base, LinearBasisTangent):
                exprs.append(v * base.expr)
                continue

            tangent = base.tangent
            info = base.info_
            if is_cyclic_expr(tangent, (a,b,c)):
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

            collection = basis_by_tangents.get(tangent)
            if collection is None:
                basis_by_tangents[tangent] = ([v], [info])
            else:
                collection[0].append(v)
                collection[1].append(info)

        for tangent, (y, base_info) in basis_by_tangents.items():
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

            collected = gcd * CyclicSum(common_base * extracted * tangent.together())
            exprs.append(collected)

        self.numerator = sp.Add(*exprs)
        self.solution = self.numerator / self.multiplier
        return self

    def as_congruence(self):
        r"""
        Note that (part of) g(a,b,c) can be represented sum of squares. For example, polynomial of degree 4 
        has form [a^2,b^2,c^2,ab,bc,ca] * M * [a^2,b^2,c^2,ab,bc,ca]' where M is positive semidefinite matrix.

        We can first reconstruct and M and then find its congruence decomposition, 
        this reduces the number of terms.

        NOTE: By experiment, in most cases rewriting the solution as congruence decomposition does not
        reduce any simplicity. Hence please do not use it.

        TODO: The function is too slow. Maybe we should cache (a-b)^i * (b-c)^j * (c-a)^k.
        """
        degree = deg(self.problem) + deg(self.multiplier.doit().as_poly(a,b,c))

        sqr_args = {}
        for i in (degree % 2, degree % 2 + 2):
            half_degree = (degree - i) // 2
            if half_degree > 0:
                n = len(generate_expr(half_degree, cyc = False)[1])
                sqr_args[i] = sp.zeros(n)

        unsqr_basis = []

        for y, base in zip(self.y, self.basis):
            if not isinstance(base, LinearBasisTangent):
                unsqr_basis.append((y, base))
                continue

            core = None
            i, j, k, m, n, p = base.info_
            if base.tangent.is_constant():
                core = base.tangent
            elif base.tangent.is_Pow:
                core = base.tangent.base ** (base.tangent.exp // 2)
            if core is None:
                unsqr_basis.append((y, base))
                continue
            core *= (a-b)**i * (b-c)**j * (c-a)**k * a**(m//2) * b**(n//2) * c**(p//2)
            core = core.as_poly(a,b,c)

            m, n, p = m % 2, n % 2, p % 2
            new_gen = None
            if m != n or n != p: # not m + n + p == 0 or 3
                if n >= p >= m:
                    new_gen = (b, c, a)
                elif p >= m >= n:
                    new_gen = (c, a, b)
            if new_gen is not None:
                core = sp.polys.Poly.from_poly(core, new_gen)

            core = arraylize_sp(core, cyc=False)
            mat = sqr_args.get(m + n + p)
            if mat is None:
                unsqr_basis.append((y, base))
                continue

            for i in range(mat.shape[0]):
                mat[:, i] += core * (core[i] * y)

        new_numerator = sum(v * b.expr for v, b in unsqr_basis)

        for key, multiplier in enumerate((S.One, a, a*b, a*b*c)):
            mat = sqr_args.get(key)
            if mat is None:
                continue

            new_numerator += congruence_as_sos(mat, multiplier = multiplier, cyc = True, cancel = True)
        # return sqr_args
        self.numerator = new_numerator
        self.solution = self.numerator / self.multiplier

        return self
