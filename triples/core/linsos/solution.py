from typing import List, Tuple, Optional, Union

import sympy as sp
from sympy.core.singleton import S
from sympy.combinatorics import PermutationGroup

from .basis import LinearBasis, LinearBasisTangent
from .updegree import LinearBasisMultiplier
from ...utils.expression.cyclic import CyclicSum
from ...utils.expression.solution import SolutionSimple
from ...utils import MonomialPerm, MonomialReduction

def _merge_common_basis(
        y: List[sp.Expr], powers: List[Tuple], symbols: Tuple[sp.Symbol, ...]
    ) -> sp.Expr:
    """
    Get the common base and the extraction of the expression.
    """
    basis = sp.Add(*(yi*sp.Mul(*[s**p for s, p in zip(symbols, power)]) for yi, power in zip(y, powers))).together()
    return basis


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
            symmetry: Union[PermutationGroup, MonomialPerm] = PermutationGroup(),
            is_equal: bool = True,
            collect: bool = True,
        ):
        """
        Parameters
        ----------
        problem: sp.Poly
            The target polynomial.
        y: List[sp.Rational]
            The coefficients of the basis.
        basis: List[LinearBasis]
            The collection of basis.
        symbols: Tuple[sp.Symbol, ...]
            The symbols of the polynomial with the homogenizer included.
        symmetry: PermutationGroup
            Every term will be wrapped by a cyclic sum of symmetryutation group.
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

        if isinstance(symmetry, PermutationGroup):
            symmetry = MonomialPerm(symmetry)

        self._symmetry = symmetry
        self._is_equal = is_equal

        multiplier = self._collect_multipliers(y, basis, problem.gens, symmetry)

        if collect:
            numerator = self._collect_common_tangents(problem.gens, symmetry)
        else:
            raise NotImplementedError
        self.solution = numerator / multiplier


    def _collect_multipliers(self, 
            y: List[sp.Rational], basis: List[LinearBasis],
            symbols: Tuple[sp.Symbol, ...], symmetry: MonomialReduction
        ) -> None:
        r"""
        Collect multipliers. For example, if we have 
        \sum (a^2-ab) * f(a,b,c) = g(a,b,c) + \sum (-ab) * f(a,b,c), then we should combine them.
        """
        multipliers = []
        non_mul_y = []
        nom_mul_basis = []
        for v, base in zip(y, basis):
            if isinstance(base, LinearBasisMultiplier):
                multipliers.append(v * base.multiplier)
            else:
                non_mul_y.append(v)
                nom_mul_basis.append(base)

        r, multiplier = sp.Add(*multipliers).as_content_primitive()
        # r = S.One

        self.y = [v / r for v in non_mul_y]
        self.basis = nom_mul_basis
        multiplier = symmetry.cyclic_sum(multiplier, symbols)
        return multiplier

    def _collect_common_tangents(self, symbols: Tuple[sp.Symbol, ...], symmetry: MonomialReduction):
        """
        Collect terms with same tangents. For example, if we have
        a*b*(a^2-b^2-ab-ac+2bc)^2 + a*c*(a^2-b^2-ab-ac+2bc)^2, then we should combine them.
        Every term will be wrapped by a cyclic sum of symmetryutation group.
        """
        basis_by_tangents = {}
        exprs = []
        for v, base in zip(self.y, self.basis):
            cls = base.__class__
            # if cls not in (LinearBasisTangent,):
            if not issubclass(cls, LinearBasisTangent):
                expr = v * base.as_expr(symbols)
                exprs.append(expr)
                continue

            # base: LinearBasisTangent 
            tangent = base.tangent(symbols)

            collection = basis_by_tangents.get((cls, tangent))
            if collection is None:
                basis_by_tangents[(cls, tangent)] = ([v], [base.powers])
            else:
                collection[0].append(v)
                collection[1].append(base.powers)

        for (cls, tangent), (y, base_powers) in basis_by_tangents.items():
            common_base = _merge_common_basis(y, base_powers, symbols) * tangent.together()
            exprs.append(common_base)

        return sp.Add(*(symmetry.cyclic_sum(expr, symbols) for expr in exprs))