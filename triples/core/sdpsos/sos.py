from collections import defaultdict
from typing import List, Tuple, Dict, Union, Optional, Callable, Any
from time import time
# from warnings import warn

from numpy import ndarray
import numpy as np
from sympy.combinatorics import PermutationGroup
from sympy import Poly, Expr, Symbol, Mul, ZZ
from sympy.core.relational import Relational
from sympy.matrices import MutableDenseMatrix as Matrix

from .abstract import AtomSOSElement
from .manifold import constrain_root_nullspace
from .solution import SolutionSDP
from ...utils import CyclicSum, verify_symmetry, MonomialManager, Root
from ...sdp import SDPProblem
from ...sdp.arithmetic import matmul, matadd
from ...sdp.utils import exprs_to_arrays, collect_constraints

CHECK_SYMMETRY = True


class SOSPoly(AtomSOSElement):
    ##########################################################
    # Manually set any variables below after construction
    # could be unsafe, and results in unexpected behaviors.
    ##########################################################

    gens: Tuple[Symbol,...]
    roots: List[Root]

    def __init__(self,
        poly,
        gens,
        qmodule: List[Union[Poly, Expr]] = [],
        ideal: List[Union[Poly, Expr]] = [],
        degree: Optional[int] = None,
        symmetry: Optional[PermutationGroup] = None,
        roots: Optional[List[Root]] = [],
        # qmodule_bases: Optional[List[Tuple[int, ...]]] = None,
        # ideal_bases: Optional[List[Tuple[int, ...]]] = None,
    ):
        self.poly = Poly(poly, gens)
        self.gens = gens

        if not isinstance(qmodule, dict):
            qmodule = dict(enumerate(qmodule))
        if not isinstance(ideal, dict):
            ideal = dict(enumerate(ideal))
        self._qmodule = {k: Poly(v, gens) for k, v in qmodule.items()}
        self._ideal = {k: Poly(v, gens) for k, v in ideal.items()}
        
        degree = degree if degree is not None else self.poly.total_degree()
        is_homogeneous = self.poly.is_homogeneous \
            and all(_.is_homogeneous for _ in self._qmodule.values()) \
            and all(_.is_homogeneous for _ in self._ideal.values())
        from .algebra import PolyRing
        self.algebra = PolyRing(len(gens), degree=degree, symmetry=symmetry, is_homogeneous=is_homogeneous)


        self.roots = roots

    def _post_construct(self, verbose: bool = False):
        self.sdp.constrain_zero_diagonals()

        _, roots = constrain_root_nullspace(self.sdp, self.poly, self._qmodule, self._ideal,
            ineq_bases=self._qmodule_bases, eq_bases=self._ideal_bases, degree=self.algebra.degree,
            roots=self.roots, symmetry=self.algebra.symmetry, verbose=verbose)
        self.roots = roots

    def as_solution(
        self,
        qmodule: Optional[Dict[Any, Expr]] = None,
        ideal: Optional[Dict[Any, Expr]] = None,
        adjoint_operator: Optional[Callable[[Expr], Expr]] = None,
        trace_operator: Optional[Callable[[Expr], Expr]] = None,
    ) -> SolutionSDP:
        """
        Retrieve the solution to the original polynomial.

        Parameters
        ----------
        qmodule: Optional[Dict[Any, Expr]]
            Conversion of polynomial inequality constraints to sympy expression forms.
        ideal: Optional[Dict[Any, Expr]]
            Conversion of polynomial equality constraints to sympy expression forms.
        """
        decomp = self.sdp.decompositions
        if decomp is None:
            raise ValueError("The problem has not been solved yet.")
        qmodule = self._qmodule if qmodule is None else qmodule
        ideal = self._ideal if ideal is None else ideal
        eqspace = {eq: x + space * self.sdp.y for eq, (x, space) in self._ideal_space.items()}

        solution = SolutionSDP.from_decompositions(self.poly, decomp, eqspace,
            qmodule          = qmodule,
            qmodule_bases    = self._qmodule_bases,
            ideal            = ideal,
            ideal_bases      = self._ideal_bases,
            state_operator   = (lambda x: CyclicSum(x, self.gens, self.algebra.symmetry))\
                                    if self.algebra.symmetry is not None else (lambda x: x),
            adjoint_operator = adjoint_operator,
        )

        # overwrite the constraints information in the form of dict((Poly, Expr))
        solution.ineq_constraints = {self._qmodule[key]: expr for key, expr in qmodule.items()}
        solution.eq_constraints = {self._ideal[key]: expr for key, expr in ideal.items()}
        return solution
