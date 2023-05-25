from typing import Optional, Dict, List

import sympy as sp

from .linsos import LinearSOS
from .structsos import StructuralSOS

from ..utils.polytools import deg, verify_hom_cyclic
from ..utils.expression.solution import Solution
from ..utils.roots import RootsInfo, findroot

NAME_TO_METHOD = {
    'LinearSOS': LinearSOS,
    'StructuralSOS': StructuralSOS
}

METHOD_ORDER = ['StructuralSOS', 'LinearSOS']

DEFAULT_CONFIGS = {
    'LinearSOS': {

    },
    'StructuralSOS': {

    }
}


a, b, c = sp.symbols('a b c')


def sum_of_square(
        poly: sp.polys.Poly,
        rootsinfo: Optional[RootsInfo] = None,
        method_order: List = METHOD_ORDER,
        configs: Dict = DEFAULT_CONFIGS
    ) -> Solution:
    """
    
    """
    assert isinstance(poly, sp.polys.Poly) and poly.gens == (a,b,c), 'Poly must be a sympy polynomial with gens (a,b,c).'
    assert deg(poly) > 1, 'Poly must be a polynomial of degree greater than 1.'
    assert verify_hom_cyclic(poly) == (True, True), 'Poly must be homogeneous and cyclic.'

    if rootsinfo is None:
        rootsinfo = findroot(poly, with_tangents=True)

    for method in method_order:
        config = configs.get(method, {})
        method = NAME_TO_METHOD[method]

        solution = method(poly, rootsinfo=rootsinfo, **config)

        if solution is not None:
            return solution

    return None