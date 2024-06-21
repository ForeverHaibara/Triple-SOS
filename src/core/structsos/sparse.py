from typing import Callable, List, Optional

import sympy as sp

from .utils import Coeff, PolynomialUnsolvableError
from ...utils import CyclicProduct

def sos_struct_extract_factors(coeff: Coeff, recurrsion: Callable, symbols: Optional[List[sp.Symbol]] = None, real: bool = True, **kwargs):
    """
    Try solving the inequality by extracting factors and changing of variables.
    It handles cyclic and acyclic inequalities both.

    For instance,
    CyclicSum(a**2bc*(a-b)*(a-c)) is converted to CyclicSum(a*(a-b)*(a-c))
    by extracting CyclicProduct(a).
    CyclicSum(a**2*(a**2-b**2)*(a**2-c**2)) is converted to proving
    Cyclic(a*(a-b)*(a-c)) by making substitution a^2,b^2,c^2 -> a,b,c.
    """
    if symbols is None:
        symbols = sp.symbols(f"a:{chr(96 + coeff.nvars)}")

    monoms, new_coeff = coeff.cancel_abc()
    if any(i > 0 for i in monoms):
        solution = recurrsion(new_coeff, real = real and all(i % 2 == 0 for i in monoms), **kwargs)
        if solution is not None:
            if coeff.nvars == 3 and all(i == monoms[0] for i in monoms):
                multiplier = CyclicProduct(symbols[0]**monoms[0])
            else:
                multiplier = sp.Mul(*[s**i for s, i in zip(symbols, monoms)])
            return multiplier * solution
        raise PolynomialUnsolvableError

    i, new_coeff = coeff.cancel_k()
    if i > 1:
        solution = recurrsion(new_coeff, real = False if (i % 2 == 0) else real, **kwargs)
        if solution is not None:
            return solution.xreplace(dict((s, s**i) for s in symbols))
        raise PolynomialUnsolvableError