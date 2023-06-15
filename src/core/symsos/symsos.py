import sympy as sp

from .representation import sym_representation, _verify_is_symmetric
from .proof import _prove_numerator
from .solution import SolutionSymmetric, SolutionSymmetricSimple
from ...utils.expression.cyclic import CyclicSum, CyclicProduct

def SymmetricSOS(
        poly,
        rootsinfo = None,    
    ):
    """
    Represent a polynomial to the symmetric form.
    """

    # check symmetricity here and (1,1,1) == 0
    if poly(1,1,1) != 0 or not _verify_is_symmetric(poly):
        return None

    numerator, denominator = sym_representation(poly, positive = True, return_poly = True)
    numerator = _prove_numerator(numerator)
    if numerator is None:
        return None
    expr = numerator / denominator

    a, b, c, x, y, z, w, p = sp.symbols('a b c x y z w p')

    expr = expr.subs({
        x: CyclicSum(a*(a-b)*(a-c)),
        y: CyclicSum(a*(b-c)**2),
        z: CyclicProduct((a-b)**2) * CyclicSum(a)**3 / CyclicProduct(a),
        w: CyclicSum(a*(a-b)*(a-c)) * CyclicSum(a*(b-c)**2)**2 / CyclicProduct(a),
        p: CyclicSum(a)
    })

    solution = SolutionSymmetric(
        problem = poly,
        solution = expr,
        is_equal = True
    ).as_simple_solution()

    return solution