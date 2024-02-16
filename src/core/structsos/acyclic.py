import sympy as sp

from .utils import CyclicSum, CyclicProduct, Coeff, prove_univariate

def sos_struct_acyclic(coeff, real = True):
    """
    Solve acyclic 3-var polynomial inequalities.
    """
    0