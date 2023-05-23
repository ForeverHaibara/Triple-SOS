import numpy as np
import sympy as sp
from sympy.core.singleton import S

from ..expression.cyclic import _is_cyclic_expr, CyclicSum, CyclicProduct
from ..basis_generator import generate_expr
from ..roots.rationalize import cancel_denominator

class Solution():
    def __init__(self, problem = None, solution = None):
        self.problem = problem
        self.solution = solution
        self.is_equal_ = None

    def __str__(self) -> str:
        return f"{self.problem} = {self.solution}"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def is_equal(self):
        return self.is_equal_

# class SolutionNull(Solution):
#     def __init__(self, problem = None, solution = None):
#         super().__init__(problem = problem, solution = None)


def _arg_sqr_core(arg):
    if arg.is_constant():
        return S.One
    if isinstance(arg, sp.Symbol):
        return arg
    if arg.is_Pow:
        return S.One if arg.args[1] % 2 == 0 else arg.args[0]
    if arg.is_Mul:
        return sp.Mul(*[_arg_sqr_core(x) for x in arg.args])
    if isinstance(arg, CyclicProduct):
        return CyclicProduct(_arg_sqr_core(arg.args[0]), arg.symbols).doit()
    return None


class SolutionSimple(Solution):
    """
    All (rational) SOS solutions can be presented in the form of f(a,b,c) = g(a,b,c) / h(a,b,c)
    where g and h are polynomials.
    """
    def __init__(self, problem = None, numerator = None, multiplier = None, is_equal = None):
        if multiplier is None:
            multiplier = 1
        self.problem = problem
        self.solution = numerator / multiplier
        self.numerator = numerator
        self.multiplier = multiplier
        self.is_equal_ = is_equal

    @property
    def is_equal(self):
        if self.is_equal_ is None:
            symbols = set(self.problem.symbols) # | set(self.numerator.free_symbols) | set(self.multiplier.free_symbols)
            difference = (self.problem  * self.multiplier - self.numerator)
            difference = difference.doit().as_poly(*symbols)
            self.is_equal_ = difference.is_zero

        return self.is_equal_

    def as_congruence(self):
        """
        Note that (part of) g(a,b,c) can be represented sum of squares. For example, polynomial of degree 4 
        has form [a^2,b^2,c^2,ab,bc,ca] * M * [a^2,b^2,c^2,ab,bc,ca]' where M is positive semidefinite matrix.

        We can first reconstruct and M and then find its congruence decomposition, 
        this reduces the number of terms.

        WARNING: WE ONLY SUPPORT 3-VAR CASE.
        """
        if not isinstance(self.numerator, sp.Add):
            # in this case, the numerator is already simplified
            return self

        return self

        # not implemented yet

        # now we only handle cyclic expressions
        sqr_args = {}
        unsqr_args = []
        
        def _is_symbol(s):
            return isinstance(s, sp.Symbol) and s in self.problem.symbols

        def _is_core_monomial(core):
            """Whether core == a^i * b^j * c^k."""
            if core.is_constant() or _is_symbol(core):
                return True
            if isinstance(core, sp.Pow) and _is_symbol(core.args[0]):
                return True
            if isinstance(core, sp.Mul):
                return all(_is_core_monomial(x) for x in core.args)
            return False

        for arg in self.numerator.args:
            core = None
            if _is_cyclic_expr(arg, self.problem.symbols):
                if isinstance(arg, CyclicSum):
                    core = _arg_sqr_core(arg.args[0])
                else:
                    core = _arg_sqr_core(arg)
                if not _is_core_monomial(core):
                    core = None

            if core is not None:
                # reduce monomial core once more, e.g. a^4 b^5 c^3 -> bc
                core = _arg_sqr_core(core)
                if len(core.free_symbols) not in sqr_args:
                    sqr_args[len(core.free_symbols)] = []
                sqr_args[len(core.free_symbols)].append(arg)
            else:
                unsqr_args.append(arg)



def congruence(M):
    """
    Write a symmetric matrix as a sum of squares.
    M = U.T @ S @ U where U is upper triangular and S is diagonal.

    Returns
    -------
    U : np.ndarray
        Upper triangular matrix.
    S : np.ndarray
        Diagonal vector (1D array).
    """
    M = M.copy()
    n = M.shape[0]
    if isinstance(M[0,0], sp.Expr):
        U, S = sp.Matrix.zeros(n), [0] * n
    else:
        U, S = np.zeros((n,n)), np.zeros(n)
    for i in range(n-1):
        if M[i,i] > 0:
            S[i] = M[i,i]
            U[i,i+1:] = M[i,i+1:] / (S[i])
            U[i,i] = 1
            M[i+1:,i+1:] -= U[i:i+1,i+1:].T @ (U[i:i+1,i+1:] * S[i])
    U[-1,-1] = 1
    S[-1] = M[-1,-1]
    return U, S


def congruence_as_sos(M, multiplier = S.One, symbols = 'a b c', cancel = True, cyc = True):
    # (n+1)(n+2)/2 = M.shape[0]
    n = round((M.shape[0] * 2 + .25)**.5 - 1.5)
    U, S = congruence(M)

    if isinstance(symbols, str):
        symbols = sp.symbols(symbols)
    a, b, c = symbols

    exprs = []
    coeffs = []

    monoms = generate_expr(n, cyc = 0)[1]
    for i, s in enumerate(S):
        if s == 0:
            continue
        val = sp.S(0)
        if cancel:
            r = cancel_denominator(U[i,i:])
        for j in range(i, len(monoms)):
            monom = monoms[j]
            val += U[i,j] / r * a**monom[0] * b**monom[1] * c**monom[2]
        exprs.append(val**2)
        coeffs.append(s * r**2)

    exprs = [multiplier * expr for expr in exprs]
    if cyc:
        exprs = [CyclicSum(expr, symbols) for expr in exprs]

    exprs = [coeff * expr for coeff, expr in zip(coeffs, exprs)]
    expr = sp.Add(*exprs)

    return expr
