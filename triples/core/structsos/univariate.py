from contextlib import AbstractContextManager, contextmanager
from typing import Tuple, List, Optional, Union

from mpmath import mp
import numpy as np
from numpy.polynomial.polynomial import polyroots as np_polyroots
from numpy.polynomial.polynomial import polyfromroots as np_polyfromroots
from sympy import Poly, Expr, Float, Integer, Symbol, QQ, construct_domain, count_roots
from sympy.core import S
from sympy.matrices import MutableDenseMatrix as Matrix
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.polyerrors import MultivariatePolynomialError
from sympy.polys.rings import PolyElement

from ...sdp import congruence
from ...utils import SOSCone, SOSElement, SOSlist

T = float
C = complex

class CTX():
    """

    Subclasses should implement:
    * float, complex:
        For converting a float/complex/int/mpmath.mpf/mpmath.mpc to a
        (possibly high-prec) float or complex object in the given context.
    * real, imag, conjugate:
        For getting the real/imag/conjugate of a complex object in the given context.
    * withdps:
        For setting the global precision to `dps`.
    * round:
        For rounding a float to a Python int.
    * matrix, mat_shape, ger:
        For creating a float matrix from a list of lists. The matrix is ensured to be REAL.
        And for matrix manipulations.
    * polyroots:
        For computing the complex roots of a real polynomial.
    * polyfromroots:
        For computing the coefficients of the polynomial built from a list of complex roots.

    This is a naive implementation and is very slow.
    In practice, use NumpyCTX for default precision and FlintCTX for arbitrary precision.
    """
    seed = 1209
    early_stop = lambda self, dim: 2 * dim + 2

    def float(self, x: Union[int, str, float]) -> T:
        return float(x)
    def complex(self, x: Union[int, str, T, complex]) -> C:
        return complex(x)
    def round(self, x: T) -> int:
        return int(round(x))
    def real(self, x: C) -> T:
        return x.real
    def imag(self, x: C) -> T:
        return x.imag
    def conjugate(self, x: C) -> C:
        return x.conjugate()
    def is_real(self, x: C) -> bool:
        imag = self.imag(x)
        return not (imag > 0 or imag < 0)
    def withdps(self, dps: int) -> AbstractContextManager:
        """
        Create a context manager that sets the global precision to `dps`.
        This is called before all arithmetics.
        * Subclasses should implement this if it is necessary.
        """
        @contextmanager
        def suppress_overflow():
            try:
                yield
            except (OverflowError, FloatingPointError):
                # might overflow when conversion from int to float
                pass
        return suppress_overflow()
    def matrix(self, m: List[List[T]]):
        """
        Create a float matrix from a list of lists. The matrix is ensured to be REAL.
        * Subclasses should implement this method to return a matrix that
        stores context-specific floats and that supports the `M[i, j]` syntax.
        """
        return Matrix(m)
    def mat_shape(self, m) -> Tuple[int, int]:
        """
        Return the shape of the matrix `m`.
        * Subclasses should implement this if `m.shape` is not supported.
        """
        return m.shape
    def ger(self, m, x, y):
        """
        General rank-one update: `m + x*y^T` where `x` and `y` are in the shape of (n, 1).
        All `m, x, y` are in same type as the output of `self.matrix`.
        Can be done either in-place or not.
        * Subclasses should implement this method to adapt for the context-specific
        matrix multiplication and transpose.
        """
        return m + x*y.transpose()

    def zeros(self, r: int, c: int):
        """
        Create a float matrix of zeros.
        * Need not to be implemented for subclasses.
        """
        zero = self.float(0)
        return self.matrix([[zero for _ in range(c)] for _ in range(r)])

    def polyroots(self, x: List[T]) -> Optional[List[C]]:
        """
        Compute the complex roots of a real polynomial. The polynomial is represented
        by a list of float coefficients. The leading coefficient is in the front.
        Returns None if there the algorithm does not converge in the precision.
        * Complexity: O(n**3) (ignoring the high-precision arithmetic)
        * Subclasses should implement this method to return context-specific complexes.
        """
        return list(map(self.complex, Poly(x, Symbol("x")).nroots()))

    def polyfromroots(self, x: List[C]):
        """
        Compute the coefficients of the polynomial built from a list of complex roots.
        The polynomial is represented by a list of complex coefficients. The leading
        coefficient is in the front. It by default uses a divide-and-conquer algorithm.
        * Complexity: O(n**2) (ignoring the high-precision arithmetic)
        * Need not to be implemented for subclasses. However, if it can be done
        more efficiently, it is advised to implement it.
        """
        def polymul(a, b):
            c = [self.complex(0)] * (len(a) + len(b) - 1)
            for i in range(len(a)):
                for j in range(len(b)):
                    c[i + j] += a[i] * b[j]
            return c
        def recur(x):
            n = len(x)
            if n == 0:
                return [self.complex(1)]
            if n == 1:
                return [-x[0], self.complex(1)]
            return polymul(recur(x[:n//2]), recur(x[n//2:]))
        return recur(x)[::-1]

    def classify_roots(self, roots: List[C]) -> Tuple[List[T], List[Tuple[C, C]]]:
        """
        Classify the roots into real and imaginary pairs. Assume all complex
        have their conjugates in the list.
        * Complexity: O(n)
        * Need not to be implemented for subclasses.
        """
        real, imag, is_real, conj = self.real, self.imag, self.is_real, self.conjugate
        real_roots = [real(r) for r in roots if is_real(r)]
        imag_roots = [(r, conj(r)) for r in roots if (not is_real(r)) and imag(r) > 0]
        return real_roots, imag_roots

    def _mat_from_roots(self, complex_roots: List[Tuple[C, C]], leading_coeff: T = 1.):
        """
        Build directly the Gram matrix from complex roots.
        By performing a convex combination of multiple Gram matrices from
        different combinations of complex roots, we obtain a Gram matrix that is
        full rank and strictly positive definite. After building, the matrix
        is passed into `_mat_rounding` to round the entries to rational.

        See the docstring of `prove_univariate` for details.

        * Complexity: O(n**3) (ignoring the high-precision arithmetic)
        * Need not to be implemented for subclasses.
        """
        dim = len(complex_roots)
        M = self.zeros(dim + 1, dim + 1)
        cnt = 0

        early_stop = self.early_stop(dim)
        real, imag = self.real, self.imag

        # Complexity: O(n) loops, each loop O(n**2) => O(n**3)
        np.random.seed(self.seed)
        for _ in range(early_stop):
            # choose randomly z_i or its conjugate
            comb = np.random.randint(0, 2, dim).tolist()
            roots_comb = [complex_roots[i][comb[i]] for i in range(dim)]
            vec = self.polyfromroots(roots_comb)

            vec_re = self.matrix([[real(v)] for v in vec])
            vec_im = self.matrix([[imag(v)] for v in vec])

            # M += vec_re * vec_re.T + vec_im * vec_im.T
            M = self.ger(M, vec_re, vec_re)
            M = self.ger(M, vec_im, vec_im)
            cnt += 1
            if cnt >= early_stop:
                break

        # M is a convex combination of multiple matrices,
        # and is thus (most-likely) full-rank and strictly positive definite
        M = M * (leading_coeff / cnt)
        return M

    def _mat_rounding(self, M, rounding: int = 2, domain = QQ) -> List[List]:
        """
        Rounding a matrix: compute `round(M * rounding) / rounding` and
        cast the result to list-of-lists. Each element should be in the domain.
        The previous `M` is of the same type as the output of `self.matrix`.
        After rounding, the output is passed into `_mat_correction` to correct
        the matrix so that it sums up to the original polynomial.
        * Complexity: O(n**2)
        * Need not to be implemented for subclasses.
        """
        _round = self.round
        rows, cols = self.mat_shape(M)
        dtype = QQ.dtype
        zero = dtype(0)
        M2 = [[zero for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for j in range(i, cols):
                M2[i][j] = dtype(_round(M[i, j] * rounding), rounding)
                M2[j][i] = M2[i][j]
        if not domain.is_QQ:
            for i in range(rows):
                for j in range(i, cols):
                    M2[i][j] = domain.convert_from(M2[i][j], QQ)
                    M2[j][i] = M2[i][j]
        return M2

    def _mat_correction(self, M2: List[List], p: List) -> List[List]:
        """
        Correct the list-of-lists `M2` so that `self.mat_to_sympy_poly(M2) == p`.
        * Complexity: O(n**2)
        * Need not to be implemented for subclasses.
        """
        n = (len(p)+1)//2
        for i in range(2*n - 1):
            m = i // 2
            if i % 2 == 0:
                l = min(m, n - m - 1)
                s = sum(M2[m - j][m + j] for j in range(1, l + 1))
                M2[m][m] = (p[i] - s * 2) if s != 0 else p[i]

            else:
                l = min(m, n - m - 2)
                s = sum(M2[m - j][m + j + 1] for j in range(1, l + 1))
                M2[m][m + 1] = (p[i] / 2 - s) if s != 0 else p[i] / 2
                M2[m + 1][m] = M2[m][m + 1]
        return M2

    def _mat_congruence(self, M2: List[List], domain = QQ) -> Optional[Tuple[Matrix, Matrix]]:
        """
        Cast a list-of-lists to a sympy Matrix and compute its LDL decomposition.
        Returns `(U, S)` such that `U.T @ diag(*S) @ U == Matrix(M2)`. If `M2`
        is not positive semidefinite, returns None.
        * Complexity: O(n**3)
        * Need not to be implemented for subclasses.
        """
        rows = len(M2)
        M3 = Matrix._fromrep(DomainMatrix.from_rep(DDM(M2, (rows, rows), domain)))
        return congruence(M3)

    def _build_sol(self, US: Tuple[Matrix, Matrix], x: Symbol) -> SOSElement:
        """
        Convert the LDL decomposition of the matrix to an SOSElement instance.
        * Complexity: O(n**2)
        * Need not to be implemented for subclasses.
        """
        U, S = US
        polys = []
        cone = SOSCone(U._rep.domain[x], S._rep.domain) # type: ignore
        coeffs = S._rep.rep.to_list_flat()

        from_dict = cone.algebra.one.ring.from_dict
        n = len(coeffs)
        rows = [[] for _ in range(n)]
        for (i, j), v in U._rep.rep.to_dok().items():
            rows[i].append((n - 1 - j, v))
        polys = [from_dict({(k,): v for k, v in row}) for row in rows]
        return SOSElement(cone, [(c, v) for c, v in zip(coeffs, polys) if c])


    def from_sympy_poly(self, p: Poly, dps: int = 15) -> List[T]:
        """
        Convert a sympy polynomial to a list of coefficients in the type of self.float.
        The leading coefficient (highest order) is in the front.
        * Need not to be implemented for subclasses.
        """
        _float = self.float
        if p.domain.is_ZZ:
            return [_float(int(_)) for _ in p.rep.all_coeffs()]
        if p.domain.is_QQ:
            return [_float(int(_.numerator)) / _float(int(_.denominator)) for _ in p.rep.all_coeffs()]
        if p.domain.is_RR:
            # sympy Float, mpmath.mpf and python-flint.arb supports safe conversion without precision loss
            return [_float(_) for _ in p.rep.all_coeffs()]
        # supports float, np.float64, Float, mpf, arb
        from sympy import re
        return [_float(re(_.n(dps, maxn=dps + 85))) for _ in p.all_coeffs()]

    def mat_to_sympy_poly(self, M2: List[List]) -> List:
        """
        * Need not to be implemented for subclass.
        """
        n = len(M2)
        ss = [0] * (2*n - 1)
        for i in range(2*n - 1):
            m = i // 2
            if i % 2 == 0:
                l = min(m, n - m - 1)
                s = sum(M2[m - j][m + j] for j in range(1, l + 1)) * 2 + M2[m][m]
            else:
                l = min(m, n - m - 2)
                s = sum(M2[m - j][m + j + 1] for j in range(l + 1)) * 2
            ss[i] = s
        return ss


class NumpyCTX(CTX):
    def float(self, x):
        return np.float64(x)
    def complex(self, x):
        return np.complex128(x)
    def polyroots(self, x):
        return np_polyroots(x[::-1])
    def matrix(self, x):
        return np.array(x, dtype=np.float64)
    def ger(self, m, x, y):
        return m + x @ y.T
    def polyfromroots(self, x):
        return np_polyfromroots(x)[::-1]

class MpmathCTX(CTX):
    """
    Mpmath supports arbitrary precision but is slower than python-flint.
    """
    maxsteps = -1
    extraprec = -1
    roots_init = True
    def float(self, x):
        return mp.mpf(x)
    def complex(self, x):
        return mp.mpc(x)
    def round(self, x):
        return int(mp.nint(x))
    def withdps(self, dps: int):
        return mp.workdps(dps)
    def polyroots(self, x):
        maxsteps = mp.dps * 2 + 50 if self.maxsteps < 0 else self.maxsteps
        extraprec = (mp.dps * 3)//2 + 10 if self.extraprec < 0 else self.extraprec
        roots_init = self.roots_init
        if roots_init is True:
            try:
                npctx = NumpyCTX()
                x2 = [npctx.float(v) for v in x]
                roots_init = npctx.polyroots(x2)
            except (OverflowError, FloatingPointError):
                roots_init = None
        try:
            return mp.polyroots(x, maxsteps=maxsteps, extraprec=extraprec,
                    roots_init = roots_init)
        except mp.NoConvergence:
            pass
    def matrix(self, x):
        return mp.matrix(x)
    def mat_shape(self, x):
        return (x.rows, x.cols)


try:
    from flint import arb, acb, arb_poly, arb_mat, acb_poly
    from flint import ctx as flint_ctx
    class FlintCTX(CTX):
        def float(self, x):
            return arb(float(x))
        def complex(self, x):
            return acb(x)
        def round(self, x):
            return int(x.mid().floor().unique_fmpz())
        def withdps(self, dps: int):
            if hasattr(flint_ctx, 'workdps'):
                flint_ctx.workdps(dps)
            @contextmanager
            def flint_workdps():
                dps0 = flint_ctx.dps
                try:
                    flint_ctx.dps = dps
                    yield
                finally:
                    flint_ctx.dps = dps0
            return flint_workdps()
        def polyroots(self, x):
            tol = self.float(10)**(-(flint_ctx.dps+5)//2)
            try:
                return arb_poly(x[::-1]).complex_roots(tol=tol)
            except ValueError:
                # roots() failed to converge: insufficient precision, or squareful input
                return None
        def matrix(self, x):
            return arb_mat(x)
        def mat_shape(self, x):
            return (x.nrows(), x.ncols())
        def polyfromroots(self, x):
            return acb_poly.from_roots(x).coeffs()[::-1]
except ImportError:
    class FlintCTX(MpmathCTX): ...


def _prove_univariate_from_mat_R(p: Poly, ctx=None, rounding=None, dps=15) -> Optional[SOSElement]:
    """See the docstring of `prove_univariate` for the description of the algorithm."""
    if ctx is None:
        ctx = NumpyCTX() if dps <= 15 else FlintCTX()
    if rounding is None:
        rounding = 2**((dps+1)//2) # if dps=15, then rounding=256
    rounding_bound = 2**int(round((dps + 1)*3.32))

    US = None

    p = p.to_field()

    with ctx.withdps(dps):
        coeffs = ctx.from_sympy_poly(p, dps)
        roots = ctx.polyroots(coeffs)
        if roots is None:
            return None
        real_roots, complex_roots = ctx.classify_roots(roots)
        if real_roots:
            return None

        M = ctx._mat_from_roots(complex_roots, coeffs[0])

        rounding_i = 1
        while rounding_i < rounding_bound and US is None:
            M2 = ctx._mat_rounding(M, rounding_i, p.domain)
            M2 = ctx._mat_correction(M2, p.rep.all_coeffs()) # exact coeffs here
            US = ctx._mat_congruence(M2, p.domain)
            rounding_i *= rounding

    if US is None:
        return None
    return ctx._build_sol(US, p.gen)


def _prove_univariate_from_mat_Rplus(p: Poly, ctx=None,
        rounding=None, dps=15) -> Optional[Tuple[SOSElement, SOSElement]]:
    if ctx is None:
        ctx = NumpyCTX() if dps <= 15 else FlintCTX()
    if rounding is None:
        rounding = 2**((dps+1)//2) # if dps=15, then rounding=256
    rounding_bound = 2**int(round((dps + 1)*3.32))

    USa, USb = None, None
    p = p.to_field()

    def insert(l, v):
        # e.g. [a,b,c] -> [a,v,b,v,c]
        return [l[i//2] if i % 2 == 0 else v for i in range(len(l)*2-1)]
    def lc(l):
        return (l[0]) if l else ctx.float(1)

    with ctx.withdps(dps):
        coeffs = ctx.from_sympy_poly(p, dps)
        roots = ctx.polyroots(coeffs)
        if roots is None:
            return None
        real_roots, complex_roots = ctx.classify_roots(roots)
        if len(real_roots) == 0:
            res = _prove_univariate_from_mat_R(p, ctx=ctx, rounding=rounding, dps=dps)
            return (res, res.zero) if res is not None else None

        zero = ctx.float(0)
        d = len(real_roots)
        real_part = [ctx.real(v) for v in ctx.polyfromroots(real_roots)]
        major_part = real_part[::2]   # containing the leading term
        minor_part = real_part[1::2] # not containing the leading term
        major_part = insert(major_part, zero)
        minor_part = insert(minor_part, zero)
        major_roots = ctx.polyroots(major_part)
        minor_roots = ctx.polyroots(minor_part)
        if major_roots is None or minor_roots is None:
            return None
        _major_real, major_roots = ctx.classify_roots(major_roots)
        _minor_real, minor_roots = ctx.classify_roots(minor_roots)
        if _major_real or _minor_real:
            return None

        Ma = ctx._mat_from_roots(major_roots + complex_roots, coeffs[0] * lc(major_part))
        Mb = ctx._mat_from_roots(minor_roots + complex_roots, coeffs[0] * lc(minor_part))
        rounding_i = 1
        while rounding_i < rounding_bound and (USa is None or USb is None):
            M2b = ctx._mat_rounding(Mb, rounding_i, p.domain)
            if len(coeffs) % 2 == 0:
                # p is odd-degree, the last term should match the constant term of p
                M2b[-1][-1] = p.rep.TC()
            USb = ctx._mat_congruence(M2b, p.domain)
            if USb is None:
                rounding_i *= rounding
                continue

            pb = ctx.mat_to_sympy_poly(M2b)
            if len(coeffs) % 2 == 1:
                # p is even-degree, the minor part should be multiplied by x
                pb.append(p.domain.zero)
            pb = [p.domain.zero] * (len(coeffs) - len(pb)) + pb
            pa = [v1 - v2 for v1, v2 in zip(p.rep.all_coeffs(), pb)]

            M2a = ctx._mat_rounding(Ma, rounding_i, p.domain)
            M2a = ctx._mat_correction(M2a, pa)
            USa = ctx._mat_congruence(M2a, p.domain)
            rounding_i *= rounding

    if USa is None or USb is None:
        return None
    if p.total_degree() % 2 == 1:
        USa, USb = USb, USa
    return ctx._build_sol(USa, p.gen), ctx._build_sol(USb, p.gen)


def _prove_univariate_sqrfree_R(p: Poly) -> Optional[SOSElement]:
    dps = 15
    res = _prove_univariate_from_mat_R(p, dps=dps)

    if res is not None:
        return res

    check = count_roots(p) # note that p is sqr-free
    if check != 0: # not nonnegative
        return None
    while dps < 200:
        # NOTE: increasing precision might not work due to instability?
        dps *= 2
        res = _prove_univariate_from_mat_R(p, dps=dps)
        if res is not None:
            return res

    # TODO: implement univsos1


def _prove_univariate_sqrfree_Rplus(p: Poly) -> Optional[Tuple[SOSElement, SOSElement]]:
    dps = 15
    res = _prove_univariate_from_mat_Rplus(p, dps=dps)

    if res is not None:
        return res

    check = count_roots(p, 0, None) # note that p is sqr-free
    if check != 0: # not nonnegative
        return None
    while dps < 200:
        dps *= 2
        res = _prove_univariate_from_mat_Rplus(p, dps=dps)
        if res is not None:
            return res


class _SOSElement_Rplus:
    """Represent a + b*x"""
    def __init__(self, x, a: SOSElement, b: Optional[SOSElement]=None):
        self.x = x
        self.a = a
        self.b = b if b is not None else a.zero
    def __mul__(self, other) -> '_SOSElement_Rplus':
        a, b, c, d = self.a, self.b, other.a, other.b
        return _SOSElement_Rplus(self.x, a*c + b*d.mul_sqr(self.x), a*d + b*c)
    def __iter__(self):
        return iter([self.a, self.b])
    @classmethod
    def prod(cls, l: List['_SOSElement_Rplus']) -> '_SOSElement_Rplus':
        # be sure that len(l) > 0
        e = l[0]
        for v in l[1:]:
            e = e * v
        return e


def _prove_univariate(p: Poly, over_positive_reals=False) -> Optional[Tuple[SOSElement, SOSElement]]:
    if p.LC() < 0:
        return None
    p = p.to_field()

    if over_positive_reals:
        if p.EC() < 0: # ending coeff
            return None
        sqrfree_solver = _prove_univariate_sqrfree_Rplus
    else:
        if p.total_degree() % 2 == 1:
            return None
        sqrfree_solver = _prove_univariate_sqrfree_R

    c, rest = p.rep.factor_list()

    cone = SOSCone(p.domain[p.gen], p.domain)
    one = cone.algebra.one
    xgen = cone.algebra(p.gen)
    elems = [_SOSElement_Rplus(xgen, SOSElement.new(cone, [(c, one)]))]

    for q, mul in rest:
        if mul % 2 == 1:
            proof = sqrfree_solver(Poly.new(q, p.gen))
            if proof is None:
                return None
            if not over_positive_reals:
                proof = (proof, proof.zero)
            elems.append(_SOSElement_Rplus(xgen, proof[0], proof[1]))
        if mul > 1:
            q_alg = one.ring.from_dict(q.to_dict())
            sqr = SOSElement.new(cone,
                [(cone.domain.one, q_alg**(mul//2))]) # domain.one != one
            elems.append(_SOSElement_Rplus(xgen, sqr))
    prod = _SOSElement_Rplus.prod(elems)
    return prod.a, prod.b


def prove_univariate(poly: Union[Poly, Expr, List], interval: Tuple[Optional[Expr], Optional[Expr]]=(None, None),
    return_type = 'expr'
) -> Optional[Union[Expr, List[Tuple[Expr, Poly]], List[Tuple[Expr, SOSlist]]]]:
    """
    Prove a univariate polynomial to be nonnegative over an interval.

    Parameters
    ----------
    poly : Poly | Expr | List
        The univariate polynomial to be proven.
        If a list, the leading coefficient should be in the front.
    interval : Tuple[Optional[Expr], Optional[Expr]]
        The interval where `poly >= 0` is to be proven.
        Use None or sympy.Infinity or sympy.NegativeInfinity to represent infinity.
    return_type : str
        The type of the return value. Currently supports 'expr', 'list', and 'soslist'.

    Returns
    ----------
    * Returns None when the polynomial is not nonnegative over the interval, or when
    there is a numerical issue that the function fails to find a certificate.

    * If return_type is 'expr', the return value is a sympy expression.

    * If return_type is 'list', the return value is in the form of:
        `[(u, [(c_0, p_0), ..., (c_a, p_a)]),
          (v, [(d_0, q_0), ..., (d_b, q_b)])]`
    where `c_0, ..., c_a, d_0, ..., d_b` are constants, `p_0, ..., p_a, q_0, ..., q_b`
    are sympy polynomials and
        `poly = u * sum(c_i * p_i**2) + v * sum(d_j * q_j**2)`
    The rule for `u` and `v` are as follows. Suppose the generator symbol of poly is `x`,
    then:
        + If `interval = (-oo, oo)`, then `(u, v) = (1, x)`,
            and the second list must be empty, i.e., b = 0.
        + If `interval = (l, oo)`, then `(u, v) = (1, x - l)`.
        + If `interval = (-oo, r)`, then `(u, v) = (1, r - x)`,
        + If `interval = (l, r)`, then `(u, v) = (x - l, r - x)` if the polynomial has odd
        degree; and `(u, v) = (1, (x - l)*(r - x))` if the polynomial has even degree.

    * If return_type is 'soslist', the return value is in the form of `[(u, s_a), (v, s_b)]`
    where `s_a, s_b` are SOSlist instances. See `SOSlist` for details.

    Examples
    ----------
    >>> from sympy.abc import x
    >>> from sympy import Rational
    >>> prove_univariate(2*x**4 - 2*x + 1) # doctest: +SKIP
    2*(x - 1/2)**2 + 2*(x**2 - 1/2)**2
    >>> prove_univariate(2*x**4 - 2*x + 1).expand()
    2*x**4 - 2*x + 1

    Use the `interval` parameter to specify the interval where the polynomial is to be proven.
    Use None (or sympy.oo, -sympy.oo) to represent the infinity.

    >>> prove_univariate(x**3 - x + 1, (0, None)) # doctest: +SKIP
    x*((x - 1/2)**2 + 3/4) + (x - 1)**2
    >>> prove_univariate(2*x**4 - 3*x**3 + Rational(1,2), (None, 0)) # doctest: +SKIP
    -x*(4*(-x - 1/4)**2 + 3/4) + 15*(-x - 4/15)**2/8 + 2*(x**2 + x/4)**2 + 11/30
    >>> prove_univariate(x**3/3 - 2*x + 1, (-2, 0)) # doctest: +SKIP
    -x*(7*x**2/24 + 7*(x/2 + 1)**2/2) + (x + 2)*(11*x**2/8 + (x/2 + 1)**2/2)

    The function also accepts type `list` as input. In this case, the leading coefficient
    should be in the front.

    >>> prove_univariate([2, -4, 0, 4, 3]) # doctest: +SKIP
    2*x**2 + 2*(x**2 - x - 1)**2 + 1
    >>> prove_univariate([2, -4, 0, 4, 3]).expand()
    2*x**4 - 4*x**3 + 4*x + 3

    Use `return_type = 'list'` to obtain weighted sum-of-squares lists as outputs.

    >>> prove_univariate([2, -4, 0, 4, 3], return_type='list') # doctest: +SKIP
    [(1,
      [(2, Poly(x**2 - x - 1, x, domain='QQ')),
       (2, Poly(x, x, domain='QQ')),
       (1, Poly(1, x, domain='QQ'))]),
     (x, [])]
    >>> prove_univariate(x**3 - x + 1, (0, None), return_type='list') # doctest: +SKIP
    [(1, [(1, Poly(x - 1, x, domain='QQ'))]),
     (x, [(1, Poly(x - 1/2, x, domain='QQ')), (3/4, Poly(1, x, domain='QQ'))])]
    >>> prove_univariate(x**3/3 - 2*x + 1, (-2, 0), return_type='list') # doctest: +SKIP
    [(x + 2,
      [(1/2, Poly(1/2*x + 1, x, domain='QQ')),
       (11/2, Poly(-1/2*x, x, domain='QQ'))]),
     (-x,
      [(7/2, Poly(1/2*x + 1, x, domain='QQ')),
       (7/6, Poly(-1/2*x, x, domain='QQ'))])]
    >>> prove_univariate(-2*x**4 - 2*x**3 + 2*x**2 + 2*x + 1, (-1, 1), return_type='list') # doctest: +SKIP
    [(1,
      [(1, Poly(3/8*x**2 + 1/2*x + 1/8, x, domain='QQ')),
       (15/4, Poly(-19/60*x**2 + 2/15*x + 11/60, x, domain='QQ')),
       (11/15, Poly(1/4*x**2 - 1/2*x + 1/4, x, domain='QQ'))]),
     ((1 - x)*(x + 1),
      [(21, Poly(1/3*x + 1/6, x, domain='QQ')),
       (11/3, Poly(-1/4*x + 1/4, x, domain='QQ'))])]

    ## Algorithms

    * _prove_univariate_from_mat_R

    Recall if a polynomial `p` is positive over R, then all its roots are paired in
    complex conjugate pairs. Suppose `z_1, ..., z_{n/2}` and their conjugates are
    the the roots of `p`, then:
    ```
        p(x) = prod(x - z_i)(x - conj(z_i)) = |prod(x - z_i)|**2
        = im(prod(x - z_i))**2 + re(prod(x - z_i))**2
    ```
    This implies a numerical sum-of-squares certificate of `p` and also a positive
    semidefinite Gram matrix of rank 2. If we replace some `z_i` by its conjugate,
    then we will obtain new solutions and new Gram matrices. By performing a convex
    combination of these Gram matrices, we will obtain a Gram matrix that is
    full rank and strictly positive definite. Then it is possible to perturb and round
    the Gram matrix to obtain a rational sum-of-squares certificate.
    """
    if not (return_type in ('expr', 'list', 'soslist')):
        raise ValueError(f"The return_type must be 'expr', 'list', or 'soslist', but got {return_type}.")

    if not isinstance(poly, Poly):
        gens = ()
        if isinstance(poly, Expr):
            gens = poly.free_symbols
            if len(gens) == 0:
                gens = (Symbol("x"),)
            elif len(gens) > 1:
                raise MultivariatePolynomialError(f"Poly must be univariate, but got {poly}.")
        elif isinstance(poly, (list, dict, tuple, int, float)):
            gens = (Symbol("x"),)
        poly = Poly(poly, *gens, extension=True)
    if len(poly.gens) != 1:
        raise MultivariatePolynomialError(f"Poly must be univariate, but got {poly}.")
    if poly.domain.is_EX or poly.domain.is_EXRAW:
        poly = Poly(poly.expr, poly.gen, extension=True)

    l = interval[0] if interval[0] is not None else S.NegativeInfinity
    r = interval[1] if interval[1] is not None else S.Infinity

    if l >= r:
        raise ValueError(f"Interval must be well-defined (l < r), but got l={l}, r={r}.")

    is_infinite = lambda x: x is S.NegativeInfinity or x is S.Infinity
    dom, _ = construct_domain(
        [_  if not is_infinite(_) else 0 for _ in [l, r]], extension=True)
    dom = poly.domain.get_field().unify(dom)
    poly = poly.set_domain(dom)

    dom = poly.domain
    convert = lambda z: dom.convert(z) if not (z is None or is_infinite(z)) else z
    dl, dr = convert(l), convert(r) # do not convert from `construct_domain`
    gen = poly.gen
    def from_list(vec: List) -> Poly:
        return Poly.new(poly.rep.from_list(vec, 0, dom), gen)
    def from_dict(vec: dict) -> Poly:
        return Poly.new(poly.rep.from_dict(vec, 0, dom), gen)


    res = None
    multipliers = (Integer(1), Integer(1))
    if l is S.NegativeInfinity and r is S.Infinity:
        res = _prove_univariate(poly, over_positive_reals=False)
        multipliers = (Integer(1), gen)

    elif (l is not S.NegativeInfinity) and r is S.Infinity:
        res = _prove_univariate(poly.shift(dl), over_positive_reals=True)
        if res is None:
            return None
        res = res[0].applyfunc(lambda x: x.shift(-dl)), res[1].applyfunc(lambda x: x.shift(-dl))
        multipliers = (Integer(1), gen - l)

    elif (l is S.NegativeInfinity) and (r is not S.Infinity):
        def neg(f: Poly) -> Poly:
            """Compute f(-x)"""
            vec = f.rep.to_list()
            new_vec = [v if i % 2 == 1 else -v for i, v in enumerate(vec, start=len(vec)%2)]
            return from_list(new_vec)

        def neg2(f: PolyElement) -> PolyElement:
            """Compute f(-x) where type(f) == PolyElement"""
            new_rep = {(k,): v if k % 2 == 0 else -v for (k,), v in f.items()}
            return f.ring.from_dict(new_rep)

        # mirror (l, r) to (-r, -l)
        res = _prove_univariate(neg(poly).shift(-dr), over_positive_reals=True)
        if res is None:
            return None
        res = res[0].applyfunc(lambda x: neg2(x.shift(dr))), res[1].applyfunc(lambda x: neg2(x.shift(dr)))
        multipliers = (Integer(1), r - gen)

    else:
        # l is not S.NegativeInfinity and r is not S.Infinity
        # y = (x - l)/(r - x)
        # x = (r*y + l)/(y + 1)
        d = poly.total_degree()
        def trans(f: Poly) -> Poly:
            """Compute f((r*y + l)/(y + 1)) * (y + 1)**d"""
            numer, denom = from_list([dr, dl]), from_list([dom.one, dom.one])
            return f.transform(numer, denom)

        if not poly.is_zero:
            def inv_trans(f: PolyElement, k: int = d//2) -> PolyElement:
                """Compute f((x - l)/(r - x)) / (y + 1)**(k) where type(f) == PolyElement
                Note that 1/(y + 1) = (r - x)/(r - l)
                """
                # convert polyelement to poly
                p0 = from_dict(dict(f))
                df = p0.total_degree()

                # be careful with the API of .transform
                numer, denom = from_list([dom.one, -dl]), from_list([-dom.one, dr])
                p = p0.transform(numer, denom)
                p = p * denom**(k - df)
                p = Poly.new(p.rep.mul_ground((dom.one/(dr - dl))**k), p.gen)

                # convert back to PolyElement
                return f.ring.from_dict(p.rep.to_dict())
        else:
            inv_trans = lambda f, k = d//2: f

        res = _prove_univariate(trans(poly), over_positive_reals=True)
        if res is None:
            return None

        # p * (y + 1)**d == res[0] + y*res[1] == res[0] + (x - l)/(r - x)*res[1]
        if d % 2 == 0:
            res = res[0].applyfunc(inv_trans), res[1].applyfunc(lambda x: inv_trans(x, d//2-1)/(dr-dl))
            multipliers = (Integer(1), (gen - l)*(r - gen))
        else:
            res = res[1].applyfunc(inv_trans)/(dr-dl), res[0].applyfunc(inv_trans)/(dr-dl)
            multipliers = (gen - l, r - gen)

    if res is None:
        return None

    if return_type == 'soslist':
        return [(multipliers[0], SOSlist.new(res[0])), (multipliers[1], SOSlist.new(res[1]))]
    elif return_type == 'list':
        def to_c_poly(elem) -> List[Tuple[Expr, Poly]]:
            return [(dom.to_sympy(c), from_dict(dict(v))) for c, v in elem]
        return [(multipliers[0], to_c_poly(res[0])), (multipliers[1], to_c_poly(res[1]))]
    # else:
    return multipliers[0] * res[0].as_expr() + multipliers[1] * res[1].as_expr()


def prove_univariate_interval(
    poly: Poly,
    interval: Tuple[Optional[Expr], Optional[Expr]] = (None, None),
    return_raw: bool = False,
    early_stop: bool = -1,
    n: int = 15
):
    """Deprecated. Use `prove_univariate` instead."""
    from warnings import warn
    warn("Deprecated: use `prove_univariate` instead.", DeprecationWarning, stacklevel=2)
    if not return_raw:
        return prove_univariate(poly, interval)
    # else:
    res = prove_univariate(poly, interval, return_type='list')
    if res is None:
        return None
    return [
        (res[0][0], [c for c, v in res[0][1]], [v for c, v in res[0][1]]),
        (res[1][0], [c for c, v in res[1][1]], [v for c, v in res[1][1]])
    ]
