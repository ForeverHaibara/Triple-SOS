from typing import Optional, Union, List, Tuple

import numpy as np
import sympy as sp

from .rationalize import rationalize
from ..basis_generator import MonomialReduction, generate_expr

def _zip_power(gens: List[sp.Symbol], powers: List[int]) -> sp.Expr:
    return sp.Mul(*[g**p for g, p in zip(gens, powers)])

def _reg_matrix(M):
    # normalize so that the largest entry in each column is 1
    reg = np.max(np.abs(M), axis = 0)
    reg = 1 / np.tile(reg, (M.shape[0], 1))
    reg = sp.Matrix(reg)
    M = M.multiply_elementwise(reg)
    return M

def _algebraic_extension(vec):
    if isinstance(vec[0], sp.Rational):
        return [vec]
    field = vec[0].mod

    # if len(field) == 3 and field[0] == 1 and field[1] == 0:
    #     f = lambda x: sp.Rational(x.numerator, x.denominator)
    #     return [[f(x.rep[0])  if len(x.rep) == 2 else sp.S(0) for x in vec], 
    #             [f(x.rep[-1]) if len(x.rep) else sp.S(0) for x in vec]]

    f = lambda x: sp.Rational(x.numerator, x.denominator)
    vecs = []
    for i in range(1, len(field)):
        vecs.append([f(x.rep[-i]) if len(x.rep) >= i else sp.S(0) for x in vec])
    return vecs

def _derv(n: int, i: int) -> int:
    """
    Compute n! / (n-i)!.
    """
    if i == 1: return n
    if n < i:
        return 0
    return sp.factorial(n) // sp.factorial(n - i)


class Root():
    """
    Root of a multivariate polynomial.
    """
    __slots__ = ('root',)
    def __new__(cls, root):
        """
        Return RootRational if each entry of root is rational.
        """
        if len(root) == 3:
            return RootTernary.__new__(RootTernary, root)
        if all(isinstance(r, (sp.Rational, int)) for r in root):
            return object.__new__(RootRational)
        return object.__new__(cls)

    def __init__(self, root):
        root = tuple(sp.S(r) for r in root)
        self.root = root

    def __getitem__(self, i):
        return self.root[i]

    def __len__(self):
        return len(self.root)

    @property
    def nvars(self):
        return len(self.root)

    @property
    def is_corner(self):
        return sum(_ == 0 for _ in self.root) > 1

    @property
    def is_border(self):
        return sum(_ == 0 for _ in self.root) > 0

    @property
    def is_symmetric(self):
        return len(set(self.root)) < self.nvars

    @property
    def is_centered(self):
        p = self.root[0]
        return all(p == _ for _ in self.root[1:])

    @property
    def is_nontrivial(self):
        return not self.is_border and not self.is_symmetric

    def __str__(self):
        return str(self.root)

    def __repr__(self):
        return repr(self.root)

    def __hash__(self):
        return hash(self.root)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Root):
            return False
        return self.root == __value.root

    def eval(self, poly: sp.Poly, rational = False) -> sp.Expr:
        """
        Evaluate poly(*root). The function only accepts sympy
        polynomials as input. For more complicated usage,
        please use subs().

        Parameters
        ----------
        poly : sympy.Poly
            The polynomial to be evaluated on.
        rational : bool
            Whether to convert the root to a Rational number and evaluate. This
            boosts the performance.
        """
        root = self.root
        if rational:
            root = tuple(map(sp.Rational, root))
        return poly(*root)

    def as_vec(self,
        n: int,
        diff: Optional[Tuple[int, ...]] = None,
        numer: bool = False,
        **options
    ) -> Union[sp.Matrix, np.ndarray]:
        """
        Construct the vector of all monomials of degree n. For example,
        for nvars = 3, n = 3,
        return f(a,b,c) = [a^3,a^2*b,a^2*c,a*b^2,a*b*c,a*c^2,b^3,b^2*c,b*c^2,c^3].

        Parameters
        ----------
        n : int
            The degree of the monomials.
        diff : Optional[Tuple[int, ...]]
            The order of differentiation. If not None, it should have the same
            length as the number of variables. For example, (1, 0, 0) means
            differentiating the first variable once and the others zero times.
        numer : bool
            If numer, return a numpy array instead of sympy Matrix.
        options : dict
            Other options for the function generate_expr.
        """
        monoms = generate_expr(self.nvars, n, **options)[1]

        if diff is None:
            f = lambda m: _zip_power(self.root, m)
            vec = [f(m) for m in monoms]

        else:
            vec = [0] * len(monoms)
            for ind, monom in enumerate(monoms):
                if any(order_m < order_diff for order_m, order_diff in zip(monom, diff)):
                    continue
                dervs = [_derv(order_m, order_diff) for order_m, order_diff in zip(monom, diff)]
                powers = [order_m - order_diff for order_m, order_diff in zip(monom, diff)]
                vec[ind] = sp.prod(dervs) * _zip_power(self.root, powers)

        return np.array(vec).astype(np.float64) if numer else sp.Matrix(vec)

    def subs(self, expr: Union[sp.Expr, sp.Matrix], gens: Optional[List[sp.Symbol]] = None) -> Union[sp.Expr, sp.Matrix]:
        """
        Substitute the root into an expression or a matrix of expressions.
        """
        if gens is None:
            gens = expr.free_symbols
        return expr.subs(zip(gens, self.root))


class RootAlgebraic(Root):
    """
    Algebraic root of a multivariate polynomial.
    """
    __slots__ = ('root', 'K', 'root_anp', '_inv_sum', '_power_cache')
    def __new__(cls, root):
        if len(root) == 3:
            return RootAlgebraicTernary.__new__(RootAlgebraicTernary, root)
        if all(isinstance(r, (sp.Rational, int)) for r in root):
            return object.__new__(RootRational)
        return object.__new__(cls)

    def __init__(self, root):
        root = tuple(sp.S(r) for r in root)
        super().__init__(root)
        self.K = None
        self.root_anp = None
        for i in range(len(root)):
            if not isinstance(root[i], sp.Rational):
                self.K = sp.QQ.algebraic_field(root[i])
                try:
                    self.root_anp = [
                        self.K.from_sympy(r) for r in root
                    ]
                except Exception as e: # Coercion Failed
                    self.K = None
                if self.K is not None:
                    break
        if self.root_anp is None:
            self.root_anp = self.root
    
        if self.K is None:
            self._inv_sum = sp.S(1) / sum(self.root_anp)
        else:
            _sum = sum(self.root_anp)
            if _sum != self.K.zero:            
                self._inv_sum = self.K.from_sympy(sp.S(1)) / sum(self.root_anp)
            else:
                self._inv_sum = sp.oo
        self._power_cache = dict((key, {}) for key in range(-1, len(self.root)))

    def __str__(self):
        if self.K is None:
            return str(self.root)
        return str(tuple(r.n(15) if not isinstance(r, sp.Rational) else r for r in self.root))

    def __repr__(self):
        if self.K is None:
            return str(self.root)
        return str(tuple(r.n(15) if not isinstance(r, sp.Rational) else r for r in self.root))

    def to_sympy(self, *args, **kwargs):
        return self.K.to_sympy(*args, **kwargs)

    def from_sympy(self, *args, **kwargs):
        return self.K.from_sympy(*args, **kwargs)

    def _single_power(self, i: int, degree: int) -> sp.polys.polyclasses.ANP:
        """
        Return self.root_anp[i] ** degree.
        """
        k = self._power_cache[i].get(degree)
        if k is None:
            if i >= 0:
                self._power_cache[i][degree] = self.root_anp[i] ** degree
            elif i == -1:
                self._power_cache[i][degree] = self._inv_sum ** degree
            k = self._power_cache[i][degree]
        return k

    def _single_power_monomial(self, monomial: Tuple[int, ...]) -> sp.polys.polyclasses.ANP:
        return sp.prod([self._single_power(i, p) for i, p in enumerate(monomial)])
    
    def span(self, n: int, diff: Optional[Tuple[int, ...]] = None, **options) -> sp.Matrix:
        """
        Compute the rational span of the Root.as_vec(n, diff, **options).      
        """
        monoms = generate_expr(self.nvars, n, **options)[1]

        vec = [None] * len(monoms)
        _single_power = self._single_power_monomial

        if diff is None:
            for ind, monom in enumerate(monoms):
                vec[ind] = _single_power(monom)
        else:
            zero = self.K.zero
            for ind, monom in enumerate(monoms):
                if any(order_m < order_diff for order_m, order_diff in zip(monom, diff)):
                    vec[ind] = zero
                else:
                    dervs = [_derv(order_m, order_diff) for order_m, order_diff in zip(monom, diff)]
                    powers = [order_m - order_diff for order_m, order_diff in zip(monom, diff)]
                    vec[ind] = sp.prod(dervs) * _single_power(powers)

        # return sp.Matrix(vec)

        vecs_extended = _algebraic_extension(vec)         
        M = sp.Matrix(vecs_extended).T

        return _reg_matrix(M)

    def _subs_poly(self, poly: sp.Poly) -> sp.polys.polyclasses.ANP:
        """
        Substitute the root into a polynomial. This returns an ANP object,
        which is different from the general method self.eval(poly). This
        function is only available for RootAlgebraic class.
        """
        s = 0
        for monom, coeff in poly.terms():
            s += self._single_power_monomial(monom) * coeff
        return s

    def subs(self, expr: Union[sp.Expr, sp.Matrix], gens: List[sp.Symbol], to_sympy = True) -> Union[sp.Expr, sp.Matrix]:
        """
        Substitute the root into an expression or a matrix of expressions.
        """
        is_single_expr = not isinstance(expr, sp.MatrixBase)
        if is_single_expr:
            expr = [expr]

        results = []
        for expr0 in expr:
            try:
                if isinstance(expr0, sp.Poly):
                    s = self._subs_poly(expr0)
                else:
                    frac0, frac1 = sp.fraction(expr0.together())
                    frac0, frac1 = frac0.as_poly(*gens), frac1.as_poly(*gens)
                    s = self._subs_poly(frac0) / self._subs_poly(frac1)
            except:
                s = expr0.subs(zip(gens, self.root))
            results.append(s)

        if to_sympy:
            results = [self.K.to_sympy(r) for r in results]
        if is_single_expr:
            return results[0]
        if not to_sympy:
            return results # ANP objects cannot be converted to sympy Matrix
        return sp.Matrix(results).reshape(*expr.shape)


class RootRational(RootAlgebraic):
    __slots__ = ('root',)
    def __new__(cls, root):
        if len(root) == 3:
            return RootRationalTernary.__new__(RootRationalTernary, root)
        return object.__new__(cls)

    def __init__(self, root):
        Root.__init__(self, root)

    def span(self, n: int, diff: Optional[Tuple[int, ...]] = None, **options) -> sp.Matrix:
        return self.as_vec(n, diff, numer = False, **options)

    def __str__(self):
        return Root.__str__(self)

    def __repr__(self):
        return Root.__repr__(self)

    def subs(self, *args, **kwargs):
        return Root.subs(self, *args, **kwargs)


###################################################################
#
#
#                         Ternary roots
#
#
###################################################################

class TernaryMixin():
    """
    Abstract mixin class for roots of 3-var cyclic polynomials.
    """
    def __init__(self):
        if self.is_centered:
            self._uv = (sp.S(2), sp.S(2))
        elif self.is_corner:
            self._uv = (sp.oo, sp.oo)
        else:
            a, b, c = self.root if not hasattr(self, 'root_anp') else self.root_anp
            one = sp.S(1) if not hasattr(self, 'K') else self.K.one
            sa2b2 = a**2*b**2 + b**2*c**2 + c**2*a**2
            sa2bc = a*b*c * (a + b + c)
            sab3 = a*b**3 + b*c**3 + c*a**3
            sa3b = a**3*b + b**3*c + c**3*a
            inv_sa2b2_sa2bc = one / (sa2b2 - sa2bc)
            u = (sab3 - sa2bc) * inv_sa2b2_sa2bc
            v = (sa3b - sa2bc) * inv_sa2b2_sa2bc
            self._uv = (u, v)

        u, v = self._uv
        self._ker = u**2 - u*v + v**2 + u + v + 1


    @classmethod
    def from_uv(cls, u: sp.Expr, v: sp.Expr) -> 'RootTernary':
        """
        Construct a root from u, v.
        """
        u, v = sp.S(u), sp.S(v)
        ker = u**2 - u*v + v**2 + u + v + 1
        sab = (u + v - 1) / ker
        abc = (u*v - 1) / ker**2
        poly = sp.Poly([1, -1, sab, -abc], sp.Symbol('x'))
        if poly.domain is sp.QQ or poly.domain is sp.ZZ:
            a, b, c = poly.all_roots()
            cls = RootAlgebraicTernary
        elif issubclass(cls, RootAlgebraic):
            raise ValueError('RootAlgebraic.from_uv currently does not support irrational generating polynomials.')
        else:
            a, b, c = sp.polys.nroots(poly)
        root = cls((a, b, c))
        u_, v_ = root.uv()
        if abs(u_ - u) + abs(v_ - v) > abs(v_ - u) + abs(u_ - v):
            a, b, c = c, b, a
            root = cls((a, b, c))
        if not issubclass(root.__class__, RootAlgebraic):
            root._uv = (u, v)
            root._ker = ker
        return root


    def uv(self, to_sympy = True):
        uv = self._uv
        if hasattr(self, 'K') and self.K is not None and to_sympy:
            return (self.K.to_sympy(uv[0]), self.K.to_sympy(uv[1]))
        return uv

    def ker(self):
        """
        Return u^2 - uv + v^2 + u + v + 1.
        """
        ker = self._ker
        if hasattr(self, 'K') and self.K is not None:
            return self.K.to_sympy(ker)
        return ker

    def poly(self, x: sp.Symbol = None) -> sp.Poly:
        """
        Return a polynomial whose three roots are (proportional to) a, b, c.
        """
        x = x or sp.Symbol('x')
        poly = x**3 - x**2 + self.cyclic_sum((1,1,0)) * x - self.cyclic_sum((1,1,1)) / 3
        return poly.as_poly(x)

    def cyclic_sum(self, monom: Tuple[int, int, int], to_sympy = True) -> sp.Expr:
        """
        Return CyclicSum(a**i * b**j * c**k) assuming standardlization a + b + c = 1,
        that is, s(a^i*b^j*c^k) / s(a)^(i+j+k).

        Note: when i == j == k, it will return 3 * p(a^i). Be careful with the coefficient.
        """
        _cyclic_sum_cache = self._cyclic_sum_cache if hasattr(self, '_cyclic_sum_cache') else {}
        i, j, k = monom
        m = min(monom)
        u, v = self.uv()
        if m >= 1:
            # is multiple of abc
            return ((u*v - 1) / self.ker()**2) ** m * self.cyclic_sum((i-m, j-m, k-m))

        if k != 0:
            if j == 0:
                i, j, k = k, i, j
            elif i == 0:
                i, j, k = j, k, i
        if i == 0:
            i, j = j, i

        s = _cyclic_sum_cache.get((i,j,k), None)
        if s is not None:
            return s

        m = max(monom)
        if m >= 3:
            # can reduce the degree by poly remainder
            a, b = sp.symbols('a b')
            mod_poly = self.poly(a)
            poly_a = (a**i).as_poly(a) % mod_poly
            poly_b = (b**j).as_poly(b) % mod_poly.replace(a, b)
            s = sp.S(0)
            for term1 in poly_a.terms():
                for term2 in poly_b.terms():
                    s += term1[1] * term2[1] * self.cyclic_sum((term1[0][0], term2[0][0], 0))
            _cyclic_sum_cache[(i,j,k)] = s
            return s


        if i == 0:
            if j == 0:
                s = sp.S(3)
            elif j == 1:
                s = sp.S(1)
            elif j == 2:
                s = 1 - 2 * self.cyclic_sum((1,1,0))
        elif i == 1:
            if j == 0:
                s = sp.S(1)
            elif j == 1:
                s = (u + v - 1) / self.ker()
            elif j == 2:
                s = ((v - u) * (u*v + u + v - 2) + u**3 + 1) / self.ker()**2
        elif i == 2:
            if j == 0:
                s = 1 - 2 * self.cyclic_sum((1,1,0))
            elif j == 1:
                s = ((u - v) * (u*v + u + v - 2) + v**3 + 1) / self.ker()**2
            elif j == 2:
                s = (u**2 + v**2 - 2*(u+v) + 3) / self.ker()**2
        _cyclic_sum_cache[(i,j,k)] = s
        return s


    
    def standardize(self, cyc: bool = False, inplace: bool = False) -> 'Root':
        """
        Standardize the root by setting a + b + c == 3.

        Parameters
        ----------
        cyc : bool
            Whether to standardize cyclically. If True, return
            (a,b,c) such that a=max(a,b,c). If a == max(b,c), then
            return (a,b,c) such that b=max(b,c).

        inplace : bool
            Whether to modify the root inplace. If False, return
            a new Root object. If True, return self.

        Returns
        ----------
        root : Root
            The standardized root. If inplace is True, return self
            after modification.
        """
        if self.is_centered:
            root = (sp.S(1), sp.S(1), sp.S(1))
        else:
            s = sum(self.root) / 3
            root = (self.root[0] / s, self.root[1] / s, self.root[2] / s)
            if cyc:
                if not self.is_symmetric:
                    m = max(root)
                    for i in range(3):
                        if root[i] == m:
                            root = (root[i], root[(i+1)%3], root[(i+2)%3])
                            break
                else:
                    m = max(root)
                    for i in range(3):
                        if root[i] == m and root[(i+2)%3] < m:
                            root = (root[i], root[(i+1)%3], root[(i+2)%3])
                            break

        if inplace:
            self.root = root
            return self
        return RootTernary(root)


    def approximate(self, tolerance = 1e-3, approximate_tolerance = 1e-6):
        """
        Approximate the root of a ternary equation, especially it is on the border or symmetric axis.
        Currently only supports constant roots.
        """
        def n(a, b, tol = tolerance):
            # whether a, b are close enough
            return abs(a - b) < tol
        a, b, c = self.root
        s = (a + b + c) / 3
        r = [a/s, b/s, c/s]
        if n(r[0], r[1]) and n(r[1], r[2]) and n(r[0], r[2]):
            return RootRationalTernary((1,1,1))

        r = [0 if n(x, 0) else x for x in r]
        if sum(x != 0 for x in r) == 1:
            # two of them are zero
            r = [0 if x == 0 else 1 for x in r]
            return RootRationalTernary(r)

        perms = [(0,1,2),(1,2,0),(2,0,1)]
        for perm in perms:
            if n(r[perm[0]], r[perm[1]]):
                r[perm[2]] = r[perm[2]] / r[perm[0]]
                r[perm[0]] = r[perm[1]] = 1
            elif n(r[perm[0]], 0):
                r[perm[2]] = r[perm[2]] / r[perm[1]]
                r[perm[1]] = 1

        for i in range(3):
            v = rationalize(r[i], rounding=.1, reliable=False)
            if n(v, r[i], approximate_tolerance):
                r[i] = v

        return RootTernary((r[0], r[1], r[2]))


class RootTernary(Root, TernaryMixin):
    """
    Cyclic root of CyclicSum((a^2-b^2+u(ab-ac)+v(bc-ab))^2).
    Clearly, it should satisfy a^2-b^2+u(ab-ac)+v(bc-ab) = 0 and its permutations.
    For example, Vasile inequality is the case of (u, v) = (1, 2).

    When uv = 1, it degenerates to a root on border, (u, 0, 1) and permutations.
    When uv != 1, the root is computed by:
    x = ((v - u)(uv + u + v - 2) + u^3 + 1)/(1 - uv)
    y = ((u - v)(uv + u + v - 2) - v^3 - 1)/(1 - uv)

    Then b/c is a root of t^3 + xt^2 + yt - 1 = 0.
    And a/c is determined by a/c = ((b/c)^2 + (b/c)(u - v) - 1)/((b/c)u - v).
    """
    def __new__(cls, root):
        if all(isinstance(r, (sp.Rational, int)) for r in root):
            return object.__new__(RootRationalTernary)
        return object.__new__(cls)

    def __init__(self, root):
        Root.__init__(self, root)
        TernaryMixin.__init__(self)
        self._cyclic_sum_cache = {}


class RootAlgebraicTernary(RootAlgebraic, RootTernary):
    """
    Algebraic root of a 3-var homogeneous cyclic polynomial.
    """
    def __new__(cls, root):
        if all(isinstance(r, (sp.Rational, int)) for r in root):
            return object.__new__(RootRationalTernary)
        return object.__new__(cls)

    def __init__(self, root):
        RootAlgebraic.__init__(self, root)
        TernaryMixin.__init__(self)

    def cyclic_sum(self, monom: Tuple[int, int, int], to_sympy = True) -> Union[sp.Expr, sp.polys.polyclasses.ANP]:
        """
        Return CyclicSum(a**i * b**j * c**k) assuming standardlization a + b + c = 1,
        that is, s(a^i*b^j*c^k) / s(a)^(i+j+k).

        Note: when i == j == k, it will return 3 * p(a^i). Be careful with the coefficient.
        """
        s = 0
        _single_power = self._single_power
        for i in range(3):
            s = _single_power(0, monom[i%3]) * _single_power(1, monom[(i+1)%3]) * _single_power(2, monom[(i+2)%3]) + s
        s *= _single_power(-1, sum(monom))
        if self.K is not None and to_sympy:
            s = self.K.to_sympy(s)
        return s


class RootRationalTernary(RootRational, RootAlgebraicTernary):
    def __new__(cls, root):
        return object.__new__(cls)

    def __init__(self, root):
        RootRational.__init__(self, root)
        TernaryMixin.__init__(self)
    
    def cyclic_sum(self, monom: Tuple[int, int, int], to_sympy: bool = True) -> sp.Expr:
        """
        Return CyclicSum(a**i * b**j * c**k) assuming standardlization a + b + c = 1,
        that is, s(a^i*b^j*c^k) / s(a)^(i+j+k).

        Note: when i == j == k, it will return 3 * p(a^i). Be careful with the coefficient.
        """
        i, j, k = monom
        a, b, c = self.root
        return (a**i * b**j * c**k + a**k * b**i * c**j + a**j * b**k * c**i) / (a + b + c)**(i + j + k)