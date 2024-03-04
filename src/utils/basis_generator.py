from typing import Union, Optional, Dict, List, Tuple, Callable

import numpy as np
import sympy as sp

def _integer_bisection(f: Callable[[int], int], left: int, right: int, v: int = 0) -> int:
    """
    Find the integer x such that f(x) <= v < f(x + 1).
    Function f should be monotonically increasing and left <= x <= right.
    """
    while left <= right:
        mid = (left + right) // 2
        if f(mid) < v:
            left = mid + 1
        elif f(mid) > v:
            right = mid - 1
        else:
            return mid
    return right if f(right) <= v else left # right < left

def _solve_combinatorial_equation(n: int, v: int) -> int:
    """
    Solve for x such that C(n + x, x) = v.
    """
    if v == 1:
        return 0
    elif n == 1:
        return v - 1
    elif n == 2:
        x = round(((8*v + 1)**.5 - 3)/2)
        return x if (x + 2) * (x + 1) == 2 * v else -1

    left, right = 0, v - n  # The range of x is [0, v - n].
    f = lambda x: sp.binomial(n + x, x)
    x = _integer_bisection(f, left, right, v)
    if f(x) == v:
        return x
    return -1


class MonomialReduction():
    """
    Reduction rules for monomials in polynomials according to the equivalence of symbols.

    For example,
    f(a,b,c) = (a^2 + b^2 + c^2)^2 - 3 * (a^3b + b^3c + c^3a)
    has the relation f(a,b,c) = f(b,c,a) = f(c,a,b).
    This implies the polynomial is invariant up to a permutation group.

    The class handles such permutation groups and reduce the number of
    degrees of freedom in the polynomial.
    """
    is_hom = False
    is_cyc = False
    _monoms = {} # map monoms to indices & map indices to monoms
    def __new__(cls, *args, **kwargs):
        if cls is MonomialReduction:
            # this is an abstract base class, use MonomialFull by default
            return super().__new__(MonomialFull)
        return super().__new__(cls)

    def base(self) -> 'MonomialReduction':
        """
        Return the base class of any reduction rule with permutation group.
        The base class does not contain any permutation group and is always
        a subclass of MonomialFull.
        """
        return MonomialReduction()

    @classmethod
    def from_poly(cls, poly: sp.Poly) -> 'MonomialReduction':
        raise NotImplementedError

    @classmethod
    def from_options(cls, **options) -> 'MonomialReduction':
        return _parse_options(**options)

    def _register_monoms(self, nvars: int, degree: int) -> Tuple[Dict[Tuple[int, ...], int], List[Tuple[int, ...]]]:
        """
        Register dict_monoms and inv_monoms for nvars and degree if not computed.
        """
        if nvars < 0 or degree < 0:
            return {}, []
        x = self._monoms.get(nvars, None)
        if x is None:
            self._monoms[nvars] = {}
        else:
            x = x.get(degree, None)
        if x is None:
            x = self._generate_monoms(nvars, degree)
            self._monoms[nvars][degree] = x
        return x

    def _generate_monoms(self, nvars: int, degree: int) -> Tuple[Dict[Tuple[int, ...], int], List[Tuple[int, ...]]]:
        """
        Generate all tuples (a0,a1,...,an) such that n = nvars and sum(ai) <= degree.
        """
        def generate_tuples(current_tuple: Tuple[int, ...], current_sum: int, remaining_vars: int) -> List[Tuple[int, ...]]:
            if remaining_vars == 0:
                return [current_tuple]
            else:
                tuples = []
                for i in range(degree - current_sum, -1, -1):
                    tuples.extend(generate_tuples(current_tuple + (i,), current_sum + i, remaining_vars - 1))
                return tuples

        inv_monoms = generate_tuples((), 0, nvars)
        inv_monoms = list(filter(self.is_standard_monom, inv_monoms))
        dict_monoms = {t: i for i, t in enumerate(inv_monoms)}
        return dict_monoms, inv_monoms

    def dict_monoms(self, nvars: int, degree: int) -> Dict[Tuple[int, ...], int]:
        return self._register_monoms(nvars, degree)[0]

    def inv_monoms(self, nvars: int, degree: int) -> List[Tuple[int, ...]]:
        return self._register_monoms(nvars, degree)[1]

    def _standard_monom(self, monom: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        For monoms in cyclic groups, only one representative is needed.
        For example, a^3b^2c, b^3c^2a, and c^3a^2b are equivalent in MonomialCyclic,
        and the standard monom (the representative) is a^3b^2c. The function
        returns the standard monom for the input monom.
        """
        return monom

    def is_standard_monom(self, monom: Tuple[int, ...]) -> bool:
        """
        For monoms in cyclic groups, only one representative is needed.
        For example, a^3b^2c, b^3c^2a, and c^3a^2b are equivalent in MonomialCyclic,
        and the standard monom (the representative) is a^3b^2c. The function
        returns True if the input monom is the standard monom.
        """
        return monom == self._standard_monom(monom)

    def index(self, monom: Tuple[int, ...], degree: int) -> int:
        nvars = len(monom)
        return self.dict_monoms(nvars, degree)[self._standard_monom(monom)]

    def _arraylize_list(self, poly: sp.Poly, expand_cyc: bool = False) -> List[int]:
        nvars = len(poly.gens)
        degree = poly.total_degree()
        dict_monoms = self.dict_monoms(nvars, degree)
        array = [0] * len(dict_monoms)
        if not expand_cyc:
            for monom, coeff in poly.terms():
                v = dict_monoms.get(monom, -1)
                if v != -1:
                    array[v] = coeff
        else:
            for monom, coeff in poly.terms():
                for monom2 in self.permute(monom):
                    v = dict_monoms.get(monom2, -1)
                    if v != -1:
                        array[v] += coeff
        return array

    def permute(self, monom: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """
        Permute the monom according to the permutation group.
        """
        return [monom]

    def permute_vec(self, nvars: int, vec: sp.Matrix, full: bool = True) -> sp.Matrix:
        """
        Permute the array representation of a polynomial according to the permutation group.
        """
        if vec.shape[1] != 1:
            raise ValueError("Input vec should be a column vector.")
        return vec

    def arraylize(self, poly: sp.Poly, expand_cyc: bool = False) -> np.ndarray:
        return np.array(self._arraylize_list(poly, expand_cyc = expand_cyc)).astype(np.float64)

    def arraylize_sp(self, poly: sp.Poly, expand_cyc: bool = False) -> sp.Matrix:
        return sp.Matrix(self._arraylize_list(poly, expand_cyc = expand_cyc))

    def _infer_degree(self, nvars: int, length: int) -> int:
        raise NotImplementedError

    def invarraylize(self, array: Union[List, np.ndarray, sp.Matrix], gens: List[sp.Symbol]) -> sp.Poly:
        raise NotImplementedError

    def cyclic_sum(self, expr: sp.Expr, gens: List[sp.Symbol]) -> sp.Expr:
        """
        Sum up the expression according to the permutation group.
        """
        raise NotImplementedError


class MonomialHomogeneous(MonomialReduction):
    is_hom = True
    is_cyc = False
    _monoms = {} # map monoms to indices & map indices to monoms
    def __new__(cls, *args, **kwargs):
        if cls is MonomialHomogeneous:
            # this is an abstract base class, use MonomialHomogeneousFull by default
            return super().__new__(MonomialHomogeneousFull)
        return super().__new__(cls)

    def base(self) -> 'MonomialHomogeneous':
        return MonomialHomogeneous()

    def _register_monoms(self, nvars: int, degree: int) -> None:
        if nvars == 0:
            if degree == 0:
                return {(): 0}, [()]
            return {}, []
        return super()._register_monoms(nvars, degree)

    def _generate_monoms(self, nvars: int, degree: int) -> Tuple[Dict[Tuple[int, ...], int], List[Tuple[int, ...]]]:
        # generate all (a0,a1,...,an) such that n = nvars and sum(ai) == degree
        def generate_tuples(current_tuple: Tuple[int, ...], current_sum: int, remaining_vars: int) -> List[Tuple[int, ...]]:
            if remaining_vars == 1:
                return [current_tuple + (degree - current_sum,)]
            else:
                tuples = []
                for i in range(degree - current_sum, -1, -1):
                    tuples.extend(generate_tuples(current_tuple + (i,), current_sum + i, remaining_vars - 1))
                return tuples

        inv_monoms = generate_tuples((), 0, nvars)
        inv_monoms = list(filter(self.is_standard_monom, inv_monoms))
        dict_monoms = {t: i for i, t in enumerate(inv_monoms)}
        return dict_monoms, inv_monoms

    def index(self, monom: Tuple[int, ...], degree: Optional[int] = None) -> int:
        nvars = len(monom)
        degree = sum(monom) if degree is None else degree
        return self.dict_monoms(nvars, degree)[self._standard_monom(monom)]

    def _infer_degree(self, nvars: int, length: int) -> int:
        return _solve_combinatorial_equation(nvars - 1, length)

    def invarraylize(self, array: Union[List, np.ndarray, sp.Matrix], gens: List[sp.Symbol]) -> sp.Poly:
        # automatic infer the nvars and degrees: ((n - 1) + d)! / (n - 1)! / d! = len(array)
        nvars = len(gens)
        degree = self._infer_degree(nvars, len(array))
        if degree == -1:
            raise ValueError(f"Array cannot be of length {len(array)} with {nvars} gens.")
        inv_monoms = self.inv_monoms(nvars, degree)
        terms_dict = dict(zip(inv_monoms, array))
        return sp.Poly(terms_dict, gens)


class MonomialFull(MonomialReduction):
    _monoms = {} # map monoms to indices & map indices to monoms

    def base(self) -> 'MonomialFull':
        return MonomialFull()

    def is_standard_monom(self, monom: Tuple[int]) -> bool:
        return True

    def _infer_degree(self, nvars: int, length: int) -> int:
        return _solve_combinatorial_equation(nvars, length)

    def invarraylize(self, array: Union[List, np.ndarray, sp.Matrix], gens: List[sp.Symbol]) -> sp.Poly:
        # automatic infer the nvars and degrees: (n + d)! / n! / d! = len(array)
        nvars = len(gens)
        degree = self._infer_degree(nvars, len(array))
        inv_monoms = self.inv_monoms(nvars, degree)
        terms_dict = dict(zip(inv_monoms, array))
        return sp.Poly(terms_dict, gens)

    def cyclic_sum(self, expr: sp.Expr, gens: Tuple[sp.Symbol]) -> sp.Expr:
        return expr


class MonomialHomogeneousFull(MonomialHomogeneous, MonomialFull):
    _monoms = {} # map monoms to indices & map indices to monoms

    def base(self) -> 'MonomialHomogeneousFull':
        return MonomialHomogeneousFull()

    def is_standard_monom(self, monom: Tuple[int]) -> bool:
        return True

    def _generate_monoms(self, nvars: int, degree: int) -> Tuple[Dict[Tuple[int, ...], int], List[Tuple[int, ...]]]:
        return MonomialHomogeneous._generate_monoms(self, nvars, degree)

    def index(self, monom: Tuple[int, ...], degree: Optional[int] = None) -> int:
        return MonomialHomogeneous.index(self, monom, degree)

    def _infer_degree(self, nvars: int, length: int) -> int:
        return MonomialHomogeneous._infer_degree(self, nvars, length)

    def invarraylize(self, array: Union[List, np.ndarray, sp.Matrix], gens: List[sp.Symbol]) -> sp.Poly:
        return MonomialHomogeneous.invarraylize(self, array, gens)


class MonomialCyclic(MonomialHomogeneous):
    is_cyc = True
    _monoms = {}

    def base(self) -> MonomialHomogeneousFull:
        return MonomialHomogeneousFull()

    def _standard_monom(self, monom: Tuple[int, ...]) -> Tuple[int, ...]:
        # first find the largest in the monom
        v = max(monom)
        best = None
        for i in range(len(monom)):
            if monom[i] == v:
                current = monom[i:] + monom[:i]
                if best is None or current > best:
                    best = current
        return best

    def permute(self, monom: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        return [monom[i:] + monom[:i] for i in range(len(monom))]

    def permute_vec(self, nvars: int, vec: sp.Matrix, full: bool = True) -> sp.Matrix:
        if vec.shape[1] != 1:
            raise ValueError("Input vec should be a column vector.")
        base = self.base()
        degree = base._infer_degree(nvars, vec.shape[0])
        dict_monoms = base.dict_monoms(nvars, degree)
        inv_monoms = base.inv_monoms(nvars, degree)
        p = lambda m: m[1:] + m[:1]
        f = lambda i: dict_monoms[p(inv_monoms[i])]
        if full:
            vecs = [vec]
            for _ in range(nvars - 1):
                new_vec = [0] * vec.shape[0]
                for i in range(vec.shape[0]):
                    new_vec[f(i)] = vec[i]
                vecs.append(sp.Matrix(new_vec))
                vec = vecs[-1]
            return sp.Matrix.hstack(*vecs)
        else:
            new_vec = [0] * vec.shape[0]
            for i in range(vec.shape[0]):
                new_vec[f(i)] = vec[i]
            return sp.Matrix(new_vec)

    def _infer_degree(self, nvars: int, length: int) -> int:
        """
        Compute d such that len(dict_monoms[nvars][k]) == length.
        In fact, len(dict_monoms[n][d]) = T(n+k, k)
        where T(n,k) is defined in https://oeis.org/A047996.

        T(n, k) = 1/n * Sum_{d|(n,k)} phi(d) * binomial(n/d, k/d)

        Lower Bound: 
        T(n+k, k) >= 1/(n+k) * binomial(n+k,k) = binomial(n+k-1,k) / n
        """
        n, v = nvars, length
        def func(k: int) -> int:
            x = self._monoms.get(n, None)
            if x is not None:
                x = x.get(k, None)
                if x is not None:
                    return len(x[0])
            return sum(
                sp.totient(d) * sp.binomial((n + k) // d, k // d)
                for d in sp.divisors(sp.gcd(n, k))
            ) // (n + k)
        left = 0
        right = _integer_bisection(lambda x: sp.binomial(n - 1 + x, x), 0, n*v - n + 1, n*v) + 1
        x = _integer_bisection(func, left, right, v)
        if func(x) == v:
            return x
        return -1 

    def invarraylize(self, array: Union[List, np.ndarray, sp.Matrix], gens: List[sp.Symbol]) -> sp.Poly:
        # automatic infer the nvars and degrees: 
        # ((n - 1) + d)! / (n - 1)! / d! <= len(array)
        nvars = len(gens)
        degree = self._infer_degree(nvars, len(array))
        if degree == -1:
            raise ValueError(f"Array cannot be of length {len(array)} with {nvars} gens.")
        inv_monoms = self.inv_monoms(nvars, degree)
        terms_dict = {}
        for coeff, monom in zip(array, inv_monoms):
            for i in range(nvars):
                monom2 = monom[i:] + monom[:i]
                terms_dict[monom2] = coeff
        return sp.Poly(terms_dict, gens)

    def cyclic_sum(self, expr: sp.Expr, gens: List[sp.Symbol]) -> sp.Expr:
        s = expr
        cyclic_replacement = {gens[i]: gens[(i+1)%len(gens)] for i in range(len(gens))}
        for i in range(1, len(gens)):
            expr = expr.xreplace(cyclic_replacement)
            s += expr
        return s


def _parse_options(**options) -> MonomialReduction:
    option = options.get('option', None)
    if option is not None:
        if isinstance(option, type) and issubclass(option, MonomialReduction):
            return option()
        elif isinstance(option, MonomialReduction):
            return option

    hom = options.get('hom', True)
    cyc = options.get('cyc', False)
    if cyc:
        cls = MonomialCyclic
    elif hom:
        cls = MonomialHomogeneousFull
    else:
        cls = MonomialFull
    return cls()


def arraylize(poly: sp.Poly, expand_cyc: bool = False, **options) -> np.ndarray:
    option = _parse_options(**options)
    return option.arraylize(poly, expand_cyc = expand_cyc)

def arraylize_sp(poly: sp.Poly, expand_cyc: bool = False, **options) -> sp.Matrix:
    option = _parse_options(**options)
    return option.arraylize_sp(poly, expand_cyc = expand_cyc)

def invarraylize(array: Union[List, np.ndarray, sp.Matrix], gens: List[sp.Symbol], **options) -> sp.Poly:
    option = _parse_options(**options)
    return option.invarraylize(array, gens)

def generate_expr(nvars: int, degree: int, **options) -> Tuple[Dict[Tuple[int, ...], int], List[Tuple[int, ...]]]:
    option = _parse_options(**options)
    return option._register_monoms(nvars, degree)