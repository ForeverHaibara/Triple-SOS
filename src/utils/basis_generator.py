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
    _monoms = {} # map monoms to indices & map indices to monoms
    def __new__(cls, *args, **kwargs):
        if cls is MonomialReduction:
            # this is an abstract base class, use MonomialFull by default
            return super().__new__(MonomialFull)
        return super().__new__(cls)

    @classmethod
    def from_poly(self, poly: sp.Poly) -> 'MonomialReduction':
        raise NotImplementedError

    def _register_monoms(self, nvars: int, degree: int) -> None:
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
        # generate all (a0,a1,...,an) such that n = nvars and sum(ai) <= degree
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
        return monom

    def is_standard_monom(self, monom: Tuple[int, ...]) -> bool:
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
        return [monom]

    def arraylize(self, poly: sp.Poly, expand_cyc: bool = False) -> np.ndarray:
        return np.array(self._arraylize_list(poly, expand_cyc = expand_cyc)).astype(np.float64)

    def arraylize_sp(self, poly: sp.Poly, expand_cyc: bool = False) -> sp.Matrix:
        return sp.Matrix(self._arraylize_list(poly, expand_cyc = expand_cyc))
    
    def invarraylize(self, array: Union[List, np.ndarray, sp.Matrix], gens: List[sp.Symbol]) -> sp.Poly:
        raise NotImplementedError

    def cyclic_sum(self, expr: sp.Expr, gens: List[sp.Symbol]) -> sp.Expr:
        raise NotImplementedError


class MonomialHomogeneous(MonomialReduction):
    _monoms = {} # map monoms to indices & map indices to monoms
    def __new__(cls, *args, **kwargs):
        if cls is MonomialHomogeneous:
            # this is an abstract base class, use MonomialHomogeneousFull by default
            return super().__new__(MonomialHomogeneousFull)
        return super().__new__(cls)

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

    def invarraylize(self, array: Union[List, np.ndarray, sp.Matrix], gens: List[sp.Symbol]) -> sp.Poly:
        # automatic infer the nvars and degrees: ((n - 1) + d)! / (n - 1)! / d! = len(array)
        nvars = len(gens)
        degree = _solve_combinatorial_equation(nvars - 1, len(array))
        if degree == -1:
            raise ValueError(f"Array cannot be of length {len(array)} with {nvars} gens.")
        inv_monoms = self.inv_monoms(nvars, degree)
        terms_dict = dict(zip(inv_monoms, array))
        return sp.Poly(terms_dict, gens)


class MonomialFull(MonomialReduction):
    _monoms = {} # map monoms to indices & map indices to monoms

    def is_standard_monom(self, monom: Tuple[int]) -> bool:
        return True

    def invarraylize(self, array: Union[List, np.ndarray, sp.Matrix], gens: List[sp.Symbol]) -> sp.Poly:
        # automatic infer the nvars and degrees: (n + d)! / n! / d! = len(array)
        nvars = len(gens)
        degree = _solve_combinatorial_equation(nvars, len(array))
        inv_monoms = self.inv_monoms(nvars, degree)
        terms_dict = dict(zip(inv_monoms, array))
        return sp.Poly(terms_dict, gens)

    def cyclic_sum(self, expr: sp.Expr, gens: Tuple[sp.Symbol]) -> sp.Expr:
        return expr


class MonomialHomogeneousFull(MonomialHomogeneous, MonomialFull):
    _monoms = {} # map monoms to indices & map indices to monoms
    def is_standard_monom(self, monom: Tuple[int]) -> bool:
        return True

    def _generate_monoms(self, nvars: int, degree: int) -> Tuple[Dict[Tuple[int, ...], int], List[Tuple[int, ...]]]:
        return MonomialHomogeneous._generate_monoms(self, nvars, degree)

    def index(self, monom: Tuple[int, ...], degree: Optional[int] = None) -> int:
        return MonomialHomogeneous.index(self, monom, degree)

    def invarraylize(self, array: Union[List, np.ndarray, sp.Matrix], gens: List[sp.Symbol]) -> sp.Poly:
        return MonomialHomogeneous.invarraylize(self, array, gens)


class MonomialCyclic(MonomialHomogeneous):
    _monoms = {}
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

    def _infer_degree(self, n: int, v: int) -> int:
        """
        Compute d such that len(dict_monoms[nvars][k]) == length.
        In fact, len(dict_monoms[n][d]) = T(n+k, k)
        where T(n,k) is defined in https://oeis.org/A047996.

        T(n, k) = 1/n * Sum_{d|(n,k)} phi(d) * binomial(n/d, k/d)

        Lower Bound: 
        T(n+k, k) >= 1/(n+k) * binomial(n+k,k) = binomial(n+k-1,k) / n
        """
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
        if issubclass(option, MonomialReduction):
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