from typing import List, Dict, Tuple, Union, Set, Optional, TYPE_CHECKING

from sympy import Expr, Poly, Rational, Add, Mul
from sympy.polys.polyerrors import BasePolynomialError

from ...utils.expressions import CyclicExpr

if TYPE_CHECKING:
    from sympy.combinatorics.perm_groups import PermutationGroup
    from sympy import Symbol

    from ..problem import InequalityProblem

    SIGNS_TYPE = Dict["Symbol", Tuple[Optional[int], Expr]]

def _is_double_rational(x):
    if isinstance(x, Rational) and (int(x.numerator) % 2 == 0 or
            int(x.denominator) % 2 == 0):
        return True
    return False

def is_nonneg_pow(x: Expr) -> bool:
    return _is_double_rational(x.exp)

def sgn_prod(signs: "SIGNS_TYPE") -> Optional[int]:
    if any(s == 0 for s in signs):
        return 0
    if any(s is None for s in signs):
        return None
    return 1 if sum([1 for s in signs if s < 0]) % 2 == 0 else -1


def _prove_poly(poly: Poly, signs: "SIGNS_TYPE", factor: bool=False) -> Optional[Expr]:
    """
    Helper function to decide the sign of a polynomial.

    Examples
    --------
    >>> from sympy.abc import a, b, c, x
    >>> _prove_poly((3*a**4*b**2 - 4*a**2*b**3).as_poly(a, b), {b: (-1, x)})
    3*a**4*b**2 + 4*a**2*b**2*x
    """
    if factor:
        const, factors = poly.factor_list()
        const = _prove_by_recur(const.as_expr(), signs)
        if const is None:
            return None
        proof = [const[0]]
        for part, mul in factors:
            if mul % 2 == 0:
                # TODO: check whether sgn(part) == 0
                proof.append(part.as_expr()**mul)
            else:
                part_proof = _prove_poly(part, signs, factor=False)
                if part_proof is None:
                    return None
                proof.append(part_proof * part.as_expr()**(mul - 1))
        return Mul(*proof)


    if poly.free_symbols_in_domain:
        poly = poly.inject()

    gens = poly.gens
    sign_gens = [signs.get(g, (None, None)) for g in gens]
    sgn = 0
    sgns = []
    for monom, coeff in poly.terms():
        s = sgn_prod([
            0 if a == 0 else (1 if m % 2 == 0 else a) \
                for m, (a, e) in zip(monom, sign_gens) if m > 0
        ])
        sgns.append(s)
        if s is None:
            return None
        if s == 0:
            continue
        if coeff < 0:
            s = -s
        if sgn != 0:
            if s != sgn:
                # has different signs among all terms
                return None
        else:
            # set sgn = 0 to sgn = s
            sgn = s
    if sgn < 0:
        return None

    terms = []
    default = lambda e, g: e if e is not None else g

    for s, (monom, coeff) in zip(sgns, poly.terms()):
        if s != 0:
            terms.append(Mul(abs(coeff), *[
                (g**m if m % 2 == 0 else g**(m-1)*default(e, g)) \
                    for m, g, (a, e) in zip(monom, gens, sign_gens) if m > 0
            ]))
        else:
            terms.append(Mul(coeff, *[
                (g**m if a != 0 else g**(m-1)*default(e, g)) \
                    for m, g, (a, e) in zip(monom, gens, sign_gens) if m > 0
            ]))
    return Add(*terms)


def _is_perm_invariant_signs(
    signs: "SIGNS_TYPE", symbols: Tuple["Symbol"], perm_group: "PermutationGroup", cache: dict
) -> bool:
    """
    Check if the signs are perm-invariant.
    """
    symbols = tuple(symbols)
    if (symbols, perm_group) in cache:
        return cache[(symbols, perm_group)]

    from ...utils.expressions.cyclic import _is_perm_invariant_dict
    directions = {k: s for k, (s, v) in signs.items() if s is not None}
    witnesses = {k: (v if v is not None else (-k if s == -1 else k))
        for k, (s, v) in signs.items() if s is not None}
    invariant = (_is_perm_invariant_dict(symbols, perm_group, directions)
        and _is_perm_invariant_dict(symbols, perm_group, witnesses))
    cache[(symbols, perm_group)] = invariant
    return invariant


class _SignProver:
    """Infer signs without recursively walking the expression tree.

    Results are ``(sign, proof)``, where sign is 1 (nonnegative), -1
    (nonpositive), 0 (zero), or None (unknown). Unlike entries in ``signs``,
    every successful proof represents the expression itself, including its
    sign. Zero proofs retain the equality witnesses rather than becoming 0.

    Each call owns its caches. Generator methods request child results from
    an explicit stack; they never recursively evaluate children or revisit
    the supplied witnesses. A separate zero query refines cheap sign results
    such as squares only when their vanishing is needed.
    """

    _UNKNOWN = (None, None)

    def __init__(self, signs, symmetry_cache=None):
        self.signs = signs
        self.results = {}
        self.numeric = {}
        self.symmetry_cache = {} if symmetry_cache is None else symmetry_cache

    def prove(self, expr):
        result = self._evaluate(expr, False)
        if result[0] is None or result[0] == -1:
            result = self._evaluate(expr, True)
        if result[0] is not None:
            return result[1], result[1] != expr
        return None

    def _evaluate(self, expr, zero):
        key = (expr, zero)
        if key in self.results:
            return self.results[key]
        stack = [(key, self._infer(expr, zero))]
        result = None
        while stack:
            key, worker = stack[-1]
            try:
                child = worker.send(result)
            except StopIteration as finished:
                result = finished.value
                self.results[key] = result
                stack.pop()
            else:
                if child in self.results:
                    result = self.results[child]
                else:
                    stack.append((child, self._infer(*child)))
                    result = None
        return result

    def _infer(self, expr, zero):
        sign, witness = self.signs.get(expr, self._UNKNOWN)
        if sign is not None and (not zero or sign == 0):
            proof = expr if witness is None else (-witness if sign == -1 else witness)
            return sign, proof
        if isinstance(expr, Rational):
            if expr == 0:
                return 0, expr
            return self._UNKNOWN if zero else (1 if expr > 0 else -1, expr)
        if expr.is_Symbol:
            return self._UNKNOWN

        if expr.is_Pow:
            result = yield from self._infer_power(expr, zero)
        elif expr.is_Add or expr.is_Mul:
            result = yield from self._infer_terms(expr, expr.args, expr.func, zero)
        elif isinstance(expr, CyclicExpr):
            result = yield from self._infer_cyclic(expr, zero)
        else:
            result = self._UNKNOWN

        if result[0] is None and self._is_number(expr) and expr.is_finite is not False:
            # Numerical expressions may need more than termwise reasoning.
            if expr.is_zero is True:
                return 0, expr
            if not zero:
                if expr.is_nonnegative is True:
                    return 1, expr
                if expr.is_nonpositive is True:
                    return -1, expr
        return result

    def _is_number(self, expr):
        # Avoid the recursive is_number property on deep symbolic expressions.
        # Only numerical candidates reach SymPy's more expensive assumptions.
        stack = [(expr, False)]
        while stack:
            node, visited = stack.pop()
            if node in self.numeric:
                continue
            if not node.args:
                self.numeric[node] = node.is_number
            elif not visited:
                stack.append((node, True))
                stack.extend((arg, False) for arg in node.args if arg not in self.numeric)
            elif not all(self.numeric[arg] for arg in node.args):
                self.numeric[node] = False
            else:
                self.numeric[node] = (node.is_Add or node.is_Mul or node.is_Pow
                    or node.is_number)
        return self.numeric[expr]

    def _infer_power(self, expr, zero):
        base, exponent = expr.args
        if zero:
            if exponent.is_positive is not True or exponent.is_real is not True:
                return self._UNKNOWN
            sign, proof = yield (base, True)
            if sign != 0:
                return self._UNKNOWN
            if proof == base:
                return 0, expr
            if exponent.is_Integer:
                return 0, base**(exponent - 1)*proof
            return 0, proof**exponent

        # Do not turn an explicitly zero denominator into a certificate.
        if exponent.is_negative and self.signs.get(base, self._UNKNOWN)[0] == 0:
            return self._UNKNOWN
        if is_nonneg_pow(expr):
            return 1, expr
        sign, proof = yield (base, False)
        if sign is None:
            return self._UNKNOWN
        if sign == 0:
            if exponent.is_positive is not True:
                return self._UNKNOWN
        elif sign == -1:
            # SymPy's fractional powers use the principal complex branch.
            if not exponent.is_Integer:
                return self._UNKNOWN
            sign = -1 if exponent % 2 else 1
        if proof == base:
            return sign, expr
        if exponent.is_Integer and exponent > 0:
            return sign, base**(exponent - 1)*proof
        return sign, proof**exponent

    def _combine(self, expr, args, results, operation):
        signs = [result[0] for result in results]
        if operation is Mul:
            sign = sgn_prod(signs)
            if sign == 0:
                # Replace all zero factors, without choosing an asymmetric anchor.
                proofs = [proof if s == 0 else arg
                    for arg, (s, proof) in zip(args, results)]
            else:
                proofs = [result[1] for result in results]
        else:
            directions = set(signs) - {0}
            if None in directions or len(directions) > 1:
                return self._UNKNOWN
            sign = next(iter(directions)) if directions else 0
            proofs = [result[1] for result in results]
        if sign is None:
            return self._UNKNOWN
        if all(proof == arg for arg, proof in zip(args, proofs)):
            return sign, expr
        return sign, operation(*proofs)

    def _infer_terms(self, expr, args, operation, zero):
        results = []
        for arg in args:
            result = yield (arg, zero)
            results.append(result)
            if zero and operation is Add and result[0] != 0:
                return self._UNKNOWN
        result = self._combine(expr, args, results, operation)
        if not zero and result[0] is None:
            # Unknown factors may be annihilated, and conflicting summands
            # may vanish. Failure to prove zero never means nonzero.
            for i, arg in enumerate(args):
                if results[i][0] != 0:
                    refined = yield (arg, True)
                    if refined[0] == 0:
                        results[i] = refined
            result = self._combine(expr, args, results, operation)
        return result

    @staticmethod
    def _intrinsic_nonnegative(expr):
        args = expr.args if expr.is_Mul else (expr,)
        return all((arg.is_Pow and is_nonneg_pow(arg))
            or (isinstance(arg, Rational) and arg >= 0) for arg in args)

    def _infer_cyclic(self, expr, zero):
        from ...utils.expressions.cyclic import _replace_symbols

        arg, symbols, group = expr.args
        if not zero and self._intrinsic_nonnegative(arg):
            return 1, expr
        if _is_perm_invariant_signs(self.signs, symbols, group, self.symmetry_cache):
            sign, proof = yield (arg, zero)
            if sign is None:
                return self._UNKNOWN
            if sign == -1:
                # Pull the common sign outside the wrapper. The product has
                # one factor per group element, including repeated orbit terms.
                direction = -1 if expr.base_func is Add else (-1)**group.order()
                return direction, direction*expr.func(-proof, symbols, group, evaluate=False)
            if proof == arg:
                return sign, expr
            return sign, expr.func(proof, symbols, group, evaluate=False)

        # Stream group elements rather than materializing group.elements.
        args = tuple(_replace_symbols(arg, dict(zip(symbols, perm(symbols))))
            for perm in group.generate())
        return (yield from self._infer_terms(expr, args, expr.base_func, zero))


def _prove_by_recur(expr: Expr, signs: "SIGNS_TYPE", cache=None) -> Optional[Tuple[Expr, bool]]:
    """Return a nonnegative certificate and whether it differs from ``expr``."""
    return _SignProver(signs, symmetry_cache=cache).prove(expr)


def sign_sos(expr: Union[Expr, Poly], signs: "SIGNS_TYPE", factor: bool = False) -> Optional[Expr]:
    """
    Very fast and simple nonnegativity check for a SymPy (commutative, real)
    expression instance given signs of symbols.

    Parameters
    ----------
    expr: Union[Expr, Poly]
        The expression to check.
    signs: Dict[Symbol, Tuple[int, Expr]]
        Signs of symbols stored in the format dict((symbol, (sign, expr))).
        The sign is 1 if the symbol is non-negative and -1 if the symbol is
        non-positive. The expr is the expression that the absolute value
        of the symbol is replaced with. If the sign is 0, the symbol
        is zero and the expr is the expression that the symbol is replaced
        with.

    Returns
    -------
    Optional[Expr]
        The nonnegatie certificate of the input expression.
        Returns None if the heuristic fails.

    Examples
    ---------
    >>> from sympy.abc import a, b, c
    >>> from sympy import Function
    >>> F = Function('F')
    >>> signs = {a: (1, F(a)), b: (1, b), c: (None, None)}
    >>> sign_sos(a*b**3*c**2 + b + (a - b)**2 + 2/a, signs)
    b**3*c**2*F(a) + b + (a - b)**2 + 2/F(a)
    >>> sign_sos(c + a**2, signs) is None
    True

    Note that the function bases on heuristics,
    and may fail on expressions that are nonnegative.

    >>> sign_sos(a**12 + b**12 + c**12 - 3*a**4*b**4*c**4, signs) is None
    True
    """
    if isinstance(expr, Poly):
        return _prove_poly(expr, signs, factor=factor)

    cache = {}
    sol = _prove_by_recur(expr.as_expr(), signs, cache=cache)
    if sol is not None:
        return sol[0]

    if factor:
        # check once more by factorization
        expr = expr.as_expr().doit().factor()
        sol = _prove_by_recur(expr, signs, cache=cache)
        if sol is not None:
            return sol[0]


class SpecialHeap:
    """
    An efficient data structure storing a list of sets `[x[0], ..., x[n-1]]`.
    It supports the following two operations:
    1. `remove(a, b)`: call x[a].remove(b).
    2. `pop()`: pop the set with minimum length.
    """
    def __init__(self, x: List[Set[int]]):
        self.heap = []
        self.pos = list(range(len(x))) # index of x[i] in heap

        for xi, yi in zip(x, self.pos):
            self.heap.append((xi, yi))
        for i in range((len(self.heap) - 2) // 2, -1, -1):
            self._bubble_down(i)

    def __len__(self):
        return len(self.heap)

    def _bubble_up(self, i):
        heap = self.heap
        while i > 0:
            parent = (i - 1) // 2
            if len(heap[i][0]) < len(heap[parent][0]):
                heap[i], heap[parent] = heap[parent], heap[i]
                y_i = heap[i][1]
                y_parent = heap[parent][1]
                self.pos[y_i] = i
                self.pos[y_parent] = parent
                i = parent
            else:
                break

    def _bubble_down(self, i):
        heap = self.heap
        n = len(heap)
        while True:
            left = 2 * i + 1
            right = 2 * i + 2
            smallest = i

            if left < n and len(heap[left][0]) < len(heap[smallest][0]):
                smallest = left
            if right < n and len(heap[right][0]) < len(heap[smallest][0]):
                smallest = right

            if smallest != i:
                heap[i], heap[smallest] = heap[smallest], heap[i]
                y_i = heap[i][1]
                y_smallest = heap[smallest][1]
                self.pos[y_i] = i
                self.pos[y_smallest] = smallest
                i = smallest
            else:
                break

    def remove(self, a: int, b: int):
        """
        Call x[a].remove(b)
        """
        i = self.pos[a]
        if i >= len(self.heap): # ignore (i is not in the heap)
            return
        xi, yi = self.heap[i]
        if yi != a: # ignore (i is not in the heap)
            return
        xi.remove(b)
        self._bubble_up(i)

    def pop(self) -> Tuple[Set[int], int]:
        """
        Get the element with the minimum length of x[i],
        and remove it from the heap.
        Returns (x[i], i).
        """
        min_xi, min_yi = self.heap[0]

        if len(self.heap) == 1:
            self.heap.pop()
            return (min_xi, min_yi)

        # fill the heap top with the last element, and bubble down
        last_element = self.heap.pop()
        self.heap[0] = last_element
        self.pos[last_element[1]] = 0
        self._bubble_down(0)

        return (min_xi, min_yi)



def _infer_separable_sign(poly: Poly, expr: Expr, s: int,
        signs: Dict[int, Tuple[int, Expr]], cmp: int = 1) -> Tuple[int, int, Expr]:
    """
    Check whether `poly` implies the sign of the symbol `poly.gens[s]` by
    rewriting `poly` as `r(...)*gens[s] + q(...) == expr (<>=) 0`.

    When cmp == 1, the constraint is `poly == expr >= 0`.
    When cmp == 0, the constraint is `poly == expr == 0`.

    Currently require:
    * `poly.gens[s]` is the only undetermined symbol in `signs`
    * `poly` is linear in `poly.gens[s]`.

    Returns `s, sgn, x` if inference succeeds, where
    * `x = abs(gens[s])` if `cmp == 1`;
    * `x = gens[s]` if `cmp == 0`.

    Returns None if inference fails.

    Examples
    ---------
    >>> from sympy.abc import a, b, c, x, y, z
    >>> p1 = (c*a - 2 - c**2 - b).as_poly(a,b,c)
    >>> signs = {1: (1, b), 2: (1, c)}
    >>> _infer_separable_sign(p1, x, 0, signs, 1)
    (0, 1, (b + c**2 + x + 2)/c)

    Note that the `expr` stands for its absolute value if the sign is -1:
    >>> p2 = (-(c + 1)*a - 2 - c**2 - b).as_poly(a,b,c)
    >>> _infer_separable_sign(p2, x, 0, signs, 1)
    (0, -1, (b + c**2 + x + 2)/(c + 1))

    When inference fails, return None.
    >>> p3 = (c*a + 2 - c**2 - b).as_poly(a,b,c)
    >>> _infer_separable_sign(p3, x, 0, signs, 1) is None
    True
    """
    if poly.degree(s) != 1:
        return None
    gens = poly.gens
    nvars = len(gens)
    def monom_sign(m, c, skip) -> Tuple[int, Expr]:
        """
        Returns (sgn(m*c), abs(m*c))
        """
        sgn = 1 if c > 0 else -1
        absm = [-c if sgn == -1 else c]
        for j in range(nvars):
            mj = m[j]
            if j == skip:
                continue
            absm.append(gens[j]**((mj//2)*2))
            if mj % 2 == 1:
                sj, e = signs[j]
                sgn *= sj
                absm.append(e)
        return sgn, Mul(*absm)

    qr = [[], []]
    qr_sign = [-cmp, 0]

    for monom, coeff in poly.terms():
        k = 0 if monom[s] == 0 else 1
        sgn, e = monom_sign(monom, coeff, s)
        if qr_sign[k] == -sgn and sgn != 0: # inconsistent
            return None
        if qr_sign[k] == 0 and sgn != 0:
            qr_sign[k] = sgn

        # qr stores the absolute value of each term
        qr[k].append(e)

    q_sign, r_sign = qr_sign
    if r_sign == 0: # division by zero
        return None

    absq = Add(*qr[0])
    absr = Add(*qr[1])

    if q_sign != 0:
        # note that we return the absolute value of gens[s]
        abss = (expr * (-q_sign) + absq) / absr
    else:
        abss = expr * r_sign / absr

    return s, -q_sign * r_sign, abss


def _get_signs_by_topological_order(ineq_constraints: Dict[Poly, Expr], eq_constraints: Dict[Poly, Expr],
        signs: "SIGNS_TYPE") -> "SIGNS_TYPE":
    """
    Algorithm to infer the signs of symbols from constraints by
    repeating the following process:
    1. Extract the constraint with fewest undetermined symbols.
    2. If the constraint has only one undetermined symbol,
        try to infer the sign of the symbol from the constraint.
    3. If the sign of the new symbol is determined,
        remove the symbol from undetermined symbols of each constraint.

    Parameters
    ----------
    ineq_constraints: Dict[Poly, Expr]
        The inequalities constraints.
    eq_constraints: Dict[Poly, Expr]
        The equalities constraints.
    signs: Dict[int, Tuple[Optional[int], Expr]]
        The signs of symbols. Note that the function modifies
        the `signs` in place.

    Note
    ----
    * The function modifies `ineq_constraints` and `eq_constraints`
    in place to remove constraints that are used or have no
    undetermined symbols.
    * The function modifies the `signs` in place.
    * The keys of `signs` are indices, not symbols.

    Returns
    -------
    signs: Dict[int, Tuple[Optional[int], Expr]]
        The signs of symbols. It is the input `signs`.
    """
    if len(ineq_constraints):
        gens = next(iter(ineq_constraints.keys())).gens
    elif len(eq_constraints):
        gens = next(iter(eq_constraints.keys())).gens
    else:
        return signs
    nvars = len(gens)

    def _is_separable(p: Poly) -> bool:
        """
        Check whether the polynomial can be written as x = q(...)/r(...)
        where q and r is independent with the variable x.
        This form helps infer the sign of the variable x.
        """
        if p.is_zero:
            return False
        return any(i == 1 for i in p.degree_list())

    neighbors = {i: set() for i in range(nvars)}
    cons = []
    cons_fs = []
    cons_cnt = 0
    for cmp, constraints in ((1, ineq_constraints), (0, eq_constraints)):
        for p, e in constraints.items():
            if not _is_separable(p):
                continue
            pd = p.degree_list()
            pfs = {i for i in range(nvars) if pd[i] > 0 and signs[i][0] is None}
            for s in pfs:
                neighbors[s].add(cons_cnt)
            cons.append((p, e, cmp))
            cons_fs.append(pfs)
            cons_cnt += 1

    # maintain a heap so that we can access i with the minimum len(cons_fs[i])
    heap = SpecialHeap(cons_fs)

    def infer(i):
        p, e, cmp = cons[i]
        s = next(iter(cons_fs[i]))
        return _infer_separable_sign(p, e, s, signs, cmp)

    # print('Constraints', constraints)
    while len(heap):
        fs, i = heap.pop()
        if len(fs) == 0:
            continue
        if len(fs) > 1:
            # cannot eliminate symbols anymore
            break
        # if len(fs) == 1:
        result = infer(i)
        if result is None:
            continue

        s_ind, sgn, abs_s = result
        signs[s_ind] = (sgn, abs_s)

        if cons[i][2] == 1:
            del ineq_constraints[cons[i][0]]
        else:
            del eq_constraints[cons[i][0]]

        # print(f'Removing Neighbors[{s_ind}] = {neighbors[s_ind]}:'
        #       f' {[cons[i] for i in neighbors[s_ind]]}')
        for j in list(neighbors[s_ind]):
            heap.remove(j, s_ind)
            neighbors[s_ind] = {}

    return signs


def get_symbol_signs(problem: "InequalityProblem") -> Dict["Symbol", Tuple[Optional[int], Expr]]:
    """
    Infer the signs of each symbol in the problem given inequality
    and equality constraints heuristically. It can also be called
    by the class method `InequalityProblem.get_symbol_signs()`.

    The inference is heuristic and incomplete.

    Returns
    -------
    Dict[Symbol, Tuple[Optional[int], Expr]]
        A dictionary mapping each symbol to a tuple of its sign and a
        representative expression. The sign is 1 if the symbol is nonnegative,
        -1 if the symbol is nonpositive, and 0 if the symbol is zero. The sign
        is None if the sign of the symbol is undetermined.

        If sign == 1 or -1, Expr is an nonnegative expression equal to the ABSOLUTE VALUE
        of the symbol. If sign == 0, Expr is an expression of zero.

    Examples
    --------
    >>> from sympy.abc import a, b, c, d, u, v, w, x, y, z
    >>> from triples.core import InequalityProblem
    >>> pro = InequalityProblem(a, {c/2 - b - 2: x, b - a: y, a - 1: z, d + 2: w}, {})
    >>> pro.get_symbol_signs()
    {a: (1, z + 1), b: (1, y + z + 1), c: (1, 2*x + 2*y + 2*z + 6), d: (None, None)}

    Note that the `expr` stands for its absolute value if the sign is -1,
    so that it always look like an nonnegative expression:
    >>> pro = InequalityProblem(a, {-a - 3: x, -b - a**2: y}, {c: v - 2})
    >>> pro.get_symbol_signs()
    {a: (-1, x + 3), b: (-1, a**2 + y), c: (0, v - 2)}
    """

    eqs0, ineqs0 = problem.eq_constraints, problem.ineq_constraints
    eqs, ineqs = {}, {}

    fs0 = problem.gens

    # polylize and make a copy
    for src, tar in ((eqs0, eqs), (ineqs0, ineqs)):
        for key, val in src.items():
            try:
                if (not isinstance(key, Poly)) or (not key.gens == fs0):
                    key = Poly(key, *fs0)
                tar[key] = val
            except BasePolynomialError:
                # not polynomial -> ignore
                pass

    signs = dict.fromkeys(range(len(fs0)), (None, None))

    # infer signs from constraints
    signs = _get_signs_by_topological_order(ineqs, eqs, signs)

    signs = {fs0[i]: signs.get(i, (None, None)) for i in range(len(fs0))}
    return signs


def recompute_constraints_from_signs(problem: "InequalityProblem") -> "InequalityProblem":
    """
    Recompute the constraints from the symbol signs.

    Examples
    --------
    >>> from sympy.abc import a, b, c, d, u, v, w, x, y, z
    >>> from triples.core import InequalityProblem
    >>> pro = InequalityProblem(a**3 - b + 1, [a**2 - b**2, a, b, a + 2*b])
    >>> pro.recompute_constraints().ineq_constraints
    {a - b: (a**2 - b**2)/(a + b), a: a, b: b}
    """
    signs = problem.get_symbol_signs()
    has_zero = any(s == 0 for s, v in signs.values())

    source = [problem.ineq_constraints, problem.eq_constraints]
    target = [{}, {}]
    for tp, src, dst in zip((1, 2), source, target):
        for p, e in src.items():
            if isinstance(p, Expr):
                if p == 0:
                    continue
                args = Mul.make_args(p.factor())
                args = [(a.base, a.exp) if a.is_Pow and a.exp.is_Rational else (a, 1) for a in args]
            elif isinstance(p, Poly):
                if p.is_zero:
                    continue
                args = p.factor_list_include()
            else:
                raise TypeError(f"Unknown type {type(p)}.")

            sgn = 1
            expr = 1
            rest_args = []
            for a, mul in args:
                # extract the positive part of each arg
                if isinstance(mul, Rational):
                    if _is_double_rational(mul):
                        expr = expr * a**mul
                        continue
                    a = a**mul
                    mul = 1
                elif mul % 2 == 0:
                    expr = expr * a.as_expr()**mul
                    continue
                s1 = sign_sos(a, signs)
                if has_zero or s1 is None:
                    s2 = sign_sos(-a, signs)
                    if s1 is not None:
                        # a == 0
                        sgn = 0
                        break
                    if s2 is not None:
                        sgn *= -1
                        expr = expr * s2**mul
                    else: # elif s2 is None:
                        rest_args.append((a, mul))
                if s1 is not None:
                    expr = expr * s1**mul

            if sgn == 0:
                # 0 >= 0 or 0 == 0 are trivial and should be omitted
                continue

            # p == sgn * expr * Mul(a**mul for a, mul in rest_args)
            e2 = e / expr
            p2 = sgn
            if len(rest_args) == 0:
                if tp == 1:
                    if sgn == 1:
                        # reduces to the trivial constraint 1 >= 0
                        # just omit it
                        continue
                if isinstance(p, Poly):
                    # cast to poly
                    p2 = p.one * sgn
            else:
                for a, mul in rest_args:
                    # TODO: make it square-free
                    p2 = p2 * a**mul

            if isinstance(p, Poly):
                if not isinstance(p2, Poly):
                    # cast to poly
                    p2 = Poly(p2, p.gens)
            elif isinstance(p2, int):
                p2 = Rational(p2)
            dst[p2] = e2

    is_poly = isinstance(problem.expr, Poly)
    for g, (s, v) in signs.items():
        if s is None:
            continue
        if is_poly:
            # cast to poly
            g = Poly(g, problem.expr.gens)
        if s == 0:
            target[1][g] = v
        elif s == 1:
            target[0][g] = v
        elif s == -1:
            target[0][-g] = v

    problem = problem.copy_new(problem.expr, target[0], target[1])
    return problem
