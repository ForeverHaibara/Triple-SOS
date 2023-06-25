import numpy as np
import sympy as sp

from .findroot import findroot_resultant
from ...utils.basis_generator import generate_expr
from ...utils.polytools import convex_hull_poly, deg
from ...utils.roots.tangents import uv_from_root
from ...utils.roots.roots import Root, RootRational
from ...utils.roots.rootsinfo import RootsInfo

_ROOT_FILTERS = {
    (0,0,0): lambda r: True,
    (1,0,0): lambda r: r[0] != 0,
    (1,1,0): lambda r: r[0] != 0 and r[1] != 0,
    (1,1,1): lambda r: r[0] != 0 and r[1] != 0 and r[2] != 0,
}

_REDUCE_KWARGS = {
    (0, 'major'): {'monom_add': (0,0,0), 'cyc': False},
    (0, 'minor'): {'monom_add': (1,1,0), 'cyc': True},
    (1, 'major'): {'monom_add': (1,0,0), 'cyc': True},
    (1, 'minor'): {'monom_add': (1,1,1), 'cyc': False},
}

def _all_roots_space(rootsinfo, n, monom_add = (0,0,0), root_filter = None, convex_hull = None):
    """
    Drepecated.
    """
    # if not hasattr(root_filter, '__call__'):
    if root_filter is None:
        root_filter = _ROOT_FILTERS[monom_add]

    handled = {}
    spaces = []
    for root in filter(root_filter, rootsinfo.strict_roots):
        if not isinstance(root, Root):
            root = Root(root)
        uv = root.uv()
        is_handled = handled.get(uv, False)
        if is_handled:
            continue
        
        handled[uv] = True
        spaces.append(root.span(n))

    spaces += _hull_space(n, convex_hull, monom_add = monom_add)

    return sp.Matrix.hstack(*spaces)


def _perp_space(rootsinfo, n, monom_add = (0,0,0), convex_hull = None):
    """
    Deprecated.
    """
    if isinstance(rootsinfo, RootsInfo):
        space = _all_roots_space(rootsinfo, n, monom_add = monom_add, convex_hull = convex_hull).T
    else:
        space = rootsinfo.T

    # HAS BUG:
    # L, U, P = sp.Matrix.LUdecomposition(space)
    
    # for i in range(min(U.shape) - 1, -1, -1):
    #     if U[i,i] != 0:
    #         break
    # else: i = -1
    # rank = i + 1
    # U = U[:rank, :]

    # U0, U1 = U[:, :U.shape[0]], U[:, U.shape[0]:]

    # # [U0, U1] @ Q = 0
    # Q = sp.Matrix.vstack(-U0.LUsolve(U1), sp.eye(U1.shape[1]))

    n = space.shape[1]
    Q = sp.Matrix(space.nullspace())
    Q = Q.reshape(Q.shape[0] // n, n).T

    # normalize to keep numerical stability
    reg = np.max(np.abs(Q), axis = 0)
    reg = 1 / np.tile(reg, (Q.shape[0], 1))
    reg = sp.Matrix(reg)
    Q = Q.multiply_elementwise(reg)

    return Q


def _hull_space(n, convex_hull = None, monom_add = (0,0,0)):
    """
    For example, s(ab(a-b)2(a+b-3c)2) does not have a^6,
    so in the positive semidefinite representation, the entry (a^3,a^3) of M is zero.
    This requires that Me_i = 0 where i is the index of a^3.
    """
    if convex_hull is None:
        return []
    inv_monoms = generate_expr(n, cyc = False)[0]

    def onehot(i):
        v = sp.Matrix.zeros(len(inv_monoms), 1)
        v[i] = 1
        return v

    space = []
    for key, value in convex_hull.items():
        if value:
            continue
        # value == False: not in convex hull

        rest_monom = (key[0] - monom_add[0], key[1] - monom_add[1], key[2] - monom_add[2])
        if rest_monom[0] % 2 or rest_monom[1] % 2 or rest_monom[2] % 2:
            continue
        rest_monom = (rest_monom[0] // 2, rest_monom[1] // 2, rest_monom[2] // 2)

        i = inv_monoms[rest_monom]
        space.append(onehot(i))

    return space


class RootSubspace():
    def __init__(self, poly):
        self.poly = poly
        self.n = deg(poly)
        self.convex_hull = convex_hull_poly(poly)[0]
        self.roots = findroot_resultant(poly)
        self.roots = [r for r in self.roots if not r.is_corner]
        self.subspaces_ = {}

    def subspaces(self, n, positive = False):
        if not positive:
            if n not in self.subspaces_:
                subspaces = []
                for root in self.roots:
                    if not root.is_corner:
                        subspaces.append(root.span(n))
                self.subspaces_[n] = subspaces
            return self.subspaces_[n]

        # if positive == True, we filter out the negative roots
        subspaces = self.subspaces(n, positive = False)
        subspaces_positive = []
        for r, subspace in zip(self.roots, subspaces):
            if r.root[0] >= 0 and r.root[1] >= 0 and r.root[2] >= 0:
                subspaces_positive.append(subspace)
        return subspaces_positive

    @property
    def positive_roots(self):
        return [r for r in self.roots if r.root[0] >= 0 and r.root[1] >= 0 and r.root[2] >= 0]

    def perp_space(self, minor = 0, positive = True):

        # if we only perform SOS over positive numbers,
        # we filter out the negative roots
        subspaces = self.subspaces(self.n // 2 - minor, positive = positive)

        monom_add = _REDUCE_KWARGS[(self.n % 2, 'minor' if minor else 'major')]['monom_add']
        if sum(monom_add):
            subspaces_filtered = []
            # filter roots on border
            monoms = generate_expr(self.n // 2 - minor, cyc = False)[0]
            for root, subspace in zip(self.positive_roots, subspaces):
                if not root.is_border:
                    subspaces_filtered.append(subspace)
                    continue
                if sum(monom_add) == 3:
                    # filter out all columns
                    continue
                reserved_cols = []
                for i in range(subspace.shape[1]):
                    # check each column
                    nonzero_monom = [0, 0, 0]
                    for monom, j in zip(monoms, range(subspace.shape[0])):
                        if subspace[j, i] != 0:
                            for k in range(3):
                                if monom[k] != 0:
                                    nonzero_monom[k] = 1
                    # if nonzero_monom[k] == 0
                    # it means that (a,b,c)[k] == 0 in this root
                    for k in range(3):
                        if monom_add[k] == 1 and nonzero_monom[k] == 0:
                            # filter out this column
                            break
                    else:
                        reserved_cols.append(i)
                subspaces_filtered.append(subspace[:, reserved_cols])
        
            subspaces = subspaces_filtered
                    

        subspaces += self.hull_space(minor)
        space = sp.Matrix.hstack(*subspaces).T

        n = space.shape[1]
        Q = sp.Matrix(space.nullspace())
        Q = Q.reshape(Q.shape[0] // n, n).T

        # normalize to keep numerical stability
        reg = np.max(np.abs(Q), axis = 0)
        reg = 1 / np.tile(reg, (Q.shape[0], 1))
        reg = sp.Matrix(reg)
        Q = Q.multiply_elementwise(reg)

        return Q

    def hull_space(self, minor = 0):
        """
        For example, if s(a4b2+4a4c2+4a3b3-6a3b2c-12a3bc2+9a2b2c2) can be written as
        sum of squares, then there would not be the term (a^3 + ...)^2.

        This is because a^6 is out of the convex hull of the polynomial. We should 
        constraint the coefficients of a^3 to be zero.
        """
        monom_add = _REDUCE_KWARGS[(self.n % 2, 'minor' if minor else 'major')]['monom_add']
        return _hull_space(self.n // 2 - minor, self.convex_hull, monom_add = monom_add)

    def cyclic_space(self, minor = 0):
        """
        If monom_add['cyc'] is False, we can impose additional cyclic constraints.
        """
        monom_add = _REDUCE_KWARGS[(self.n % 2, 'minor' if minor else 'major')]['monom_add']
        if monom_add['cyc']:
            return []

        monoms = generate_expr(self.n // 2 - minor, cyc = False)[0]
        return []

    def __str__(self):
        return 'Subspace [\n    roots = %s\n    uv    = %s\n]'%(
            self.roots, [_.uv() for _ in self.roots]
        )

class LowRankHermitian():
    """
    A Hermitian matrix M such that M is in rowspace of Q.
    That is, M = Q @ S @ Q.T for some low rank S.
    Note that by inertia theorem, if M is positive definite, then so is S.

    Knowledge of kronecker product tells us that vec(M) = kron(Q,Q) @ vec(S).
    """
    def __init__(self, Q: sp.Matrix = None, S = None):
        self.Q = Q
        self.QQ = None

        if Q is not None:
            if isinstance(Q, sp.Matrix):
                self.QQ = sp.kronecker_product(Q, Q)
            elif isinstance(Q, np.ndarray):
                self.QQ = np.kron(Q, Q)

        self.S = None
        self.construct_from_vector(S)

    @property
    def M(self):
        if self.S is None or self.Q is None:
            return None
        if isinstance(self.Q, sp.Matrix):
            return self.Q * self.S * self.Q.T
        elif isinstance(self.Q, np.ndarray):
            return self.Q @ self.S @ self.Q.T

    def construct_from_vector(self, S, full = False):
        """
        Construct S from vec(S).
        """
        k = self.Q.shape[1] if hasattr(self.Q, 'shape') else 0
        if S is not None:
            if len(S.shape) == 2:
                if S.shape[0] == S.shape[1]:
                    self.S = S
                    return self
                size = S.shape[0] * S.shape[1]
                S = S.reshape(size, 1)
                if isinstance(S, np.ndarray):
                    S = S.flatten()
                
            # infer k automatically
            if full:
                k = round(S.shape[0] ** 0.5)
            else:
                k = round((2 * S.shape[0] + .25) ** 0.5 - .5)

            S_ = sp.zeros(k, k)
            pointer = 0
            if size != k ** 2:
                # S is only the upper triangular part
                for i in range(k):
                    for j in range(i, k):
                        S_[i,j] = S_[j,i] = S[pointer]
                        pointer += 1
            else:
                for i in range(k):
                    for j in range(k):
                        S_[i,j] = S[pointer]
                        pointer += 1
            S = S_

        self.S = S
        return self

    def reduce(self, n, monom_add = (0,0,0), cyc = False):
        """
        For example, Vasile inequality 2s(a2)2 - 6s(a3b) has a
        (nonpositive) 6*6 matrix representation v' * M * v where
        v = [a^2,b^2,c^2,ab,bc,ca]' and M = 
        [[ 2, 1, 1,-3, 0, 0],
         [ 1, 2, 1, 0,-3, 0],
         [ 1, 1, 2, 0, 0,-3],
         [-3, 0, 0, 2, 0, 0],
         [ 0,-3, 0, 0, 2, 0],
         [ 0, 0,-3, 0, 0, 2]]
        The a^2b^2 coefficient is (1 + 1 + 2) = 4. We note that each coefficient
        may be the sum of several entries. We should reduce the entry number to 
        coefficient number. As a result, the shape of kron(Q,Q) gets reduced.
        """
        monoms = generate_expr(n, cyc = False)[1]
        m, k = self.Q.shape # m = len(monoms)

        m0, n0, p0 = monom_add
        inv_monoms = generate_expr(2*n + sum(monom_add), cyc = False)[0]
        QQ_reduced = sp.zeros(len(inv_monoms), k ** 2)

        if cyc:
            permute = lambda a,b,c: [(a,b,c), (b,c,a), (c,a,b)]
        else:
            permute = lambda a,b,c: [(a,b,c)]

        for i in range(m):
            m1, n1, p1 = monoms[i]
            m1, n1, p1 = m1 + m0, n1 + n0, p1 + p0
            for j in range(m):
                m2, n2, p2 = monoms[j]
                new_monom_ = (m1 + m2, n1 + n2, p1 + p2)
                for new_monom in permute(*new_monom_):
                    index = inv_monoms[new_monom]
                    QQ_reduced[index, :] += self.QQ[i * m + j, :]

        # also cancel diagonal entries
        QQ_reduced2 = []
        for i in range(k):
            QQ_reduced2.append(QQ_reduced[:, i * k + i])
            for j in range(i + 1, k):
                QQ_reduced2.append(QQ_reduced[:, i * k + j] + QQ_reduced[:, j * k + i])
        QQ_reduced = sp.Matrix.hstack(*QQ_reduced2)

        return QQ_reduced