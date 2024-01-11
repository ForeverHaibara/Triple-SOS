from functools import lru_cache

import numpy as np
import sympy as sp

from ...utils import (
    deg, generate_expr, verify_is_symmetric,
    convex_hull_poly,
    findroot_resultant
)


_REDUCE_KWARGS = {
    (0, 'major'): {'monom_add': (0,0,0), 'cyc': False},
    (0, 'minor'): {'monom_add': (1,1,0), 'cyc': True},
    (1, 'major'): {'monom_add': (1,0,0), 'cyc': True},
    (1, 'minor'): {'monom_add': (1,1,1), 'cyc': False},
}


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


def _compute_multiplicity_sym(poly):
    """
    Compute the multiplicity of zero (1,1,c) on the symmetric axis when c -> 0.
    This is equivalent to the number of zeros in the leading row sums of the
    coefficient triangle.

    Returns zero when the polynomial is a multiple of p(a-b)2.
    """
    a, b = sp.symbols('a b')
    poly = poly.subs({a:1, b:1})
    return min(poly.monoms())[0]


def _compute_multiplicity_hessian(poly):
    """
    Compute the multiplicity of zero at (1,1,1) by Hessian.
    For now, when poly(1,1,1) != 0, return 0. Otherwise,
    return maximum n that any partial derivatives of order 2(n-1) is zero.
    Similar definition of multivariate multiplicity can be referred to Slusky's theorem.
    """
    a, b, c = sp.symbols('a b c')
    part_poly = poly.subs({c:1})
    if part_poly.subs({a:1,b:1}) != 0:
        return 0
    part_poly = part_poly.as_poly(a).shift(1)
    part_poly = part_poly.as_poly(b).shift(1).as_poly(a,b)
    monoms = map(lambda x: x[0] + x[1], part_poly.monoms())
    n = min(monoms) - 1

    # now that part_poly is centered at (0,0)
    return n // 2 + 1


class RootSubspace():
    """
    If an inequality has nontrivial equality cases, known as roots, then
    the sum-of-squares representation should be zero at these roots. This implies
    the result lies on a subspace perpendicular to the roots. This class
    investigates the roots and generates the subspace.
    """
    def __init__(self, poly):
        self.poly = poly
        self.n = deg(poly)
        self.convex_hull = convex_hull_poly(poly)[0]
        self.roots = findroot_resultant(poly)
        self.roots = [r for r in self.roots if not r.is_corner]
        self.subspaces_ = {}
        self.multiplicity_sym_ = _compute_multiplicity_sym(poly)
        self.multiplicity_hes_ = _compute_multiplicity_hessian(poly)

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

    def reduce_kwargs(self, minor = 0):
        return _REDUCE_KWARGS[(self.n % 2, 'minor' if minor else 'major')]

    def monom_add(self, minor = 0):
        return self.reduce_kwargs(minor)['monom_add']

    def perp_space(self, minor = 0, positive = True):
        """
        The most important idea of sum of squares is to find a subspace
        constrained by the roots of the polynomial. For example, if (1,2,3) is a root of 
        a quartic cyclic polynomial. And the quartic can be written as x'Mx where 
        x = [a^2, b^2, c^2, ab, bc, ca]. Then we must have Mx = 0 when (a,b,c) = (1,2,3)
        since M >= 0 is a positive semidefinite matrix. This means that the M lies in the
        orthogonal complement of the subspace spanned by (1,2,3).

        This function returns the orthogonal complement of the subspace spanned by the roots.
        """

        # if we only perform SOS over positive numbers,
        # we filter out the negative roots
        subspaces = self.subspaces(self.n // 2 - minor, positive = positive)

        monom_add = self.monom_add(minor)
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
        subspaces += self.mult_sym_space(minor)
        subspaces += self.mult_hes_space(minor)
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
        monom_add = self.monom_add(minor)
        return _hull_space(self.n // 2 - minor, self.convex_hull, monom_add = monom_add)

    def mult_sym_space(self, minor = 0):
        """
        If the first n rows of a coefficient triangle sum to zero, respectively, then it means that
        the order of zero at (1,1,c) when c -> 0 is n. This implies a multiplicity constraint. For example,
        if we require f(c) = x'Mx = 0 at (1,1,c) when c -> 0 with order n. If n = 3, we have that
        the first derivative (order = 2): (dx/dc)'Mx = 0. This implies nothing because Mx = 0 already.
        However, the second derivative (order = 3): (dx/dc)'M(dx/dc) + ... = 0 => M(dx/dc) = 0.
        This yields extra constraints.
        """
        if self.multiplicity_sym_ <= 1:
            # either no root at (1,1,0) or multiplicity is merely 1
            return []
        monom_add = self.monom_add(minor)
        monoms = generate_expr(self.n // 2 - minor, cyc = False)[1]

        vecs_all = []

        for i in range(3): # stands for a, b, c
            # compute the order, note that monom_add might cancel one order
            dm = monom_add[i]
            num = (self.multiplicity_sym_ - 1 + dm) // 2
            # num = self.multiplicity_sym_ - 1 - dm

            vecs = [[0] * len(monoms) for _ in range(num)]
            for j in range(len(monoms)):
                monom = monoms[j]
                if -dm < monom[i] <= num - dm:
                    # sum of coefficients where monom[i] == constant
                    # actually it should be the factorial after derivative, (monom[i]!), 
                    # however it is equivalent to ones
                    vecs[monom[i] - 1 + dm][j] = 1
            vecs = [sp.Matrix(_) for _ in vecs]
        
            vecs_all.extend(vecs)
        
        return vecs_all

    def mult_hes_space(self, minor = 0):
        """
        If the Hessian of the polynomial at (1,1,1) is zero, then we have extra constraints.
        M * (dv/da) == 0, M * (dv/db) == 0, M * (dv/dc) == 0 at (1,1,1).
        """
        if self.multiplicity_hes_ <= 1:
            # either no root at (1,1,1) or multiplicity is merely 1
            return []
        monoms = generate_expr(self.n // 2 - minor, cyc = False)[1]

        # @lru_cache()
        def derv(n, i):
            # compute n*(n-1)*..*(n-i+1)
            if i == 1: return n
            if n < i:
                return 0
            return sp.prod((n - j) for j in range(i))

        vecs_all = []
        for i in range(3): # stands for a, b, c
            vecs = [[0] * len(monoms) for _ in range(self.multiplicity_hes_ - 1)]
            for d in range(1, self.multiplicity_hes_):
                vec = vecs[d - 1]
                for j in range(len(monoms)):
                    vec[j] = derv(monoms[j][i], d)
            vecs = [sp.Matrix(_) for _ in vecs]
        
            vecs_all.extend(vecs)

        return vecs_all


    def __str__(self):
        # roots = [abs(root[2]) >= abs(root[1]) and abs(root[2]) >= abs(root[0]) for a,b,c in roots]
        def _formatter(root):
            uv = root.uv()
            if hasattr(root, 'root_anp') and isinstance(root.root_anp[0], sp.polys.polyclasses.ANP):
                if len(root.root_anp[0].mod) > 4:
                    uv = (uv[0].n(15), uv[1].n(15))
            return uv
        return 'Subspace [\n    roots  = %s\n    uv     = %s\n    rowsum = %s\n    hessian = %s\n]'%(
            self.roots, [_formatter(_) for _ in self.roots], self.multiplicity_sym_, self.multiplicity_hes_
        )


def coefficient_matrix(Q, n, monom_add = (0,0,0), cyc = False):
    """
    For example, the Vasile inequality 2s(a2)2 - 6s(a3b) has a 
    (nonnegative) 6*6 matrix representation v' * M * v where
    v = [a^2,b^2,c^2,ab,bc,ca]' and M = 
    [[ 2, 1, 1,-3, 0, 0],
    [ 1, 2, 1, 0,-3, 0],
    [ 1, 1, 2, 0, 0,-3],
    [-3, 0, 0, 2, 0, 0],
    [ 0,-3, 0, 0, 2, 0],
    [ 0, 0,-3, 0, 0, 2]]
    The a^2b^2 coefficient is (1 + 1 + 2) = 4. We note that each coefficient
    may be the sum of several entries. We learn that the final equation is A @ vec(M) = p
    where p is the vector of coefficients of the original polynomial.

    Hence, we have A @ kron(Q,Q) @ vec(S) = p.
    And we reduce kron(Q,Q) to such A @ kron(Q,Q) = QQ_reduced.

    Futhermore, S is symmetric, so we only need the upper triangular part to
    form the vec(S). So we can further reduce the size of matrix, so that
    QQ_reduced @ vec(S)_reduced = p.
    """
    monoms = generate_expr(n, cyc = False)[1]
    m, k = Q.shape # m = len(monoms)
    QQ = sp.kronecker_product(Q, Q)

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
                QQ_reduced[index, :] += QQ[i * m + j, :]

    # also cancel symmetric entries
    QQ_reduced2 = []
    for i in range(k):
        QQ_reduced2.append(QQ_reduced[:, i * k + i])
        for j in range(i + 1, k):
            QQ_reduced2.append(QQ_reduced[:, i * k + j] + QQ_reduced[:, j * k + i])
    QQ_reduced = sp.Matrix.hstack(*QQ_reduced2)

    return QQ_reduced


def add_cyclic_constraints(sdp_problem):
    """
    If monom_add['cyc'] is False, we can impose additional cyclic constraints.

    Parameters
    ----------
    sdp_problem : SDPProblem
        SDP problem.

    Returns
    ----------
    sdp_problem : SDPProblem
        SDP problem. The function modifies the input sdp_problem inplace.
    """
    degree = sdp_problem.poly_degree

    transforms = [
        lambda x,y,z: (y,z,x),
        lambda x,y,z: (x,z,y)
    ]
    if not verify_is_symmetric(sdp_problem.poly):
        transforms = transforms[:1]

    for key in ('major', 'minor'):
        if sdp_problem.Q[key] is None:
            continue
        reduced_kwargs = _REDUCE_KWARGS[(degree % 2, key)]
        if reduced_kwargs['cyc']:
            continue

        # cyc == False
        rows = []

        inv_monoms, monoms = generate_expr(degree // 2 - (key == 'minor'), cyc = False)
        m = len(monoms)
        Q = sdp_problem.Q[key]
        QQ = sp.kronecker_product(Q, Q)
        for i1, m1 in enumerate(monoms):
            for j1, m2 in enumerate(monoms[i1:], start = i1):
                for transform in transforms:
                    i2 = inv_monoms[transform(*m1)]
                    if i2 <= i1:
                        continue
                    j2 = inv_monoms[transform(*m2)]

                    # equivalent entries must be equal
                    row = QQ[i1 * m + j1, :] - QQ[i2 * m + j2, :]
                    rows.append(row)

        if len(rows):
            rows = sp.Matrix.vstack(*rows)

            # cancel symmetric entries
            k = Q.shape[1]
            QQ_reduced, QQ_reduced2 = rows, []
            for i in range(k):
                QQ_reduced2.append(QQ_reduced[:, i * k + i])
                for j in range(i + 1, k):
                    QQ_reduced2.append(QQ_reduced[:, i * k + j] + QQ_reduced[:, j * k + i])
            rows = sp.Matrix.hstack(*QQ_reduced2)

            eq = sdp_problem.eq
            eq[key] = sp.Matrix.vstack(eq[key], rows)

            # align other components with zero matrices
            for other_key in eq.keys():
                if other_key == key or eq[other_key] is None:
                    continue
                eq[other_key] = sp.Matrix.vstack(
                    eq[other_key], sp.zeros(rows.shape[0], eq[other_key].shape[1])
                )
            
            sdp_problem.vecP = sp.Matrix.vstack(sdp_problem.vecP, sp.zeros(rows.shape[0], 1))
    
    return sdp_problem