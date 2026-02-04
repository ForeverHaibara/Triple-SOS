"""
This file is directly copied from `sympy.polys.matrices.normalforms`
to support SymPy version < 1.12
"""

'''Functions returning normal forms of matrices'''

from sympy.polys.matrices.domainmatrix import DomainMatrix

# TODO (future work):
#  There are faster algorithms for Smith and Hermite normal forms, which
#  we should implement. See e.g. the Kannan-Bachem algorithm:
#  <https://www.researchgate.net/publication/220617516_Polynomial_Algorithms_for_Computing_the_Smith_and_Hermite_Normal_Forms_of_an_Integer_Matrix>


def smith_normal_form(m):
    '''
    Return the Smith Normal Form of a matrix `m` over the ring `domain`.
    This will only work if the ring is a principal ideal domain.
    '''
    invs = invariant_factors(m)
    smf = DomainMatrix.diag(invs, m.domain, m.shape)
    return smf


def is_smith_normal_form(m):
    '''
    Checks that the matrix is in Smith Normal Form
    '''
    domain = m.domain
    shape = m.shape
    zero = domain.zero
    m = m.to_list()

    for i in range(shape[0]):
        for j in range(shape[1]):
            if i == j:
                continue
            if not m[i][j] == zero:
                return False

    upper = min(shape[0], shape[1])
    for i in range(1, upper):
        if m[i-1][i-1] == zero:
            if m[i][i] != zero:
                return False
        else:
            r = domain.div(m[i][i], m[i-1][i-1])[1]
            if r != zero:
                return False

    return True


def add_columns(m, i, j, a, b, c, d):
    # replace m[:, i] by a*m[:, i] + b*m[:, j]
    # and m[:, j] by c*m[:, i] + d*m[:, j]
    for k in range(len(m)):
        e = m[k][i]
        m[k][i] = a*e + b*m[k][j]
        m[k][j] = c*e + d*m[k][j]


def invariant_factors(m):
    '''
    Return the tuple of abelian invariants for a matrix `m`
    (as in the Smith-Normal form)

    References
    ==========

    [1] https://en.wikipedia.org/wiki/Smith_normal_form#Algorithm
    [2] https://web.archive.org/web/20200331143852/https://sierra.nmsu.edu/morandi/notes/SmithNormalForm.pdf

    '''
    domain = m.domain
    shape = m.shape
    m = m.to_list()
    return _smith_normal_decomp(m, domain, shape=shape, full=False)


def smith_normal_decomp(m):
    '''
    Return the Smith-Normal form decomposition of matrix `m`.
    '''
    domain = m.domain
    rows, cols = shape = m.shape
    m = m.to_list()

    invs, s, t = _smith_normal_decomp(m, domain, shape=shape, full=True)
    smf = DomainMatrix.diag(invs, domain, shape).to_dense()

    s = DomainMatrix(s, domain=domain, shape=(rows, rows))
    t = DomainMatrix(t, domain=domain, shape=(cols, cols))
    return smf, s, t


def _smith_normal_decomp(m, domain, shape, full):
    '''
    Return the tuple of abelian invariants for a matrix `m`
    (as in the Smith-Normal form). If `full=True` then invertible matrices
    ``s, t`` such that the product ``s, m, t`` is the Smith Normal Form
    are also returned.
    '''
    if not domain.is_PID:
        msg = f"The matrix entries must be over a principal ideal domain, but got {domain}"
        raise ValueError(msg)

    rows, cols = shape
    zero = domain.zero
    one = domain.one

    def eye(n):
        return [[one if i == j else zero for i in range(n)] for j in range(n)]

    if 0 in shape:
        if full:
            return (), eye(rows), eye(cols)
        else:
            return ()

    if full:
        s = eye(rows)
        t = eye(cols)

    def add_rows(m, i, j, a, b, c, d):
        # replace m[i, :] by a*m[i, :] + b*m[j, :]
        # and m[j, :] by c*m[i, :] + d*m[j, :]
        for k in range(len(m[0])):
            e = m[i][k]
            m[i][k] = a*e + b*m[j][k]
            m[j][k] = c*e + d*m[j][k]

    def clear_column():
        # make m[1:, 0] zero by row and column operations
        pivot = m[0][0]
        for j in range(1, rows):
            if m[j][0] == zero:
                continue
            d, r = domain.div(m[j][0], pivot)
            if r == zero:
                add_rows(m, 0, j, 1, 0, -d, 1)
                if full:
                    add_rows(s, 0, j, 1, 0, -d, 1)
            else:
                a, b, g = domain.gcdex(pivot, m[j][0])
                d_0 = domain.exquo(m[j][0], g)
                d_j = domain.exquo(pivot, g)
                add_rows(m, 0, j, a, b, d_0, -d_j)
                if full:
                    add_rows(s, 0, j, a, b, d_0, -d_j)
                pivot = g

    def clear_row():
        # make m[0, 1:] zero by row and column operations
        pivot = m[0][0]
        for j in range(1, cols):
            if m[0][j] == zero:
                continue
            d, r = domain.div(m[0][j], pivot)
            if r == zero:
                add_columns(m, 0, j, 1, 0, -d, 1)
                if full:
                    add_columns(t, 0, j, 1, 0, -d, 1)
            else:
                a, b, g = domain.gcdex(pivot, m[0][j])
                d_0 = domain.exquo(m[0][j], g)
                d_j = domain.exquo(pivot, g)
                add_columns(m, 0, j, a, b, d_0, -d_j)
                if full:
                    add_columns(t, 0, j, a, b, d_0, -d_j)
                pivot = g

    # permute the rows and columns until m[0,0] is non-zero if possible
    ind = [i for i in range(rows) if m[i][0] != zero]
    if ind and ind[0] != zero:
        m[0], m[ind[0]] = m[ind[0]], m[0]
        if full:
            s[0], s[ind[0]] = s[ind[0]], s[0]
    else:
        ind = [j for j in range(cols) if m[0][j] != zero]
        if ind and ind[0] != zero:
            for row in m:
                row[0], row[ind[0]] = row[ind[0]], row[0]
            if full:
                for row in t:
                    row[0], row[ind[0]] = row[ind[0]], row[0]

    # make the first row and column except m[0,0] zero
    while (any(m[0][i] != zero for i in range(1,cols)) or
           any(m[i][0] != zero for i in range(1,rows))):
        clear_column()
        clear_row()

    def to_domain_matrix(m):
        return DomainMatrix(m, shape=(len(m), len(m[0])), domain=domain)

    if m[0][0] != 0:
        c = domain.canonical_unit(m[0][0])
        if domain.is_Field:
            c = 1 / m[0][0]
        if c != domain.one:
            m[0][0] *= c
            if full:
                s[0] = [elem * c for elem in s[0]]

    if 1 in shape:
        invs = ()
    else:
        lower_right = [r[1:] for r in m[1:]]
        ret = _smith_normal_decomp(lower_right, domain,
                shape=(rows - 1, cols - 1), full=full)
        if full:
            invs, s_small, t_small = ret
            s2 = [[1] + [0]*(rows-1)] + [[0] + row for row in s_small]
            t2 = [[1] + [0]*(cols-1)] + [[0] + row for row in t_small]
            s, s2, t, t2 = list(map(to_domain_matrix, [s, s2, t, t2]))
            s = s2 * s
            t = t * t2
            s = s.to_list()
            t = t.to_list()
        else:
            invs = ret

    if m[0][0]:
        result = [m[0][0]]
        result.extend(invs)
        # in case m[0] doesn't divide the invariants of the rest of the matrix
        for i in range(len(result)-1):
            a, b = result[i], result[i+1]
            if b and domain.div(b, a)[1] != zero:
                if full:
                    x, y, d = domain.gcdex(a, b)
                else:
                    d = domain.gcd(a, b)

                alpha = domain.div(a, d)[0]
                if full:
                    beta = domain.div(b, d)[0]
                    add_rows(s, i, i + 1, 1, 0, x, 1)
                    add_columns(t, i, i + 1, 1, y, 0, 1)
                    add_rows(s, i, i + 1, 1, -alpha, 0, 1)
                    add_columns(t, i, i + 1, 1, 0, -beta, 1)
                    add_rows(s, i, i + 1, 0, 1, -1, 0)

                result[i+1] = b * alpha
                result[i] = d
            else:
                break
    else:
        if full:
            if rows > 1:
                s = s[1:] + [s[0]]
            if cols > 1:
                t = [row[1:] + [row[0]] for row in t]
        result = invs + (m[0][0],)

    if full:
        return tuple(result), s, t
    else:
        return tuple(result)
