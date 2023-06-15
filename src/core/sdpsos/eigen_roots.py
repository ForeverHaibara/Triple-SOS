"""
Deprecated
"""
import numpy as np
import sympy as sp

from ...utils.basis_generator import generate_expr
from ...utils.roots.findroot import findroot

def _as_vec(n, root):
    basis = generate_expr(n, cyc = 0)[1]
    vec = np.zeros(len(basis))
    a, b, c = root
    for i, (n1, n2, n3) in enumerate(basis):
        vec[i] = a**n1 * b**n2 * c**n3
    return vec

def _filter_positive_roots(roots, positive = True):
    if positive:
        return list(filter(lambda x: x[0] >= 0 and x[1] >= 0 and x[2] >= 0, roots))
    return roots

def _get_standard_roots(
        roots = None, 
        positive = True, 
        vanish_tolerance = 1e-4,
        normalize = True
    ):
    """
    Get standard roots from rootsinfo.
    """
    def _vanish(x):
        if hasattr(x, '__len__'):
            return list(map(_vanish, x))
        if abs(x) < vanish_tolerance:
            return 0
        if abs(x - 1) < vanish_tolerance:
            return 1
        return x

    def _align_length(x):
        if len(x) == 2:
            return x[0], x[1], 1
        return x

    def _max_ahead(x):
        if x[1] > x[0] and x[1] > x[2]:
            return x[1], x[2], x[0]
        if x[2] > x[0] and x[2] > x[1]:
            return x[2], x[0], x[1]
        return x

    def _expand_cyclic(roots):
        roots_ = []
        for x in roots:
            if x[0] == x[1] and x[1] == x[2]:
                roots_.append(x)
                continue
            roots_.append(x)
            roots_.append((x[1], x[2], x[0]))
            roots_.append((x[2], x[0], x[1]))
        return roots_

    if isinstance(roots, sp.polys.Poly):
        roots = findroot(roots, most = 10)
    if isinstance(roots, dict) and roots.get('strict_roots') is not None:
        roots = roots['strict_roots']

    roots = [_vanish(x) for x in roots]
    roots = [_align_length(x) for x in roots]
    roots = _filter_positive_roots(roots, positive)
    roots = [tuple(x) for x in roots]
    roots = [_max_ahead(x) for x in roots]
    roots = set(roots)
    roots = _expand_cyclic(roots)

    if normalize:
        roots = np.array(roots).astype('float64')
        roots /= np.linalg.norm(roots, axis = 1, keepdims = True)

    return roots