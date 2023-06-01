import numpy as np
import sympy as sp

from .utils import solve_undetermined_linear, split_vector
from .solver import sdp_solver
from .manifold import _perp_space, LowRankHermitian
from ...utils.basis_generator import arraylize, arraylize_sp
from ...utils.polytools import deg
from ...utils.roots.roots import Root
from ...utils.roots.findroot import findroot


_REDUCE_KWARGS = {
    (0, 'major'): {'monom_add': (0,0,0), 'cyc': False},
    (0, 'minor'): {'monom_add': (1,1,0), 'cyc': True},
    (1, 'major'): {'monom_add': (1,0,0), 'cyc': True},
    (1, 'minor'): {'monom_add': (1,1,1), 'cyc': False},
}


def _discard_zero_rows(A, b, rank_tolerance = 1e-10):
    A_rowmax = np.abs(A).max(axis = 1)
    nonvanish =  A_rowmax > rank_tolerance * A_rowmax.max()
    rows = np.extract(nonvanish, np.arange(len(nonvanish)))
    return A[rows], b[rows]



def SDPSOS(
        poly,
        rootsinfo = None,
        minor = False,
    ):
    degree = deg(poly)
    if rootsinfo is None:
        rootsinfo = findroot(poly, with_tangents = False)

    collection = {}
    for key in ('Q', 'deg', 'M', 'eq', 'Q_numer'):
        collection[key] = {'major': None, 'minor': None, 'multiplier': None}
    
    collection['deg']['major'] = degree // 2
    collection['Q']['major'] = _perp_space(rootsinfo, degree // 2, 
                                           root_filter = _REDUCE_KWARGS[(degree % 2, 'major')]['monom_add'])

    if minor and degree > 2:
        collection['deg']['minor'] = degree // 2 - 1
        collection['Q']['minor'] = _perp_space(rootsinfo, degree // 2 - 1,
                                           root_filter = _REDUCE_KWARGS[(degree % 2, 'minor')]['monom_add'])

    for key in collection['Q'].keys():
        Q = collection['Q'][key]
        if Q is not None:
            M = LowRankHermitian(Q)
            collection['M'][key] = M
            collection['Q_numer'][key] = np.array(Q).astype(np.float64)

            eq = M.reduce(collection['deg'][key], **_REDUCE_KWARGS[(degree%2, key)])
            collection['eq'][key] = eq
    print([Root(_).uv for _ in rootsinfo.strict_roots])

    # perform SDP with numerical operations

    vecM = arraylize_sp(poly, cyc = False)
    eq = sp.Matrix.hstack(*filter(lambda x: x is not None, collection['eq'].values()))

    # we have eq @ vecS = vecM
    # so that vecS = x0 + space * y where x0 is a particular solution and y is arbitrary
    x0, space = solve_undetermined_linear(eq, vecM)
    # return x0, space

    splits = split_vector(collection['eq'].values())
    print('Degree of freedom: %d'%space.shape[1])
    print([(key, value.shape if value is not None else 'None') for key, value in collection['eq'].items()])

    not_none_keys = [key for key, value in collection['Q'].items() if value is not None]
    sos_result = sdp_solver(x0, space, splits, not_none_keys, verbose = 1)
    if sos_result is None:
        return None
    collection.update(sos_result)

    for key, S in collection['S'].items():
        if collection['Q'][key] is None:
            continue
        M = collection['M'][key].construct_from_vector(S)
        collection['S'][key] = M.S
        collection['M'][key] = M.M

    collection.update({'x0': x0, 'space': space})
    return collection
