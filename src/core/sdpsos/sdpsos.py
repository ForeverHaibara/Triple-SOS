import numpy as np
import sympy as sp

from .utils import solve_undetermined_linear, split_vector
from .solver import sdp_solver
from .manifold import RootSubspace, LowRankHermitian, _REDUCE_KWARGS
from ...utils.basis_generator import arraylize, arraylize_sp
from ...utils.polytools import deg
from ...utils.roots.roots import Root
from ...utils.roots.findroot import findroot


def _discard_zero_rows(A, b, rank_tolerance = 1e-10):
    A_rowmax = np.abs(A).max(axis = 1)
    nonvanish =  A_rowmax > rank_tolerance * A_rowmax.max()
    rows = np.extract(nonvanish, np.arange(len(nonvanish)))
    return A[rows], b[rows]



def SDPSOS(
        poly,
        minor = False,
        verbose = True
    ):
    degree = deg(poly)

    collection = {'poly': poly}
    for key in ('Q', 'deg', 'M', 'eq', 'Q_numer'):
        collection[key] = {'major': None, 'minor': None, 'multiplier': None}

    manifold = RootSubspace(poly)
    if verbose:
        print(manifold)

    positive = not (degree % 2 == 0 and minor == 0)
    collection['deg']['major'] = degree // 2
    collection['Q']['major'] = manifold.perp_space(minor = 0)

    if minor and degree > 2:
        collection['deg']['minor'] = degree // 2 - 1
        collection['Q']['minor'] = manifold.perp_space(minor = 1)


    for key in collection['Q'].keys():
        Q = collection['Q'][key]
        if Q is not None and Q.shape[1] > 0:
            M = LowRankHermitian(Q)
            collection['M'][key] = M
            collection['Q_numer'][key] = np.array(Q).astype(np.float64)

            eq = M.reduce(collection['deg'][key], **_REDUCE_KWARGS[(degree%2, key)])
            collection['eq'][key] = eq
        else:
            collection['Q'][key] = None

    # perform SDP with numerical operations

    vecM = arraylize_sp(poly, cyc = False)
    eq = sp.Matrix.hstack(*filter(lambda x: x is not None, collection['eq'].values()))
    # return collection['Q']

    # we have eq @ vecS = vecM
    # so that vecS = x0 + space * y where x0 is a particular solution and y is arbitrary
    x0, space = solve_undetermined_linear(eq, vecM)

    splits = split_vector(collection['eq'].values())
    collection.update({'x0': x0, 'space': space, 'splits': splits})

    if verbose:
        print('Degree of freedom: %d'%space.shape[1])

    not_none_keys = [key for key, value in collection['Q'].items() if value is not None]
    sos_result = sdp_solver(x0, space, splits, not_none_keys, reg = 0, verbose = verbose)
    if sos_result is None:
        return collection
    collection.update(sos_result)

    for key, S in collection['S'].items():
        if collection['Q'][key] is None:
            continue
        M = collection['M'][key].construct_from_vector(S)
        collection['S'][key] = M.S
        collection['M'][key] = M.M

    return collection
