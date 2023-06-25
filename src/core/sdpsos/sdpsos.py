from typing import List, Optional, Union, Tuple, Dict

import sympy as sp

from .utils import solve_undetermined_linear, split_vector, indented_print
from .solver import sdp_solver
from .manifold import RootSubspace, LowRankHermitian, _REDUCE_KWARGS
from .solution import create_solution_from_M, SolutionSDP
from ...utils.basis_generator import arraylize, arraylize_sp
from ...utils.polytools import deg


def _sdp_sos(
        poly: sp.Poly,
        manifold: RootSubspace = None,
        minor: bool = False,
        verbose: bool = True
    ) -> Optional[Dict]:
    """
    Solve a polynomial SOS problem with SDP.

    Parameters
    ----------
    poly : sp.Poly
        Polynomial to be solved.
    manifold : RootSubspace
        Manifold should be a RootSubspace object, RootSubspace(poly) will be used if None.
        When the polynomial has nontrivial roots, then the solution positive semidefinite
        matrix should be zero at these roots. This implies linear constraint of the matrix
        at these roots. The manifold finds the roots of the polynomial and generate the
        linear constraints.
    minor : bool
        For a problem of even degree, if it holds for all real numbers, it might be in the 
        form of sum of squares. However, if it only holds for positive real numbers, then
        it might be in the form of \sum (...)^2 + \sum ab(...)^2. Note that we have an
        additional term in the latter, which is called the minor term. If we need to 
        add the minor term, please set minor = True.

    verbose : bool
        If True, print the information of the problem.
    """
    degree = deg(poly)

    collection = {'poly': poly}
    for key in ('Q', 'deg', 'M', 'eq'):
        collection[key] = {'major': None, 'minor': None, 'multiplier': None}

    if manifold is None:
        manifold = RootSubspace(poly)
        if verbose:
            print(manifold)

    positive = not (degree % 2 == 0 and minor == 0)
    collection['deg']['major'] = degree // 2
    collection['Q']['major'] = manifold.perp_space(minor = 0, positive = positive)

    if minor and degree > 2:
        collection['deg']['minor'] = degree // 2 - 1
        collection['Q']['minor'] = manifold.perp_space(minor = 1, positive = positive)


    for key in collection['Q'].keys():
        Q = collection['Q'][key]
        if Q is not None and Q.shape[1] > 0:
            M = LowRankHermitian(Q)
            collection['M'][key] = M

            eq = M.reduce(collection['deg'][key], **_REDUCE_KWARGS[(degree%2, key)])
            collection['eq'][key] = eq
        else:
            collection['Q'][key] = None


    vecM = arraylize_sp(poly, cyc = False)
    eq = sp.Matrix.hstack(*filter(lambda x: x is not None, collection['eq'].values()))
    # return collection['Q']

    # we have eq @ vecS = vecM
    # so that vecS = x0 + space * y where x0 is a particular solution and y is arbitrary
    try:
        x0, space = solve_undetermined_linear(eq, vecM)
    except:
        print('Linear system no solution. Please higher the degree by multiplying something %s.'%(
            'or use the minor term' if not minor else ''
        ))
        return collection

    splits = split_vector(collection['eq'].values())
    collection.update({'x0': x0, 'space': space, 'splits': splits})

    not_none_keys = [key for key, value in collection['Q'].items() if value is not None]

    # Main SOS solver
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


def SDPSOS(
        poly: sp.Poly,
        minor: Union[List[bool], bool] = [False, True],
        degree_limit: int = 12,
        verbose: bool = True,
        **kwargs
    ) -> Optional[SolutionSDP]:
    """
    Solve a polynomial SOS problem with SDP.

    Parameters
    ----------
    poly : sp.Poly
        Polynomial to be solved.
    minor : Union[List[bool], bool]
        For a problem of even degree, if it holds for all real numbers, it might be in the
        form of sum of squares. However, if it only holds for positive real numbers, then
        it might be in the form of \sum (...)^2 + \sum ab(...)^2. Note that we have an
        additional term in the latter, which is called the minor term. If we need to
        add the minor term, please set minor = True.
        The function also supports multiple trials. The default is [False, True], which
        first tries to solve the problem without the minor term.
    degree_limit : int
        The maximum degree of the polynomial to be solved. When the degree is too high,
        return None.
    verbose : bool
        If True, print the information of the problem.
    """
    degree = deg(poly)
    if degree > degree_limit or degree < 2:
        return None

    manifold = RootSubspace(poly)
    if verbose:
        print(manifold)

    if isinstance(minor, (bool, int)):
        minor = [minor]

    for minor_ in minor:
        if verbose:
            print('SDP Minor = %d:'%minor_)

        with indented_print():
            try:
                collection = _sdp_sos(poly, manifold, minor_, verbose)
                if isinstance(collection['M']['major'], sp.Matrix):
                    return create_solution_from_M(
                        collection['poly'], 
                        collection['M']
                    )
                    if verbose:
                        print('Success.')
            except Exception as e:
                print(e)
            print('Failed.')

    return None