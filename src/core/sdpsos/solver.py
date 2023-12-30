from typing import List, Optional, Tuple, Callable

import numpy as np
import sympy as sp

from .rationalize import (
    rationalize_with_mask, rationalize_simutaneously,
    verify_is_pretty, verify_is_positive
)

def _sos_early_stop(sos, verbose = False):
    """
    Python package PICOS solving SOS problem with CVXOPT will oftentimes
    faces ZeroDivisionError. This is due to the iterations is large while
    working precision is not enough.

    This function is a workaround to solve this. It flexibly reduces the
    number of iterations and tries to solve the problem again until
    the problem is solved or the number of iterations is less than 10.

    Parameters
    ----------
    sos : picos.Problem
        The SOS problem.
    verbose : bool
        If True, print the number of iterations.

    Returns
    -------
    solution : Optional[picos.Problem]
        The solution of the SOS problem. If the problem is not solved,
        return None.
    """
    if verbose:
        print('Retry Early Stop SOS Max Iters = %d' % sos.options.max_iterations)

    try:
        solution = sos._strategy.execute()
        return solution # .primals[sos.variables['y']]
    except Exception as e:
        if isinstance(e, ZeroDivisionError):
            max_iters = sos.options.max_iterations
            if max_iters > 9:
                sos.options.max_iterations = max_iters // 2
                return _sos_early_stop(sos)
        return None
    return None


def _add_sdp_eq(sos, x0, space, y, reg = 0, suffix = '_0'):
    """
    Add a new variable S to a PICOS.Problem such that S is a symmetric positive
    semidefinite matrix and the upper triangular part of S = x0 + space * y.

    Parameters
    ----------
    sos : picos.Problem
        The SOS problem.
    x0 : np.ndarray
        The constant part of the equation. Stands for a particular solution
        of the space of S.
    space : np.ndarray
        The space of S. Stands for the space constraint of S.
    y : picos.RealVariable
        The underlying generator of S.
    reg : float
        The regularization term of S. We require S >> reg * I.
    suffix : str
        A string suffix to the name of the variable.

    Returns
    -------
    S : picos.SymmetricVariable
        The variable S.
    """
    import picos

    # k(k+1)/2 = len(x0)
    k = round(np.sqrt(2 * len(x0) + .25) - .5)
    S = picos.SymmetricVariable('S%s'%suffix, (k,k))
    sos.add_constraint(S >> reg)

    target = space * y + x0.reshape((-1,1))
    pointer = 0
    for i in range(k):
        for j in range(i, k):
            sos.add_constraint(S[i,j] == target[pointer])
            pointer += 1
    return S


def _sdp_constructor(
        x0: sp.Matrix, 
        space: sp.Matrix, 
        splits: List[slice], 
        keys: List[str],
        reg: float = 0
    ):
    """
    Construct SDP problem: find feasible y such that
    `x0[splits[i]] + space[splits[i],:] * y = uppervec(S[i])`
    with constraint that each `S[i]` is symmetric positive semidefinite.

    Parameters
    ----------
    x0 : sp.Matrix
        The constant part of the equation. Stands for a particular solution
        of the space of S.
    space : sp.Matrix
        The space of S. Stands for the space constraint of S.
    splits : List[slice]
        The splits of the space. Each split is a slice object.
    keys : List[str]
        The keys of the variables.
    reg : float
        The regularization term of S. We require S >> reg * I.

    Returns
    -------
    sos : picos.Problem
        The SOS problem.
    y : picos.RealVariable
        The underlying generator of S.
    """
    import picos

    # SDP should use numerical algorithm
    x0_numer = np.array(x0).astype(np.float64).flatten()
    space_numer = np.array(space).astype(np.float64)

    sos = picos.Problem()
    y = picos.RealVariable('y', space.shape[1])
    for key, split in zip(keys, splits):
        S = _add_sdp_eq(sos, x0_numer[split], space_numer[split], y, reg = reg, suffix = '_%s'%key)

    return sos, y


def _sdp_solver(sos, x0, space, splits, objectives = None, allow_numer = False, verbose = False):
    """
    Solve the SDP problem. See details at `sdp_solver` function.

    Parameters
    ----------
    sos : picos.Problem
        The SOS problem.
    x0 : sp.Matrix
        The constant part of the equation. Stands for a particular solution
        of the space of S.
    space : sp.Matrix
        The space of S. Stands for the space constraint of S.
    splits : List[slice]
        The splits of the symmetric matrices. Each split is a slice object.
    objectives : List[Tuple[str, Callable]]
        The objectives of the SOS problem.
    allow_numer : bool
        Whether to allow numerical solution. If True, then the function will return numerical solution
        if the rational solution does not exist.
    verbose : bool
        If True, print the details.

    Returns
    -------
    y_rational: sp.Matrix
        The rational solution of y.
    decompositions: List[Tuple[sp.Matrix, sp.Matrix, List]]
        The congruence decomposition of each symmetric matrix S. Each item is a tuple
        of `(S, U, diag)` where `S = U.T * diag(diag) * U` where `U` is upper triangular
        and `diag` is a diagonal matrix.
    """
    try:
        import picos
    except ImportError:
        if verbose:
            print('Cannot import picos, please use command "pip install picos" to install it.')
        return None

    if objectives is None:
        # x = np.random.random((6,6))
        # objectives = [('max', lambda sos: sos.variables['S_0']|x)]
        obj_key = 'S_minor' if 'S_minor' in sos.variables else 'S_major'
        objectives = [
            ('max', lambda sos: sos.variables[obj_key].tr),
            ('max', lambda sos: sos.variables[obj_key]|1)
        ]
        x = np.random.randn(*sos.variables[obj_key].shape)
        objectives.append(('max', lambda sos: sos.variables[obj_key]|x))

    # record all numerical solution of y
    # so that we can take the convex combination in the final step
    ys = []

    for objective in objectives:
        # try each of the objectives

        sos.set_objective(objective[0], objective[1](sos))
        y = None

        from picos.modeling.strategy import Strategy
        sos.options.max_iterations = 50
        sos._strategy = Strategy.from_problem(sos)
        solution = _sos_early_stop(sos)

        if solution is not None:
            try:
                y = solution.primals[sos.variables['y']]
            except KeyError:
                if verbose:
                    print('Cannot find numerical solution for y.')
                return None

            # NOTE: PICOS uses a different vectorization of symmetric matrices
            #       (off-diagonal elements are divided by sqrt(2))
            #       so if we need to convert it back, we had better use its API.
            # S0 = (SymmetricVectorization((6,6)).devectorize(cvxopt.matrix(list(solution.primals.values())[0])))
            # print(np.linalg.eigvalsh(np.array(S0)))
            # print(S0)
        
        if y is None:
            continue

        # perform rationalization
        y = np.array(y)
        ys.append(y)
        y_rational = rationalize_with_mask(y, zero_tolerance = 1e-7)

        # verify -> ok -> return / no: next objective
        if verify_is_pretty(y_rational):
            decompositions = verify_is_positive(x0 + space * y_rational, splits)
            if decompositions is not None:
                return y_rational, decompositions


    # Final try: convex combination
    # Although SDP often presents low-rank solution and perturbation of low-rank solution
    # is not guaranteed to be positive semidefinite. 
    # We can take the convex combination of multiple solutions, which yields an interior point
    # in the feasible set.
    # An interior point must have rational approximant.

    if len(ys) > 1 and not allow_numer:
        y = np.array(ys).mean(axis = 0)

        lcm = max(1260, sp.prod(set.union(*[set(sp.primefactors(_.q)) for _ in space])))
        times = int(10 / sp.log(lcm, 10).n(15) + 3)
        for y_rational in rationalize_simutaneously(y, lcm, times = times):
            if verify_is_pretty(y_rational):
                decompositions = verify_is_positive(x0 + space * y_rational, splits)
                if decompositions is not None:
                    return y_rational, decompositions

    if len(ys) > 0 and allow_numer:
        y = sp.Matrix(ys[0])
        decompositions = verify_is_positive(x0 + space * y, splits, allow_numer = allow_numer)
        if decompositions is not None:
            return y, decompositions

    if len(ys) > 0 and (not allow_numer) and verbose:
        print('Failed to find a rational solution despite having a numerical solution. '
              'Try other multipliers might be useful. '
              'Currently using lcm = %d, times = %d'%(lcm, times))
        # print(ys)
        # print('The mixed solution is:\n')
        # for y in ys:
        #     y = x0 + space * sp.Matrix(y.flatten().tolist())
        #     for split in splits:
        #         M = LowRankHermitian(None, sp.Matrix(y[split]).n(20)).S
        #         print('Eigenvals =', M.eigenvals(), '\n', M)

    return None


def sdp_solver(
        x0: sp.Matrix, 
        space: sp.Matrix, 
        splits: List[slice],
        keys: List[str],
        reg: float = 0,
        objectives: Optional[List[Tuple[str, Callable]]] = None,
        allow_numer: bool = False,
        verbose: bool = False
    ):
    """
    Solve SDP problem: find feasible y such that
    `x0[splits[i]] + space[splits[i],:] * y = uppervec(S[i])`
    with constraint that each `S[i]` is symmetric positive semidefinite.

    Parameters
    ----------
    x0 : sp.Matrix
        The constant part of the equation. Stands for a particular solution
        of the space of S.
    space : sp.Matrix
        The space of S. Stands for the space constraint of S.
    splits : List[slice]
        Vector `x0 + space * y` is the concatenation of multiple
        vectors `uppervec(S[i])`. Each `S[i]` is a symmetric matrix.
        This parameter indicates how to split the vector `x0 + space * y`
        into multiple vectors.
    keys: List[str]
        Represent the name of each symmetric matrix `S[i]`. Should match
        the length of splits.
    reg : float
        We require `S[i]` to be positive semidefinite, but in practice
        we might want to add a small regularization term to make it
        positive definite >> reg * I.
    objectives : Optional[List[Tuple[str, Callable]]]
        Although it suffices to find one feasible solution, we might 
        use objective to find particular feasible solution that has 
        good rational approximant. This parameter takes in multiple objectives, 
        and the solver will try each of the objective. If still no 
        approximant is found, the final solution will average this 
        SOS solution and perform rationalization. Note that SDP problem is 
        convex so the convex combination is always feasible and not on the
        boundary.

        Example: 
        ```
        objectives = [
            ('max', lambda sos: sos.variables['S_major'].tr),
            ('max', lambda sos: sos.variables['S_major']|1)
        ]
        ```        
    allow_numer : bool
        Whether to allow numerical solution. If True, then the function will return numerical solution
        if the rational solution does not exist.

    Returns
    -------
    Returns a dict if the problem is solved, otherwise return None.
    The dict contains the following keys:
        sos : picos.Problem
            The SOS problem.
        y : sp.Matrix
            The rational solution of y.
        S : Dict[str, sp.Matrix]
            The solution of each symmetric matrix S.
        decompositions : Dict[str, Tuple[sp.Matrix, List[sp.Rational]]]
            The congruence decomposition of each symmetric matrix S. Each item is a tuple.
    """
    if not isinstance(keys, list):
        keys = list(keys) # avoid troubles of iterator

    if verbose:
        print('Degree of freedom: %d'%space.shape[1])

    if space.shape[1] > 0:
        sos, y = _sdp_constructor(x0, space, splits, keys, reg = reg)
        solution = _sdp_solver(sos, x0, space, splits, objectives = objectives, allow_numer = allow_numer, verbose = verbose)
    else:
        sos = None
        solution = _degenerated_solver(x0, space, splits, verbose = verbose)

    if solution is None:
        return None

    return {
        'sos': sos,
        'y': solution[0],
        'S': dict((key, S[0]) for key, S in zip(keys, solution[1])),
        'decompositions': dict((key, S[1:]) for key, S in zip(keys, solution[1]))
    }


def _degenerated_solver(
        x0: sp.Matrix,
        space: sp.Matrix,
        splits: List[slice],
        verbose: bool = True
    ):
    """
    When there is zero degree of freedom, the space degenerates to matrix with shape[1] == 0.
    Then the unique solution is presented by x0.

    The function will construct S from x0.
    """
    y = x0

    decompositions = verify_is_positive(x0, splits)
    if decompositions is None:
        return None

    return y, decompositions
