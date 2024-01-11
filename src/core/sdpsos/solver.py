from typing import List, Optional, Tuple, Callable, Dict
from contextlib import contextmanager

import numpy as np
import sympy as sp

from .rationalize import rationalize, rationalize_and_decompose
from .utils import symmetric_matrix_from_upper_vec, S_from_y


class SDPResult():
    def __init__(self, sos, solution: Dict = None):
        self.sos = sos
        self.y = None
        self.S = None
        self.decompositions = None
        self.success = False
        if solution is not None:
            self.y = solution['y']
            self.S = solution['S']
            self.decompositions = solution['decompositions']
            self.success = True

    def __getitem__(self, key):
        return getattr(self, key)

    def as_dict(self):
        return {
            'sos': self.sos,
            'y': self.y,
            'S': self.S,
            'decompositions': self.decompositions,
            'success': self.success
        }


def _check_picos(verbose = False):
    """
    Check whether PICOS is installed.
    """
    try:
        import picos
    except ImportError:
        if verbose:
            print('Cannot import picos, please use command "pip install picos" to install it.')
        return False
    return True


def _sdp_solve_with_early_stop(sos, max_iters = 50, min_iters = 10, verbose = False):
    """
    Python package PICOS solving SDP problem with CVXOPT will oftentimes
    faces ZeroDivisionError. This is due to the iterations is large while
    working precision is not enough.

    This function is a workaround to solve this. It flexibly reduces the
    number of iterations and tries to solve the problem again until
    the problem is solved or the number of iterations is less than min_iters.

    Parameters
    ----------
    sos : picos.Problem
        The SDP problem.
    max_iters : int
        Maximum number of iterations. It cuts down to half if ZeroDivisionError is raised. Defaults to 50. 
    min_iters : int
        Minimum number of iterations. Return None if max_iters < min_iters. Defaults to 10.
    verbose : bool
        If True, print the number of iterations.

    Returns
    -------
    solution : Optional[picos.Problem]
        The solution of the SDP problem. If the problem is not solved,
        return None.
    """
    sos.options.max_iterations = max_iters
    if verbose:
        print('Retry Early Stop SOS Max Iters = %d' % sos.options.max_iterations)

    try:
        solution = sos._strategy.execute()
        return solution # .primals[sos.variables['y']]
    except Exception as e:
        if isinstance(e, ZeroDivisionError):
            if max_iters // 2 >= min_iters and max_iters > 1:
                return _sdp_solve_with_early_stop(
                            sos, 
                            max_iters = max_iters // 2, 
                            min_iters = min_iters, 
                            verbose = verbose
                        )
        return None
    return None


def _add_sdp_eq(
        sos, 
        x0: np.ndarray,
        space: np.ndarray,
        y: np.ndarray,
        reg: float = 0,
        suffix: str = '_0',
        method: str = 'lmi'
    ):
    """
    Add a new variable S to a PICOS.Problem such that S is a symmetric positive
    semidefinite matrix and the upper triangular part of S = x0 + space * y.

    Parameters
    ----------
    sos : picos.Problem
        The SDP problem.
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
    method : str
        The method to construct S. Currently supports 'lmi', 'intermediate' and 'direct'.

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

    if method == 'lmi':
        x0_sym = symmetric_matrix_from_upper_vec(x0)
        space_sym = symmetric_matrix_from_upper_vec(space).reshape(k**2, -1)
        sos.add_constraint(S.vec == x0_sym + space_sym * y)

    elif method == 'intermediate':
        z = picos.RealVariable('z%s'%suffix, len(x0))
        sos.add_constraint(z == x0 + space * y)
        target = z
    elif method == 'direct':
        target = space * y + x0.reshape((-1,1))
    else:
        raise ValueError('Method %s is not supported.'%method)

    if method in ['direct', 'intermediate']:
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
        reg: float = 0,
        constraints: Optional[List[Callable]] = None
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
    constraints : Optional[List[Callable]]
        Extra constraints of the SDP problem.

        Example:
        ```
        constraints = [
            lambda sos: sos.variables['y'][0] == 0,
        ]
        ```

    Returns
    -------
    sos : picos.Problem
        The SDP problem.
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

    for constraint in constraints or []:
        sos.add_constraint(constraint(sos))

    return sos, y


def _sdp_solver(
        sos,
        x0: sp.Matrix,
        space: sp.Matrix,
        splits: List[slice],
        objectives: Optional[List[Tuple[str, Callable]]] = None,
        allow_numer: bool = False,
        verbose: bool = False,
        return_ys: bool = False
    ):
    """
    Solve the SDP problem. See details at `sdp_solver` function.

    Parameters
    ----------
    sos : picos.Problem
        The SDP problem.
    x0 : sp.Matrix
        The constant part of the equation. Stands for a particular solution
        of the space of S.
    space : sp.Matrix
        The space of S. Stands for the space constraint of S.
    splits : List[slice]
        The splits of the symmetric matrices. Each split is a slice object.
    objectives : List[Tuple[str, Callable]]
        The objectives of the SDP problem.
    allow_numer : bool
        Whether to allow numerical solution. If True, then the function will return numerical solution
        if the rational solution does not exist.
    verbose : bool
        If True, print the details.
    return_ys : bool
        This is used for debugging. If True, return all numerical solutions of y without
        performing rationalization or decompositions.

    Returns
    -------
    y_rational: sp.Matrix
        The rational solution of y.
    decompositions: List[Tuple[sp.Matrix, sp.Matrix, List]]
        The congruence decomposition of each symmetric matrix S. Each item is a tuple
        of `(S, U, diag)` where `S = U.T * diag(diag) * U` where `U` is upper triangular
        and `diag` is a diagonal matrix.
    """

    if objectives is None:
        obj_key = 'S_minor' if 'S_minor' in sos.variables else 'S_major'
        objectives = [
            ('max', lambda sos: sos.variables[obj_key].tr),
            ('min', lambda sos: sos.variables[obj_key].tr),
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
        sos._strategy = Strategy.from_problem(sos)
        solution = _sdp_solve_with_early_stop(sos, max_iters = 50)

        if solution is not None:
            # try:
            y = solution.primals[sos.variables['y']]
            # except KeyError:

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
        
        decomp = rationalize_and_decompose(y, x0, space, splits, 
            try_rationalize_with_mask = True, times = 0, check_pretty = True
        )

    # Final try: convex combination
    # Although SDP often presents low-rank solution and perturbation of low-rank solution
    # is not guaranteed to be positive semidefinite. 
    # We can take the convex combination of multiple solutions, which yields an interior point
    # in the feasible set.
    # An interior point must have rational approximant.

    if not allow_numer:
        y = np.array(ys).mean(axis = 0)


        S_numer = S_from_y(y, x0, space, splits)
        if all(_.is_positive_definite for _ in S_numer):
            lcm, times = 1260, 5
        else:
            lcm = max(1260, sp.prod(set.union(*[set(sp.primefactors(_.q)) for _ in space])))
            times = int(10 / sp.log(lcm, 10).n(15) + 3)

        if verbose:
            print('Minimum Eigenvals = %s'%[min(map(lambda x:sp.re(x), _.eigenvals())) for _ in S_numer])

        decomp = rationalize_and_decompose(y, x0, space, splits,
            try_rationalize_with_mask = False, lcm = 1260, times = times
        )
        if decomp is not None:
            return decomp

    if return_ys:
        return ys

    if len(ys) > 0 and allow_numer:
        y = sp.Matrix(ys[0])
        decomp = rationalize_and_decompose(y, x0, space, splits,
            try_rationalize_with_mask = False, times = 0, perturb = True, check_pretty = False
        )
        return decomp

    if len(ys) > 0 and (not allow_numer) and verbose:
        print('Failed to find a rational solution despite having a numerical solution. '
            'Try other multipliers might be useful.')

    return None


def _sdp_solver_partial_deflation(
        sos,
        x0: sp.Matrix,
        space: sp.Matrix,
        splits: List[slice],
        objectives: Optional[List[Tuple[str, Callable]]] = None,
        deflation_sequence: Optional[List[int]] = None,
        allow_numer: bool = False,
        verbose: bool = False,     
    ):
    """
    Solve the SDP problem. See details at `sdp_solver` function.
    We use the following idea to generate a rational solution:
    1. Solve SDP with objectives = max(y[-1]) and min(y[-1]).
    2. Set y[-1] = (max + min) / 2 as a new constraint and solve SDP again.
    3. Repeat step 2 until the solution is rational.

    Parameters
    ----------
    sos : picos.Problem
        The SDP problem.
    x0 : sp.Matrix
        The constant part of the equation. Stands for a particular solution
        of the space of S.
    space : sp.Matrix
        The space of S. Stands for the space constraint of S.
    splits : List[slice]
        The splits of the symmetric matrices. Each split is a slice object.
    objectives : List[Tuple[str, Callable]]
        IT DOES NOT SUPPORT OBJECTIVES.
    deflation_sequence : Optional[List[int]]
        The deflation sequence. If None, we use range(n) where n is the
        number of variables.
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
    assert not objectives, 'Method "partial deflation" does not support objectives.'

    @contextmanager
    def restore_constraints(sos):
        constraints_num = len(sos.constraints)
        yield
        for i in range(len(sos.constraints) - 1, constraints_num - 1, -1):
            sos.remove_constraint(i)

    n = space.shape[1]
    deflation_sequence = deflation_sequence or range(n)

    with restore_constraints(sos):
        for i in deflation_sequence:
            bounds = []
            objectives = [
                ('max', lambda sos: sos.variables['y'][i]),
                ('min', lambda sos: sos.variables['y'][i])
            ]
            solution = _sdp_solver(sos, x0, space, splits, objectives = objectives, allow_numer = False, return_ys = True, verbose = verbose)

            if solution is None or isinstance(solution, tuple):
                return solution
            elif len(solution) < 2:
                # not enough solutions
                return None

            bounds = [solution[0][i], solution[1][i]]

            # fix == (max + min) / 2
            fix = (bounds[0] + bounds[1]) / 2
            eps = (bounds[0] - bounds[1]) / 2
            if eps <= 1e-7:
                # this implies bounds[0] == bounds[1]
                fix = rationalize(fix, reliable = True) if abs(fix) > 1e-7 else 0
            elif bounds[0] > round(fix) > bounds[1]:
                fix = round(fix)
            else:
                fix = rationalize(fix, rounding = eps * .8, reliable = False)

            if verbose:
                print('Deflate y[%d] = %s Bounds = %s'%(i, fix, bounds))

            sos.add_constraint(sos.variables['y'][i] == float(fix))

    if allow_numer and len(solution) == 1:
        y = sp.Matrix(solution[0])
        decomp = rationalize_and_decompose(y, x0, space, splits,
            try_rationalize_with_mask = False, times = 0, perturb = True, check_pretty = False
        )
        return decomp

    return None


def _sdp_solver_relax(
        sos,
        x0: sp.Matrix,
        space: sp.Matrix,
        splits: List[slice],
        objectives: Optional[List[Tuple[str, Callable]]] = None,
        allow_numer: bool = False,
        verbose: bool = False,     
    ):
    """
    Solve the SDP problem. See details at `sdp_solver` function.
    We modify the problem to be a relaxation of the original problem:
    S - a * I >> 0
    and optimize max(a).

    Parameters
    ----------
    sos : picos.Problem
        The SDP problem.
    x0 : sp.Matrix
        The constant part of the equation. Stands for a particular solution
        of the space of S.
    space : sp.Matrix
        The space of S. Stands for the space constraint of S.
    splits : List[slice]
        The splits of the symmetric matrices. Each split is a slice object.
    objectives : List[Tuple[str, Callable]]
        IT DOES NOT SUPPORT OBJECTIVES.
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
    assert not objectives, 'Method "relax" does not support objectives.'

    import picos
    from picos.constraints.con_lmi import LMIConstraint

    obj_key = 'S_minor' if not 'S_major' in sos.variables else 'S_major'
    lamb = picos.RealVariable('lamb', 1)
    obj = sos.variables[obj_key]

    @contextmanager
    def restore_constraints(sos, obj, lamb):    
        for i, constraint in enumerate(sos.constraints):
            if isinstance(constraint, LMIConstraint) and obj in constraint.variables:
                # remove obj >> 0
                sos.remove_constraint(i)
                break
        sos.add_constraint((obj - lamb * picos.I(obj.shape[0])) >> 0)
        sos.add_constraint(lamb >= 0)

        yield
        sos.remove_constraint(-1)
        sos.remove_constraint(-1)
        sos.set_objective('max', obj.tr)

    with restore_constraints(sos, obj, lamb):
        objectives = [('max', lambda sos: sos.variables['lamb'])]
        solution = _sdp_solver(sos, x0, space, splits, objectives = objectives, allow_numer = allow_numer, verbose = verbose)

    return solution



def _check_method(method):
    """
    Return the corresponding function of the method.
    """
    method = method.lower()
    METHODS = {
        'partial deflation': _sdp_solver_partial_deflation,
        'trivial': _sdp_solver,
        'relax': _sdp_solver_relax
    }
    assert method in METHODS, 'Method %s is not supported. Currently supports %s'%(method, METHODS.keys())

    return METHODS[method]

def sdp_solver(
        x0: sp.Matrix, 
        space: sp.Matrix, 
        splits: List[slice],
        keys: List[str],
        method: str = 'partial deflation',
        reg: float = 0,
        constraints: Optional[List[Callable]] = None,
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
    method: str
        The method to solve the SDP problem. Currently supports:
        'partial deflation' and 'relax' and 'trivial'
    keys: List[str]
        Represent the name of each symmetric matrix `S[i]`. Should match
        the length of splits.
    reg : float
        We require `S[i]` to be positive semidefinite, but in practice
        we might want to add a small regularization term to make it
        positive definite >> reg * I.
    constraints : Optional[List[Callable]]
        Extra constraints of the SDP problem. This is not called when the problem is degenerated
        (when the degree of freedom is zero).

        Example:
        ```
        constraints = [
            lambda sos: sos.variables['y'][0] == 0,
        ]
        ```
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
            The SDP problem.
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
        sos, y = _sdp_constructor(x0, space, splits, keys, reg = reg, constraints = constraints)

        func = _check_method(method)
        if _check_picos(verbose = verbose):
            try:
                solution = func(sos, x0, space, splits, objectives = objectives, allow_numer = allow_numer, verbose = verbose)
            except KeyError:
                # This implies that the SDP problem is infeasible (even solved numerically).
                if verbose:
                    print('Cannot find numerical solution for y.')
                solution = None
    else:
        sos = None
        solution = _degenerated_solver(x0, space, splits, verbose = verbose)

    if solution is not None:
        solution = {
            'y': solution[0],
            'S': dict((key, S[0]) for key, S in zip(keys, solution[1])),
            'decompositions': dict((key, S[1:]) for key, S in zip(keys, solution[1]))
        }

    return SDPResult(sos, solution)


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

    decomp = rationalize_and_decompose(
        sp.Matrix([]).reshape(0,1), x0, space, splits,
        check_pretty = False
    )
    return decomp