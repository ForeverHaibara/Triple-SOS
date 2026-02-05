from typing import Tuple, Dict, List, Optional, Union, Callable, Any
from warnings import warn

import numpy as np
from sympy import Poly, Expr, Symbol

from .preprocess import ProofNode, ProofTree, SolvePolynomial
from .preprocess.reparam import Reparametrization
from .linsos.linsos import LinearSOSSolver
from .pivoting.pivoting import Pivoting
from .structsos.structsos import StructuralSOSSolver
from .symsos import SymmetricSubstitution
from .sdpsos.sdpsos import SDPSOSSolver

from .solution import Solution
from ..utils import PolyReader

NAME_TO_METHOD = {
    'StructuralSOS': StructuralSOSSolver,
    'LinearSOS': LinearSOSSolver,
    'SDPSOS': SDPSOSSolver,
    'SymmetricSOS': SymmetricSubstitution,
    'Pivoting': Pivoting,
    'Reparametrization': Reparametrization,
}
NAME_TO_METHOD.update({v.__name__: v for v in NAME_TO_METHOD.values()})


DEFAULT_SAVE_SOLUTION = lambda x: (str(x.solution) if x is not None else '')


def sum_of_squares(
    expr: Expr,
    ineq_constraints: Union[List[Expr], Dict[Expr, Expr]] = {},
    eq_constraints: Union[List[Expr], Dict[Expr, Expr]] = {},
    *,
    roots: Optional[List[Union[Tuple[Expr, ...], Dict[Symbol, Expr]]]] = None,
    verbose: bool = False,
    time_limit: float = 3600.,
    methods: Optional[List[str]] = None,
    configs: Dict[str, Dict] = {},
    mode: str = "fast",
    method_order: Optional[List[str]] = None, # deprecated
) -> Optional[Solution]:
    """
    Main function for sum-of-squares decomposition.

    Examples
    ----------
    The function relies on SymPy for symbolic computation. First, import necessary items:

        >>> from sympy.abc import x, y, a, b, c
        >>> from sympy import Expr, Function

    Call the function by passing in a SymPy expression:

        >>> result = sum_of_squares(a**2+b**2+c**2-a*b-b*c-c*a)

    The result will be `None` if the function fails. However, when the function fails
    it does not mean the polynomial is non positive semidefinite or non sum-of-squares. It only
    means the function could not find a solution.
    If result is not `None`, it will be a solution class. To access the expression, use .solution:

        >>> print(isinstance(result.solution, Expr), result.solution) # doctest: +SKIP
        True (Σ(a - b)**2)/2

    The solution expression might involve `CyclicSum` and `CyclicProduct` classes, which are not native
    to SymPy, but defined in this package. The permutation groups are not displayed by default and
    might be sometimes misleading. To avoid ambiguity and to expand them, use `.doit()` on SymPy expressions:

        >>> result.solution.doit() # doctest: +SKIP
        (-a + c)**2/2 + (a - b)**2/2 + (b - c)**2/2


    ### Constraints

    If we want to add constraints for the domain of the variables, we can pass in a list of inequality
    or equality constraints. This should be the second and the third argument respectively. Here is
    an example for the constraints a,b,c >= 0:

        >>> sum_of_squares(a*(a-b)*(a-c)+b*(b-c)*(b-a)+c*(c-a)*(c-b), [a,b,c]).solution # doctest: +SKIP
        ((Σ(a - b)**2*(a + b - c)**2)/2 + Σa*b*(a - b)**2)/(Σa)

    If we want to track the constraints, we can also pass in a dictionary to imply the "name" of the
    constraints:

        >>> sum_of_squares(((a+2)*(b+2)*(c+2)*(a**2/(2+a)+b**2/(2+b)+c**2/(2+c)-1)).cancel(), [a,b,c], {a*b*c-1:x}).solution # doctest: +SKIP
        x*(Σ(2*a + 13))/6 + Σa*(b - c)**2 + (Σa*b*(c - 1)**2)/6 + 5*(Σ(a - 1)**2)/6 + 7*(Σ(a - b)**2)/12

        >>> sum_of_squares(x+y+z-(x*y+y*z+z*x), {x:x, y:y, z:z, 4-(x*y+y*z+z*x+x*y*z):a}).solution # doctest: +SKIP
        (a*(Σ(x**2 + 2*x*y)) + Σx*y*(x - y)**2 + (Σx*y*z*(x - y)**2)/2)/(Σ(x*y*z + 4*x*y + 4*x))

        >>> G = Function("G")
        >>> sum_of_squares(x*(y-z)**2+y*(z-x)**2+z*(x-y)**2, {x:G(x),y:G(y),z:G(z)}).solution # doctest: +SKIP
        Σ(x - y)**2*G(z)


    ### Assumptions

    In the current, all SymPy symbol assumptions are ignored and symbols are treated as
    real variables. To claim nonnegativity of symbols, just add them to `ineq_constraints`.
    Integer or noncommutative symbol assumptions are not supported in the current either:

        >>> from sympy import Symbol
        >>> _x = Symbol("x", positive=True)
        >>> sum_of_squares(_x**2 + 3*_x + 1) is None
        True
        >>> sum_of_squares(_x**2 + 3*_x + 1, [_x]) is not None
        True


    Parameters
    ----------
    expr: Expr
        The expression to perform sum of squares on.
    ineq_constraints: Union[List[Expr], Dict[Expr, Expr]]
        Inequality constraints to the problem. This assumes g_1(x) >= 0, g_2(x) >= 0, ...
    eq_constraints: Union[List[Expr], Dict[Expr, Expr]]
        Equality constraints to the problem. This assumes h_1(x) = 0, h_2(x) = 0, ...
    roots: Optional[List[Union[Tuple[Expr, ...], Dict[Symbol, Expr]]]]
        Equality cases of the expression. This saves the time for searching equality
        cases if provided.
    verbose: bool
        Whether to print information during the solving process. Defaults to False.
    time_limit: float
        The time limit (in seconds) for the solver. Defaults to 3600. When the time limit is
        reached, the solver is killed when it returns to the main loop.
        However, it might not be killed instantly if it is stuck in an internal function.
    configs: Dict[str, Dict]
        The configurations for each method.
        It should be a dictionary containing the ProofNode classes as keys and the kwargs as values.
    methods: Optional[List[str]]
        The methods to try.
    mode: str
        Experimental. The mode of the solver. Defaults to 'fast'. Supports 'fast' and 'pretty'.
        If 'pretty', it traverses all methods and selects the most pretty solution.


    Returns
    ----------
    Optional[Solution]
        The solution. If no solution is found, None is returned.
    """

    problem = ProofNode.new_problem(expr, ineq_constraints, eq_constraints)
    problem.set_roots(roots)

    if method_order is not None:
        warn("method_order is deprecated. Use methods instead.", DeprecationWarning, stacklevel=2)
        methods = methods or method_order

    if methods is not None:
        _methods = [NAME_TO_METHOD[_] for _ in methods if _ in NAME_TO_METHOD\
                    or isinstance(_, ProofNode)]
        if len(_methods) != len(methods):
            diff = set(methods) - set(_methods)
            raise ValueError(f"Methods {diff} are not supported.")
        methods = _methods

    _configs = {
        ProofTree: {
            "mode": mode,
            "verbose": verbose,
            "time_limit": time_limit,
        },
        ProofNode: {
            "verbose": verbose,
        },
        SolvePolynomial: {
            "solvers": methods,
        },
    }
    _configs.update(configs)
    return problem.sum_of_squares(_configs)


def sum_of_squares_multiple(
    polys: Union[List[Union[Poly, str]], str],
    ineq_constraints: Union[List[Expr], Dict[Expr, Expr]] = {},
    eq_constraints: Union[List[Expr], Dict[Expr, Expr]] = {},
    poly_reader_configs: Dict[str, Any] = {},
    save_result: Union[bool, str] = True,
    save_solution_method: Callable[[Optional[Solution]], str] = DEFAULT_SAVE_SOLUTION,
    verbose_progress: bool = True,
    **sos_kwargs
):
    """
    TODO: This function is currently unmaintained, and is not intended to be used.

    Decompose multiple polynomials into sum of squares and return the results
    as a pandas DataFrame.

    Parameters
    ----------
    polys : Union[List[Union[Poly, str]], str]
        The polynomials to solve. If it is a string, it will be treated as a file name.
        If it is a list of strings, each string will be treated as a polynomial.
        Empty lines will be ignored.
    ineq_constraints: Union[List[Expr], Dict[Expr, Expr]]
        Inequality constraints to all problems. This assumes g_1(x) >= 0, g_2(x) >= 0, ...
    eq_constraints: Union[List[Expr], Dict[Expr, Expr]]
        Equality constraints to all problems. This assumes h_1(x) = 0, h_2(x) = 0, ...
    poly_reader_configs : Dict[str, Any]
        The configurations for the PolyReader. It should be a dictionary containing the kwargs.
    save_result : Union[bool, str]
        Whether to save the results. If True, it will be saved to a csv file in the same directory
        as the input file. If False, it will not be saved. If it is a string, it will be treated
        as the file name to save the results.
    save_solution_method : Callable[[Optional[Solution]], str]
        The method to convert a solution to string for saving the result.
        It will be applied on each solution. It should handle None as well.
    verbose_progress : bool
        Whether to show the progress bar. Requires tqdm to be installed.
    **sos_kwargs
        The keyword arguments to pass to the `sum_of_squares` function.

    Returns
    ----------
    pd.DataFrame
        The results as a pandas DataFrame.
    """
    from time import time

    try:
        import pandas as pd
    except ImportError:
        raise ImportError('Please install pandas to use this function.')

    if 'ignore_errors' not in poly_reader_configs:
        poly_reader_configs['ignore_errors'] = True
    read_polys = PolyReader(polys, **poly_reader_configs)
    read_polys_zip = zip(read_polys.polys, read_polys)
    if verbose_progress:
        try:
            from tqdm import tqdm
            read_polys_zip = tqdm(read_polys_zip, total = len(read_polys))
        except ImportError:
            raise ImportError('Please install tqdm to use this function with verbose_progress=True.')

    records = []
    for poly_str, poly in read_polys_zip:
        if poly is None:
            record = {
                'problem': poly_str,
                'deg': 0,
                'method': None,
                'solution': None,
                'time': np.nan,
                'status': 'invalid',
            }
            records.append(record)
            continue

        record = {'problem': poly_str, 'deg': poly.total_degree()}
        try:
            t0 = time()
            solution = sum_of_squares(poly, ineq_constraints, eq_constraints, **sos_kwargs)
            used_time = time() - t0
            if solution is None:
                record['status'] = 'fail'
                record['solution'] = None
                record['method'] = None
            else:
                # if not solution.is_Exact:
                #     record['status'] = 'inaccurate'
                # else:
                record['status'] = 'success'
                record['solution'] = solution
                record['method'] = solution.method
        except Exception as e:
            used_time = time() - t0
            record['status'] = 'error'
            record['solution'] = None
            record['method'] = None

        record['time'] = used_time
        records.append(record)

    return _process_records(records, save_result, save_solution_method, polys)


def _process_records(
    records: List[Dict],
    save_result: Union[bool, str] = True,
    save_solution_method: Callable[[Optional[Solution]], str] = DEFAULT_SAVE_SOLUTION,
    source: Optional[str] = None
) -> Any:
    """
    Process the records returned by sum_of_square_multiple and return a pandas DataFrame.

    Parameters
    ----------
    records : List[Dict]
        The records returned by sum_of_square_multiple.
    save_result : Union[bool, str]
        Whether to save the results. If True, it will be saved to a csv file in the same directory
        as the input file. If False, it will not be saved. If it is a string, it will be treated
        as the file name to save the results.
    save_solution_method : Callable[[Optional[Solution]], str]
        The method to convert a solution to string for saving the result.
        It will be applied on each solution. It should handle None as well.
    source : Optional[str]
        The file name of the input file. If None, it will not be used.
    """
    from time import strftime
    import os

    import pandas as pd
    records_pd = pd.DataFrame(records, index = range(1, len(records)+1))
    if save_result:
        records_save = records_pd.copy()
        records_save['solution'] = records_save['solution'].apply(save_solution_method)

        if save_result is True or os.path.isdir(save_result):
            if isinstance(source, str):
                filename = os.path.basename(source) + '_'
            else:
                filename = 'sos_'
            filename += strftime('%Y%m%d_%H-%M-%S') + '.csv'
            if save_result is True:
                if isinstance(source, str):
                    dirname = os.path.dirname(source)
                else:
                    dirname = os.getcwd()
            else:
                dirname = save_result
            filename = os.path.join(dirname, filename)
        elif save_result.endswith('.csv'):
            filename = save_result
        else:
            raise ValueError('save_result must be a directory / csv file name / True.')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        records_save.to_csv(filename)

    return records_pd
