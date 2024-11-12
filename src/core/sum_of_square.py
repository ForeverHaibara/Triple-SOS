from typing import Optional, Dict, List, Union, Callable, Any

import numpy as np
import sympy as sp
from sympy import Expr

from .linsos import LinearSOS
from .structsos import StructuralSOS
from .symsos import SymmetricSOS
from .sdpsos import SDPSOS
from .shared import sanitize_input

from ..utils import deg, PolyReader, Solution

NAME_TO_METHOD = {
    'LinearSOS': LinearSOS,
    'StructuralSOS': StructuralSOS,
    'SymmetricSOS': SymmetricSOS,
    'SDPSOS': SDPSOS
}

METHOD_ORDER = ['StructuralSOS', 'LinearSOS', 'SDPSOS', 'SymmetricSOS']

DEFAULT_CONFIGS = {
    'LinearSOS': {

    },
    'StructuralSOS': {

    },
    'SymmetricSOS': {

    },
    'SDPSOS': {

    }
}


# @sanitize_input(homogenize=True)
def sum_of_square(
        poly: Union[sp.Poly, sp.Expr],
        ineq_constraints: Union[List[Expr], Dict[Expr, Expr]] = {},
        eq_constraints: Union[List[Expr], Dict[Expr, Expr]] = {},
        method_order: Optional[List[str]] = METHOD_ORDER,
        configs: Optional[Dict[str, Dict]] = DEFAULT_CONFIGS
    ) -> Optional[Solution]:
    """
    Main function for sum of square decomposition.

    Examples    
    ----------
    The function relies on SymPy for symbolic computation. We first import some items:

        >>> from sympy.abc import x, y, a, b, c
        >>> from sympy import Expr, Function

    Call the function by passing in a SymPy polynomial or polynomial-like expression:

        >>> result = sum_of_square(a**2+b**2+c**2-a*b-b*c-c*a)

    The result will be None if the function fails. However, when the function fails
    it does not mean the polynomial is non positive semidefinite or non sum-of-squares. It only
    means the function could not find a solution.
    If result is not None, it will be a solution class. To access the expression, use .solution:

        >>> print(isinstance(result.solution, Expr), result.solution)
        True (Σ(a - b)**2)/2

    The solution expression might involve CyclicSum and CyclicProduct classes, which are not native
    to SymPy, but defined in this package. The permutation groups are not displayed by default and
    might be sometimes misleading. To avoid ambiguity and to expand them, use .doit() on SymPy expressions:

        >>> result.solution.doit()
        (-a + c)**2/2 + (a - b)**2/2 + (b - c)**2/2

    If we want to add constraints for the domain of the variables, we can pass in a list of inequality
    or equality constraints. This should be the second and the third argument respectively. Here is
    an example for the constraints a,b,c >= 0:

        >>> sum_of_square(a*(a-b)*(a-c)+b*(b-c)*(b-a)+c*(c-a)*(c-b), [a,b,c]).solution
        ((Σ(a - b)**2*(a + b - c)**2)/2 + Σa*b*(a - b)**2)/(Σa)

    If we want to track the constraints, we can also pass in a dictionary to imply the "name" of the
    constraints:

        >>> sum_of_square(((a+2)*(b+2)*(c+2)*(a**2/(2+a)+b**2/(2+b)+c**2/(2+c)-1)).cancel(), [a,b,c], {a*b*c-1:x}).solution
        x*(Σa)/3 + 13*x + Σa*(b - c)**2 + (Σa*b*(c - 1)**2)/6 + 5*(Σ(a - 1)**2)/6 + 7*(Σ(a - b)**2)/12

        >>> sum_of_square(x+y+z-(x*y+y*z+z*x), {x:x, y:y, z:z, 4-(x*y+y*z+z*x+x*y*z):a}).solution
        (6*a*(Σx) + 3*a*(Σx**2) + 3*(Σx*y*(x - y)**2) + 3*(Σx*y*z*(x - y)**2)/2)/(Σ(x*y*z + 6*x*y + 12*x + 8))

        >>> G = Function("G")
        >>> sum_of_square(x*(y-z)**2+y*(z-x)**2+z*(x-y)**2, {x:G(x),y:G(y),z:G(z)}).solution
        Σ(x - y)**2*G(z)

    Parameters
    ----------
    poly: Union[sp.Poly, sp.Expr]
        The polynomial to perform SOS on.
    ineq_constraints: Union[List[Expr], Dict[Expr, Expr]]
        Inequality constraints to the problem. This assumes g_1(x) >= 0, g_2(x) >= 0, ...
    eq_constraints: Union[List[Expr], Dict[Expr, Expr]]
        Equality constraints to the problem. This assumes h_1(x) = 0, h_2(x) = 0, ...
    method_order: List[str]
        The order of methods to try. Defaults to METHOD_ORDER.
    configs: Dict[str, Dict]
        The configurations for each method. Defaults to DEFAULT_CONFIGS.
        It should be a dictionary containing the method names as keys and the kwargs as values.

    Returns
    ----------
    Optional[Solution]
        The solution. If no solution is found, None is returned.
    """
    if method_order is None:
        method_order = METHOD_ORDER
    if configs is None:
        configs = DEFAULT_CONFIGS

    for method in method_order:
        config = configs.get(method, {})

        method = NAME_TO_METHOD[method]
        solution = method(poly, ineq_constraints, eq_constraints, **config)
        if solution is not None:
            return solution

    return None


def sum_of_square_multiple(
        polys: Union[List[Union[sp.Poly, str]], str],
        ineq_constraints: Union[List[Expr], Dict[Expr, Expr]] = {},
        eq_constraints: Union[List[Expr], Dict[Expr, Expr]] = {},
        method_order: List[str] = METHOD_ORDER,
        configs: Dict[str, Dict] = DEFAULT_CONFIGS,
        poly_reader_configs: Dict[str, Any] = {},
        save_result: Union[bool, str] = True,
        save_solution_method: Union[str, Callable] = 'str_formatted',
        verbose_sos: bool = False,
        verbose_progress: bool = True
    ):
    """
    Decompose multiple polynomials into sum of squares and return the results
    as a pandas DataFrame.

    Parameters
    ----------
    polys : Union[List[Union[sp.Poly, str]], str]
        The polynomials to solve. If it is a string, it will be treated as a file name.
        If it is a list of strings, each string will be treated as a polynomial.
        Empty lines will be ignored.
    ineq_constraints: Union[List[Expr], Dict[Expr, Expr]]
        Inequality constraints to all problems. This assumes g_1(x) >= 0, g_2(x) >= 0, ...
    eq_constraints: Union[List[Expr], Dict[Expr, Expr]]
        Equality constraints to all problems. This assumes h_1(x) = 0, h_2(x) = 0, ...
    method_order : List[str]
        The order of methods to try. Defaults to METHOD_ORDER.
    configs : Dict[str, Dict]
        The configurations for each method. Defaults to DEFAULT_CONFIGS.
        It should be a dictionary containing the method names as keys and the kwargs as values.
    poly_reader_configs : Dict[str, Any]
        The configurations for the PolyReader. It should be a dictionary containing the kwargs.
    save_result : Union[bool, str]
        Whether to save the results. If True, it will be saved to a csv file in the same directory
        as the input file. If False, it will not be saved. If it is a string, it will be treated
        as the file name to save the results.
    save_solution_method : Union[str, Callable]
        The method to save the solution. If it is a string, it will be treated as the attribute
        name of the solution to save. If it is a callable, it will be called on the solution to
        save the result. It should handle None as well.
    verbose_sos : bool
        Whether to send verbose=True to the sum_of_square function.
    verbose_progress : bool
        Whether to show the progress bar. Requires tqdm to be installed.

    Returns
    ----------
    pd.DataFrame
        The results as a pandas DataFrame.
    """
    from time import time
    from copy import deepcopy

    try:
        import pandas as pd
    except ImportError:
        raise ImportError('Please install pandas to use this function.')

    configs = deepcopy(configs)
    for key in DEFAULT_CONFIGS:
        if key not in configs:
            configs[key] = {}
        if key in ('LinearSOS', 'SDPSOS'):
            configs[key]['verbose'] = verbose_sos

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

        record = {'problem': poly_str, 'deg': deg(poly)}
        try:
            t0 = time()
            solution = sum_of_square(poly, ineq_constraints, eq_constraints,
                method_order=method_order, configs=configs
            )
            used_time = time() - t0
            if solution is None:
                record['status'] = 'fail'
                record['solution'] = None
                record['method'] = None
            else:
                if not solution.is_equal:
                    record['status'] = 'inaccurate'
                else:
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
        save_solution_method: Union[str, Callable] = 'str_formatted',
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
    save_solution_method : Union[str, Callable]
        The method to save the solution. If it is a string, it will be treated as the attribute
        name of the solution to save. If it is a callable, it will be called on the solution to
        save the result.
    source : Optional[str]
        The file name of the input file. If None, it will not be used.
    """
    from time import strftime
    import os
    if isinstance(save_solution_method, str):
        attr = save_solution_method
        save_solution_method = lambda x: getattr(x, attr) if x is not None else None

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