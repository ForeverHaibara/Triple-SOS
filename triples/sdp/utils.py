from typing import Union, Optional, Tuple, List, Dict, Callable, Generator, Any

from numpy import ndarray
import numpy as np
from sympy import Matrix, MatrixBase, Expr, Rational, Symbol, re, eye, collect
from sympy.core.relational import GreaterThan, StrictGreaterThan, LessThan, StrictLessThan, Equality, Relational
from sympy.core.singleton import S as singleton

from .arithmetic import vec2mat, reshape

def S_from_y(
        y: Matrix,
        x0_and_space: Dict[str, Tuple[Matrix, Matrix]]
    ) -> Dict[str, Matrix]:
    """
    Return the symmetric matrices S from the vector y.

    Parameters
    ----------
    y : Matrix
        The vector to be checked.
    x0_and_space : Dict[str, Tuple[Matrix, Matrix]]
        vec(S[key]) = x0[key] + space[key] @ y.
    splits : List[slice]
        The splits of the symmetric matrices. Each split is a slice object.
    mode : int
        Mode of conversion. 0: direct, 1: upper part, 2: isometric.

    Returns
    ----------
    S_dict : Matrix
        Each S[key] satisfies that vec(S[key]) = x0[key] + space[key] @ y.
    """
    if not isinstance(y, MatrixBase):
        y = Matrix(y)

    S_dict = {}
    for key, (x0, space) in x0_and_space.items():
        vecS = x0 + space * y
        S = vec2mat(vecS)
        S_dict[key] = S
    return S_dict



_RELATIONAL_TO_OPERATOR = {
    GreaterThan: (1, '>='),
    StrictGreaterThan: (1, '>'),
    LessThan: (-1, '>='),
    StrictLessThan: (-1, '>'),
    Equality: (1, '=='),
    '>': (1, '>'),
    '>=': (1, '>='),
    '__gt__': (1, '>'),
    '__ge__': (1, '>='),
    '<': (-1, '>'),
    '<=': (-1, '>='),
    '__lt__': (-1, '>'),
    '__le__': (-1, '>='),
    '==': (1, '=='),
    '=': (1, '=='),
    '__eq__': (1, '=='),
}


def decompose_matrix(
        M: Matrix,
        variables: Optional[List[Symbol]] = None
    ) -> Tuple[Matrix, Matrix, Matrix]:
    """
    Decomposes a symbolic matrix into the form vec(M) = x + A @ v
    where x is a constant vector, A is a constant matrix, and v is a vector of variables.

    Please always ensure that the matrix is linear with respect to the variables.
    Nonlinear terms will be ignored.

    See also in `sympy.solvers.solveset.linear_eq_to_matrix`.

    Parameters
    ----------
    M : Matrix
        The matrix to be decomposed.
    variables : List[Symbol]
        The variables to be used in the decomposition. If None, it uses M.free_symbols.

    Returns
    ----------
    x : Matrix
        The constant vector.
    A : Matrix
        The constant matrix.
    v : Matrix
        The vector of variables.
    """
    rows, cols = M.shape
    if variables is None:
        variables = list(M.free_symbols)
        variables = sorted(variables, key = lambda x: x.name)
    variable_index = {var: idx for idx, var in enumerate(variables)}

    v = Matrix(variables)
    x = Matrix.zeros(rows * cols, 1)
    A = Matrix.zeros(rows * cols, len(variables))

    for i in range(rows):
        for j in range(cols):
            expr = M[i, j]
            terms = collect(expr, variables, evaluate=False)

            constant_term = terms.pop(singleton.One, 0)  # Extract and remove constant term for x
            x[i * cols + j] = constant_term

            for term, coeff in terms.items():
                A[i * cols + j, variable_index[term]] = coeff  # Extract coefficients for A

    return x, A, v


def exprs_to_arrays(locals: Dict[str, Any], symbols: List[Symbol],
        exprs: List[Union[Callable, Expr, Relational, Tuple[Matrix, float], Tuple[Matrix, float, str]]],
        dtype: Optional[Any] = None
    ) -> List[Union[Tuple[Matrix, Matrix], Tuple[Matrix, Matrix, str]]]:
    """
    Convert linear expressions to arrays with respect to the given symbols.

    Parameters
    ----------
    locals : Dict[str, Any]
        The local variables.
    symbols : List[Symbol]
        The free symbols.
    exprs : List[Union[Callable, Expr, Relational, Matrix]]
        For each expression, it can be a Callable, Expr, Relational.
        If it is a Callable, it is first applied on the locals.
        If it is a Expr, it should be linear with respect to the given symbols.
        If it is a Relational, it should be linear with respect to the given symbols.
    dtype : Optional[Any]
        The data type of the returned arrays. If given, matrices are converted
        to numpy arrays with the given data type.

    Returns
    ----------
    List[Tuple[A, b, [, operator]]] :
        For each expression, the returned `A`, `b` satisfy that `A @ Matrix(symbols) - b = expression`.
        If the expression is a Relational, `A @ Matrix(symbols) (operator) b` should be an
        inequality equivalent to the expression.
        Matrix `A` must be 2D and `b` must be a vector.

    Examples
    ----------
    >>> from sympy.abc import a, b, c
    >>> from sympy import Matrix, Eq
    >>> exprs_to_arrays({'x': a}, [a,b,c],
    ...      [2*a+3*b+4*c+1,
    ...      a-2*b<=3*c-5,
    ...      Eq(a/2-b/3-c/4,1),
    ...      lambda l: (l['x']-2),
    ...      (Matrix([[7],[8],[9]]), 2),
    ...      ([0.2,0.3,0.4], [-0.5], '>')])  # doctest: +NORMALIZE_WHITESPACE
    [(Matrix([[2, 3, 4]]), Matrix([[-1]])),
     (Matrix([[-1, 2, 3]]), Matrix([[5]]), '>='),
     (Matrix([[1/2, -1/3, -1/4]]), Matrix([[1]]), '=='),
     (Matrix([[1, 0, 0]]), Matrix([[2]])),
     (Matrix([[7, 8, 9]]), Matrix([[2]])),
     (Matrix([[0.2, 0.3, 0.4]]), Matrix([[-0.5]]), '>')]

    2D matrices are also supported:

    >>> exprs_to_arrays(None, [a,b,c], 
    ...     [([[4,5,6],[1,2,3]], [[-4,-8]])], dtype=int) # doctest: +NORMALIZE_WHITESPACE
    [(array([[4, 5, 6], [1, 2, 3]]), array([-4, -8]))]
    """
    op_list = []
    vec_list = []
    index_list = []
    result = [None for _ in range(len(exprs))]
    for i, expr in enumerate(exprs):
        c, op = 0, None
        if callable(expr):
            expr = expr(locals)
        if isinstance(expr, tuple):
            if len(expr) == 3:
                expr, c, op = expr
            elif len(expr) == 2:
                expr, c = expr
            else:
                raise ValueError("The tuple should be of length 2 or 3.")
        if isinstance(expr, Relational):
            sign, op = _RELATIONAL_TO_OPERATOR[expr.__class__]
            expr = expr.lhs - expr.rhs if sign == 1 else expr.rhs - expr.lhs
            c = -c if sign == -1 else c
        if isinstance(expr, (Expr, int, float)):
            vec_list.append(expr)
            op_list.append(op)
            index_list.append(i)
        elif isinstance(expr, (list, ndarray, MatrixBase)):
            if isinstance(expr, list):
                expr = Matrix(expr)
            if isinstance(c, list):
                c = Matrix(c)
            if op is not None:
                if op in _RELATIONAL_TO_OPERATOR:
                    sign, op = _RELATIONAL_TO_OPERATOR[op]
                else:
                    raise ValueError(f"The operator {op} at line {i} is not supported.")
                if sign == -1:
                    expr, c = -expr, -c
                result[i] = (expr, c, op)
            else:
                result[i] = (expr, c)
        else:
            raise ValueError(f"The expression {type(expr)} at line {i} is not supported.")

    const, A, _ = decompose_matrix(Matrix(vec_list), symbols)

    for j in range(len(index_list)):
        i = index_list[j]
        if op_list[j] is not None:
            result[i] = (A[j,:], -const[j], op_list[j])
        else:
            result[i] = (A[j,:], -const[j])

    nvars = len(symbols)
    for i in range(len(result)):
        expr, c = result[i][0], result[i][1]
        if not isinstance(c, (ndarray, MatrixBase)):
            c = Matrix([c])
        elif isinstance(c, ndarray):
            c = c.flatten()
        if isinstance(expr, ndarray):
            expr = expr.reshape(1, -1)
        elif isinstance(expr, MatrixBase):
            if expr.shape[0] == nvars and expr.shape[1] == 1: # column vec
                expr = reshape(expr, (1, nvars))
            if expr.shape[1] != nvars:
                raise ValueError(f"Invalid shape of expr matrix, expected (*,{nvars}), but got {expr.shape}.")
            # expr = expr.reshape(expr.shape[0]*expr.shape[1]//nvars, nvars)
        if len(result[i]) == 3:
            result[i] = (expr, c, result[i][2])
        else:
            result[i] = (expr, c)

    if dtype is not None:
        f = lambda x: x.astype(dtype) if isinstance(x, ndarray) else np.array(x.tolist()).astype(dtype)
        for i in range(len(result)):
            if len(result[i]) == 3:
                result[i] = (f(result[i][0]), f(result[i][1]).flatten(), result[i][2])
            else:
                result[i] = (f(result[i][0]), f(result[i][1]).flatten())

    return result


def merge_constraints(constraints: List[Tuple[ndarray, ndarray, str]], dof: int) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    ineq_lhs, ineq_rhs = [], []
    eq_lhs, eq_rhs = [], []
    for lhs, rhs, op in constraints:
        if op in ('>', '>='):
            ineq_lhs.append(lhs)
            ineq_rhs.append(rhs)
        elif op in ('<', '<='):
            ineq_lhs.append(-lhs)
            ineq_rhs.append(-rhs)
        elif op in ('==', '='):
            eq_lhs.append(lhs)
            eq_rhs.append(rhs)
        else:
            raise ValueError(f"Unknown operator {op}.")

    if len(ineq_lhs):
        ineq_lhs = np.vstack(ineq_lhs)
        ineq_rhs = np.concatenate(ineq_rhs)
    else:
        ineq_lhs = np.zeros((0, dof))
        ineq_rhs = np.zeros((0,))

    if len(eq_lhs):
        eq_lhs = np.vstack(eq_lhs)
        eq_rhs = np.concatenate(eq_rhs)
    else:
        eq_lhs = np.zeros((0, dof))
        eq_rhs = np.zeros((0,))
    return ineq_lhs, ineq_rhs, eq_lhs, eq_rhs


class IteratorAlignmentError(Exception): ...

def align_iters(
        iters: List[Union[Any, List[Any]]],
        default_types: List[Union[List[Any], Callable[[Any], bool]]],
        raise_exception: bool = False
    ) -> List[List[Any]]:
    """
    Align the iterators with the default types.

    Parameters
    ----------
    iters : List[Union[Any, List[Any]]]
        The iterators to be aligned.
    default_types : List[Union[List[Any], Callable[[Any], bool]]]
        The default types for each iterator. For each element, it can be a list of types or a function.
        If it is a function, it should return True if the element is of the type.
    raise_exception : bool
        If True, raise exception when it cannot align the iterators.
        If False, use the minimum length of the iterators.
    """
    if len(iters) == 0:
        return None
    check_tp = lambda i, tp: (callable(tp) and not isinstance(tp, type) and tp(i)) or isinstance(i, tp)
    aligned_iters = []
    for i, tp in zip(iters, default_types):
        if i is None:
            aligned_iters.append([])
            continue
        if isinstance(i, list):
            if len(i) == 0 and not check_tp(i, tp):
                # unrecognized type
                # return [[] for _ in range(len(iters))]
                aligned_iters.append([])
                continue
            elif len(i) and check_tp(i[0], tp):
                # every element of the list is of the expected type
                aligned_iters.append(i)
                continue
        if check_tp(i, tp):
            # single element is of the expected type
            aligned_iters.append([i])
        else:
            # unrecognized type
            aligned_iters.append([])
    lengths = [len(i) for i in aligned_iters]
    if raise_exception and len(set(lengths)) > 1:
        raise IteratorAlignmentError("Iterator lengths incompatible", lengths)
    min_len = min(lengths)
    return [i[:min_len] for i in aligned_iters]
