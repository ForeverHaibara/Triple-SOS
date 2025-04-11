from contextlib import contextmanager
from math import sqrt
from typing import Union, Optional, Tuple, List, Dict, Callable, Generator, Any

from numpy import zeros as np_zeros
from numpy import ndarray
from sympy import Matrix, MatrixBase, Expr, Rational, Symbol, re, eye, collect
from sympy.core.relational import GreaterThan, StrictGreaterThan, LessThan, StrictLessThan, Equality, Relational
from sympy.core.singleton import S as singleton

from .arithmetic import vec2mat

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
    GreaterThan: (1, '__ge__'),
    StrictGreaterThan: (1, '__ge__'),
    LessThan: (-1, '__ge__'),
    StrictLessThan: (-1, '__ge__'),
    Equality: (1, '__eq__')
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
        exprs: List[Union[Callable, Expr, Relational, Union[Tuple[Matrix, Rational], Tuple[Matrix, Rational, str]]]]
    ) -> List[Union[Tuple[Matrix, Rational], Tuple[Matrix, Rational, str]]]:
    """
    Convert expressions to arrays with respect to the free symbols.

    Parameters
    ----------
    locals : Dict[str, Any]
        The local variables.
    symbols : List[Symbol]
        The free symbols.
    exprs : List[Union[Callable, Expr, Relational, Matrix]]
        For each expression, it can be a Callable, Expr, Relational, or matrix.
        If it is a Callable, it should be a function that calls on the locals and returns Expr/Relational/Matrix.
        If it is a Expr, it should be with respect to the free symbols.
        If it is a Relational, it should be with respect to the free symbols.

    Returns
    ----------
    Matrix, Rational, [, operator] : Union[Tuple[Matrix, Rational], Tuple[Matrix, Rational, str]]
        The coefficient vector with respect to the free symbols and the Rational of RHS (constant).
        If it is a Relational, it returns the operator also.
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
            else:
                expr, c = expr
        if isinstance(expr, Relational):
            sign, op = _RELATIONAL_TO_OPERATOR[expr.__class__]
            expr = expr.lhs - expr.rhs if sign == 1 else expr.rhs - expr.lhs
            c = -c if sign == -1 else c
        if isinstance(expr, (Expr, int, float)):
            vec_list.append(expr)
            op_list.append(op)
            index_list.append(i)
        else:
            if op is not None:
                result[i] = (expr, c, op)
            else:
                result[i] = (expr, c)

    const, A, _ = decompose_matrix(Matrix(vec_list), symbols)

    for j in range(len(index_list)):
        i = index_list[j]
        if op_list[j] is not None:
            result[i] = (A[j,:], -const[j], op_list[j])
        else:
            result[i] = (A[j,:], -const[j])
    return result


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
