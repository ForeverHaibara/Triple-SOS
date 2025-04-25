from typing import Union, Optional, Any, Tuple, List, Dict

import numpy as np
from sympy import Expr, Symbol, Rational, MatrixBase
from sympy.core.relational import Relational
from sympy import MutableDenseMatrix as Matrix

from .abstract import Decomp
from .arithmetic import solve_undetermined_linear, rep_matrix_from_numpy, sqrtsize_of_mat
from .backends import SDPError, solve_numerical_dual_sdp
from .rationalize import RationalizeWithMask, RationalizeSimultaneously
from .transforms import TransformableDual

from .utils import S_from_y, decompose_matrix


def _get_unique_symbols(used_symbols, dof: int, xname: str = 'y'):
    """
    Generate `dof` unique symbols that differ from the existing symbols.

    Parameters
    ----------
    used_symbols: List[Symbol]
        The existing symbols.
    dof : int
        The number of symbols to generate.
    xname : str
        The prefix of the symbol name.
    """
    used_symbols = set([_.name for _ in used_symbols])
    xname = xname + '_{'
    n = len(xname)
    used_symbols = set(s[n:-1] for s in used_symbols if s.startswith(xname) and s[-1] == '}')
    digits = '0123456789'
    used_digits = list(map(int, filter(lambda x: all(d in digits for d in x), used_symbols)))
    max_digit = max(used_digits, default=-1) + 1
    return [Symbol(xname + str(i) + '}') for i in range(max_digit, max_digit + dof)]


class SDPProblem(TransformableDual):
    """
    Class to solve dual SDP problems, which is in the form of

        S_i = C_i + y_1 * A_i1 + y_2 * A_i2 + ... + y_n * A_in >> 0.
    
    where C, A_ij ... are known symmetric matrices, and y_i are free variables.

    It can be rewritten in the form of

        vec(S_i) = x_i + space_i @ y >> 0.

    And together they are vec([S_1, S_2, ...]) = [x_1, x_2, ...] + [space_1, space_2, ...] @ y
    where x_i and space_i are known. The problem is to find a feasible solution y such that S_i >> 0
    and minimize a linear objective function c^Ty.

    ## Solving Dual SDP

    Here is a simple tutorial to use this class to solve SDPs. Consider the example from
    https://github.com/vsdp/SDPLIB/tree/master:

        min   10 * x1 + 20 * x2

        s.t.  X = F1 * x1 + F2 * x2 - F0
              X >= 0

    where:

        F0 = Matrix(4,4,[1,0,0,0,0,2,0,0,0,0,3,0,0,0,0,4])
        F1 = Matrix(4,4,[1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])
        F2 = Matrix(4,4,[0,0,0,0,0,1,0,0,0,0,5,2,0,0,2,6])

    To solve this problem, we view X >> 0 as (A @ x + b) >> 0, where x = [x1,x2]
    and A is a matrix of shape 16x2 and b = -vec(F0). We initialize the problem as:

        >>> from sympy import Matrix
        >>> F0 = Matrix(4,4,[1,0,0,0,0,2,0,0,0,0,3,0,0,0,0,4])
        >>> F1 = Matrix(4,4,[1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])
        >>> F2 = Matrix(4,4,[0,0,0,0,0,1,0,0,0,0,5,2,0,0,2,6])
        >>> A = Matrix.hstack(F1.reshape(16, 1), F2.reshape(16, 1))
        >>> b = -F0.reshape(16, 1)
        >>> sdp = SDPProblem({'X': (b, A)})
        >>> sdp
        <SDPProblem dof=2 size={'X': 4}>

    We can take a look at the symbolic matrix by calling the `S_from_y` method:

        >>> sdp.S_from_y()
        {'X': Matrix([
        [y_{0} - 1,                 0,           0,           0],
        [        0, y_{0} + y_{1} - 2,           0,           0],
        [        0,                 0, 5*y_{1} - 3,     2*y_{1}],
        [        0,                 0,     2*y_{1}, 6*y_{1} - 4]])}

    Then we can solve the problem by calling the `solve_obj` method
    by passing in the objective vector, and it is expected to return the solution vector.

        >>> sdp.solve_obj([10, 20]) # doctest: +SKIP
        Matrix([
        [0.999999990541857],
        [0.999999992256747]])

    After the solution is found, the solution can also be accessed by `sdp.y`, `sdp.S`
    and `sdp.decompositions`. As the solving process is numerical, the matrix could
    be slightly nonpositive semidefinite up to a small numerical error.

        >>> sdp.y # doctest: +SKIP
        Matrix([
        [0.999999990541857],
        [0.999999992256747]])
        >>> sdp.S # doctest: +SKIP
        {'X': Matrix([
        [-9.45814337960371e-9,                  0.0,              0.0,              0.0],
        [                 0.0, -1.72013967514317e-8,              0.0,              0.0],
        [                 0.0,                  0.0, 1.99999996128373, 1.99999998451349],
        [                 0.0,                  0.0, 1.99999998451349, 1.99999995354048]])}
        >>> sdp.decompositions # doctest: +SKIP
        {'X': (Matrix([
        [0.0, 0.0,  0.707106780502134, -0.707106781870961],
        [0.0, 1.0,                0.0,                0.0],
        [1.0, 0.0,                0.0,                0.0],
        [0.0, 0.0, -0.707106781870961, -0.707106780502134]]), Matrix([
        [            0.0],
        [            0.0],
        [            0.0],
        [3.9999999419256]]))}

    ### Initialization by from_matrix

    Apart from initializing by the tuple of "x" and "space", there is also
    a more flexible approach to initialize the problem by a dictionary of SymPy matrices.
    This is done by calling the `.from_matrix` classmethod.

        >>> from sympy.abc import x, y
        >>> X = x*F1 + y*F2 - F0 # we wish to solve the SDP problem X >> 0
        >>> sdp2 = SDPProblem.from_matrix(X)
        >>> sdp2.S_from_y() # visualize the symbolic matrix
        {0: Matrix([
        [x - 1,         0,       0,       0],
        [    0, x + y - 2,       0,       0],
        [    0,         0, 5*y - 3,     2*y],
        [    0,         0,     2*y, 6*y - 4]])}
        >>> sdp2.solve_obj([10, 20]) # doctest: +SKIP
        Matrix([
        [0.999999990541857],
        [0.999999992256747]])

    It also supports linear objectives represented in the variables, such as:

        >>> sdp2.solve_obj(10*x + 20*y) # doctest: +SKIP
        Matrix([
        [0.999999990541857],
        [0.999999992256747]])

    The solution can also be accessed by `sdp.as_params()`, which returns a dictionary of parameters.

        >>> sdp2.as_params() # doctest: +SKIP
        {x: 0.999999990541857, y: 0.999999992256747}

    ### Initialization by multiple matrices

    Since the target matrix X is block-diagonal, X >> 0 is equivalent to three matrices S_1, S_2, S_3 >> 0.
    We can initialize the problem by passing in a dictionary of matrices, which means each of the matrices
    should be positive semidefinite.

        >>> S1 = Matrix([[x - 1]])
        >>> S2 = Matrix([[x + y - 2]])
        >>> S3 = Matrix([[5*y - 3, 2*y], [2*y, 6*y - 4]])
        >>> sdp3 = SDPProblem.from_matrix({'S1': S1, 'S2': S2, 'S3': S3})
        >>> sdp3.S_from_y()
        {'S1': Matrix([[x - 1]]), 'S2': Matrix([[x + y - 2]]), 'S3': Matrix([
        [5*y - 3,     2*y],
        [    2*y, 6*y - 4]])}
        >>> sdp3.solve_obj(10*x + 20*y) # doctest: +SKIP
        Matrix([
        [0.99999998837194],
        [0.99999999348851]])

    ### Solving with constraints

    Because S1 and S2 are one-dimensional, the constraint is equivalent to x-1>=0, x+y-2>=0 and S3>>0.
    It is also supported to initialize the SDP by S3 only, and pass in the linear constraints to
    `solve_obj`. However, since S3 does not contain the symbol x, the symbols must be explicitly passed
    to the `gens` argument when initialization.

        >>> sdp4 = SDPProblem.from_matrix({'S3': S3}, gens=(x,y))
        >>> sdp4.gens
        [x, y]
        >>> sdp4.solve_obj(10*x + 20*y, constraints=[x>=1, x+y-2>=0]) # doctest: +SKIP
        Matrix([
        [ 0.99999999751827],
        [0.999999998305915]])

    Note that equality constraints are not represented directly by == operator. This is
    because equalities like `x == 1` would be a boolean False. To correctly enforce equality
    constraints, the "== 0" should be omitted, or use the sympy.Eq class. Here is an example
    to solve the SDP given constraints x == 1 and x + y - 2 == 0 using two different
    representations of equality constraints:

        >>> from sympy import Eq
        >>> sdp4.solve_obj(10*x + 20*y, constraints=[Eq(x, 1), x + y - 2]) # doctest: +SKIP
        Matrix([
        [1.0],
        [1.0]])


    ## Solving SOS programming

    We next illustrate an example of SOS programming from https://hankyang.seas.harvard.edu/Semidefinite/SOS.html,
    Example 4.13:

        min   -a - b

        s.t.  x^4+a*x+(2+b) is SOS
              (a-b+1)*x^2 + b*x + 1 is SOS

    To define the problem, we assume the two SOS polys to be v1'*Q1*v1, v2'*Q2*v2,
    where v1 = [1,x,x^2], v2 = [1,x]. We need to create two symmetric matrices
    using SymPy matrices and symbols.

        >>> from sympy import Matrix, Symbol
        >>> from sympy.abc import x, a, b
        >>> v1, v2 = Matrix([1,x,x**2]), Matrix([1,x])
        >>> Q1 = Matrix([[Symbol(f'Q1_{min(i,j)}_{max(i,j)}') for j in range(3)] for i in range(3)])
        >>> Q2 = Matrix([[Symbol(f'Q2_{min(i,j)}_{max(i,j)}') for j in range(2)] for i in range(2)])
        >>> Q1 # check the matrix Q1, which must be symmetric
        Matrix([
        [Q1_0_0, Q1_0_1, Q1_0_2],
        [Q1_0_1, Q1_1_1, Q1_1_2],
        [Q1_0_2, Q1_1_2, Q1_2_2]])

    We need to compute the relations of the variables by comparing the coefficients:

        >>> p1 = ((v1.T @ Q1 @ v1)[0,0] - (x**4 + a*x + (2+b))).as_poly(x)
        >>> p2 = ((v2.T @ Q2 @ v2)[0,0] - ((a-b+1)*x**2 + b*x + 1)).as_poly(x)
        >>> eq = p1.coeffs() + p2.coeffs()
        >>> eq # all following values should be zeros
        [Q1_2_2 - 1, 2*Q1_1_2, 2*Q1_0_2 + Q1_1_1, 2*Q1_0_1 - a, Q1_0_0 - b - 2, Q2_1_1 - a + b - 1, 2*Q2_0_1 - b, Q2_0_0 - 1]

    We can then initialize the SDPProblem object and solve it via:

        >>> symbols = list(Q1.free_symbols) + list(Q2.free_symbols) + [a,b] # collect all symbols
        >>> sdp = SDPProblem.from_matrix({'Q1': Q1, 'Q2': Q2}, gens=symbols)
        >>> sol = sdp.solve_obj(-a-b, constraints=eq)

    After solving, calling the `.as_params()` method will return a dictionary of parameter values:

        >>> print('(a, b) =', (sdp.as_params()[a], sdp.as_params()[b])) # doctest: +SKIP
        (a, b) = (6.61890359483024, 3.87159385296789)

    It is also possible to access the values of the matrices by:

        >>> sdp.S # doctest: +SKIP
        {'Q1': Matrix([
        [ 5.87159385296789, 3.30945179741512, -1.39899944005445],
        [ 3.30945179741512,  2.7979988801089,               0.0],
        [-1.39899944005445,              0.0,               1.0]]), 'Q2': Matrix([
        [             1.0, 1.93579692648395],
        [1.93579692648395, 3.74730974186235]])}

    """
    is_dual = True
    is_primal = False

    _x0_and_space: Dict[Any, Tuple[Matrix, Matrix]] = None

    def __init__(
        self,
        x0_and_space: Union[Dict[str, Tuple[Matrix, Matrix]], List[Tuple[Matrix, Matrix]]],
        gens = None
    ):
        super().__init__()

        self._init_space(x0_and_space.copy(), '_x0_and_space')
        x0_and_space = self._x0_and_space

        # check that each space has the same number of columns
        _free_symbols_in_domain = set()
        dof = 0
        for x0, space in x0_and_space.values():
            if space.shape[1] != 0:
                dof = space.shape[1]
                break
        for key, (x0, space) in x0_and_space.items():
            # if space.shape[0] * space.shape[1] == 0:
            #     x0_and_space[key] = (x0, Matrix.zeros(x0.shape[0], dof))
            if x0.shape[0] != space.shape[0]:
                raise ValueError(f"The number of rows of x0 and space should be the same, but got"
                                 f" {x0.shape[0]} and {space.shape[0]} for key {key}.")

            if space.shape[1] != dof:
                raise ValueError(f"The number of columns of spaces should be the same, but got"
                                 f" {space.shape[1]} and {dof} for key {key}.")

            if hasattr(x0, 'free_symbols'):
                _free_symbols_in_domain.update(x0.free_symbols)
            if hasattr(space, 'free_symbols'):
                _free_symbols_in_domain.update(space.free_symbols)

        _free_symbols_in_domain = sorted(list(_free_symbols_in_domain), key=lambda x: x.name)
        self._free_symbols_in_domain = _free_symbols_in_domain

        if gens is not None:
            if len(gens)!= dof:
                raise ValueError(f"Length of free_symbols and space should be the same, but got"
                                 f" {len(gens)} and {dof}.")
            self._gens = list(gens)
        else:
            self._gens = _get_unique_symbols(_free_symbols_in_domain, dof, xname='y')



    def keys(self, filter_none: bool = False) -> List[Any]:
        """
        Get the keys of the SDP problem.

        Parameters
        ----------
        filter_none : bool, optional
            If True, filter out the keys with size 0, by default False.

        Returns
        ----------
        keys : List[Any]
            The keys of the SDP problem.

        Examples
        ----------
        >>> from sympy import Matrix
        >>> from sympy.abc import a, b
        >>> sdp = SDPProblem.from_matrix(Matrix([[a+1+b, 1+b], [1+b, 2-a]]))
        >>> sdp.keys()
        [0]
        >>> sdp = SDPProblem.from_matrix({'S1': Matrix([[a,b],[b,a]]), 'S2': Matrix([[b,a],[a,b]]), 'S3': Matrix([])})
        >>> sdp.keys()
        ['S1', 'S2', 'S3']
        >>> sdp.keys(filter_none=True)
        ['S1', 'S2']
        """
        space = self._x0_and_space
        keys = list(space.keys())
        if filter_none:
            _size = lambda key: space[key][1].shape[1] * space[key][1].shape[0]
            keys = [key for key in keys if _size(key) != 0]
        return keys

    @property
    def size(self) -> Dict[Any, int]:
        """
        The dimension of each symmetric matrix in the SDP problem.

        Returns
        ----------
        size : Dict[Any, int]
            The size of the SDP problem.

        Examples
        ----------
        >>> from sympy import Matrix
        >>> from sympy.abc import a, b
        >>> sdp = SDPProblem.from_matrix({'A': Matrix([[a+1+b, 1+b], [1+b, 2-a]]), 'B': Matrix([[b]])})
        >>> sdp.size
        {'A': 2, 'B': 1}
        """
        return super().size

    @property
    def free_symbols(self) -> List[Symbol]:
        """
        The free symbols of the SDP problem, including parameters.

        Returns
        ----------
        free_symbols : List[Symbol]
            The free symbols of the SDP problem.

        Examples
        ----------
        >>> from sympy import Matrix
        >>> from sympy.abc import a, b, t
        >>> sdp = SDPProblem.from_matrix(Matrix([[t**2*a, 1+b], [1+b, 2+t-a]]), gens=[a,b])
        >>> sdp.free_symbols
        [a, b, t]
        >>> sdp.gens
        [a, b]

        See also
        ----------
        gens : The variables of the SDP problem, excluding parameters.
        """
        return self._gens + self._free_symbols_in_domain

    @property
    def gens(self) -> List[Symbol]:
        """
        The variables of the SDP problem.

        Returns
        ----------
        gens : List[Symbol]
            The variables of the SDP problem.
    
        Examples
        ----------
        >>> from sympy import Matrix
        >>> from sympy.abc import a, b
        >>> sdp = SDPProblem.from_matrix(Matrix([[a+1+b, 1+b], [1+b, 2-a]]))
        >>> sdp.gens
        [a, b]

        See also
        ----------
        free_symbols : The free symbols of the SDP problem, including parameters.
        """
        return self._gens

    @property
    def dof(self) -> int:
        """
        The degree of freedom of the SDP problem, equals to the number of generators.

        Returns
        ----------
        dof : int
            The degree of freedom of the SDP problem.

        Examples
        ----------
        >>> from sympy import Matrix
        >>> from sympy.abc import a, b
        >>> sdp = SDPProblem.from_matrix(Matrix([[a+1+b, 1+b], [1+b, 2-a]]))
        >>> sdp.gens
        [a, b]
        >>> sdp.dof
        2
        """
        return len(self.gens)

    @classmethod
    def from_full_x0_and_space(cls, x0: Matrix, space: Matrix, splits: Union[Dict[Any, int], List[int]],
        gens: Optional[List[Symbol]]=None, constrain_symmetry: bool = False
    ) -> 'SDPProblem':
        """
        Initialize a SDP problem with the compressed x0 and space matrix.

        Parameters
        -----------
        x0 : Matrix
            The concatenated x0 of all matrices Si.
        space : Matrix
            The concatednated (vstack) spaces of all matrices Si.
        splits : Union[Dict[Any, int], List[int]]
            The dimension of each matrix.
        gens : Optional[List[Symbol]]
            The variable names.
        constrain_symmetry : bool
            If each column of the split space is not the vector form of
            a symmetric matrix, the flag should be set to True to impose symmetry.

        Returns
        ---------
        SDPProblem :
            The created SDPProblem instance.

        Examples
        ----------
        Consider a SDP problem with 3 positive semidefinite matrices:

            vec(S1) = [[1,0]] @ [x,y] + [-1]
            vec(S2) = [[1,1]] @ [x,y] + [-2]
            vec(S3) = [[0,5],[0,-2],[0,-2],[0,6]] @ [x,y] + [-3,0,0,-4]

        Together they can be concatenated into a single x0 and space:

            >>> from sympy import Matrix
            >>> from sympy.abc import x, y
            >>> x0 = Matrix([-1,-2,-3,0,0,-4])
            >>> space = Matrix([[1,0], [1,1], [0,5],[0,-2],[0,-2],[0,6]])

        To initialize the SDP for S1,S2,S3 >> 0, use `splits = [1,1,2]` to indicate
        the dimension of each matrix. A dictionary is also accepted to contain the names
        of each matrix.

            >>> sdp = SDPProblem.from_full_x0_and_space(x0, space,
            ...          splits={'S1': 1, 'S2': 1, 'S3': 2}, gens=(x,y))
            >>> sdp.S_from_y()
            {'S1': Matrix([[x - 1]]), 'S2': Matrix([[x + y - 2]]), 'S3': Matrix([
            [5*y - 3,    -2*y],
            [   -2*y, 6*y - 4]])}
        """
        keys = None
        if isinstance(splits, dict):
            keys = list(splits.keys())
            splits = list(splits.values())

        x0_and_space = []
        start = 0

        size = sum(n**2 for n in splits)
        if x0.shape[0] != size or space.shape[0] != size:
            raise ValueError(f"The size of x0 and space does not match the splits, expected"
                             f" {size}, but got {x0.shape} and {space.shape}.")

        for n in splits:
            x0_ = x0[start:start+n**2,:]
            space_ = space[start:start+n**2,:]
            x0_and_space.append((x0_, space_))
            start += n**2

        if keys is not None:
            x0_and_space = dict(zip(keys, x0_and_space))
        sdp = SDPProblem(x0_and_space, gens=gens)

        if constrain_symmetry:
            sdp = sdp.constrain_symmetry()
            sdp._transforms.clear()
        return sdp

    @classmethod
    def from_equations(cls, eq: Matrix, rhs: Matrix, splits: Optional[Union[Dict[str, int], List[int]]]=None
    ) -> 'SDPProblem':
        """
        Assume the SDP problem can be rewritten in the form of

            eq * [vec(S1); vec(S2); ...] = rhs
        
        where Si.shape[0] = splits[i].
        The function formulates the SDP problem from the given equations.
        This is also the primal form of the SDP problem.

        Parameters
        ----------
        eq : Matrix
            The matrix eq.
        rhs : Matrix
            The matrix rhs.
        splits : Optional[Union[Dict[str, int], List[int]]]
            The splits of the size of each symmetric matrix.
            If None, it assumes there is only one matrix.

        Returns
        ---------
        sdp : SDPProblem
            The created SDP problem instance.

        Examples
        ---------
        Consider the example from https://clarabel.org/stable/examples/example_sdp/
        where the SDP problem is given by

            min trace(X)

            s.t. Avec(X) = b, X >> 0

        where:

            A = Matrix([[1,2,4,2,3,5,4,5,6]])
            b = Matrix([1])

        To initialize the problem, just use the `from_equations` method.
 
            >>> from sympy import Matrix
            >>> A = Matrix([[1,2,4,2,3,5,4,5,6]])
            >>> b = Matrix([1])
            >>> sdp = SDPProblem.from_equations(A, b); sdp
            <SDPProblem dof=5 size={0: 3}>
            >>> sdp.S_from_y() # doctest: +SKIP
            {0: Matrix([
            [-4*y_{0} - 3*y_{1} - 8*y_{2} - 10*y_{3} - 6*y_{4} + 1, y_{0}, y_{2}],
            [                                                y_{0}, y_{1}, y_{3}],
            [                                                y_{2}, y_{3}, y_{4}]])}
            >>> sol = sdp.solve_obj(sdp.S_from_y()[0].trace())
            >>> sdp.S # doctest: +SKIP
            {0: Matrix([
            [0.0128900409812068, 0.0177186109536194, 0.0251905619896497],
            [0.0177186109536194,  0.024355954451504, 0.0346268797866644],
            [0.0251905619896497, 0.0346268797866644, 0.0492290596776602]])}
            >>> sdp.S[0].trace() # doctest: +SKIP
            0.0864750551103711
        """
        if splits is None:
            splits = [sqrtsize_of_mat(eq.shape[1])]
        x0, space = solve_undetermined_linear(eq, rhs)
        return cls.from_full_x0_and_space(x0, space, splits, constrain_symmetry=True)

    @classmethod
    def from_matrix(cls, S: Union[Matrix, List[Matrix], Dict[str, Matrix]], gens: Optional[List[Symbol]]=None,
    ) -> 'SDPProblem':
        """
        Construct a `SDPProblem` from symbolic symmetric matrices.
        The problem is to solve a parameter set such that all given
        symmetric matrices are positive semidefinite. The result can
        be obtained by `SDPProblem.as_params()`.

        Parameters
        ----------
        S : Union[Matrix, List[Matrix], Dict[str, Matrix]]
            The symmetric matrices that SDP requires to be positive semidefinite.
            Each entry of the matrix should be linear in the free symbols (gens).
        gens : Optional[List[Symbol]], optional
            The free symbols of the matrices, by default None.
            If None, it will be inferred from the matrices and sorted by names.

        Returns
        ----------
        sdp : SDPProblem
            The created SDP problem instance.


        """
        keys = None
        if isinstance(S, dict):
            keys = list(S.keys())
            S = list(S.values())

        if isinstance(S, Matrix):
            S = [S]

        if gens is None:
            gens = set()
            for s in S:
                if not isinstance(s, MatrixBase):
                    raise ValueError("S must be a list of Matrix or a dict of Matrix.")
                gens |= set(s.free_symbols)

        gens = list(gens)
        gens = sorted(gens, key = lambda x: x.name)

        x0_and_space = []
        for s in S:
            x0, space, _ = decompose_matrix(s, gens)
            x0_and_space.append((x0, space))

        if keys is not None:
            x0_and_space = dict(zip(keys, x0_and_space))

        return SDPProblem(x0_and_space, gens=gens)

    def S_from_y(self, y: Optional[Union[Matrix, np.ndarray, Dict]] = None) -> Dict[str, Matrix]:
        m = self.dof
        if y is None:
            y = Matrix(self.gens).reshape(m, 1)
        elif isinstance(y, MatrixBase):
            if m == 0 and y.shape[0] * y.shape[1] == 0:
                y = Matrix.zeros(0, 1)
            elif y.shape == (1, m):
                y = y.T
            elif y.shape != (m, 1):
                raise ValueError(f"Vector y must be a matrix of shape ({m}, 1), but got {y.shape}.")
        elif isinstance(y, np.ndarray):
            if y.size != m:
                raise ValueError(f"Vector y must be an array of shape ({m},) or ({m}, 1), but got {y.shape}.")
            y = rep_matrix_from_numpy(y)
        elif isinstance(y, dict):
            y = Matrix([y.get(v, v) for v in self.gens]).reshape(m, 1)

        return S_from_y(y, self._x0_and_space)

    def as_params(self) -> Dict[Symbol, Expr]:
        """
        Return the dictionary of free symbols and their values after solving the SDP.

        Returns
        ---------
        params : Dict[Symbol, Expr]
            The dictionary of variable values.

        Examples
        ---------
        >>> from sympy import Matrix
        >>> from sympy.abc import a, b
        >>> sdp = SDPProblem.from_matrix(Matrix([[a, 1], [1, b]]))
        >>> sdp.solve_obj(4*a + b) # doctest: +SKIP
        Matrix([
        [0.499986210665879],
        [ 2.00005510893102]])
        >>> sdp.as_params() # doctest: +SKIP
        {a: 0.499986210665879, b: 2.00005510893102}
        """
        return super().as_params()

    def _solve_numerical_sdp(self,
        objective: np.ndarray,
        constraints: List[Tuple[np.ndarray, np.ndarray, str]] = [],
        solver: Optional[str] = None,
        return_result: bool = False,
        kwargs: Dict[str, Any] = {}
    ) -> Optional[np.ndarray]:
        return solve_numerical_dual_sdp(
            self._x0_and_space, objective=objective, constraints=constraints,
            solver=solver, return_result=return_result, **kwargs
        )

    def solve_obj(self,
        objective: Union[Expr, Matrix, List],
        constraints: List[Union[Relational, Expr, Tuple[Matrix, Matrix, str]]] = [],
        solver: Optional[str] = None,
        solve_child: bool = True,
        propagate_to_parent: bool = True,
        verbose: bool = False,
        kwargs: Dict[Any, Any] = {}
    ) -> Optional[Matrix]:
        """
        Solve the SDP problem numerically with the given objective.

        Parameters
        ----------
        objective : Expr, Matrix, or list
            Objective to minimize. If it is a sympy expression, it must be
            affine with respect to the variables. If it is a matrix (a column vector) or a list,
            the objective is the inner product of the vector and the variable vector.

        constraints : List[Union[Relational, Expr, Tuple[Matrix, Matrix, str]]]
            Additional affine constraints over variables. Each element of the list
            must be one of the following:
            
            A sympy affine relational expression, e.g., `x > 0` or `Eq(x + y, 1)`.
              Note that equality constraints must use `sympy.Eq` class instead of `==` operator,
              because the latter `x + y == 1` will be evaluated to a boolean value.

            A sympy affine expression, e.g., `x + y - 1`, they are treated as equality constraints.

            A tuple of (lhs, rhs, operator), where lhs is a 2D matrix, rhs is a 1D vector, and
              operator is a string. It is considered as `lhs @ variables (operator) rhs`.
              The operator can be one of '>', '<' or '='.

        solver : Optional[str]
            Backend solver to the numerical SDP, e.g., 'mosek', 'clarabel', 'cvxopt'.
            Corresponding packages must be installed. If None, the solver will be
            automatically selected. For a full list of supported backends, see `sdp.backends.caller.py`.

        solve_child : bool
            If there is a transformation graph of the SDP, whether to solve the child node and
            then convert the solution back to the parent node. This reduces the degree of freedom.
            Defaults to True.

        propagate_to_parent : bool
            If there is a transformation graph of the SDP, whether to propagate the solution of
            the SDP to its parents. Defaults to True.

        verbose : bool
            Whether to allow the backend SDP solver to print the log. Defaults to False.
            This argument will be suppressed if `kwargs` contains a `verbose` key.

        kwargs : Dict
            Extra kwargs passed to `sdp.backends.solve_numerical_dual_sdp`. Accepted kwargs keys:
            `verbose`, `max_iters`, `tol_gap_abs`, `tol_gap_rel`, `tol_fsb_abs`, `tol_fsb_rel`, `solver_options`,
            etc.

        Examples
        ----------
        Here we illustrate the example from "Semidefinite Optimization and Convex Algebraic Geometry"
        by Blekherman, Parillo and Thomas, Example 2.7. Consider the SDP
        [[x+1, 0, y], [0, 2, -x-1], [y, -x-1, 2]] >> 0, whose feasible set
        is part of the elliptic curve: 3+x-x^3-3*x^2-2*y^2>=0 && x>=-1. We wish
        to maximize x+2*y (i.e., minimize -x-2*y) in the positive semidefinite cone.
        The SDP problem can be then initialized using the `from_matrix` method:

            >>> from sympy import Matrix
            >>> from sympy.abc import x, y, a, b, c
            >>> sdp = SDPProblem.from_matrix(Matrix([[x+1, 0, y], [0, 2, -x-1], [y, -x-1, 2]]))

        We can solve the SDP problem by calling `.solve_obj`:

            >>> sdp.solve_obj(-x-2*y) # doctest: +SKIP
            Matrix([
            [0.453950762287364],
            [ 1.17093779835239]])

        After the SDP is solved, the solution can be obtained by `.y` and `.S`:

            >>> sdp.y # doctest: +SKIP
            Matrix([
            [0.453950762287364],
            [ 1.17093779835239]])
            >>> sdp.S # doctest: +SKIP
            {0: Matrix([
            [1.45395076228736,               0.0,  1.17093779835239],
            [             0.0,               2.0, -1.45395076228736],
            [1.17093779835239, -1.45395076228736,               2.0]])}

        It is also supported to obtain a dictionary of values via `.as_params()`,
        which can be used in `.subs` method of sympy expressions:

            >>> sdp.as_params() # doctest: +SKIP
            {x: 0.453950762287364, y: 1.17093779835239}
            >>> (x+2*y).subs(sdp.as_params()) # doctest: +SKIP
            2.79582635899214


        Apart from using a sympy expression to express the objective, the objective can also
        be a vector (list), standing for the inner product of the vector and the variable vector.
        For example, using [-1,-2] as objective is equivalent to minimizing -x-2*y:

            >>> sdp.solve_obj([-1,-2]) # doctest: +SKIP
            Matrix([
            [0.453950762287364],
            [ 1.17093779835239]])

        ### Solving with constraints

        We can also add additional affine constrains to the SDP problem when
        calling `.solve_obj`. For example, we can add the constraint x+y<0:

            >>> sdp.solve_obj(-x-2*y, constraints=[x+y<0]) # doctest: +SKIP
            Matrix([
            [-0.729181517672415],
            [ 0.729181519072328]])

        Constraints can also be passed a (lhs, rhs, op) tuple, where op is one of
        '>', '<', '=', lhs is a 2D matrix and rhs is a 1D matrix. For example, the constraint
        x+y<0 can be written as (Matrix([[1,1]]), Matrix([0]), '<'). The following code is
        equivalent to the previous example:

            >>> sol = sdp.solve_obj(-x-2*y, constraints=[(Matrix([[1,1]]), Matrix([0]), '<')])
    
        The function will sanitize the input so it is also acceptable to pass in lists or numpy
        arrays instead of sympy matrices.

        Nonlinear objectives or constraints are not supported, e.g.,

            >>> sdp.solve_obj(-x-2*y, constraints=[y<=x**2]) # doctest: +SKIP
            Traceback (most recent call last):
            ...
            NonlinearError: nonlinear term: x**2

        ### Handling exceptions

        If the solver does not find the optimal solution, e.g., when the problem is
        infeasible or unbounded, or the solution is inaccurate given the tolerance,
        the function will raise an error. Below is an example of infeasible SDP
        that Matrix([[a, 2], [2, 1-a]]) >> 0. It is infeasible since a*(1-a) < 4.

            >>> sdp = SDPProblem.from_matrix(Matrix([[a, 2], [2, 1-a]]))
            >>> sdp.solve_obj(a) # doctest: +SKIP
            Traceback (most recent call last):
            ...
            SDPError: SDP solution failed: optimal=False, infeasible=True, inf_or_unb=True

            >>> try:
            ...     sdp.solve_obj(a)
            ... except SDPError as e:
            ...     print((e.infeasible, e.unbounded, e.inf_or_unb))
            (True, False, True)

        Below is an example of unbounded SDP: min -a, s.t. [[a, 1], [1, a]] >> 0.

            >>> sdp = SDPProblem.from_matrix(Matrix([[a, 1], [1, a]]))
            >>> try:
            ...     sdp.solve_obj(-a)
            ... except SDPError as e:
            ...     print((e.infeasible, e.unbounded, e.inf_or_unb))
            (False, True, True)

        More statuses of SDPErrors can be found in the `SDPError` class.

        ### Using kwargs

        The `kwargs` argument can be used to pass extra arguments to the backend SDP solver.
        We have the follwing example to solve an SDP problem with CVXOPT solver and increased
        precision:

            >>> sdp = SDPProblem.from_matrix(Matrix([[a, 1], [1, a]]))
            >>> sdp.solve_obj(a, solver='cvxopt',
            ... kwargs={'verbose':True, 'tol_gap_abs':1e-12, 'tol_gap_rel':1e-12}) # doctest: +SKIP
                 pcost       dcost       gap    pres   dres   k/t
             0:  0.0000e+00 -0.0000e+00  2e+00  2e+00  5e-10  1e+00
             1:  8.2427e-01  8.2427e-01  2e-01  2e-01  5e-11  9e-02
             2:  9.9824e-01  9.9824e-01  2e-03  2e-03  5e-13  1e-03
             3:  9.9998e-01  9.9998e-01  2e-05  2e-05  5e-15  1e-05
             4:  1.0000e+00  1.0000e+00  2e-07  2e-07  1e-16  1e-07
             5:  1.0000e+00  1.0000e+00  2e-09  2e-09  7e-16  1e-09
             6:  1.0000e+00  1.0000e+00  2e-11  2e-11  2e-16  1e-11
             7:  1.0000e+00  1.0000e+00  2e-13  2e-13  4e-15  1e-13
            Optimal solution found.
            Matrix([[0.999999999999824]])
        """
        return super().solve_obj(
            objective, constraints=constraints, solver=solver,
            solve_child=solve_child, propagate_to_parent=propagate_to_parent,
            verbose=verbose, kwargs=kwargs
        )

    def solve(self,
        solver: Optional[str] = None,
        solve_child: bool = True,
        propagate_to_parent: bool = True,
        verbose: bool = False,
        allow_numer: int = 0,
        kwargs: Dict[Any, Any] = {}
    ) -> Optional[Matrix]:
        """
        Solve a feasible SDP problem. If the SDPProblem is rational,
        it tries to find a rational solution. However, the search for
        rational solutions is heuristic and could fail for weakly feasible SDPs.

        Parameters
        ----------
        solver : Optional[str]
            Backend solver to the numerical SDP, e.g.,'mosek', 'clarabel', 'cvxopt'.
            Corresponding packages must be installed. If None, the solver will be
            automatically selected. For a full list of supported backends, see `sdp.backends.caller.py`.

        solve_child : bool
            If there is a transformation graph of the SDP, whether to solve the child node and
            then convert the solution back to the parent node. This reduces the degree of freedom.
            Defaults to True.

        propagate_to_parent : bool
            If there is a transformation graph of the SDP, whether to propagate the solution of
            the SDP to its parents. Defaults to True.

        allow_numer : bool
            Whether to allow inexact, numerical feasible solutions. This is useful when the
            SDP is weakly feasible and no rational solution is found successfully.

        kwargs : Dict
            Extra kwargs passed to `sdp.backends.solve_numerical_dual_sdp`. Accepted kwargs keys:
            `verbose`, `max_iters`, `tol_gap_abs`, `tol_gap_rel`, `tol_fsb_abs`, `tol_fsb_rel`, `solver_options`,
            etc.

        Returns
        ----------
        y : Matrix
            The solution of the SDP problem. If it fails, return None.

        Examples
        ----------
        Here we illustrate an example from "Moment and Polynomial Optimization" by Jiawang Nie,
        Section 3.1 to prove 1+x+x^2+x^3+x^4+x^5+x^6 >= 0 via sum of squares. The polynomial
        can always be represented as [1,x,x^2,x^3]^T @ X @ [1,x,x^2,x^3] where X is defined as:

            >>> from sympy import Matrix, Rational
            >>> from sympy.abc import a, b, c, x
            >>> half = Rational(1,2)
            >>> X = Matrix([[1,half,a,b],[half,1-2*a,half-b,c],[a,half-b,1-2*c,half],[b,c,half,1]])

        We can check that this is correct:

            >>> xx = Matrix([1,x,x**2,x**3])
            >>> (xx.T @ X @ xx).expand()
            Matrix([[x**6 + x**5 + x**4 + x**3 + x**2 + x + 1]])

        Our target is to find a solution of [a,b,c] so that X >> 0 holds. We can create the SDP problem
        as follows:

            >>> sdp = SDPProblem.from_matrix(X)
            >>> sdp.solve() # doctest: +SKIP
            Matrix([
            [  -433/2302],
            [8357/134991],
            [  -433/2302]])

        The `solve` method tries to find a rational solution if the SDP problem is rational
        by rationalization. After solving, the result can be accessed via `.y`, `.S` and `.as_params()`:

            >>> sdp.y # doctest: +SKIP
            Matrix([
            [  -433/2302],
            [8357/134991],
            [  -433/2302]])
            >>> sdp.S # doctest: +SKIP
            {0: Matrix([
            [          1,           1/2,     -433/2302, 8357/134991],
            [        1/2,     1584/1151, 118277/269982,   -433/2302],
            [  -433/2302, 118277/269982,     1584/1151,         1/2],
            [8357/134991,     -433/2302,           1/2,           1]])}
            >>> sdp.as_params() # doctest: +SKIP
            {a: -433/2302, b: 8357/134991, c: -433/2302}

        It is also possible to access the decompositions via `.decompositions`,
        which contains a dictionary of (U, D) tuple such that U.T@diag(D)@U=S.

            >>> sdp.decompositions # doctest: +SKIP
            {0: (Matrix([
            [1, 1/2,           -433/2302,                               8357/134991],
            [0,   1, 330724757/699928335,                       -27228004/139985667],
            [0,   0,                   1, 38499884941424340985/68179658483640649487],
            [0,   0,                   0,                                         1]]), Matrix([
            [                                                                   1],
            [                                                           5185/4604],
            [                         136359316967281298974/125172531956581997985],
            [866868545756278356280599749609294/1430012304144122904319093542412497]]))}

        ### Registering solutions

        The rationalization might fail, or generate very nasty solutions, and we may
        want to manually register a solution. The `register_y` method can be used to register
        a solution. By registration, the feasibility will be verified and the matrices
        and decompositions will be automatically updated.

            >>> sdp.register_y([0,0,0])
            True
            >>> sdp.S
            {0: Matrix([
            [  1, 1/2,   0,   0],
            [1/2,   1, 1/2,   0],
            [  0, 1/2,   1, 1/2],
            [  0,   0, 1/2,   1]])}
            >>> sdp.decompositions
            {0: (Matrix([
            [1, 1/2,   0,   0],
            [0,   1, 2/3,   0],
            [0,   0,   1, 3/4],
            [0,   0,   0,   1]]), Matrix([
            [  1],
            [3/4],
            [2/3],
            [5/8]]))}

        We can then obtain a sum-of-squares proof via the decomposition:

            >>> U, D = sdp.decompositions[0]
            >>> sos = sum(coeff*p**2 for coeff, p in zip(D, U@xx)); sos
            5*x**6/8 + (x/2 + 1)**2 + 3*(2*x**2/3 + x)**2/4 + 2*(3*x**3/4 + x**2)**2/3
            >>> sos.expand()
            x**6 + x**5 + x**4 + x**3 + x**2 + x + 1

        ### Allowing numerical solutions

        Although the SDPProblem tries to find a rational solution if the problem is
        rational, a rational solution might not exist even if the SDP is feasible.
        Consider the example [[a,2],[2,2*a]] >> 0, [[2,a],[a,1]] >> 0, which is a
        two-block SDP problem. The only solution is a = sqrt(2), which is irrational.
        Any number a other than sqrt(2) will make one of the matrices not positive
        semidefinite, so it will fail to find a feasible solution:

            >>> S1 = Matrix([[a,2],[2,2*a]])
            >>> S2 = Matrix([[2,a],[a,1]])
            >>> sdp = SDPProblem.from_matrix({'S1': S1, 'S2': S2})
            >>> sdp.solve() is None
            True

        However, we can allow numerical solutions by setting `allow_numer=True`.
        The `solve` method will then return a numerical solution up to a small
        tolerance:

            >>> sdp.solve(allow_numer=True) # doctest: +SKIP
            Matrix([[1.41421356161294]])
            >>> sdp.S # doctest: +SKIP
            {'S1': Matrix([
            [1.41421356161294,              2.0],
            [             2.0, 2.82842712322588]]), 'S2': Matrix([
            [             2.0, 1.41421356161294],
            [1.41421356161294,              1.0]])}
        """
        original_self = self
        if solve_child:
            self = self.get_last_child()
        if self.dof == 0:
            y = Matrix.zeros(0, 1)
            try:
                self.register_y(y)
                return True
            except ValueError:
                raise SDPError.from_kwargs(infeasible=True)
            return False

        success = False

        x0_and_space = {}
        size = self.size
        for key, (x0, space) in self._x0_and_space.items():
            n = size[key]
            # diagonals = Matrix([-int(i%(n+1)==0) for i in range(n**2)])
            diagonals = -np.eye(n).reshape(n**2, 1)
            x0_and_space[key] = (x0, np.hstack([np.array(space).astype(float), diagonals]))
        objective = np.zeros(self.dof + 1)
        objective[-1] = -1
        constraints = [(objective, 0, '<'), (objective, -5, '>')]
        sol = solve_numerical_dual_sdp(
            x0_and_space, objective=objective, constraints=constraints,
            solver=solver, return_result=True, **kwargs
        )

        if sol.y is not None and (not sol.infeasible):
            y = sol.y[:-1] # discard the eigenvalue relaxation
            self._ys.append(y)
            solution = self.rationalize(y, verbose=verbose,
                rationalizers=[RationalizeWithMask(), RationalizeSimultaneously([1,1260,1260**3])])
            if solution is not None:
                self.y = solution[0]
                self.S = dict((key, S[0]) for key, S in solution[1].items())
                self.decompositions = dict((key, S[1:]) for key, S in solution[1].items())
                success = True
            elif allow_numer:
                self.register_y(y, perturb=True, propagate_to_parent=propagate_to_parent)
                success = True

        if propagate_to_parent:
            self.propagate_to_parent(recursive=True)

        return original_self.y if success else None