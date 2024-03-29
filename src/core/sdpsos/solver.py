from contextlib import contextmanager, nullcontext, AbstractContextManager
from typing import Union, Optional, Any, Tuple, List, Dict, Callable, Generator

import numpy as np
import sympy as sp

from .rationalize import rationalize, rationalize_and_decompose
from .ipm import (
    SDPConvergenceError, SDPNumericalError, SDPInfeasibleError, SDPRationalizeError
)
from .utils import (
    solve_undetermined_linear, S_from_y, Mat2Vec, congruence_with_perturbation,
    is_empty_matrix
)


Decomp = Dict[str, Tuple[sp.Matrix, sp.Matrix, List[sp.Rational]]]
PicosExpression = Any

_RELATIONAL_TO_FUNC = {
    sp.GreaterThan: '__ge__',
    sp.StrictGreaterThan: '__ge__',
    sp.LessThan: '__le__',
    sp.StrictLessThan: '__le__',
    sp.Equality: '__eq__'
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


def _decompose_matrix(
        M: sp.Matrix,
        variables: Optional[List[sp.Symbol]] = None
    ) -> Tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    """
    Decomposes a symbolic matrix into the form vec(M) = x + A @ v
    where x is a constant vector, A is a constant matrix, and v is a vector of variables.

    See also in `sympy.solvers.solveset.linear_eq_to_matrix`.

    Parameters
    ----------
    M : sp.Matrix
        The matrix to be decomposed.
    variables : List[sp.Symbol]
        The variables to be used in the decomposition. If None, it uses M.free_symbols.

    Returns
    ----------
    x : sp.Matrix
        The constant vector.
    A : sp.Matrix
        The constant matrix.
    v : sp.Matrix
        The vector of variables.
    """
    rows, cols = M.shape
    variables = list(M.free_symbols) if variables is None else variables
    variable_index = {var: idx for idx, var in enumerate(variables)}

    v = sp.Matrix(variables)
    x = sp.zeros(rows * cols, 1)
    A = sp.zeros(rows * cols, len(variables))

    for i in range(rows):
        for j in range(cols):
            expr = M[i, j]
            terms = sp.collect(expr, variables, evaluate=False)

            constant_term = terms.pop(sp.S.One, 0)  # Extract and remove constant term for x
            x[i * cols + j] = constant_term

            for term, coeff in terms.items():
                A[i * cols + j, variable_index[term]] = coeff  # Extract coefficients for A

    return x, A, v



class SDPTransformation():
    """
    Class that represents transformation between SDPProblems.
    """
    def __init__(self, parent_node, *args, **kwargs):
        self.parent_node: SDPProblem = parent_node
        self.child_node: SDPProblem = self._init_child_node(*args, **kwargs)
        self.parent_node._transforms.append(self)
        self.child_node._transforms.append(self)
    def _init_child_node(self, *args, **kwargs) -> 'SDPProblem':
        return
    def is_parent(self, sdp):
        return sdp is self.parent_node
    def is_child(self, sdp):
        return sdp is self.child_node
    def propagate_to_parent(self, recursive: bool = True):
        raise NotImplementedError
    def propagate_to_child(self, recursive: bool = True):
        raise NotImplementedError

class SDPIdentityTransform(SDPTransformation):
    """
    Identity transformation. It is used as an empty transformation.
    It can neither be found by SDPProblem.parents nor by SDPProblem.children.
    So its methods will not be called.

    This is class is used when the transformation is not needed,
    created by __new__ from other classes. And it should not be used directly.
    """
    __slots__ = ('parent_node', 'child_node')
    def __new__(cls, parent_node):
        obj = object.__new__(cls)
        obj.parent_node = parent_node
        obj.child_node = parent_node
        return obj
    def __init__(self, parent_node, *args, **kwargs):
        self.parent_node = parent_node
        self.child_node = parent_node        
    def is_parent(self, sdp):
        return False
    def is_child(self, sdp):
        return False
    def propagate_to_parent(self, recursive: bool = True):
        return
    def propagate_to_child(self, recursive: bool = True):
        return


class SDPMatrixTransform(SDPTransformation):
    """
    Assume the original problem to be S1 >= 0, ... Sn >= 0.
    We assume that Si = Ui * Mi * Ui' given matrices U1, ... Un.
    An equivalence form is Si * Vi = 0 given matrices Vi, ... Vn.
    Here we have orthogonal relation Ui' * Vi = 0.

    This constrains the columnspace / nullspace of Si and we can perform
    rank reduction. The problem becomes to solve for M1 >= 0, ... Mn >= 0.

    In more detail, recall our SDP problem is in the form of

        Si = A_i0 + y_1 * A_i1 + ... + y_n * A_in >> 0.

    For each A_ij, we can always decompose that

        A_ij = Ui * X_ij * Ui' + Vi * Y_ij * Vi' + (Ui * Z_ij * Vi' + Vi * Z_ij' * Ui')

    where X_ij = (Ui'Ui)^{-1} * Ui'A_ijUi * (Ui'Ui)^{-1}.

    So the problem is equivalent to:

        (Ui'A_i0Ui) + y_1 * (Ui'A_i1Ui) + ... + y_n * (Ui'A_inUi) >> 0.

    with constraints that

        (A_i0Vi) + y_1 * (A_i1Vi) + ... + y_n * (A_inVi) = 0.
    """
    def __new__(cls, parent_node, columnspace = None, nullspace = None):
        if columnspace is None and nullspace is None:
            raise ValueError("Columnspace and nullspace cannot both be None.")


        if nullspace is None:
            if all(is_empty_matrix(mat) for mat in columnspace.values()):
                return SDPIdentityTransform.__new__(SDPIdentityTransform, parent_node)
        elif columnspace is None:
            if all(is_empty_matrix(mat) for mat in nullspace.values()):
                return SDPIdentityTransform.__new__(SDPIdentityTransform, parent_node)

        return super().__new__(cls)


    def __init__(self, parent_node: 'SDPProblem', columnspace: Dict[str, sp.Matrix] = None, nullspace: Dict[str, sp.Matrix] = None):
        def _reg(X, key):
            return sp.Matrix.hstack(*X.columnspace())
        def _perp(X, key):
            return sp.Matrix.hstack(*X.T.nullspace())

        if nullspace is None:
            columnspace = {key: _reg(columnspace[key], key) for key in columnspace}
            nullspace = {key: _perp(columnspace[key], key) for key in columnspace}
        elif columnspace is None:
            nullspace = {key: _reg(nullspace[key], key) for key in nullspace}
            columnspace = {key: _perp(nullspace[key], key) for key in nullspace}

        columnspace = parent_node._standardize_mat_dict(columnspace)
        nullspace = parent_node._standardize_mat_dict(nullspace)

        self._columnspace = columnspace
        self._nullspace = nullspace
        self._trans_x0 = None
        self._trans_space = None
        super().__init__(parent_node, columnspace, nullspace)

    def _init_child_node(self, columnspace, nullspace):
        parent_node = self.parent_node
        if parent_node is None:
            return

        # form the constraints of y by computing Sum(y_i * A_ij * Vi) = -A_i0 * Vi
        # TODO: faster fraction arithmetic
        eq_list = []
        x0_list = []
        for key, (x0, space) in parent_node._x0_and_space.items():
            V = self._nullspace[key]
            if is_empty_matrix(V):
                continue
            eq_mat = []
            for i in range(space.shape[1]):
                Aij = Mat2Vec.vec2mat(space[:,i])
                eq = list(Aij * V)
                eq_mat.append(eq)

            eq_mat = sp.Matrix(eq_mat).T
            eq_list.append(eq_mat)

            Ai0 = Mat2Vec.vec2mat(x0)
            new_x0 = list(Ai0 * V)
            x0_list.extend(new_x0)


        # eq * y + x0 = 0 => y = trans_x0 + trans_space * z
        eq_list = sp.Matrix.vstack(*eq_list)
        x0_list = sp.Matrix(x0_list)
        trans_x0, trans_space = solve_undetermined_linear(eq_list, -x0_list)
        self._trans_x0, self._trans_space = trans_x0, trans_space

        # Sum(Ui' * Aij * Ui * (trans_x0 + trans_space * z)[j]) >> 0
        new_x0_and_space = {}
        for key, (x0, space) in parent_node._x0_and_space.items():
            U = self._columnspace[key]
            if is_empty_matrix(U):
                continue
            eq_mat = []
            for i in range(space.shape[1]):
                Aij = Mat2Vec.vec2mat(space[:,i])
                eq = U.T * Aij * U
                eq_mat.append(list(eq))

            eq_mat = sp.Matrix(eq_mat).T
            new_space = eq_mat * trans_space

            Ai0 = Mat2Vec.vec2mat(x0)
            new_x0 = Mat2Vec.mat2vec(U.T * Ai0 * U) + eq_mat * trans_x0

            new_x0_and_space[key] = (new_x0, new_space)

        return SDPProblem(new_x0_and_space)

    def propagate_to_parent(self, recursive: bool = True):
        parent_node, child_node = self.parent_node, self.child_node
        if child_node.y is None:
            return parent_node
        y = self._trans_space * child_node.y + self._trans_x0
        parent_node.register_y(y, propagate_to_parent = recursive)
        return parent_node


class SDPRowMasking(SDPMatrixTransform):
    """
    Mask several rows and cols of the matrix so that they are assumed
    to be zero. This is useful when we want to reduce the rank of the
    matrix or when there are zero diagonal entries.    
    """
    def __new__(cls, parent_node, masks: Dict[str, List[int]]):
        if all(len(mask) == 0 for mask in masks.values()):
            return SDPIdentityTransform.__new__(SDPIdentityTransform, parent_node)
        return object.__new__(cls)

    def __init__(self, parent_node, masks: Dict[str, List[int]]):
        self.masks = masks
        self.unmasks = {}
        for key, n in parent_node.size.items():
            mask = set(masks[key])
            self.unmasks[key] = [i for i in range(n) if i not in mask]

        def onehot(n, i):
            return sp.Matrix([1 if j == i else 0 for j in range(n)])

        def onehot_dict(masks):
            mats = {}
            for key, n in parent_node.size.items():
                mat = [onehot(n, i) for i in masks[key]]
                if len(mat) == 0:
                    mats[key] = sp.zeros(n, 0)
                else:
                    mats[key] = sp.Matrix.hstack(*mat)
            return mats
        self.nullspace = onehot_dict(masks)
        self.columnspace = onehot_dict(self.unmasks)

        self._trans_x0 = None
        self._trans_space = None
        SDPTransformation.__init__(self, parent_node)

    def _init_child_node(self):
        masks, unmasks = self.masks, self.unmasks
        parent_node = self.parent_node
        eqs = []
        rhs = []
        nonzero_inds = {}
        for key, (x0, space) in parent_node._x0_and_space.items():
            mask = set(masks[key])
            n = Mat2Vec.length_of_mat(x0.shape[0])
            for i in mask:
                for j in range(n):
                    eqs.append(space[i*n+j,:])
                    rhs.append(x0[i*n+j])

            unmask = unmasks[key]
            nonzero_inds_ = []
            for i in unmask:
                for j in unmask:
                    nonzero_inds_.append(i*n+j)
            nonzero_inds[key] = nonzero_inds_

        eqs = sp.Matrix.vstack(*eqs)
        rhs = sp.Matrix(rhs)
        trans_x0, trans_space = solve_undetermined_linear(eqs, -rhs)
        self._trans_x0, self._trans_space = trans_x0, trans_space

        new_x0_and_space = {}
        for key, (x0, space) in parent_node._x0_and_space.items():
            space2 = space[nonzero_inds[key],:]
            new_x0 = x0[nonzero_inds[key],:] + space2 * trans_x0
            new_space = space2 * trans_space
            new_x0_and_space[key] = (new_x0, new_space)
        return SDPProblem(new_x0_and_space)

    @classmethod
    def _get_zero_diagonals(cls, x0_and_space: Dict[str, Tuple[sp.Matrix, sp.Matrix]]) -> Dict[str, List[int]]:
        """
        Return a dict indicating the indices of diagonal entries
        that are sure to be zero.
        """
        zero_diagonals = {}
        for key, (x0, space) in x0_and_space.items():
            n = Mat2Vec.length_of_mat(x0.shape[0])
            zero_diagonals[key] = []
            for i in range(n):
                if x0[i*n+i] == 0 and not any(space[i*n+i,:]):
                    zero_diagonals[key].append(i)
        return zero_diagonals

    @classmethod
    def constrain_zero_diagonals(cls, parent_node: 'SDPProblem', recursive: bool = True) -> 'SDPRowMasking':
        """
        If a diagonal of the positive semidefinite matrix is zero,
        then the corresponding row must be all zeros. This function
        constrains the solution to satisfy this condition.
        """
        zero_diagonals = parent_node._get_zero_diagonals()
        sdp = parent_node
        while any(zero_diagonals.values()):
            sdp = SDPRowMasking(sdp, zero_diagonals).child_node
            if not recursive:
                break
            zero_diagonals = sdp._get_zero_diagonals()
        return sdp._transforms[-1] if sdp._transforms else SDPIdentityTransform(parent_node)


class SDPVectorTransform(SDPTransformation):
    """
    Assume the original problem to be S1 >= 0, ... Sn >= 0
    where Si = xi + spacei @ y.
    Now we make the transformation y = A @ z + b.
    The new problem is to solve for z such that S1 >= 0, ... Sn >= 0.
    """
    def __init__(self, parent_node, A: sp.Matrix, b: sp.Matrix):
        self._A = A
        self._b = b
        super().__init__(parent_node, A, b)

    @classmethod
    def from_equations(cls, parent, eqs: sp.Matrix, rhs: sp.Matrix):
        """
        Set constraints that eqs * y = rhs => y = A * z + b.
        """
        b, A = solve_undetermined_linear(eqs, rhs)
        return SDPVectorTransform(parent, A, b)

    def _init_child_node(self, A, b):
        parent_node = self.parent_node
        if parent_node is None:
            return
        x0_and_space = {}
        for key, (x0, space) in parent_node._x0_and_space.items():
            x0_ = x0 + space @ b
            space_ = space @ A
            x0_and_space[key] = (x0_, space_)
        return SDPProblem(x0_and_space)

    def propagate_to_parent(self, recursive: bool = True):
        parent_node, child_node = self.parent_node, self.child_node
        if child_node.y is None:
            return parent_node
        parent_node.y = self._A @ child_node.y + self._b
        parent_node.S = child_node.S
        parent_node.decompositions = child_node.decompositions
        if recursive:
            parent_node.propagate_to_parent(recursive = recursive)
        return parent_node
    




class SDPProblem():
    """
    Class to solve rational SDP feasible problems, which is in the form of

        S_i = C_i + y_1 * A_i1 + y_2 * A_i2 + ... + y_n * A_in >> 0.
    
    where C, A_ij ... are known symmetric matrices, and y_i are free variables.

    It can be rewritten in the form of

        vec(S_i) = x_i + space_i @ y >> 0.

    And together they are vec([S_1, S_2, ...]) = [x_1, x_2, ...] + [space_1, space_2, ...] @ y
    where x_i and space_i are known. The problem is to find a rational solution y such that S_i >> 0.
    This is the standard form of our rational SDP feasible problem.
    """
    _has_picos = _check_picos(verbose = True)
    def __init__(
        self,
        x0_and_space: Union[Dict[str, Tuple[sp.Matrix, sp.Matrix]], List[Tuple[sp.Matrix, sp.Matrix]]],
        free_symbols = None
    ):
        """
        Initializing a SDPProblem object.
        """
        if isinstance(x0_and_space, list):
            keys = ['S_%d'%i for i in range(len(x0_and_space))]
            x0_and_space = dict(zip(keys, x0_and_space))
        elif isinstance(x0_and_space, dict):
            keys = list(x0_and_space.keys())
        else:
            raise TypeError("x0_and_space should be a dict or a list containing tuples.")

        self._x0_and_space = x0_and_space

        self.y = None
        self.S = None
        self.decompositions = None

        if len(keys):
            dof = x0_and_space[keys[0]][1].shape[1]
            for x0, space in x0_and_space.values():
                if space.shape[1] != dof:
                    raise ValueError("The number of columns of spaces should be the same.")

            if free_symbols is not None:
                if len(free_symbols) != dof:
                    raise ValueError("Length of free_symbols and space should be the same. But got %d and %d."%(len(free_symbols), self.space.shape[1]))
                self.free_symbols = list(free_symbols)
            else:
                self.free_symbols = list(sp.Symbol('y_{%d}'%i) for i in range(dof))
        else:
            self.free_symbols = []

        self.sdp = None

        # record the numerical solutions
        self._ys = []

        # record the transformation dependencies
        self._transforms: List[SDPTransformation] = []

    @property
    def dof(self):
        """
        The degree of freedom of the SDP problem.
        """
        return len(self.free_symbols)

    def keys(self, filter_none: bool = False) -> List[str]:
        keys = list(self._x0_and_space.keys())
        if filter_none:
            _size = lambda key: self._x0_and_space[key][1].shape[1] * self._x0_and_space[key][1].shape[0]
            keys = [key for key in keys if _size(key) != 0]
        return keys

    def get_size(self, key: str) -> int:
        return Mat2Vec.length_of_mat(self._x0_and_space[key][1].shape[0])

    @property
    def size(self) -> Dict[str, int]:
        return {key: self.get_size(key) for key in self.keys()}

    def __repr__(self):
        return "<SDPProblem dof=%d size=%s>"%(self.dof, self.size)

    def _standardize_mat_dict(self, mat_dict: Dict[str, sp.Matrix]) -> Dict[str, sp.Matrix]:
        """
        Standardize the matrix dictionary.
        """
        if not set(mat_dict.keys()) == set(self.keys()):
            print(mat_dict.keys(), self.keys())
            raise ValueError("The keys of the matrix dictionary should be the same as the keys of the SDP problem.")
        for key, X in mat_dict.items():
            if not isinstance(X, sp.MatrixBase):
                raise ValueError("The values of the matrix dictionary should be sympy MatrixBase.")
            if is_empty_matrix(X):
                n = self.get_size(key)
                mat_dict[key] = sp.zeros(n, 0)
        return mat_dict

    @classmethod
    def from_full_x0_and_space(
        cls,
        x0: sp.Matrix,
        space: sp.Matrix,
        splits: Union[Dict[str, int], List[int]]
    ) -> 'SDPProblem':
        keys = None
        if isinstance(splits, dict):
            keys = list(splits.keys())
            splits = list(splits.values())

        x0_and_space = []
        start = 0
        for n in splits:
            x0_ = x0[start:start+n**2,:]
            space_ = space[start:start+n**2,:]
            x0_and_space.append((x0_, space_))
            start += n**2

        if keys is not None:
            x0_and_space = dict(zip(keys, x0_and_space))
        return SDPProblem(x0_and_space)
        

    @classmethod
    def from_equations(
        cls,
        eq: sp.Matrix,
        rhs: sp.Matrix,
        splits: Union[Dict[str, int], List[int]]
    ) -> 'SDPProblem':
        """
        Assume the SDP problem can be rewritten in the form of

            eq * [vec(S1); vec(S2); ...] = rhs
        
        where Si.shape[0] = splits[i].
        The function formulates the SDP problem from the given equations.

        Parameters
        ----------
        eq : sp.Matrix
            The matrix eq.
        rhs : sp.Matrix
            The matrix rhs.
        splits : Union[Dict[str, int], List[int]]
            The splits of the size of each symmetric matrix.

        Returns
        ----------
        sdp : SDPProblem
            The SDP problem constructed.    
        """
        x0, space = solve_undetermined_linear(eq, rhs)
        return cls.from_full_x0_and_space(x0, space, splits)

    @classmethod
    def from_matrix(
        cls,
        S: Union[sp.Matrix, List[sp.Matrix], Dict[str, sp.Matrix]],
    ) -> 'SDPProblem':
        """
        Construct a `SDPProblem` from symbolic symmetric matrices.
        The problem is to solve a parameter set such that all given
        symmetric matrices are positive semidefinite. The result can
        be obtained by `SDPProblem.as_params()`.

        Parameters
        ----------
        S : Union[sp.Matrix, List[sp.Matrix], Dict[str, sp.Matrix]]
            The symmetric matrices that SDP requires to be positive semidefinite.
            Each entry of the matrix should be linear in the free symbols.

        Returns
        ----------
        sdp : SDPProblem
            The SDP problem constructed.
        """

        keys = None
        if isinstance(S, dict):
            keys = list(S.keys())
            S = list(S.values())

        if isinstance(S, sp.Matrix):
            S = [S]

        free_symbols = set()
        for s in S:
            if not isinstance(s, sp.Matrix):
                raise ValueError("S must be a list of sp.Matrix or dict of sp.Matrix.")
            free_symbols |= set(s.free_symbols)

        free_symbols = list(free_symbols)

        x0_and_space = []
        for s in S:
            x0, space, _ = _decompose_matrix(s, free_symbols)
            x0_and_space.append((x0, space))

        if keys is not None:
            x0_and_space = dict(zip(keys, x0_and_space))

        return SDPProblem(x0_and_space, free_symbols = free_symbols)

    def S_from_y(self, 
            y: Optional[Union[sp.Matrix, np.ndarray, Dict]] = None
        ) -> Dict[str, sp.Matrix]:
        """
        Given y, compute the symmetric matrices. This is useful when we want to see the
        symbolic representation of the SDP problem.

        This function does not register the result to self.S.

        Parameters
        ----------
        y : Optional[Union[sp.Matrix, np.ndarray]]
            The generating vector. If None, it uses a symbolic vector.

        Returns
        ----------
        S : Dict[str, sp.Matrix]
            The symmetric matrices that SDP requires to be positive semidefinite.
        """
        m = self.dof
        if y is None:
            y = sp.Matrix(self.free_symbols).reshape(m, 1)
        elif isinstance(y, sp.MatrixBase):
            if y.shape != (m, 1):
                raise ValueError(f"Vector y must be a matrix of shape ({m}, 1).")
        elif isinstance(y, np.ndarray):
            if y.size != m:
                raise ValueError(f"Vector y must be an array of shape ({m},) or ({m}, 1).")
            y = sp.Matrix(y.flatten())
        elif isinstance(y, dict):
            y = sp.Matrix([y.get(v, v) for v in self.free_symbols]).reshape(m, 1)

        return S_from_y(y, self._x0_and_space)

    def register_y(self,
            y: Union[sp.Matrix, np.ndarray, Dict],
            perturb: bool = False,
            propagate_to_parent: bool = True
        ) -> None:
        """
        Manually register a solution y to the SDP problem.

        Parameters
        ----------
        y : Union[sp.Matrix, np.ndarray, Dict]
            The solution to the SDP problem.
        perturb : bool
            If perturb == True, it must return the result by adding a small perturbation * identity to the matrices.
            This is useful when the given y is numerical.
        propagate_to_parent : bool
            If True, propagate the solution to the parent SDP problem.
        """
        S = self.S_from_y(y)
        decomps = {}
        for key, s in S.items():
            decomp = congruence_with_perturbation(s, perturb = perturb)
            if decomp is None:
                raise ValueError(f"Matrix {key} is not positive semidefinite given y.")
            decomps[key] = decomp
        self.y = y
        self.S = S
        self.decompositions = decomps
        if propagate_to_parent:
            self.propagate_to_parent(recursive = True)


    def as_params(self) -> Dict[sp.Symbol, sp.Rational]:
        """
        Return the free symbols and their values.
        """
        return dict(zip(self.free_symbols, self.y))


    def _construct_sdp(self,
            reg: float = 0,
            constraints: List[Union[PicosExpression, sp.Expr, Callable]] = []
        ):
        """
        Construct picos.Problem from self. The function
        is automatically called when __init__.

        Parameters
        ----------
        reg : float
            For symmetric matrix S, we require S >> reg * I.
        constraints : List[Union[PicosExpression, sp.Expr, Callable]]:
            Additional constraints.

            Example:
            ```
                constraints = [
                    lambda sdp: sdp.variables['y'][0] == 1
                ]
            ```

        Returns
        ---------
        sdp : picos.Problem
            Picos problem created. If there is no degree of freedom,
            return None.
        """
        if self.dof == 0:
            return None


        try:
            import picos

            # SDP should use numerical algorithm
            sdp = picos.Problem()
            y = picos.RealVariable('y', self.dof)
            for key, (x0, space) in self._x0_and_space.items():
                k = Mat2Vec.length_of_mat(x0.shape[0])
                if k == 0:
                    continue
                x0_ = np.array(x0).astype(np.float64).flatten()
                space_ = np.array(space).astype(np.float64)
                S = picos.SymmetricVariable(key, (k,k))
                sdp.add_constraint(S >> reg)

                self._add_sdp_eq(sdp, S, x0_, space_, y)

            for constraint in constraints or []:
                constraint = self._align_constraint(constraint)
                sdp.add_constraint(constraint)
        except Exception as e:
            raise e
            return None

        self.sdp = sdp
        return sdp

    def _add_sdp_eq(self, sdp, S, x0, space, y):
        """
        Helper function that add the constraint
        S.vec == x0 + space * y to the sdp.
        """
        sdp.add_constraint(S.vec == x0 + space * y)


    def _nsolve_with_early_stop(
            self,
            max_iters: int = 50,
            min_iters: int = 10,
            verbose: bool = False
        ) -> Any:
        """
        Numerically solve the sdp with PICOS.

        Python package PICOS solving SDP problem with CVXOPT will oftentimes
        faces ZeroDivisionError. This is due to the iterations is large while
        working precision is not enough.

        This function is a workaround to solve this. It flexibly reduces the
        number of iterations and tries to solve the problem again until
        the problem is solved or the number of iterations is less than min_iters.

        Parameters
        ----------
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
        sdp = self.sdp
        # verbose = self.verbose

        sdp.options.max_iterations = max_iters
        if verbose:
            print('Retry Early Stop sdp Max Iters = %d' % sdp.options.max_iterations)

        try:
            solution = sdp._strategy.execute()
            return solution # .primals[sdp.variables['y']]
        except Exception as e:
            if isinstance(e, ZeroDivisionError):
                if max_iters // 2 >= min_iters and max_iters > 1:
                    return self._nsolve_with_early_stop(
                                max_iters = max_iters // 2, 
                                min_iters = min_iters, 
                                verbose = verbose
                            )
            return None
        return None


    def _get_defaulted_objectives(self):
        """
        Get the default objectives of the SDP problem.
        """
        obj_key = self.keys(filter_none = True)[0]
        objectives = [
            ('max', self.sdp.variables[obj_key].tr),
            ('min', self.sdp.variables[obj_key].tr),
            ('max', self.sdp.variables[obj_key]|1)
        ]
        # x = np.random.randn(*sdp.variables[obj_key].shape)
        # objectives.append(('max', lambda sdp: sdp.variables[obj_key]|x))
        return objectives

    def _align_objective(
            self,
            objective: Tuple[str, Union[PicosExpression, sp.Expr, Callable]]
        ) -> Tuple[str, PicosExpression]:
        """
        Align the objective to an expression of sdp.variables.
        """
        indicator, x = objective
        if isinstance(x, Callable):
            x = x(self.sdp)
        elif isinstance(x, sp.Expr):
            f = sp.lambdify(self.free_symbols, x)
            x = f(*self.sdp.variables['y'])
        return indicator, x

    def _align_constraint(
        self,
        constraint: Union[PicosExpression, sp.Expr, Callable]
    ) -> PicosExpression:
        """
        Align the constraint to an expression of sdp.variables.
        """
        x = constraint
        if isinstance(x, Callable):
            x = x(self.sdp)
        elif isinstance(x, sp.core.relational.Relational):
            lhs = sp.lambdify(self.free_symbols, x.lhs)(*self.sdp.variables['y'])
            rhs = sp.lambdify(self.free_symbols, x.rhs)(*self.sdp.variables['y'])
            sym = _RELATIONAL_TO_FUNC[x.__class__]
            x = getattr(lhs, sym)(rhs)
        return x


    def _nsolve_with_obj(
            self,
            objectives: List[Tuple[str, Union[PicosExpression, sp.Expr, Callable]]],
            context: Optional[AbstractContextManager] = None
        ) -> Generator[Optional[np.ndarray], None, None]:
        """
        Numerically solve a SDP problem with multiple objectives.
        This returns a generator of ndarray.

        Parameters
        ---------- 
        objectives : Optional[List[Tuple[str, Union[Any, Callable]]]]
            Although it suffices to find one feasible solution, we might 
            use objective to find particular feasible solution that has 
            good rational approximant. This parameter takes in multiple objectives, 
            and the solver will try each of the objective. If still no 
            approximant is found, the final solution will average this 
            sdp solution and perform rationalization. Note that SDP problem is 
            convex so the convex combination is always feasible and not on the
            boundary.

            Example: 
            ```
            objectives = [
                ('max', lambda sdp: sdp.variables['S_major'].tr),
                ('max', lambda sdp: sdp.variables['S_major']|1)
            ]
            ```
        context : Optional[AbstractContextManager]
            Context that the SDP is solved in.

        Yields
        ---------
        y: Optional[np.ndarray]
            Numerical solution y. Return None if y unfound.
        """
        from picos.modeling.strategy import Strategy
        sdp = self.sdp

        if context is None:
            context = nullcontext()
        if objectives is None:
            objectives = self._get_defaulted_objectives()

        with context:
            for objective in objectives:
                # try each of the objectives
                sdp.set_objective(*self._align_objective(objective))

                sdp._strategy = Strategy.from_problem(sdp)
                solution = self._nsolve_with_early_stop(max_iters = 50)

                if solution is not None:
                    try:
                        y = np.array(solution.primals[sdp.variables['y']])
                    except KeyError:
                        raise SDPInfeasibleError("SDP problem numerically infeasible.")

                    self._ys.append(y)
                    yield y

                # NOTE: PICOS uses an isometric vectorization of symmetric matrices
                #       (off-diagonal elements are divided by sqrt(2))
                #       so if we need to convert it back, we had better use its API.
                # S0 = (SymmetricVectorization((6,6)).devectorize(cvxopt.matrix(list(solution.primals.values())[0])))

                yield None

    def _nsolve_with_rationalization(
            self,
            objectives: List[Tuple[str, Union[PicosExpression, sp.Expr, Callable]]],
            context: Optional[AbstractContextManager] = None,
            **kwargs
        ) -> Optional[Tuple[sp.Matrix, Decomp]]:
        """
        Solve the SDP problem and returns the rational solution if any.

        Parameters
        ----------
        objectives : List[Tuple[str, Union[PicosExpression, sp.Expr, Callable]]]
            See details in self._nsolve_with_obj.
        context : Optional[AbstractContextManager]
            See details in self._nsolve_with_obj.
        kwargs : Any
            Keyword arguments that passed into self.rationalize.

        Returns
        ----------
        y, decompositions : Optional[Tuple[sp.Matrix, Decomp]]
            If the problem is solved, return the congruence decompositions `y, [(S, U, diag)]`
            So that each `S = U.T * diag(diag) * U` where `U` is upper triangular.
            Otherwise, return None.
        """
        for y in self._nsolve_with_obj(objectives, context):
            if y is not None:
                ra = self.rationalize(y, **kwargs)
                if ra is not None:
                    return ra

    def rationalize(
            self,
            y: np.ndarray,
            try_rationalize_with_mask: bool = True,
            times: int = 1,
            check_pretty: bool = True,
        ) -> Optional[Tuple[sp.Matrix, Decomp]]:
        """
        Rationalize a numerical vector y so that it produces a rational solution to SDP.

        Parameters
        ----------
        y : np.ndarray
            Numerical solution y.
        kwargs : Any
            Arguments that passed into rationalize_and_decompose.

        Returns
        ----------
        y, decompositions : Optional[Tuple[sp.Matrix, Decomp]]
            If the problem is solved, return the congruence decompositions `y, [(S, U, diag)]`
            So that each `S = U.T * diag(diag) * U` where `U` is upper triangular.
            Otherwise, return None.
        """
        decomp = rationalize_and_decompose(y, self._x0_and_space,
            try_rationalize_with_mask=try_rationalize_with_mask, times=times, check_pretty=check_pretty
        )
        return decomp

    def rationalize_combine(
            self,
            ys: List[np.ndarray] = None,
            verbose: bool = False,
        ) ->  Optional[Tuple[sp.Matrix, Decomp]]:
        """
        Linearly combine all numerical solutions [y] to produce a rational solution.

        Parameters
        ----------
        y : np.ndarray
            Numerical solution y.
        verbose : bool
            Whether to print out the eigenvalues of the combined matrix. Defaults
            to False.

        Returns
        ----------
        y, decompositions : Optional[Tuple[sp.Matrix, Decomp]]
            If the problem is solved, return the congruence decompositions `y, [(S, U, diag)]`
            So that each `S = U.T * diag(diag) * U` where `U` is upper triangular.
            Otherwise, return None.
        """
        if ys is None:
            ys = self._ys

        if len(ys) == 0:
            return None

        y = np.array(ys).mean(axis = 0)

        S_numer = S_from_y(y, self._x0_and_space)
        if all(_.is_positive_definite for _ in S_numer.values()):
            lcm, times = 1260, 5
        else:
            # spaces = [space for x0, space in self._x0_and_space.values()]
            # lcm = max(1260, sp.prod(set.union(*[set(sp.primefactors(_.q)) for _ in spaces if isinstance(_, sp.Rational)])))
            # times = int(10 / sp.log(lcm, 10).n(15) + 3)
            times = 5

        if verbose:
            print('Minimum Eigenvals = %s'%[min(map(lambda x:sp.re(x), _.eigenvals())) for _ in S_numer.values()])

        decomp = rationalize_and_decompose(y, self._x0_and_space,
            try_rationalize_with_mask = False, lcm = 1260, times = times
        )
        return decomp


    def _solve_trivial(
            self,
            objectives: Optional[List[Tuple[str, Union[PicosExpression, sp.Expr, Callable]]]] = None,
            constraints: List[Union[PicosExpression, sp.Expr, Callable]] = []
        ) -> Optional[Tuple[sp.Matrix, Decomp]]:
        """
        Solve SDP numerically with given objectives and constraints.
        """
        # a context that imposes additional constraints
        if len(constraints):
            @contextmanager
            def restore_constraints(sdp, constraints):
                constraints_num = len(sdp.constraints)
                for constraint in constraints:
                    constraint = self._align_constraint(constraint)
                    sdp.add_constraint(constraint)
                yield
                for i in range(len(sdp.constraints) - 1, constraints_num - 1, -1):
                    sdp.remove_constraint(i)
            context = restore_constraints(self.sdp, constraints)
        else:
            context = nullcontext()
        return self._nsolve_with_rationalization(objectives, context)


    def _solve_relax(
            self
        ) -> Optional[Tuple[sp.Matrix, Decomp]]:
        """
        Solve SDP with such objective:
            S - l * I >= 0.
            max(l)
        """
        import picos
        from picos.constraints.con_lmi import LMIConstraint

        sdp = self.sdp
        obj_key = self.keys()[0]
        lamb = picos.RealVariable('lamb', 1)
        obj = sdp.variables[obj_key]

        @contextmanager
        def restore_constraints(sdp, obj, lamb):
            for i, constraint in enumerate(sdp.constraints):
                if isinstance(constraint, LMIConstraint) and obj in constraint.variables:
                    # remove obj >> 0
                    sdp.remove_constraint(i)
                    break
            sdp.add_constraint((obj - lamb * picos.I(obj.shape[0])) >> 0)
            sdp.add_constraint(lamb >= 0)

            yield
            sdp.remove_constraint(-1)
            sdp.remove_constraint(-1)
            sdp.set_objective('max', obj.tr)

        objectives = [('max', lambda sdp: sdp.variables['lamb'])]
        context = restore_constraints(sdp, obj, lamb)
        return self._nsolve_with_rationalization(objectives, context)


    def _solve_partial_deflation(
            self,
            deflation_sequence: Optional[List[int]] = None,
            verbose: bool = False
        ) -> Optional[Tuple[sp.Matrix, Decomp]]:
        """
        We use the following idea to generate a rational solution:
        1. Solve SDP with objectives = max(y[-1]) and min(y[-1]).
        2. Set y[-1] = (max + min) / 2 as a new constraint and solve SDP again.
        3. Repeat step 2 until the solution is rational.
        """
        @contextmanager
        def restore_constraints(sdp):
            constraints_num = len(sdp.constraints)
            yield
            for i in range(len(sdp.constraints) - 1, constraints_num - 1, -1):
                sdp.remove_constraint(i)

        n = self.dof
        sdp = self.sdp
        if deflation_sequence is None:
            deflation_sequence = range(n)

        with restore_constraints(sdp):
            for i in deflation_sequence:
                bounds = []
                objectives = [
                    ('max', lambda sdp: sdp.variables['y'][i]),
                    ('min', lambda sdp: sdp.variables['y'][i])
                ]
                cnt_ys = len(self._ys)
                ra = self._nsolve_with_rationalization(objectives)
                cnt_sol = len(self._ys) - cnt_ys

                if cnt_sol == 0 or isinstance(ra, tuple):
                    return ra
                elif cnt_sol < 2:
                    # not enough solutions
                    return None

                ra = self.rationalize_combine(verbose = verbose)
                if ra is not None:
                    return ra

                bounds = [self._ys[-2][i], self._ys[-1][i]]

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

                sdp.add_constraint(sdp.variables['y'][i] == float(fix))


    def _solve_degenerated(
            self
        ) -> Optional[Tuple[sp.Matrix, Decomp]]:
        """
        Solve the SDP if degree of freedom is zero.
        In this case it does not rely on any optimization package.
        """
        if self.dof == 0:
            decomp = rationalize_and_decompose(
                sp.Matrix([]).reshape(0,1), self._x0_and_space,
                check_pretty = False
            )
            return decomp


    def _solve_wrapped(
        self,
        method: str = 'trivial',
        allow_numer: int = 0,
        verbose: bool = False,
        **kwargs
    ) -> Optional[Tuple[sp.Matrix, Decomp]]:
        """
        Solve SDP with given method. Moreover, we try to make a convex combinations
        of all numerical solution to test whether it produces a rational solution.
        Finally, if allow_numer == 1, return one of the numerical solution if rationalization fails.
        If allow_numer == 2, force to return the first numerical solution.

        Parameters
        ----------
        method : str
            The method to solve the SDP problem. Currently supports:
            'partial deflation' and 'relax' and 'trivial'.
        allow_numer : int
            Whether to allow numerical solution. If 0, then the function will return None if
            the rational feasible solution does not exist. If 1, then the function will return a numerical solution
            if the rational feasible solution does not exist. If 2, then the function will return the first
            numerical solution  without any rationalization.
        verbose : bool
            If True, print the information of the solving process.

        Returns
        ----------
        y, decompositions : Optional[Tuple[sp.Matrix, Decomp]]
            If the problem is solved, return the congruence decompositions `y, [(S, U, diag)]`
            So that each `S = U.T * diag(diag) * U` where `U` is upper triangular.
            Otherwise, return None.
        """
        method = method.lower()

        num_sol = len(self._ys)

        if method == 'trivial':
            ra = self._solve_trivial(**kwargs)
        elif method == 'relax':
            ra = self._solve_relax(**kwargs)
        elif method == 'partial deflation':
            ra = self._solve_partial_deflation(verbose=verbose, **kwargs)
        else:
            raise ValueError("Method %s is not supported."%method)

        if allow_numer < 2:
            if ra is not None:
                return ra

            ra = self.rationalize_combine(verbose = verbose)
            if ra is not None:
                return ra

        if len(self._ys) > num_sol:
            if allow_numer:
                y = sp.Matrix(self._ys[-1])
                decomp = rationalize_and_decompose(y, self._x0_and_space,
                    try_rationalize_with_mask = False, times = 0, perturb = True, check_pretty = False
                )
                return decomp
            else:
                raise SDPRationalizeError(
                    "Failed to find a rational solution despite having a numerical solution."
                )

        return None

    def solve(
            self,
            method: str = 'trivial',
            allow_numer: int = 0,
            verbose: bool = False,
            solve_child: bool = True,
            propagate_to_parent: bool = True,
            **kwargs
        ) -> bool:
        """
        Interface for solving the SDP problem.

        Parameters
        ----------
        method : str
            The method to solve the SDP problem. Currently supports:
            'partial deflation' and 'relax' and 'trivial'.
        allow_numer : int
            Whether to allow numerical solution. If 0, then the function will return None if
            the rational feasible solution does not exist. If 1, then the function will return a numerical solution
            if the rational feasible solution does not exist. If 2, then the function will return the first
            numerical solution without any rationalization.
        verbose : bool
            If True, print the information of the solving process.
        solve_child : bool
            Whether to solve the problem from the child node. Defaults to True. If True,
            it only uses the newest child node to solve the problem. If no child node is found,
            it defaults to solve the problem by itself.
        propagate_to_parent : bool
            Whether to propagate the result to the parent node. Defaults to True.

        Returns
        ----------
        bool
            Whether the problem is solved. If True, the result can be accessed by
            SDPProblem.y and SDPProblem.S and SDPProblem.decompositions.
        """
        if solve_child:
            child = self.get_last_child()
            if child is not self:
                return child.solve(
                    method = method,
                    allow_numer = allow_numer,
                    verbose = verbose,
                    solve_child = solve_child,
                    propagate_to_parent = propagate_to_parent,
                    **kwargs
                )

        if self.dof == 0:
            solution = self._solve_degenerated()
        elif self._has_picos:
            self._construct_sdp()
            solution = self._solve_wrapped(method = method, allow_numer = allow_numer, verbose = verbose, **kwargs)
        else:
            solution = None

        if solution is not None:
            self.y = solution[0]
            self.S = dict((key, S[0]) for key, S in solution[1].items())
            self.decompositions = dict((key, S[1:]) for key, S in solution[1].items())

        if propagate_to_parent:
            self.propagate_to_parent()

        return (solution is not None)

    @property
    def parents(self) -> List['SDPProblem']:
        """
        Return the parent nodes.
        """
        return [transform.parent_node for transform in self._transforms if transform.is_child(self)]

    @property
    def children(self) -> List['SDPProblem']:
        """
        Return the child nodes.
        """
        return [transform.child_node for transform in self._transforms if transform.is_parent(self)]

    def get_last_child(self) -> 'SDPProblem':
        """
        Get the last child node of the current node recursively.
        """
        children = self.children
        if len(children):
            return children[-1].get_last_child()
        return self

    def common_transform(self, other: 'SDPProblem') -> SDPTransformation:
        """
        Return the common transformation between two SDP problems.
        """
        for transform in self._transforms:
            if transform.is_parent(self) and transform.is_child(other):
                return transform
            elif transform.is_parent(other) and transform.is_child(self):
                return transform

    def print_graph(self) -> None:
        """
        Print the dependency graph of the SDP problem.
        """
        _MAXLEN = 30
        _PAD = (_MAXLEN - 10) // 2
        print(" " * _PAD + "SDPProblem" + " " * _PAD + self.__str__())
        sdp = self

        def _formatter(a):
            filler_length = _MAXLEN - len(a) - 9
            filler = '-' * (filler_length // 2)
            filler2 = '-' * (filler_length - len(filler))
            return f"---{filler} {a} {filler2}-->"

        while len(sdp.children):
            sdp2 = sdp.children[-1]
            transform = sdp.common_transform(sdp2)
            print(_formatter(transform.__class__.__name__) + " " + sdp2.__str__())
            sdp = sdp2

    def propagate_to_parent(self, recursive: bool = True):
        """
        Propagate the result to the parent node.
        """
        for transform in self._transforms:
            if transform.is_child(self):
                transform.propagate_to_parent(recursive = recursive)

    def constrain_subspace(self, columnspaces: Dict[str, sp.Matrix], to_child: bool = False) -> 'SDPProblem':
        """
        Assume Si = Qi * Mi * Qi.T where Qi are given.
        Then the problem becomes to find Mi >> 0.

        Parameters
        ----------
        columnspaces : Dict[str, sp.Matrix]
            The matrices that represent the subspace. The keys of dictionary
            should match the keys of self.keys().
        to_child : bool
            If True, apply the constrain to the child node. Otherwise, apply
            the constrain to the current node. Defaults to False.

        Returns
        ----------
        SDPProblem
            The new SDP problem.

        Raises
        ----------
        ValueError
            If there is no solution to the linear system Si = Qi * Mi * Qi.T,
            then it raises an error.
        """
        if to_child:
            raise NotImplementedError("Constraining to child node is not implemented yet.")
        transform = SDPMatrixTransform(self, columnspace=columnspaces)
        return transform.child_node

    def constrain_nullspace(self, nullspaces: Dict[str, sp.Matrix], to_child: bool = False) -> 'SDPProblem':
        """
        Assume Si * Ni = 0 where Ni are given, which means that there exists Qi
        such that Si = Qi * Mi * Qi.T where Qi are nullspaces of Ni.
        Then the problem becomes to find Mi >> 0.

        Parameters
        ----------
        nullspaces : Dict[str, sp.Matrix]
            The matrices that represent the nullspace. The keys of dictionary
            should match the keys of self.keys().
        to_child : bool
            If True, apply the constrain to the child node. Otherwise, apply
            the constrain to the current node. Defaults to False.

        Returns
        ----------
        SDPProblem
            The new SDP problem.

        Raises
        ----------
        ValueError
            If there is no solution to the linear system Si = Qi * Mi * Qi.T,
            then it raises an error.
        """
        sdp: SDPProblem = self
        nullspaces = self._standardize_mat_dict(nullspaces)
        if to_child:
            while len(sdp.children):
                sdp2 = sdp.children[-1]
                transform = sdp.common_transform(sdp2)
                nullspaces = {key: transform.columnspace[key].T * nullspaces[key] for key in sdp2.keys()}
                sdp = sdp2
        transform = SDPMatrixTransform(sdp, nullspace=nullspaces)
        return transform.child_node

    def constrain_symmetry(self) -> 'SDPProblem':
        """
        Constrain the solution to be symmetric. This is useful to reduce
        the degree of freedom when the given symbolic matrix is not symmetric.
        """
        # first solve for the nullspace the y should lie in
        eqs = []
        rhs = []
        for key, (x0, space) in self._x0_and_space.items():
            n = Mat2Vec.length_of_mat(x0.shape[0])
            for i in range(1, n):
                for j in range(i):
                    eqs.append(space[i*n+j, :] - space[j*n+i, :])
                    rhs.append(x0[j*n+i] - x0[i*n+j])
        if len(eqs) == 0:
            return self
        eqs = sp.Matrix.vstack(*eqs)
        rhs = sp.Matrix(rhs)
        transform = SDPVectorTransform.from_equations(self, eqs, rhs)
        return transform.child_node

    def constrain_equal_entries(self, entry_tuples: Dict[str, List[Tuple[Tuple[int, int], Tuple[int, int]]]]) -> 'SDPProblem':
        """
        Constrain some of the entries to be equal. This is a generalization of
        `constrain_symmetry`.

        Parameters
        ----------
        entry_tuples : Dict[str, List[Tuple[Tuple[int, int], Tuple[int, int]]]]
            The keys of the dictionary should match the keys of self.keys().
            The value of the dictionary should be a list of tuples. Each tuple
            contains two pairs of indices.
        """
        eqs = []
        rhs = []
        for key, (x0, space) in self._x0_and_space.items():
            n = Mat2Vec.length_of_mat(x0.shape[0])
            for (i, j), (k, l) in entry_tuples.get(key, []):
                if i == k and j == l:
                    continue
                eq = space[i*n+j, :] - space[k*n+l, :]
                eqs.append(eq)
                rhs.append(x0[k*n+l] - x0[i*n+j])
        if len(eqs) == 0:
            return self
        eqs = sp.Matrix.vstack(*eqs)
        rhs = sp.Matrix(rhs)
        transform = SDPVectorTransform.from_equations(self, eqs, rhs)
        return transform.child_node

    def _get_zero_diagonals(self) -> Dict[str, List[int]]:
        return SDPRowMasking._get_zero_diagonals(self._x0_and_space)

    def constrain_zero_diagonals(self, recursive: bool = True) -> 'SDPProblem':
        """
        If a diagonal of the positive semidefinite matrix is zero,
        then the corresponding row must be all zeros. This function
        constrains the solution to satisfy this condition.
        """
        return SDPRowMasking.constrain_zero_diagonals(self, recursive = recursive).child_node