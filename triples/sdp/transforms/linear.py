"""
This module contains basic linear transformations of SDP problems.
"""

from typing import Tuple, List, Dict, Union, Optional, Callable, Any, TYPE_CHECKING

from sympy.matrices import MutableDenseMatrix as Matrix

from .transform import SDPTransformation
from ..arithmetic import (
    ArithmeticTimeout, is_empty_matrix, vec2mat, reshape,
    solve_undetermined_linear, solve_nullspace, solve_columnspace,
    matmul, matadd, matmul_multiple, symmetric_bilinear, symmetric_bilinear_multiple,
    principal_basis, sqrtsize_of_mat
)

if TYPE_CHECKING:
    from ..abstract import SDPProblemBase


class SDPLinearTransform(SDPTransformation):
    """
    Standard linear transformation class of SDP problems.

    Parameters
    -----------
    parent_node : SDPProblemBase
        The parent node of the transformation.
    child_node : SDPProblemBase
        The child node of the transformation.
    A : Matrix
        The matrix A in y = Ay' + b.
        If None, the linear transformation is assumed to be the identity y = y'.
    b : Matrix
        The matrix b in y = Ay' + b.
        If None, the linear transformation is assumed to be the identity y = y'.
    """
    _A = None
    _b = None
    def __init__(self, parent_node, child_node, A: Optional[Matrix]=None, b: Optional[Matrix]=None):
        super().__init__(parent_node, child_node)
        self._A = A
        self._b = b
        if (A is None) ^ (b is None):
            raise ValueError("A and b should be both None or both not None.")

    def get_y_transform_from_child(self) -> Tuple[Matrix, Matrix]:
        return self._A, self._b

    def propagate_to_parent(self):
        """
        Propagate the solution of the child problem to the parent problem.
        """
        parent, child = self.parent_node, self.child_node
        A, b = self.get_y_transform_from_child()
        if A is None and b is None:
            y = child.y
        else:
            y = A @ child.y + b
        parent.y = y
        parent.S = child.S
        parent.decompositions = child.decompositions

    def propagate_nullspace_to_child(self, nullspace: Dict[Any, Matrix]) -> Dict[Any, Matrix]:
        return nullspace

    def propagate_affine_to_child(self, A: Matrix, b: Matrix) -> Tuple[Matrix, Matrix]:
        """
        Get `A'`, `b'` such that `A@y + b = A'@y' + b'`, where `y` and `y'`
        are the solutions of the parent and child problems respectively.
        """
        U, v = self.get_y_transform_from_child()
        if U is None and v is None:
            return A, b
        return matmul(A, U), matadd(matmul(A, v), b)


    @classmethod
    def apply_from_affine(cls, parent_node: "SDPProblemBase", A: Matrix, b: Matrix,
            to_child: bool=True, time_limit: Optional[Union[Callable, float]]=None) -> "SDPProblemBase":
        time_limit = ArithmeticTimeout.make_checker(time_limit)
        if to_child:
            # this should use inverse, or converted to from_equations
            if parent_node.get_last_child() != parent_node:
                raise NotImplementedError
            time_limit()
        if parent_node.is_dual:
            x0_and_space = {}
            for key, (x0, space) in parent_node._x0_and_space.items():
                new_x0 = x0 + matmul(space, b, time_limit=time_limit)
                new_space = matmul(space, A, time_limit=time_limit)
                x0_and_space[key] = (new_x0, new_space)
                time_limit()
            child_node = parent_node.__class__(x0_and_space)
            transform = cls(parent_node, child_node, A=A, b=b)
            return transform.child_node
        raise NotImplementedError

    @classmethod
    def apply_from_equations(cls, parent_node, eqs: Matrix, rhs: Matrix,
            to_child: bool=True, time_limit: Optional[Union[Callable, float]]=None) -> "SDPProblemBase":
        if to_child:
            # eqs * y - rhs == new_eqs * y' + new_nrhs
            eqs, new_nrhs = parent_node.propagate_affine_to_child(eqs, -rhs, recursive=True)
            parent_node = parent_node.get_last_child()
            rhs = -new_nrhs
        b, A = solve_undetermined_linear(eqs, rhs)
        return cls.apply_from_affine(parent_node, A, b,
            to_child=False, time_limit=time_limit)


class SDPMatrixTransform(SDPLinearTransform):
    """
    Standard matrix transformation class of SDP problems.

    Parameters
    -----------
    parent_node : SDPProblemBase
        The parent node of the transformation.
    child_node : SDPProblemBase
        The child node of the transformation.
    columnspace : Dict[Any, Matrix]
        The column spaces of the transformation. Should be a dictionary of
        (key, new_key): matrix. If the dict is None, the mapping is assumed to be the identity.
    nullspace : Dict[Any, Matrix]
        The null spaces of the transformation. Should be a dictionary of
        (key, new_key): matrix. If the dict is None, the mapping is assumed to be the identity.
    A : Matrix
        The matrix A in y = Ay' + b.
        If None, the linear transformation is assumed to be the identity y = y'.
    b : Matrix
        The matrix b in y = Ay' + b.
        If None, the linear transformation is assumed to be the identity y = y'.
    """
    _columnspace = None
    _nullspace = None
    _A = None
    _b = None
    def __init__(self, parent_node, child_node,
        columnspace: Optional[Dict[Any, Matrix]]=None,
        nullspace: Optional[Dict[Any, Matrix]]=None,
        A: Optional[Matrix]=None,
        b: Optional[Matrix]=None
    ):
        super().__init__(parent_node, child_node)
        self._columnspace = columnspace
        self._nullspace = nullspace
        self._A = A
        self._b = b
        if (A is None) ^ (b is None):
            raise ValueError("A and b should be both None or both not None.")
        if (columnspace is None) ^ (nullspace is None):
            raise ValueError("The columnspace and nullspace should be both None or both not None.")

    def propagate_to_parent(self):
        parent = self.parent_node
        child = self.child_node
        if child.y is None:
            # no solution
            return

        if self._A is None:
            # no linear transformation
            parent.y = child.y
        else:
            parent.y = self._A @ child.y + self._b

        if self._columnspace is None:
            # no columnspace transformation
            parent.S = child.S
            parent.decompositions = child.decompositions
        else:
            parent.register_y(parent.y, perturb=True)

    def propagate_nullspace_to_child(self, nullspace):
        # S = W * H * W.T sends S * V = 0 to H * (W.T * V) = 0.
        columnspace = self._columnspace or {}
        child_keys = set(self.child_node.keys())
        return {key: (matmul(columnspace[key].T, V) if key in columnspace else V)
                for key, V in nullspace.items() if key in child_keys}

    @classmethod
    def apply(cls, parent_node: "SDPProblemBase", columnspace: Dict[Any, Matrix]=None, nullspace: Dict[Any, Matrix]=None,
            to_child: bool=True, time_limit: Optional[Union[Callable, float]]=None) -> "SDPProblemBase":
        if parent_node.is_dual:
            return DualMatrixTransform.apply(parent_node, columnspace=columnspace, nullspace=nullspace,
                to_child=to_child, time_limit=time_limit)
        raise NotImplementedError


def _nullspace_equations(blocks, nullspace, time_limit=None):
    """Build E and f for `S(y)*V = 0`, written as `E*y = f`."""
    eqs, rhs = [], []
    for key, V in nullspace.items():
        x0, space = blocks[key]
        eqs.append(matmul_multiple(space.T, V, time_limit=time_limit).T)
        rhs.append(reshape(matmul(vec2mat(x0), V, time_limit=time_limit), (V.rows*V.cols, 1)))
        if time_limit is not None:
            time_limit()
    return Matrix.vstack(*eqs), -Matrix.vstack(*rhs)


def _extract_principal_block(x0: Matrix, space: Matrix, indices: List[int]) -> Tuple[Matrix, Matrix]:
    n = sqrtsize_of_mat(x0.rows)
    entries = [i*n+j for i in indices for j in indices]
    return x0.extract(entries, [0]), space.extract(entries, range(space.cols))


class DualMatrixTransform(SDPMatrixTransform):
    """Impose S(y) * V = 0 and restrict PSD to the orthogonal complement of V.

    For an exact nullspace constraint, let J be the free columns of the
    row reduction of V.T. Choose a basis W of ker(V.T) with W[J, :] = I.
    Then S(y) = W * S(y)[J, J] * W.T on the constrained affine space,
    so PSD is equivalent to PSD of the principal block S(y)[J, J].
    For a congruence child H = U.T * S(y) * U, use the lift
    W = U * (U.T * U)^(-1).
    """

    @classmethod
    def apply(cls, parent_node: "SDPProblemBase", columnspace: Dict[Any, Matrix]=None, nullspace: Dict[Any, Matrix]=None,
            to_child: bool=True, time_limit: Optional[Union[Callable, float]]=None) -> "SDPProblemBase":
        if not parent_node.is_dual:
            raise ValueError("The parent node should be dual.")
        if columnspace is None and nullspace is None:
            raise ValueError("At least one of columnspace and nullspace should be provided.")

        use_principal = columnspace is None
        time_limit = ArithmeticTimeout.make_checker(time_limit)
        if nullspace is None:
            columnspace = {key: (solve_columnspace(space), time_limit())[0] for key, space in columnspace.items()}
            nullspace = {key: (solve_nullspace(space), time_limit())[0] for key, space in columnspace.items()}
        else:
            nullspace = {key: (solve_columnspace(space), time_limit())[0] for key, space in nullspace.items()}

        if all(is_empty_matrix(V) for V in nullspace.values()):
            return parent_node.get_last_child() if to_child else parent_node
        if to_child:
            columnspace = None
            nullspace = parent_node.propagate_nullspace_to_child(nullspace, recursive=True)
            parent_node = parent_node.get_last_child()

        blocks = parent_node._x0_and_space
        nullspace = {key: V for key, V in nullspace.items()
                     if key in blocks and not is_empty_matrix(V) and V.is_zero_matrix is not True}
        if not nullspace:
            return parent_node

        extractions = {}
        if columnspace is None:
            columnspace = {}
            for key, V in nullspace.items():
                if use_principal and V._rep.domain.is_Exact:
                    extractions[key], columnspace[key] = principal_basis(V)
                else:
                    columnspace[key] = solve_nullspace(V)
                time_limit()
        else:
            columnspace = {key: columnspace[key] for key in nullspace}

        # Solve S(y) * V = 0 for the affine map y = A * z + b.
        eqs, rhs = _nullspace_equations(blocks, nullspace, time_limit)
        b, A = solve_undetermined_linear(eqs, rhs, time_limit=time_limit)

        new_x0_and_space = {}
        for key, (x0, space) in blocks.items():
            if key in extractions:
                # triggered when it uses a principal basis
                indices = extractions[key]
                if not indices:
                    continue
                x0, space = _extract_principal_block(x0, space, indices)
            elif key in columnspace:
                U = columnspace[key]
                if is_empty_matrix(U):
                    continue
                space = symmetric_bilinear_multiple(U, space.T, time_limit=time_limit).T
                x0 = symmetric_bilinear(U, x0, is_A_vec=True,
                    return_shape=(U.cols**2, 1), time_limit=time_limit)
                gram = matmul(U.T, U, time_limit=time_limit)
                columnspace[key] = matmul(U, gram.inv(), time_limit=time_limit)
            new_x0 = matadd(x0, matmul(space, b, time_limit=time_limit))
            new_space = matmul(space, A, time_limit=time_limit)
            new_x0_and_space[key] = (new_x0, new_space)
            time_limit()

        if not new_x0_and_space and A.cols:
            # Keep the remaining affine coordinates when every PSD block vanishes.
            key = next(iter(blocks))
            new_x0_and_space[key] = (Matrix.zeros(0, 1), Matrix.zeros(0, A.cols))
        child = parent_node.__class__(new_x0_and_space)
        return cls(parent_node, child, columnspace=columnspace, nullspace=nullspace, A=A, b=b).child_node


class SDPCongruence(SDPLinearTransform):
    """
    Congruence transformation of SDP problems.

    Assume the original problem is in the form of

        S1 = A_10 + y_1 * A_11 + ... + y_n * A_1n >> 0
        ...
        Sm = A_m0 + y_1 * A_m1 + ... + y_n * A_mn >> 0.

    Given invertible matrices P1, ..., Pn, it can be
    written as

        Pi.T * Si * Pi >> 0.
    """
    _basis = None
    _inv_basis = None
    def __init__(self, parent_node, child_node, basis: Dict[Any, Matrix], inv_basis: Optional[Dict[Any, Matrix]]=None):
        super().__init__(parent_node, child_node)
        self._basis = basis
        self._inv_basis = inv_basis if inv_basis is not None else \
            {key: basis.inv() for key, basis in basis.items()}

    def propagate_to_parent(self):
        parent, child = self.parent_node, self.child_node
        parent.y = child.y
        inv = self._inv_basis
        size = parent.size
        parent.S = {key: reshape(space @ parent.y + x0, (size[key], size[key]))
            for key, (x0, space) in parent._x0_and_space.items()}
        parent.decompositions = {key: ((matmul(U, inv[key]) if key in inv else U), S)
            for key, (U, S) in child.decompositions.items()}

    def propagate_nullspace_to_child(self, nullspace):
        b = self._inv_basis
        return {key: ((matmul(b[key], ns) if key in b else ns))
                 for key, ns in nullspace.items()}

    def propagate_affine_to_child(self, A, b):
        return A, b

    @classmethod
    def apply(cls, parent_node: "SDPProblemBase", basis: Dict[Any, Matrix], time_limit: Optional[Union[Callable, float]]=None):
        if parent_node.is_primal:
            raise NotImplementedError
        return DualSDPCongruence.apply(parent_node, basis=basis, time_limit=time_limit)


class DualSDPCongruence(SDPCongruence):
    @classmethod
    def apply(cls, parent_node: "SDPProblemBase", basis: Dict[Any, Matrix],
              time_limit: Optional[Union[Callable, float]]=None):
        new_x0_and_space = {}
        time_limit = ArithmeticTimeout.make_checker(time_limit)

        for key, (x0, space) in parent_node._x0_and_space.items():
            if key not in basis:
                new_x0_and_space[key] = (x0, space)
                continue
            new_space = symmetric_bilinear_multiple(basis[key], space.T, time_limit=time_limit).T
            new_x0 = symmetric_bilinear(basis[key], x0, is_A_vec = True,
                        return_shape=x0.shape, time_limit=time_limit)
            new_x0_and_space[key] = (new_x0, new_space)
        child = parent_node.__class__(new_x0_and_space, gens=parent_node.gens)
        transform = cls(parent_node, child, basis)
        return transform.child_node
