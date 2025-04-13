"""
This module contains basic linear transformations of SDP problems.
"""

from typing import List, Tuple, Dict, Optional, Any

from sympy.matrices import MutableDenseMatrix as Matrix

from .transform import SDPTransformation, SDPProblemBase
from ..arithmetic import (
    is_empty_matrix, vec2mat,
    solve_undetermined_linear, solve_nullspace, solve_columnspace,
    matmul, matmul_multiple, symmetric_bilinear, symmetric_bilinear_multiple
)


class SDPMatrixTransform(SDPTransformation):
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
            b: Optional[Matrix]=None):
        super().__init__(parent_node, child_node)
        self._columnspace = columnspace
        self._nullspace = nullspace
        self._A = A
        self._b = b
        if (A is None) ^ (b is None):
            raise ValueError("A and b should be both None or both not None.")
        if (columnspace is None) ^ (nullspace is None):
            raise ValueError("The columnspace and nullspace should be both None or both not None.")

    def get_y_transform_from_child(self) -> Tuple[Matrix, Matrix]:
        return self._A, self._b

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
        columnspace = self._columnspace
        return {key: matmul(columnspace[key].T, nullspace[key]) for key in nullspace.keys()}


    @classmethod
    def apply(cls, parent_node, columnspace: Dict[Any, Matrix]=None, nullspace: Dict[Any, Matrix]=None, to_child: bool=True) -> SDPProblemBase:
        if parent_node.is_dual:
            return DualMatrixTransform.apply(parent_node, columnspace=columnspace, nullspace=nullspace, to_child=to_child)
        raise NotImplementedError

    @classmethod
    def apply_from_affine(cls, parent_node, A: Matrix, b: Matrix) -> SDPProblemBase:
        if parent_node.is_dual:
            return DualMatrixTransform.apply_from_affine(parent_node, A, b)
        raise NotImplementedError

    @classmethod
    def apply_from_equations(cls, parent_node, eqs: Matrix, rhs: Matrix) -> SDPProblemBase:
        b, A = solve_undetermined_linear(eqs, rhs)
        return cls.apply_from_affine(parent_node, A, b)


class DualMatrixTransform(SDPMatrixTransform):
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
    @classmethod
    def apply(cls, parent_node, columnspace: Dict[Any, Matrix]=None, nullspace: Dict[Any, Matrix]=None, to_child: bool=True) -> SDPProblemBase:
        if not parent_node.is_dual:
            raise ValueError("The parent node should be dual.")
        if columnspace is None and nullspace is None:
            raise ValueError("At least one of columnspace and nullspace should be provided.")
        if nullspace is None:
            columnspace = {key: solve_columnspace(space) for key, space in columnspace.items()}
            nullspace = {key: solve_nullspace(space) for key, space in columnspace.items()}
        else:
            nullspace = {key: solve_columnspace(space) for key, space in nullspace.items()}

        if all(is_empty_matrix(m) for m in nullspace.values()):
            return parent_node.get_last_child() if to_child else parent_node

        if to_child:
            columnspace = None
            nullspace = parent_node.propagate_nullspace_to_child(nullspace, recursive=True)
            parent_node = parent_node.get_last_child()
        if columnspace is None:
            columnspace = {key: solve_nullspace(space) for key, space in nullspace.items()}


        def _get_new_params(x0_and_space, columnspace, nullspace):
            # form the constraints of y by computing Sum(y_i * A_ij * Vi) = -A_i0 * Vi
            # TODO: faster fraction arithmetic
            # NOTE: the major bottleneck lies in:
            # 1. solve_undetermined_linear
            # 2. matmul_multiple

            eq_list = []
            x0_list = []
            # from time import time
            # time0 = time()
            for key, (x0, space) in x0_and_space.items():
                V = nullspace[key]
                if is_empty_matrix(V):
                    continue

                eq_mat = matmul_multiple(space.T, V).T
                eq_list.append(eq_mat)

                Ai0 = vec2mat(x0)
                # new_x0 = list(Ai0 * V)
                new_x0 = list(matmul(Ai0, V))
                x0_list.extend(new_x0)


            # eq * y + x0 = 0 => y = trans_x0 + trans_space * z
            eq_list = Matrix.vstack(*eq_list)
            x0_list = Matrix(x0_list)
            trans_x0, trans_space = solve_undetermined_linear(eq_list, -x0_list)

            # Sum(Ui' * Aij * Ui * (trans_x0 + trans_space * z)[j]) >> 0
            new_x0_and_space = {}
            for key, (x0, space) in x0_and_space.items():
                U = columnspace[key]
                if is_empty_matrix(U):
                    continue
                eq_mat = symmetric_bilinear_multiple(U, space.T).T
                new_space = eq_mat * trans_space
                # new_space = matmul(eq_mat, trans_space)

                # Ai0 = vec2mat(x0)
                # new_x0 = mat2vec(U.T * Ai0 * U) + eq_mat * trans_x0
                new_x0 = symmetric_bilinear(U, x0, is_A_vec = True, return_shape=(U.shape[1]**2, 1))
                new_x0 += matmul(eq_mat, trans_x0)
                # new_x0 += matmul(eq_mat, trans_x0)

                new_x0_and_space[key] = (new_x0, new_space)
            # print(f"Time: {time() - time0}")
            return new_x0_and_space, trans_space, trans_x0

        new_x0_and_space, A, b = _get_new_params(parent_node._x0_and_space, columnspace, nullspace)
        child_node = parent_node.__class__(new_x0_and_space)
        transform = cls(parent_node, child_node, columnspace=columnspace, nullspace=nullspace, A=A, b=b)
        return transform.child_node

    @classmethod
    def apply_from_affine(cls, parent_node, A: Matrix, b: Matrix) -> SDPProblemBase:
        x0_and_space = {}
        for key, (x0, space) in parent_node._x0_and_space.items():
            x0_ = x0 + matmul(space, b)
            space_ = matmul(space, A)
            x0_and_space[key] = (x0_, space_)
        child_node = parent_node.__class__(x0_and_space)
        transform = cls(parent_node, child_node, A=A, b=b)
        return transform.child_node