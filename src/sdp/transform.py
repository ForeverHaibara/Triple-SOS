from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple, List

import sympy as sp

from .arithmetic import (
    solve_undetermined_linear,
    matmul, matmul_multiple, symmetric_bilinear, symmetric_bilinear_multiple
)
from .abstract import SDPProblemBase
from .utils import is_empty_matrix, Mat2Vec

class SDPTransformation():
    """
    Class that represents transformation between SDPProblems.
    """
    parent_node: 'SDPTransformMixin'
    child_node: 'SDPTransformMixin'
    def __init__(self, parent_node, *args, **kwargs):
        self.parent_node = parent_node
        self.child_node = self._init_child_node(*args, **kwargs)
        self.parent_node._transforms.append(self)
        self.child_node._transforms.append(self)
    def _init_child_node(self, *args, **kwargs) -> SDPProblemBase:
        return
    def is_parent(self, sdp):
        return sdp is self.parent_node
    def is_child(self, sdp):
        return sdp is self.child_node
    def propagate_to_parent(self, recursive: bool = True):
        raise NotImplementedError
    def propagate_to_child(self, recursive: bool = True):
        raise NotImplementedError
    @classmethod
    def _create_sdpproblem(cls, *args, **kwargs) -> SDPProblemBase:
        # Avoid circular import
        from .dual import SDPProblem
        return SDPProblem(*args, **kwargs)

class SDPIdentityTransform(SDPTransformation):
    """
    Identity transformation. It is used as an empty transformation.
    It can neither be found by SDPProblem.parents nor by SDPProblem.children.
    So its methods will not be called.

    This is class is used when the transformation is not needed,
    created by __new__ from other classes. And it should not be used directly.
    """
    # __slots__ = ('parent_node', 'child_node')
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

class SDPCopyTransform(SDPTransformation):
    """
    This is a transformation that copies the SDPProblem.    
    """
    def propagate_to_child(self, recursive: bool = True):
        parent, child = self.parent_node, self.child_node
        child.y = parent.y
        child.S = parent.S
        child.decompositions = parent.decompositions
        return child.propagate_to_child(recursive)

    def propagate_to_parent(self, recursive: bool = True):
        parent, child = self.parent_node, self.child_node
        parent.y = child.y
        parent.S = child.S
        parent.decompositions = child.decompositions
        return parent.propagate_to_parent(recursive)

class SDPMatrixTransform(SDPTransformation):
    def __new__(cls, parent_node, columnspace = None, nullspace = None):
        if parent_node.is_dual:
            return DualMatrixTransform(parent_node, columnspace, nullspace)
        if parent_node.is_primal:
            return PrimalMatrixTransform(parent_node, columnspace, nullspace)

        raise ValueError("The parent node should be either primal or dual.")
        # return super().__new__(cls)

    def __init__(self, parent_node, *args, **kwargs):
        super().__init__(parent_node, *args, **kwargs)

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
    def __new__(cls, parent_node, columnspace = None, nullspace = None):
        if columnspace is None and nullspace is None:
            raise ValueError("Columnspace and nullspace cannot both be None.")

        if nullspace is None:
            if all(is_empty_matrix(mat, check_all_zeros=True) for mat in columnspace.values()):
                return SDPIdentityTransform.__new__(SDPIdentityTransform, parent_node)
        elif columnspace is None:
            if all(is_empty_matrix(mat, check_all_zeros=True) for mat in nullspace.values()):
                return SDPIdentityTransform.__new__(SDPIdentityTransform, parent_node)
        return object.__new__(cls)


    def __init__(self, parent_node: SDPProblemBase, columnspace: Optional[Dict[str, sp.Matrix]] = None, nullspace: Optional[Dict[str, sp.Matrix]] = None):
        def _reg(X, key):
            m = sp.Matrix.hstack(*X.columnspace())
            if is_empty_matrix(m):
                return sp.zeros(X.shape[0], 0)
            return m
        def _perp(X, key):
            m = sp.Matrix.hstack(*X.T.nullspace())
            if is_empty_matrix(m):
                return sp.zeros(X.shape[0], 0)
            return m

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
        # from time import time
        # time0 = time()
        for key, (x0, space) in parent_node._x0_and_space.items():
            V = self._nullspace[key]
            if is_empty_matrix(V):
                continue
            eq_mat = matmul_multiple(space.T, V).T
            eq_list.append(eq_mat)

            Ai0 = Mat2Vec.vec2mat(x0)
            # new_x0 = list(Ai0 * V)
            new_x0 = list(matmul(Ai0, V))
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
            eq_mat = symmetric_bilinear_multiple(U, space.T).T
            new_space = eq_mat * trans_space
            # new_space = matmul(eq_mat, trans_space)

            # Ai0 = Mat2Vec.vec2mat(x0)
            # new_x0 = Mat2Vec.mat2vec(U.T * Ai0 * U) + eq_mat * trans_x0
            new_x0 = sp.Matrix(symmetric_bilinear(U, x0, is_A_vec = True, return_vec = True))
            new_x0 += eq_mat * trans_x0
            # new_x0 += matmul(eq_mat, trans_x0)

            new_x0_and_space[key] = (new_x0, new_space)

        # print('Used time', time() - time0)
        return self._create_sdpproblem(new_x0_and_space)

    def propagate_to_parent(self, recursive: bool = True):
        parent_node, child_node = self.parent_node, self.child_node
        if child_node.y is None:
            return parent_node
        y = self._trans_space * child_node.y + self._trans_x0
        parent_node.register_y(y, propagate_to_parent = recursive)
        return parent_node


class SDPRowMasking(DualMatrixTransform):
    """
    Mask several rows and cols of the matrix so that they are assumed
    to be zero. This is useful when we want to reduce the rank of the
    matrix or when there are zero diagonal entries.    
    """
    def __new__(cls, parent_node, masks: Dict[str, List[int]]):
        if all(len(mask) == 0 for mask in masks.values()):
            return SDPIdentityTransform.__new__(SDPIdentityTransform, parent_node)
        return object.__new__(cls)

    def __init__(self, parent_node: 'SDPTransformMixin', masks: Dict[str, List[int]]):
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

        return self._create_sdpproblem(new_x0_and_space)

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
    def constrain_zero_diagonals(cls, parent_node: 'SDPTransformMixin', recursive: bool = True) -> 'SDPRowMasking':
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
        if not parent_node.is_dual:
            raise ValueError("The parent node should be a dual problem for vector transformation.")
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
        return self._create_sdpproblem(x0_and_space)

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


class PrimalMatrixTransform(SDPMatrixTransform, SDPCopyTransform):
    """
    Assume the original problem to be S1 >= 0, ... Sn >= 0 where they satisfy certain constraints.
    The class imposes additional constraints on the matrices given columnspace / nullspace of Si.
    """
    
    def __new__(cls, parent_node, columnspace = None, nullspace = None):
        if columnspace is None and nullspace is None:
            raise ValueError("Columnspace and nullspace cannot both be None.")

        if nullspace is None:
            if all(is_empty_matrix(mat, check_all_zeros=True) for mat in columnspace.values()):
                return SDPIdentityTransform.__new__(SDPIdentityTransform, parent_node)
        elif columnspace is None:
            if all(is_empty_matrix(mat, check_all_zeros=True) for mat in nullspace.values()):
                return SDPIdentityTransform.__new__(SDPIdentityTransform, parent_node)
        return object.__new__(cls)


    def _init_child_node(self, columnspace: Optional[Dict[str, sp.Matrix]] = None, nullspace: Optional[Dict[str, sp.Matrix]] = None):
        parent_node = self.parent_node
        if columnspace is not None:
            raise NotImplementedError

        space = {k: v.copy() for k, v in parent_node._space.items()}
        eq_num = 0
        for key, mat in nullspace.items():
            eqs = []
            m = mat.shape[0]
            for i in range(mat.shape[1]):
                # consider every column of the nullspace
                if not any(mat[:, i]):
                    continue
                # each row inner product with the column should be zero
                for j in range(m):
                    eq = sp.zeros(1, m**2)
                    for k in range(m):
                        eq[j*m+k] += mat[k, i]
                        eq[k*m+j] += mat[k, i] # keep the matrix constraint symmetric
                    eqs.append(eq)
            if len(eqs):
                eqs = sp.Matrix.vstack(*eqs)
                eq_num += eqs.shape[0]
                space[key] = sp.Matrix.vstack(space[key], eqs)
                for other_key, other_mat in space.items():
                    if other_key == key:
                        continue
                    # align eqs with other_mat
                    space[other_key] = sp.Matrix.hstack(other_mat, sp.zeros(eqs.shape[0], other_mat.shape[1]))

        x0 = sp.Matrix.vstack(parent_node._x0, sp.zeros(eq_num, 1))
        return parent_node.__class__(space, x0)



class SDPTransformMixin(SDPProblemBase):
    def __init__(self, *args, **kwargs):
        # record the transformation dependencies
        self._transforms: List[SDPTransformation] = []
        super().__init__(*args, **kwargs)

    @property
    def parents(self) -> List['SDPTransformMixin']:
        """
        Return the parent nodes.
        """
        return [transform.parent_node for transform in self._transforms if transform.is_child(self)]

    @property
    def children(self) -> List['SDPTransformMixin']:
        """
        Return the child nodes.
        """
        return [transform.child_node for transform in self._transforms if transform.is_parent(self)]

    def get_last_child(self) -> 'SDPTransformMixin':
        """
        Get the last child node of the current node recursively.
        """
        children = self.children
        if len(children):
            return children[-1].get_last_child()
        return self

    def common_transform(self, other: SDPProblemBase) -> SDPTransformation:
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

    def propagate_to_child(self, recursive: bool = True):
        """
        Propagate the result to the child node.
        """
        for transform in self._transforms:
            if transform.is_parent(self):
                transform.propagate_to_child(recursive = recursive)

    
    @abstractmethod
    def constrain_subspace(self, columnspaces: Dict[str, sp.Matrix], to_child: bool = False) -> SDPProblemBase:
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

    @abstractmethod
    def constrain_nullspace(self, nullspaces: Dict[str, sp.Matrix], to_child: bool = False) -> SDPProblemBase:
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
            the constrain to the current node. Defaults to False. This is only
            supported when the child is the result of a chain of matrix transformations.

        Returns
        ----------
        SDPProblemBase
            The new SDP problem.

        Raises
        ----------
        ValueError
            If there is no solution to the linear system Si = Qi * Mi * Qi.T,
            then it raises an error.
        """

    @abstractmethod
    def constrain_symmetry(self) -> SDPProblemBase:
        """
        Constrain the solution to be symmetric. This is useful to reduce
        the degree of freedom when the given symbolic matrix is not symmetric.
        """


class DualTransformMixin(SDPTransformMixin):
    def _get_zero_diagonals(self) -> Dict[str, List[int]]:
        return SDPRowMasking._get_zero_diagonals(self._x0_and_space)

    def constrain_subspace(self, columnspaces: Dict[str, sp.Matrix], to_child: bool = False) -> SDPProblemBase:
        if to_child:
            raise NotImplementedError("Constraining to child node is not implemented yet.")
        transform = SDPMatrixTransform(self, columnspace=columnspaces)
        return transform.child_node

    def constrain_nullspace(self, nullspaces: Dict[str, sp.Matrix], to_child: bool = False) -> SDPProblemBase:
        sdp = self
        nullspaces = self._standardize_mat_dict(nullspaces)
        if to_child:
            while len(sdp.children):
                sdp2 = sdp.children[-1]
                transform = sdp.common_transform(sdp2)
                if not isinstance(transform, DualMatrixTransform):
                    raise ValueError("Transformations should be a chain of DualMatrixTransform when to_child is True.")
                nullspaces = {key: transform.columnspace[key].T * nullspaces[key] for key in sdp2.keys()}
                sdp = sdp2
        transform = SDPMatrixTransform(sdp, nullspace=nullspaces)
        return transform.child_node

    def constrain_symmetry(self) -> SDPProblemBase:
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

    def constrain_equal_entries(self, entry_tuples: Dict[str, List[Tuple[Tuple[int, int], Tuple[int, int]]]]) -> SDPProblemBase:
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

    def constrain_zero_diagonals(self, recursive: bool = True) -> SDPProblemBase:
        """
        If a diagonal of the positive semidefinite matrix is zero,
        then the corresponding row must be all zeros. This function
        constrains the solution to satisfy this condition.
        """
        return SDPRowMasking.constrain_zero_diagonals(self, recursive = recursive).child_node


class PrimalTransformMixin(SDPTransformMixin):
    def constrain_subspace(self, columnspaces: Dict[str, sp.Matrix], to_child: bool = False) -> SDPProblemBase:
        raise NotImplementedError("Constraint to subspace is not implemented for primal problems.")

    def constrain_nullspace(self, nullspaces: Dict[str, sp.Matrix], to_child: bool = False) -> SDPProblemBase:
        sdp = self
        # nullspaces = self._standardize_mat_dict(nullspaces)
        if to_child:
            while len(sdp.children):
                sdp2 = sdp.children[-1]
                transform = sdp.common_transform(sdp2)
                if not isinstance(transform, PrimalMatrixTransform):
                    raise ValueError("Transformations should be a chain of PrimalMatrixTransform when to_child is True.")
                sdp = sdp2
        transform = SDPMatrixTransform(sdp, nullspace=nullspaces)
        return transform.child_node

    def constrain_symmetry(self) -> SDPProblemBase:
        raise NotImplementedError("Symmetry constraint needs not to be applied to primal problems.")