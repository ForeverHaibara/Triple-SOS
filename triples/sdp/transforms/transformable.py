from typing import List, Tuple, Dict, Optional, Union, Callable, Any

from sympy.matrices import MutableDenseMatrix as Matrix
from sympy import Symbol

from .transform import SDPTransformation
from .linear import SDPLinearTransform, SDPMatrixTransform, SDPCongruence
from .parametric import SDPDeparametrization
from .polytope import get_zero_diagonals, SDPRowExtraction
from .diagonalize import get_block_structures, SDPBlockDiagonalization
from ..abstract import SDPProblemBase
from ..arithmetic import sqrtsize_of_mat



def _propagate_args_to_last(self, next_node, func, recursive, *args):
    if next_node == 'child':
        get_next = lambda x: x.child_node
        check_transform = lambda x, self: x.is_parent(self)
    elif next_node == 'parent':
        get_next = lambda x: x.parent_node
        check_transform = lambda x, self: x.is_child(self)
    else:
        raise ValueError("The next_node should be either 'child' or 'parent'.")

    start = True
    while recursive or start:
        start = False
        transform = None
        for _transform in self._transforms[::-1]:
            if check_transform(_transform, self) and not (get_next(_transform) is self):
                transform = _transform
                break
        if transform is None:
            break
        args = func(transform, *args)
        self = get_next(transform)
    return args


class TransformableProblem(SDPProblemBase):
    def __init__(self, *args, **kwargs):
        # record the transformation dependencies
        self._transforms: List[SDPTransformation] = []
        super().__init__(*args, **kwargs)

    @property
    def parents(self) -> List['TransformableProblem']:
        """
        Return the parent nodes.
        """
        return [transform.parent_node for transform in self._transforms if transform.is_child(self)]

    @property
    def children(self) -> List['TransformableProblem']:
        """
        Return the child nodes.
        """
        return [transform.child_node for transform in self._transforms if transform.is_parent(self)]

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

    def get_last_child(self) -> SDPProblemBase:
        """
        Get the last child node of the current node recursively.
        """
        children = self.children
        if not children:
            return self
        return children[-1].get_last_child()

    def clear_parents(self):
        """
        Clear all the parent nodes.
        """
        self._transforms = [_ for _ in self._transforms if not _.is_child(self)]

    def clear_children(self):
        """
        Clear all the child nodes.
        """
        self._transforms = [_ for _ in self._transforms if not _.is_parent(self)]

    def common_transform(self, other: SDPProblemBase) -> SDPTransformation:
        """
        Return the common transformation (linkage) between two SDP problems.
        """
        for transform in self._transforms:
            if transform.is_parent(self) and transform.is_child(other):
                return transform
            elif transform.is_parent(other) and transform.is_child(self):
                return transform

    def propagate_to_parent(self, recursive: bool = True):
        """
        Propagate the result to the parent node.
        """
        for transform in self._transforms:
            if transform.is_child(self):
                transform.propagate_to_parent()
                if recursive:
                    transform.parent_node.propagate_to_parent(recursive=recursive)

    def propagate_to_child(self, recursive: bool = True):
        """
        Propagate the result to the child node.
        """
        for transform in self._transforms:
            if transform.is_parent(self):
                transform.propagate_to_child(recursive=recursive)
                if recursive:
                    transform.child_node.propagate_to_child(recursive=recursive)

    def propagate_nullspace_to_child(self, nullspace: Dict[Any, Matrix], recursive: bool=False) -> Dict[Any, Matrix]:
        func = lambda transform, nullspace: (transform.propagate_nullspace_to_child(nullspace),)
        args = _propagate_args_to_last(self, 'child', func, recursive, nullspace)
        return args[0]

    def propagate_affine_to_child(self, A: Matrix, b: Matrix, recursive: bool=False) -> Tuple[Matrix, Matrix]:
        func = lambda transform, A, b: transform.propagate_affine_to_child(A, b)
        args = _propagate_args_to_last(self, 'child', func, recursive, A, b)
        return args

    def constrain_columnspace(self, columnspace: Dict[Any, Matrix], to_child: bool=False,
            time_limit: Optional[Union[Callable, float]]=None) -> 'TransformableProblem':
        """
        Constrain the columnspace of the SDP problem.
        """
        return SDPMatrixTransform.apply(self, columnspace=columnspace, to_child=to_child, time_limit=time_limit)

    def constrain_nullspace(self, nullspace: Dict[Any, Matrix], to_child: bool=False,
            time_limit: Optional[Union[Callable, float]]=None) -> 'TransformableProblem':
        """
        Constrain the nullspace of the SDP problem.
        """
        return SDPMatrixTransform.apply(self, nullspace=nullspace, to_child=to_child, time_limit=time_limit)

    def constrain_congruence(self, basis: Dict[Any, Matrix],
            time_limit: Optional[Union[Callable, float]]=None) -> 'TransformableProblem':
        """
        Constrain the congruence of the SDP problem.
        """
        return SDPCongruence.apply(self, basis=basis, time_limit=time_limit)

    def get_zero_diagonals(self) -> Dict[Any, List[int]]:
        return get_zero_diagonals(self)

    def get_block_structures(self) -> Dict[Any, List[List[int]]]:
        return get_block_structures(self)

    def constrain_zero_diagonals(self, extractions: Optional[Dict[Any, List[int]]]=None, masks: Optional[Dict[Any, List[int]]]=None,
            time_limit: Optional[Union[Callable, float]]=None) -> 'TransformableProblem':
        return SDPRowExtraction.apply(self, extractions=extractions, masks=masks, time_limit=time_limit)

    def constrain_block_structures(self, blocks: Optional[Dict[Any, List[int]]]=None):
        return SDPBlockDiagonalization.apply(self, blocks=blocks)

    def deparametrize(self, symbols: Optional[List[Symbol]]=None) -> 'TransformableProblem':
        return SDPDeparametrization.apply(self, symbols=symbols)


class TransformableDual(TransformableProblem):
    def constrain_symmetry(self) -> SDPProblemBase:
        # first solve for the nullspace the y should lie in
        eqs = []
        rhs = []
        size = self.size
        for key, (x0, space) in self._x0_and_space.items():
            n = size[key]
            for i in range(1, n):
                for j in range(i):
                    eqs.append(space[i*n+j, :] - space[j*n+i, :])
                    rhs.append(x0[j*n+i] - x0[i*n+j])
        if len(eqs) == 0:
            return self
        eqs = Matrix.vstack(*eqs)
        rhs = Matrix(rhs)
        return SDPLinearTransform.apply_from_equations(self, eqs, rhs)

class TransformablePrimal(TransformableProblem):
    def constrain_symmetry(self) -> SDPProblemBase:
        for key, m in self.size.items():
            space = self._space[key]
            n = space.shape[0]
            for i in range(m):
                for j in range(i):
                    k1, k2 = i*m+j, j*m+i
                    for row in range(n):
                        if space[row, k1] != space[row, k2]:
                            mid = (space[row, k1] + space[row, k2]) / 2
                            space[row, k1] = mid
                            space[row, k2] = mid
        return self
