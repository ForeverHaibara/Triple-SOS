from typing import List, Tuple, Dict, Any

from sympy.matrices import MutableDenseMatrix as Matrix

from .transform import SDPTransformation
from .linear import SDPMatrixTransform
from .polytope import get_zero_diagonals, SDPRowExtraction
from ..abstract import SDPProblemBase
from ..arithmetic import sqrtsize_of_mat

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
                transform.propagate_to_parent(recursive=recursive)

    def propagate_to_child(self, recursive: bool = True):
        """
        Propagate the result to the child node.
        """
        for transform in self._transforms:
            if transform.is_parent(self):
                transform.propagate_to_child(recursive=recursive)


    def constrain_columnspace(self, columnspace: Dict[Any, Matrix], to_child: bool=True):
        """
        Constrain the columnspace of the SDP problem.
        """
        return SDPMatrixTransform.apply(self, columnspace=columnspace, to_child=to_child)

    def constrain_nullspace(self, nullspace: Dict[Any, Matrix], to_child: bool=True):
        """
        Constrain the nullspace of the SDP problem.
        """
        return SDPMatrixTransform.apply(self, nullspace=nullspace, to_child=to_child)

    def get_zero_diagonals(self) -> Dict[Any, List[int]]:
        return get_zero_diagonals(self)

    def constrain_zero_diagonals(self):
        return SDPRowExtraction.apply(self)

    def deparametrize(self):
        # TODO
        return self


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
        return SDPMatrixTransform.apply_from_equations(self, eqs, rhs)

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