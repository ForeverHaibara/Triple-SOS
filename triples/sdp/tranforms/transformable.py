from typing import List, Tuple, Dict, Any

from sympy.matrices import MutableDenseMatrix as Matrix

from ..abstract import SDPProblemBase

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

    def propagate_to_child(self, recursive: bool = True) -> None:
       ...

    def propagate_to_parent(self, recursive: bool = True) -> None:
        ...