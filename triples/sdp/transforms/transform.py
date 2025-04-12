from typing import Tuple, Dict, Any

from sympy.matrices import Matrix

from ..abstract import SDPProblemBase

class SDPTransformation:
    """
    SDP transformations represent the relationship between two SDP problems.
    When initialized, it registers itself to the parent and child SDP problems,
    so that it can be accessed by the parent and child SDP problems.

    Transformations record how the problems are related and can propagate
    the information (e.g., the solution) of one problem to the other.
    """
    parent_node: SDPProblemBase = None
    child_node: SDPProblemBase = None
    def __init__(self, parent_node, child_node, *args, **kwargs):
        self.parent_node = parent_node
        self.child_node = child_node
        self.parent_node._transforms.append(self)
        self.child_node._transforms.append(self)
    def is_parent(self, sdp) -> bool:
        """Check if the given SDP is the parent of the current transformation."""
        return sdp is self.parent_node
    def is_child(self, sdp) -> bool:
        """Check if the given SDP is the child of the current transformation."""
        return sdp is self.child_node

    def get_y_transform_from_child(self) -> Tuple[Matrix, Matrix]:
        """If `y = A@y' + b`, where `y` and `y'` are the solutions of the parent and child problems respectively,
        return `A` and `b`."""
        raise NotImplementedError

    def propagate_to_parent(self):
        """
        Propagate the solution of the child problem to the parent problem.
        """
        parent, child = self.parent_node, self.child_node
        A, b = self.get_y_transform_from_child()
        y = A @ child.y + b
        parent.register_y(y)

    def propagate_to_child(self):
        """
        Propagate the solution of the parent problem to the child problem.
        """
        raise NotImplementedError

    def propagate_nullspace_to_child(self, nullspace: Dict[Any, Matrix]) -> Dict[Any, Matrix]:
        raise NotImplementedError

    def propagate_affine_to_child(self, A, b) -> Tuple[Matrix, Matrix]:
        """
        Get `A'`, `b'` such that `A@y + b = A'@y' + b'`, where `y` and `y'`
        are the solutions of the parent and child problems respectively.
        """
        U, v = self.get_y_transform_from_child()
        return A@U, A@v + b


    @classmethod
    def apply(cls, parent_node, *args, **kwargs) -> SDPProblemBase:
        raise NotImplementedError

class SDPIdentityTransform(SDPTransformation):
    """
    Identity transformation. It is used as an empty transformation.
    It can neither be found by SDPProblem.parents nor by SDPProblem.children.
    So its methods will not be called.

    The class is used when the transformation is not needed,
    created by __new__ from other classes. And it should not be used directly.
    """
    def __new__(cls, parent_node, *args, **kwargs):
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
    @classmethod
    def apply(cls, parent_node) -> SDPProblemBase:
        return parent_node

