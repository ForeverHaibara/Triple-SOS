from ..abstract import SDPProblemBase

class SDPTransformation:
    parent_node: SDPProblemBase = None
    child_node: SDPProblemBase = None
    def __init__(self, parent_node, child_node, *args, **kwargs):
        self.parent_node = parent_node
        self.child_node = child_node
        self.parent_node._transforms.append(self)
        self.child_node._transforms.append(self)
    def is_parent(self, sdp) -> bool:
        return sdp is self.parent_node
    def is_child(self, sdp) -> bool:
        return sdp is self.child_node
    def _propagate_to_parent(self):
        raise NotImplementedError
    def _propagate_to_child(self):
        raise NotImplementedError
    def propagate_to_parent(self, recursive: bool = True):
        self._propagate_to_parent()
        if recursive:
            for parent in self.parent_node.parents:
                parent.propagate_to_parent(recursive=recursive)
    def propagate_to_child(self, recursive: bool = True):
        self._propagate_to_child()
        if recursive:
            for child in self.child_node.children:
                child.propagate_to_child(recursive=recursive)
    """
    Should also propagate objectives, constraints, e.g. linear operators,..,
    nullspace / columnspaces / ...
    Basic: Matrix Transformations and Vector Transformations.
    Composite transformations
    """


    def _propagate_nullspace_to_child(self, nullspace):
        raise NotImplementedError
    def propagate_nullspace_to_child(self, nullspace, recursive: bool = True):
        spaces = self._propagate_nullspace_to_child()
        latest_spaces = spaces
        if recursive:
            for child in self.child_node.children:
                latest_spaces = child.propagate_nullspace_to_child(spaces, recursive=recursive)
        return latest_spaces

    @classmethod
    def apply(cls, parent_node, *args, **kwargs):
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
    def apply(cls, parent_node):
        return parent_node

