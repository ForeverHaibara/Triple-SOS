from typing import List, Optional

from sympy import Matrix, Symbol

from .transform import SDPTransformation
from .linear import SDPLinearTransform, SDPMatrixTransform
from ..utils import decompose_matrix
from ..arithmetic import free_symbols_of_mat

class SDPDeparametrization(SDPLinearTransform):
    @classmethod
    def apply(cls, parent_node, *args, **kwargs):
        if parent_node.is_dual:
            return DualDeparametrization.apply(parent_node, *args, **kwargs)
        elif parent_node.is_primal:
            raise NotImplementedError
        raise TypeError('Parent_node should be a SDPProblemBase object.')

class DualDeparametrization(SDPDeparametrization):
    @classmethod
    def apply(cls, parent_node, symbols: Optional[List[Symbol]]=None):
        if parent_node.is_primal:
            raise TypeError('Problem should be a SDPProblemBase object.')

        if symbols is None:
            symbols = set()
            for key, (x0, space) in parent_node._x0_and_space.items():
                # if hasattr(x0, 'free_symbols'):
                #     symbols.update(x0.free_symbols)
                symbols.update(free_symbols_of_mat(x0))
        symbols = list(symbols)

        if len(symbols) == 0:
            return parent_node

        new_x0_and_space = {}
        for key, (x0, space) in parent_node._x0_and_space.items():
            if free_symbols_of_mat(x0): # is not empty
                x1, A, v = decompose_matrix(x0, symbols)
            else:
                x1 = x0
                A = Matrix.zeros(x0.shape[0], len(symbols))
            new_x0_and_space[key] = (x1, Matrix.hstack(space, A))

        child = parent_node.__class__(new_x0_and_space, gens=parent_node.gens + symbols)
        A = Matrix.eye(parent_node.dof, child.dof)
        b = Matrix.zeros(parent_node.dof, 1)
        return cls(parent_node, child, A=A, b=b).child_node
