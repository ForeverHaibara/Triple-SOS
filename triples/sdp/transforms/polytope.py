from math import sqrt
from typing import Any, Dict, List, Tuple

from sympy.matrices import MutableDenseMatrix as Matrix
from sympy.polys.domains import ZZ
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.matrices.sdm import SDM

from .transform import SDPTransformation, SDPIdentityTransform
from .linear import SDPMatrixTransform
from ..arithmetic import sqrtsize_of_mat, solve_undetermined_linear

def complement(n: int, mask: List[int]) -> List[int]:
    """Get the complement of a mask."""
    mask = set(mask)
    return [i for i in range(n) if i not in mask]

def onehot(n: int, inds: int) -> Matrix:
    """Create a matrix where M[i, inds[i]] = 1 and others are 0."""
    sdm = {i: {j: ZZ.one} for i, j in enumerate(inds)}
    mat = Matrix._fromrep(DomainMatrix.from_rep(SDM(sdm, (n, len(inds)), ZZ)))
    return mat


def _get_zero_diagonals_of_dual(self) -> Dict[Any, List[int]]:
    zero_diagonals = {}
    size = self.size
    for key, (x0, space) in self._x0_and_space.items():
        n = size[key]
        zero_diagonals[key] = []
        for i in range(n):
            if x0[i*n+i] == 0 and not any(space[i*n+i,:]):
                zero_diagonals[key].append(i)
    return zero_diagonals

def _get_zero_diagonals_of_primal(self) -> Dict[Any, List[int]]:
    zero_diagonals = {key: set() for key in self.keys()}
    zero_eqs = set([i for i in range(self._x0.shape[0]) if self._x0[i] == 0])
    new_zero_found = True
    while new_zero_found:
        new_zero_found = False
        for eq in zero_eqs:
            # chances are that sum(diag entries) == 0
            eq_fail = False
            for key, m in self.size.items():
                zeros = zero_diagonals[key] # rows/cols that are zeros
                space = self._space[key]
                for i in range(m):
                    if i in zeros:
                        continue
                    for j in range(i):
                        if j in zeros:
                            continue
                        if space[eq, i*m+j] != 0: # nonzero off-diagonal constraint
                            eq_fail = True
                            break
                    if eq_fail:
                        break
                if eq_fail:
                    break
            else:
                eq_fail = False
                new_zeros = {key: set() for key in self.keys()}
                for key, m in self.size.items():
                    zeros = zero_diagonals[key]
                    space = self._space[key]
                    for i in range(m):
                        if i in zeros:
                            continue
                        if space[eq, i*m+i] < 0: # nonzero diagonal constraint
                            eq_fail = True
                            break
                        if space[eq, i*m+i] != 0:
                            new_zero_found = True
                            new_zeros[key].add(i)
                    if eq_fail:
                        new_zero_found = False
                        break
                if (not eq_fail) and new_zero_found:
                    for key in self.keys():
                        zero_diagonals[key] |= new_zeros[key]
                    zero_eqs.remove(eq) # has been processed
                    break
    return {key: list(zero_diagonals[key]) for key in self.keys()}

def get_zero_diagonals(self) -> Dict[Any, List[int]]:
    if self.is_dual:
        return _get_zero_diagonals_of_dual(self)
    elif self.is_primal:
        return _get_zero_diagonals_of_primal(self)


class SDPRowExtraction(SDPMatrixTransform):
    _extractions = None
    _A = None
    _b = None
    def __new__(cls, parent_node, child_node, extractions: Dict[Tuple[Any, Any], List[int]], A: Matrix=None, b: Matrix=None):
        size = parent_node.size
        identical = True
        for (key, new_key), extraction in extractions.items():
            if key not in size:
                raise ValueError(f'Key {key} not found in the parent node.')
            if key != new_key:
                identical = False
                break
            n = size[key]
            if len(extraction) != n:
                identical = False
                break
        if identical:
            # no need to create a new object
            return SDPIdentityTransform(parent_node)
        return super().__new__(cls, parent_node, child_node, A=A, b=b)

    def __init__(self, parent_node, child_node, extractions: Dict[Tuple[Any, Any], List[int]], A: Matrix=None, b: Matrix=None):
        SDPTransformation.__init__(self, parent_node, child_node, A=A, b=b)
        self._extractions = extractions
        self._A = A
        self._b = b

    @property
    def _columnspace(self):
        columnspace = {}
        size = self.parent_node.size
        for key, extraction in self._extractions.items():
            columnspace[key] = onehot(size[key], extraction)
        return columnspace

    @property
    def _nullspace(self):
        nullspace = {}
        size = self.parent_node.size
        for key, extraction in self._extractions.items():
            nullspace[key] = onehot(size[key], complement(size[key], extraction))
        return nullspace

    @classmethod
    def apply(cls, parent_node, extractions: Dict[Any, List[int]]=None, masks: Dict[Any, List[int]]=None):
        if parent_node.is_dual:
            return DualRowExtraction.apply(parent_node, extractions=extractions, masks=masks)
        elif parent_node.is_primal:
            raise NotImplementedError
        raise TypeError('Parent_node should be a SDPProblemBase object.')


class DualRowExtraction(SDPRowExtraction):
    @classmethod
    def apply(cls, parent_node, extractions: Dict[Any, List[int]]=None, masks: Dict[Any, List[int]]=None):
        if not parent_node.is_dual:
            raise TypeError('Parent_node should be a SDPProblem object.')
        if masks is None and extractions is None:
            raise ValueError('At least one of extractions or masks should be provided.')
        if extractions is None:
            extractions = {key: complement(n, masks.get(key, tuple())) for key, n in parent_node.size.items()}
        if masks is None:
            masks = {key: complement(n, extractions.get(key, tuple())) for key, n in parent_node.size.items()}

        def _get_new_params(x0_and_space, extractions, masks):
            eqs = []
            rhs = []
            nonzero_inds = {}
            for key, (x0, space) in x0_and_space.items():
                mask = set(masks[key])
                n = sqrtsize_of_mat(x0)
                rows = []
                for i in mask:
                    rows.extend(list(range(i*n, (i+1)*n)))
                eqs.append(space.extract(rows, list(range(space.shape[1]))))
                rhs.append(x0.extract(rows, [0]))

                unmask = extractions[key]
                nonzero_inds_ = []
                for i in unmask:
                    for j in unmask:
                        nonzero_inds_.append(i*n+j)
                nonzero_inds[key] = nonzero_inds_

            eqs = Matrix.vstack(*eqs)
            rhs = Matrix.vstack(*rhs)
            trans_x0, trans_space = solve_undetermined_linear(eqs, -rhs)

            new_x0_and_space = {}
            for key, (x0, space) in x0_and_space.items():
                space2 = space[nonzero_inds[key],:]
                new_x0 = x0[nonzero_inds[key],:] + space2 * trans_x0
                new_space = space2 * trans_space
                new_x0_and_space[key] = (new_x0, new_space)
            return new_x0_and_space, trans_space, trans_x0

        x0_and_space = parent_node._x0_and_space
        new_x0_and_space, A, b = _get_new_params(x0_and_space, extractions, masks)

        child_node = parent_node.__class__(new_x0_and_space)
        extractions = {(key, key): value for key, value in extractions.items()}
        return cls(parent_node, child_node, extractions=extractions, A=A, b=b)
