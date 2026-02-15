from typing import Tuple, List, Dict, Any, Optional

from sympy.matrices.repmatrix import RepMatrix
from sympy import MutableDenseMatrix as Matrix
from .transform import SDPIdentityTransform
from .linear import SDPLinearTransform

from ..arithmetic import permute_matrix_rows

def _get_nonzero_entries_of_dual(self) -> Dict[Any, List[Tuple[int, int]]]:
    nonzero_entries = {}
    size = self.size
    for key, (x0, space) in self._x0_and_space.items():
        n = size[key]
        if isinstance(x0, RepMatrix) and isinstance(space, RepMatrix):
            x_rep, space_rep = x0._rep.rep.to_sdm(), space._rep.rep.to_sdm()
            nonzero_entries[key] = [(i,j) for i in range(n) for j in range(i+1,n)
                                 if (i*n+j in x_rep) or (i*n+j in space_rep)]
        else:
            nonzero_entries[key] = []
            for i in range(n):
                for j in range(i+1,n):
                    if x0[i*n+j] != 0 or any(space[i*n+j,:]):
                        nonzero_entries[key].append((i,j))
    return nonzero_entries


def get_block_structures(self) -> Dict[Any, List[List[int]]]:
    if self.is_primal:
        raise NotImplementedError
    nonzero_entries = _get_nonzero_entries_of_dual(self)
    all_blocks = {}

    size = self.size
    for key, nz in nonzero_entries.items():
        n = size[key]
        if n == 0:
            all_blocks[key] = [[]]
            continue

        ufs = {i: i for i in range(n)}

        def find(ufs, i):
            if ufs[i] != i:
                ufs[i] = find(ufs, ufs[i])
            return ufs[i]
        def union(ufs, i, j):
            ufs[find(ufs, i)] = find(ufs, j)

        for i, j in nz:
            union(ufs, i, j)

        blocks = {}
        for i in range(n):
            if find(ufs, i) == i:
                blocks[i] = []
        for i in range(n):
            blocks[find(ufs, i)].append(i)
        blocks = list(blocks.values())
        all_blocks[key] = blocks
    return all_blocks


class SDPBlockDiagonalization(SDPLinearTransform):
    _blocks = None
    _child_keys = None
    _A = None
    _b = None
    def __init__(self, parent_node, child_node,
        blocks: Dict[Any, List[int]],
        child_keys: Optional[List[Any]]=None,
        A: Optional[Matrix]=None,
        b: Optional[Matrix]=None
    ):
        super().__init__(parent_node, child_node, A, b)
        self._blocks = blocks
        self._child_keys = child_keys

    def propagate_to_parent(self):
        parent, child = self.parent_node, self.child_node
        A, b = self.get_y_transform_from_child()
        if A is None and b is None:
            y = child.y
        else:
            y = A @ child.y + b
        parent.y = y
        parent.S = child.S
        parent.register_y(parent.y, perturb=True)

    @classmethod
    def apply_block_structures(cls, parent_node):
        bs = get_block_structures(parent_node)
        if all(len(blocks) == 1 for blocks in bs.values()):
            return SDPIdentityTransform(parent_node, parent_node)

    @classmethod
    def apply(cls, parent_node, blocks: Optional[Dict[Any, List[int]]]=None):
        if blocks is None:
            blocks = get_block_structures(parent_node)
        else:
            blocks = blocks.copy()
            for key, size in parent_node.size:
                if key not in blocks:
                    blocks[key] = [list(range(size))]
                elif sum(len(b) for b in blocks[key]) != size:
                    raise ValueError(f"Invalid blocks for {key}: {blocks[key]}")
        if all(len(blocks) == 1 for blocks in blocks.values()):
            return SDPIdentityTransform(parent_node, parent_node).child_node

        keys = set(parent_node._x0_and_space.keys())

        size = parent_node.size
        new_x0_and_space = {}
        new_keys = {}
        for key, (x0, space) in parent_node._x0_and_space.items():
            block = blocks[key]
            m = size[key]
            new_keys[key] = []
            for b in block:
                counter = 0
                new_key = (key, counter)
                while new_key in keys:
                    counter += 1
                    new_key = (key, counter)
                keys.add(new_key)
                new_keys[key].append(new_key)

                rows = [i*m + j for i in b for j in b]
                new_x0 = permute_matrix_rows(x0, rows)
                new_space = permute_matrix_rows(space, rows)
                new_x0_and_space[new_key] = (new_x0, new_space)

        child = parent_node.__class__(new_x0_and_space, gens=parent_node.gens)
        transform = cls(parent_node, child, blocks, new_keys)
        return transform.child_node
