from typing import List, Tuple, Dict, Optional, Any, TYPE_CHECKING
from collections import Counter

from sympy.matrices import MutableDenseMatrix as Matrix

from .linear import SDPLinearTransform, SDPMatrixTransform, SDPCongruence
from .parametric import SDPDeparametrization
from .polytope import get_zero_diagonals, SDPRowExtraction
from .diagonalize import get_block_structures, SDPBlockDiagonalization
from ..arithmetic import ArithmeticTimeout, is_empty_matrix
from ..abstract import SDPProblemBase

if TYPE_CHECKING:
    from sympy import Symbol

    from .transform import SDPTransformation
    from ..arithmetic import TimeLimit


def _format_sdp_summary(sdp: SDPProblemBase, short: int) -> str:
    if short <= 0:
        return str(sdp)
    if short == 1:
        sizes = sorted(sdp.size.values(), reverse=True)
        return f"<dof={sdp.dof}, size={sizes}>"
    counts = Counter(sdp.size.values())
    terms = [f"{size}²×{count}" if count > 1 else f"{size}²"
             for size, count in sorted(counts.items(), reverse=True)]
    return f"<dof={sdp.dof}, size=({' + '.join(terms)})>"


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
            if check_transform(_transform, self) and get_next(_transform) is not self:
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

    def print_graph(self, short: int = 0) -> None:
        """
        Print the dependency graph of the SDP problem.
        """
        short = int(short)
        _MAXLEN = 30
        _PAD = (_MAXLEN - 10) // 2
        print(" " * _PAD + "SDPProblem" + " " * _PAD + _format_sdp_summary(self, short))
        sdp = self

        def _formatter(a):
            filler_length = _MAXLEN - len(a) - 9
            filler = '-' * (filler_length // 2)
            filler2 = '-' * (filler_length - len(filler))
            return f"---{filler} {a} {filler2}-->"

        while len(sdp.children):
            sdp2 = sdp.children[-1]
            transform = sdp.common_transform(sdp2)
            sdp_str = _format_sdp_summary(sdp2, short)
            print(_formatter(transform.__class__.__name__) + " " + sdp_str)
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

    def common_transform(self, other: SDPProblemBase) -> 'SDPTransformation':
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

    def constrain_affine(self, A: Matrix, b: Matrix, to_child: bool=True,
            time_limit: "TimeLimit"=None) -> 'TransformableProblem':
        """
        Constrain the affine transformation of the SDP problem.
        """
        return SDPLinearTransform.apply_from_affine(self,
            A=A, b=b, to_child=to_child, time_limit=time_limit)

    def constrain_equations(self, eqs: Matrix, rhs: Matrix, to_child: bool=True,
            time_limit: "TimeLimit"=None) -> 'TransformableProblem':
        """
        Constrain the equations of the SDP problem.
        """
        return SDPLinearTransform.apply_from_equations(self,
            eqs=eqs, rhs=rhs, to_child=to_child, time_limit=time_limit)

    def constrain_columnspace(self, columnspace: Dict[Any, Matrix], to_child: bool=True,
            time_limit: "TimeLimit"=None) -> 'TransformableProblem':
        """
        Constrain the columnspace of the SDP problem.
        """
        return SDPMatrixTransform.apply(self,
            columnspace=columnspace, to_child=to_child, time_limit=time_limit)

    def constrain_nullspace(self, nullspace: Dict[Any, Matrix], to_child: bool=True,
            time_limit: "TimeLimit"=None) -> 'TransformableProblem':
        """
        Constrain the nullspace of the SDP problem.
        """
        return SDPMatrixTransform.apply(self,
            nullspace=nullspace, to_child=to_child, time_limit=time_limit)

    def constrain_congruence(self, basis: Dict[Any, Matrix],
            time_limit: "TimeLimit"=None) -> 'TransformableProblem':
        """
        Constrain the congruence of the SDP problem.
        """
        return SDPCongruence.apply(self, basis=basis, time_limit=time_limit)

    def get_zero_diagonals(self) -> Dict[Any, List[int]]:
        return get_zero_diagonals(self)

    def get_block_structures(self) -> Dict[Any, List[List[int]]]:
        return get_block_structures(self)

    def constrain_zero_diagonals(self, extractions: Optional[Dict[Any, List[int]]]=None, masks: Optional[Dict[Any, List[int]]]=None,
            time_limit: "TimeLimit"=None) -> 'TransformableProblem':
        return SDPRowExtraction.apply(self,
            extractions=extractions, masks=masks, time_limit=time_limit)

    def constrain_block_structures(self, blocks: Optional[Dict[Any, List[int]]]=None):
        return SDPBlockDiagonalization.apply(self, blocks=blocks)

    def deparametrize(self, symbols: Optional[List['Symbol']]=None) -> 'TransformableProblem':
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
        if is_empty_matrix(eqs, check_all_zeros=True) and \
                is_empty_matrix(rhs, check_all_zeros=True):
            return self
        return SDPLinearTransform.apply_from_equations(self, eqs, rhs)

    def constrain_coordinate_faces(self, level: int = 2, to_child: bool = True,
            time_limit: "TimeLimit" = None) -> 'TransformableDual':
        """Repeatedly reduce the PSD blocks using exact coordinate certificates.

        Level 1 searches for nonnegative combinations of the diagonal forms
        that vanish identically across the blocks. Level 2 also uses the
        forms in directions e_i + e_j and e_i - e_j. Level 0 disables
        the search. Start at the last child if to_child is True.
        """
        from ..facial import find_exposing_nullspace
        if level not in (0, 1, 2):
            raise ValueError("level must be 0, 1 or 2")
        node = self.get_last_child() if to_child else self
        if level == 0:
            return node

        check_time = ArithmeticTimeout.make_checker(time_limit)

        while node.dof > 0:
            check_time()
            nullspace = find_exposing_nullspace(node._x0_and_space,
                coordinate_level=level, time_limit=check_time)
            if not nullspace:
                return node
            node = node.constrain_nullspace(nullspace, to_child=False, time_limit=check_time)

        return node


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
