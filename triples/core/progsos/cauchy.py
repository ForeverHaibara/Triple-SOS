from typing import List, Tuple, TYPE_CHECKING

from sympy import MutableDenseMatrix as Matrix
from sympy import QQ
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.matrices.sdm import SDM

from ...sdp import SDPProblem

if TYPE_CHECKING:
    from sympy.polys.domains import Domain


class _SDPBuilder:
    """Build homogeneous linear, rotated SOC, and power-cone constraints."""
    def __init__(self, nvars: int, size: int = 2147483647):
        self._nvars = nvars
        self._dof = nvars
        self._linear_blocks = []
        self._soc_blocks = []
        self._size = size

    def new_variable(self) -> int:
        """Reserve and return the index of one auxiliary variable."""
        variable = self._dof
        self._dof += 1
        if self._dof >= self._size:
            raise MemoryError("Number of variables exceeds size")
        return variable

    def _variable(self, i: int, domain: "Domain" = QQ) -> "SDM":
        return SDM.new({i: {0: domain.one}}, (self._size, 1), domain)

    def add_linear(self, form: "SDM") -> None:
        """Add the homogeneous scalar constraint ``form @ y >= 0``."""
        self._linear_blocks.append(form)

    def add_rotated_soc(self, left: "SDM", middle: "SDM", right: "SDM") -> None:
        """Add ``[[left, middle], [middle, right]] >> 0``."""
        self._soc_blocks.append((left, middle, right))

    def add_power_cone(
        self,
        value: "SDM",
        scale_index: int,
        base: "SDM",
        power: int
    ) -> None:
        """
        Add ``value**power <= scale * base**(power - 1)``.

        The constraint is represented by a dyadic binary tree of rotated
        second-order cones. The sign and domain constraints are added here so
        callers only need to provide the three homogeneous linear forms.
        """
        if power < 2:
            raise ValueError("The power of a power cone must be at least 2")

        if power == 2:
            self.add_rotated_soc(self._variable(scale_index), value, base)
            return

        total = 1
        while total < power:
            total *= 2

        target = self.new_variable()
        target_form = self._variable(target)
        self.add_linear(target_form)
        self.add_linear(target_form - value)
        if power % 2 == 0:
            self.add_linear(target_form + value)

        root = (1, power - 1, total - power)
        leaves = (
            self._variable(scale_index),
            base,
            target_form,
        )
        nodes = {root: target_form}
        pending = [root]

        while pending:
            counts = pending.pop()
            children = self._split_counts(counts, total)
            if not children:
                continue

            child_forms = []
            for child in children:
                if child not in nodes:
                    if max(child) == total:
                        nodes[child] = leaves[child.index(total)]
                    else:
                        nodes[child] = self._variable(self.new_variable())
                        pending.append(child)
                child_forms.append(nodes[child])

            self.add_rotated_soc(child_forms[0], nodes[counts], child_forms[1])

    def add_weighted_pnorm_cone(
        self,
        A: Matrix,
        weight: Matrix,
        affine: Matrix,
        power: int
    ) -> None:
        """
        Add ``<weight, (A @ x)**power> <= <affine, x>**power``.

        All forms are homogeneous in the original variables. For ``power >=
        2``, one power cone is added for every nonzero weight, together with
        their weighted epigraph sum. The linear case is handled directly.
        """
        if power < 1:
            raise ValueError("Each p must be an integer greater than or equal to 1")
        power = int(power)

        if weight.shape != (A.rows, 1):
            raise ValueError("The shape of weight must be (A.rows, 1)")
        if affine.shape != (self._nvars, 1):
            raise ValueError("The shape of affine must be (A.cols, 1)")

        Arep = A._rep.rep.to_sdm()
        values = [Arep.get(i, {}) for i in range(Arep.shape[0])]
        values = [Arep.new({0: row} if row else {},
                      (1, self._size), Arep.domain).transpose()
                      for row in values]

        sdm = affine._rep.rep.to_sdm()
        aff = sdm.new(dict(sdm), (self._size, 1), sdm.domain)

        if power == 1:
            for w, v in zip(weight, values):
                if w:
                    aff -= w * v
            self.add_linear(aff)
            return

        self.add_linear(aff.copy())
        scales = []
        for i, value in enumerate(values):
            if weight[i] == 0:
                continue
            scale = self.new_variable()
            self.add_power_cone(value, scale, aff, power)
            scales.append((weight[i], self._variable(scale)))
        for w, v in scales:
            aff -= w * v
        self.add_linear(aff)

    @staticmethod
    def _split_counts(counts: Tuple[int, ...], total: int) -> Tuple[Tuple[int, ...], ...]:
        """Split dyadic multiplicities into two equal-sized children."""
        if max(counts) == total:
            return ()

        child1 = [0] * len(counts)
        child2 = [2 * count for count in counts]
        bit = total
        while bit:
            for index, count in enumerate(child2):
                if count >= bit:
                    child2[index] -= bit
                    child1[index] += bit
                if sum(child1) == total:
                    return tuple(child1), tuple(child2)
            bit //= 2
        raise ValueError("Unable to decompose the power cone")

    def get_x0_and_space(self) -> List[Tuple[Matrix, Matrix]]:
        """Return the low-level ``(x0, space)`` blocks for ``SDPProblem``."""
        blocks = []
        def to_mat(sdm: "SDM") -> Matrix:
            sdm = sdm.new(dict(sdm), (self._dof, 1), sdm.domain).transpose()
            return Matrix._fromrep(DomainMatrix.from_rep(sdm))

        for form in self._linear_blocks:
            blocks.append((Matrix.zeros(1, 1), to_mat(form)))
        for left, middle, right in self._soc_blocks:
            l, m, r = to_mat(left), to_mat(middle), to_mat(right)
            blocks.append((
                Matrix.zeros(4, 1),
                Matrix.vstack(l, m, m, r)
            ))
        return blocks


def norm_cone_prog_problem(
    As: List[Matrix],
    weights: List[Matrix],
    affines: List[Matrix],
    ps: List[int]
) -> SDPProblem:
    """
    Model ``<weight, (A @ x)**p> <= <affine, x>**p`` as an SDP.

    Powers in ``(A @ x)**p`` are entrywise. The original variables are the
    first generators of the returned ``SDPProblem``.
    """
    if not len(As) == len(weights) == len(affines) == len(ps):
        raise ValueError("As, weights, affines, and ps must have the same length")
    if not As:
        return SDPProblem({})

    builder = _SDPBuilder(As[0].shape[1])
    for A, weight, affine, power in zip(As, weights, affines, ps):
        builder.add_weighted_pnorm_cone(A, weight, affine, power)

    return SDPProblem(builder.get_x0_and_space())
