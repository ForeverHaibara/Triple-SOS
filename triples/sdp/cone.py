from typing import List, Tuple, Optional, TYPE_CHECKING

from sympy import MutableDenseMatrix as Matrix
from sympy import QQ
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.matrices.sdm import SDM

from .dual import SDPProblem

if TYPE_CHECKING:
    from sympy.polys.domains import Domain


class ConeSDPBuilder:
    """Helper class to build SDPProblem from cone constraints."""
    def __init__(self, nvars: int, size: int = 2147483647):
        self._nvars = nvars
        self._dof = nvars
        self._linear_blocks = []
        self._soc_blocks = []
        self._size = size

    def new_variable(self) -> int:
        """Reserve and return the index of one auxiliary variable."""
        variable = self._dof
        if self._dof >= self._size:
            raise MemoryError("Number of variables exceeds size")
        self._dof += 1
        return variable

    def _variable(self, i: int, domain: "Domain" = QQ) -> "SDM":
        return SDM.new({0: {i: domain.one}}, (1, self._size), domain)

    def add_linear(self, form: "SDM") -> None:
        """
        Add the homogeneous scalar constraint ``form @ y >= 0``.
        Stored as a row vector in the SDM format.
        """
        self._linear_blocks.append(form)

    def add_rotated_soc(self, left: "SDM", middle: "SDM", right: "SDM") -> None:
        """
        Add ``[[left, middle], [middle, right]] >> 0``.
        Stored as three row vectors in the SDM format.
        """
        self._soc_blocks.append((left, middle, right))

    def add_power_cone(
        self,
        value: "SDM",
        scale_index: int,
        base: "SDM",
        power: int
    ) -> None:
        """
        Add ``|value|**power <= scale * base**(power - 1)``.

        The constraint is represented by a dyadic binary tree of rotated
        second-order cones. The sign and domain constraints are added here so
        callers only need to provide the three homogeneous linear forms.
        """
        power = int(power)

        if power < 2:
            raise ValueError("The power of a power cone must be at least 2")

        if power == 2:
            self.add_rotated_soc(self._variable(scale_index), value, base)
            return

        total = 1
        while total < power:
            total *= 2

        abs_form = self._variable(self.new_variable())
        self.add_linear(abs_form - value)
        self.add_linear(abs_form + value)

        root = (1, power - 1, total - power)
        leaves = (
            self._variable(scale_index),
            base,
            abs_form,
        )
        nodes = {root: abs_form}
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

            # child[0]*child[1] >= nodes[counts]**2
            self.add_rotated_soc(child_forms[0], nodes[counts], child_forms[1])

    def add_pnorm_cone(
        self,
        A: Matrix,
        affine: Matrix,
        p: int,
        weight: Optional[Matrix] = None,
    ) -> None:
        """
        Add ``(<weight, |A @ x|**p>)^(1/p) <= <affine, x>``. If `weight`
        is not provided, then the weight is taken to be 1.

        All forms are homogeneous in the original variables. For ``p >=
        2``, one p-norm cone is added for every nonzero weight, together with
        their weighted epigraph sum. The linear case is handled directly.
        """
        if p < 1:
            raise ValueError("Each p must be an integer greater than or equal to 1")

        if weight is not None:
            ws = weight._rep.rep.to_sdm().transpose()
            if not ws:
                return
        else:
            dom = A._rep.rep.domain
            ws = SDM({0: {i: dom.one} for i in range(A.shape[0])},
                     (1, self._size), dom)

        Arep = A._rep.rep.to_sdm()
        aff = affine._rep.rep.to_sdm().transpose()
        aff = aff.new(dict(aff), (1, self._size), aff.domain)

        dom = Arep.domain.unify(aff.domain).unify(ws.domain)
        Arep = Arep.convert_to(dom)
        aff = aff.convert_to(dom)
        ws = ws.convert_to(dom).get(0, {})

        rows = [Arep.get(i, {}) for i in ws]
        rows = [Arep.new({0: row} if row else {}, (1, self._size), dom)
                    for row in rows]

        if p == 1:
            for ind, v in zip(ws, rows):
                aff -= v.mul(ws[ind])
            self.add_linear(aff)
            return

        affcopy = aff.copy()
        for ind, v in zip(ws, rows):
            scale = self.new_variable()
            self.add_power_cone(v, scale, affcopy, p)
            aff -= self._variable(scale, domain=dom).mul(ws[ind])

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
            sdm = sdm.new(dict(sdm), (1, self._dof), sdm.domain)
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

    def build(self) -> SDPProblem:
        """Build the SDPProblem from the constraints."""
        return SDPProblem(self.get_x0_and_space())
