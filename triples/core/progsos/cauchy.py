from typing import Any, Dict, List, Tuple

from sympy import Matrix

from ...sdp import SDPProblem


_Affine = Tuple[Any, Dict[int, Any]]


def _add(lhs: _Affine, rhs: _Affine) -> _Affine:
    constant = lhs[0] + rhs[0]
    coefficients = dict(lhs[1])
    for index, coefficient in rhs[1].items():
        coefficients[index] = coefficients.get(index, 0) + coefficient
        if coefficients[index] == 0:
            del coefficients[index]
    return constant, coefficients


def _scale(coefficient: Any, value: _Affine) -> _Affine:
    if coefficient == 0:
        return 0, {}
    return value[0] * coefficient, {
        index: coefficient * value_ for index, value_ in value[1].items()
    }


def _linear_form(coefficients: List[Any], variables: List[_Affine]) -> _Affine:
    value = (0, {})
    for coefficient, variable in zip(coefficients, variables):
        value = _add(value, _scale(coefficient, variable))
    return value


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


class _SDPBuilder:
    """Collect affine scalar and rotated second-order cone constraints."""

    def __init__(self, nvars: int):
        self.variables = [(0, {index: 1}) for index in range(nvars)]
        self._dof = nvars
        self._blocks = []

    def new_variable(self) -> _Affine:
        variable = (0, {self._dof: 1})
        self._dof += 1
        return variable

    def add_linear(self, value: _Affine) -> None:
        self._blocks.append((value,))

    def add_rotated_soc(self, left: _Affine, middle: _Affine, right: _Affine) -> None:
        # [[left, middle], [middle, right]] >> 0.
        self._blocks.append((left, middle, right))

    def matrices(self) -> List[Tuple[Matrix, Matrix]]:
        matrices = []
        for block in self._blocks:
            if len(block) == 1:
                value = block[0]
                matrices.append((
                    Matrix([value[0]]),
                    Matrix([[value[1].get(index, 0) for index in range(self._dof)]])
                ))
            else:
                left, middle, right = block
                matrices.append((
                    Matrix([left[0], middle[0], middle[0], right[0]]),
                    Matrix([
                        [left[1].get(index, 0) for index in range(self._dof)],
                        [middle[1].get(index, 0) for index in range(self._dof)],
                        [middle[1].get(index, 0) for index in range(self._dof)],
                        [right[1].get(index, 0) for index in range(self._dof)],
                    ])
                ))
        return matrices

    def build(self) -> SDPProblem:
        return SDPProblem(self.matrices())


def _add_power_cone(
    builder: _SDPBuilder,
    value: _Affine,
    scale: _Affine,
    base: _Affine,
    power: int,
) -> None:
    """Add ``value**power <= scale * base**(power - 1)``."""
    total = 1
    while total < power:
        total *= 2

    if power == 2:
        builder.add_rotated_soc(scale, value, base)
        return

    target = builder.new_variable()
    builder.add_linear(target)
    builder.add_linear(_add(target, _scale(-1, value)))
    if power % 2 == 0:
        builder.add_linear(_add(target, value))

    # The dyadic weights are (1 / p, (p - 1) / p, (q - p) / q),
    # where q is the next power of two. The last weight pads the tree.
    root = (1, power - 1, total - power)
    leaves = (scale, base, target)
    nodes = {root: target}
    pending = [root]

    while pending:
        counts = pending.pop()
        children = _split_counts(counts, total)
        if not children:
            continue

        child_nodes = []
        for child in children:
            if child not in nodes:
                if max(child) == total:
                    nodes[child] = leaves[child.index(total)]
                else:
                    nodes[child] = builder.new_variable()
                    pending.append(child)
            child_nodes.append(nodes[child])

        builder.add_rotated_soc(child_nodes[0], nodes[counts], child_nodes[1])


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

    nvars = As[0].cols
    if any(A.cols != nvars for A in As):
        raise ValueError("All A matrices must have the same number of columns")

    builder = _SDPBuilder(nvars)
    for A, weight, affine, power in zip(As, weights, affines, ps):
        if power < 1:
            raise ValueError("Each p must be an integer greater than or equal to 1")
        power = int(power)

        if weight.shape != (A.rows, 1):
            raise ValueError("The shape of weight must be (A.rows, 1)")
        if affine.shape != (nvars, 1):
            raise ValueError("The shape of affine must be (A.cols, 1)")

        values = [
            _linear_form([A[i, j] for j in range(nvars)], builder.variables)
            for i in range(A.rows)
        ]
        base = _linear_form(list(affine), builder.variables)

        if power == 1:
            lhs = (0, {})
            for coefficient, value in zip(weight, values):
                lhs = _add(lhs, _scale(coefficient, value))
            builder.add_linear(_add(base, _scale(-1, lhs)))
            continue

        builder.add_linear(base)
        weighted_scale = (0, {})
        for coefficient, value in zip(weight, values):
            if coefficient == 0:
                continue
            scale = builder.new_variable()
            _add_power_cone(builder, value, scale, base, power)
            weighted_scale = _add(weighted_scale, _scale(coefficient, scale))
        builder.add_linear(_add(base, _scale(-1, weighted_scale)))

    return builder.build()
