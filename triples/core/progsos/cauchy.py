from numbers import Integral
from typing import Any, Dict, List, Tuple

from sympy import Matrix, Symbol

from ...sdp import SDPProblem


_Affine = Tuple[Any, Dict[int, Any]]


def _as_vector(value: Any, name: str) -> List[Any]:
    """Convert a row or column matrix to a list."""
    value = Matrix(value)
    if value.cols == 1:
        return [value[i, 0] for i in range(value.rows)]
    if value.rows == 1:
        return [value[0, i] for i in range(value.cols)]
    raise ValueError("%s must be a row or column matrix" % name)


def norm_cone_prog_problem(
    As: List[Matrix],
    weights: List[Matrix],
    affines: List[Matrix],
    ps: List[int]
) -> SDPProblem:
    """
    Model ``<weight, (A @ x)**p> <= <affine, x>**p`` as an SDP.

    Powers in ``(A @ x)**p`` are entrywise.  The original variables are
    generated automatically and are placed before all auxiliary variables in
    ``SDPProblem.gens``.
    """
    if len(As) != len(weights) or len(As) != len(affines) or len(As) != len(ps):
        raise ValueError("As, weights, affines, and ps must have the same length")
    if len(As) == 0:
        return SDPProblem({})

    As = [Matrix(A) for A in As]
    weights = [Matrix(weight) for weight in weights]
    affines = [Matrix(affine) for affine in affines]

    nvars = As[0].cols
    used_names = set()
    for value in As + weights + affines:
        used_names.update(symbol.name for symbol in value.free_symbols)

    gens = []
    blocks = []
    variables = 0

    def new_symbol(prefix: str) -> Symbol:
        index = 0
        while True:
            symbol = Symbol("%s_{%d}" % (prefix, index))
            if symbol.name not in used_names:
                used_names.add(symbol.name)
                return symbol
            index += 1

    def new_variable(prefix: str) -> _Affine:
        nonlocal variables
        symbol = new_symbol(prefix)
        gens.append(symbol)
        index = variables
        variables += 1
        return 0, {index: 1}

    # Reserve the original variables first, as required by the modeling API.
    original = []
    for _ in range(nvars):
        symbol = new_symbol("x")
        gens.append(symbol)
        original.append((0, {variables: 1}))
        variables += 1

    def add(left: _Affine, right: _Affine) -> _Affine:
        constant = left[0] + right[0]
        coefficients = dict(left[1])
        for index, coefficient in right[1].items():
            coefficients[index] = coefficients.get(index, 0) + coefficient
            if coefficients[index] == 0:
                del coefficients[index]
        return constant, coefficients

    def scale(coefficient: Any, value: _Affine) -> _Affine:
        if coefficient == 0:
            return 0, {}
        return value[0] * coefficient, {
            index: coefficient * value_ for index, value_ in value[1].items()
        }

    def negate(value: _Affine) -> _Affine:
        return scale(-1, value)

    def linear_form(coefficients: List[Any]) -> _Affine:
        value = (0, {})
        for coefficient, variable in zip(coefficients, original):
            value = add(value, scale(coefficient, variable))
        return value

    def add_scalar_constraint(value: _Affine) -> None:
        blocks.append((value,))

    def add_geometric_mean_constraint(left: _Affine, node: _Affine, right: _Affine) -> None:
        # [[left, node], [node, right]] >> 0 is node**2 <= left * right.
        blocks.append((left, node, right))

    def add_power_constraint(value: _Affine, scale_variable: _Affine,
                             base: _Affine, power: int) -> None:
        leaf_count = 1
        while leaf_count < power:
            leaf_count *= 2

        target = new_variable("t")
        add_scalar_constraint(target)
        add_scalar_constraint(add(target, negate(value)))
        if power % 2 == 0:
            add_scalar_constraint(add(target, value))

        # The three entries count s, v, and the target t in a dyadic
        # completion of the weights (1 / p, (p - 1) / p).
        root = (1, power - 1, leaf_count - power)
        leaves = (scale_variable, base, target)
        nodes = {root: target}

        def split_counts(counts: Tuple[int, ...]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
            if max(counts) == leaf_count:
                return (), ()
            child1 = [0] * len(counts)
            child2 = [2 * count for count in counts]
            bit = leaf_count
            while bit:
                for index, count in enumerate(child2):
                    if count >= bit:
                        child2[index] -= bit
                        child1[index] += bit
                    if sum(child1) == leaf_count:
                        return tuple(child1), tuple(child2)
                bit //= 2
            raise ValueError("Unable to decompose the power cone")

        todo = [root]
        for counts in todo:
            children = split_counts(counts)
            if not children[0]:
                continue
            child1, child2 = children
            child_nodes = []
            for child in children:
                if child not in nodes:
                    if max(child) == leaf_count:
                        nodes[child] = leaves[child.index(leaf_count)]
                    else:
                        nodes[child] = new_variable("z")
                child_nodes.append(nodes[child])
                if child not in todo:
                    todo.append(child)
            add_geometric_mean_constraint(child_nodes[0], nodes[counts], child_nodes[1])

    for A, weight, affine, power in zip(As, weights, affines, ps):
        if not isinstance(power, Integral) or isinstance(power, bool) or power < 1:
            raise ValueError("Each p must be an integer greater than or equal to 1")
        power = int(power)
        if A.cols != nvars:
            raise ValueError("All A matrices must have the same number of columns")

        weight_vector = _as_vector(weight, "weight")
        affine_vector = _as_vector(affine, "affine")
        if len(weight_vector) != A.rows:
            raise ValueError("The length of weight must equal the number of rows of A")
        if len(affine_vector) != nvars:
            raise ValueError("The length of affine must equal the number of columns of A")

        values = [linear_form([A[i, j] for j in range(nvars)]) for i in range(A.rows)]
        base = linear_form(affine_vector)

        if power == 1:
            constraint = negate(base)
            for coefficient, value in zip(weight_vector, values):
                constraint = add(constraint, scale(coefficient, value))
            add_scalar_constraint(negate(constraint))
            continue

        add_scalar_constraint(base)
        weighted_scale = (0, {})
        for coefficient, value in zip(weight_vector, values):
            # A zero weight does not need an epigraph variable or a cone.
            if coefficient == 0:
                continue
            scale_variable = new_variable("s")
            add_power_constraint(value, scale_variable, base, power)
            weighted_scale = add(weighted_scale, scale(coefficient, scale_variable))

        add_scalar_constraint(add(base, negate(weighted_scale)))

    def row(value: _Affine) -> List[Any]:
        return [value[1].get(index, 0) for index in range(variables)]

    x0_and_space = []
    for block in blocks:
        if len(block) == 1:
            value = block[0]
            x0_and_space.append((
                Matrix([value[0]]),
                Matrix([row(value)])
            ))
        else:
            left, node, right = block
            x0_and_space.append((
                Matrix([left[0], node[0], node[0], right[0]]),
                Matrix([row(left), row(node), row(node), row(right)])
            ))

    return SDPProblem(x0_and_space, gens=gens)
