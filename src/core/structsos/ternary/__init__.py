from .sparse  import sos_struct_sparse, sos_struct_heuristic
from .quadratic import sos_struct_quadratic, sos_struct_acyclic_quadratic
from .cubic   import sos_struct_cubic, sos_struct_acyclic_cubic
from .quartic import sos_struct_quartic, sos_struct_acyclic_quartic
from .quintic import sos_struct_quintic
from .sextic  import sos_struct_sextic
from .septic  import sos_struct_septic
from .octic   import sos_struct_octic
from .nonic   import sos_struct_nonic
from .acyclic import sos_struct_acyclic_sparse
from .nonhomogeneous import sos_struct_nonhomogeneous

from .solver import structural_sos_3vars, _structural_sos_3vars_cyclic, _structural_sos_3vars_acyclic
from ...solver import SS

__all__ = [
    'structural_sos_3vars'
]

_registry = [
    sos_struct_sparse, sos_struct_heuristic,
    sos_struct_quadratic, sos_struct_acyclic_quadratic,
    sos_struct_cubic, sos_struct_acyclic_cubic,
    sos_struct_quartic, sos_struct_acyclic_quartic,
    sos_struct_quintic,
    sos_struct_sextic,
    sos_struct_septic,
    sos_struct_octic,
    sos_struct_nonic,
    sos_struct_acyclic_sparse,
    sos_struct_nonhomogeneous,
    _structural_sos_3vars_cyclic, _structural_sos_3vars_acyclic, structural_sos_3vars
]
SS._register_solver('structsos', 'ternary', _registry)