from warnings import warn

from .monomials import *

warn("Importing basis_generator from triples.utils is deprecated. Please import from triples.utils.monomials instead.", DeprecationWarning,
     stacklevel=2)