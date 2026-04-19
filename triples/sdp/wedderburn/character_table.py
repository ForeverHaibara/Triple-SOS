from typing import List, Set, Optional

from sympy.combinatorics import Permutation, PermutationGroup

from .dixon import dixon_character_table
from .symmetric import murnaghan_nakayama_character_table

def character_table(G: PermutationGroup, cc: Optional[List[Set[Permutation]]]=None):
    """
    Computes the character table of a permutation group,
    the order of rows and columns are not fixed.
    """
    if G.is_symmetric:
        return murnaghan_nakayama_character_table(G.degree, cc=cc)

    if cc is None:
        cc = G.conjugacy_classes()
    return dixon_character_table(cc)
