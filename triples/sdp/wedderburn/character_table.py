from typing import List, Set, Optional

from sympy.combinatorics import Permutation, PermutationGroup

from .dixon import dixon_character_table

def character_table(G: PermutationGroup, cc: Optional[List[Set[Permutation]]]=None):
    # TODO: implement other logics
    cc = cc if cc is not None else G.conjugacy_classes()
    return dixon_character_table(cc)
