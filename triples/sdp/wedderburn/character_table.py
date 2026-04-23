from typing import List, Set, Optional

from sympy.combinatorics import Permutation, PermutationGroup

from .dixon import dixon_character_table
from .symmetric import murnaghan_nakayama_character_table

def character_table(G: PermutationGroup, cc: Optional[List[Set[Permutation]]]=None):
    """
    Compute the character table of a permutation group.
    The order of rows and columns are not fixed.

    Parameters
    ----------
    G: PermutationGroup
        The permutation group.
    cc: Optional[List[Set[Permutation]]], optional
        The conjugacy classes of G.
        If provided, columns are sorted to match the conjugacy classes.

    Returns
    -------
    Matrix
        The character table of G.

    Examples
    --------
    >>> from sympy.combinatorics import SymmetricGroup, AlternatingGroup
    >>> character_table(SymmetricGroup(4)) # doctest: +SKIP
    Matrix([
    [1,  1,  1,  1,  1],
    [1, -1,  1,  1, -1],
    [2,  0, -1,  2,  0],
    [3, -1,  0, -1,  1],
    [3,  1,  0, -1, -1]])
    >>> character_table(AlternatingGroup(4)) # doctest: +SKIP
    Matrix([
    [1,          1,          1,  1],
    [1, -1 - zeta3,      zeta3,  1],
    [1,      zeta3, -1 - zeta3,  1],
    [3,          0,          0, -1]])
    """
    if G.is_symmetric:
        return murnaghan_nakayama_character_table(G.degree, cc=cc)

    if cc is None:
        cc = G.conjugacy_classes()
    return dixon_character_table(cc)
