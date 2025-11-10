from typing import Tuple, List, Dict, Union, Optional, Generator

from sympy import Expr, Symbol

from .roots import Root

class RootList(list):
    """
    A list-like object that stores a list of roots.
    """
    symbols: Tuple[Symbol, ...]
    def __new__(cls, symbols, roots=None):
        roots = roots if roots is not None else []
        if any(len(r) != len(symbols) for r in roots):
            raise ValueError("All roots must have the same length as the number of symbols.")
        roots = [Root(root) for root in roots]
        return cls.new(tuple(symbols), roots)

    def __init__(self, symbols, roots=None):
        pass

    @property
    def roots(self):
        return list(self)

    @classmethod
    def new(cls, symbols: Tuple[Symbol, ...], roots: List[Root]):
        """Initialization without sanity checks."""
        obj = list.__new__(cls)
        obj.symbols = symbols
        list.__init__(obj, roots)
        return obj

    def copy(self) -> 'RootList':
        return RootList(self.symbols, list.copy(self))

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"RootList({self.symbols!r}, {super().__repr__()})"

    def __iter__(self) -> Generator[Root, None, None]:
        return super().__iter__()

    def __getitem__(self, i) -> Union[Root, 'RootList']:
        item = super().__getitem__(i)
        if isinstance(item, list):
            return RootList.new(self.symbols, item)
        return item

    def to_dicts(self) -> List[Dict[Symbol, Expr]]:
        """
        Convert the RootList to a list of dictionaries.

        Returns
        ----------
        List[Dict[Symbol, Expr]]
            A list of dictionaries, where each dictionary maps a symbol
            to the corresponding value in the root.

        Examples
        ----------
        >>> from sympy.abc import a, b, c
        >>> roots = RootList((a, b, c), [(1, 2, 3), (4, 5, 6)])
        >>> roots.to_dicts()
        [{a: 1, b: 2, c: 3}, {a: 4, b: 5, c: 6}]
        """
        return [dict(zip(self.symbols, root)) for root in self]

    def reorder(self, new_symbols: Tuple[Union[Symbol, int], ...]) -> 'RootList':
        """
        Reorder the symbols of the RootList.

        Parameters
        ----------
        new_symbols :
            The new symbols or indices to order the RootList by.

        Returns
        ----------
        RootList
            A new RootList with the reordered symbols.

        Examples
        ----------
        >>> from sympy.abc import a, b, c
        >>> from sympy import sqrt
        >>> roots = RootList((a, b, c), [(1, 2, 3), (4, 5, 6)])
        >>> roots.reorder([c, b, a])
        RootList((c, b, a), [(3, 2, 1), (6, 5, 4)])
        >>> roots.reorder([b, c])
        RootList((b, c), [(2, 3), (5, 6)])
        >>> roots.reorder((-1, 0, 1))
        RootList((c, a, b), [(3, 1, 2), (6, 4, 5)])
        """
        inds = [symbol if isinstance(symbol, int) else self.symbols.index(symbol)
                for symbol in new_symbols]
        new_symbols = tuple([self.symbols[i] for i in inds])
        return RootList.new(new_symbols, [root[inds] for root in self])

    def n(self, *args, **kwargs) -> 'RootList':
        """
        Return a new RootList with the numerical evaluation of the roots. See also sympy.n().

        Parameters
        ----------
        args :
            The arguments to pass to the `n` method of each root.
        kwargs :
            The keyword arguments to pass to the `n` method of each root.
        
        Returns
        ----------
        RootList
            A new RootList with the rounded roots.

        Examples
        ----------
        >>> from sympy.abc import a, b, c
        >>> from sympy import sqrt
        >>> roots = RootList((a,b,c), [Root((1, 2, 3))/4, Root((sqrt(2), 1, 1))]); roots
        RootList((a, b, c), [(1/4, 1/2, 3/4), (sqrt(2), 1, 1)])
        >>> roots.n(4)
        RootList((a, b, c), [(0.2500, 0.5000, 0.7500), (1.414, 1.000, 1.000)])
        """
        return RootList.new(self.symbols, [root.n(*args, **kwargs) for root in self])

    def transform(self, subs: Union[Dict[Symbol, Expr], List[Expr]],
            new_symbols: Optional[Tuple[Symbol, ...]] = None) -> 'RootList':
        """
        Transform the RootList by substituting symbols with expressions.

        Parameters
        ----------
        subs :
            A dictionary or list of expressions to substitute for the symbols in the RootList.
        new_symbols :
            The new symbols to use in the transformed RootList.
            If None, the keys of `subs` are used.

        Returns
        ----------
        RootList
            A new RootList with the transformed roots.

        Examples
        ----------
        >>> from sympy.abc import a, b, c, x, y, z, w
        >>> from sympy import sqrt
        >>> roots = RootList((a, b, c), [(1, 2, 3), (4, 5, 6)])
        >>> roots.transform({x: a-b, y: b-2*c, z: c/2, w: b**2/(a + 1)}, [w, x, y, z])
        RootList((w, x, y, z), [(2, -1, -4, 3/2), (5, -1, -7, 3)])
        >>> roots.transform([sqrt(a), sqrt(b), (a+b)/c], (x, y, z))
        RootList((x, y, z), [(1, sqrt(2), 1), (2, sqrt(5), 3/2)])
        """
        keys = new_symbols
        if isinstance(subs, list):
            if new_symbols is None:
                raise ValueError("new_symbols must be specified when subs is a list.")
            elif len(new_symbols) != len(subs):
                raise ValueError(f"new_symbols must have the same length as subs, but got {len(new_symbols)} != {len(subs)}.")
            keys = list(range(len(subs)))
        new_symbols = tuple(new_symbols if new_symbols is not None else subs.keys())
        return RootList.new(new_symbols,
            [root.transform(self.symbols, subs, keys) for root in self])
