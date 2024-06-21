from typing import List, Tuple, Union

import sympy as sp
import numpy as np

from ..utils.text_process import wrap_desmos

def show_dets(M: sp.Matrix):
    """
    Display the determinants of the leading principal submatrices of a matrix
    in a Desmos graph.

    Parameters
    ----------
    M : sympy.Matrix
        The matrix whose determinants are to be displayed.    
    """
    from IPython.display import display, HTML
    syms = list(M.free_symbols)
    assert len(syms) <= 2, "The matrix must have no more than 2 free symbols."
    syms = sorted(syms, key = lambda x: x.name)
    syms = dict(zip(syms, sp.symbols('x y')))
    det = []
    for i in range(1, 1+(M.shape[0])):
        value = (sp.GreaterThan(M[:i,:i].det().factor().subs(syms), 0))
        if value is sp.true:
            value = sp.S(1)
        det.append(value)
    txt = wrap_desmos(det)
    display(HTML(txt))


def plot_contour(
        f: sp.Expr,
        xrange: Tuple[Union[int, float], Union[int, float]] = (-5, 5),
        yrange: Tuple[Union[int, float], Union[int, float]] = (-5, 5),
        figsize: Tuple[int, int] = (10, 7),
        cmap: str = 'RdGy'
    ):
    """
    Plot the contour plot of a function of two variables.

    Parameters
    ----------
    f : sympy.Expr
        The function to be plotted.
    xrange : Tuple[Union[int, float], Union[int, float]]
        The range of the x-axis, by default (-5, 5.
    yrange : Tuple[Union[int, float], Union[int, float]]
        The range of the y-axis, by default (-5, 5).
    figsize : Tuple[int, int]
        The size of the plot, by default (10, 7).
    cmap : str
        The colormap to be used, by default 'RdGy'.
    """
    from matplotlib import pyplot as plt
    vars = list(f.free_symbols)
    assert len(vars) == 2, "The function must be a function of exactly two variables."
    f_lambdified = sp.lambdify(vars, f, "numpy")
    x_values = np.linspace(*xrange, 400)
    y_values = np.linspace(*yrange, 400)
    X, Y = np.meshgrid(x_values, y_values)
    Z = f_lambdified(X, Y)
    plt.figure(figsize=figsize)
    plt.imshow(Z, extent=list(xrange)+list(yrange), origin='lower', cmap=cmap, alpha=0.5)
    plt.colorbar()
    contours = plt.contour(X, Y, Z, colors='black', levels=20)
    plt.clabel(contours, inline=True, fontsize=8)
    plt.xlabel(vars[0]) and plt.ylabel(vars[1])
    plt.show()


def plot_f(
        f: sp.Expr,
        gens: List[sp.Symbol],
        xrange: Tuple[Union[int, float], Union[int, float]] = (-5, 5),
        yrange: Tuple[Union[int, float], Union[int, float]] = (-5, 5),
        n: int = 100,
        figsize: Tuple[int, int] = (10, 7),
        cmap: str = 'coolwarm'
    ):
    """
    Plot the function f of two variables.

    Parameters
    ----------
    f : sympy.Expr
        The function to be plotted.
    gens : List[sp.Symbol]
        The list of the symbols in the function.
    xrange : Tuple[Union[int, float], Union[int, float]]
        The range of the x-axis, by default (-5, 5.
    yrange : Tuple[Union[int, float], Union[int, float]]
        The range of the y-axis, by default (-5, 5).
    n : int
        The number of points to be plotted for each axis, by default 100.
    figsize : Tuple[int, int]
        The size of the plot, by default (10, 7).
    cmap : str
        The colormap to be used, by default 'RdGy'.
    """
    from matplotlib import pyplot as plt
    lambf = sp.lambdify(gens, f, 'numpy')
    x = np.linspace(*xrange,n)
    y = np.linspace(*yrange,n)
    X, Y = np.meshgrid(x, y)
    Z = lambf(X,Y)
    plt.figure(figsize=figsize)
    plt.imshow(Z, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap=cmap, alpha=0.5)
    plt.show()