# from math import prod
try:
    from math import prod # python 3.8
except ImportError:
    from functools import reduce
    prod = lambda lst: reduce(lambda x, y: x * y, lst, 1)

from typing import List, Tuple

import sympy as sp
import numpy as np
# from sympy.plotting.experimental_lambdify import vectorized_lambdify

from ..utils import Root, generate_monoms

class GridPoly():
    """
    A grid to store value information of a 3-var polynomial. The GridPoly class should be
    created by function `GridRender.render`.
    """
    def __init__(self, poly = None, size = 60, grid_coor = None, grid_value = None, grid_color = None):
        self.poly = poly
        self.size = size
        self.grid_coor = grid_coor
        self.grid_value = grid_value
        self.grid_color = grid_color

    def save_heatmap(self, path=None, dpi=None, backgroundcolor=211):
        """
        Save a heatmap to the given path. And return the numpy array of the heatmap.
        Only supports 3-var polynomials.

        Parameters
        ----------
        path : str
            The path to save the heatmap.
        dpi : int
            The dpi of the saved image.
        backgroundcolor : int
            The background color of the heatmap.
        """
        n = self.size
        x = np.full((n+1,n+1,3), backgroundcolor, dtype='uint8')
        grid_color = self.grid_color
        # get_color = lambda r,j: grid_color[r*(2*n+3-r)//2+j]
        get_color = lambda r,j: grid_color[(r+j+2)*(r+j+1)//2-j-1]
        for i in range(n+1):
            t = i * 15 // 26   # i * 15/26 ~ i / sqrt(3)
            r = t * 2         # row of grid triangle = i / (sqrt(3)/2)
            if r > n:          # n+1-r <= 0
                break
            for j in range(n+1-r):  # j < n+1-r
                # row r, entry j: a^(n-r-j)*b^j*c^r
                color = get_color(r,j)
                x[i,j+t,0] = color[0]
                x[i,j+t,1] = color[1]
                x[i,j+t,2] = color[2]

        if path is not None:
            import matplotlib.pyplot as plt
            plt.imshow(x,interpolation='nearest')
            plt.axis('off')
            plt.savefig(path, dpi=dpi, bbox_inches ='tight')
            plt.close()

        return x

    def save_coeffs(self, path, dpi=500, fontsize=20):
        """
        Save the coefficient triangle (as an image) to path.

        Parameters
        ----------
        path : str
            The path to save the coefficient triangle.
        dpi : int
            The dpi of the saved image.
        fontsize : int
            The fontsize of the saved image.
        """

        import matplotlib.pyplot as plt

        coeffs = self.poly.coeffs()
        monoms = self.poly.monoms()
        monoms.append((-1,-1,0))  # tail flag

        maxlen = 1
        for coeff in coeffs:
            maxlen = max(maxlen,len(f'{round(float(coeff),4)}'))

        distance = max(maxlen + maxlen % 2 + 1, 8)
        # print(maxlen,distance)
        n = (self.poly).total_degree()
        strings = [((distance + 1) // 2 * i) * ' ' for i in range(n+1)]

        t = 0
        for i in range(n+1):
            for j in range(i+1):
                if monoms[t][0] == n - i and monoms[t][1] == i - j:
                    if isinstance(coeffs[t], sp.core.numbers.Float):
                        txt = f'{round(float(coeffs[t]),4)}'
                    else:
                        txt = f'{coeffs[t].p}' + (f'/{coeffs[t].q}' if coeffs[t].q != 1 else '')
                        if len(txt) > distance:
                            txt = f'{round(float(coeffs[t]),4)}'
                    t += 1
                else:
                    txt = '0'
                    
                strings[j] += ' ' * (max(0, distance - len(txt))) + txt
        monoms.pop()

        for i in range(len(strings[0])):
            if strings[0][i] != ' ':
                break
        strings[0] += i * ' '
        string = '\n\n'.join(strings)

        # set the figure small enough
        # even though the text cannot be display as a whole in the window
        # it will be saved correctly by setting bbox_inches = 'tight'
        plt.figure(figsize=(0.3,0.3))
        plt.text(-0.3,0.9,string, fontsize=fontsize, fontfamily='Times New Roman')
        plt.xlim(0,6)
        plt.ylim(0,1)
        plt.axis('off')
        plt.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close()

        return string


    def local_minima(self, filter_nontrivial: bool = True, cyc: bool = True) -> List[Root]:
        """
        Search for local minima. Currently only 3-var polynomials are supported.

        Parameters
        ----------
        filter_nontrivial : bool
            Whether to remove the trivial extrema.
        cyc : bool
            Whether reduce the roots by the cyclic symmetry.

        Returns
        ----------
        extrema : List[Root]
            A list of local minima.
        """
        if len(self.poly.gens) != 3:
            # raise NotImplementedError('Only 3-var polynomials are supported.')
            return [Root((1,)*len(self.poly.gens))] if not filter_nontrivial else []

        grid_coor, grid_value = self.grid_coor, self.grid_value
        # grid_coor[k] = (i,j) stands for the value  f(n-i-j, i, j)
        # (grid_size + 1) * (grid_size + 2) // 2 = len(grid_coor)
        n = round((2 * len(grid_coor) + .25) ** .5 - 1.5)
        grid_dict = dict(zip(grid_coor, grid_value))

        extrema = []

        if not cyc:
            iterator = zip(grid_coor, grid_value)
        else:
            # trunc = (2*n + 3 - n // 3) * (n // 3) // 2
            trunc = (2*n//3) * (2*n//3 + 1) // 2
            iterator = zip(grid_coor[:trunc], grid_value[:trunc])

        def _get_local_checker(cyc = True):
            def _basic_checker(i, j, k, v):
                if i == 0 or j == 0 or k == 0:
                    return False
                for coor in ((i,j-1,k+1),(i+1,j-1,k),(i-1,j,k+1),(i-1,j+1,k),(i+1,j,k-1),(i,j+1,k-1)):
                    if v >= grid_dict[coor]:
                        return False
                return True
            if cyc:
                def _cyc_checker(i, j, k, v):
                    # without loss of generality we may assume i = max(i,j,k)
                    # need to be locally convex
                    if i < j or i < k:
                        return False
                    return _basic_checker(i, j, k, v)
                return _cyc_checker
            return _basic_checker

        checker = _get_local_checker(cyc)
        for (i, j, k), v in iterator:
            if checker(i, j, k, v):
                if cyc:
                    extrema.append(Root((j,k,i)))
                else:
                    extrema.append(Root((i,j,k)))

        if filter_nontrivial:
            extrema = [r for r in extrema if r.is_nontrivial]
        return extrema

def _grid_init_precal(size: int, degree_limit: int) -> List[List[int]]:
    """
    Pre-calculate stores the powers of integer
    precal[i][j] = i ** j    where i <= grid_size = 60 ,   j <= deglim = 18
    bound = 60 ** 18 = 60^18 = 1.0156e+032
    """
    grid_precal = []
    for i in range(size + 1):
        grid_precal.append( [1, i] )
        for _ in range(degree_limit - 1):
            grid_precal[i].append(grid_precal[i][-1] * i)
    return grid_precal
    

class GridRender():
    """
    Class to generate the grid of a homogeneous polynomial. See details in `GridRender.render`.
    """
    color_level = 12
    size_limit = 60
    degree_limit = 18

    # initialize the grid and preprocess some values
    # grid_coor_np_ = np.hstack([size - np.array(grid_coor), np.array(grid_coor)]).T
    grid_precal = _grid_init_precal(size_limit, degree_limit)

    @classmethod
    def grid_coor(cls, nvars: int = 3, size: int = 60) -> List[Tuple[int, int]]:
        return generate_monoms(nvars, size)[1]

    @classmethod
    def _render_grid_value(cls, poly: sp.Poly, size: int = 60, value_method: str = 'integer_lambdify') -> List[float]:
        """
        Render the grid by computing the values.
        """
        nvars = len(poly.gens)
        grid_coor = cls.grid_coor(nvars, size)
        grid_value = [0] * len(grid_coor)
        if size > cls.size_limit or poly.total_degree() > cls.degree_limit:
            return grid_value
        

        if value_method == 'integer':
            # integer arithmetic runs much faster than accuarate floating point arithmetic
            # example: computing 45**18   is around ten times faster than  (1./45)**18

            # convert sympy.core.number.Rational / Float / Integer to python int / float
            coeffs, monoms = poly.coeffs(), poly.monoms()
            for i in range(len(coeffs)):
                if coeffs[i].is_Integer:
                    coeffs[i] = int(coeffs[i])
                else:
                    coeffs[i] = float(coeffs[i])

            # pointer, but not deepcopy
            pc = cls.grid_precal

            for k in range(len(grid_coor)):
                base = [prod(pc[i][d] for i, d in zip(grid_coor[k], monom)) for monom in monoms]
                grid_value[k] = sum(coeff * b for coeff, b in zip(coeffs, base))

        elif value_method == 'integer_lambdify':
            f = sp.lambdify(poly.gens, poly.as_expr())
            grid_coor_np = np.array(grid_coor) / cls.degree_limit # normalize
            grid_value = f(*grid_coor_np.T)

        return grid_value


    @classmethod
    def _render_grid_color(cls, 
            poly: sp.Poly,
            size: int = 60,
            value_method: str = 'integer_lambdify',
            color_method: str = 'numpy',
            power: float = 1./3
        ) -> Tuple[List[float], List[Tuple[int, int, int]]]:
        """
        Render the grid by computing the values and setting the grid_val to rgba colors.

        Parameters
        ----------
        poly : sp.Poly
            The polynomial to be rendered.
        size : int
            The size of the grid. It is the number of sampling point in each axis.
        value_method : str
            See `_render_grid_value`.
        color_method : str
            'numpy' or 'integer'
            If 'numpy', use numpy to compute the colors.
            If 'integer', use integer arithmetic to compute the colors.
        power : float
            Map function values to colors by raising to this power.

        Returns
        ----------
        grid_value : List[float]
            A list of function values.
        grid_color : List[Tuple[int, int, int]]
            A list of RGB colors.
        """
        WHITE = (255,255,255)
        nvars = len(poly.gens)
        grid_coor = cls.grid_coor(nvars, size)
        grid_value = cls._render_grid_value(poly, size=size, value_method=value_method)

        if poly.total_degree() > cls.degree_limit:
            grid_color = [WHITE] * len(grid_coor)
            return grid_value, grid_color

        if color_method == 'integer':
            grid_color = [WHITE] * len(grid_coor)

            # preprocess the levels
            max_v, min_v = max(grid_value), min(grid_value)
            if max_v >= 0:
                max_levels = [(i / cls.color_level) ** (1 / power) * max_v for i in range(cls.color_level+1)]
            if min_v <= 0:
                min_levels = [(i / cls.color_level) ** (1 / power) * min_v for i in range(cls.color_level+1)]
            for k in range(len(grid_coor)):
                v = grid_value[k]
                if v > 0:
                    for i, level in enumerate(max_levels):
                        if v <= level:
                            v = 255 - (i-1)*255//(cls.color_level-1)
                            grid_color[k] = (255, v, 0)
                            break
                    else:
                        grid_color[k] = (255, 0, 0)
                elif v < 0:
                    for i, level in enumerate(min_levels):
                        if v >= level:
                            v = 255 - (i-1)*255//(cls.color_level-1)
                            grid_color[k] = (0, v, 255)
                            break
                    else:
                        grid_color[k] = (0, 0, 0)
                else: # v == 0:
                    grid_color[k] = WHITE

            return grid_value, grid_color

        elif color_method == 'numpy':
            value_numpy = np.array(grid_value)
            color_numpy = np.full((len(grid_coor), 3), 255, dtype = np.uint8)

            max_v, min_v = value_numpy.max(), value_numpy.min()
            if max_v > 0:
                positive = value_numpy > 0
                color_numpy[:, 2] = np.where(positive, 0, 255)
                levels = np.clip(value_numpy / max_v, 0, None) ** (power) * 255
                levels = levels.astype(np.uint8) ^ 255
                color_numpy[:, 1] = np.where(positive, levels, color_numpy[:, 1])

            if min_v < 0:
                negative = value_numpy < 0
                color_numpy[:, 0] = np.where(negative, 0, 255)
                levels = np.clip(value_numpy / min_v, 0, None) ** (power) * 255
                levels = levels.astype(np.uint8) ^ 255
                color_numpy[:, 1] = np.where(negative, levels, color_numpy[:, 1])

            grid_color = color_numpy.tolist()
            return grid_value, grid_color
            

    @classmethod
    def render(cls,
            poly: sp.Poly,
            size: int = 60,
            value_method: str = 'integer',
            color_method: str = 'numpy',
            with_color: bool = False,
            handle_error: bool = True,
        ) -> GridPoly:
        """
        Render the grid by computing the values and setting the grid_val to rgba colors.

        Parameters
        ----------
        poly : sp.Poly
            The polynomial to be rendered.
        size : int
            The size of the grid. It is the number of sampling point in each axis.
        value_method : str
            'integer' or 'integer_lambdify'
            If 'integer', use integer arithmetic to compute the values.
            If 'integer_lambdify', use integer arithmetic to compute the values, but use lambdify to speed up.
        color_method : str
            'numpy' or 'integer'
            If 'numpy', use numpy to compute the colors.
            If 'integer', use integer arithmetic to compute the colors.
        with_color : bool
            If True, return both grid_value and grid_color.
            If False, return only grid_value.
        handle_error : bool
            If True, return a zero grid when error occurs.
            If False, raise the error.

        Returns
        ----------
        grid : GridPoly
            A GridPoly object.
        """
        try:
            if with_color:
                grid_value, grid_color = cls._render_grid_color(poly, size=size, value_method=value_method, color_method=color_method)
            else:
                grid_value = cls._render_grid_value(poly, size=size, value_method = value_method)
                grid_color = None

            return GridPoly(
                poly,
                size = size,
                grid_coor = cls.grid_coor(len(poly.gens), size),
                grid_value = grid_value,
                grid_color = grid_color
            )
        except Exception as e:
            if handle_error:
                return cls.zero_grid()
            else:
                raise e

    @classmethod
    def zero_grid(cls, nvars: int = 3, size: int = 60) -> GridPoly:
        """
        Returns a zero grid.
        """
        grid_coor = cls.grid_coor(nvars, size)
        return GridPoly(
                sp.Poly(0, sp.symbols(f'x:{nvars}')),
                size = size,
                grid_coor = grid_coor,
                grid_value = [0] * len(grid_coor),
                grid_color = [(255,255,255,255)] * len(grid_coor)
            )
