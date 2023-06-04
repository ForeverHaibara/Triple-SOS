import sympy as sp
import numpy as np

from ..polytools import deg

class GridPoly():
    def __init__(self, poly = None, size = 60, grid_coor = None, grid_value = None, grid_color = None):
        self.poly = poly
        self.size = size
        self.grid_coor = grid_coor
        self.grid_value = grid_value
        self.grid_color = grid_color

    def save_heatmap(self, path, dpi=None, backgroundcolor=211):
        """
        Save a heatmap to the given path.
        """
        import matplotlib.pyplot as plt

        n = self.size
        x = np.full((n+1,n+1,3), backgroundcolor, dtype='uint8')
        for i in range(n+1):
            t = i * 15 // 26   # i * 15/26 ~ i / sqrt(3)
            r = t * 2         # row of grid triangle = i / (sqrt(3)/2)
            if r > n:          # n+1-r <= 0
                break
            base = r*(2*n+3-r)//2   # (n+1) + n + ... + (n+2-r)
            for j in range(n+1-r):  # j < n+1-r
                x[i,j+t,0] = self.grid_color[base+j][0]
                x[i,j+t,1] = self.grid_color[base+j][1]
                x[i,j+t,2] = self.grid_color[base+j][2]
    
        plt.imshow(x,interpolation='nearest')
        plt.axis('off')
        plt.savefig(path, dpi=dpi, bbox_inches ='tight')
        plt.close()

    def save_coeffs(self, path, dpi=500, fontsize=20):
        """
        Save the coefficient triangle (as an image) to path.
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
        n = deg(self.poly)
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

        # set the figure small enough
        # even though the text cannot be display as a whole in the window
        # it will be saved correctly by setting bbox_inches = 'tight'
        plt.figure(figsize=(0.3,0.3))
        plt.text(-0.3,0.9,'\n\n'.join(strings), fontsize=fontsize, fontfamily='Times New Roman')
        plt.xlim(0,6)
        plt.ylim(0,1)
        plt.axis('off')
        plt.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close()

    def latex_coeffs(self, tabular=True):
        """
        Return the LaTeX format of the coefficient triangle.
        """
        n = deg(self.poly)
        emptyline = '\\\\ ' + '\\ &' * (n * 2) + '\\  \\\\ '
        strings = ['' for i in range(n+1)]
        
        if self.poly is not None:
            coeffs = self.poly.coeffs()
            monoms = self.poly.monoms()
        else:  # all coefficients are treated as zeros
            coeffs = []
            monoms = []
        monoms.append((-1,-1,0))  # tail flag
        t = 0
        for i in range(n+1):
            for j in range(i+1):
                if monoms[t][0] == n - i and monoms[t][1] == i - j:
                    txt = sp.latex(coeffs[t])
                    t += 1
                else:
                    txt = '0'
                strings[j] = strings[j] + '&\\ &' + txt if len(strings[j]) != 0 else txt
        monoms.pop()

        for i in range(n+1):
            strings[i] = '\\ &'*i + strings[i] + '\\ &'*i + '\\ '
        s = emptyline.join(strings)
        if tabular:
            s = '\\left[\\begin{matrix}\\begin{tabular}{' + 'c' * (n * 2 + 1) + '} ' + s
            s += ' \\end{tabular}\\end{matrix}\\right]'
        else:
            s = '\\left[\\begin{matrix} ' + s
            s += ' \\end{matrix}\\right]'

        return s



def _grid_init_coor(size):
    """
    Initialize the grid and preprocess some values.
    """
    # grid_coor[k] = (i,j) stands for the value  f(n-i-j, i, j)
    grid_coor = []
    for i in range(size + 1):
        for j in range(size + 1 - i):
            grid_coor.append((j,i))
    return grid_coor

def _grid_init_precal(size, degree_limit):
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
    color_level = 12
    size = 60
    degree_limit = 18

    # initialize the grid and preprocess some values
    grid_coor = _grid_init_coor(size)
    grid_precal = _grid_init_precal(size, degree_limit)

    _zero_grid = None

    @classmethod
    def _render_grid_value(cls, poly, value_method = 'integer'):
        """
        Render the grid by computing the values.
        """
        grid_value = [0] * len(cls.grid_coor)
        if deg(poly) > cls.degree_limit:
            return grid_value

        if value_method == 'integer':
            # integer arithmetic runs much faster than accuarate floating point arithmetic
            # example: computing 45**18   is around ten times faster than  (1./45)**18

            # convert sympy.core.number.Rational / Float / Integer to python int / float
            coeffs = poly.coeffs()
            for i in range(len(coeffs)):
                if int(coeffs[i]) == coeffs[i]:
                    coeffs[i] = int(coeffs[i])
                else:
                    coeffs[i] = float(coeffs[i])
            
            # pointer, but not deepcopy
            pc = cls.grid_precal

            for k in range(len(cls.grid_coor)):
                b , c = cls.grid_coor[k]
                a = cls.size - b - c
                v = 0
                for coeff, monom in zip(coeffs, poly.monoms()):
                    # coeff shall be the last to multiply, as it might be float while others int
                    v += pc[a][monom[0]] * pc[b][monom[1]] * pc[c][monom[2]] * coeff
                grid_value[k] = v

        else:
            pass
        return grid_value


    @classmethod
    def _render_grid_color(cls, poly, value_method = 'integer', color_method = 'numpy', power = 1./3):
        """
        Render the grid by computing the values and setting the grid_val to rgba colors.

        TODO:
        1. use numpy to accelerate
        2. continuous color levels?
        """
        grid_value = cls._render_grid_value(poly, value_method = value_method)

        if deg(poly) > cls.degree_limit:
            grid_color = [(255,255,255,255)] * len(cls.grid_coor)
            return grid_value, grid_color

        if color_method == 'integer':
            grid_color = [(255,255,255,255)] * len(cls.grid_coor)

            # preprocess the levels
            max_v, min_v = max(grid_value), min(grid_value)
            if max_v >= 0:
                max_levels = [(i / cls.color_level) ** (1 / power) * max_v for i in range(cls.color_level+1)]
            if min_v <= 0:
                min_levels = [(i / cls.color_level) ** (1 / power) * min_v for i in range(cls.color_level+1)]
            for k in range(len(cls.grid_coor)):
                v = grid_value[k]
                if v > 0:
                    for i, level in enumerate(max_levels):
                        if v <= level:
                            v = 255 - (i-1)*255//(cls.color_level-1)
                            grid_color[k] = (255, v, 0, 255)
                            break
                    else:
                        grid_color[k] = (255, 0, 0, 255)
                elif v < 0:
                    for i, level in enumerate(min_levels):
                        if v >= level:
                            v = 255 - (i-1)*255//(cls.color_level-1)
                            grid_color[k] = (0, v, 255, 255)
                            break
                    else:
                        grid_color[k] = (0, 0, 0, 255)
                else: # v == 0:
                    grid_color[k] = (255, 255, 255, 255)
            
            return grid_value, grid_color

        elif color_method == 'numpy':
            value_numpy = np.array(grid_value)
            color_numpy = np.full((len(cls.grid_coor), 4), 255, dtype = np.uint8)

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
            poly, 
            value_method = 'integer',
            color_method = 'numpy',
            with_color = False,
            handle_error = True,
        ):
        # """
        # Render the grid by computing the values and setting the grid_val to rgba colors.
        # """
        # try:
            if with_color:
                grid_value, grid_color = cls._render_grid_color(poly, value_method = value_method, color_method = color_method)
            else:
                grid_value = cls._render_grid_value(poly, value_method = value_method)
                grid_color = None

            return GridPoly(
                poly,
                size = cls.size,
                grid_coor = cls.grid_coor,
                grid_value = grid_value,
                grid_color = grid_color
            )
        # except Exception as e:
        #     if handle_error:
        #         return cls.zero_grid()
        #     else:
        #         raise e

    @classmethod
    def zero_grid(cls):
        if cls._zero_grid is None:
            a, b, c = sp.symbols('a b c')
            cls._zero_grid = GridPoly(
                sp.polys.Poly(0, (a, b, c)),
                size = cls.size,
                grid_coor = cls.grid_coor,
                grid_value = [0] * len(cls.grid_coor),
                grid_color = [(255,255,255,255)] * len(cls.grid_coor)
            )
        return cls._zero_grid