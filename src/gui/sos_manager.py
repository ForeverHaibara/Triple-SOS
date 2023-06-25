# author: https://github.com/ForeverHaibara
import sympy as sp

from ..utils import deg, verify_hom_cyclic, poly_get_factor_form, poly_get_standard_form
from ..utils.text_process import preprocess_text, degree_of_zero
from ..utils.roots import RootsInfo, GridRender, findroot
from ..core.sum_of_square import sum_of_square


class SOS_Manager():
    """
    SOS Manager is a class for web GUI.    
    """
    _zero_grid = GridRender.zero_grid()

    def __init__(self):
        self.solution = None

        self._poly_info = {
            'polytxt': None,
            'poly': sp.S(0).as_poly(*sp.symbols('a b c')),
            'ishom': False,
            'iscyc': False,
            'isfrac': False,
            'grid': self._zero_grid,
            'deg': 0,
        }

        self.updeg = 11
        self.deglim = 18

        self._roots_info = RootsInfo()

    @property
    def poly(self):
        return self._poly_info['poly']

    @property
    def deg(self):
        return self._poly_info['deg']

    @property
    def grid(self):
        return self._poly_info['grid']

    def set_poly(self, txt, cancel=True, render_grid=True):
        """
        Set the processed polynomial to some text. If render_grid == True, the grid will also be updated.
        
        Warning: The result might not refresh if the input is invalid.
        """

        if self._poly_info['polytxt'] == txt:
            return True
        
        try:
            poly, isfrac = preprocess_text(txt, cancel = cancel)
            self._poly_info['poly'] = poly
            self._roots_info = RootsInfo()

            if poly.is_zero:
                n = degree_of_zero(txt)
                if n > 0:
                    self._poly_info['deg'] = n
                self._poly_info['isfrac'] = False
                self._poly_info['ishom'] = True
                self._roots_info = RootsInfo()
                self._poly_info['grid'] = self._zero_grid
                return True

            self._poly_info['deg'] = deg(self.poly)
        except: # invalid input
            return False


        if render_grid:
            self._poly_info['grid'] = GridRender.render(self.poly, with_color=True)
        
        self._poly_info['polytxt'] = txt
        self._poly_info['isfrac'] = isfrac
        self._poly_info['ishom'], self._poly_info['iscyc'] = verify_hom_cyclic(self.poly)
        return True


    def get_standard_form(self, formatt = 'short'):
        if not self.poly.domain.is_Numerical:
            return self._poly_info['polytxt']
        if formatt == 'short':
            return poly_get_standard_form(self.poly, formatt = 'short', is_cyc = self._poly_info['iscyc'])
        elif formatt == 'factor':
            return poly_get_factor_form(self.poly)

    def findroot(self):
        if self.deg <= 1 or (not self._poly_info['iscyc']) or (self.poly.is_zero) or (not self._poly_info['ishom']):
            self._roots_info = RootsInfo()
            return self._roots_info

        if not self.poly.domain.is_Numerical:
            self._roots_info = RootsInfo()
            return self._roots_info

        self._roots_info = findroot(
            self.poly, 
            most = 5, 
            grid = self._poly_info['grid'], 
            with_tangents = True
        )
        self._roots_info.sort_tangents()
        print(self._roots_info)
        return self._roots_info

    def sum_of_square(self, method_order = None, configs = None):
        if self.deg <= 1 or (not self._poly_info['iscyc']) or (self.poly.is_zero) or (not self._poly_info['ishom']):
            return
        
        if not self.poly.domain.is_Numerical:
            return

        self.solution = sum_of_square(
            self.poly, 
            rootsinfo = self._roots_info, 
            method_order = method_order,
            configs = configs
        )
        return self.solution

    
    def save_heatmap(self, *args, **kwargs):
        return self._poly_info['grid'].save_heatmap(*args, **kwargs)

    def save_coeffs(self, *args, **kwargs):
        return self._poly_info['grid'].save_coeffs(*args, **kwargs)

    def latex_coeffs(self, *args, **kwargs):
        return self._poly_info['grid'].latex_coeffs(*args, **kwargs)



def _render_LaTeX(a, path, usetex=True, show=False, dpi=500, fontsize=20):
    '''render a text in LaTeX and save it to path'''
    
    import matplotlib.pyplot as plt

    acopy = a
    # linenumber = a.count('\\\\') + 1
    # plt.figure(figsize=(12,10 ))
    
    # set the figure small enough
    # even though the text cannot be display as a whole in the window
    # it will be saved correctly by setting bbox_inches = 'tight'
    plt.figure(figsize=(0.3,0.3))
    if usetex:
        try:
            a = '$\\displaystyle ' + a.strip('$') + ' $'
            #plt.figure(figsize=(12, linenumber*0.5 + linenumber**0.5 * 0.3 ))
            #plt.text(-0.3,0.75+min(0.35,linenumber/25), a, fontsize=15, usetex=usetex)
            #fontfamily='Times New Roman')
            plt.text(-0.3,0.9, a, fontsize=fontsize, usetex=usetex)#
        except:
            usetex = False
    
    if not usetex:
        a = acopy
        a = a.strip('$')
        a = '\n'.join([' $ '+_+' $ ' for _ in a.split('\\\\')])
        plt.text(-0.3,0.9, a, fontsize=fontsize, usetex=usetex)#, fontfamily='Times New Roman')
        
    plt.ylim(0,1)
    plt.xlim(0,6)
    plt.axis('off')
    plt.savefig(path, dpi=dpi, bbox_inches ='tight')
    if show:
        plt.show()
    else:
        plt.close()
