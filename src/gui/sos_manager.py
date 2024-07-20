# author: https://github.com/ForeverHaibara
import sympy as sp

from ..utils import deg, verify_hom_cyclic, poly_get_factor_form, poly_get_standard_form, latex_coeffs
from ..utils.text_process import preprocess_text, degree_of_zero, coefficient_triangle
from ..utils.roots import RootsInfo, GridRender, findroot
from ..core.sum_of_square import sum_of_square
from ..core.linsos import root_tangents


class SOS_Manager():
    """
    SOS Manager is a class for web GUI.
    """

    # self._poly_info = {
    #     'polytxt': None,
    #     'poly': sp.S(0).as_poly(*sp.symbols('a b c')),
    #     'ishom': False,
    #     'iscyc': False,
    #     'isfrac': False,
    #     'grid': self._zero_grid,
    #     'deg': 0,
    # }

    @classmethod
    def set_poly(cls, 
            txt,
            render_triangle = True,
            render_grid = True,
            factor = False,
        ):
        """
        0
        """
        try:
            poly, denom = preprocess_text(txt, cancel = True)

            if poly.is_zero:
                n = degree_of_zero(txt)
            else:
                n = deg(poly)
        except:
            return None

        if poly is None:
            return None

        try:
            if factor:
                txt2 = poly_get_factor_form(poly)
            elif not denom.degree() == 0:
                txt2 = poly_get_standard_form(poly)
            if isinstance(txt2, str):
                txt = txt2
        except:
            pass


        return_dict = {'poly': poly, 'degree': n, 'txt': txt}
        if render_triangle:
            return_dict['triangle'] = coefficient_triangle(poly)

        if render_grid:
            if cls.check_poly(poly):
                grid = GridRender.render(poly, with_color=True)
            else:
                grid = GridRender.zero_grid()
            return_dict['grid'] = grid
        return return_dict
        # if self._poly_info['polytxt'] == txt:
        #     return True
        
        # try:
        #     poly, isfrac = preprocess_text(txt, cancel = cancel)
        #     self._poly_info['poly'] = poly
        #     self._roots_info = RootsInfo()

        #     if poly.is_zero:
        #         n = degree_of_zero(txt)
        #         if n > 0:
        #             self._poly_info['deg'] = n
        #         self._poly_info['isfrac'] = False
        #         self._poly_info['ishom'] = True
        #         self._roots_info = RootsInfo()
        #         self._poly_info['grid'] = self._zero_grid
        #         return True

        #     self._poly_info['deg'] = deg(self.poly)
        # except: # invalid input
        #     return False


        # if render_grid:
        #     self._poly_info['grid'] = GridRender.render(self.poly, with_color=True)
        
        # self._poly_info['polytxt'] = txt
        # self._poly_info['isfrac'] = isfrac
        # self._poly_info['ishom'], self._poly_info['iscyc'] = verify_hom_cyclic(self.poly)
        # return True

    @classmethod
    def check_poly(cls, poly):
        if poly is None or (not isinstance(poly, sp.Poly)):
            return False
        if len(poly.gens) != 3 or (poly.is_zero) or (not poly.is_homogeneous) or deg(poly) < 1:
            return False
        if not poly.domain.is_Numerical:
            return False
        return True

    @classmethod
    def get_standard_form(cls, poly, formatt = 'short'):
        if (not cls.check_poly(poly)) or not (poly.domain in (sp.ZZ, sp.QQ, sp.RR)):
            return None
        if formatt == 'short':
            return poly_get_standard_form(poly, formatt = 'short') #, is_cyc = self._poly_info['iscyc'])
        elif formatt == 'factor':
            return poly_get_factor_form(poly)

    @classmethod
    def findroot(cls, poly, grid = None, verbose = True):
        if not cls.check_poly(poly):
            return RootsInfo()

        roots_info = findroot(
            poly, 
            most = 5, 
            grid = grid, 
            with_tangents = root_tangents
        )
        roots_info.sort_tangents()
        if verbose:
            print(roots_info)
        return roots_info

    @classmethod
    def sum_of_square(cls, poly, rootsinfo = None, method_order = None, configs = None):
        if not cls.check_poly(poly):
            return None

        solution = sum_of_square(
            poly,
            rootsinfo = rootsinfo, 
            method_order = method_order,
            configs = configs
        )
        return solution

    
    def save_heatmap(self, *args, **kwargs):
        return self._poly_info['grid'].save_heatmap(*args, **kwargs)

    def save_coeffs(self, *args, **kwargs):
        return self._poly_info['grid'].save_coeffs(*args, **kwargs)

    def latex_coeffs(self, *args, **kwargs):
        return latex_coeffs(self._poly_info['poly'], *args, **kwargs)



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
