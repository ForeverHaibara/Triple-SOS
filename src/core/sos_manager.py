# author: https://github.com/ForeverHaibara
from numbers import Number as PythonNumber
import re
# import warnings

import sympy as sp
import numpy as np
# from scipy.optimize import linprog
# from scipy.optimize import OptimizeWarning
# from matplotlib import pyplot as plt

from ..utils import deg, verify_hom_cyclic, poly_get_factor_form, poly_get_standard_form
from ..utils.text_process import preprocess_text, degree_of_zero
# from ..utils.text_process import deg, verify_hom_cyclic, degree_of_zero, poly_get_standard_form, poly_get_factor_form
# from ..utils.text_process import PreprocessText, prettyprint, text_compresser, text_sorter, text_multiplier
# from ..utils.basis_generator import arraylize, invarraylize, generate_expr, generate_basis, append_basis, reduce_basis
from ..utils.roots import RootsInfo, GridRender, findroot
# from .sum_of_square import exact_coefficient, up_degree, SOS_Special


class SOS_Manager():
    def __init__(self, GUI=None):
        self.GUI = GUI
        self.linefeed = 2
        self.sosresults = ['','','']

        self._poly_info = {
            'polytxt': None,
            'poly': sp.S(0).as_poly(*sp.symbols('a b c')),
            'ishom': False,
            'iscyc': False,
            'isfrac': False,
            'grid': None,
            'deg': 0,
        }

        self.updeg = 11
        self.deglim = 18

        self._roots_info = RootsInfo()
        self.precision = 8
        self.stage = 60

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
        if formatt == 'short':
            return poly_get_standard_form(self.poly, formatt = 'short', is_cyc = self._poly_info['iscyc'])
        elif formatt == 'factor':
            return poly_get_factor_form(self.poly)

    def findroot(self):
        if self.deg <= 1 or (not self._poly_info['iscyc']) or (self.poly.is_zero) or (not self._poly_info['ishom']):
            return
        
        self._roots_info = findroot(
            self.poly, 
            most = 5, 
            grid = self._poly_info['grid'], 
            with_tangents = True
        )
        self._roots_info.sort_tangents()
        print(self._roots_info)
        return self._roots_info


    def GUI_prettyResult(self, y, names, multipliers = None, equal = True):
        y , names = text_compresser(y, names)

        y , names , linefeed = text_sorter(y, names)
        
        multiplier_txt = text_multiplier(multipliers)

        # 0: LaTeX
        self.sosresults[0] = prettyprint(y, names,
                        precision = self.precision, linefeed=linefeed).strip('$')
                        
        # if n - origin_deg >= 2:
        #     self.sosresults[0] = '$$\\left(\\sum a^{%d}\\right)f(a,b,c) = '%(n - self.deg) + self.sosresults[0] + '$$'
        #     self.sosresults[2] = 's(a^{%d})f(a,b,c) = '%(n - self.deg)
        # elif n - origin_deg == 1:
        #     self.sosresults[0] = '$$\\left(\\sum a\\right)f(a,b,c) = ' + self.sosresults[0] + '$$'
        #     self.sosresults[2] = 's(a)f(a,b,c) = '
        # else:
        self.sosresults[0] = '$$' + multiplier_txt[0] + 'f(a,b,c) = ' + self.sosresults[0] + '$$'
        self.sosresults[2] = multiplier_txt[1] + 'f(a,b,c) = '

        # 1: txt
        self.sosresults[1] = self.sosresults[0].strip('$').replace(' ','')
        self.sosresults[1] = self.sosresults[1].replace('left','').replace('right','')
        self.sosresults[1] = self.sosresults[1].replace('cdot','')
        self.sosresults[1] = self.sosresults[1].replace('\\','').replace('sum','Σ')
        
        parener = lambda x: '(%s)'%x if '+' in x or '-' in x else x
        self.sosresults[1] = re.sub('frac\{(.*?)\}\{(.*?)\}',
                        lambda x: '%s/%s'%(parener(x.group(1)), parener(x.group(2))),
                        self.sosresults[1])

        self.sosresults[1] = self.sosresults[1].replace('{','').replace('}','')

        self.sosresults[1] = self.sosresults[1].replace('sqrt','√')

        for i, idx in enumerate('²³⁴⁵⁶⁷⁸⁹'):
            self.sosresults[1] = self.sosresults[1].replace('^%d'%(i+2),idx)

        # 2: formatted
        self.sosresults[2] += prettyprint(y, names,
                        precision=self.precision, linefeed=self.linefeed, formatt=2, dectofrac=True)
                        
        if not equal:
            self.sosresults[0] = self.sosresults[0].replace('=', '\\approx')
            self.sosresults[1] = self.sosresults[1].replace('=', '≈')
            self.sosresults[2] = self.sosresults[2].replace('=', '≈')


    def GUI_stateUpdate(self, stage = None):
        if stage is not None:
            self.stage = stage
        if self.GUI is not None:
            self.GUI.displaySOS()


    def GUI_SOS(self, txt, skip_setpoly = False, skip_findroots = False, skip_tangents = False,
                verbose_updeg = False, use_structural_method = True):
        self._roots_info['rootsinfo'] = ''
        self.stage = 0
        if (not skip_setpoly) and (not self.setPoly(txt)):
            self.stage = 70
            return ''
            
        origin_deg = self.deg
        if origin_deg <= 1 or origin_deg >= self.deglim or (not self._poly_info['iscyc']) or self.poly is None:
            self.GUI_stateUpdate(70)
            return ''
        self.GUI_stateUpdate(10)
        self.updeg = min(self.updeg, self.deglim)

        self.GUI_stateUpdate(20)

        if not skip_findroots:
            self.GUI_findRoot()
        if not skip_tangents:
            self.GUI_getTangents()

        self.GUI_stateUpdate(30)
        
        # copy here to avoid troubles in async
        original_poly = self.poly
        strict_roots = self._roots_info['strict_roots'].copy()
        tangents = self._roots_info['tangents'].copy()

        # initialization with None
        x , names , polys , basis , multipliers = None , None , None , None , []

        for multiplier, poly, n in up_degree(self.poly, origin_deg, self.updeg):
            multipliers = [multiplier]

            # try SOS on special structures
            if use_structural_method:
                special_result = SOS_Special(poly, n)
                if special_result is not None:
                    new_multipliers , y , names = special_result
                    multipliers += new_multipliers
                    polys = None
                    break

            b = arraylize(poly)
                
            # generate basis with degree n
            # make use of already-generated ones
            names, polys, basis = generate_basis(n)
            names, polys, basis = append_basis(n, tangents, names = names, polys = polys, basis = basis)
        
            # reduce the basis according to the strict roots
            names, polys, basis = reduce_basis(n, strict_roots, names = names, polys = polys, basis = basis)

            x = None

            if len(names) > 0:
                with warnings.catch_warnings(record=True) as __warns:
                    warnings.simplefilter('once')
                    try:
                        x = linprog(np.ones(basis.shape[0]), A_eq=basis.T, b_eq=b, method='simplex')
                        y = x.x
                    #, options={'tol':1e-9})
                    except:
                        pass
        
            if len(names) != 0 and (x is not None) and x.success:
                self.stage = 50
                #if self.GUI is not None: self.GUI.repaint()
                break

            self.GUI_stateUpdate(30+n)
            if verbose_updeg:
                print('Failed with degree %d'%n)
                
        else: # failed
            self.GUI_stateUpdate(70)
            return ''
        
        y, names, equal = exact_coefficient(original_poly, multipliers, y, names, polys, self)
        

        # obtain the LaTeX format
        self.GUI_prettyResult(y, names, multipliers, equal = equal)
        self.stage = 50

        if self.GUI is not None:
            self.GUI_stateUpdate(50)
            #self.GUI.txt_displayResult.setText(self.sosresults[self.GUI.btn_displaymodeselect])
            #self.GUI.repaint()

            _render_LaTeX(self.sosresults[0],'Formula.png')
            self.GUI_stateUpdate(60)

        return self.sosresults[0]

    
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
    #linenumber = a.count('\\\\') + 1
    #plt.figure(figsize=(12,10 ))
    
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


# examples
if __name__ == '__main__':
    # initialize only once!
    sos = SOS_Manager()

    # setPoly first, then get heatmap / coefficient triangle
    s = '3a3+3b3-6c3+3b2c-c2a-2a2b-2bc2+16ca2-14ab2'     # input
    sos.setPoly(s)   # input the text
    sos.save_heatmap('heatmap.png')  # save the heatmap to 'heatmap.png'
    print(sos.getStandardForm())
    
    # another example
    s = 's(a6)+12p(a2)-78p(a(a-b))'   # input
    sos.setPoly(s)   # input the text
    sos.save_coeffs('coeffs.png')  # save the coefficient triangle
    x = sos.latex_coeffs()  # get the latex coefficients
    print(sos.getStandardForm())
    print(x, end='\n\n')

    # auto sum of square
    s = 's(a2)2-3s(a3b)'
    s = '4s(ab)s((a+b)^2(a+c)^2)-9p((a+b)^2)'
    # no need to call setPoly when using GUI_SOS
    x = sos.GUI_SOS(s).strip('$')
    print(x, end='\n\n')



    # -------  Return empty string when cases are undefined  -------
    check_undefined_cases = False

    if check_undefined_cases:
        # Undefined cases 1, 2
        s = 's(a%?!!!asdquwve'    # invalid inputs
        #s = 's(a)2-s(a2+2ab)'    # zero polynomial
        sos.setPoly(s)
        sos.save_heatmap('heatmap2.png')
        x = sos.latex_coeffs()     # empty string
        print(x, end='\n\n')
        print(sos.GUI_SOS(s), end='\n\n')  # empty result
        
        # Undefined case 3
        s = 's(a)2-2s(a2+2ab)'    # zero polynomial
        x = sos.GUI_SOS(s)        # sum of square attempt fails
        print(x, end='\n\n')      # empty string
