from sum_of_square import *
import sympy as sp 
import numpy as np 
import matplotlib.pyplot as plt
from copy import deepcopy
from numbers import Number as PythonNumber 

class SOS_Manager():
    def __init__(self,GUI=None):
        self.names = {}
        self.basis = {}
        self.polys = {}
        self.dict_monoms = {}
        self.inv_monoms  = {}
        self.GUI = GUI
        self.linefeed = 2
        self.sosresults = ['','','']

        self.zeropoly = sp.polys.polytools.Poly('a+b+c') - sp.polys.polytools.Poly('a+b+c')
        self.polytxt = None 
        self.poly = None
        self.poly_ishom = False 
        self.poly_iscyc = False 
        self.poly_isfrac = False 
        self.multiplier = None
        self.deg = 0
        self.updeg = 10
        self.deglim = 18

        self.maxiter = 5000
        self.precision = 8
        self.mod = None
        self.roots = []
        self.strict_roots = []
        self.rootsinfo = ''
        self.tangents_default = []
        self.tangents = []
        self.stage = 60

        self.std_monoms = []
        self.grid_resolution = 12
        self.grid_size = 60
        self.grid_coor = []
        self.grid_precal = []
        self.grid_val = []
        self.grid_deglim = 18
        self.gridInit()


    def gridInit(self):
        '''
        Initialize the grid and preprocess some values.
        '''
        # grid_coor[k] = (i,j) stands for the value  f(n-i-j, j, i)
        self.grid_coor = []
        for i in range(self.grid_size + 1):
            for j in range(self.grid_size + 1 - i):
                self.grid_coor.append((j,i))

        # initialize the colors by white
        self.grid_val = [(255,255,255,255)] * len(self.grid_coor)

        # precal stores the powers of integer
        # precal[i][j] = i ** j    where i <= grid_size = 60 ,   j <= deglim = 18
        # bound = 60 ** 18 = 60^18 = 1.0156e+032
        for i in range(self.grid_size + 1):
            self.grid_precal.append( [1, i] )
            for _ in range(self.grid_deglim - 1):
                self.grid_precal[i].append(self.grid_precal[i][-1] * i)


    def renderGrid(self):
        '''
        Render the grid by computing the values and setting the grid_val to rgba colors.
        '''
        
        # integer arithmetic runs much faster than accuarate floating point arithmetic
        # example: computing 45**18   is around ten times faster than  (1./45)**18 

        # convert sympy.core.number.Rational / Float / Integer to python int / float
        if self.deg > self.grid_deglim :
            return 
        coeffs = self.poly.coeffs()
        for i in range(len(coeffs)):
            if int(coeffs[i]) == coeffs[i]:
                coeffs[i] = int(coeffs[i])
            else:
                coeffs[i] = float(coeffs[i])
        
        # pointer, but not deepcopy
        pc = self.grid_precal

        max_v , min_v = 0 , 0
        for k in range(len(self.grid_coor)):
            b , c = self.grid_coor[k]
            a = self.grid_size - b - c
            v = 0
            for coeff, monom in zip(coeffs, self.std_monoms):
                # coeff shall be the last to multiply, as it might be float while others int
                v += pc[a][monom[0]] * pc[b][monom[1]] * pc[c][monom[2]] * coeff
            max_v , min_v = max(v, max_v), min(v, min_v)
            self.grid_val[k] = v

        # preprocess the levels
        if max_v >= 0:
            max_levels = [(i / self.grid_resolution)**2 * max_v for i in range(self.grid_resolution+1)] 
        if min_v <= 0:
            min_levels = [(i / self.grid_resolution)**2 * min_v for i in range(self.grid_resolution+1)] 
    
        for k in range(len(self.grid_coor)):
            v = self.grid_val[k]
            if v > 0:
                for i, level in enumerate(max_levels):
                    if v <= level: 
                        v = 255 - (i-1)*255//(self.grid_resolution-1)
                        self.grid_val[k] = (255, v, 0, 255)
                        break 
                else:
                    self.grid_val[k] = (255, 0, 0, 255)
            elif v < 0:
                for i, level in enumerate(min_levels):
                    if v >= level:
                        v = 255 - (i-1)*255//(self.grid_resolution-1)
                        self.grid_val[k] = (0, v, 255, 255)
                        break
                else:
                    self.grid_val[k] = (0, 0, 0, 255)
            else: # v == 0:
                self.grid_val[k] = (255, 255, 255, 255)
        

    def setPoly(self, txt, cancel = False, render_grid=True):
        '''
        Set the processed polynomial to some text. If render_grid == True, the grid will also be updated.
        
        Warning: The result might not refresh if the input is invalid.
        '''

        if self.polytxt == txt:
            return True 
        
        try:
            poly , isfrac = PreprocessText(txt, cancel = cancel)
            if poly is not None:
                # add a zero polynomial, to ensure it has variables 'a','b','c'
                self.poly = self.zeropoly + poly
                self.deg = deg(self.poly)
                self.multiplier = None
                self.roots = []
                self.strict_roots = []
            else:
                # zero polynomial (or invalid input)
                if self.GUI is None: self.poly = None
                self.rootsinfo = ''
                self.grid_val = [(255,255,255,255)] * len(self.grid_coor)
                self.std_monoms = []
                return True
        except:
            return False

        
        args = self.poly.args
        if len(args) == 4:
            self.std_monoms = self.poly.monoms()
        elif len(args) == 3: 
            if args[1].name == 'a':
                if args[2].name == 'b':
                    self.std_monoms = [(i,j,0) for (i,j) in self.poly.monoms()]
                else: #args[2].name == 'c':
                    self.std_monoms = [(i,0,j) for (i,j) in self.poly.monoms()]
            else: # b , c
                self.std_monoms = [(0,i,j) for (i,j) in self.poly.monoms()]
        elif len(args) == 2:
            if args[1].name == 'a':
                self.std_monoms = [(i,0,0) for (i,) in self.poly.monoms()]
            elif args[1].name == 'b':
                self.std_monoms = [(0,i,0) for (i,) in self.poly.monoms()]
            else: #args[1].name == 'c':
                self.std_monoms = [(0,0,i) for (i,) in self.poly.monoms()]
        else: # empty
            self.std_monoms = []
            

        if render_grid:
            try:
                self.renderGrid()
            except:
                pass
        
        self.polytxt = txt 
        self.poly_isfrac = isfrac
        self.poly_ishom, self.poly_iscyc = CheckHomCyclic(self.poly, self.deg)
        return True 


    def getStandardForm(self, formatt = 'short'): 
        if formatt == 'short':
            def TitleParser(char, deg):
                return '' if deg == 0 else (char if deg == 1 else (char + str(deg)))
            def Formatter(x):
                if x == 1:
                    return '+'
                elif x >= 0:
                    return f'+{x}'
                elif x == -1:
                    return f'-'
                else:
                    return f'{x}'
            if self.poly_iscyc:
                txt = ''
                for coeff, monom in zip(self.poly.coeffs(), self.poly.monoms()):
                    a , b , c = monom 
                    if a >= b and a >= c:
                        if a == b and a == c:
                            txt += Formatter(coeff/3) + TitleParser('a',a) + TitleParser('b',b) + TitleParser('c',c)
                        elif (a != b and a != c) or a == b:
                            txt += Formatter(coeff) + TitleParser('a',a) + TitleParser('b',b) + TitleParser('c',c)
                if txt.startswith('+'):
                    txt = txt[1:]
                return 's(' + txt + ')'

            else: # not cyclic 
                txt = ''
                for coeff, monom in zip(self.poly.coeffs(), self.poly.monoms()):
                    a , b , c = monom
                    txt += Formatter(coeff) + TitleParser('a',a) + TitleParser('b',b) + TitleParser('c', c)
                if txt.startswith('+'):
                    txt = txt[1:]
                return txt 

    def GUI_findRoot(self):
        if self.deg <= 1 or (not self.poly_iscyc):
            return 
        self.roots, self.strict_roots = findroot(self.poly, maxiter=self.maxiter, roots=self.roots)
        if len(self.roots) > 0:
            self.rootsinfo = 'Local Minima Approx:'
            print(self.roots)
            def Formatter(root, precision = self.precision, maxlen = 20):
                if isinstance(root, PythonNumber):
                    return round(root, precision)
                elif len(str(root)) > maxlen:
                    return round(complex(root).real, precision)
                else:
                    return root 
            for root in self.roots:
                self.rootsinfo += f'\n({Formatter(root[0])},{Formatter(root[1])},1)'
                self.rootsinfo += f' = {Formatter(self.poly(complex(root[0]).real, float(root[1]),1))}'
        else:
            self.rootsinfo = ''
        return self.rootsinfo
                

    def GUI_getTangents(self):
        self.tangents = sorted(self.tangents_default[:] + root_tangents(self.roots), key = lambda x:len(x))
        return self.tangents


    def GUI_prettyResult(self, y, names, n):
        # 0: LaTeX
        self.sosresults[0] = prettyprint(y, names, 
                        precision=self.precision, linefeed=self.linefeed).strip('$')
                        
        if n - self.deg >= 2:
            self.sosresults[0] = '$$\\left(\\sum a^{%d}\\right)f(a,b,c) = '%(n - self.deg) + self.sosresults[0] + '$$'
            self.sosresults[2] = 's(a^{%d})f(a,b,c) = '%(n - self.deg)
        elif n - self.deg == 1:
            self.sosresults[0] = '$$\\left(\\sum a\\right)f(a,b,c) = ' + self.sosresults[0] + '$$'
            self.sosresults[2] = 's(a)f(a,b,c) = '
        else:
            self.sosresults[0] = '$$f(a,b,c) = ' + self.sosresults[0] + '$$'
            self.sosresults[2] = 'f(a,b,c) = '

        # 1: txt
        self.sosresults[1] = self.sosresults[0].strip('$').replace('}{','/')
        self.sosresults[1] = self.sosresults[1].replace(' ','').replace('left','').replace('right','')
        self.sosresults[1] = self.sosresults[1].replace('\\','').replace('sum','Σ').replace('frac','')
        self.sosresults[1] = self.sosresults[1].replace('{','').replace('}','')
        for i, idx in enumerate('²³⁴⁵⁶⁷⁸⁹'):
            self.sosresults[1] = self.sosresults[1].replace('^%d'%(i+2),idx)

        # 2: formatted
        self.sosresults[2] += prettyprint(y, names, 
                        precision=self.precision, linefeed=self.linefeed, formatt=2, dectofrac=True)
                        

    def GUI_stateUpdate(self, stage = None):
        if stage is not None: 
            self.stage = stage 
        if self.GUI is not None:
            self.GUI.displaySOS()


    def GUI_SOS(self, txt, skip_setpoly = False, skip_findroots = False, skip_tangents = False,
                verbose_updeg = False):
        self.rootsinfo = ''
        self.stage = 0
        if (not skip_setpoly) and (not self.setPoly(txt)):
            self.stage = 70
            return ''
            
        if self.deg <= 1 or self.deg >= self.deglim or (not self.poly_iscyc) or self.poly is None:
            self.GUI_stateUpdate(70)
            return ''
        self.GUI_stateUpdate(10)
        self.updeg = min(self.updeg, self.deglim)

        if self.maxiter > 0 and not skip_findroots:
            self.GUI_findRoot()
            
        self.GUI_stateUpdate(20)

        if not skip_tangents:
            self.GUI_getTangents()

        self.GUI_stateUpdate(30)
        
        # copy here to avoid troubles in async
        strict_roots = self.strict_roots.copy()
        tangents = self.tangents.copy()

        for multiplier, poly, n in UpDegree(self.poly, self.deg, self.updeg):
            if not (n in self.dict_monoms.keys()):
                self.dict_monoms[n] , self.inv_monoms[n] = generate_expr(n)
            
            dict_monom = self.dict_monoms[n]
            inv_monom  = self.inv_monoms[n]        
            b = arraylize(poly, dict_monom, inv_monom)    
                
            # generate basis with degree n
            # make use of already-generated ones
            if not (n in self.names.keys()):
                self.names[n], self.polys[n], self.basis[n] = generate_basis(n,dict_monom,inv_monom,['a2-bc','a3-bc2','a3-b2c'],[])

            names, polys, basis = deepcopy(self.names[n]), deepcopy(self.polys[n]), self.basis[n].copy()
            names, polys, basis = append_basis(n, dict_monom, inv_monom, names, polys, basis, tangents)
        
            # reduce the basis according to the strict roots
            names, polys, basis = reduce_basis(n,dict_monom,inv_monom,names,polys,basis,strict_roots)
            x = None

            if len(names) > 0:
                with warnings.catch_warnings(record=True) as __warns:
                    warnings.simplefilter('once')
                    try:
                        x = linprog(np.ones(basis.shape[0]), A_eq=basis.T, b_eq=b, method='simplex')
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
        
        # Approximates the coefficients to fractions if possible
        rounding = 0.1
        y = rationalize_array(x.x, rounding=rounding, mod=self.mod, reliable=True)

        # check if the approximation works, if not, cut down the rounding and retry
        while (not verify(y,polys,poly,tol=1e-8)) and rounding > 1e-9:
            rounding *= 0.1
            y = rationalize_array(x.x, rounding=rounding, mod=self.mod, reliable=True)
            
        # obtain the LaTeX format
        self.GUI_prettyResult(y, names, n)
        self.stage = 50

        if self.GUI is not None: 
            self.GUI_stateUpdate(50)
            #self.GUI.txt_displayResult.setText(self.sosresults[self.GUI.btn_displaymodeselect])
            #self.GUI.repaint()

            renderLaTeX(self.sosresults[0],'Formula.png')
            self.GUI_stateUpdate(60)

        return self.sosresults[0]

    
    def saveHeatmap(self,path,dpi=None,backgroundcolor=211):
        '''save the heatmap to the path'''
        n = self.grid_size
        x = np.full((n+1,n+1,3),backgroundcolor,dtype='uint8')
        for i in range(n+1):
            t = i * 15 // 26   # i * 15/26 ~ i / sqrt(3)
            r = t << 1         # row of grid triangle = i / (sqrt(3)/2)
            if r > n:          # n+1-r <= 0
                break
            base = r*(2*n+3-r)//2   # (n+1) + n + ... + (n+2-r)
            for j in range(n+1-r):  # j < n+1-r
                x[i,j+t,0] = self.grid_val[base+j][0]
                x[i,j+t,1] = self.grid_val[base+j][1]
                x[i,j+t,2] = self.grid_val[base+j][2]
    
        plt.imshow(x,interpolation='nearest')
        plt.axis('off')
        plt.savefig(path, dpi=dpi, bbox_inches ='tight')
        plt.close()


    def saveCoeffs(self,path,dpi=500,fontsize=20):
        '''save the coefficient triangle (as an image) to path'''
        coeffs = self.poly.coeffs()
        monoms = self.std_monoms
        monoms.append((-1,-1,0))  # tail flag

        maxlen = 1
        for coeff in coeffs: 
            maxlen = max(maxlen,len(f'{round(float(coeff),4)}'))

        distance = max(maxlen + maxlen % 2 + 1, 8)
        #print(maxlen,distance)
        n = self.deg
        strings = [((distance + 1) // 2 * i) * ' ' for i in range(n+1)]

        t = 0
        for i in range(n+1):
            for j in range(i+1):
                if monoms[t][0] == n - i and monoms[t][1] == i - j:
                    if isinstance(coeffs[t],sp.core.numbers.Float):
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

    
    def LaTeXCoeffs(self,tabular=True):
        '''return the LaTeX format of the coefficient triangle'''
        n = self.deg
        emptyline = '\\\\ ' + '\\ &'*((n<<1)) + '\\  \\\\ '
        strings = ['' for i in range(n+1)]
        
        if self.poly is not None:
            coeffs = self.poly.coeffs()
            monoms = self.std_monoms
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
            s = '\\left[\\begin{matrix}\\begin{tabular}{' + 'c' * ((n<<1)|1) + '} ' + s
            s += ' \\end{tabular}\\end{matrix}\\right]'
        else:
            s = '\\left[\\begin{matrix} ' + s
            s += ' \\end{matrix}\\right]'

        return s



def renderLaTeX(a, path, usetex=True, show=False, dpi=500, fontsize=20):
    '''render a text in LaTeX and save it to path'''
    
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
    sos.saveHeatmap('heatmap.png')  # save the heatmap to 'heatmap.png'
    print(sos.getStandardForm())
    
    # another example
    s = 's(a6)+12p(a2)-78p(a(a-b))'   # input
    sos.setPoly(s)   # input the text
    sos.saveCoeffs('coeffs.png')  # save the coefficient triangle
    x = sos.LaTeXCoeffs()  # get the latex coefficients
    print(sos.getStandardForm())
    print(x, end='\n\n')

    # auto sum of square
    s = 's(a2)2-3s(a3b)'
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
        sos.saveHeatmap('heatmap2.png')
        x = sos.LaTeXCoeffs()     # empty string
        print(x, end='\n\n')
        print(sos.GUI_SOS(s), end='\n\n')  # empty result
        
        # Undefined case 3
        s = 's(a)2-2s(a2+2ab)'    # zero polynomial
        x = sos.GUI_SOS(s)        # sum of square attempt fails
        print(x, end='\n\n')      # empty string