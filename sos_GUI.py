from sum_of_square import *
from PySide6 import QtCore, QtWidgets, QtGui
import matplotlib.pyplot as plt
from copy import deepcopy

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
        self.poly = None
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
        self.gridInit()

    def gridInit(self):
        self.grid_coor = []
        for i in range(self.grid_size + 1):
            for j in range(self.grid_size + 1 - i):
                self.grid_coor.append((j,i))
        self.grid_val = [(255,255,255,255)] * len(self.grid_coor)

        # precal stores the powers of integer
        # precal[i][j] = i ** j    where i <= grid_size = 60 ,   j <= deglim = 18
        # bound = 60 ** 18 = 60^18 = 1.0156e+032
        for i in range(self.grid_size + 1):
            self.grid_precal.append( [1, i] )
            for _ in range(self.deglim - 1):
                self.grid_precal[i].append(self.grid_precal[i][-1] * i)

    def renderGrid(self):
        # integer arithmetic runs much faster than accuarate floating point arithmetic
        # example: computing 45**18   is around ten times faster than  (1./45)**18 

        # convert sympy.core.number.Rational / Float / Integer to python int / float
        if self.deg > self.deglim :
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

    def setPoly(self,txt,render_grid=True):
        try:
            poly = PreprocessText(txt)
            if poly is not None:
                self.poly = self.zeropoly + poly
                self.deg = deg(self.poly)
                self.multiplier = None
                self.roots = []
                self.strict_roots = []
            else:
                # zero polynomial
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
        
        return True 


    def GUI_findRoot(self):
        self.roots, self.strict_roots = findroot(self.poly, maxiter=self.maxiter, roots=self.roots)
        if len(self.roots) > 0:
            self.rootsinfo = 'Local Minima Approx:'
            for root in self.roots:
                self.rootsinfo += f'\n({round(root[0],self.precision)},{round(root[1],self.precision)},1)'
                self.rootsinfo += f' = {round(float(self.poly(*root,1)),self.precision)}'

    def GUI_getTangents(self):
        self.tangents = self.tangents_default[:] + root_tengents(self.roots)

    def GUI_prettyResult(self, y, names, n):
        # 0: LaTeX
        self.sosresults[0] = prettyprint(y, names, 
                        precision=self.precision, linefeed=self.linefeed).strip('$')
        if n - self.deg >= 2:
            self.sosresults[0] = '$$(\\sum a^{%d})f(a,b,c) = '%(n - self.deg) + self.sosresults[0] + '$$'
            self.sosresults[2] = 's(a^{%d})f(a,b,c) = '%(n - self.deg)
        elif n - self.deg == 1:
            self.sosresults[0] = '$$(\\sum a)f(a,b,c) = ' + self.sosresults[0] + '$$'
            self.sosresults[2] = 's(a)f(a,b,c) = '
        else:
            self.sosresults[0] = '$$f(a,b,c) = ' + self.sosresults[0] + '$$'
            self.sosresults[2] = 'f(a,b,c) = '

        # 1: txt
        self.sosresults[1] = self.sosresults[0].strip('$').replace('}{','/')
        self.sosresults[1] = self.sosresults[1].replace(' ','').replace('left','').replace('right','')
        self.sosresults[1] = self.sosresults[1].replace('\\','').replace('sum','Σ').replace('frac','')
        self.sosresults[1] = self.sosresults[1].replace('{','').replace('}','')

        # 2: formatted
        #self.sosresults[2] = self.sosresults[1].replace('\\','').replace('Σ','s')
        self.sosresults[2] += prettyprint(y, names, 
                        precision=self.precision, linefeed=self.linefeed, formatt=2, dectofrac=True)
                        
        
        # 1: txt
        for i, idx in enumerate('²³⁴⁵⁶⁷⁸⁹'):
            self.sosresults[1] = self.sosresults[1].replace('^%d'%(i+2),idx)


    def GUI_SOS(self,txt):
        self.rootsinfo = ''
        self.stage = 0
        if not self.setPoly(txt):
            self.stage = 70
            return False
        self.stage = 10
        if self.deg <= 1 or self.deg >= self.deglim:
            self.stage = 70
            if self.GUI is not None: self.GUI.displaySOS() #self.GUI.interface = 0
            return False
        if self.GUI is not None: self.GUI.displaySOS() #self.GUI.repaint()
        self.updeg = min(self.updeg, self.deglim)

        if self.maxiter > 0:
            self.GUI_findRoot()
        self.stage = 20

        if self.GUI is not None: self.GUI.displaySOS() #self.GUI.repaint()
        self.GUI_getTangents()
        self.stage = 30

        if self.GUI is not None: self.GUI.displaySOS() #self.GUI.repaint()
        retry = 1
        while retry:
            poly = self.poly 
            n = self.deg
            if self.multiplier != None:
                poly = poly * self.multiplier 
                n += deg(self.multiplier)

            if not (n in self.dict_monoms.keys()):
                self.dict_monoms[n] , self.inv_monoms[n] = generate_expr(n)
            
            dict_monom = self.dict_monoms[n]
            inv_monom  = self.inv_monoms[n]           
            b = arraylize(poly,dict_monom,inv_monom)    
                
            # generate basis with degree n
            # make use of already-generated ones
            if not (n in self.names.keys()):
                self.names[n], self.polys[n], self.basis[n] = generate_basis(n,dict_monom,inv_monom,['a2-bc','a3-bc2','a3-b2c'],[])

            names, polys, basis = deepcopy(self.names[n]), deepcopy(self.polys[n]), self.basis[n].copy()
            names, polys, basis = append_basis(n, dict_monom, inv_monom, names, polys, basis, self.tangents)
        
            # reduce the basis according to the strict roots
            names, polys, basis = reduce_basis(n,dict_monom,inv_monom,names,polys,basis,self.strict_roots)
            x = None

            if len(names) > 0:
                with warnings.catch_warnings(record=True) as __warns:
                    warnings.simplefilter('once')
                    try:
                        x = linprog(np.ones(basis.shape[0]), A_eq=basis.T, b_eq=b, method='simplex')
                    #, options={'tol':1e-9})
                    except:
                        pass
        
            if len(names) == 0 or x is None or not x.success:
                if n < self.updeg:
                    # move up a degree and retry!
                    codeg = n + 1 - self.deg
                    self.multiplier = sp.polys.polytools.Poly(f'a^{codeg}+b^{codeg}+c^{codeg}')

                    self.stage = 30 + n
                    if self.GUI is not None: self.GUI.displaySOS() #self.GUI.repaint()
                else: # failed
                    self.stage = 70
                    if self.GUI is not None: self.GUI.displaySOS() #self.GUI.repaint()
                    return ''
            else:
                retry = 0
                self.stage = 50
                #if self.GUI is not None: self.GUI.repaint()
        
        # Approximates the coefficients to fractions if possible
        rounding = 0.1
        y = rationalize_array(x.x, rounding=rounding, mod=self.mod)

        # check if the approximation works, if not, cut down the rounding and retry
        while (not verify(y,polys,poly,tol=1e-8)) and rounding > 1e-9:
            rounding *= 0.1
            y = rationalize_array(x.x, rounding=rounding, mod=self.mod)
            
        # obtain the LaTeX format
        self.GUI_prettyResult(y, names, n)
        self.stage = 50

        if self.GUI is not None: 
            self.GUI.displaySOS() #
            #self.GUI.txt_displayResult.setText(self.sosresults[self.GUI.btn_displaymodeselect])
            #self.GUI.repaint()

            renderLaTeX(self.sosresults[0],'Formula.png')
            self.stage = 60
            
            self.GUI.displaySOS() #self.GUI.repaint()

        return self.sosresults[0]
        



def printPolyTriangle(self):
    '''print the coefficient triangle in the Qt canva'''
    poly = self.SOS_Manager.poly
    if poly is None:
        return 
    n = deg(poly)
    if n == 0:
        return
    w = self.width()
    h = self.height()

    ctrx = int(w*0.36)   # display centerx 
    ctry = int(h*0.3926) # display centery
    bw = w*7//10
    bh = h-h//18-h//20
    l = min(bw,bh)//(n+1)  # the length of each equilateral triangle
    #QtGui.QPainter.drawText(QtGui.QPainter(self), ctrx, ctry, "Center")
    ulx = ctrx - (l*n>>1)
    uly = ctry - 13*n*l//45

    coeffs = poly.coeffs()
    monoms = self.SOS_Manager.std_monoms
    monoms.append((-1,-1,0))  # tail flag

    t = 0
    qp = QtGui.QPainter(self)
    fontsize = max(8, 15-n//3)
    qp.setFont(QtGui.QFont('Times New Roman',fontsize))
    for i in range(n+1):
        for j in range(i+1):
            if monoms[t][0] == n - i and monoms[t][1] == i - j:
                if isinstance(coeffs[t],sp.core.numbers.Float):
                    txt = f'{round(float(coeffs[t]),4)}'
                else:
                    txt = f'{coeffs[t].p}' + (f'/{coeffs[t].q}' if coeffs[t].q != 1 else '')
                    if len(txt) > 10:
                        txt = f'{round(float(coeffs[t]),4)}'
                t += 1
            else:
                txt = '0'
            QtGui.QPainter.drawText(qp, 
                            ulx + l*(2*i-j)//2 - len(txt)*(fontsize-1)//2, uly + l*j*13//15, txt)
    
    # retrieve
    monoms.pop()


def printGridTriangle(self):
    '''print the heat map in the Qt canva'''
    n = self.SOS_Manager.grid_size
    w = self.width()
    h = self.height()

    ctrx = int(w*0.856)   # display centerx 
    ctry = int(h*0.133)   # display center
    bw = w*21//80
    bh = h*60//120
    l = min(bw,bh)//n  # the length of each equilateral triangle
    #QtGui.QPainter.drawText(QtGui.QPainter(self), ctrx, ctry, "Center")
    ulx = ctrx - (l*n>>1)
    uly = ctry - 13*n*l//45
    
    qp = QtGui.QPainter(self)
    t = 0
    color = self.SOS_Manager.grid_val
    for i in range(n+1):
        for j in range(n+1-i):
            QtGui.QPainter.fillRect(qp, 
                            ulx + l*(i+j*2)//2, uly + l*i*13//15, l, l, QtGui.QColor(*color[t]))
            t += 1



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


if __name__ == '__main__':
    sos = SOS_Manager()
    txt = 's(a10)'
    sos.setPoly(txt)
    print(type(round(sp.sympify('1/3*1.0'),4)))
    #print(sos.grid_val)
    #sos.GUI_SOS(txt)
    #print(sos.sosresults)
    '''
    from time import time as timer
    txt = 's(a9)s(a3)+9p(a4)-6a3b3c3s(a3)'
    t = timer()
    print(sos.GUI_SOS(txt))
    print(f'Time = {timer()-t}\n')
    t = timer()
    print(sos.GUI_SOS(txt))
    print(f'Time = {timer()-t}\n')
    print('Roots = ',sos.roots)
    print('Tangents = ',sos.tangents)
    #print(sos.basis[6].shape)
    '''
    #renderLaTeX(r'c:\ |z-z_0|=1 \\ \frac{1}{2\pi i}\int_c \frac{f(z)}{z-z_0}dz = Res_{z=z_0}f(z)','Formula.png')