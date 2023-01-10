import sympy as sp

from ..utils.text_process import deg

try:
    from PySide6 import QtCore, QtWidgets, QtGui
except:
    from PySide2 import QtCore, QtWidgets, QtGui

def _print_grid_triangle(self):
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
    
    # restore
    monoms.pop()


def _print_poly_triangle(self):
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
    color = self.SOS_Manager.grid_color
    for i in range(n+1):
        for j in range(n+1-i):
            QtGui.QPainter.fillRect(qp, 
                            ulx + l*(i+j*2)//2, uly + l*i*13//15, l, l, QtGui.QColor(*color[t]))
            t += 1



if __name__ == '__main__':
    # sos = SOS_Manager()
    # txt = 's(a10)'
    # sos.setPoly(txt)
    # print(type(round(sp.sympify('1/3*1.0'),4)))
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
