# todos:
# 1. transfer to flask (html support)
# 2. enable async / await
# 3. enable mousehover hint / highlight 



#pyinstaller --clean -Fw -p D:/Qt graphics_main.py 
#pyinstaller --clean -w -p D:/Qt --hidden-import PySide6  graphics_main.py 
import sys

try:
    from PySide6 import QtCore, QtWidgets, QtGui
    from PySide6.QtWidgets import QApplication
except:
    from PySide2 import QtCore, QtWidgets, QtGui
    from PySide2.QtWidgets import QApplication

from sos_manager import SOS_Manager
from sos_GUI import *

class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.interface = 0
        self.setWindowTitle('Sum of Square')

        self.SOS_Manager = SOS_Manager(self)

        self.txt_input = QtWidgets.QLineEdit(self)
        self.btn_sos = QtWidgets.QPushButton(self)
        self.btn_sos.setText("Sum of Square")
        self.btn_sos.clicked.connect(self.autosos)

        self.btn_displaymode = [None,None,None]
        self.btn_displaymodeselect = 0
        for i, name in enumerate(('LaTeX','txt','formatted')):
            self.btn_displaymode[i] = QtWidgets.QPushButton(self)
            self.btn_displaymode[i].setText(name) 
            self.btn_displaymode[i].clicked.connect(self.displaymodeselect)
            if self.btn_displaymodeselect == i:
                self.btn_displaymode[i].setStyleSheet("background-color: white") 
            else:
                self.btn_displaymode[i].setStyleSheet("border: none") 
        self.txt_displayResult = QtWidgets.QTextEdit(self)

        self.spinbox_searchstep = QtWidgets.QSpinBox(self)
        self.spinbox_searchstep.setRange(0,200000)
        self.spinbox_searchstep.setValue(self.SOS_Manager.maxiter)

        self.txt_inputtg  = QtWidgets.QTextEdit(self)
        self.txt_extratg = QtWidgets.QTextEdit(self)

        self.shadow = QtWidgets.QPushButton(self)
        self.shadow.setStyleSheet('background-color: rgba(0,0,0,150)')
        #self.shadow.clicked.connect(self.displaySOSexit)
        self.display = QtWidgets.QLabel(self)
        self.display.setStyleSheet('border:2px groove gray')
        self.display.setAlignment(QtCore.Qt.AlignCenter)
        self.displayProgress = QtWidgets.QLabel(self)
        self.displayProgress.setStyleSheet('border:2px groove gray; color:white')
        self.displayProgress.setAlignment(QtCore.Qt.AlignCenter)
        qf = QtGui.QFont()
        qf.setPointSize(20)
        self.displayProgress.setFont(qf)
        self.shadow.hide()
        self.display.hide()
        self.displayProgress.hide()
    
        
    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        w = self.width()
        h = self.height()

        self.txt_input.setGeometry(w//80, h//60, w*7//10, h//18)
        
        self.btn_sos.setGeometry(w*7//10+w//40,h*34//80, w*21//80, h//18)

        for i in range(3):
            self.btn_displaymode[i].setGeometry(w*29//40 + i*w*21//240, h*32//40,w*10//120,h//20)
        self.txt_displayResult.setGeometry(w*29//40, h*34//40, w*21//80, h*8//60)

        self.spinbox_searchstep.setGeometry(w*36//40,h*59//120,w*3//40,h*5//120)
        self.txt_inputtg.setGeometry(w*29//40,h*45//80,w*21//80,h*5//60)
        self.txt_extratg.setGeometry(w*29//40,h*54//80,w*21//80,h*7//60)
        
        self.shadow.setGeometry(0,0,w,h)
        self.display.setGeometry(w//30,h//30,28*w//30,28*h//30)
        self.displayProgress.setGeometry(w//30,h//30,28*w//30,28*h//30)

        try:
            pix = QtGui.QPixmap('Formula.png').scaled(28*w//30,28*h//30,
                                    QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.display.setPixmap(pix)
        except:
            pass

        return super().resizeEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        w = self.width()    
        h = self.height()

        QtGui.QPainter.drawRect(QtGui.QPainter(self),w//80,h*16//180,w*7//10,h*322//360)
        #QtGui.QPainter.drawRect(QtGui.QPainter(self),w*7//10+w//40,h//60,w*21//80,h*14//40)
        QtGui.QPainter.drawText(QtGui.QPainter(self),w*25//80,h*320//360,w*31//80,h*33//360,
                                QtGui.Qt.AlignRight|QtGui.Qt.AlignBottom,self.SOS_Manager.rootsinfo)
        QtGui.QPainter.drawText(QtGui.QPainter(self),w*7//10+w//40,h*20//40,w*7//40,h//40,
                                    QtGui.Qt.AlignLeft,"Local Minima")
        QtGui.QPainter.drawText(QtGui.QPainter(self),w*57//80,h*20//40,w*7//40,h//40,
                                    QtGui.Qt.AlignRight,"Steps")
        QtGui.QPainter.drawText(QtGui.QPainter(self),w*29//40,h*43//80,w*7//40,h//40,
                                    QtGui.Qt.AlignLeft,"Additional Tangents")
        QtGui.QPainter.drawText(QtGui.QPainter(self),w*29//40,h*52//80,w*7//40,h//40,
                                    QtGui.Qt.AlignLeft,"Generated Tangents")
        
        #print('Hi')
        printPolyTriangle(self)
        printGridTriangle(self)

        return super().paintEvent(event)

    def displaySOS(self):
        w = self.width()
        h = self.height()
        self.shadow.show()
        self.shadow.update()

        if self.SOS_Manager.stage == 60:
            self.txt_displayResult.setText(self.SOS_Manager.sosresults[self.btn_displaymodeselect])
            self.displayProgress.hide()
            self.display.show()
            
            pix = QtGui.QPixmap('Formula.png').scaled(28*w//30,28*h//30,
                                    QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.display.setPixmap(pix)
            return 
        else:
            self.display.hide()
            txt = ''
            if self.SOS_Manager.stage >= 10:
                txt += 'Searching Roots...'
                if self.SOS_Manager.stage >= 20:
                    txt += '\nCreating Tangents...'
                    if self.SOS_Manager.stage >= 30:
                        txt += '\nComputing Decomposition...'
                        if self.SOS_Manager.stage == 30:
                            self.txt_extratg.setText('\n'.join(self.SOS_Manager.tangents))
                        elif self.SOS_Manager.stage < 50:
                            txt += '\n' + '\n'.join([f'Failed with degree {i}...' 
                                    for i in range(self.SOS_Manager.deg,self.SOS_Manager.stage-30+1)])
                        elif self.SOS_Manager.stage == 50:
                            txt += '\nSuccess\nRendering LaTeX...'
                        elif self.SOS_Manager.stage == 70:
                            txt += '\nFailed'
            self.displayProgress.setText(txt)
            self.displayProgress.show()
            self.displayProgress.repaint()
            return 

    def displaymodeselect(self):
        for i in range(3):
            if self.sender() == self.btn_displaymode[i]:
                if self.btn_displaymodeselect == i and i == 0:
                    self.interface = 1
                    self.SOS_Manager.stage = 60
                    self.displaySOS()
                else:
                    self.btn_displaymodeselect = i
                    self.txt_displayResult.setText(self.SOS_Manager.sosresults[i])
                    self.btn_displaymode[i].setStyleSheet("background-color: white") 
                    self.repaint()
            else:
                self.btn_displaymode[i].setStyleSheet("border: none") 
                self.repaint()
        
    def backToMain(self):
        self.shadow.hide()
        self.display.hide()
        self.displayProgress.hide()
        self.interface = 0
        self.repaint()
        

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        qtkey = QtGui.QKeyEvent.key(event)
        if qtkey == QtCore.Qt.Key_Return:
            if self.interface == 0:
                self.SOS_Manager.setPoly(self.txt_input.text())
                self.repaint()
            elif self.interface == 1 and self.SOS_Manager.stage >= 60:
                self.backToMain()
        elif qtkey == QtCore.Qt.Key_Escape:
            if self.interface == 1 and self.SOS_Manager.stage >= 60:
                self.backToMain()
            
        return super().keyPressEvent(event)

    @QtCore.Slot()
    def autosos(self):
        self.interface = 1
        self.SOS_Manager.tangents_default = [tg 
                        for tg in self.txt_inputtg.toPlainText().split('\n') if len(tg)>0]
        self.SOS_Manager.maxiter = self.spinbox_searchstep.value()
        self.SOS_Manager.GUI_SOS(self.txt_input.text())


if __name__ == "__main__":
    app = QApplication([])

    widget = MyWidget()
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec())