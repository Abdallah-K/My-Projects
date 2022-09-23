
from PyQt5 import QtCore, QtGui, QtWidgets
import test
from PyQt5.QtCore import QTimer
from app import  Ui_MainWindow2


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(531, 423)
        MainWindow.setStyleSheet("background-color: rgb(0, 255, 0);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.logo = QtWidgets.QLabel(self.centralwidget)
        self.logo.setGeometry(QtCore.QRect(80, 80, 371, 251))
        self.logo.setStyleSheet("image: url(:/logo/logo.png);")
        self.logo.setText("")
        self.logo.setObjectName("logo")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(110, 350, 321, 51))
        self.progressBar.setStyleSheet("QProgressBar{\n"
"border:2px solid grey;\n"
"border-radius:10px;\n"
"text-align:center;\n"
"color:white;\n"
"}\n"
"QProgressBar::chunk{\n"
"background-color: rgb(0, 0, 0);\n"
"width:2.15px;\n"
"margin:1px;\n"
"}")
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(50, 10, 431, 41))
        font = QtGui.QFont()
        font.setFamily("Segoe Script")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.timer = QTimer()
        self.timer.timeout.connect(self.run_bar)
        self.timer.start(1000)
        self.timer2 = QTimer()
        self.timer2.singleShot(6220, self.open)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Arabic letters Recognition"))
        MainWindow.setWindowIcon(QtGui.QIcon("logo.png"))
        self.label_2.setText(_translate("MainWindow", "Convolution Neural Network"))
    
    def run_bar(self):
        bar=self.progressBar.value()
        self.progressBar.setValue(bar+20)

    
    def open(self):
        self.MainWindow2 = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow2()
        self.ui.setupUi(self.MainWindow2)
        self.MainWindow2.show()



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
