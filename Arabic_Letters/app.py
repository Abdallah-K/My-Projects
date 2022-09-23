

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
from keras.models import load_model
from PyQt5.QtWidgets import QFileDialog, QDialog
from PyQt5.QtGui import QPixmap
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import glob
import math


model=load_model("final.h5")
arabic_dic={0:'أ',
	    1:'ب',
	    2:'ب في اول الكلمة',
	    3:'ب في وسط الكلمة',
	    4:'ت',
	    5:'ت في أخر الكلمة',
	    6:'ث',
	    7:'ث في أول الكلمة',
	    8:'ج',
        9:'ج في أول الكلمة',
	    10:'ح',
	    11:'ح فب أول الكلمة',
	    12:'ح في وسط الكلمة',
	    13:'خ',
	    14:'د',
 	    15:'ذ',
	    16:'ر',
	    17:'ر في أخر الكلمة',
	    18:'ز',
	    19:'س',
	    20:'ش',
	    21:'ص',
	    22:'ص في أول الكلمة',
	    23:'ض',
	    24:'ط',
	    25:'ظ',
	    26:'ع',
	    27:'ع في أول الكلمة',
	    28:'غ',
	    29:'غ في وسط الكلمة',
	    30:'ف',
	    31:'ف في وسط الكلمة',
	    32:'ق',
	    33:'ك',
  	    34:'ك في أول الكلمة',
	    35:'ل',
	    36:'م',
	    37:'م في أول الكلمة',
	    38:'م في وسط الكلمة',
	    39:'ن',
	    40:'ن في أول الكلمة',
	    41:'ن في وسط الكلمة',
	    42:'ه',
	    43:'و',
	    44:'ي'}


class Ui_MainWindow2(QDialog):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(583, 500)
        MainWindow.setStyleSheet("background-color: rgb(0, 255, 0);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(100, 0, 381, 51))
        font = QtGui.QFont()
        font.setFamily("Snap ITC")
        font.setPointSize(18)
        self.label.setFont(font)
        self.label.setStyleSheet("color:rgb(255, 255, 255);")
        self.label.setObjectName("label")
        self.dir = QtWidgets.QLineEdit(self.centralwidget)
        self.dir.setGeometry(QtCore.QRect(140, 60, 301, 51))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.dir.setFont(font)
        self.dir.setStyleSheet("")
        self.dir.setObjectName("dir")
        self.photo = QtWidgets.QLabel(self.centralwidget)
        self.photo.setGeometry(QtCore.QRect(190, 130, 201, 331))
        self.photo.setText("")
        self.photo.setPixmap(QtGui.QPixmap("../Arabic letters/logo.png"))
        self.photo.setScaledContents(True)
        self.photo.setObjectName("photo")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 120, 161, 341))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox.setFont(font)
        self.groupBox.setStatusTip("")
        self.groupBox.setObjectName("groupBox")
        self.create = QtWidgets.QPushButton(self.groupBox)
        self.create.setGeometry(QtCore.QRect(10, 50, 141, 51))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.create.setFont(font)
        self.create.setToolTip("")
        self.create.setStyleSheet("color:white;\n"
"background-color: rgb(58, 58, 58);")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../Arabic letters/dataset1.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.create.setIcon(icon)
        self.create.setIconSize(QtCore.QSize(25, 20))
        self.create.setObjectName("create")
        self.clear = QtWidgets.QPushButton(self.groupBox)
        self.clear.setGeometry(QtCore.QRect(10, 140, 141, 51))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.clear.setFont(font)
        self.clear.setStyleSheet("color:white;\n"
"background-color: rgb(58, 58, 58);")
        self.clear.setObjectName("clear")
        self.remove = QtWidgets.QPushButton(self.groupBox)
        self.remove.setGeometry(QtCore.QRect(10, 230, 141, 51))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.remove.setFont(font)
        self.remove.setToolTip("")
        self.remove.setStyleSheet("color:white;\n"
"background-color: rgb(58, 58, 58);")
        self.remove.setIconSize(QtCore.QSize(25, 20))
        self.remove.setObjectName("remove")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(410, 120, 161, 341))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.upload = QtWidgets.QPushButton(self.groupBox_2)
        self.upload.setGeometry(QtCore.QRect(10, 50, 141, 51))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.upload.setFont(font)
        self.upload.setToolTip("")
        self.upload.setStyleSheet("color:white;\n"
"background-color: rgb(58, 58, 58);")
        self.upload.setIconSize(QtCore.QSize(25, 20))
        self.upload.setObjectName("upload")
        self.show = QtWidgets.QPushButton(self.groupBox_2)
        self.show.setGeometry(QtCore.QRect(10, 140, 141, 51))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.show.setFont(font)
        self.show.setToolTip("")
        self.show.setStyleSheet("color:white;\n"
"background-color: rgb(58, 58, 58);")
        self.show.setIconSize(QtCore.QSize(25, 20))
        self.show.setObjectName("show")
        self.predict = QtWidgets.QPushButton(self.groupBox_2)
        self.predict.setGeometry(QtCore.QRect(10, 230, 141, 51))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.predict.setFont(font)
        self.predict.setToolTip("")
        self.predict.setStyleSheet("color:white;\n"
"background-color: rgb(58, 58, 58);")
        self.predict.setObjectName("predict")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.create.clicked.connect(self.dataset)
        self.upload.clicked.connect(self.browse_ima)
        self.predict.clicked.connect(self.pred)
        self.show.clicked.connect(self.predcv)
        self.clear.clicked.connect(self.delete)
        self.remove.clicked.connect(self.exit)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Arabic letters Recognition"))
        MainWindow.setWindowIcon(QtGui.QIcon("logo.png"))
        self.label.setText(_translate("MainWindow", "Arabic Letters Recognition"))
        self.dir.setToolTip(_translate("MainWindow", "Dataset"))
        self.dir.setStatusTip(_translate("MainWindow", "Name Of Dataset"))
        self.dir.setPlaceholderText(_translate("MainWindow", "Enter The Name Of Dataset"))
        self.photo.setStatusTip(_translate("MainWindow", "Photo"))
        self.groupBox.setTitle(_translate("MainWindow", "Dataset"))
        self.create.setStatusTip(_translate("MainWindow", "Create Dataset"))
        self.create.setText(_translate("MainWindow", "Create"))
        self.clear.setStatusTip(_translate("MainWindow", "Clear Dataset"))
        self.clear.setText(_translate("MainWindow", "Clear"))
        self.remove.setStatusTip(_translate("MainWindow", "Delete Dataset"))
        self.remove.setText(_translate("MainWindow", "Delete "))
        self.groupBox_2.setTitle(_translate("MainWindow", "Image processing"))
        self.upload.setStatusTip(_translate("MainWindow", "Upload image"))
        self.upload.setText(_translate("MainWindow", "Upload"))
        self.show.setStatusTip(_translate("MainWindow", "Display images"))
        self.show.setText(_translate("MainWindow", "Show"))
        self.predict.setStatusTip(_translate("MainWindow", "Predict the images"))
        self.predict.setText(_translate("MainWindow", "Predict"))


    def dataset(self):
           dir=self.dir.text()
           if(dir == ""):
                msg=QMessageBox()
                msg.setWindowTitle("CNN")
                msg.setText("Please enter name of dataset...")
                msg.setWindowIcon(QtGui.QIcon("dataset1.png"))
                msg.setIcon(QMessageBox.Warning)
                x=msg.exec_()
           else:
                os.mkdir(dir)
                msg = QMessageBox()
                msg.setWindowTitle("Create Dataset")
                msg.setWindowIcon(QtGui.QIcon("dataset1.png"))
                msg.setText(f" Dataset {dir} has been created!")
                msg.setIcon(QMessageBox.Information)
                msg.setStandardButtons(QMessageBox.Cancel |
                               QMessageBox.Ok)
                msg.setDefaultButton(QMessageBox.Ok)
                msg.setDetailedText(f"In {dir} you can put all the letters that you would like to predict it..")
                msg.buttonClicked.connect(self.pop_button)

                x = msg.exec_()

    def pop_button(self, i):
        print(i.text())
    

    def browse_ima(self):
            dir=self.dir.text()
            if(dir == ""):
                msg=QMessageBox()
                msg.setWindowTitle("Upload Image")
                msg.setWindowIcon(QtGui.QIcon("logo.png"))
                msg.setText("Please enter name of dataset...")
                msg.setIcon(QMessageBox.Warning)
                x=msg.exec_()
            else:
                fname = QFileDialog.getOpenFileName(
                    self, 'open image', f'c\\Users\HP\Desktop\\', 'image files (*.png)')

                imagepath = fname[0]
                piximap = QPixmap(imagepath)
                self.photo.setPixmap(QPixmap(piximap))
                self.resize(piximap.width(), piximap.height())
    

    def pred(self):
            dir=self.dir.text()
            if(dir == ""):
                msg=QMessageBox()
                msg.setWindowTitle("Predict")
                msg.setText("Please enter name of dataset to predict it...")
                msg.setWindowIcon(QtGui.QIcon("logo.png"))
                msg.setIcon(QMessageBox.Warning)
                x=msg.exec_()
            else:
                for file in os.listdir(dir):
                    img=image.load_img(dir+'//' + file,target_size=(180,180))
                    img_array = keras.preprocessing.image.img_to_array(img)
                    img_array = tf.expand_dims(img_array, 0)
                    predictions = model.predict(img_array)
                    score = tf.nn.softmax(predictions[0])
                    name=arabic_dic[np.argmax(score)]
                    msg=QMessageBox()
                    msg.setWindowTitle("Prediction")
                    msg.setWindowIcon(QtGui.QIcon("logo.png"))
                    msg.setText(f"هذا هو حرف {name} بنسبة %{math.floor(100*np.max(score))}")
                    msg.setIcon(QMessageBox.Information)
                    x=msg.exec_()



    def predcv(self):
        dir=self.dir.text()
        if(dir == ""):
            msg=QMessageBox()
            msg.setWindowTitle("Predict")
            msg.setText("Please enter name of dataset to show it...")
            msg.setWindowIcon(QtGui.QIcon("logo.png"))
            msg.setIcon(QMessageBox.Warning)
            x=msg.exec_()
        else:
            path=glob.glob(f"{dir}/*")
            for file in path:
                img=cv2.imread(file)
                plt.figure(file)
                plt.imshow(img)
                plt.show()
    

    def delete(self):
            dir=self.dir.text()
            fol=f"{dir}/"
            if(dir == ""):
                msg=QMessageBox()
                msg.setWindowTitle("Clear Dataset")
                msg.setWindowIcon(QtGui.QIcon("dataset1.png"))
                msg.setText("Please enter name of dataset...")
                msg.setIcon(QMessageBox.Warning)
                x=msg.exec_()
            else:
                for file in os.listdir(fol):
                    if file.endswith(".png"):
                        os.unlink(fol+file)
                        msg = QMessageBox()
                msg.setWindowTitle("Reset")
                msg.setText("your data has been reseted...")
                msg.setWindowIcon(QtGui.QIcon("dataset1.png"))
                msg.setIcon(QMessageBox.Information)
                msg.setStandardButtons(QMessageBox.Ok |
                               QMessageBox.Cancel)
                x = msg.exec_()


    def exit(self):
        dir=self.dir.text()
        if(dir == ""):
            msg=QMessageBox()
            msg.setWindowTitle("Clear Dataset")
            msg.setWindowIcon(QtGui.QIcon("dataset1.png"))
            msg.setText("Please enter name of dataset...")
            msg.setIcon(QMessageBox.Warning)
            x=msg.exec_()
        else:
            os.rmdir(dir)
            msg=QMessageBox()
            msg.setWindowTitle("Delete Dataset")
            msg.setWindowIcon(QtGui.QIcon("dataset1.png"))
            msg.setText(f"Dataset {dir} has been deleted...")
            msg.setIcon(QMessageBox.Information)
            x=msg.exec_()



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow2 = QtWidgets.QMainWindow()
    ui = Ui_MainWindow2()
    ui.setupUi(MainWindow2)
    MainWindow2.show()
    sys.exit(app.exec_())
