# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 00:02:12 2018
@author: frank
"""

import sys

from PyQt5.QtWidgets import QMainWindow, QTextEdit, QAction, QApplication, QFileDialog, QMessageBox, QProgressBar
from PyQt5.QtWidgets import QVBoxLayout,QHBoxLayout,QPushButton,QSizePolicy,QSplitter#, QCheckBox
from PyQt5.QtWidgets import QComboBox,QDialog, QLabel,QLineEdit#, QTableWidget,QStackedWidget
from PyQt5.QtWidgets import QDateTimeEdit,QDialogButtonBox
from PyQt5.QtWebKitWidgets import QWebView
from PyQt5.QtWebKit import QWebSettings
from PyQt5.QtGui import QIcon,QFont,QKeySequence#,QVBoxLayout,QHBoxLayout
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QThread
from PyQt5.Qt import QDateTime
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage,QPixmap

import cv2
import os

image_exts = ['.jpeg','.jpg','.png','.bmp']
video_exts = ['.mpg','.mpeg','.mp4','.avi', '.m4v']

import numpy as np

class Thread(QThread):
    changePixmap = pyqtSignal(QPixmap)

    def __init__(self, parent=None):
        QThread.__init__(self, parent=parent)
        self.filepath = None
        self.width = 640
        self.height = 480
        self.isfinished = False
        self.detector = None
        self.image = np.zeros((self.height,self.width,3), dtype=np.uint8)

    def isFinished(self):
        return self.isfinished
 
    def refresh(self):
        rgbImage = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
        convertToQtFormat = QPixmap.fromImage(convertToQtFormat)
        p = convertToQtFormat.scaled(self.width, self.height, Qt.KeepAspectRatio)
        self.changePixmap.emit(p)        
        
    def run(self):
        if not self.filepath or not os.path.exists(self.filepath):
            return
        print(self.filepath)
        filepath,filename=os.path.split(self.filepath)
        fname,ext = os.path.splitext(filename)
        print('ext:%s'%ext)
        if ext in image_exts:
            print('processing image...')
            frame = cv2.imread(self.filepath)
            frame = self.detector.markize(frame) if self.detector else frame
            rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
            convertToQtFormat = QPixmap.fromImage(convertToQtFormat)
            p = convertToQtFormat.scaled(self.width, self.height, Qt.KeepAspectRatio)
            self.changePixmap.emit(p)
            self.isfinished = True
            self.image = frame.copy()
            
        elif ext in video_exts:
            cap = cv2.VideoCapture(os.path.abspath(self.filepath))
            while True:
                ret, frame = cap.read()
                if ret is False:
                    break
                frame = self.detector.markize(frame) if self.detector else frame
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
                convertToQtFormat = QPixmap.fromImage(convertToQtFormat)
                p = convertToQtFormat.scaled(self.width, self.height, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
                self.image = frame.copy()
            self.isfinished = True

    def setdetector(self,detector):
        self.detector = detector
        
    def setpath(self,path):
        self.filepath = os.path.abspath(path)
        self.isfinished = False
        
    def setsize(self,width,height):
        self.width,self.height = width,height
   
class Dialog(QDialog):
    up_camera_signal = QtCore.pyqtSignal(QImage)
    def __init__(self, detector):
        super(Dialog, self).__init__(None)
        self.isfinished = False
        self.detector = detector
        self.initialize()

    def act_fileopen(self):
        filepath,extensions = QFileDialog.getOpenFileName(self, r'File Open','',"Image/Video files (*.jpeg *.jpg *.png *.avi *.mpeg *.mpg *.mp4 *.m4v)")
        self.thread.setpath(filepath)
        self.thread.start()
 
    def closeEvent(self, event):
        
        if self.isfinished:
            self.deleteLater()
            return
        
        reply = QMessageBox.question(self, 'warning',
                                     "Are you sure to quit dialog?", QMessageBox.Yes | 
                                     QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.thread.quit()
            self.deleteLater()
            event.accept()
        else:
            event.ignore()
            
    def initialize(self):
        ###
        self.up_camera = None
        ###
        self.label = QLabel(self)
        #label.move(180, 120)
        self.label.resize(350, 350)
        button = QPushButton(self)
        button.setText('open')
        button.released.connect(self.act_fileopen)
        ###
        vbox = QVBoxLayout()
        vbox.addStretch(1)
        vbox.addWidget(self.label)
        vbox.addWidget(button)
        self.setLayout(vbox)
        ###
        self.thread = Thread(self)
        self.thread.setdetector(self.detector)
        self.thread.changePixmap.connect(self.label.setPixmap)
        ###
        self.setMinimumSize(500,350)
        self.setWindowTitle("Demo")  
        self.show()

    def resizeEvent(self, event):
        #self.resized.emit()
        width = self.frameGeometry().width()
        height = self.frameGeometry().height()
        self.thread.setsize(width-30,height-30)
        if not self.thread.isFinished():
            self.thread.refresh()
        self.label.setGeometry(QtCore.QRect(0, height-20, width-20, 20))
        self.label.setVisible(True)
        self.update()
        return super(Dialog, self).resizeEvent(event)

    def setdetector(self,detector):
        self.detector = detector
        self.thread.detector(detector)

def opencv2skimage(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
def main():
    from detector import VehicleDetector
    import time
    #
    import gc
    gc.collect()
    det = VehicleDetector()
    # Create an PyQT5 application object.
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    ex = Dialog(detector=det)
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    main()