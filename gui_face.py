# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 00:02:12 2018
@author: frank
"""
import cv2
import os
import sys

image_exts = ['.jpeg','.jpg','.png','.bmp']
video_exts = ['.mpg','.mpeg','.mp4','.avi','m4v']

#from PyQt5.QtWidgets import QMainWindow, QTextEdit, QAction, QApplication, QMessageBox, QProgressBar
from PyQt5.QtWidgets import QApplication,QMessageBox,QFileDialog
from PyQt5.QtWidgets import QVBoxLayout,QHBoxLayout,QPushButton
#from PyQt5.QtWidgets import QSizePolicy,QSplitter, QCheckBox
from PyQt5.QtWidgets import QDialog, QLabel
#from PyQt5.QtWidgets import QComboBox,,LineEdit, QTableWidget,QStackedWidget
#from PyQt5.QtWidgets import QDateTimeEdit,QDialogButtonBox
#from PyQt5.QtWebKitWidgets import QWebView
#from PyQt5.QtWebKit import QWebSettings
#from PyQt5.QtGui import QIcon,QFont,QKeySequence#,QVBoxLayout,QHBoxLayout
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, pyqtSignal, QThread#, QObject
#from PyQt5.Qt import QDateTime
#from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtGui import QPainter, QPen, QBrush#, QPainterPath
from PyQt5.QtCore import QLine, QPoint, QRect#,QSize, 

import numpy as np

import face_recognition
from load_faces import queryface

class Thread(QThread):
    changePixmap = pyqtSignal(QPixmap)

    def __init__(self, parent=None):
        QThread.__init__(self, parent=parent)
        self.filepath = None
        self.width, self.height = 640,480
        self.wratio,self.hratio=1,1
        self.curFrame = np.zeros((self.height,self.width,3), dtype=np.uint8)
        self.initialize()

    def initialize(self):
        ###
        self.isdrawing = False
        self.isfinished = True
        self.isfirstscreen_displayed = False
        self.ispaused = True
        self.isprocessing = False
        self.ismarked = False
        self.isinterrupted = False
        ###        
        
    def isDrawing(self):
        return self.isdrawing
        
    def isFinished(self):
        return self.isfinished

    def isMarked(self):
        return self.ismarked
    
    def isPaused(self):
        return self.ispaused

    def isProcessing(self):
        return self.isprocessing
    
    def process(self,frame):
        ### Calculate Ratio
        h,w,c=frame.shape
        self.wratio,self.hratio=w/self.width,h/self.height
        ### Drawing
        if self.isMarked():
            print('drawing rectangle')
            x1,y1,x2,y2=self.roiRect.left()*self.wratio,self.roiRect.top()*self.hratio,self.roiRect.right()*self.wratio,self.roiRect.bottom()*self.hratio
            print(x1,y1,x2,y2)            
            frame = cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), int(self.wratio if self.wratio > 2 else 2))
        ### Processing
        if self.isprocessing:
            rgb_frame = frame[:, :, ::-1]
            # Find all the faces and face enqcodings in the frame of video
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            # Loop through each face in this frame of video
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # See if the face is a match for the known face(s)
                name = queryface(face_encoding)
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        ### Projecting to Label    
        rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        convertToQtFormat = QImage(rgbImage.data, w, h, QImage.Format_RGB888)
        convertToQtFormat = QPixmap.fromImage(convertToQtFormat)
        p = convertToQtFormat.scaled(self.width, self.height)#, Qt.KeepAspectRatio)
        self.changePixmap.emit(p)
        
    def refresh(self):
        ### projecting 
        self.process(self.curFrame.copy())
        
    def run(self):
        if not self.filepath or not os.path.exists(self.filepath):
            return
        print(self.filepath)
        filepath,filename=os.path.split(self.filepath)
        fname,ext = os.path.splitext(filename)
        print('ext:%s'%ext)
        if ext.lower() in image_exts:
            print('processing image...')
            self.isfinished = False
            frame = cv2.imread(self.filepath)
            if frame is None:
                print('no file exists')
                return
            ### Handling Exception
            h,w,c=frame.shape
            frame = cv2.resize(frame,(int(w/4)*4,int(h/4)*4))
            ### Processing
            self.process(frame.copy())
            ### Handling Idle
            self.isfinished = True
            self.curFrame = frame.copy()
            
        elif ext.lower() in video_exts:
            cap = cv2.VideoCapture(os.path.abspath(self.filepath))
            self.isfirstscreen_displayed = False
            while True:
                ### check if the first screen or not
                if not self.isfirstscreen_displayed:
                    self.isfirstscreen_displayed=True
                ### check if paused key pressed or not
                elif self.ispaused:
                    continue
                ret, frame = cap.read()
                if not ret:
                    break
                ### Handling Exception
                h,w,c=frame.shape
                frame = cv2.resize(frame,(int(w/4)*4,int(h/4)*4))
                ### Processing
                self.process(frame.copy())
                ### Handling Idle
                self.curFrame = frame.copy()
            self.isfinished = True
            
    def setRect(self,rect):
        self.roiRect = rect
        
    def setpath(self,path):
        self.filepath = os.path.abspath(path)
        self.initialize()

    def setPaused(self,flag=True):
        self.ispaused = flag
        
    def setProcessing(self,flag=True):
        self.isprocessing = flag
    
    def setMarked(self,flag=True):
        self.ismarked = flag
        if not self.ismarked:
            self.setRect(QRect())
        
    def setDrawing(self,flag):
        self.isdrawing = flag
        
    def setsize(self,width,height):
        self.width,self.height = width,height
    
class Dialog(QDialog):
    up_camera_signal = QtCore.pyqtSignal(QImage)
    def __init__(self, parent = None):
        super(Dialog, self).__init__(parent)
        self.isFinished = False
        self.isDragging = False
        self.line = QLine()
        self.roiRect = QRect()
        self.initialize()
        self.curFilePath = None

    def act_draw(self):
        self.thread.setMarked(not self.thread.isMarked())
        self.drawbutton.setText('draw') if not self.thread.isMarked() else self.drawbutton.setText('erase')
        self.thread.refresh()
    
    def act_fileopen(self):
        ispaused = self.thread.isPaused()
        self.thread.setPaused(True)
        self.playbutton.setText('play') if self.thread.isPaused() else self.playbutton.setText('paused')
        filepath,extensions = QFileDialog.getOpenFileName(self, r'File Open','',"Image/Video Files (*.jpeg *.jpg *.png *.avi *.mpeg *.mpg *.mp4 *m4v)")
        if not self.curFilePath:
            self.curFilePath = filepath
        if filepath == self.curFilePath:
            print('continuing original playing')
            self.thread.setPaused(ispaused)
            self.playbutton.setText('play') if self.thread.isPaused() else self.playbutton.setText('paused')
        else:
            print('exiting original playing...')
            self.thread.terminate()
            self.curFilePath = filepath
        self.thread.setpath(filepath)
        self.thread.start()

    def act_play(self):
        self.thread.setPaused(not self.thread.isPaused())
        self.playbutton.setText('play') if self.thread.isPaused() else self.playbutton.setText('paused')
        self.update()

    def act_process(self):
        self.thread.setProcessing(not self.thread.isProcessing())
        self.processbutton.setText('enable') if not self.thread.isProcessing() else self.processbutton.setText('disable')
        self.update()
        
    def closeEvent(self, event):
        
        if self.isFinished:
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
        ### Label
        self.label = QLabel(self)
        #label.move(180, 120)
        self.label.resize(350, 350)
        
        ### Buttons
        self.openbutton = QPushButton(self)
        self.openbutton.setText('open')
        self.openbutton.released.connect(self.act_fileopen)
        self.playbutton = QPushButton(self)
        self.playbutton.setText('play')
        self.playbutton.released.connect(self.act_play)        
        self.drawbutton = QPushButton(self)
        self.drawbutton.setText('draw')
        self.drawbutton.released.connect(self.act_draw)
        self.processbutton = QPushButton(self)
        self.processbutton.setText('enable')
        self.processbutton.released.connect(self.act_process)  
        
        ### Horizontal Layout
        hbox = QHBoxLayout()
        hbox.addStretch(3)
        hbox.addWidget(self.openbutton)
        hbox.addWidget(self.playbutton)
        hbox.addWidget(self.drawbutton)
        hbox.addWidget(self.processbutton)
        
        ### Vertical Layout
        vbox = QVBoxLayout()
        vbox.addStretch(1)
        vbox.addWidget(self.label)
        vbox.addLayout(hbox)
        
        ### Arrange
        self.setLayout(vbox)
        
        ### Creating and Connecting Thread
        self.thread = Thread(self)
        self.thread.changePixmap.connect(self.label.setPixmap)
        ### Resize Windows
        self.setMinimumSize(1080,640)
        self.setWindowTitle("Demo")
        self.showMaximized()
        self.show()

    def initializeROI(self):
        self.roiRect.setTopLeft(QPoint(0,0))
        self.roiRect.setBottomRight(QPoint(0,0))

    def mouseDoubleClickEvent(self, event):
        if self.isMaximized():
            pass#self.showMinimized()
        else:
            self.showMaximized()
            
    def mousePressEvent(self, event):
 
        imgRect = self.label.contentsRect()
        if imgRect.contains(event.pos()):
            self.isDragging = True
            self.initializeROI()
            self.drag_offset = imgRect.topLeft() - event.pos()
            self.point_s = event.pos()
        else:
            self.isDragging = False
     
    def mouseMoveEvent(self, event):
     
        if not self.isDragging:
            return
        
        imgRect = self.label.contentsRect()
        ###
        left = imgRect.left()
        right = imgRect.right()
        top = imgRect.top()
        bottom = imgRect.bottom()
        ###
        point = event.pos() + self.drag_offset
        point.setX(max(left, min(point.x(), right)))
        point.setY(max(top, min(point.y(), bottom)))
        ### set roiRect
        print(left,right,top,bottom)
        print(imgRect.topLeft())
        self.roiRect.setTopLeft(self.point_s)#-imgRect.TopLeft())
        self.roiRect.setBottomRight(event.pos())#-imgRect.TopLeft())
        ### project to label
        self.thread.setRect(self.roiRect)
        self.thread.refresh()
        ###        
        self.update()
         
    def mouseReleaseEvent(self, event):
        self.isDragging = False

    def paintEvent(self,event):
        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QBrush(Qt.green), 1, Qt.DashLine))
        painter.drawRect(self.roiRect)
        painter.end()
        
    def resizeEvent(self, event):
        #self.resized.emit()
        width = self.frameGeometry().width()
        height = self.frameGeometry().height()
        self.thread.setsize(width-30,height-30)
        #self.thread.setsize(width,height)
        if self.thread.isFinished():
            self.thread.refresh()
        self.label.setGeometry(QtCore.QRect(0, height-20, width-20, 20))
        #self.label.setGeometry(QtCore.QRect(0, height, width, 0))
        self.label.setVisible(True)
        self.update()
        return super(Dialog, self).resizeEvent(event)
        
if __name__ == "__main__":
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    ex = Dialog()
    sys.exit(app.exec_())
    