# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 00:02:12 2018
@author: frank
"""
import cv2
import os
import sys

#from PyQt5.QtWidgets import QMainWindow, QTextEdit, QAction, QApplication, QMessageBox, QProgressBar
from PyQt5.QtWidgets import QApplication,QMessageBox,QFileDialog
from PyQt5.QtWidgets import QVBoxLayout,QHBoxLayout,QPushButton
#from PyQt5.QtWidgets import QSizePolicy,QSplitter, QCheckBox
from PyQt5.QtWidgets import QDialog, QLabel
from PyQt5.QtWidgets import QComboBox
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
#import random
#import time

from detector import VehicleDetector

detector = VehicleDetector()

#from alpr_hyper import recognize_plate

from entity import Entity,EntityManager

EM = EntityManager()

from constants import image_exts,video_exts
from constants import labels
from constants import bg_color
from constants import screen_size

from util import drawMark,drawRectBox,drawStatus

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler('vehicle.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)

logger.info('Start recording ssd results...')

(cv_major_ver, cv_minor_ver, cv_subminor_ver) = (cv2.__version__).split('.')

global fps
global fourcc
global writer
if int(cv_major_ver) < 3 :
    fourcc = cv2.cv.CV_FOURCC('D','I','V','X')
    #writer = cv2.VideoWriter('record.AVI',	fourcc, fps, (width,height), 1)
else:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #writer = cv2.VideoWriter('record.AVI',	fourcc, fps, (width,height), True)

from constants import overtime_color,restricted_color,wrong_color,crossover_color

class Thread(QThread):
    changePixmap = pyqtSignal(QPixmap)

    def __init__(self, parent=None):
        QThread.__init__(self, parent=parent)
        self.filepath = None
        self.width, self.height = 640,480
        self.wratio,self.hratio=1,1
        self.fps = 15
        self.timelimit=2
        
        self.redline_pt1,self.redline_pt2 = (0,0),(0,0)
        self.direction_pt1,self.direction_pt2 = (0,0),(0,0)
        self.direction = (0,0)
        self.roiRect = QRect()
        
        self.rline_pt1,self.rline_pt2=(0,0),(0,0)
        self.restricted = (0,0,0,0)
        
        self.curFrame = np.zeros((self.height,self.width,3), dtype=np.uint8)
        self.out = None
        
        self.initialize()

    def initialize(self):
        # 
        self.isdrawing = False
        self.isfinished = True
        self.isfirstscreen_displayed = False
        self.ispaused = True
        self.isprocessing = False
        self.ismarked = False
        self.isinterrupted = False
        
        self.enable_license_plate_recognition = True
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
            self.restricted = (x1,y1,x2-x1,y2-y1)
            #print(x1,y1,x2,y2)            
            frame = cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), int(self.wratio if self.wratio > 2 else 2))
            print('drawing redline')
            x1,y1,x2,y2=self.redline_pt1[0]*self.wratio,self.redline_pt1[1]*self.hratio,\
                        self.redline_pt2[0]*self.wratio,self.redline_pt2[1]*self.hratio
            self.rline_pt1,self.rline_pt2 = (x1,y1),(x2,y2)
            cv2.line(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,255), 5)
            print('drawing arrowline')
            x1,y1,x2,y2=self.direction_pt1[0]*self.wratio,self.direction_pt1[1]*self.hratio,\
                        self.direction_pt2[0]*self.wratio,self.direction_pt2[1]*self.hratio
            self.direction = (x2-x1,y2-y1)
            cv2.arrowedLine(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,255), 5)
        ### Processing
        if self.isprocessing:
            rclasses, rscores, rbboxes =  detector.process_image(frame)
            candidates = []
            for i in range(rclasses.shape[0]):
                if labels[rclasses[i]] not in ['bus','car','motorbike','pedestrian']:
                    continue
                top = int(rbboxes[i, 0] * h)
                left = int(rbboxes[i, 1] * w)
                bottom = int(rbboxes[i, 2] * h)
                right = int(rbboxes[i, 3] * w)
                width = int((right-left)/4)*4
                height= int((bottom-top)/4)*4
                label = 'vehicle' if labels[rclasses[i]] in ['car','bus'] else labels[rclasses[i]]
                candidates.append([top,left,bottom,right,width,height,label])
            EM.refresh(candidates)
            for entity in EM.get_alives():
                # object detection and display result
                rect = entity.get_rect()
                label = entity.get_label()
                index = entity.getIndex()
                print('%s,%s,%s'%(rect,label,index))
                drawMark(frame,rect,label)
                level=0
                if entity.is_in_restricted(self.restricted):
                    elapsed_time = entity.calc_time(self.fps)
                    if elapsed_time > self.timelimit:
                        inform_str = '%dth target is stuck.'%index
                        #drawMark(frame,rect,label,overtime_color)
                        drawStatus(frame,rect,inform_str,level=level)
                    else:
                        inform_str = '%dth target is in restricted zone.'%index
                        #drawMark(frame,rect,label,restricted_color)
                        drawStatus(frame,rect,inform_str,level=level)
                    level+=1
                if self.direction and entity.is_in_wrong_direction(self.direction):
                    inform_str = '%dth target is in wrong direction.'%index
                    #drawMark(frame,rect,label,wrong_color)
                    drawStatus(frame,rect,inform_str,level=level)
                    level+=1
                if self.direction and self.rline_pt1 and self.rline_pt2 and \
                   entity.is_over_red_line(self.redline_pt1,self.redline_pt2,self.direction):
                    inform_str = '%dth target crossed the red line.'%index
                    #drawMark(frame,rect,label,crossover_color)
                    drawStatus(frame,rect,inform_str,level=level)
                    level+=1
                self.out.write(frame) if self.out is not None else None
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
            self.out = cv2.VideoWriter('out.avi',fourcc, 10, (screen_size[0],screen_size[1]),True)
            cap = cv2.VideoCapture(os.path.abspath(self.filepath))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width,height = cap.get(cv2.CAP_PROP_FRAME_WIDTH),\
                           cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            EM.setScreenSize([width,height])
            cnt=0
            pick_rate = int(fps/self.fps)
            print('pick_rate:%d'%pick_rate)
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
                cnt += 1
                if cnt%pick_rate != 1:
                    continue
                ### Handling Exception
                h,w,c=frame.shape
                frame = cv2.resize(frame,(int(w/4)*4,int(h/4)*4))
                ### Processing
                self.process(frame.copy())
                ### Handling Idle
                self.curFrame = frame.copy()
            self.isfinished = True
        self.out.release() if self.out is not None else None

    def setDirection(self,pt1,pt2):
        self.direction_pt1 = (pt1.x(),pt1.y())
        self.direction_pt2 =(pt2.x(),pt2.y())
        self.direction=[pt2.x()-pt1.x(),pt2.y()-pt1.y()]

    def setRedLinePts(self,pt1,pt2):
        self.redline_pt1 = (pt1.x(),pt1.y())
        self.redline_pt2 = (pt2.x(),pt2.y())
        
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

from handy.misc import switch
    
class Dialog(QDialog):
    up_camera_signal = QtCore.pyqtSignal(QImage)
    def __init__(self, parent = None):
        super(Dialog, self).__init__(parent)
        self.isFinished = False
        self.isDragging = False
        self.curFilePath = None
        ###
        self.selected_option = 'Direction'
        self.pos_start,self.pos_end=QPoint(),QPoint()
        self.line = QLine()
        self.roiRect = QRect()
        ###
        self.initialize()
        
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
        self.processbutton.setText('enable') if not self.thread.isProcessing() else self.processbutton.setText('disable')
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
             
        ### ComboBox
        self.combo = QComboBox(self)
        self.combo.setMinimumWidth(150)
        self.combo.setMinimumHeight(20)
        self.combo.addItem('Direction')
        self.combo.addItem('Forbidden')
        self.combo.addItem('RedLine')
        self.combo.currentIndexChanged.connect(self.selection_change)
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
        hbox.addStretch(4)
        
        hbox.addWidget(self.openbutton)
        hbox.addWidget(self.playbutton)
        hbox.addWidget(self.processbutton)
        hbox.addWidget(self.combo)
        hbox.addWidget(self.drawbutton)
        
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
            self.pos_start= event.pos()
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
        #print(left,right,top,bottom)
        #print(imgRect.topLeft())
        self.pos_end=event.pos()
        ### project to label
        for case in switch(self.selected_option):
            if case('Direction'):
                self.thread.setDirection(self.pos_start,self.pos_end)
                break
            if case('Forbidden'):
                self.roiRect.setTopLeft(self.pos_start)#-imgRect.TopLeft())
                self.roiRect.setBottomRight(self.pos_end)#-imgRect.TopLeft())
                self.thread.setRect(self.roiRect)
                break
            if case('RedLine'):
                self.thread.setRedLinePts(self.pos_start,self.pos_end)
                break
        self.thread.refresh()
        ###        
        self.update()
         
    def mouseReleaseEvent(self, event):
        self.isDragging = False

    def paintEvent(self,event):
        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)        
        for case in switch(self.selected_option):
            if case('Direction'):
                painter.setPen(QPen(QBrush(Qt.red), 1, Qt.DashLine))
                painter.drawLine(QLine(self.pos_start,self.pos_end))
                break
            if case('Forbidden'):
                painter.setPen(QPen(QBrush(Qt.green), 1, Qt.DashLine))
                painter.drawRect(self.roiRect)
                break
            if case('RedLine'):
                painter.setPen(QPen(QBrush(Qt.yellow), 1, Qt.DashLine))
                painter.drawLine(QLine(self.pos_start,self.pos_end))
                break
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
    
    def selection_change(self,i):
        self.selected_option = self.combo.currentText()
        '''
        for case in switch(self.selected_option):
            if case('Direction'):
                self.drawType = 0
                break
            if case('Forbidden'):
                self.drawType = 1
                break
            if case('RedLine'):
                self.drawType = 2
                break
        '''
        
if __name__ == "__main__":
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    ex = Dialog()
    sys.exit(app.exec_())
    