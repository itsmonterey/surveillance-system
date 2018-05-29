# -*- coding: utf-8 -*-
"""
Created on Sun May 27 23:38:11 2018

@author: stephen
"""

import cv2
#import imp
#import sys
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import numpy as np

#imp.reload(sys)
fontC = ImageFont.truetype("./Font/platech.ttf", 14, 0)

from constants import left_margin,top_margin
from constants import label_height,label_length
from constants import __top_margin__,__left_margin__
from constants import text_color,bg_color

font = cv2.FONT_HERSHEY_DUPLEX
ispaused=False        

def drawMark(frame,rect,label,bg_colr=bg_color,fg_colr=text_color,direction=True):#top-down if True,bottom-up if False
    left,top,width,height = rect
    right,bottom = left+width,top+height
    cv2.rectangle(frame,(left,top),(right,bottom),bg_color,3)
    if (right-left) > (label_length+2*left_margin+2*__left_margin__) or\
        not direction:
        left_ = left+left_margin
        top_ = top+top_margin if direction else top-label_height
        top_ = max(0,top_)
        right_ = left+label_length
        bottom_ = top_+label_height
        cv2.rectangle(frame, (left_, top_), (right_, bottom_), bg_colr, cv2.FILLED)
        cv2.putText(frame, label, (left_+__left_margin__, bottom_-__top_margin__), font, 1, fg_colr, 2)
    else:#right
        left_ = right+left_margin
        top_ = top+top_margin
        right_ = right+label_length
        bottom_ = top_+label_height
        cv2.rectangle(frame, (left_, top_), (right_, bottom_), bg_colr, cv2.FILLED)
        cv2.putText(frame, label, (left_+__left_margin__, bottom_-__top_margin__), font, 1, fg_colr, 2)    

from constants import label_lenght__

def drawStatus(frame,rect,label,bg_colr=bg_color,fg_colr=text_color,level=0):
    left,top,width,height = rect
    right,bottom = left+width,top+height
    cv2.rectangle(frame,(left,top),(right,bottom),bg_color,3)
    left_ = left+left_margin
    top_ = top-label_height*(1+level)
    right_ = left+label_lenght__
    bottom_ = top_+label_height
    cv2.rectangle(frame, (left_, top_), (right_, bottom_), bg_colr, cv2.FILLED)
    cv2.putText(frame, label, (left_+__left_margin__, bottom_-__top_margin__), font, 1, fg_colr, 2)    

def drawRectBox(image,rect,addText,bg_colr=(0,0, 255),fg_colr=(255, 255, 255)):

    cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), bg_colr, 2, cv2.LINE_AA)
    cv2.rectangle(image, (int(rect[0]-1), int(rect[1])-16), (int(rect[0] + 115), int(rect[1])), bg_colr, -1, cv2.LINE_AA)

    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)

    draw.text((int(rect[0]+1), int(rect[1]-16)), addText, fg_colr, font=fontC)
    imagex = np.array(img)
   
    return imagex

def saveImg(index,findex,image):
    cv2.imwrite("./out/%d/%d.png"%(index,findex),image)