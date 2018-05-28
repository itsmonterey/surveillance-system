# -*- coding: utf-8 -*-
"""
Created on Sun May 27 08:24:10 2018

@author: stephen
"""

import os
import cv2
import time

#import imp
#import sys
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import numpy as np

#imp.reload(sys)
fontC = ImageFont.truetype("./Font/platech.ttf", 14, 0)

from detector import VehicleDetector

detector = VehicleDetector()

from constants import image_exts,video_exts
from constants import labels
from constants import left_margin,top_margin
from constants import label_height,label_length
from constants import label_gap
from constants import __top_margin__,__left_margin__
from constants import text_color,bg_color
from constants import screen_size,display_size

from hyperlpr_py3 import pipline as pp

draw_plate_in_image_enable = 0
plateTypeName = ["蓝", "黄", "绿", "白", "黑 "]
        
def recognize_plate(image):
    t0 = time.time()
    h,w,c=image.shape
    images = pp.detect.detectPlateRough(
        image, image.shape[0], top_bottom_padding_rate=0.1)
    res_set = []
    y_offset = 32
    for j, plate in enumerate(images):
        plate, rect, origin_plate = plate

        plate = cv2.resize(plate, (136, 36 * 2))
        #t1 = time.time()
        h_,w_,c_=plate.shape
        #
        if y_offset+h_ > h or w_ > w:
            continue
        
        plate_type = pp.td.SimplePredict(plate)
        plate_color = plateTypeName[plate_type]

        if (plate_type > 0) and (plate_type < 5):
            plate = cv2.bitwise_not(plate)

        if draw_plate_in_image_enable == 1:
            image[y_offset:y_offset + plate.shape[0], 0:plate.shape[1]] = plate
            y_offset = y_offset + plate.shape[0] + 4

        image_rgb = pp.fm.findContoursAndDrawBoundingBox(plate)

        if draw_plate_in_image_enable == 1:
            image[y_offset:y_offset + image_rgb.shape[0],
                  0:image_rgb.shape[1]] = image_rgb
            y_offset = y_offset + image_rgb.shape[0] + 4

        image_rgb = pp.fv.finemappingVertical(image_rgb)

        if draw_plate_in_image_enable == 1:
            image[y_offset:y_offset + image_rgb.shape[0],
                  0:image_rgb.shape[1]] = image_rgb
            y_offset = y_offset + image_rgb.shape[0] + 4

        pp.cache.verticalMappingToFolder(image_rgb)

        if draw_plate_in_image_enable == 1:
            image[y_offset:y_offset + image_rgb.shape[0],
                  0:image_rgb.shape[1]] = image_rgb
            y_offset = y_offset + image_rgb.shape[0] + 4

        e2e_plate, e2e_confidence = pp.e2e.recognizeOne(image_rgb)
        print("e2e:", e2e_plate, e2e_confidence)

        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        #print("校正", time.time() - t1, "s")
        #t2 = time.time()
        val = pp.segmentation.slidingWindowsEval(image_gray)
        # print val
        #print("分割和识别", time.time() - t2, "s")
        res=""
        confidence = 0
        if len(val) == 3:
            blocks, res, confidence = val
            if confidence / 7 > 0.7:
                #image = pp.drawRectBox(image, rect, res)
                if draw_plate_in_image_enable == 1:
                    for i, block in enumerate(blocks):
                        block_ = cv2.resize(block, (24, 24))
                        block_ = cv2.cvtColor(block_, cv2.COLOR_GRAY2BGR)
                        image[j * 24:(j * 24) + 24, i *
                              24:(i * 24) + 24] = block_
                        if image[j * 24:(j * 24) + 24,
                                 i * 24:(i * 24) + 24].shape == block_.shape:
                            pass

        res_set.append([res,
                        confidence / 7,
                        rect,
                        plate_color,
                        e2e_plate,
                        e2e_confidence,
                        len(blocks)])
        print("seg:",res,confidence/7)
    print(time.time() - t0, "s")

    print("---------------------------------")
    return image, res_set

cnt = 0
fps = 8
font = cv2.FONT_HERSHEY_DUPLEX
ispaused=False        

def drawMark(frame,rect,label,direction=True):#top-down if True,bottom-up if False
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
        cv2.rectangle(frame, (left_, top_), (right_, bottom_), bg_color, cv2.FILLED)
        cv2.putText(frame, label, (left_+__left_margin__, bottom_-__top_margin__), font, 1, text_color, 2)
    else:#right
        left_ = right+left_margin
        top_ = top+top_margin
        right_ = right+label_length
        bottom_ = top_+label_height
        cv2.rectangle(frame, (left_, top_), (right_, bottom_), bg_color, cv2.FILLED)
        cv2.putText(frame, label, (left_+__left_margin__, bottom_-__top_margin__), font, 1, text_color, 2)    

def drawRectBox(image,rect,addText):

    cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (0,0, 255), 2, cv2.LINE_AA)
    cv2.rectangle(image, (int(rect[0]-1), int(rect[1])-16), (int(rect[0] + 115), int(rect[1])), (0, 0, 255), -1, cv2.LINE_AA)

    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)

    draw.text((int(rect[0]+1), int(rect[1]-16)), addText, (255, 255, 255), font=fontC)
    imagex = np.array(img)
   
    return imagex

if __name__ == "__main__":
    path = './samples/9.avi'
    cap = cv2.VideoCapture(path)
    #fps = cap.get(cv2.CAP_PROP_FPS)
    if os.path.exists(path):
        print('processing...')
        while True:
            #if ispaused:
            #    continue
            # reads frames from a video
            ret, frame = cap.read()
            if ret is False:
                break
            h,w,c=frame.shape
            cnt += 1
            if cnt%4 != 1:
                continue
            # object detection
            rclasses, rscores, rbboxes =  detector.process_image(frame.copy())
            for i in range(rclasses.shape[0]):
                if labels[rclasses[i]] not in ['bus','car','motorbike','pedestrian']:
                    continue
                top = int(rbboxes[i, 0] * h)
                left = int(rbboxes[i, 1] * w)
                bottom = int(rbboxes[i, 2] * h)
                right = int(rbboxes[i, 3] * w)
                width = int((right-left)/4)*4
                height= int((bottom-top)/4)*4
                #if (right-left) * 2 > w:
                #    continue
                # extract roi                
                roi = frame[top:top+height,left:left+width,:]
                # object detection and display result
                cv2.rectangle(frame,(left,top),(right,bottom),bg_color,3)
                drawMark(frame,(left,top,width,height),labels[rclasses[i]])
                # license recognition
                if labels[rclasses[i]] not in ['bus','car']:
                    continue
                roi, cands = recognize_plate(roi)
                # mark 
                for lp_str,conf,rect,lp_colr,lp_str_,conf,length in cands:
                    if conf > 0.7:
                        left_,top_,width_,height_ = rect
                        frame[top:top+height,left:left+width,:] = \
                                drawRectBox(roi,(int(left_),int(top_),int(width_),int(height_)),lp_str)
                    break
            
            frame = cv2.resize(frame,screen_size)
            cv2.imshow('all',frame)
            # Wait for Esc key to stop
            if cv2.waitKey(33) == 27:
                break