# -*- coding: utf-8 -*-
"""
Created on Sat May 26 05:53:05 2018

@author: stephen
"""

from hyperlpr_py3 import pipline as pp

import cv2
import time

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
                image = pp.drawRectBox(image, rect, res)
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

import os     
from constants import screen_size
    
def video_test(path):
    cap = cv2.VideoCapture(path)
    #fps = cap.get(cv2.CAP_PROP_FPS)
    if not os.path.exists(path):
        return
    #cnt = 0
    ispaused=False
    while True:
        if ispaused:
            continue
        # reads frames from a video
        ret, frame = cap.read()
        if ret is False:
            break
        #cnt += 1
        #if cnt%2 != 1:
        #    continue
        frame, cands = recognize_plate(frame.copy())
        # license plate detection and display result
        '''
        for cand in cands:
            plate_string = cand[0]
            confidence = cand[1]
            x,y,w,h = cand[2]
            left=int(x)
            top=int(y)
            right=int(x+w)
            #bottom=int(y+h)
            #plate_colr = cand[3]
            font = cv2.FONT_HERSHEY_DUPLEX
            if confidence > 0.7:
                cv2.rectangle(frame, (left, top-20), (right, top), bg_color, cv2.FILLED)
                cv2.putText(frame, plate_string.encode('UTF-8'), (left, top-20), font, 1, text_color, 2)
        '''
        frame = cv2.resize(frame,screen_size)
        # Display frames in a window
        cv2.imshow('video2',frame)
        # Wait for Esc key to stop
        if cv2.waitKey(33) == 27:
            break

def image_test(path='./samples/lp.jpg'):
    path = os.path.abspath('lp.jpg')
    if os.path.exists(path):
        #image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
        image = cv2.imread(path)
        image, res_set = recognize_plate(image)
        print(res_set)
        
if __name__ == "__main__":
    video_test('./samples/lp.avi')

    