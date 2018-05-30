# -*- coding: utf-8 -*-
"""
Created on Sun May 27 08:24:10 2018

@author: stephen
"""

import os
import cv2
import time

from detector import VehicleDetector

detector = VehicleDetector()

from hyperlpr_py3 import pipline as pp

from constants import labels
from constants import screen_size,video_size#,display_size
#from constants import image_exts,video_exts

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
        blocks = []
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

from util import drawMark,drawRectBox
from constants import bg_color

(cv_major_ver, cv_minor_ver, cv_subminor_ver) = (cv2.__version__).split('.')
fps = 10
mode = 'recoding'#'displaying','mix'

from handy.misc import switch

if __name__ == "__main__":
    ### basic setting
    path = './samples/9.avi'
    cap = cv2.VideoCapture(path)
    fps_ = cap.get(cv2.CAP_PROP_FPS)
    pick_rate = int(fps_/fps) if fps_ >= 2*fps else 2
    ### video writer settting
    record_size=screen_size
    if mode is 'recoding':
        record_size=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if int(cv_major_ver) < 3 :
        fourcc = cv2.cv.CV_FOURCC('D','I','V','X')
        writer = cv2.VideoWriter('record.AVI',	fourcc, fps, record_size, 1)
    else:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter('record.AVI',	fourcc, fps, record_size, True)
    ###
    cnt = 0
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
            if cnt%pick_rate != 1:
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
                    if conf < 0.7:
                        continue
                    left_,top_,width_,height_ = rect
                    frame[top:top+height,left:left+width,:] = \
                            drawRectBox(roi,(int(left_),int(top_),int(width_),int(height_)),lp_str)
                    break
            for case in switch(mode):
                if case('displaying'):
                    frame = cv2.resize(frame,screen_size)
                    cv2.imshow('demo',frame)
                    break
                if case('recoding'):
                    writer.write(frame)
                    break
                if case('mix'):
                    frame = cv2.resize(frame,screen_size)
                    cv2.imshow('demo',frame)
                    writer.write(frame)
                    break
             # Wait for Esc key to stop
            if cv2.waitKey(5) == 27:
                cv2.destroyAllWindows()
                
    cv2.destroyAllWindows()
    writer.release()