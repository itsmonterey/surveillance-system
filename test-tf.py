# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 21:41:41 2018

@author: stephen
"""

from detector import VehicleDetector
import os
import cv2
import time

def opencv2skimage(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def main():
    import gc
    gc.collect()
    detector = VehicleDetector()
    framecount = 0
    print(os.path.abspath('uproad.m4v'))
    cap = cv2.VideoCapture(os.path.abspath('uproad.m4v'))
    while(1):
       ret, frame = cap.read()
       #cv2.imshow('frame',frame)
       if ret is False:
           break
       framecount+=1
       #if framecount % 5 != 0:
       #    continue
       s = time.time()
       frame = cv2.resize(frame, (0,0), fx=0.3, fy=0.3)
       rclasses, rscores, rbboxes =  detector.process_image(frame)
       height = frame.shape[0]
       width = frame.shape[1]
       for i in range(rclasses.shape[0]):
           ymin = int(rbboxes[i, 0] * height)
           xmin = int(rbboxes[i, 1] * width)
           ymax = int(rbboxes[i, 2] * height)
           xmax = int(rbboxes[i, 3] * width)
           cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),(0,255,0),3)
       cv2.imshow('video',frame)
       e = time.time()
       print(e-s)
       k = cv2.waitKey(1) & 0xFF
       if k == 27:
           cv2.destroyAllWindows()
           break
    print('releasing net...')
    detector.close()
    
if __name__ == "__main__":
    main()