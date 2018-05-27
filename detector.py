#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 10:14:48 2017

@author: ubuntu
"""
import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

import matplotlib.image as mpimg

import cv2
import math

import sys
sys.path.append('../')

from nets import ssd_vgg_300
from nets import ssd_vgg_512
from nets import np_methods#, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
#from notebooks import visualization

def showResult(name,
               img):
    cv2.imshow(name,img)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()

def analyzeBBox(bbox):
    pt1 = (int((bbox[0][0] + bbox[3][0]) / 2),int((bbox[0][1] + bbox[3][1]) / 2))
    pt2 = (int((bbox[1][0] + bbox[2][0]) / 2),int((bbox[1][1] + bbox[2][1]) / 2))
    pt3 = (int((bbox[0][0] + bbox[1][0]) / 2),int((bbox[0][1] + bbox[1][1]) / 2))
    pt4 = (int((bbox[2][0] + bbox[3][0]) / 2),int((bbox[2][1] + bbox[3][1]) / 2))
    center= (int((pt1[0]+pt2[0])/2),int((pt1[1]+pt2[1])/2))
    w = np.linalg.norm(np.array(pt1)-np.array(pt2))
    h = np.linalg.norm(np.array(pt3)-np.array(pt4))
    ###
    if w < h:
        tmp = w
        w = h
        h = tmp
        if pt3[0] < pt4[0]:
            pt_left = pt3
            pt_right = pt4
            [pt0_,pt3_] = [bbox[0],bbox[1]] if bbox[0][1] < bbox[1][1] else [bbox[1],bbox[0]]
            [pt1_,pt2_] = [bbox[2],bbox[3]] if bbox[2][1] < bbox[3][1] else [bbox[3],bbox[2]]
        else:
            pt_left = pt4
            pt_right = pt3
            [pt0_,pt3_] = [bbox[2],bbox[3]] if bbox[2][1] < bbox[3][1] else [bbox[3],bbox[2]]
            [pt1_,pt2_] = [bbox[0],bbox[1]] if bbox[0][1] < bbox[1][1] else [bbox[1],bbox[0]]
    else:
        if pt1[0] < pt2[0]:
            pt_left = pt1
            pt_right = pt2
            [pt0_,pt3_] = [bbox[0],bbox[3]] if bbox[0][1] < bbox[3][1] else [bbox[3],bbox[0]]
            [pt1_,pt2_] = [bbox[1],bbox[2]] if bbox[1][1] < bbox[2][1] else [bbox[2],bbox[1]]
        else:
            pt_left = pt2
            pt_right = pt1
            [pt0_,pt3_] = [bbox[1],bbox[2]] if bbox[1][1] < bbox[2][1] else [bbox[2],bbox[1]]
            [pt1_,pt2_] = [bbox[0],bbox[3]] if bbox[0][1] < bbox[3][1] else [bbox[3],bbox[0]]
            
    angle_radian = math.atan(float(pt_right[1] - pt_left[1])/w)
    angle_degree = angle_radian * (180.0 / math.pi)            
    
    return pt0_,pt1_,pt2_,pt3_,w,h,center,pt_left,pt_right,angle_degree
        
def drawBBox(img,
            bboxes_lp,
            bbox_car=None,
            bxcolor1 = (255,255,0),
            bxcolor2 = (0,255,255),
            linecolor = (0,255,0),
            drawmline=False,
            debug=False):
    
    out = img.copy()
    
    if bbox_car is not None:
        cv2.drawContours(out,[bbox_car],-1,bxcolor1,2)
        
    if bboxes_lp is None:
        return out
    
    for bbox in bboxes_lp:
        cv2.drawContours(out,[bbox],-1,bxcolor2,2)     
        # Direction Line ?x,y?y,x
        if drawmline:
            pt0_,pt1_,pt2_,pt3_,w,h,center,pt_left,pt_right,angle_degree = analyzeBBox(bbox)
            out = cv2.line(out,pt_left,pt_right,linecolor,2)
    if debug:
        showResult("drawBBox:result",out)
    return out

MODEL = 300

class VehicleDetector(object):
    def __init__(self, params=None):
        # TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
        self.gpu_options = tf.GPUOptions(allow_growth=True)
        self.config = tf.ConfigProto(log_device_placement=False, gpu_options=self.gpu_options)
        # Input placeholder.
        self.net_shape = (512,512) if MODEL == 512 else (300, 300)
        self.data_format = 'NHWC'
        self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
        # Evaluation pre-processing: resize to SSD net shape.
        self.image_pre, self.labels_pre, self.bboxes_pre, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
            self.img_input, None, None, self.net_shape, self.data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
        self.image_4d = tf.expand_dims(self.image_pre, 0)
        # Define the SSD model.
        try:
            tf.get_variable('ssd_512_vgg/conv1/conv1_1/weights') if MODEL == 512 else tf.get_variable('ssd_300_vgg/conv1/conv1_1/weights')
            self.reuse  = True# if tf.variable_scope('ssd_300_vgg/conv1/conv1_1/weights') else None
        except ValueError:
            print('model loading failed')
            self.reuse  = None
        self.ssd_net = ssd_vgg_512.SSDNet() if MODEL == 512 else ssd_vgg_300.SSDNet()
        with slim.arg_scope(self.ssd_net.arg_scope(data_format=self.data_format)):
            self.predictions, self.localisations, _, _ = self.ssd_net.net(self.image_4d, is_training=False, reuse=self.reuse)
        # Restore SSD model.
        self.ckpt_filename = './checkpoints/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt' if MODEL == 512 else './checkpoints/ssd_300_vgg.ckpt' 
        # SSD default anchor boxes.
        self.ssd_anchors = self.ssd_net.anchors(self.net_shape)

        self.isess = tf.InteractiveSession(config=self.config)     
        # Load Model     
        self.isess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.saver.restore(self.isess, self.ckpt_filename)
        
    def __enter__(self):
        return self
    
    # Main image processing routine.
    def process_image(self,
                      img,
                      select_threshold=0.5,
                      nms_threshold=.45,
                      net_shape=(300, 300)):
        # Run SSD network.
        rimg, rpredictions, rlocalisations, rbbox_img = self.isess.run([self.image_4d,self.predictions, self.localisations, self.bbox_img],
                                                                  feed_dict={self.img_input: img})
        
        # Get classes and bboxes from the net outputs.
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
                rpredictions, rlocalisations, self.ssd_anchors,
                select_threshold=select_threshold, img_shape=self.net_shape, num_classes=21, decode=True)
        
        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
        # Resize bboxes to original image shape. Note: useless for Resize.WARP!
        rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
        return rclasses, rscores, rbboxes
    

    
    def detect(self,
               img,
               debug=False):
        
        rclasses, rscores, rbboxes =  self.process_image(img)
        #visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
        # Refine BBoxes
        bbox = VehicleDetector.pick_one_vehicle(img,rclasses, rscores, rbboxes)
        if debug:
            drawBBox(img,[bbox],drawmline=False)
        return bbox
                
    
    def detect_by_filename(self,
                           path,
                           debug=False):        
        # Load File
        img = mpimg.imread(path)#img = cv2.imread(path)
        # Detect
        bbox = self.detect(img)#mpimg.imread(path)
        # Check Result
        if bbox is not None and debug:
            drawBBox(img,[bbox])
        ##
        return bbox
        
    def detect_by_data(self,
                       img,
                       debug=False):         
        # Detect
        bbox = self.detect(img)#mpimg.imread(path)
        # Check Result
        if bbox is not None and debug:
            drawBBox(img,[bbox])
        ##
        return bbox

    @staticmethod    
    def rbbox2bbox(img,
                   rbbox):
        height,width,channels = img.shape
        ymin = int(rbbox[0] * height)
        xmin = int(rbbox[1] * width)
        ymax = int(rbbox[2] * height)
        xmax = int(rbbox[3] * width)
        return np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])
    
    @staticmethod
    def pick_one_vehicle(img,rclasses, rscores, rbboxes):
        size = 0
        selected = None
        for i,rclass in enumerate(rclasses):
            if rclass == 7 and rscores[i]>0.5:
                bbox = VehicleDetector.rbbox2bbox(img,rbboxes[i])
                pt0_,pt1_,pt2_,pt3_,w,h,center,pt_left,pt_right,angle_degree = analyzeBBox(bbox)
                if w*h > size:
                    selected = bbox
                    size = w*h
        return selected
    
    @staticmethod   
    def pick_vehicles(img,rclasses, rscores, rbboxes):
        bboxes = []
        sizes = []
        for i,rclass in enumerate(rclasses):
            if rclass == 7 and rscores[i]>0.5:
                bbox = VehicleDetector.rbbox2bbox(img,rbboxes[i])
                bboxes.append(rbboxes[i])
                pt0_,pt1_,pt2_,pt3_,w,h,center,pt_left,pt_right,angle_degree = analyzeBBox(bbox)
                sizes.append(w*h)
        return bboxes
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.isess.close()
    
    def close(self):
        self.isess.close()
        
if __name__ == '__main__':
    pass
    