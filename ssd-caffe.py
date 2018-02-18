#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 14:10:16 2017

@author: frank
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
os.chdir('/home/frank/caffe-ssd/examples')

caffe_root = '../' # this file is expected to be in {caffe_root}/examples
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

from google.protobuf import text_format
from caffe.proto import caffe_pb2

sys.path.append('/home/frank/caffe-ssd/python')

labelmap_file = 'data/VOC0712/labelmap_voc.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

model_def = '/home/frank/caffe-ssd/models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt'
model_weights = '/home/frank/caffe-ssd/models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel'

net = caffe.Net(model_def, # defines the structure of the model
model_weights, # contains the trained weights
caffe.TEST) # use test mode (e.g., don't perform dropout)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel

transformer.set_channel_swap('data', (2,1,0)) # the reference model has channels in BGR order instead of RGB

image_resize = 300
net.blobs['data'].reshape(1,3,image_resize,image_resize)

colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

def detect_this_image(image,frame):
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[0,...] = transformed_image
    # net.blobs['data'].data[0,...] = image
    # Forward pass.
    detections = net.forward()['detection_out']
    # Parse the outputs.
    det_label = detections[0,0,:,1]
    det_conf = detections[0,0,:,2]
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]
    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]
    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    #define a numpy empty array of detections 
    dets = np.empty((0,5),int)
    #dets = [] 

    for i in xrange(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * image.shape[1]))
        ymin = int(round(top_ymin[i] * image.shape[0]))
        xmax = int(round(top_xmax[i] * image.shape[1]))
        ymax = int(round(top_ymax[i] * image.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
        label_name = top_labels[i]
        color = colors[label]
        
        dets = np.append(dets,np.array([[xmin,ymin,xmax,ymax,label]]),axis=0)
        #dets.append([xxmin,yymin,xxmax,yymax,score])
        #dets = np.asarray(dets)
    return dets

import cv2
import time

cap = cv2.VideoCapture('/media/frank/Data/Test/bgs/testvideos/uproad.m4v')
framecount = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret is False:
        continue
    framecount += 1
    if framecount % 5 != 0:
        continue
    frame = cv2.resize(frame,(0,0),fx=0.3,fy=0.3)
    image = cv2.resize(frame, (int(300), int(300)), interpolation=cv2.INTER_CUBIC)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('frame', res)
    # cv2.waitKey()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # break
    detss = detect_this_image(image,frame)

    for d in detss:
        #print xxmin,yymin,xxmax,yymax,track_id,category
        #print int(d[0]),int(d[1]),int(d[2]),int(d[3]),d[4],d[5]
        
        fheight,fwidth = frame.shape[:2]
        xxmin=int(round(d[0]*fwidth/300))
        yymin=int(round(d[1]*fheight/300))
        xxmax=int(round(d[2]*fwidth/300))
        yymax=int(round(d[3]*fheight/300))   
        #draw the trackers
        d = d.astype(np.uint32)
        color=colors[10]
        cv2.rectangle(frame, (xxmin,yymin), (xxmax,yymax),(int(color[0]*240),int(color[1]*240),int(color[2]*240)),2)
        #cv2.rectangle(frame, (int(round(d[0]*fwidth/300)),int(round(d[1]*fheight/300))), (int(round(d[2]*fwidth/300)), int(round(d[3]*fheight/300))),(int(color[0]*240),int(color[1]*240),int(color[2]*240)), 2)
        #cv2.imshow('frame', frame)
        cv2.putText(frame,str(d[4]),(xxmin,yymin),cv2.FONT_HERSHEY_SIMPLEX,2,255)
        
    cv2.imshow('frame', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
# image = cv2.resize(image, (int(900), int(900)), interpolation=cv2.INTER_CUBIC)
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('image', 600, 600)
cap.release()
cv2.destroyAllWindows()
