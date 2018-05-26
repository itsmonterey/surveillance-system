# -*- coding: utf-8 -*-
"""
Created on Sat May 26 00:24:47 2018

@author: stephen
"""

image_exts = ['.jpeg','.jpg','.png','.bmp']
video_exts = ['.mpg','.mpeg','.mp4','.avi','m4v']
labels = ['background',
          'aeroplane',
          'bicycle',
          'bird',
          'boat',
          'bottle',
          'bus',
          'car',
          'cat',
          'chair',
          'cow',
          'diningtable',
          'dog',
          'horse',
          'motorbike',
          'pedestrian',
          'pottedplant',
          'sheep',
          'sofa',
          'train',
          'tvmonitor']

left_margin = 20
top_margin = 20

label_height = 40
label_gap = 3
label_length = 150
__top_margin__ = 4
__left_margin__ = 4

text_height = label_height - 2 * __top_margin__

