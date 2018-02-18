#!/usr/bin/env python
from PIL import Image
# import pytesseract
# import urllib2
import signal
import argparse
import csv
import datetime
import threading
import timeit
import urllib
# import SimpleHTTPServer
# import SocketServer

import copy
import imutils
import time
import cv2
import logging

import os
import sys

import multiprocessing

import itertools
import numpy as np

#cv2.ocl.setUseOpenCL(False)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
fh = logging.FileHandler('log.txt')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s]:{} %(levelname)s %(message)s'.format(os.getpid()))
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)


class BufferedFrameWriter:
    MAX_BUF_SIZE = 1000

    def __init__(self):
        self.buf = []
        self.thread = threading.Thread(target=self.write_buf, args=())
        self.thread_stop = False
        self.thread.start()

    def write_buf(self):
        while not self.thread_stop:
            if len(self.buf) == 0:
                time.sleep(0.1)
                continue
            item = self.buf.pop()
            if item:
                path, image = item
                cv2.imwrite(path, image)

    def push(self, path, image):
        if len(self.buf) < BufferedFrameWriter.MAX_BUF_SIZE:
            self.buf.append((path, image))
        else:
            logger.error(
                'Output image buffer overflow, image storage speed is too low or host''s CPU usage is too high to encode images on time')

    def release(self):
        self.thread_stop = True


class MjpgStreamCap:
    MAX_BETWEEN_FRAMES_MS = 5000
    NEW_FRAME_CHECK_INTERVAL_MS = 100

    def __init__(self, url):
        self.url = url
        self.frame = None
        self.cur_frame_num = 0
        self.last_read_frame = 0
        self.thread = threading.Thread(target=self.read_stream, args=())
        self.thread_stop = False
        self.thread.start()

    def isOpened(self):
        return True

    def read(self):
        wait_counter = 0
        while ((self.frame is None) or (
                        self.cur_frame_num == self.last_read_frame and wait_counter < MjpgStreamCap.MAX_BETWEEN_FRAMES_MS)) and not self.thread_stop:
            time.sleep(MjpgStreamCap.NEW_FRAME_CHECK_INTERVAL_MS / 1000.0)
            wait_counter += MjpgStreamCap.NEW_FRAME_CHECK_INTERVAL_MS
            # logger.info('Frame not ready')
        if wait_counter >= MjpgStreamCap.MAX_BETWEEN_FRAMES_MS:
            logger.warn('Max interframe interval for stream reached, exiting')
        self.last_read_frame = self.cur_frame_num
        return (0, self.frame)

    def get(self, flag):
        return -1

    def read_stream(self):
        try:
            self.stream = urllib.urlopen(self.url)
            bytes = ''
            while not self.thread_stop:
                bytes += self.stream.read(500 * 1024)
                a = bytes.find('\xff\xd8')
                b = bytes.find('\xff\xd9')
                if a != -1 and b != -1:
                    jpg = bytes[a:b + 2]
                    bytes = bytes[b + 2:]
                    self.frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    self.cur_frame_num += 1
        except Exception as e:
            self.thread_stop = True
            logger.exception('Error opening stream')

    def release(self):
        self.thread_stop = True


def pos_to_literal(pos):
    """
    :param pos: Position in video in seconds or datetime
    :return:
    """
    if isinstance(pos, datetime.datetime):
        return pos.strftime('%H_%M_%S') + '_' + str(int(pos.microsecond / 1000)).ljust(2, '0')
    elif pos is None:
        return 'None'
    else:
        return str(datetime.timedelta(seconds=int(pos))).replace('.',
                                                                 '_').replace(':', '_')


def get_out_dir(params):
    if params.date_dir:
        out_dir = os.path.join(params.outdir, params.file_dir_name, datetime.datetime.now().strftime('%Y-%m-%d'))
        if out_dir not in params.existing_dirs:
            if not os.path.exists(out_dir):
                if params.file_dir_name and not os.path.exists(os.path.join(params.outdir, params.file_dir_name)):
                    os.mkdir(os.path.join(params.outdir, params.file_dir_name))
                os.mkdir(out_dir)
            params.existing_dirs.append(out_dir)
        return out_dir
    else:
        return params.outdir


def create_image_name(params, seconds, num, frame_number, suffix='', sight_num=0):
    now = datetime.datetime.now()
    seconds_since_mn = now.second + now.minute * 60 + now.hour * 3600
    return os.path.join(get_out_dir(params),
                        '{}{}_{}_{}_{}.{}'.format(suffix,
                                                  (
                                                      sight_num + num) if params.is_master else SightingHandler.CUR_SIGHTING,
                                                  os.path.splitext(os.path.basename(params.video))[0],
                                                  pos_to_literal(seconds), frame_number
                                                  , params.imgformat))


# class SightingHandler(SimpleHTTPServer.SimpleHTTPRequestHandler, object):
#     CUR_SIGHTING = 'X'
#
#     def do_GET(self):
#         SightingHandler.CUR_SIGHTING = str(self.path.split('/sighting/')[-1])
#         self.send_response(200)




def dump_presence(filename, presence):
    with open(filename, 'wb') as output_file:
        dict_writer = csv.DictWriter(output_file, ('start_time', 'end_time', 'start_frame', 'end_frame'))
        dict_writer.writeheader()
        dict_writer.writerows(presence)


ts_x = 195
ts_y = 1067
sym_w = 6
sym_h = 10
intersym = 2
cnt = 0

templates = dict()
for s_path in os.listdir('digits'):
    img = Image.open('digits/' + s_path)
    templates[s_path.split('.')[0]] = np.array(img.getdata()).reshape(img.size[1], img.size[0])


def ocr_timestamp(frame):
    try:
        global cnt
        # ts_area = frame[int(0.98618 * frame.shape[0]):frame.shape[0],
        #           int(0.09322 * frame.shape[1]):int(0.14739 * frame.shape[1]),
        #           :]
        # ts_area = np.pad(ts_area, [(ts_area.shape[0], ts_area.shape[0]), (0, 0), (0, 0)], mode='constant')
        # #ts_area = np.where(ts_area[:, :, 0] + ts_area[:, :, 1] + ts_area[:, :, 2] > 300, 0, 255).astype(np.uint8)
        # img = Image.fromarray(ts_area)
        # img.save('tst2.bmp')
        # 187, 1067
        text = ''
        for i in range(11):
            if i not in [2, 5, 8]:
                sym_strt = ts_x + i * sym_w + intersym * i
                sym_area = frame[ts_y:ts_y + sym_h, sym_strt:(sym_strt + sym_w)]
                sym_area = np.where((sym_area[:, :, 0] + sym_area[:, :, 1] + sym_area[:, :, 2]) > 100, 255, 0).astype(
                    np.uint8)
                # img = Image.fromarray(sym_area)
                # img.save('digits/' + str(i) + '_' + str(cnt) + '.bmp')
                # cnt += 1
                sym = None
                for c, t in templates.iteritems():
                    if np.mean(np.abs(sym_area - t)) < 50:
                        sym = c
                        break
                if sym:
                    text += sym
        time_a = [text[0:2], text[2:4], text[4:6]]
        time_ms_str = text[6:8]
        if len(time_ms_str) == 1:
            time_ms_str = time_ms_str + '0'
        time_ms = int(time_ms_str) * 10
        return datetime.datetime.combine(datetime.datetime.today(),
                                         datetime.time(hour=int(time_a[0]), minute=int(time_a[1]),
                                                       second=int(time_a[2]),
                                                       microsecond=time_ms * 1000))
    except Exception as e:
        # logger.exception('Couldn''t read timestamp')
        return None


def get_source_params(args):
    for s, r in zip(args.video.split(';'), itertools.cycle(args.roi)):
        if os.path.isdir(s):
            for f in sorted(os.listdir(s)):
                file = os.path.join(s, f)
                if os.path.isfile(file) and os.path.splitext(f)[1].lower() in ['.mp4', '.mpg', '.mpeg', '.mkv', '.mov',
                                                                               '.avi', '.264', '.h264', '.xvid', '.m4v',
                                                                               '.divx']:
                    yield os.path.join(s, f), r
        else:
            yield s, r

"""
Created on Wed Aug  2 14:10:16 2017

@author: jack

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

model_def = 'models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt'
model_weights = 'models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel'

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

class MaskBuffer:
    ZERO_BUFFER_SIZE = 10
    MIN_EXISTING_RATIO = 0.12
    def __init__(self,size):
        self.old = np.zeros(size,dtype=np.uint8)
        self.new = np.zeros(size,dtype=np.uint8)
        self.zeroframe_count = 0
    def add(self,mask):
        if np.count_nonzero(mask) == 0:
            self.zeroframe_count += 1
        else:
            self.zeroframe_count = 0
            
        if self.zeroframe_count == self.ZERO_BUFFER_SIZE:
            self.zeroframe_count = 0
        
        if np.count_nonzero(self.new) > np.prod(self.old.shape) * self.MIN_EXISTING_RATIO and self.zeroframe_count != 0:
            return
        
        self.old = self.new
        self.new = mask
        
    def get(self):
        return self.new
#def refine_mog():
    
import time

def scan_video(params):
    if not params.is_master:
        # start slave sighting listener
        Handler = SightingHandler
        httpd = SocketServer.TCPServer(("", params.slave_port), Handler)
        sght_thread = threading.Thread(target=lambda: httpd.serve_forever(), args=())
        sght_thread.start()

    cap = MjpgStreamCap(params.video) if params.video.startswith('http') else cv2.VideoCapture(params.video)

    if params.offset > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, params.offset)
        logger.info('Offset set to {0} ms'.format(params.offset))
    kernel = np.ones((3, 3), np.uint8)
    bgs = cv2.BackgroundSubtractorMOG2()#cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    frame_writer = BufferedFrameWriter()
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    cur_time = timeit.default_timer()
    scan_size = 320
    frame_number = 0
    frame_rel_time = 0
    presence_list = []
    presence_num = 1
    play = True
    frame_time = None
    cur_presence = None
    params.existing_dirs = []
    if params.file_dir:
        if params.is_stream:
            params.file_dir_name = params.video[
                                   (params.video.find('://') + 3):params.video.rfind('/')] + '_' + params.video[(
                params.video.rfind('/') + 1):(None if params.video.rfind('?') == -1 else params.video.rfind('?'))]
            params.file_dir_name = params.file_dir_name.replace('.', '_').replace('/', '_').replace(':', '_')
        else:
            params.file_dir_name = os.path.basename(params.video)
    logger.info('Video FPS is: {0}'.format(fps))
    logger.info('File length is: {0} frames'.format(length))
    time_start = datetime.datetime.now()
    if params.preview:
        cv2.namedWindow(params.video)
        
    """
    jack-todo
    """
    (ret, frame) = cap.read()
    sframe = imutils.resize(frame, width=scan_size)
    fgmask_buffer = MaskBuffer(sframe.shape)
    while cap.isOpened():
        (ret, frame) = cap.read()
        if frame is None:
            break

        # lower resolution to speed things up
        sframe = imutils.resize(frame, width=scan_size)#cv2.resize(frame,(0,0),fx=0.3,fy=0.3)#

        # create ROI mask and OCR timestamp
        if frame_number == 0:
            mask = np.zeros_like(sframe)
            roi = np.array([(int(x * mask.shape[1]), int(y * mask.shape[0])) for x, y in params.roi])
            scan_roi = np.array(
                [(int(x * scan_size), int((y * mask.shape[0] * scan_size) / float(mask.shape[1]))) for x, y in
                 params.roi])
            cv2.fillConvexPoly(mask, roi, color=(255, 255, 255))
            mask_area = np.count_nonzero(mask[:, :, 0]) / float(np.prod(sframe.shape[:2]))
        frame_number += 1

        if not frame_number % 300:
            scan_fps = 1 / ((timeit.default_timer() - cur_time) / 300)
            cur_time = timeit.default_timer()
            logger.info('Scan FPS is: {0} Progress: {1} / {2} frames'.format(scan_fps, frame_number, length))

        # re-align timestamp
        if not frame_number % 1 or frame_number == 1:
            ocr_frame_time = ocr_timestamp(frame)
            # update only if recognized
            if frame_number == 1 or ocr_frame_time:
                # logger.debug('OCR timestamp is: {}'.format(ocr_frame_time))
                frame_time = ocr_frame_time
        else:
            if frame_time:
                frame_time = frame_time + datetime.timedelta(milliseconds=1000.0 / fps)

        # subtract roi mask
        sframe = cv2.bitwise_and(sframe, mask)

        if params.light:
            # ignore L component to adjust for camera's auto-brightness
            sframe = cv2.cvtColor(sframe, cv2.COLOR_RGB2HLS)
            channels = cv2.split(sframe)
            channels[1] = np.zeros(channels[1].shape, np.uint8)
            sframe = cv2.merge(channels)

        fgmask = bgs.apply(sframe, learningRate=params.learningrate)
        # let bg subtractor learn
        if frame_number < 5:
            continue

        if params.noise:
            fgmask = cv2.erode(fgmask, kernel, iterations=1)
            fgmask = cv2.dilate(fgmask, None, iterations=4)

        th, fgmask = cv2.threshold(fgmask, 0, 255, cv2.THRESH_BINARY);
        
        """
        Created on Wed Aug  2 14:10:16 2017
        
        @author: jack
        
        """
        image = cv2.resize(frame, (int(300), int(300)), interpolation=cv2.INTER_CUBIC)
        detss = detect_this_image(image,frame)
        ssd_mask = np.zeros(fgmask.shape[:2], dtype=np.uint8)
             
        for d in detss:
            
            single_mask = np.zeros(fgmask.shape[:2], dtype=np.uint8)
            fheight,fwidth = fgmask.shape[:2]    
            xmin=int(round(d[0]*fwidth/300))
            ymin=int(round(d[1]*fheight/300))
            xmax=int(round(d[2]*fwidth/300))
            ymax=int(round(d[3]*fheight/300))  
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
            
            if d[4] == 4 or d[4] == 6 or d[4] == 7 or d[4] == 19:
                cv2.rectangle(frame, (xxmin,yymin), (xxmax,yymax),(int(color[0]*240),int(color[1]*240),int(color[2]*240)),2)
                #cv2.rectangle(frame, (int(round(d[0]*fwidth/300)),int(round(d[1]*fheight/300))), (int(round(d[2]*fwidth/300)), int(round(d[3]*fheight/300))),(int(color[0]*240),int(color[1]*240),int(color[2]*240)), 2)
                #cv2.imshow('frame', frame)
                #cv2.putText(frame,str(d[4]),(xxmin,yymin),cv2.FONT_HERSHEY_SIMPLEX,2,255)
                rc = np.array([[[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]]], dtype=np.int32)
                cv2.fillPoly(single_mask, rc, 255)
                ssd_mask = cv2.bitwise_or(ssd_mask,single_mask)
  
        #cv2.imshow("mog",fgmask)
        #cv2.waitKey(1)
         
        fgmask_buffer.add(ssd_mask)
        fgmask = fgmask_buffer.get()#cv2.bitwise_and(fgmask.copy(),ssd_mask)#,mask=ssd_mask)#
            
        #cv2.imshow('ssd', fgmask)
        #cv2.waitKey(1)
        
        #(_, cnts, _) = cv2.findContours(fgmask.copy(), cv2.RETR_TREE,
        #                                cv2.CHAIN_APPROX_SIMPLE)
        cnts,hierarchy = cv2.findContours(fgmask.copy(), cv2.cv.CV_RETR_TREE,
                                cv2.cv.CV_CHAIN_APPROX_SIMPLE)
        
        # loop over the contours
        max_cnt = None
        max_area = 0
        for c in cnts:
            # if the contour is too small, ignore it
            area = cv2.contourArea(c)
            if area > max_area:
                max_cnt = c
                max_area = area
        if params.debug and max_area > scan_size * 10:
            # create debug image
            d_image = cv2.bitwise_or(sframe, sframe, mask=fgmask)
            frame_writer.push(
                create_image_name(params, frame_time if not params.is_stream else datetime.datetime.now(),
                                  presence_num, frame_number % 5000, suffix='debug_', sight_num=params.sight_num),
                d_image)
        if max_area > params.size * np.prod(sframe.shape[:2]) * mask_area:
            if params.dumpimages:
                frame_writer.push(
                    create_image_name(params, frame_time if not params.is_stream else datetime.datetime.now(),
                                      presence_num, frame_number % 5000, sight_num=params.sight_num),
                    frame)
            if not cur_presence:
                cur_presence = {}
                cur_presence['start_frame'] = frame_number
                f_time = frame_rel_time if not params.is_stream else datetime.datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S')
                cur_presence['start_time'] = f_time
                if params.is_master and not params.alone:
                    urllib2.urlopen("http://127.0.0.1:{0}/sighting/{1}".format(params.slave_port, presence_num)).read()
                logger.debug('roi_frame {0} Time {1} presence start'.format(frame_number, f_time))
        else:
            if cur_presence:#if cur_presence:
                presence_num += 1#increase the vehicle index
                cur_presence['end_frame'] = frame_number
                f_time = frame_rel_time if not params.is_stream else datetime.datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S')
                cur_presence['end_time'] = f_time
                presence_list.append(cur_presence)
                if params.is_master and not params.alone:
                    urllib2.urlopen("http://127.0.0.1:{0}/sighting/X".format(params.slave_port)).read()
                logger.debug('roi_frame {0} Time {1} presence end'.format(frame_number, f_time))
            cur_presence = None
            #cur_presence = None
        if not frame_number % 100 and not params.dump_interval == 0 and (
                    datetime.datetime.now() - time_start).total_seconds() > params.dump_interval:
            dump_presence(
                os.path.join(get_out_dir(params), 'presence_' + pos_to_literal(time_start) + '__' + pos_to_literal(
                    datetime.datetime.now()) + '.csv'),
                presence_list)
            time_start = datetime.datetime.now()
            presence_list = []
        if params.preview:
            pframe = imutils.resize(frame, width=scan_size)
            if cur_presence:
                # compute the bounding box for the contour, draw it on the sframe,
                (x, y, w, h) = cv2.boundingRect(max_cnt)
                cv2.rectangle(pframe, (x, y), (x + w, y + h), (0, 255, 0), 1)
            for i in range(len(scan_roi) - 1):
                cv2.line(pframe, (scan_roi[i][0], scan_roi[i][1]), (scan_roi[i + 1][0], scan_roi[i + 1][1]),
                         color=(0, 0, 255))
            cv2.line(pframe, (scan_roi[-1][0], scan_roi[-1][1]), (scan_roi[0][0], scan_roi[0][1]), color=(0, 0, 255))
            cv2.imshow(params.video, pframe)
            key = cv2.waitKey(1)
    if cur_presence:
        cur_presence['end_frame'] = frame_number
        cur_presence['end_time'] = frame_rel_time
        presence_list.append(cur_presence)
    cap.release()
    cv2.destroyAllWindows()
    frame_writer.release()
    return presence_list


if __name__ == '__main__':
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("-v", "--video",
                        help="Path to the video files or stream URLs separated by ; If there are 2 streams, first is considered 'master' and provides sighting number to the 'slave'")
        ap.add_argument("-o", "--outdir", help="path to the dir with output files", default='out')
        ap.add_argument("-d", "--dumpimages", help="dump images with motion", default=True)
        ap.add_argument("-s", "--size", type=float, default=0.3, help="minimum relative size of the object (0-1)")
        ap.add_argument("-i", "--imgformat", default='jpeg',
                        help="output images format, one of: jpeg, png or bmp (uncompressed)")
        ap.add_argument("-g", "--roi", default='[[(0.0,0.0),(1.0,0.0),(1.0,1.0),(0.0,1.0)]]',
                        help="Region Of Interest - list of list of points pairs describing polygon in relative float coordinates in python format, clockwise. One for all sources, or an entry for each source. Example: '[[(0.0,0.0), (0.9,0.0), (0.9,0.9), (0.0,0.9)]]' ")
        ap.add_argument("-l", "--learningrate", type=float, default=-1,
                        help="background learning rate (0.0 - 1.0, omit for auto). Smaller values should improve performance when motion is slow.")
        ap.add_argument("-r", "--parallel", type=int, default=0,
                        help="Max number of worker processes (level of parallelization), default is number of cores minus 1")
        ap.add_argument("--debug", default=False, dest='debug', action='store_true',
                        help="Write output images with debug info")
        ap.add_argument("--preview", default=False, dest='preview', action='store_true',
                        help="Show output in a small window. Warning - slows down processing.")
        ap.add_argument("--no-noise", action='store_false', default=True, dest='noise',
                        help="Disable noise reduction (if there are false negatives)")
        ap.add_argument("--no-light", action='store_false', default=True, dest='light',
                        help="Disable HSL conversion and L component ignoring (if there are false negatives)")
        ap.add_argument("-p", "--presence-dump-interval", type=int, default=0, dest='dump_interval',
                        help="Interval of dumping presence records to CSV when in stream mode. 0 - don't save presence to CSV.")
        ap.add_argument("-a", "--no-date-dir", action='store_false', default=True, dest='date_dir',
                        help="Create date dir in destination dir")
        ap.add_argument("-f", "--no-file-dir", action='store_false', default=True, dest='file_dir',
                        help="Create file or stream dir in destination dir")
        ap.add_argument("-t", "--slave-port", default=7800, type=int, dest='slave_port',
                        help="Slaves starting port number")
        ap.add_argument("-m", "--master-offset", default=0, type=int, dest='master_offset',
                        help="Master offset for file sources. If greater than zero, master file is offset specified number of milliseconds, "
                             "if below zero - slave.")
        ap.add_argument("-n", "--sight-num", default=0, type=int, dest='sight_num',
                        help="Sighting number to start from")
        args = ap.parse_args()
        if vars(args).get('help'):
            exit(0)
        logger.info('Arguments: {0}'.format(vars(args)))
        if args.roi:
            args.roi = eval(args.roi)
            if len(args.roi) > 1 and len(args.roi) != len(args.video.split(';')):
                raise Exception('Length of ROI list doesn''t match number of sources')
        mp_args = []
        for i, (v, r) in enumerate(get_source_params(args)):
            if not (v.startswith('rtsp') or v.startswith('http') or os.path.exists(v)):
                raise Exception('Unknown stream type or video file doesn''t exist')
            args_c = copy.copy(args)
            args_c.offset = 0
            args_c.alone = True
            args_c.is_master = True
            # if i == 0:
            #     args_c.is_master = True
            # else:
            #     args_c.is_master = False
            #     args_c.slave_port = args.slave_port + (i - 1)
            args_c.video = v
            args_c.roi = r
            args_c.is_stream = v.startswith('http') or v.startswith('rtsp')
            mp_args.append(args_c)
        # if len(mp_args) == 2 and os.path.exists(mp_args[0].video) and os.path.exists(mp_args[1].video):
        #     if args.master_offset > 0:
        #         mp_args[0].offset = args.master_offset
        #     elif args.master_offset < 0:
        #         mp_args[1].offset = abs(args.master_offset)
        #     else:
        #         # set seek offsets based on file creation time
        #         master_time = os.path.getctime(mp_args[0].video)
        #         slave_time = os.path.getctime(mp_args[1].video)
        #         if slave_time > master_time:
        #             mp_args[0].offset = slave_time - master_time
        #         else:
        #             mp_args[1].offset = master_time - slave_time
        # if len(mp_args) == 1:
        #     mp_args[0].alone = True
        # else:
        #     mp_args[0].alone = False
        
        print mp_args
        print args.parallel
        
        try:
            if len(mp_args) > 1 and args.parallel > 0:
                logger.info('Starting parallel processing...')
                p = multiprocessing.Pool(((
                                              multiprocessing.cpu_count() - 1) if multiprocessing.cpu_count() > 1 else 1) if not args.parallel else args.parallel)
                presence = p.map(scan_video, mp_args)
            else:
                logger.info('Starting sequential processing...')
                cur_sno = args.sight_num
                for i, ar in enumerate(mp_args):
                    logger.info('Processing source {2}, {0} of {1}'.format(i, len(mp_args), ar.video))
                    ar.sight_num = cur_sno
                    try:
                        presence = scan_video(ar)
                        cur_sno += len(presence)
                    except:
                        logger.exception('Error processing source, skipping:')
            print  zip(presence, mp_args)
            for p, a in zip(presence, mp_args):
                if not a.is_stream and p:
                    video_name = a.video[a.video.rfind('/') + 1:] if args.is_stream else os.path.basename(a.video)
                    video_name = video_name or 'video'
                    video_name = os.path.join(args.outdir, video_name)
                    dump_presence('presence_' + video_name + '.csv', p)
                    logger.info('Continuous presence cases count: {0}'.format(len(p)))
                    logger.info('Output file name: {0}'.format(video_name + '.csv'))
        except KeyboardInterrupt:
            p.terminate()
            p.join()
        else:
            logger.info('No presence detected')
    except Exception as exc:
        if exc is not SystemExit:
            logger.exception('Error processing video')
    input("Press any key to exit")
