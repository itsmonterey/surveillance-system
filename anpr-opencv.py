from openalpr import Alpr
from argparse import ArgumentParser
import sys
import numpy as np
import cv2
import cv
import time
#-c us --config "/etc/openalpr/openalpr.conf" --runtime_data "/usr/share/openalpr/runtime_data" "5799KE.jpg"
#https://github.com/openalpr/openalpr/issues/543
#https://github.com/openalpr/openalpr/issues/508
#which one to be selected : https://github.com/openalpr/openalpr/releases/tag/v2.2.0
#On ubuntu,compilation : http://doc.openalpr.com/compiling.html#linux
#Anaconda makes problem : https://github.com/openstreetmap/Nominatim/issues/577
#Change CMakeList.txt : https://github.com/openalpr/openalpr/issues/237
#Anaconda or Different Installation : https://github.com/openalpr/openalpr/issues/414
#https://github.com/openalpr/openalpr/blob/fa561a063a203a1702a13f042fbfcff24d0b651f/src/bindings/python/openalpr/openalpr.py
#(1.download https://github.com/openalpr/openalpr/tree/fa561a063a203a1702a13f042fbfcff24d0b651f()
# 2.disable anaconda(~/.bashrc)
# 3.You can edit the file : /openalpr/src/CMakeLists.txt Change
#
#        if ( NOT DEFINED WITH_TESTS )
#                SET(WITH_TESTS ON)
#        to
#
#        if ( NOT DEFINED WITH_TESTS )
#                SET(WITH_TESTS OFF)
#        ENDIF()
# 4.Follow Install Guide
#       # Install prerequisites
#       sudo apt-get install libopencv-dev libtesseract-dev git cmake build-essential libleptonica-dev
#       sudo apt-get install liblog4cplus-dev libcurl3-dev
#       # If using the daemon, install beanstalkd
#       sudo apt-get install beanstalkd
#       # Clone the latest code from GitHub
#       git clone https://github.com/openalpr/openalpr.git
#       # Setup the build directory
#       cd openalpr/src
#       mkdir build
#       cd build
#       # setup the compile environment
#       cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr -DCMAKE_INSTALL_SYSCONFDIR:PATH=/etc ..
#       # compile the library
#       make
#       # Install the binaries/libraries to your local system (prefix is /usr)
#       sudo make install
#       # Test the library
#       wget http://plates.openalpr.com/h786poj.jpg -O lp.jpg
#       alpr lp.jpg

alpr = Alpr("us", "/etc/openalpr/openalpr.conf", "/usr/share/openalpr/runtime_data")
if not alpr.is_loaded():
        print("Error loading OpenALPR")
        sys.exit(1)

alpr.set_top_n(20)
alpr.set_default_region("au")

'''
cap = cv2.VideoCapture("/media/frank/Data/Test/bgs/testvideos/uproad.m4v")#"rtsp://ip:port/snl/live/1/2?tcp")
ret, frame = cap.read()

while(True):

    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#cv2.imread("5799KE.jpg",0)   
    results = alpr.recognize_ndarray(gray)
    
    #print results['results']
    i = 0
    for plate in results['results']:
        i += 1
        #print "Plate #%d" % i
        #print "   %12s %12s" % ("Plate", "Confidence")
        for candidate in plate['candidates']:
            prefix = "-"
            if candidate['matches_template']:
                    prefix = "*"
            print "  %s %12s%12f"%(prefix, candidate['plate'], candidate['confidence'])
            break
    resized = cv2.resize(frame,(0,0),fx=0.3,fy=0.3)
    cv2.imshow("frame",resized)
    cv2.waitKey(1)

cap.release()
'''

gray = cv2.imread("/home/frank/Tmp/2.jpg",0)   
s = time.time()
results = alpr.recognize_ndarray(gray)
#print results['results']

i = 0
for plate in results['results']:
    i += 1
    #print "Plate #%d" % i
    #print "   %12s %12s" % ("Plate", "Confidence")
    for candidate in plate['candidates']:
        prefix = "-"
        if candidate['matches_template']:
                prefix = "*"
        print "  %s %12s%12f"%(prefix, candidate['plate'], candidate['confidence'])
        break
e = time.time()
print e-s
#resized = cv2.resize(gray,(0,0),fx=0.3,fy=0.3)
cv2.imshow("frame",gray)
cv2.waitKey(1000)                
alpr.unload()
