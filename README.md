# Surveillance System
This repos, a preprocessing module for surveillance-system, takes out keeps vehicle frames from video stream. It also contains openalpr-based license plate recognition module.

# Requirement
   caffe: [installation](http://caffe.berkeleyvision.org/install_apt.html), [source-code](https://github.com/weiliu89/caffe/tree/ssd)
   
   openalpr:[source-code](https://github.com/openalpr/openalpr)

# How to Use
    
    python car_detector.py\
          -v "/media/XXX/Data/Test/bgs/testvideos/uproad.m4v"\
          -o "/media/XXX/Data/Test/bgs/out"\
          --preview\
          -i jpeg\
          --no-date-dir\
          --no-file-dir\
          --no-light\
          --no-noise\
          -s 0.3 -l -1\
          -m 0 
          -n 0\
          -g [[(0.48,0.05),(0.16,0.93),(0.92,0.91),(0.93,0.06)]]
    
# Thanks
  * I would like to thanks [weiliu](https://github.com/weiliu89/caffe/tree/ssd) for his wonderful caffe-based ssd and also thanks [balancap](https://github.com/balancap/SSD-Tensorflow) for his tensorflow ssd job. I couldn't use tensorflow ssd for my job because it is too slow, but I think it could be useful someday.
