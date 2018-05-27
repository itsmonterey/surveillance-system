# -*- coding: utf-8 -*-
"""
Created on Sat May 26 05:26:12 2018

@author: stephen
"""

import face_recognition

known_face_encodings,known_face_names = [],[]

import os
import glob
import time

def file2encoding(filepath):
    image = face_recognition.load_image_file(filepath)
    encodings = face_recognition.face_encodings(image)
    return encodings[0] if encodings else None

for filepath in glob.glob('./database/face/*.png'):
    start = time.time()
    face_encoding = file2encoding(filepath)
    end = time.time()
    if face_encoding is None:
        continue
    print('elapsed time = %.1f'%(end-start))
    dir_,file = os.path.split(filepath)
    fname,ext = os.path.splitext(file)
    known_face_encodings.append(face_encoding)
    known_face_names.append(fname)
    
def queryface(face_encoding):
    name = "Unknown"
    if face_encoding is None:
        return name
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    # If a match was found in known_face_encodings, just use the first one.
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]
    return name

if __name__ == "__main__":
    print(known_face_encodings)
    print(known_face_names)
    for filepath in glob.glob('./database/face/*.png'):
        print('%s'%filepath,queryface(file2encoding(filepath)))


    