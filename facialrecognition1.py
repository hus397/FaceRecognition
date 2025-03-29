import cv2
import numpy as np
import os
import sys

reffile = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'
sub_data = 'Bottle'

path = os.path.join(datasets, sub_data)
if not os.path.isdir(path):
    os.mkdir(path)

width, height = (130, 100)
face_cascade = cv2.CascadeClassifier(reffile)
webcam = cv2.VideoCapture(0)
count = 1
while count <= 30:
    (_, im) = webcam.read()
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(imgray, 1.3, 4)
    for (x, y, w, h) in faces:
        rect = cv2.rectangle(imgray, (x, y), (x+w, y+h), (100, 0, 0), 3)
        face = imgray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (width, height))
        cv2.imwrite('%s/%s.png'%(path, count), face_resized)
    count = count + 1
    cv2.imshow('image', im)
    k = cv2.waitKey()
    if k == 27:
        break

        
        
    
    

