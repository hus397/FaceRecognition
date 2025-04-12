import cv2
import numpy as np
import os
import sys

reffile = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'

print('Your webcam is on')
images, labels, names, id = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
    for subdirs in dirs:
        names[id] = subdirs
        path = os.path.join(datasets, subdirs)
        for file in os.listdir(path):
            path1 = path + '/' + file
            label = id
            images.append(cv2.imread(path1, 0))
            labels.append(int(label))
        id = id + 1

(w, h) = (130, 100)            
(images, labels) = [np.array(lis) for lis in [images, labels]]
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)
#collected images from folders and used to train the model

face_cascade= cv2.CascadeClassifier(reffile)
webcam = cv2.VideoCapture(0)
while True:
    (_, im) = webcam.read()
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(imgray, 1.3, 4)
    for (x, y, w, h) in faces:
        rect = cv2.rectangle(imgray, (x, y), (x+w, y+h), (100, 0, 0), 3)
        face = imgray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (w, h))
        prediction = model.predict(face_resized)
        if prediction[1] < 500:
            cv2.putText(im, '%s - %.0f'%(names[prediction[0]], prediction[1]), (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 1, (000, 256, 000))     
        else:
            cv2.putText(im, 'Image not recognised', (x, y-25), cv2.FONT_ITALIC, 1, (256, 000, 000))
    cv2.imshow('Image', im)
    k = cv2.waitKey(10)
    if k == 27:
        break

        
        
