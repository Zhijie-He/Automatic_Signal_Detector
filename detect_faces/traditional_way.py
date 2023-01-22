import numpy as np
import cv2
import os
import sys

current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
# adding the parent directory to the sys.path.
sys.path.append(parent_path)
from common import helper 

# object detection with Haar Cascades
face_cascade_path = 'Haar_File\haarcascade_frontalface_default.xml'
eye_cascade_path = "Haar_File\haarcascade_eye.xml"

# Load Haar Cascades
face_cascade = helper.load_cascade(os.path.join(parent_path, face_cascade_path))
eye_cascade = helper.load_cascade(os.path.join(parent_path, eye_cascade_path))

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

frames = []

while True:
    # Get image
    ret, frame = cap.read()
    
    # convert the image into gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces which is a array(bottom left point x , y, faces width, faces height). Here we may detect multiple faces, so we use for loop to draw all of them
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
     # loop faces array to draw rectangle
    for (x, y, w, h) in faces: 
      # draw a red rectangle around the face
      cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)
      # retrive th face sub region (gray and colored)
      face_roi_gray = gray[y:y+h, x:x+w]
      face_roi_im = frame[y:y+h, x:x+w]

      # detect eyes
      eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.3, 4)
      for(e_x,e_y,e_w,e_h)in eyes:
        #draw a blue rectangle around each eye
        cv2.rectangle(face_roi_im,(e_x,e_y),(e_x+e_w,e_y+e_h),(255,0,0),2)
    frames.append(frame)
    cv2.imshow('img', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

# save gif
helper.save_gif(os.path.join(parent_path, "images", "tradition_way.gif"), frames)
