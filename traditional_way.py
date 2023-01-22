import numpy as np
import cv2
from common import helper 
import os
current_path = os.getcwd()

# object detection with Haar Cascades
face_cascade_path = 'Haar_File\haarcascade_frontalface_default.xml'
eye_cascade_path = "Haar_File\haarcascade_eye.xml"

# Load Haar Cascades
face_cascade = helper.load_cascade(os.path.join(current_path, face_cascade_path))
eye_cascade = helper.load_cascade(os.path.join(current_path, eye_cascade_path))

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer= cv2.VideoWriter('basicvideo.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 20, (width,height))

while True:
    # Get image
    ret, img = cap.read()
    
    # convert the image into gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect faces which is a array(bottom left point x , y, faces width, faces height). Here we may detect multiple faces, so we use for loop to draw all of them
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
     # loop faces array to draw rectangle
    for (x, y, w, h) in faces: 
      # draw a rectangle around the face
      cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),2)
      # retrive th face sub region (gray and colored)
      # here dont be conused with the order of x and y
      # think the array slice, first is rows, and second is columns
      face_roi_gray = gray[y:y+h, x:x+w]
      face_roi_im = img[y:y+h, x:x+w]

      # detect eyes
      eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.3, 4)
      for(e_x,e_y,e_w,e_h)in eyes:
        #draw a rectangle around each eye
        cv2.rectangle(face_roi_im,(e_x,e_y),(e_x+e_w,e_y+e_h),(255,0,0),2)

    writer.write(img)
    cv2.imshow('img', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()