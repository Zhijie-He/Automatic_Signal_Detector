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

previous_bbox = None
margin = 50

frames = []

while True:
    # Get image
    ret, frame = cap.read()
    region_to_use = frame
    # get image shape
    (img_height, img_width) = frame.shape[0], frame.shape[1]

    if previous_bbox is not None:
        (new_x,new_y,new_width,new_height) = helper.compute_optimized_search_region(previous_bbox, img_height, img_width, margin)
        # retrieve the sub region
        region_to_use = frame[new_y:new_y+new_height, new_x:new_x+new_width]
        # draw a red rectangle  BGR
        cv2.rectangle(frame, (new_x,new_y), (new_x+new_width, new_y+new_height),(255,0,0),5)

    # Detect faces in subregion
    faces = helper.detect_faces(region_to_use, face_cascade)
    if len(faces) == 1:
        #update the face bounding box
        face = faces[0]
        # (bottom left point x , y, faces width, faces height)
        (x,y,w,h) = face if previous_bbox is None else (new_x+face[0], new_y+face[1], face[2], face[3])
        # update previous_bbox
        previous_bbox = (x,y,w,h)
        #draw a green rectangle around the detected face
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)

        # here if we can find the face region, we also could find eyes in the face region
        face_roi_im = frame[y:y+h, x:x+w]
        face_roi_gray = cv2.cvtColor(face_roi_im,cv2.COLOR_BGR2GRAY)
        # detect eyes
        eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.3, 4)
        for(e_x,e_y,e_w,e_h)in eyes:
            #draw a rectangle around each eye
            cv2.rectangle(face_roi_im,(e_x,e_y),(e_x+e_w,e_y+e_h),(0,0,255),2)
    else:
        previous_bbox= None
        
    frames.append(frame)
    cv2.imshow('img', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

# save gif
helper.save_gif(os.path.join(parent_path, "images", "search_region.gif"), frames)
