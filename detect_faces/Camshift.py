import numpy as np
import cv2
import time
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

while True:
    # Get image
    ret, img = cap.read()
    img_copy = img.copy()
    # Detect faces in subregion
    faces = helper.detect_faces(img, face_cascade)
    if len(faces) == 1:
        face = faces[0]
        # draw a green rectangle around the face
        cv2.rectangle(img_copy,(face[0],face[1]),(face[0] + face[2], face[1] + face[3]),(0,255,0),2)
    else:
        continue
    
    cv2.imshow('img_copy', img_copy)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    response = input("Use this bounding box? [y or n]:")
    if response == 'y':
        # frame contains the face region # grayscale picture
        frame = img[face[1]:face[1]+face[3], face[0]:face[0]+face[2]]
        tracking_window = face
        break
cap.release()
cv2.destroyAllWindows()

frame_hist = helper.transform_face2hist(frame)

# These mean: Stop the mean-shift algorithm iff we effectuated 10 iterations or the computed mean does not change by more than 1pt ~ 1.3px in both directions
# stop when the next center is very close
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
frames = []

while True:
    # Take a capture
    # Get image
    ret, img = cap.read()

    # Convert the capture to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Compute an inRange mask  as before with the frame
    mask = cv2.inRange(hsv, np.array((0., 64., 32.)), np.array((180., 255., 200.)))

    # Back project the frame histogram into the hsv image. Use only channel 0 (Hue), range of [0,180] and scale of 1
    prob = cv2.calcBackProject([hsv], [0], frame_hist, [0,180], scale = 1)
    # Bitwise and the back projection and the previously computed mask in order to remove very bright or very dark pixels (you can use `&` of python or cv2.bitwise_and in opencv)
    prob = prob & mask

    #bbox, tracking_window = cv2.CamShift('''back_projection here''', tracking_window '''This has been first computed in the beginning''', term_crit)
    bbox, tracking_window = cv2.CamShift(prob, tracking_window , term_crit)

    # Array of polygonal curves.
    pts = cv2.boxPoints(bbox).astype(np.int32)
    # cv2.polylines() method is used to draw a polygon on any image.
    cv2.polylines(img, [pts], True, (255, 0 , 0), 2)
    frames.append(img)
    cv2.imshow('img CamShift', img)
    # exit
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

# save gif
helper.save_gif(os.path.join(parent_path, "images", "CamShift.gif"), frames)
