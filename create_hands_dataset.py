import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from common import helper 
import os
import argparse


def create_dataset(signal_name):
    current_path = os.path.dirname(os.path.realpath(__file__))
    # object detection with Haar Cascades
    face_cascade_path = 'Haar_File\haarcascade_frontalface_default.xml'

    # Load Haar Cascades
    face_cascade = helper.load_cascade(os.path.join(current_path, face_cascade_path))

    cap = cv2.VideoCapture(0)

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
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    path = os.path.join(current_path, 'Dataset', signal_name)
    if not os.path.exists(path):
        os.makedirs(path)

    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    im_width, im_height = img.shape[1],img.shape[0]
    # Because we dont know where our hands are, we search it in entire image
    tracking_window_hand = (0, 0, im_width, im_height) # Define the initial tracking window for the hand. It spans the entire caption
    cpt = -1

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

        # First look up for the face using cam shift starting from `tracking_window`
        (x,y,w,h) = tracking_window
        bbox, tracking_window = cv2.CamShift(prob, tracking_window, term_crit)

        # Retrieve the rotated bounding rectangle
        pts = cv2.boxPoints(bbox).astype(np.int32)
        # Scale the rotated bounding box 1.5x times  
        scaled_pts = helper.scale_contour(pts, 1, 1)# Use `scale_contour`

        # Fill the rotated face bounding box with 0 in the prob map
        # Use `cv2.fillPoly`
        cv2.fillPoly(prob, [scaled_pts], 0)
        # Draw the boundix box around the face
        #cv2.polylines(prob, [pts], True, (255, 255, 255), 2)
        # Draw the scaled boundix box around the face
        cv2.polylines(img, [scaled_pts], True, (255, 0 , 0), 2)

        prob[y-20:y+h+100, x-20:x+w+20] = 0
        # Now look up for the hand using cam shift starting from `tracking_window_hand`
        bbox, tracking_window_hand = cv2.CamShift(prob, tracking_window_hand, term_crit)
        pts = cv2.boxPoints(bbox).astype(np.int32)
        # Scale the contour around the hand
        pts = helper.scale_contour(pts, 1.8, 1.8)

        #Detect hand
        cropped_hand_bbox = helper.crop_hand(pts, im_width, im_height)
        cv2.rectangle(img, cropped_hand_bbox[0], cropped_hand_bbox[1], (0, 255, 0), 2)

        cv2.imshow('img camShift', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        if k == 121: # "y"
            if cropped_hand_bbox[1][1] !=0:
                try:
                    cropped_hand = cv2.cvtColor(img[cropped_hand_bbox[0][1]:cropped_hand_bbox[1][1], cropped_hand_bbox[0][0]:cropped_hand_bbox[1][0]], cv2.COLOR_BGR2GRAY)
                    cropped_hand = cv2.resize(cropped_hand, (32, 32))
                    cpt += 1
                    # print(cpt)
                    cv2.imwrite(os.path.join(path, str(cpt)+'.jpg'), cropped_hand)
                    print("record img", os.path.join(path, str(cpt)+'.jpg'))
                except:
                    pass

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('signal', type=str, help='select the signal')

    args = parser.parse_args()
    create_dataset(args.signal)