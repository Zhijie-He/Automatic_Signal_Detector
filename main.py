import numpy as np
import cv2


#This function computes the sub region in which we want to detect faces
def compute_optimized_search_region(previous, img_height, img_width, margin):
  # here previous is the array of faces (x of top left, y of top left, face width, face height)
  # img_height and img_width is the whole img captured by the webcam, we use this to avoid the bounding box over the image
  # margin is the distance between bounding box and face region
  x_tl, y_tl = previous[0] - margin, previous[1] - margin
  x_br, y_br = previous[0] + previous[2] + margin, previous[1] + previous[3] + margin
  #set bounding box constraints
  x_tl, y_tl = max(x_tl, 0), max(y_tl, 0)
  x_br, y_br = min(x_br, img_width), min(y_br, img_height)
  # return the new bounding box coordinates
  return (x_tl, y_tl, x_br - x_tl, y_br - y_tl)

def detect_faces(img, cascades):
  #transform image into grayscale
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  #use cascades to detect faces
  faces = cascades.detectMultiScale(gray, 1.3, 5)
  return faces

# object detection with Haar Cascades
# Load Haar Cascades files
face_cascade = cv2.CascadeClassifier('Haar_File\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Haar_File\haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
previous_bbox = None
margin = 50

while True:
    # Get image
    ret, img = cap.read()
    region_to_use = img
    # get image shape
    (img_height, img_width) = img.shape[0], img.shape[1]

    if previous_bbox is not None:
        (new_x,new_y,new_width,new_height) = compute_optimized_search_region(previous_bbox, img_height, img_width, margin)
        # retrieve the sub region
        region_to_use = img[new_y:new_y+new_height, new_x:new_x+new_width]
        # draw a red rectangle  BGR
        cv2.rectangle(img, (new_x,new_y), (new_x+new_width, new_y+new_height),(255,0,0),5)

    # Detect faces in subregion
    faces = detect_faces(region_to_use, face_cascade)
    if len(faces) == 1:
        #update the face bounding box
        face = faces[0]
        # (bottom left point x , y, faces width, faces height)
        (x,y,w,h) = face if previous_bbox is None else (new_x+face[0], new_y+face[1], face[2], face[3])
        # update previous_bbox
        previous_bbox = (x,y,w,h)
        #draw a green rectangle around the detected face
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)

        # here if we can find the face region, we also could find eyes in the face region
        face_roi_im = img[y:y+h, x:x+w]
        face_roi_gray = cv2.cvtColor(face_roi_im,cv2.COLOR_BGR2GRAY)
        # detect eyes
        eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.3, 4)
        for(e_x,e_y,e_w,e_h)in eyes:
            #draw a rectangle around each eye
            cv2.rectangle(face_roi_im,(e_x,e_y),(e_x+e_w,e_y+e_h),(0,0,255),2)
    else:
        previous_bbox= None

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()