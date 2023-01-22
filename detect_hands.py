import numpy as np
import cv2
import time
import matplotlib.pyplot as plt


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

def show_hist(hist):
  bin_count = hist.shape[0]
  bin_w = 24
  img = np.zeros((256, bin_count*bin_w, 3), np.uint8)
  for i in range(bin_count):
      h = int(hist[i])
      cv2.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  plt.imshow(img)
  plt.show()

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
    img_copy = img.copy()
    # Detect faces in subregion
    faces = detect_faces(img, face_cascade)
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

# Transform face into histogram

# Transform the frame into HSV space
frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Create an inRange mask for pixels. Limit the saturation in [64., 255.] and brightness in [32., 200.]
# only contains white and black color
mask = cv2.inRange(frame_hsv, np.array((0., 64., 32.)), np.array((180., 255., 200.)))

# Compute the histogram of the frame (use only the HUE channel). See `https://bit.ly/3pdVUEd`
# Take into account only pixels which are not too bright and not too dark (use the previous mask)
# Use 16 bins and speicfy the range of the hue ([0, 180])
frame_hist = cv2.calcHist([frame_hsv], [0], mask, [16], [0,180])

# Normalize the histogram between 0 (lowest intensity) and 255 (highest intensity) (use MinMax normalization `cv.NORM_MINMAX`) using the method `https://bit.ly/3jMGhCj`
frame_hist = cv2.normalize(frame_hist, 0, 255, cv2.NORM_MINMAX)

# Reashape the histogram into a 1-D array (use `.reshape(-1)`)
frame_hist = frame_hist.reshape(-1)

# # Show the histogram
# show_hist(frame_hist)


# These mean: Stop the mean-shift algorithm iff we effectuated 10 iterations or the computed mean does not change by more than 1pt ~ 1.3px in both directions
# stop when the next center is very close
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

cap = cv2.VideoCapture(0)

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

    #_, tracking_window = cv2.meanShift('''back_projection here''', tracking_window '''This has been first computed in the beginning''', term_crit)
    _, tracking_window = cv2.meanShift(prob, tracking_window, term_crit)
    (x, y, w, h) = tracking_window
    
    # set the face region prob = 0
    prob[y-20:y+h+100, x-20:x+w+20] = 0
    
    # plot a bounding box with coordiantes `tracking_window` in the image
    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow('img meanShift', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()








# # This function scales a rotated rectangle by a factor of scale_x (width) and scale_y (height)
# def scale_contour(pts, scale_x, scale_y):
#     M = cv2.moments(pts)

#     if M['m00'] == 0:
#       return pts

#     cx = int(M['m10']/M['m00'])
#     cy = int(M['m01']/M['m00'])

#     cnt_norm = pts - [cx, cy]
#     cnt_scaled = cnt_norm * np.array([scale_x, scale_y])
#     cnt_scaled = cnt_scaled + [cx, cy]
#     cnt_scaled = cnt_scaled.astype(np.int32)

#     return cnt_scaled



# # These mean: Stop the mean-shift algorithm iff we effectuated 10 iterations or the computed mean does not change by more than 1pt ~ 1.3px in both directions
# # stop when the next center is very close
# term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

# cap = cv2.VideoCapture(0)
# ret, img = cap.read()
# im_width, im_height = img.shape[1],img.shape[0]
# # Because we dont know where our hands are, we search it in entire image
# tracking_window_hand = (0, 0, im_width, im_height) # Define the initial tracking window for the hand. It spans the entire caption

# while True:
#     # Take a capture
#     # Get image
#     ret, img = cap.read()

#     # Convert the capture to HSV
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     # Compute an inRange mask  as before with the frame
#     mask = cv2.inRange(hsv, np.array((0., 64., 32.)), np.array((180., 255., 200.)))
    
#     # Back project the frame histogram into the hsv image. Use only channel 0 (Hue), range of [0,180] and scale of 1
#     prob = cv2.calcBackProject([hsv], [0], frame_hist, [0,180], scale = 1)
#     # Bitwise and the back projection and the previously computed mask in order to remove very bright or very dark pixels (you can use `&` of python or cv2.bitwise_and in opencv)
#     prob = prob & mask

#     # First look up for the face using cam shift starting from `tracking_window`
#     bbox, tracking_window = cv2.CamShift(prob, tracking_window, term_crit)
#     (x,y,w,h) = tracking_window

#     # Retrieve the rotated bounding rectangle
#     pts = cv2.boxPoints(bbox).astype(np.int32)
#     # Scale the rotated bounding box 1.5x times  
#     scaled_pts = scale_contour(pts, 1.5, 1.5)# Use `scale_contour`

#     # Fill the rotated face bounding box with 0 in the prob map
#     # Use `cv2.fillPoly`
#     cv2.fillPoly(prob, [scaled_pts], 0)
#     # # Draw the boundix box around the face
#     # cv2.polylines(img, [pts], True, (255, 0, 0), 2)
#     # # Draw the scaled boundix box around the face
#     # cv2.polylines(img, [scaled_pts], True, (0, 255 , 0), 2)

#     prob[y-20:y+h+100, x-20:x+w+20] = 0
#     # Now look up for the hand using cam shift starting from `tracking_window_hand`
#     bbox, tracking_window_hand = cv2.CamShift(prob, tracking_window_hand, term_crit)
#     pts = cv2.boxPoints(bbox).astype(np.int32)

#     # Scale the contour around the hand
#     pts = scale_contour(pts, 1.8, 1.5)
#     cv2.polylines(img, [pts], True, (0, 0, 255), 2)

#     cv2.imshow('img camShift', img)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()