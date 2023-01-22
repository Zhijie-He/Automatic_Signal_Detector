
import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt

def load_cascade(cascade_file_path):
    cascade = cv2.CascadeClassifier(cascade_file_path)
    return cascade


def save_gif(gif_path, frames):
    print("Saving GIF file", gif_path)
    with imageio.get_writer(gif_path, mode="I") as writer:
        for idx, frame in enumerate(frames):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(rgb_frame)

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

def transform_face2hist(frame):
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
  return frame_hist

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

# This function scales a rotated rectangle by a factor of scale_x (width) and scale_y (height)
def scale_contour(pts, scale_x, scale_y):
    M = cv2.moments(pts)

    if M['m00'] == 0:
      return pts

    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = pts - [cx, cy]
    cnt_scaled = cnt_norm * np.array([scale_x, scale_y])
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled

# Will be needed in task 6 maybe
def crop_hand(pts, im_width, im_height):
  x_tl, y_tl = max(0, min(pts[:, 0])), max(0, min(pts[:, 1]))
  x_br, y_br = min(im_width, max(pts[:, 0])), min(im_height, max(pts[:, 1]))

  return(x_tl, y_tl),(x_br, y_br)