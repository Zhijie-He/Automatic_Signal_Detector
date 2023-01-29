
import cv2
import imageio
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import keras.utils as image
from keras.applications.vgg19 import preprocess_input
from sklearn.model_selection import train_test_split

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


def save_plot(history, path, model_name="latest"):
  plt.figure(figsize=(10,5))
  plt.subplot(1,2,1)
  plt.plot(history.history['accuracy'], label='train accuracy')
  plt.plot(history.history['val_accuracy'], label = 'val accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend(loc='lower right')
  plt.subplot(1,2,2)
  plt.plot(history.history['loss'], label='train loss')
  plt.plot(history.history['val_loss'], label = 'val loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend(loc='lower right')
  plt.suptitle(model_name)
  plt.savefig(os.path.join(path, model_name + '_performance.png'))


def MLP_load_data(path):
  samples = []
  letters = []
  for folder in glob.glob(path+'/*'):

    label = folder.split('\\')[-1]
    # convert label from str to int
    label = ord(label)-ord('A')

    for images in glob.glob(folder+'/*'):
      img = cv2.imread(images)
      # convert img into grayscale
      img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      # here it's important to use flatten to reduce one dim
      img_gray = np.reshape(img_gray, (1024,-1)).flatten()
      samples.append(img_gray)
      letters.append(label)
  return np.array(samples), np.array(letters)

def CNN_load_data(path):
  samples = []
  letters = []
  for folder in glob.glob(path+'/*'):

    label = folder.split('\\')[-1]
    # convert label from str to int
    label = ord(label)-ord('A')

    for images in glob.glob(folder+'/*'):
      img = cv2.imread(images, cv2.IMREAD_GRAYSCALE)
      # convert img into grayscale
      #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      # here it's important to use flatten to reduce one dim
      samples.append(img)
      letters.append(label)
      
  return np.array(samples), np.array(letters)

def get_images_number(path_to_images):
  number = []
  paths_list = sorted(glob.glob(path_to_images + "/*"))
  for label in paths_list:
    number.append(len(glob.glob(label + "/*")))
  return number
  
def TL_load_data(path_to_images):
  # get the numbers of each label
  images_number = get_images_number(path_to_images)
  # create samples array
  samples = np.empty((sum(images_number), 32, 32, 3))
  letters = []
  index = 0
  for folder in sorted(glob.glob(path_to_images+'/*')):
    label = folder.split('\\')[-1]
    # convert label from str to int
    label = ord(label)-ord('A')
    
    for image_path in sorted(glob.glob(folder+"/*")):
      # load the image
      img = image.load_img(image_path)
      # convert the image into array
      img = image.img_to_array(img)
      # convert the images from RGB to BGR, then will zero-center each color channel with respect to the ImageNet dataset, without scaling.
      img = preprocess_input(img)
      # save the solved img 
      samples[index] = img
      index += 1
      letters.append(label)
  return samples, np.array(letters)

def split_dataset(samples, letters):
  # split dataset
  x_train, x_test, y_train, y_test = train_test_split(samples, letters, test_size=0.1, random_state=42)

  x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

  return x_train, y_train, x_test, y_test, x_val, y_val