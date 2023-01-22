import cv2
import imageio
from common import helper
import os
current_path = os.getcwd()
frames = []
image_count = 0
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow("frame", frame)

    image_count += 1
    frames.append(frame)
    print("Adding new image:", image_count)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

helper.save_gif(os.path.join(current_path, "images", "test.gif"), frames)
