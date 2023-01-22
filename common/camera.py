import cv2

class Camera():
    def __init__(self) -> None:
        self.cap = cv2.VideoCapture(0)
    def get_camera_width():
        return 