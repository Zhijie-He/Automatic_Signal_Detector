
import cv2
import imageio

def load_cascade(cascade_file_path):
    cascade = cv2.CascadeClassifier(cascade_file_path)
    return cascade


def save_gif(gif_path, frames):
    print("Saving GIF file")
    with imageio.get_writer(gif_path, mode="I") as writer:
        for idx, frame in enumerate(frames):
            print("Adding frame to GIF file: ", idx + 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(rgb_frame)