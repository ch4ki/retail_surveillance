import numpy as np
import cv2
from person_detector import PersonDetector

def main():
    # Set video input path
    video_path = 'test.mp4'

    # detection_map = np.zeros((int(streamer.video.get(4)), int(streamer.video.get(3))), dtype=np.uint64)

    person_detector = PersonDetector(video_path)
    person_detector.detect()
    person_detector.save_detections("detections.npy")

    return 0

if __name__ == "__main__":
    main()