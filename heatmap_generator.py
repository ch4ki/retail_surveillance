import numpy as np
import cv2

class HeatmapGenerator:
    def __init__(self, detections_path, video_width, video_height):
        self.detections = np.load(detections_path)
        self.video_width = video_width
        self.video_height = video_height
        self.detection_map = np.zeros((video_height, video_width), dtype=np.uint32)
        self.heatmap = np.zeros((video_height, video_width), dtype=np.uint8)

    def create_heatmap(self, first_frame, last_frame):
        relevant_detections = self.detections[np.logical_and(self.detections[:, 0] >= first_frame, self.detections[:, 0] <= last_frame)]
        for detection in relevant_detections:
            self.detection_map[detection[2], detection[1]] += 1

        self.heatmap = (self.detection_map > 0).astype(np.uint8) * 255
        cv2.imshow("heatmap", self.heatmap)
        cv2.waitKey(0)
