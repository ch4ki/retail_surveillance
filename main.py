import numpy as np
import torch
import cv2
from pathlib import Path
from stream_handler import VideoStreamHandler

def main():
    
 

    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model.conf = 0.6
    model.classes = [0]

    # Set video input path
    video_path = 'path_to_the_video.mp4'

    # Open the video file
    streamer = VideoStreamHandler(video_path)

    while True:
        # Read the next frame from the video
        frame = streamer.get_frame()

        # Perform person detection on the frame
        results = model(frame)

        # Draw bounding boxes and labels on the frame
        results.render()
        
        bboxes = results.xyxy[0].cpu().numpy()
        ground_positions = np.zeros((bboxes.shape[0], 2))
        ground_positions[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2
        ground_positions[:, 1] = bboxes[:, 3]
        

        # Convert the frame back to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # draw circles to the positions where people are standing
        for center in ground_positions:
            cv2.circle(frame, center.astype(int), 10, 0, -1)

        # Display the resulting frame
        cv2.imshow('YOLOv5 Person Detection', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) == ord('q'):
            break


    
    return 0

if __name__ == "__main__":
    main()