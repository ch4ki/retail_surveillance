
   
import torch
import cv2
from pathlib import Path

def main():
    
 

    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    # Set video input path
    video_path = Path('path_to_the_video.mp4')

    # Open the video file
    video = cv2.VideoCapture(str(video_path))

    while True:
        # Read the next frame from the video
        ret, frame = video.read()

        # Check if the frame was successfully read
        if not ret:
            break

        # Convert the frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform person detection on the frame
        results = model(frame)

        # Draw bounding boxes and labels on the frame
        results.render()
        
        bboxes = results.xyxy
        

        # Convert the frame back to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Display the resulting frame
        cv2.imshow('YOLOv5 Person Detection', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) == ord('q'):
            break


    
    return 0

if __name__ == "__main__":
    main()