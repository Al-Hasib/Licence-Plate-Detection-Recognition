from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

# Load the YOLOv8 model
model = YOLO(r"C:\Users\abdullah\projects\Computer_Vision\Licence-Plate-Detection-Recognition\runs\detect\train\weights\best.pt")

# Open the video file
video_path = r"C:\Users\abdullah\projects\Computer_Vision\Licence-Plate-Detection-Recognition\video\3206742-uhd_2160_3840_30fps.mp4"
cap = cv2.VideoCapture(video_path)

# Get the video's width, height, and FPS
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs like 'XVID', 'MJPG', etc.
output_path = r"C:\Users\abdullah\projects\Computer_Vision\Licence-Plate-Detection-Recognition\video\output2.mp4"
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))


output_images = r"C:\Users\abdullah\projects\Computer_Vision\Licence-Plate-Detection-Recognition\video\License_images"
# Store the track history
track_history = defaultdict(lambda: [])
value = 0
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    
    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)
        print(f"Results : \n{results}\n\n")
        
        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        classes_id = results[0].boxes.cls.int().cpu().tolist()
        # print(f"\nClass : {class_}")
        # print(f"boxes :   \n\n{boxes} \n\n")
        # print(f"track ids : \n\n{track_ids}")
    
        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        
        # Plot the tracks
        for box, track_id,class_ in zip(boxes, track_ids,classes_id):
            if class_ != 2:
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)
            else:
                x_center, y_center, width, height = box
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
                # Ensure coordinates are within frame bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                
                
                # Extract the ROI
                roi = frame[y1:y2, x1:x2]

                # plt.imshow(roi)
                # plt.show()
                image_name = f"image_{value}.jpg"
                cv2.imwrite(os.path.join(output_images, image_name), roi)
                



            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        # Write the annotated frame to the output video
        out.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        value+=1

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture and writer objects, and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()
