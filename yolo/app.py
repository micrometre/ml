import cv2 as cv
from glob import glob
import os
from ultralytics import YOLO

# Read in video paths
videos = glob('*.mp4')
print("Available videos:", videos)

# Check if there are any videos to process
if not videos:
    print("No .mp4 files found in the current directory.")
    exit()

# Pick pre-trained model
model_pretrained = YOLO('yolov8n.pt')

# Read video by index (ensure the index is within the range of available videos)
video_index = 1  # Change this index if needed
if video_index >= len(videos):
    print(f"Video index {video_index} is out of range. Available videos: {len(videos)}")
    exit()

video = cv.VideoCapture(videos[video_index])

# Get video dimensions
frame_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
size = (frame_width, frame_height)

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'DIVX')
output_path = './outputs/uk_dash_2.avi'

# Ensure the output directory exists
os.makedirs('./outputs', exist_ok=True)

out = cv.VideoWriter(output_path, fourcc, 20.0, size)

# Read frames
while True:
    ret, frame = video.read()

    if not ret:
        break  # Exit the loop if no more frames are available

    # Detect & track objects
    results = model_pretrained.track(frame, persist=True)

    # Plot results
    composed = results[0].plot()

    # Save video
    out.write(composed)

# Release resources
out.release()
video.release()
print(f"Video processing complete. Output saved to {output_path}")