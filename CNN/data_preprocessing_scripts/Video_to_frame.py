import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import face_recognition
from tqdm.autonotebook import tqdm
import gc
from tqdm import tqdm
import glob
import json

# Create directory in the parent of CNN
os.makedirs("../../processed_videos_frames", exist_ok=True)

def save_cropped_face_frames_with_labels(video_path, json_path, output_dir, frame_step=1):

    """
    Process RGB face video, crop faces, and save them into 'drowsy' or 'non-drowsy' folders based on JSON annotations.

    Args:
        video_path (str): Path to the RGB face video file.
        json_path (str): Path to the JSON annotation file.
        output_dir (str): Directory where cropped frames will be saved.
        frame_step (int): Number of frames to skip before processing the next frame.
    """
    # Load JSON annotations
    with open(json_path, 'r') as f:
        annotations = json.load(f)

    # Extract actions and intervals from JSON
    actions = annotations.get('openlabel', {}).get('actions', {})
    drowsy_intervals = []
    non_drowsy_intervals = []

    for action_id, action_data in actions.items():
        action_type = action_data.get('type', 'Unknown')
        frame_intervals = action_data.get('frame_intervals', [])
        if action_type in ["eyes_state/closed", "yawning/Yawning with hand", "yawning/Yawning without hand"]:
            drowsy_intervals.extend(frame_intervals)
        elif action_type in ["eyes_state/open"]:
            non_drowsy_intervals.extend(frame_intervals)

    # Flatten intervals for faster lookup
    def flatten_intervals(intervals):
        return set(
            frame
            for interval in intervals
            for frame in range(interval["frame_start"], interval["frame_end"] + 1)
        )

    drowsy_frames = flatten_intervals(drowsy_intervals)
    non_drowsy_frames = flatten_intervals(non_drowsy_intervals)

    # Create output directories
    drowsy_dir = os.path.join(output_dir, "drowsy")
    non_drowsy_dir = os.path.join(output_dir, "non-drowsy")
    os.makedirs(drowsy_dir, exist_ok=True)
    os.makedirs(non_drowsy_dir, exist_ok=True)

    # Process video
    vidObj = cv2.VideoCapture(video_path)
    if not vidObj.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return

    frame_index = 0
    frames_saved = 0

    try:
        success, frame = vidObj.read()
        while success:
            # Only process frames on the specified step
            if frame_index % frame_step == 0:
                # Determine the label based on frame index
                if frame_index in drowsy_frames:
                    label_dir = drowsy_dir
                elif frame_index in non_drowsy_frames:
                    label_dir = non_drowsy_dir
                else:
                    # Skip frames not in annotations
                    success, frame = vidObj.read()
                    frame_index += 1
                    continue

                # Detect faces
                faces = face_recognition.batch_face_locations([frame], number_of_times_to_upsample=0, batch_size=1)
                if not faces:
                    print(f"No face detected in frame {frame_index} of video {os.path.basename(video_path)}")
                for face_locations in faces:
                    for top, right, bottom, left in face_locations:
                        try:
                            cropped_frame = cv2.resize(frame[top:bottom, left:right], (112, 112))
                            # Include video name in the filename to avoid conflicts
                            frame_file_path = os.path.join(
                                label_dir,
                                f"{os.path.basename(video_path).split('.')[0]}_frame_{frame_index:05d}.jpg"
                            )
                            cv2.imwrite(frame_file_path, cropped_frame)
                            frames_saved += 1
                        except Exception as crop_error:
                            print(f"Error cropping or saving frame {frame_index} in video {os.path.basename(video_path)}: {str(crop_error)}")
            success, frame = vidObj.read()
            frame_index += 1
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")

    vidObj.release()
    gc.collect()
    print(f"Video processed: {os.path.basename(video_path)}")
    print(f"Total frames saved: {frames_saved}")
    print("-" * 40)

# Paths and parameters
base_dir = "../../dataset/DMD"  # Path to the dataset folder within the "dataset" directory
output_base_dir = "../../dataset/processed_videos_frames"  # Base output directory for processed frames
frame_step = 3  # Process every 3 frames

# Define subfolders for train, val, and test
subfolders = ["train", "val", "test"]

# Process each subfolder
for subfolder in subfolders:
    video_dir = os.path.join(base_dir, subfolder)  # e.g., /DMD/train
    output_dir = os.path.join(output_base_dir, subfolder)  # e.g., /processed_videos_frames/train
    os.makedirs(output_dir, exist_ok=True)

    # Get list of all mp4 files in the current subfolder
    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))

    # Process each video file
    for video_path in video_files:
        # Construct corresponding JSON file path
        video_filename = os.path.basename(video_path)
        video_name = video_filename.split('.')[0]
        json_filename = f"{video_name}_ann_drowsiness.json"
        json_path = os.path.join(video_dir, json_filename)

        # Check if JSON file exists
        if not os.path.exists(json_path):
            print(f"No corresponding JSON file found for video {video_filename}. Skipping.")
            continue

        # Call the processing function
        save_cropped_face_frames_with_labels(video_path, json_path, output_dir, frame_step)
