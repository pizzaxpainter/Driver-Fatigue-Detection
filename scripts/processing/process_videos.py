import os
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from tqdm import tqdm

# Define a semaphore to limit the number of concurrent threads
MAX_CONCURRENT_TASKS = 16  # Adjust based on your system's capacity
semaphore = Semaphore(MAX_CONCURRENT_TASKS)

def extract_frames_from_video(video_path, output_dir, progress_bar=None):
    """
    Extract frames from a video and save them as images in a dedicated directory.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Path to the directory where frames will be saved.
        progress_bar (tqdm): Progress bar to update during processing.
    """
    # Acquire semaphore before processing
    with semaphore:
        try:
            # Get the video name without the extension
            video_name = os.path.splitext(os.path.basename(video_path))[0]

            # Create a dedicated directory for this video
            video_output_dir = os.path.join(output_dir, video_name)
            os.makedirs(video_output_dir, exist_ok=True)

            # Open the video file
            video_capture = cv2.VideoCapture(video_path)
            if not video_capture.isOpened():
                print(f"Failed to open video: {video_path}")
                return

            frame_number = 0
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break

                # Save each frame as an image
                frame_filename = f"{video_name}_frame_{frame_number:06d}.jpg"
                frame_path = os.path.join(video_output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                frame_number += 1

            video_capture.release()
            print(f"Frames saved for video: {video_path}")

        except Exception as e:
            print(f"Error processing video {video_path}: {e}")

        finally:
            if progress_bar:
                progress_bar.update(1)

def process_videos_recursively(input_dir, output_dir):
    """
    Recursively process all AVI and MP4 videos in a directory and extract frames,
    recreating the directory structure in the output directory.

    Args:
        input_dir (str): Root directory containing video files.
        output_dir (str): Root directory where frames will be saved.
    """
    video_paths = []

    # Traverse directories to collect video paths
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.avi', '.mp4')):
                # Get the relative path of the current directory to the input directory
                relative_path = os.path.relpath(root, input_dir)

                # Create the corresponding directory in the output directory
                video_output_dir = os.path.join(output_dir, relative_path)
                os.makedirs(video_output_dir, exist_ok=True)

                # Add the video path and output directory to the list
                video_path = os.path.join(root, file)
                video_paths.append((video_path, video_output_dir))

    # Initialize the progress bar
    with tqdm(total=len(video_paths), desc="Processing Videos") as progress_bar:
        try:
            # Process videos in parallel with a semaphore and progress bar
            with ThreadPoolExecutor() as executor:
                future_to_video = {
                    executor.submit(extract_frames_from_video, video_path, video_output_dir, progress_bar): (video_path)
                    for video_path, video_output_dir in video_paths
                }
                for future in as_completed(future_to_video):
                    video_path = future_to_video[future]
                    try:
                        future.result()  # Get the result to catch exceptions
                    except Exception as e:
                        print(f"Exception processing {video_path}: {e}")

        except KeyboardInterrupt:
            print("\nProcessing interrupted by user. Shutting down gracefully...")
            for future in future_to_video.keys():
                future.cancel()
            raise

# Set the input directory (where videos are stored) and output directory (where frames will be saved)
input_directory = "../DMD"
output_directory = "./outputs"

# Process all AVI and MP4 videos in the input directory recursively
try:
    process_videos_recursively(input_directory, output_directory)
except KeyboardInterrupt:
    print("\nProgram terminated by user.")
except Exception as e:
    print(f"\nUnexpected error: {e}")
