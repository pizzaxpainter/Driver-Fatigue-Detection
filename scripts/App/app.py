import streamlit as st
import cv2
import numpy as np
from collections import deque
import time
from PIL import Image
import os
import sys

# Add project path to import custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.pretrained import VisionTransformer
from utils.helpers import load_model
from utils.settings import get_device

# Function to convert frames to a list of PIL Images
def convert_to_pil(frames):
    return [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]

# Function to classify the drowsiness state
def classify_frames(model, frames):
    pil_frames = convert_to_pil(frames)
    input_tensor = model.preprocess_frames(pil_frames)  # Assuming your model has a preprocessing method
    prediction = model.predict(input_tensor)  # Assuming your model has a predict method
    return "Drowsy" if prediction == 1 else "Non-Drowsy"

# Initialize the Vision Transformer model
st.title("Webcam Recorder with Drowsiness Detection")

device = get_device()

    # Load the trained model
model = VisionTransformer(
    num_classes=2,
    model_name='vit_base_patch16_224',
    use_temporal_modeling=True,
    temporal_hidden_size=512,
    dropout_p=0.5,
    rnn_num_layers=2,
    bidirectional=False,
    freeze_vit=True,
)
model.to(device)
fpath = os.path.join(os.getcwd(), 'best_model.pth')
load_model(model, device, fpath)

# Button state
if "recording" not in st.session_state:
    st.session_state.recording = False

# Circular buffer to store the last 128 frames
FRAME_BUFFER_SIZE = 128
if "frame_buffer" not in st.session_state:
    st.session_state.frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)

# Status display
status_placeholder = st.empty()
if st.session_state.recording:
    status_placeholder.markdown("### Status: **Live Recording...**")
else:
    status_placeholder.markdown("### Status: **Playback**")

# Start/Stop Button
if st.button("Start/Stop"):
    st.session_state.recording = not st.session_state.recording

# Webcam stream handling
video_placeholder = st.empty()
if st.session_state.recording:
    cap = cv2.VideoCapture(0)  # Initialize webcam
    if not cap.isOpened():
        st.error("Unable to access the camera. Please check your webcam.")
    else:
        while st.session_state.recording:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame. Stopping recording.")
                break

            # Display live stream and store frame in the buffer
            video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            st.session_state.frame_buffer.append(frame)

            # Update the status
            status_placeholder.markdown("### Status: **Live Recording...**")

            # Allow Streamlit to refresh UI
            time.sleep(0.03)  # Approx. 30 FPS

        cap.release()

# Stop recording: Loop the buffered frames and classify the last 16
if not st.session_state.recording and st.session_state.frame_buffer:
    # Take the last 16 frames from the buffer
    WINDOW_SIZE = 16
    if len(st.session_state.frame_buffer) >= WINDOW_SIZE:
        last_frames = list(st.session_state.frame_buffer)[-WINDOW_SIZE:]
        prediction = classify_frames(model, last_frames)
        status_placeholder.markdown(f"### Status: **Last 16 Frames Detected: {prediction}**")
    else:
        status_placeholder.markdown("### Status: **Not enough frames for prediction.**")

    # Loop through buffered frames for playback
    while True:
        for frame in list(st.session_state.frame_buffer):
            video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            time.sleep(0.03)  # Display at 30 FPS
