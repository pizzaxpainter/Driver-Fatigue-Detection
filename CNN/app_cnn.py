import streamlit as st
import cv2
import numpy as np
from collections import deque
import time
from PIL import Image
import torch
from torchvision.transforms import Normalize, ToTensor, Compose
from model import CNN3D  
#from preprocess import transform
from torchvision.transforms import CenterCrop
from torchvision.transforms import Resize

# Initialize Streamlit
st.title("Driver Drowsiness Detection (CNN3D)")

# Circular buffer for frames
FRAME_BUFFER_SIZE = 16
if "frame_buffer" not in st.session_state:
    st.session_state.frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)

# Load the CNN3D model
@st.cache_resource  # Cache the model to avoid reloading it on every interaction
def load_model():
    model = CNN3D(num_classes=1)
    model.load_state_dict(torch.load("./model_weight/cnn2.pth", weights_only=True))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device

model, device = load_model()

transform = Compose([
    Resize((128, 128)),  
    CenterCrop((112, 112)),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Function to preprocess frames
def preprocess_frames(frames):
    processed_frames = []
    if len(frames) > 10:
        # Select 10 frames evenly spaced from the original sequence
        step = len(frames) // 10
        frames = frames[::step][:10]
    for frame in frames:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = transform(image)
        processed_frames.append(image)
    return torch.stack(processed_frames).permute(1, 0, 2, 3).unsqueeze(0)

# Perform inference
def predict_drowsiness(frames, model, device):
    if len(frames) < FRAME_BUFFER_SIZE:
        return "Waiting for more frames..."

    # Preprocess frames
    inputs = preprocess_frames(frames)
    #print(f"Input shape: {inputs.shape}")  # Debugging: Print input shape
    inputs = inputs.to(device)

    # Model prediction
    with torch.no_grad():
        outputs = model(inputs)
        #print(f"Output shape: {outputs.shape}")  # Debugging: Print output shape
        prediction = torch.sigmoid(outputs).item()
        return "Drowsy" if prediction > 0.5 else "Non-Drowsy"

# Streamlit interface for live recording
status_placeholder = st.empty()
video_placeholder = st.empty()

if "recording" not in st.session_state:
    st.session_state.recording = False

if st.button("Start/Stop"):
    st.session_state.recording = not st.session_state.recording

if st.session_state.recording:
    status_placeholder.markdown("### Status: **Recording...**")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Unable to access the webcam.")
    else:
        while st.session_state.recording:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame.")
                break
            
            # Add the frame to the buffer
            st.session_state.frame_buffer.append(frame)

            # Perform prediction
            prediction = predict_drowsiness(list(st.session_state.frame_buffer), model, device)
            
            # Display the prediction and frame
            video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            status_placeholder.markdown(f"### Prediction: **{prediction}**")

            # Refresh Streamlit UI
            time.sleep(0.03)  # Approx 30 FPS

        cap.release()
else:
    status_placeholder.markdown("### Status: **Stopped**")
    if st.session_state.frame_buffer:
        video_placeholder.image(cv2.cvtColor(st.session_state.frame_buffer[-1], cv2.COLOR_BGR2RGB), channels="RGB")