import streamlit as st
import cv2
import numpy as np
from collections import deque
import time
from PIL import Image
from tqdm import tqdm
from PIL import Image, ImageFilter
import torch
from torchvision.transforms import Normalize, ToTensor
from torch.amp import autocast
from collections import Counter
import os
import sys

# Add project path to import custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.pretrained import VisionTransformerLSTMv1
from utils.helpers import load_model
from utils.settings import get_device

class ResizePadSharpenTransform:
    def __init__(self, target_size=(224, 224), mean=None, std=None):
        """
        Initialize the transform.

        Args:
            target_size (tuple): Desired output size as (width, height).
            mean (list): Mean for normalization.
            std (list): Standard deviation for normalization.
        """
        self.target_size = target_size
        self.mean = mean if mean is not None else [0.485, 0.456, 0.406]
        self.std = std if std is not None else [0.229, 0.224, 0.225]
        self.normalize = Normalize(mean=self.mean, std=self.std)
        self.to_tensor = ToTensor()

    def __call__(self, image):
        """
        Apply the transform to an image.

        Args:
            image (PIL.Image.Image): Input image.

        Returns:
            torch.Tensor: Transformed image tensor.
        """
        # Convert image to RGB if not already
        image = image.convert('RGB')

        # Determine the appropriate resampling filter
        try:
            # For Pillow >= 10.0.0
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            try:
                # For Pillow < 10.0.0
                resample = Image.LANCZOS
            except AttributeError:
                # Fallback to BICUBIC if LANCZOS is not available
                resample = Image.BICUBIC

        # Resize the image while maintaining aspect ratio
        image.thumbnail(self.target_size, resample)

        # Create a new image with a black background
        new_img = Image.new("RGB", self.target_size, (0, 0, 0))

        # Calculate padding to center the image
        paste_x = (self.target_size[0] - image.width) // 2
        paste_y = (self.target_size[1] - image.height) // 2
        new_img.paste(image, (paste_x, paste_y))

        # Apply the sharpening filter
        # You can use SHARPEN or UnsharpMask for more control
        sharpened_img = new_img.filter(ImageFilter.SHARPEN)
        # For more control:
        # sharpened_img = new_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

        # Convert to tensor and normalize
        image_tensor = self.to_tensor(sharpened_img)
        image_tensor = self.normalize(image_tensor)

        return image_tensor

class InferencePipeline:
    def __init__(self, model, device, transform, window_size=16, stride=8, classes=['neg', 'pos']):
        """
        Initialize the inference pipeline.

        Args:
            model (torch.nn.Module): Trained PyTorch model.
            device (torch.device): Device to run inference on.
            transform (callable): Image preprocessing transform.
            window_size (int): Number of frames in a sliding window.
            stride (int): Step size for the sliding window.
            classes (list): List of class names.
        """
        self.model = model
        self.device = device
        self.transform = transform
        self.window_size = window_size
        self.stride = stride
        self.classes = classes
        self.model.eval()

    def preprocess_images_from_frames(self, frames):
        """
        Preprocess a list of PIL Image frames into a tensor suitable for the model.

        Args:
            frames (list): List of PIL Image frames.

        Returns:
            torch.Tensor: Preprocessed image tensor with shape [1, window_size, C, H, W].
            torch.Tensor: Sequence mask tensor with shape [1, window_size].
        """
        images = []
        mask = []
        for frame in frames:
            if frame is not None:
                try:
                    image = frame.convert("RGB")
                    image = self.transform(image)
                    images.append(image)
                    mask.append(1)
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    continue
            else:
                # Create a dummy image for padding
                dummy_image = Image.new('RGB', (224, 224), color=(0, 0, 0))
                dummy_image = self.transform(dummy_image)
                images.append(dummy_image)
                mask.append(0)

        # Validate consistent dimensions
        if len(images) > 0:
            image_shape = images[0].shape
            assert all(img.shape == image_shape for img in images), "All images must have the same dimensions after transformation."

        # Stack and add batch dimension
        images = torch.stack(images).unsqueeze(0)  # Shape: [1, window_size, C, H, W]
        mask = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)  # Shape: [1, window_size]
        return images, mask
    
    def predict_from_buffer(self, frame_buffer, threshold=0.5):
        """
        Perform inference on a buffer of frames (numpy arrays).

        Args:
            frame_buffer (list): List of frames (numpy arrays).
            threshold (float): Threshold for binary classification.

        Returns:
            str: Predicted label (e.g., "neg" or "pos").
        """
        if not frame_buffer:
            raise ValueError("The frame buffer is empty.")

        # Convert frames in the buffer to PIL Images
        frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frame_buffer]

        # Create sliding windows
        windows = []
        for start_idx in range(0, len(frames), self.stride):
            window = frames[start_idx:start_idx + self.window_size]
            if len(window) < self.window_size:
                # Pad the window with None if not enough frames
                window += [None] * (self.window_size - len(window))
            windows.append(window)

        # Batch processing and memory optimization
        predictions = []
        for window in windows:
            # Preprocess frames and generate input tensors
            inputs, mask = self.preprocess_images_from_frames(window)  # [1, window_size, C, H, W], [1, window_size]
            inputs = inputs.to(self.device)
            mask = mask.to(self.device)

            with torch.no_grad():
                if "cuda" in self.device.type:
                    with autocast(self.device.type):
                        outputs = self.model(inputs, img_mask=mask, seq_mask=mask)  # [1, num_classes]
                else:
                    outputs = self.model(inputs, img_mask=mask, seq_mask=mask)

                # Compute probabilities and predictions
                probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of positive class
                predictions.append((probs > threshold).long().cpu().item())

        # Majority voting for final prediction
        most_common = Counter(predictions).most_common(1)[0][0]
        return self.classes[most_common]        

    def predict_webcam(self, threshold=0.5, frame_skip=1, video_placeholder=None, status_placeholder=None):
        """
        Perform inference on live webcam footage and display results in Streamlit.

        Args:
            threshold (float): Threshold for binary classification.
            frame_skip (int): Number of frames to skip between processed frames.
            video_placeholder: Streamlit placeholder for displaying video frames.
            status_placeholder: Streamlit placeholder for displaying the status.
        """
        # Initialize webcam capture
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            if status_placeholder:
                status_placeholder.error("Error accessing the webcam. Please check your camera.")
            return

        frames_buffer = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                if status_placeholder:
                    status_placeholder.error("Failed to grab frame. Stopping webcam stream.")
                break

            # Skip frames based on frame_skip
            if frame_count % frame_skip == 0:
                # Convert frame to PIL Image and add to buffer
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                frames_buffer.append(image)

                # Ensure the buffer has the latest `window_size` frames
                if len(frames_buffer) > self.window_size:
                    frames_buffer.pop(0)

                # Perform inference if we have enough frames in the buffer
                if len(frames_buffer) == self.window_size:
                    inputs, mask = self.preprocess_images_from_frames(frames_buffer)
                    inputs = inputs.to(self.device)
                    mask = mask.to(self.device)

                    with torch.no_grad():
                        if "cuda" in self.device.type:
                            with autocast(self.device.type):
                                outputs = self.model(inputs, img_mask=mask, seq_mask=mask)
                        else:
                            outputs = self.model(inputs, img_mask=mask, seq_mask=mask)

                        probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of "drowsy"
                        prediction = (probs > threshold).long().cpu().item()
                        label = self.classes[prediction]

                        # Update status_placeholder
                        if status_placeholder:
                            status_placeholder.markdown(f"### Status: **{label.capitalize()}**")

            # Display the live video stream in video_placeholder
            if video_placeholder:
                video_placeholder.image(frame_rgb, channels="RGB")

            # Stop loop if ESC key is pressed
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

# Function to display a frame in Streamlit
def display_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame, channels="RGB")

# Initialize Streamlit app
st.title("Drowsiness Detection Demo")

# Button state
if "recording" not in st.session_state:
    st.session_state.recording = False


# Status display
status_placeholder = st.empty()
if st.session_state.recording:
    status_placeholder.markdown("### Status: **Live Recording...**")
else:
    status_placeholder.markdown("### Status: **Playback**")

# Start/Stop Button
if st.button("Start/Stop"):
    st.session_state.recording = not st.session_state.recording
    
device = get_device()
model = VisionTransformerLSTMv1(
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

transform = ResizePadSharpenTransform(
    target_size=(224, 224),  # Target size as (width, height)
    mean=[0.485, 0.456, 0.406],  # Mean for normalization
    std=[0.229, 0.224, 0.225]    # Std for normalization
)

pipeline = InferencePipeline(
    model=model,
    device=device,
    transform=transform,
    window_size=16,
    stride=8,
    classes=['non-drowsy', 'drowsy']
)


# Webcam stream handling
video_placeholder = st.empty()
if st.session_state.recording:
    pipeline.predict_webcam(
        threshold=0.5,
        frame_skip=1,
        video_placeholder=video_placeholder,
        status_placeholder=status_placeholder
    )
"""
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
            
            if len(st.session_state.frame_buffer) >= 16:
                # Extract the last 16 frames from the buffer
                last_frames = list(st.session_state.frame_buffer)[-16:]

                pred = pipeline.predict_from_buffer(last_frames)

                # Display the prediction
                prediction_label = "Drowsy" if pred == 1 else "Non-Drowsy"
                status_placeholder.markdown(f"### Status: **{prediction_label}**")
            # Allow Streamlit to refresh UI
            time.sleep(0.01)  # Approx. 30 FPS

        cap.release()

# Stop recording: Loop the buffered frames
if not st.session_state.recording and st.session_state.frame_buffer:
    while True:
        for frame in list(st.session_state.frame_buffer):
            video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            status_placeholder.markdown("### Status: **Processing**")
            #images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))]
            time.sleep(0.03)  # Display at 30 FPS
"""