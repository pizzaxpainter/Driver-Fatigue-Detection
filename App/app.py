import streamlit as st
import cv2
import numpy as np
from collections import deque
import time
from threading import Lock, Thread

class StridedFrameProcessor:
    def __init__(self, buffer_size=128, batch_size=16, stride=16):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.stride = stride
        self.frame_buffer = deque(maxlen=buffer_size)
        self.lock = Lock()
        self.processing = False
        self.current_batch_start = 0
    
    def add_frame(self, frame):
        """
        Add a frame to the buffer. If buffer is full, oldest frames are automatically dropped.
        Returns True if frame was added, False if buffer is locked.
        """
        with self.lock:
            self.frame_buffer.append(frame)
            return True
    
    def get_next_batch(self):
        """
        Get the next batch of frames based on stride.
        Returns None if not enough frames or currently processing.
        """
        with self.lock:
            if self.processing:
                return None
                
            if len(self.frame_buffer) < self.batch_size:
                return None
            
            self.processing = True
            # Convert deque to list for easier slicing
            frames = list(self.frame_buffer)
            # Get the latest batch_size frames
            batch = frames[-self.batch_size:]
            # Remove processed frames based on stride
            for _ in range(self.stride):
                if self.frame_buffer:
                    self.frame_buffer.popleft()
            
            self.processing = False
            return batch

def main():
    st.title("Live Stream Frame Processor")
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    # Initialize processor with larger buffer
    processor = StridedFrameProcessor(
        buffer_size=128,  # Can hold many frames
        batch_size=16,    # Process 16 at a time
        stride=16         # Move forward 16 frames after processing
    )
    
    # Create placeholders for UI
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    buffer_status = st.empty()
    
    def process_batch(frames):
        """
        Placeholder for your model processing.
        Replace this with your actual model inference code.
        """
        time.sleep(0.5)  # Simulate processing time
        return ["Class X" for _ in frames]
    
    def processing_loop():
        while True:
            batch = processor.get_next_batch()
            if batch is not None:
                status_placeholder.text("Processing batch...")
                predictions = process_batch(batch)
                status_placeholder.text(f"Processed batch: {len(predictions)} predictions")
            time.sleep(0.01)  # Small delay to prevent CPU overuse
    
    # Start processing in separate thread
    processing_thread = Thread(target=processing_loop, daemon=True)
    processing_thread.start()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Error reading from camera")
                break
            
            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Add frame to processor
            processor.add_frame(frame_rgb.copy())
            
            # Display current frame
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Update buffer status
            buffer_status.text(f"Frames in buffer: {len(processor.frame_buffer)}")
            
            # Small delay to prevent overwhelming the UI
            time.sleep(0.01)
            
    finally:
        cap.release()

if __name__ == "__main__":
    main()