import os
import torch
from torch.amp import autocast
from collections import Counter
import time
import psutil
import gc
from torchvision import transforms
from torchvision.transforms import Normalize, ToTensor
from ..utils.helpers import load_model
from ..utils.settings import get_device
from ..models.pretrained import VisionTransformerLSTM
from PIL import Image, ImageFilter
import cv2
from tqdm import tqdm


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

    def predict_video(self, video_path, threshold=0.5, frame_skip=1):
        """
        Perform inference on a video file.

        Args:
            video_path (str): Path to the video file.
            threshold (float): Threshold for binary classification.
            frame_skip (int): Number of frames to skip between processed frames.

        Returns:
            str: Predicted label (e.g., "neg" or "pos").
        """
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Error opening video file {video_path}")

        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                # Convert frame to PIL Image
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                frames.append(image)

            frame_count += 1

        cap.release()

        if not frames:
            raise ValueError(f"No frames extracted from video {video_path}.")

        # Create sliding windows
        windows = []
        for start_idx in tqdm(range(0, len(frames), self.stride)):
            window = frames[start_idx:start_idx + self.window_size]
            if len(window) < self.window_size:
                # Pad the window with None
                window += [None] * (self.window_size - len(window))
            windows.append(window)

        # Batch processing and memory optimization
        predictions = []
        for window in windows:
            inputs, mask = self.preprocess_images_from_frames(window)  # [1, window_size, C, H, W], [1, window_size]
            inputs = inputs.to(self.device)
            mask = mask.to(self.device)

            with torch.no_grad():
                if "cuda" in self.device.type:
                    with autocast(self.device.type):
                        outputs = self.model(inputs, img_mask=mask, seq_mask=mask)  # [1, num_classes]
                else:
                    outputs = self.model(inputs, img_mask=mask, seq_mask=mask)

                probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of positive class
                predictions.append((probs > threshold).long().cpu().item())

        # Majority voting
        most_common = Counter(predictions).most_common(1)[0][0]
        return self.classes[most_common]

    def predict_webcam(self, threshold=0.5, frame_skip=1):
        """
        Perform inference on live webcam footage.

        Args:
            threshold (float): Threshold for binary classification.
            frame_skip (int): Number of frames to skip between processed frames.

        """
        # Initialize webcam capture
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise ValueError("Error opening webcam")

        frames_buffer = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            if frame_count % frame_skip == 0:
                # Convert frame to PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                frames_buffer.append(image)

                if len(frames_buffer) >= self.window_size:
                    # Get the latest window_size frames
                    window = frames_buffer[-self.window_size:]

                    inputs, mask = self.preprocess_images_from_frames(window)
                    inputs = inputs.to(self.device)
                    mask = mask.to(self.device)

                    with torch.no_grad():
                        if "cuda" in self.device.type:
                            with autocast(self.device.type):
                                outputs = self.model(inputs, img_mask=mask, seq_mask=mask)  # [1, num_classes]
                        else:
                            outputs = self.model(inputs, img_mask=mask, seq_mask=mask)

                        probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of positive class
                        prediction = (probs > threshold).long().cpu().item()
                        label = self.classes[prediction]

                        # Display the prediction on the frame
                        cv2.putText(frame, f"Prediction: {label}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Optionally, we can slide the window by stride
                    if self.stride < self.window_size:
                        frames_buffer = frames_buffer[self.stride:]
                    else:
                        frames_buffer = []

            # Display the frame
            cv2.imshow('Webcam', frame)

            key = cv2.waitKey(1)
            if key == 27:  # ESC key to exit
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()


def measure_memory_inference(model, input_data, device='cuda'):
    """
    Measures the memory usage of a PyTorch model during inference.

    Args:
        model (torch.nn.Module): The model to evaluate.
        input_data (torch.Tensor): Input data for the model.
        device (str): Device to use ('cuda' or 'cpu').

    Returns:
        dict: Memory usage statistics and inference time.
    """
    # Move model and data to the specified device
    model.to(device)
    input_data = input_data.to(device)

    # Synchronize and clear cache for accurate measurement
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Measure initial memory usage
    process = psutil.Process()
    initial_mem_cpu = process.memory_info().rss / 1024**2  # Resident memory in MB
    initial_mem_gpu = torch.cuda.memory_allocated(device) / 1024**2 if device == 'cuda' else 0.0

    # Start measuring inference time
    start_time = time.time()

    # Run inference
    with torch.no_grad():
        _ = model(input_data)

    # Synchronize and measure memory after inference
    if device == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()

    # Capture memory usage after inference
    final_mem_cpu = process.memory_info().rss / 1024**2
    final_mem_gpu = torch.cuda.memory_allocated(device) / 1024**2 if device == 'cuda' else 0.0

    # Calculate memory deltas
    delta_mem_cpu = final_mem_cpu - initial_mem_cpu
    delta_mem_gpu = final_mem_gpu - initial_mem_gpu

    # Inference time
    inference_time = end_time - start_time

    # Cleanup
    del input_data
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    # Return memory and time statistics
    return {
        "cpu_memory_before_mb": initial_mem_cpu,
        "cpu_memory_after_mb": final_mem_cpu,
        "cpu_memory_delta_mb": delta_mem_cpu,
        "gpu_memory_before_mb": initial_mem_gpu,
        "gpu_memory_after_mb": final_mem_gpu,
        "gpu_memory_delta_mb": delta_mem_gpu,
        "inference_time_s": inference_time
    }



if __name__ == "__main__":
    # Define the device
    device = get_device()

    # Load the trained model
    model = VisionTransformerLSTM(
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

    # Load the model weights
    fpath = os.path.join(os.getcwd(), 'best_model.pth')
    load_model(model, device, fpath)

    # Define the transformation
    transform = ResizePadSharpenTransform(
        target_size=(224, 224),  # Target size as (width, height)
        mean=[0.485, 0.456, 0.406],  # Mean for normalization
        std=[0.229, 0.224, 0.225]    # Std for normalization
    )

    # Initialize the inference pipeline
    pipeline = InferencePipeline(
        model=model,
        device=device,
        transform=transform,
        window_size=16,
        stride=8,
        classes=['non-drowsy', 'drowsy']
    )

    # # Perform prediction on a video file
    # video_prediction = pipeline.predict_video("test.mp4", threshold=0.5, frame_skip=1)
    # print(f"Predicted label for video: {video_prediction}")

    # Perform real-time prediction using webcam
    pipeline.predict_webcam(threshold=0.5, frame_skip=1)