import os
import torch
from PIL import Image
from torch.amp import autocast
from collections import Counter
import time
import psutil
import gc


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

    def preprocess_images(self, image_paths):
        """
        Preprocess a list of image paths into a tensor suitable for the model.

        Args:
            image_paths (list): List of image file paths.

        Returns:
            torch.Tensor: Preprocessed image tensor with shape [1, window_size, C, H, W].
            torch.Tensor: Sequence mask tensor with shape [1, window_size].
        """
        images = []
        mask = []
        for path in image_paths:
            if path is not None:
                try:
                    image = Image.open(path).convert("RGB")
                    image = self.transform(image)
                    images.append(image)
                    mask.append(1)
                except Exception as e:
                    print(f"Error processing image {path}: {e}")
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

    def predict(self, image_dir, threshold=0.5):
        """
        Perform inference on a sequence of images in a directory.

        Args:
            image_dir (str): Path to the directory containing the images.
            threshold (float): Threshold for binary classification.

        Returns:
            str: Predicted label (e.g., "neg" or "pos").
        """
        # Load and sort images from the directory
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        image_paths = sorted(
            [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.lower().endswith(image_extensions)]
        )

        if not image_paths:
            raise ValueError(f"No valid images found in directory {image_dir}.")

        # Create sliding windows
        windows = []
        for start_idx in range(0, len(image_paths), self.stride):
            window = image_paths[start_idx:start_idx + self.window_size]
            if len(window) < self.window_size:
                # Pad the window with None
                window += [None] * (self.window_size - len(window))
            windows.append(window)

        # Batch processing and memory optimization
        predictions = []
        for window in windows:
            inputs, mask = self.preprocess_images(window)  # [1, window_size, C, H, W], [1, window_size]
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

    def predict_batch(self, image_dirs, threshold=0.5):
        """
        Perform inference on multiple directories of images.

        Args:
            image_dirs (list): List of paths to directories containing images.
            threshold (float): Threshold for binary classification.

        Returns:
            list: Predicted labels for each directory.
        """
        batch_predictions = []
        for image_dir in image_dirs:
            try:
                label = self.predict(image_dir, threshold)
                batch_predictions.append(label)
            except ValueError as e:
                print(f"Skipping directory {image_dir}: {e}")
                batch_predictions.append(None)
        return batch_predictions


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