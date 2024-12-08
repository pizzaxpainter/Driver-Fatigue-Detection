import torch
import torchvision.transforms.functional as TF
import random

class DrowsinessAugmentationPipeline:
    """
    Comprehensive augmentation pipeline for driver drowsiness detection.
    Excludes brightness, contrast, and rotation adjustments to avoid redundancy with ColorJitter.
    """
    def __init__(self, 
                 noise_factor=0.02,
                 occlusion_prob=0.2,
                 motion_blur_prob=0.3,
                 noise_prob=0.2,
                 sharpness_range=(0.8, 1.2),
                 sharpness_prob=0.3,
                 colorfulness_range=(0.8, 1.2),
                 colorfulness_prob=0.3):

        self.noise_factor = noise_factor
        self.noise_prob = noise_prob
        self.occlusion_prob = occlusion_prob
        self.motion_blur_prob = motion_blur_prob
        self.sharpness_range = sharpness_range
        self.sharpness_prob = sharpness_prob
        self.colorfulness_range = colorfulness_range
        self.colorfulness_prob = colorfulness_prob

    def add_gaussian_noise(self, image):
        """Add random Gaussian noise to simulate camera sensor noise."""
        noise = torch.randn_like(image) * self.noise_factor
        noisy_image = image + noise
        return torch.clamp(noisy_image, 0, 1)

    def simulate_motion_blur(self, image, kernel_size=5, angle=None):
        """Simulate motion blur from head movement."""
        if angle is None:
            angle = random.uniform(0, 360)
        
        # Create a motion blur kernel (horizontal for simplicity)
        kernel = torch.zeros((3, 1, kernel_size, kernel_size))
        kernel[:, 0, kernel_size//2, :] = 1.0 / kernel_size
        
        # Apply convolution
        image = image.unsqueeze(0)  # Shape: [1, 3, H, W]
        blurred = torch.nn.functional.conv2d(image, kernel, padding=kernel_size//2, groups=3)
        return blurred.squeeze(0)  # Shape: [3, H, W]

    def random_occlusion(self, image, max_boxes=2):
        """Simulate partial face occlusions (e.g., hand movement, hair)."""
        result = image.clone()
        _, h, w = image.shape

        for _ in range(random.randint(1, max_boxes)):
            box_w = random.randint(w//8, w//4)
            box_h = random.randint(h//8, h//4)
            x = random.randint(0, w - box_w)
            y = random.randint(0, h - box_h)

            fill_value = random.uniform(0, 1)
            result[:, y:y+box_h, x:x+box_w] = fill_value

        return result

    def adjust_sharpness(self, image):
        """Adjust the sharpness of the image."""
        sharpness_factor = random.uniform(*self.sharpness_range)
        return TF.adjust_sharpness(image, sharpness_factor)

    def adjust_colorfulness(self, image):
        """Adjust the colorfulness (saturation) of the image."""
        saturation_factor = random.uniform(*self.colorfulness_range)
        return TF.adjust_saturation(image, saturation_factor)

    def __call__(self, image):
        """Apply augmentation pipeline."""
        # Apply motion blur
        if random.random() < self.motion_blur_prob:
            image = self.simulate_motion_blur(TF.to_tensor(image))
            image = TF.to_pil_image(image)

        # Apply Gaussian noise
        if random.random() < self.noise_prob:
            image = self.add_gaussian_noise(TF.to_tensor(image))
            image = TF.to_pil_image(image)

        # Apply partial occlusions
        if random.random() < self.occlusion_prob:
            image = self.random_occlusion(TF.to_tensor(image))
            image = TF.to_pil_image(image)

        # Apply sharpness adjustment
        if random.random() < self.sharpness_prob:
            image = self.adjust_sharpness(image)

        # Apply colorfulness adjustment
        if random.random() < self.colorfulness_prob:
            image = self.adjust_colorfulness(image)

        return image

class RandomZoom:
    def __init__(self, zoom_range=(0.8, 1.2), zoom_prob=0.3):
        """
        Initialize RandomZoom.

        Args:
            zoom_range (tuple): Range of zoom factors.
            zoom_prob (float): Probability to apply zoom.
        """
        self.zoom_range = zoom_range
        self.zoom_prob = zoom_prob

    def __call__(self, img):
        """
        Apply Random Zoom to the image.

        Args:
            img (PIL.Image): Input image.

        Returns:
            PIL.Image: Zoomed image.
        """
        if torch.rand(1) < self.zoom_prob:
            zoom_factor = torch.empty(1).uniform_(*self.zoom_range).item()
            w, h = img.size
            new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
            img_zoomed = TF.resize(img, [new_h, new_w])

            if zoom_factor < 1.0:
                pad_w_total = w - new_w
                pad_h_total = h - new_h
                padding = (
                    pad_w_total // 2, pad_h_total // 2,
                    pad_w_total - pad_w_total // 2,
                    pad_h_total - pad_h_total // 2
                )
                img_zoomed = TF.pad(img_zoomed, padding, fill=0)
            else:
                crop_left = (new_w - w) // 2
                crop_top = (new_h - h) // 2
                img_zoomed = TF.crop(img_zoomed, crop_top, crop_left, h, w)

            return img_zoomed
        return img
