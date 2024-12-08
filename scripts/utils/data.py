import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import random


class DriverDrowsinessDataset(Dataset):
    """
    Dataset for driver drowsiness detection using sequential image data.
    Handles loading, preprocessing, and providing access to image sequences
    with corresponding drowsiness labels.

    The dataset supports:
    - Multiple data splits (train/val/test)
    - Sequence padding for incomplete sequences
    - Synchronized transformations across frames during training
    - Masking for valid/invalid frames
    - Limiting the number of image frames per sequence via max_length
    """

    def __init__(self, root_dir, split='train', transform=None, seq_len=16, padding_value=0.0, default_img_size=224, stride=None, max_length=None):
        """
        Initialize dataset with configuration parameters and load data paths.

        Args:
            root_dir (str): Root directory containing split subdirectories
            split (str): Dataset split ('train', 'val', 'test')
            transform (callable): Optional transforms to apply to images
            seq_len (int): Length of image sequences
            padding_value (float): Value for padding incomplete sequences
            default_img_size (int): Size for dummy images when padding
            stride (int, optional): Stride for creating overlapping sequences
            max_length (int, optional): Maximum number of image frames to include per sequence. 
                                         If None, includes all frames.
        """
        # Store initialization parameters
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.seq_len = seq_len
        self.padding_value = padding_value
        self.sequences = []  # List to store image paths
        self.labels = []     # List to store corresponding labels
        self.default_size = default_img_size
        self.stride = stride if stride is not None else seq_len // 2
        self.max_length = max_length  # NEW: Store max_length parameter

        # Validate split parameter
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be one of 'train', 'val', 'test'.")

        split_dir = os.path.join(root_dir, split)
        
        # Process positive and negative samples
        for label in ['pos', 'neg']:
            label_dir = os.path.join(split_dir, label)
            if not os.path.exists(label_dir):
                print(f"Warning: Directory {label_dir} not found.")
                continue
            
            # Process each sequence directory
            for sequence in os.listdir(label_dir):
                sequence_dir = os.path.join(label_dir, sequence)
                # Get sorted list of valid image files
                images = sorted([
                    img for img in os.listdir(sequence_dir)
                    if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
                ])
                
                # Apply max_length if specified
                if self.max_length is not None:
                    images = images[:self.max_length]  # NEW: Limit the number of images per sequence

                # Create overlapping sequences with stride
                for i in range(0, len(images), self.stride):
                    seq = images[i:i + self.seq_len]
                    if len(seq) == self.seq_len:
                        # Store complete sequences
                        self.sequences.append([os.path.join(sequence_dir, img) for img in seq])
                        self.labels.append(1 if label == 'pos' else 0)
                    else:
                        # Pad incomplete sequences
                        padded_seq = seq + [None] * (self.seq_len - len(seq))
                        self.sequences.append([os.path.join(sequence_dir, img) if img is not None else None for img in padded_seq])
                        self.labels.append(1 if label == 'pos' else 0)

    def __len__(self):
        """Return total number of sequences in the dataset."""
        return len(self.sequences)

    def _validate_transformed_data(self, images):
        """
        Validate transformed image tensor for NaN or Inf values.
        
        Args:
            images (torch.Tensor): Tensor of transformed images
        
        Raises:
            ValueError: If NaN or Inf values are detected
        """
        if torch.isnan(images).any() or torch.isinf(images).any():
            raise ValueError("NaN or Inf detected after transformations.")
    
    def __getitem__(self, idx):
        """
        Get a sequence of images and corresponding label.
        
        For training split, applies synchronized transformations across
        all frames in the sequence using index-based seeding.

        Args:
            idx (int): Index of sequence to retrieve

        Returns:
            tuple: Contains:
                - images (torch.Tensor): Image sequence tensor [seq_len, C, H, W]
                - mask (torch.Tensor): Boolean mask for valid frames [seq_len]
                - label (int): Sequence label (0 or 1)
        """
        sequence = self.sequences[idx]
        label = self.labels[idx]
        images = []
        mask = []
        
        # Load images and create validity mask
        for img_path in sequence:
            if img_path is not None:
                try:
                    image = Image.open(img_path).convert('RGB')
                    images.append(image)
                    mask.append(1)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    dummy_image = Image.new('RGB', (self.default_size, self.default_size), color=0)
                    images.append(dummy_image)
                    mask.append(0)
            else:
                dummy_image = Image.new('RGB', (self.default_size, self.default_size), color=0)
                images.append(dummy_image)
                mask.append(0)

        # Apply transformations
        if self.transform and self.split == 'train':
            # Synchronize transformations across sequence
            random_state = torch.get_rng_state()
            torch.manual_seed(idx)
            
            transformed_images = []
            for image in images:
                torch.set_rng_state(random_state)  # Use same random state for each frame
                transformed_images.append(self.transform(image))
            
            images = transformed_images
        elif self.transform:
            images = [self.transform(image) for image in images]
        else:
            to_tensor = transforms.ToTensor()
            images = [to_tensor(image) for image in images]

        # Stack and validate images
        images = torch.stack(images)
        self._validate_transformed_data(images)
        mask = torch.tensor(mask, dtype=torch.bool)
        return images, mask, label
    
class DriverDrowsinessDatasetv2(Dataset):
    """
    Dataset for driver drowsiness detection using sequential image data.
    Handles loading, preprocessing, and providing access to image sequences
    with corresponding drowsiness labels.

    The dataset supports:
    - Multiple data splits (train/val/test)
    - Sequence padding for incomplete sequences
    - Synchronized transformations across frames during training
    - Masking for valid/invalid frames
    - Limiting the number of image frames per sequence via max_length,
      with random slicing during training if sequences exceed max_length
      and maintaining overlapping sub-sequences.
    """

    def __init__(
        self,
        root_dir,
        split='train',
        transform=None,
        seq_len=16,
        padding_value=0.0,
        default_img_size=224,
        stride=None,
        max_length=None
    ):
        """
        Initialize dataset with configuration parameters and load data paths.

        Args:
            root_dir (str): Root directory containing split subdirectories
            split (str): Dataset split ('train', 'val', 'test')
            transform (callable): Optional transforms to apply to images
            seq_len (int): Length of image sequences
            padding_value (float): Value for padding incomplete sequences
            default_img_size (int): Size for dummy images when padding
            stride (int, optional): Stride for creating overlapping sequences
            max_length (int, optional): Maximum number of image frames to include per sequence.
                                         If None, includes all frames. If specified and the
                                         sequence length exceeds max_length, during training,
                                         a random slice of max_length consecutive frames is selected.
                                         For validation and testing, the first max_length frames are used.
        """
        # Store initialization parameters
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.seq_len = seq_len
        self.padding_value = padding_value
        self.sequences = []  # List to store image paths
        self.labels = []     # List to store corresponding labels
        self.default_size = default_img_size
        self.max_length = max_length  # Store max_length parameter

        # Set stride: default to half of seq_len or a fraction of max_length if provided
        if stride is not None:
            self.stride = stride
        elif self.max_length is not None:
            self.stride = max(1, self.max_length // 4)  # 75% overlap
        else:
            self.stride = seq_len // 2  # 50% overlap

        # Validate split parameter
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be one of 'train', 'val', 'test'.")

        split_dir = os.path.join(root_dir, split)

        # Process positive and negative samples
        for label in ['pos', 'neg']:
            label_dir = os.path.join(split_dir, label)
            if not os.path.exists(label_dir):
                print(f"Warning: Directory {label_dir} not found.")
                continue

            # Process each sequence directory
            for sequence in os.listdir(label_dir):
                sequence_dir = os.path.join(label_dir, sequence)
                if not os.path.isdir(sequence_dir):
                    print(f"Warning: {sequence_dir} is not a directory.")
                    continue

                # Get sorted list of valid image files
                images = sorted([
                    img for img in os.listdir(sequence_dir)
                    if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
                ])

                # Apply max_length if specified
                if self.max_length is not None and len(images) > self.max_length:
                    if self.split == 'train':
                        # Randomly select a starting index for the slice
                        max_start = len(images) - self.max_length
                        start_idx = random.randint(0, max_start)
                        images = images[start_idx:start_idx + self.max_length]
                    else:
                        # Deterministically select the first max_length frames
                        images = images[:self.max_length]

                # Create overlapping sequences with stride
                for i in range(0, len(images), self.stride):
                    seq = images[i:i + self.seq_len]
                    if len(seq) == self.seq_len:
                        # Store complete sequences
                        self.sequences.append([os.path.join(sequence_dir, img) for img in seq])
                        self.labels.append(1 if label == 'pos' else 0)
                    else:
                        # Pad incomplete sequences
                        padded_seq = seq + [None] * (self.seq_len - len(seq))
                        self.sequences.append([os.path.join(sequence_dir, img) if img is not None else None for img in padded_seq])
                        self.labels.append(1 if label == 'pos' else 0)

    def __len__(self):
        """Return total number of sequences in the dataset."""
        return len(self.sequences)

    def _validate_transformed_data(self, images):
        """
        Validate transformed image tensor for NaN or Inf values.
        
        Args:
            images (torch.Tensor): Tensor of transformed images
        
        Raises:
            ValueError: If NaN or Inf values are detected
        """
        if torch.isnan(images).any() or torch.isinf(images).any():
            raise ValueError("NaN or Inf detected after transformations.")

    def __getitem__(self, idx):
        """
        Get a sequence of images and corresponding label.
        
        For training split, applies synchronized transformations across
        all frames in the sequence using index-based seeding.

        Args:
            idx (int): Index of sequence to retrieve

        Returns:
            tuple: Contains:
                - images (torch.Tensor): Image sequence tensor [seq_len, C, H, W]
                - mask (torch.Tensor): Boolean mask for valid frames [seq_len]
                - label (int): Sequence label (0 or 1)
        """
        sequence = self.sequences[idx]
        label = self.labels[idx]
        images = []
        mask = []

        # Load images and create validity mask
        for img_path in sequence:
            if img_path is not None:
                try:
                    image = Image.open(img_path).convert('RGB')
                    images.append(image)
                    mask.append(1)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    dummy_image = Image.new('RGB', (self.default_size, self.default_size), color=0)
                    images.append(dummy_image)
                    mask.append(0)
            else:
                dummy_image = Image.new('RGB', (self.default_size, self.default_size), color=0)
                images.append(dummy_image)
                mask.append(0)

        # Apply transformations
        if self.transform and self.split == 'train':
            # Synchronize transformations across sequence
            random_state = torch.get_rng_state()
            torch.manual_seed(idx)

            transformed_images = []
            for image in images:
                torch.set_rng_state(random_state)  # Use same random state for each frame
                transformed_images.append(self.transform(image))

            images = transformed_images
        elif self.transform:
            images = [self.transform(image) for image in images]
        else:
            to_tensor = transforms.ToTensor()
            images = [to_tensor(image) for image in images]

        # Stack and validate images
        images = torch.stack(images)
        self._validate_transformed_data(images)
        mask = torch.tensor(mask, dtype=torch.bool)
        return images, mask, label

class DriverDrowsinessDatasetv3(Dataset):
    """
    Dataset for driver drowsiness detection using sequential image data.
    Handles loading, preprocessing, and providing access to image sequences
    with corresponding drowsiness labels.

    The dataset supports:
    - Multiple data splits (train/val/test)
    - Sequence padding for incomplete sequences
    - Synchronized transformations across frames during training
    - Masking for valid/invalid frames
    - Limiting the number of image frames per sequence via max_length,
      with random slicing during training if sequences exceed max_length
      and maintaining overlapping sub-sequences.
    """

    def __init__(
        self,
        root_dir,
        split='train',
        transform=None,
        seq_len=16,
        padding_value=0.0,
        default_img_size=224,
        stride=None,
        max_length=None
    ):
        """
        Initialize dataset with configuration parameters and load data paths.

        Args:
            root_dir (str): Root directory containing split subdirectories
            split (str): Dataset split ('train', 'val', 'test')
            transform (callable): Optional transforms to apply to images
            seq_len (int): Length of image sequences
            padding_value (float): Value for padding incomplete sequences
            default_img_size (int): Size for dummy images when padding
            stride (int, optional): Stride for creating overlapping sequences
            max_length (int, optional): Maximum number of image frames to include per sequence.
                                         If None, includes all frames. If specified and the
                                         sequence length exceeds max_length, during training,
                                         a random slice of max_length consecutive frames is selected.
                                         For validation and testing, the first max_length frames are used.
        """
        # Store initialization parameters
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.seq_len = seq_len
        self.padding_value = padding_value
        self.sequences = []  # List to store image paths
        self.labels = []     # List to store corresponding labels
        self.default_size = default_img_size
        self.max_length = max_length  # Store max_length parameter

        # Set stride based on split and max_length
        if stride is not None:
            self.stride = stride
        elif self.max_length is not None:
            if self.split == 'train':
                self.stride = max(1, self.max_length // 4)  # 75% overlap
            else:
                self.stride = max(1, self.max_length // 2)  # 50% overlap for val/test
        else:
            self.stride = seq_len // 2  # Default 50% overlap

        # Validate split parameter
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be one of 'train', 'val', 'test'.")

        split_dir = os.path.join(root_dir, split)

        # Process positive and negative samples
        for label in ['pos', 'neg']:
            label_dir = os.path.join(split_dir, label)
            if not os.path.exists(label_dir):
                print(f"Warning: Directory {label_dir} not found.")
                continue

            # Process each sequence directory
            for sequence in os.listdir(label_dir):
                sequence_dir = os.path.join(label_dir, sequence)
                if not os.path.isdir(sequence_dir):
                    print(f"Warning: {sequence_dir} is not a directory.")
                    continue

                # Get sorted list of valid image files
                images = sorted([
                    img for img in os.listdir(sequence_dir)
                    if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
                ])

                # Skip if no images found
                if not images:
                    print(f"Warning: No valid images found in {sequence_dir}.")
                    continue

                # Apply max_length if specified
                if self.max_length is not None and len(images) > self.max_length:
                    if self.split == 'train':
                        # Randomly select a starting index for the slice
                        max_start = len(images) - self.max_length
                        start_idx = random.randint(0, max_start)
                        images = images[start_idx:start_idx + self.max_length]
                    else:
                        # Deterministically select the first max_length frames
                        images = images[:self.max_length]

                # Sequence generation based on split
                if self.split == 'train':
                    self._generate_training_sequences(images, sequence_dir, label)
                else:
                    self._generate_evaluation_sequences(images, sequence_dir, label)

    def _generate_training_sequences(self, images, sequence_dir, label):
        """
        Generate overlapping sequences for training split.

        Args:
            images (list): List of image filenames
            sequence_dir (str): Directory path of the current sequence
            label (str): Label of the sequence ('pos' or 'neg')
        """
        for i in range(0, len(images), self.stride):
            seq = images[i:i + self.seq_len]
            if len(seq) == self.seq_len:
                # Store complete sequences
                self.sequences.append([os.path.join(sequence_dir, img) for img in seq])
                self.labels.append(1 if label == 'pos' else 0)
            else:
                # Pad incomplete sequences
                padded_seq = seq + [None] * (self.seq_len - len(seq))
                self.sequences.append([
                    os.path.join(sequence_dir, img) if img is not None else None for img in padded_seq
                ])
                self.labels.append(1 if label == 'pos' else 0)

    def _generate_evaluation_sequences(self, images, sequence_dir, label):
        """
        Generate distinct sequences for validation/test splits.

        Args:
            images (list): List of image filenames
            sequence_dir (str): Directory path of the current sequence
            label (str): Label of the sequence ('pos' or 'neg')
        """
        num_subsequences = (len(images) - self.seq_len) // self.stride + 1
        if num_subsequences < 1:
            # If not enough frames, pad the sequence
            padded_seq = images + [None] * (self.seq_len - len(images))
            self.sequences.append([
                os.path.join(sequence_dir, img) if img is not None else None
                for img in padded_seq
            ])
            self.labels.append(1 if label == 'pos' else 0)
        else:
            # Create overlapping sub-sequences
            for i in range(0, len(images) - self.seq_len + 1, self.stride):
                seq = images[i:i + self.seq_len]
                self.sequences.append([
                    os.path.join(sequence_dir, img) for img in seq
                ])
                self.labels.append(1 if label == 'pos' else 0)

            # Handle remaining frames by padding if necessary
            remaining = len(images) - (num_subsequences * self.stride + self.seq_len)
            if remaining > 0:
                seq = images[-self.seq_len:]
                padded_seq = seq if len(seq) == self.seq_len else seq + [None] * (self.seq_len - len(seq))
                self.sequences.append([
                    os.path.join(sequence_dir, img) if img is not None else None
                    for img in padded_seq
                ])
                self.labels.append(1 if label == 'pos' else 0)

    def __len__(self):
        """Return total number of sequences in the dataset."""
        return len(self.sequences)

    def _validate_transformed_data(self, images):
        """
        Validate transformed image tensor for NaN or Inf values.

        Args:
            images (torch.Tensor): Tensor of transformed images

        Raises:
            ValueError: If NaN or Inf values are detected
        """
        if torch.isnan(images).any() or torch.isinf(images).any():
            raise ValueError("NaN or Inf detected after transformations.")

    def __getitem__(self, idx):
        """
        Get a sequence of images and corresponding label.

        For training split, applies synchronized transformations across
        all frames in the sequence using index-based seeding.

        Args:
            idx (int): Index of sequence to retrieve

        Returns:
            tuple: Contains:
                - images (torch.Tensor): Image sequence tensor [seq_len, C, H, W]
                - mask (torch.Tensor): Boolean mask for valid frames [seq_len]
                - label (int): Sequence label (0 or 1)
        """
        sequence = self.sequences[idx]
        label = self.labels[idx]
        images = []
        mask = []

        # Load images and create validity mask
        for img_path in sequence:
            if img_path is not None:
                try:
                    image = Image.open(img_path).convert('RGB')
                    images.append(image)
                    mask.append(1)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    dummy_image = Image.new('RGB', (self.default_size, self.default_size), color=0)
                    images.append(dummy_image)
                    mask.append(0)
            else:
                dummy_image = Image.new('RGB', (self.default_size, self.default_size), color=0)
                images.append(dummy_image)
                mask.append(0)

        # Apply transformations
        if self.transform and self.split == 'train':
            # Synchronize transformations across sequence
            random_state = torch.get_rng_state()
            torch.manual_seed(idx)

            transformed_images = []
            for image in images:
                torch.set_rng_state(random_state)  # Use same random state for each frame
                transformed_images.append(self.transform(image))

            images = transformed_images
        elif self.transform:
            images = [self.transform(image) for image in images]
        else:
            to_tensor = transforms.ToTensor()
            images = [to_tensor(image) for image in images]

        # Stack and validate images
        images = torch.stack(images)
        self._validate_transformed_data(images)
        mask = torch.tensor(mask, dtype=torch.bool)
        return images, mask, label

def visualize_sequence(dataset, idx, rows=2):
    """
    Visualizes a sequence of images from the dataset with their masks.
    
    Args:
        dataset: DriverDrowsinessDataset instance
        idx: Index of sequence to visualize
        rows: Number of rows for subplot grid
    """
    images, mask, label = dataset[idx]
    cols = dataset.seq_len // rows
    
    # Convert tensors back to displayable format
    if isinstance(images, torch.Tensor):
        denorm = transforms.Normalize(
            mean=[-m/s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
            std=[1/s for s in [0.229, 0.224, 0.225]]
        )
        images = denorm(images).clip(0, 1)
    
    plt.figure(figsize=(20, 8))
    for i in range(dataset.seq_len):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i].permute(1, 2, 0))
        border_color = 'green' if mask[i] else 'red'
        plt.gca().spines['bottom'].set_color(border_color)
        plt.gca().spines['top'].set_color(border_color)
        plt.gca().spines['left'].set_color(border_color)
        plt.gca().spines['right'].set_color(border_color)
        plt.title(f'Frame {i}\nValid: {mask[i].item()}')
        plt.axis('off')
    
    plt.suptitle(f'Sequence {idx} - Label: {"Drowsy" if label == 1 else "Alert"}')
    plt.tight_layout()
    plt.show()