import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class DriverDrowsinessDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, seq_len=16, padding_value=0.0, default_img_size=224):
        """
        Initializes the dataset by listing image sequences and labels, with padding for incomplete windows.

        Args:
            root_dir (str): Root directory containing 'train', 'val', 'test' splits.
            split (str, optional): One of 'train', 'val', 'test'. Defaults to 'train'.
            transform (callable, optional): Transformations to apply to images.
            seq_len (int, optional): Number of images per sequence. Defaults to 16.
            padding_value (float, optional): Value to use for padding incomplete sequences. Defaults to 0.0.
            default_img_size (int, optional): Value to use for image padding. Defaults to 224.
        """
        # Initialize attributes
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.seq_len = seq_len
        self.padding_value = padding_value
        self.sequences = []  # Store sequences of file paths
        self.labels = []  # Store corresponding labels for sequences
        self.default_size = default_img_size

        # Validate the split argument
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be one of 'train', 'val', 'test'.")

        # Construct the path for the current split
        split_dir = os.path.join(root_dir, split)
        
        # Process each label directory (positive and negative samples)
        for label in ['pos', 'neg']:
            label_dir = os.path.join(split_dir, label)
            if not os.path.exists(label_dir):
                # Skip if the label directory is missing
                print(f"Warning: Directory {label_dir} not found.")
                continue
            for sequence in os.listdir(label_dir):
                sequence_dir = os.path.join(label_dir, sequence)
                images = sorted([
                    img for img in os.listdir(sequence_dir)
                    if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
                ])  # Sort images to ensure temporal order
                
                # Create overlapping sequences with stride seq_len // 2
                stride = self.seq_len // 2
                for i in range(0, len(images), stride):
                    seq = images[i:i + self.seq_len]
                    if len(seq) == self.seq_len:
                        # Sequence is complete
                        self.sequences.append([os.path.join(sequence_dir, img) for img in seq])
                        self.labels.append(1 if label == 'pos' else 0)
                    else:
                        # Pad incomplete sequences with None
                        padded_seq = seq + [None] * (self.seq_len - len(seq))
                        self.sequences.append([os.path.join(sequence_dir, img) if img is not None else None for img in padded_seq])
                        self.labels.append(1 if label == 'pos' else 0)

    def __len__(self):
        """
        Returns the total number of sequences in the dataset.
        """
        return len(self.sequences)

    def _validate_transformed_data(self, images):
        if torch.isnan(images).any() or torch.isinf(images).any():
            raise ValueError("NaN or Inf detected after transformations.")
    
    def __getitem__(self, idx):
        """
        Retrieves a sequence of images and its label.

        Args:
            idx (int): Index of the sequence to retrieve.

        Returns:
            Tuple[Tensor, Tensor, int]: 
                - Tensor of images [seq_len, C, H, W]
                - Tensor mask [seq_len] indicating valid (1) or padded (0) frames
                - Integer label (1 for positive, 0 for negative)
        """
        sequence = self.sequences[idx]
        label = self.labels[idx]
        images = []  # Store image tensors
        mask = []  # Store mask indicating valid or padded frames
        
        for img_path in sequence:
            try:
                if img_path is not None:
                    # Load and transform a valid image
                    image = Image.open(img_path).convert('RGB')
                    if self.transform:
                        image = self.transform(image)
                    images.append(image)
                    mask.append(1)  # Mark as valid
                else:
                    # Create a black dummy image tensor for padding
                    dummy_image = torch.zeros((3, self.default_size, self.default_size))
                    images.append(dummy_image)
                    mask.append(0)  # Mark as padded
            except Exception as e:
                # Handle errors gracefully and use a dummy image for problematic files
                print(f"Error loading image {img_path}: {e}")
                dummy_image = torch.zeros((3, self.default_size, self.default_size))
                images.append(dummy_image)
                mask.append(0)
        
        # Stack image tensors into a single tensor [seq_len, C, H, W]
        images = torch.stack(images)
        self._validate_transformed_data(images)
        # Convert mask to tensor [seq_len]
        mask = torch.tensor(mask, dtype=torch.bool)
        return images, mask, label