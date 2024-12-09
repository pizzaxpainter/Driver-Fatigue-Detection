import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score, classification_report
from collections import Counter
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomHorizontalFlip, ColorJitter, RandomRotation, RandomAffine, RandomPerspective, ToTensor, Normalize
from model import CNN3D  
from dataset import VideoFrameDataset3D  
from train import train_model2  
from evaluate import evaluate_model2, plot_training_metrics
from preprocess import transform

# Paths to train, validation, and test directories
data_base_dir = "../dataset/processed_videos_frames"
train_dir = os.path.join(data_base_dir, "train")
val_dir = os.path.join(data_base_dir, "val")
test_dir = os.path.join(data_base_dir, "test")

# Load datasets directly from the respective directories
sequence_length = 10
train_dataset = VideoFrameDataset3D(train_dir, transform=transform, sequence_length=sequence_length)
val_dataset = VideoFrameDataset3D(val_dir, transform=transform, sequence_length=sequence_length)
test_dataset = VideoFrameDataset3D(test_dir, transform=transform, sequence_length=sequence_length)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Compute class counts for ImageFolder dataset
train_class_counts = Counter(train_dataset.targets)  # `targets` contains the class indices for all samples
val_class_counts = Counter(val_dataset.targets)
test_class_counts = Counter(test_dataset.targets)

# Get the count for each class
drowsy_count = train_class_counts[train_dataset.class_to_idx['drowsy']]
non_drowsy_count = train_class_counts[train_dataset.class_to_idx['non-drowsy']]

# Compute class weights
class_weights = non_drowsy_count / drowsy_count
print(f"Train Dataset - Drowsy: {drowsy_count}, Non-Drowsy: {non_drowsy_count}")
print(f"Class weights (Train): {class_weights}")

# Print for validation and test datasets
print("Validation Dataset:", val_class_counts)
print("Test Dataset:", test_class_counts)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the 3D CNN model
model = CNN3D().to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_weights).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# Train the model
train_losses, val_losses, val_accuracies, val_f1_scores = train_model2(
    model, train_loader, val_loader, num_epochs=10
)

# Plot training metrics
plot_training_metrics(train_losses, val_losses, val_accuracies, val_f1_scores)

# Save the model
torch.save(model.state_dict(), "./model_weight/cnn2.pth")

# Load the model for inference
model.load_state_dict(torch.load("./model_weight/cnn2.pth"))
model.eval()

class_names = ['non-drowsy', 'drowsy']
evaluate_model2(model, test_loader, class_names)