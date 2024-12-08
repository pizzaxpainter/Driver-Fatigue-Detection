import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from collections import Counter
from PIL import Image

# Data Preprocessing
class VideoFrameDataset3D(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, sequence_length=10):
        """
        Custom Dataset for 3D CNN using sequences of frames.
        Args:
            data_dir (str): Directory where frames are stored (e.g., train, val, test).
            transform (callable, optional): Transformations to apply to each frame.
            sequence_length (int): Number of frames in each sequence.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples, self.targets = self._load_samples()  # Load samples and targets

    def _load_samples(self):
        samples = []
        targets = []
        for cls in self.classes:
            cls_dir = os.path.join(self.data_dir, cls)
            frames = sorted(os.listdir(cls_dir))
            frame_paths = [os.path.join(cls_dir, frame) for frame in frames]
            for i in range(len(frame_paths) - self.sequence_length + 1):
                sequence = frame_paths[i:i + self.sequence_length]
                samples.append(sequence)
                targets.append(self.class_to_idx[cls])  # Append class index as target
        return samples, targets

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sequence, label = self.samples[idx], self.targets[idx]
        frames = []
        for frame_path in sequence:
            image = Image.open(frame_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            frames.append(image)
        frames = torch.stack(frames)  # Shape: (sequence_length, C, H, W)
        return frames, label

# Model Architecture
class CNN3D(nn.Module):
    def __init__(self, num_classes=1):
        super(CNN3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        # Adjust the input size of the fully connected layer
        # Calculate the output size of the last Conv3D+Pool3D layer
        example_input = torch.randn(1, 3, 10, 112, 112)  # (batch_size, channels, depth, height, width)
        with torch.no_grad():
            example_output = self.pool(self.bn3(self.conv3(self.pool(self.bn2(self.conv2(self.pool(self.bn1(self.conv1(example_input)))))))))
            flattened_size = example_output.numel()  # Calculate the number of features

        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x).squeeze(-1)
        return x

def train_model2(model, train_loader, val_loader, num_epochs=10, patience=3):
    """
    Train and validate a model with early stopping.

    Args:
        model: The PyTorch model to be trained.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        num_epochs: Number of training epochs.
        patience: Number of epochs to wait for improvement before early stopping.

    Returns:
        train_losses: List of training losses per epoch.
        val_losses: List of validation losses per epoch.
        val_accuracies: List of validation accuracies per epoch.
        val_f1_scores: List of validation F1 scores per epoch.
    """
    train_losses, val_losses, val_accuracies, val_f1_scores = [], [], [], []
    best_val_loss = float('inf')
    best_model_weights = None
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for sequences, labels in train_loader:
            # Adjust dimensions for Conv3D: (batch_size, C, depth, H, W)
            sequences = sequences.permute(0, 2, 1, 3, 4)  # From (batch_size, sequence_length, C, H, W)
            sequences, labels = sequences.to(device), labels.to(device).float()

            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation loop
        model.eval()
        val_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for sequences, labels in val_loader:
                # Adjust dimensions for Conv3D: (batch_size, C, depth, H, W)
                sequences = sequences.permute(0, 2, 1, 3, 4)
                sequences, labels = sequences.to(device), labels.to(device).float()

                outputs = model(sequences)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Collect predictions and labels for F1 score
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        val_accuracy = correct / total
        val_f1 = f1_score(all_labels, all_preds, average="binary")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_f1_scores.append(val_f1)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")

        # Save best model weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict()  # Save best weights
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # Update learning rate
        scheduler.step()

    # Load best model weights
    if best_model_weights:
        model.load_state_dict(best_model_weights)

    return train_losses, val_losses, val_accuracies, val_f1_scores

def plot_training_metrics(train_losses, val_losses, val_accuracies, val_f1_scores):
    """
    Plot training and validation metrics including loss, accuracy, and F1 score.

    Args:
        train_losses (list): Training loss for each epoch.
        val_losses (list): Validation loss for each epoch.
        val_accuracies (list): Validation accuracy for each epoch.
        val_f1_scores (list): Validation F1 score for each epoch.
    """
    # Plot training and validation loss
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot validation accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(val_accuracies, label='Validation Accuracy', marker='^')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot F1 score
    plt.figure(figsize=(8, 5))
    plt.plot(val_f1_scores, label='Validation F1 Score', marker='^', color='purple')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score')
    plt.legend()
    plt.grid(True)
    plt.show()

def evaluate_model2(model, test_loader, class_names):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for sequences, labels in test_loader:
            # Ensure the input dimensions are correct for Conv3D
            sequences, labels = sequences.to(device), labels.to(device).float()
            sequences = sequences.permute(0, 2, 1, 3, 4)  # Convert to (batch_size, channels, depth, height, width)

            # Forward pass
            outputs = model(sequences)
            predicted = (torch.sigmoid(outputs) > 0.5).float()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

    # Calculate precision, recall, and F1 score
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

def predict_image(image_path, model, transform, class_names, device):
    """
    Predict the class of a given image using the model.

    Args:
        image_path (str): Path to the image.
        model (torch.nn.Module): Trained model.
        transform (torchvision.transforms.Compose): Transformations to apply to the image.
        class_names (list): List of class names.
        device (torch.device): Device to run the model on (CPU or GPU).

    Returns:
        str: Predicted class label.
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')

    # Display the image
    plt.imshow(image)
    plt.axis('off')
    plt.title("Input Image")
    plt.show()

    # Apply transformations and add batch dimension
    image_tensor = transform(image).unsqueeze(0).to(device)  # Move to the same device as the model

    # Predict
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        predicted = (output.squeeze() > 0.5).float().item()

    return class_names[int(predicted)]

# Define transformations for the data
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=15),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

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