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

# Define the Enhanced CNN Model
class EnhancedDrowsinessCNN(nn.Module):
    def __init__(self):
        super(EnhancedDrowsinessCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 12 * 12, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x).squeeze(-1)
        return x

def train_model(model, train_loader, val_loader, num_epochs=10):
    train_losses, val_losses, val_accuracies, val_f1_scores = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()
            labels = labels.view(-1)

            # Forward pass
            outputs = model(images)
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
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()
                labels = labels.view(-1)
                outputs = model(images)
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
        val_f1 = f1_score(all_labels, all_preds, average="binary")  # Calculate F1 score

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_f1_scores.append(val_f1)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")

        # Update learning rate
        scheduler.step()

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

def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float()
            outputs = model(images)
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

# Paths to train, validation, and test directories
data_base_dir = "../dataset/processed_videos_frames"
train_dir = os.path.join(data_base_dir, "train")
val_dir = os.path.join(data_base_dir, "val")
test_dir = os.path.join(data_base_dir, "test")

# Load datasets directly from the respective directories
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Class names
class_names = train_dataset.classes  # The same classes apply to all datasets
print(f"Classes: {class_names}")
print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = EnhancedDrowsinessCNN().to(device)

# Class weights (if the dataset is imbalanced)
class_weights = 5.0
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_weights).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# Train the model
train_losses, val_losses, val_accuracies, val_f1_scores = train_model(model, train_loader, val_loader, num_epochs=10)

# Plot training metrics
plot_training_metrics(train_losses, val_losses, val_accuracies, val_f1_scores)

# Save the model
torch.save(model.state_dict(), "./model_weight/cnn.pth")

# Load the model for inference
model.load_state_dict(torch.load("./model_weight/cnn.pth"))
model.eval()

evaluate_model(model, test_loader)

# Ensure the model and input are on the same device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Predict on a new person - drowsy
test_image_path = "../dataset/processed_videos_frames/test/drowsy/gC_13_s5_2019-03-12T10;03;00+01;00_rgb_face_frame_01311.jpg"
prediction = predict_image(test_image_path, model, transform, class_names, device)
print(f"Prediction: {prediction}")

# Predict on a new person - non-drowsy
test_image_path = "../dataset/processed_videos_frames/test/non-drowsy/gC_13_s5_2019-03-12T10;03;00+01;00_rgb_face_frame_00000.jpg"
prediction = predict_image(test_image_path, model, transform, class_names, device)
print(f"Prediction: {prediction}")