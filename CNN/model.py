import torch
import torch.nn as nn
import torch.nn.functional as F

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


