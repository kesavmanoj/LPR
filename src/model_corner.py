import torch
import torch.nn as nn
import torch.nn.functional as F

class LPRCornerNet(nn.Module):
    def __init__(self):
        super(LPRCornerNet, self).__init__()
        # A simple CNN for regression.
        # Input shape: [batch, 3, 64, 256] (height=64, width=256)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)  # halves spatial dimensions

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # After three poolings, from 64x256 -> 32x128 -> 16x64 -> 8x32

        self.fc1 = nn.Linear(128 * 8 * 32, 512)
        self.fc2 = nn.Linear(512, 8)  # Predict 8 coordinates

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)  # Shape: [batch, 128, 8, 32]

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        coords = self.fc2(x)  # Shape: [batch, 8]
        return coords

if __name__ == "__main__":
    model = LPRCornerNet()
    x = torch.randn(4, 3, 64, 256)
    preds = model(x)
    print("Predicted corner coordinates shape:", preds.shape)
