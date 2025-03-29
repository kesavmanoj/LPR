import torch
import torch.nn as nn
import torch.nn.functional as F

class LPRNet(nn.Module):
    def __init__(self, num_classes_list=[34, 25, 35, 35, 35, 35, 35]):
        """
        A simple CNN for license plate recognition.

        num_classes_list:
          - 1st character (province): 34 classes
          - 2nd character (alphabet): 25 classes
          - 3rd to 7th characters (alphanumerics): 35 classes each.
          Total = 234 classes.
        """
        super(LPRNet, self).__init__()
        self.num_classes_list = num_classes_list
        self.total_classes = sum(num_classes_list)  # 234

        # Example CNN architecture:
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        # With input (3,64,256) and 3 poolings, output shape is (128, 8, 32)
        self.fc1 = nn.Linear(128 * 8 * 32, 512)
        self.fc2 = nn.Linear(512, self.total_classes)

    def forward(self, x):
        # x shape: [batch, 3, 64, 256]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)  # Now shape: [batch, 128, 8, 32]
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)  # shape: [batch, 234]
        # Split logits into a list for each character position:
        outputs = []
        start = 0
        for n in self.num_classes_list:
            outputs.append(logits[:, start:start+n])
            start += n
        return outputs

if __name__ == "__main__":
    model = LPRNet()
    x = torch.randn(4, 3, 64, 256)
    outputs = model(x)
    for i, out in enumerate(outputs):
        print(f"Output for character {i+1}: shape {out.shape}")
