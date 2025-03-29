import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.model_corner import LPRCornerNet
from src.utils import load_data
from tqdm import tqdm

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LPRCornerNet().to(device)
    train_loader, _ = load_data()

    # Use SmoothL1Loss (Huber loss) for regression
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    num_epochs = 15

    checkpoint_dir = "../model"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_corner.pth")

    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming corner training from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for batch in pbar:
            images = torch.stack([sample["image"] for sample in batch]).to(device)
            batch_size = images.size(0)
            # Ground truth corner coordinates: tensor shape [batch, 8]
            targets = torch.stack([sample["vertices"] for sample in batch]).to(device)
            
            optimizer.zero_grad()
            outputs = model(images)  # shape: [batch, 8]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")
        scheduler.step()

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint_data, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch+1}")
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "LPRCornerNet.pth"))
    print("Final corner model saved to ../model/LPRCornerNet.pth")

if __name__ == "__main__":
    train()
