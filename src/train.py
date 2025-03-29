import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.model import LPRNet
from src.utils import load_data
from tqdm import tqdm

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LPRNet().to(device)
    train_loader, _ = load_data()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Adjusted learning rate
    num_epochs = 15

    # Checkpoint setup
    checkpoint_dir = "../model"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")

    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for batch in pbar:
            # Preprocess images: shape [batch, 3, 64, 256]
            images = torch.stack([
                torch.tensor(sample["image"]).permute(2, 0, 1).float() / 255.0
                for sample in batch
            ]).to(device)

            batch_size = images.size(0)
            # Build target tensors from metadata using the actual plate_indices
            targets = []
            for i in range(7):
                target_i = torch.tensor([sample["metadata"]["plate_indices"][i] for sample in batch],
                                        dtype=torch.long).to(device)
                targets.append(target_i)

            # Debug: print unique targets for the first batch (optional)
            # for i in range(7):
            #     unique_targets = torch.unique(targets[i])
            #     print(f"Unique targets for character {i+1}: {unique_targets}")

            optimizer.zero_grad()
            outputs = model(images)  # List of 7 outputs, each [batch, num_classes]
            loss = 0
            for out, target in zip(outputs, targets):
                loss += criterion(out, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")

        # Save checkpoint after each epoch
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint_data, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch+1}")

    # Save final model
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "vertexNet.pth"))
    print("Final model saved to ../model/vertexNet.pth")

if __name__ == "__main__":
    train()
