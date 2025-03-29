import torch
import torch.nn as nn
from src.model import LPRNet
from src.utils import load_data

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the model
    model = LPRNet().to(device)
    model.load_state_dict(torch.load("../model/vertexNet.pth", map_location=device))
    model.eval()
    
    _, val_loader = load_data()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            images = torch.stack([
                torch.tensor(sample["image"]).permute(2, 0, 1).float() / 255.0 
                for sample in batch
            ]).to(device)
            batch_size = images.size(0)
            # --- Dummy targets for demonstration ---
            target_dims = [34, 25, 35, 35, 35, 35, 35]
            targets = [torch.zeros(batch_size, dtype=torch.long).to(device) for _ in target_dims]
            # -------------------------------------------
            
            outputs = model(images)
            loss = 0
            for out, target in zip(outputs, targets):
                loss += criterion(out, target)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    test()
