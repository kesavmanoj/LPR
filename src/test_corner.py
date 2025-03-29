import torch
from src.model_corner import LPRCornerNet
from src.utils import load_data

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LPRCornerNet().to(device)
    checkpoint_path = "../model/LPRCornerNet.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    _, val_loader = load_data()
    print("Testing corner detection on validation data:")
    for batch in val_loader:
        images = torch.stack([sample["image"] for sample in batch]).to(device)
        batch_size = images.size(0)
        with torch.no_grad():
            outputs = model(images)  # shape: [batch, 8]
        for i in range(batch_size):
            predicted_corners = outputs[i].cpu().tolist()
            actual_corners = batch[i]["vertices"].tolist()
            print(f"Sample {i+1}:")
            print("  Actual Corners:    ", actual_corners)
            print("  Predicted Corners: ", predicted_corners)
        break

if __name__ == "__main__":
    test()
