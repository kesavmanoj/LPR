import torch
import torch.nn as nn
from src.model import LPRNet
from src.utils import load_data

# Lookup arrays for decoding
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
             "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
             "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
             "新", "警", "学", "O"]

alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M',
             'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']

ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N',
       'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
       '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model
    model = LPRNet().to(device)
    checkpoint_path = "../model/vertexNet.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Load validation data
    _, val_loader = load_data()

    print("Testing the model on validation data:")
    for batch in val_loader:
        images = torch.stack([
            torch.tensor(sample["image"]).permute(2, 0, 1).float() / 255.0
            for sample in batch
        ]).to(device)
        batch_size = images.size(0)
        
        with torch.no_grad():
            outputs = model(images)
        
        # For each output tensor, get the predicted indices
        preds = [torch.argmax(out, dim=1).cpu().tolist() for out in outputs]
        
        # Decode predictions and display actual vs predicted license plate
        for i in range(batch_size):
            predicted_plate = (
                provinces[preds[0][i]] +
                alphabets[preds[1][i]] +
                "".join(ads[preds[j][i]] for j in range(2, 7))
            )
            actual_plate = batch[i]["metadata"]["plate_number"]
            print(f"Sample {i+1}:")
            print("  Actual License Plate:    ", actual_plate)
            print("  Predicted License Plate: ", predicted_plate)
        break  # Process one batch for demonstration

if __name__ == "__main__":
    test()
