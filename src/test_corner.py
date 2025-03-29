import torch
import csv
import os
import numpy as np
from src.model_corner import LPRCornerNet
from src.utils import load_data

# Define the dataset path used in the dataset loader
DATASET_PATH = "/content/LPR-Project/dataset/ccpd-preprocess/CCPD2019/ccpd_base"

def test_and_save_csv(csv_filename="corner_results.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LPRCornerNet().to(device)
    checkpoint_path = "../model/LPRCornerNet.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    _, val_loader = load_data()
    print("Testing corner detection on validation data...")

    results = []  # List to hold CSV rows
    for batch in val_loader:
        batch_size = len(batch)
        images = torch.stack([sample["image"] for sample in batch]).to(device)
        
        with torch.no_grad():
            outputs = model(images)  # shape: [batch, 8]

        for i in range(batch_size):
            predicted_corners = outputs[i].cpu().tolist()  # List of 8 floats
            actual_corners = batch[i]["vertices"].cpu().tolist()  # List of 8 floats
            filename = batch[i]["filename"]
            image_path = os.path.join(DATASET_PATH, filename)
            
            # Correctly join the corner values as strings
            actual_str = ", ".join([f"{v:.2f}" for v in actual_corners])
            predicted_str = ", ".join([f"{v:.2f}" for v in predicted_corners])
            
            results.append({
                "filename": filename,
                "image_path": image_path,
                "actual_corners": actual_str,
                "predicted_corners": predicted_str
            })
        break  # Process one batch for demonstration

    with open(csv_filename, mode="w", newline="") as csv_file:
        fieldnames = ["filename", "image_path", "actual_corners", "predicted_corners"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    print(f"Results saved to {csv_filename}")

if __name__ == "__main__":
    test_and_save_csv()
