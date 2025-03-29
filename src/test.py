import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.model_corner import LPRCornerNet
from src.utils import load_data

def denormalize_image(tensor):
    """
    Reverses the normalization from the transform.
    Our transform uses mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
    Converts a tensor of shape [3, H, W] back to a uint8 image (H, W, 3).
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # tensor to numpy, shape: [H, W, 3]
    image = tensor.cpu().numpy().transpose(1, 2, 0)
    image = image * std + mean  # inverse normalization
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    return image

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LPRCornerNet().to(device)
    checkpoint_path = "../model/LPRCornerNet.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    _, val_loader = load_data()
    print("Testing corner detection on validation data:")
    for batch in val_loader:
        # Images are already transformed to tensor via our transform
        images = torch.stack([sample["image"] for sample in batch]).to(device)
        batch_size = images.size(0)
        with torch.no_grad():
            outputs = model(images)  # shape: [batch, 8]
        for i in range(batch_size):
            predicted_corners = outputs[i].cpu().tolist()  # List of 8 floats
            actual_corners = batch[i]["vertices"].cpu().tolist()  # List of 8 floats
            print(f"Sample {i+1}:")
            print("  Actual Corners:    ", actual_corners)
            print("  Predicted Corners: ", predicted_corners)
            
            # Denormalize image for display (convert from normalized tensor to uint8 image)
            img = denormalize_image(batch[i]["image"])
            
            # Reshape corner coordinates into arrays of shape (4,2)
            pred_pts = np.array(predicted_corners, dtype=np.int32).reshape((4, 2))
            gt_pts = np.array(actual_corners, dtype=np.int32).reshape((4, 2))
            
            # Draw polygons:
            # Draw predicted corners in blue (BGR: (255, 0, 0))
            cv2.polylines(img, [pred_pts], isClosed=True, color=(255, 0, 0), thickness=2)
            # Draw ground truth corners in red (BGR: (0, 0, 255))
            cv2.polylines(img, [gt_pts], isClosed=True, color=(0, 0, 255), thickness=2)
            
            # Convert BGR (OpenCV default) to RGB for matplotlib
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Display the image using matplotlib
            plt.figure(figsize=(6,3))
            plt.imshow(img_rgb)
            plt.title(f"Sample {i+1}")
            plt.axis("off")
            plt.show()
        break  # Process one batch for demonstration

if __name__ == "__main__":
    test()
