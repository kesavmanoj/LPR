import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

def parse_ccpd_filename(filename):
    """
    Parses a CCPD image filename and extracts metadata.

    Expected filename format (example):
    "0415948275862-90_83-160,417_581,529-573,528_182,521_171,414_562,421-0_0_11_14_33_33_31-32-104.jpg"

    Fields:
      1. Area ratio (not used for regression here)
      2. Tilt degree (not used here)
      3. Bounding box coordinates (not used here)
      4. Four vertices locations: 8 numbers, in order [x1, y1, x2, y2, x3, y3, x4, y4]
      5. License plate number indices (not used here)
      6. Brightness (not used)
      7. Blurriness (not used)
    """
    base = os.path.splitext(filename)[0]
    fields = base.split('-')
    if len(fields) != 7:
        raise ValueError(f"Expected 7 fields in filename, got {len(fields)}: {filename}")

    # Field 4: Four vertices locations (e.g., "573,528_182,521_171,414_562,421")
    vertices_str = fields[3].replace(',', '_')
    vertices_parts = vertices_str.split('_')
    if len(vertices_parts) != 8:
        raise ValueError(f"Expected 8 vertex values, got {len(vertices_parts)}: {vertices_str}")
    vertices = [int(v) for v in vertices_parts]

    return {
        "vertices": vertices  # absolute coordinates from the original image
    }

# We'll use data augmentation transforms for training.
# For regression, we usually want to apply the same geometric transforms to both the image and the coordinates.
# For simplicity, here we perform only a basic resizing and conversion to tensor in the transform.
# (In a production system you might want to write a custom transform that applies the same random flip/rotation to the coordinates.)
basic_transform = transforms.Compose([
    transforms.ToPILImage(),
    # Here you could add additional transforms (e.g. RandomHorizontalFlip, ColorJitter, RandomRotation)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class CCPDDataset(Dataset):
    """
    PyTorch Dataset for CCPD images, for corner regression.
    For each image, we read the image, resize it to a fixed size,
    and compute the ground truth corner coordinates (rescaled accordingly).
    """
    def __init__(self, root_dir, target_size=(256, 64), transform=None, max_images=None):
        """
        Args:
            root_dir (str): Directory containing CCPD images.
            target_size (tuple): (width, height) for resizing images.
            transform (callable, optional): Transform to apply to the image.
            max_images (int, optional): Limit the number of images.
        """
        self.root_dir = root_dir
        self.target_size = target_size  # (width, height)
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname)
                            for fname in os.listdir(root_dir) if fname.endswith('.jpg')]
        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # Load image in BGR (OpenCV default)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        orig_h, orig_w = image.shape[:2]
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Parse metadata to get original corner coordinates
        metadata = parse_ccpd_filename(os.path.basename(image_path))
        gt_vertices = metadata["vertices"]  # list of 8 integers

        # Resize image to target_size (width, height)
        target_w, target_h = self.target_size
        image_resized = cv2.resize(image, (target_w, target_h))
        
        # Compute scale factors to adjust the coordinates
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        # Rescale ground truth vertices
        gt_vertices_resized = []
        for i, v in enumerate(gt_vertices):
            if i % 2 == 0:  # x coordinate
                gt_vertices_resized.append(v * scale_x)
            else:           # y coordinate
                gt_vertices_resized.append(v * scale_y)
        
        # Optionally apply transform to image (the transform expects a PIL image, so we apply it on image_resized)
        if self.transform:
            image_transformed = self.transform(image_resized)
        else:
            # Convert image_resized to tensor if no transform provided
            image_transformed = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0

        sample = {
            "image": image_transformed,   # Tensor shape: [3, target_h, target_w]
            "vertices": torch.tensor(gt_vertices_resized, dtype=torch.float32),  # shape: [8]
            "filename": os.path.basename(image_path)
        }
        return sample

def custom_collate_fn(batch):
    """Return a list of sample dictionaries."""
    return batch

def load_data():
    """
    Loads the CCPD dataset for corner regression and returns train and validation DataLoaders.
    """
    dataset_path = "/content/LPR-Project/dataset/ccpd-preprocess/CCPD2019/ccpd_base"
    dataset = CCPDDataset(dataset_path, target_size=(256, 64), transform=basic_transform, max_images=4000)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)
    return train_loader, val_loader

if __name__ == "__main__":
    # Quick test: print one batch's filenames and ground truth vertices.
    train_loader, _ = load_data()
    for batch in train_loader:
        print("Filenames in batch:", [sample["filename"] for sample in batch])
        print("Ground truth vertices for first sample:", batch[0]["vertices"])
        break
