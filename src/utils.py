import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split

def parse_ccpd_filename(filename):
    """
    Parses a CCPD image filename and extracts metadata.
    
    Filename format (example):
    "025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg"
    
    Fields:
        1. Area ratio: Area ratio of license plate to the whole image.
        2. Tilt degree: Horizontal and vertical tilt (separated by '_').
        3. Bounding box coordinates: Left-up and right-bottom vertices (using '&' as separator).
        4. Four vertices: (x, y) pairs for the license plate starting from the right-bottom vertex.
        5. License plate number indices: Indices for province, alphabet, and alphanumeric characters.
        6. Brightness: Brightness of the license plate region.
        7. Blurriness: Blurriness of the license plate region.
    """
    # Remove file extension
    filename = os.path.splitext(filename)[0]
    
    # Split the filename into fields
    parts = filename.split('-')
    
    # Field 1: Area ratio (convert to a decimal value)
    area_ratio = float(parts[0]) / 100  
    
    # Field 2: Tilt degree (horizontal, vertical)
    tilt_degree = tuple(map(int, parts[1].split('_')))
    
    # Field 3: Bounding box coordinates (format: "x1&y1_x2&y2")
    bbox = tuple(map(int, parts[2].replace('&', '_').split('_')))
    
    # Field 4: Four vertices locations (format: "x1&y1_x2&y2_x3&y3_x4&y4")
    vertices = tuple(map(int, parts[3].replace('&', '_').split('_')))
    
    # Field 5: License plate number indices
    provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
    alphabets = list("ABCDEFGHJKLMNPQRSTUVWXYZO")
    ads = list("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789O")
    
    plate_indices = list(map(int, parts[4].split('_')))
    # Construct the license plate number using the indices
    plate_number = provinces[plate_indices[0]] + alphabets[plate_indices[1]] + ''.join(ads[i] for i in plate_indices[2:])
    
    # Field 6 & 7: Brightness and blurriness
    brightness = int(parts[5])
    blurriness = int(parts[6])
    
    return {
        "area_ratio": area_ratio,
        "tilt_degree": tilt_degree,
        "bounding_box": bbox,
        "vertices": vertices,
        "plate_number": plate_number,
        "brightness": brightness,
        "blurriness": blurriness
    }

class CCPDDataset(Dataset):
    """
    A PyTorch Dataset for the CCPD dataset.
    
    Args:
        root_dir (str): Directory containing CCPD images.
        transform (callable, optional): Optional transform to be applied on an image.
        max_images (int, optional): Limit the number of images loaded.
    """
    def __init__(self, root_dir, transform=None, max_images=None):
        self.root_dir = root_dir
        self.transform = transform
        # List only .jpg files
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.endswith('.jpg')
        ]
        # Optionally limit the number of images
        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Get image path and load the image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        # Convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Parse metadata from filename
        metadata = parse_ccpd_filename(os.path.basename(image_path))
        
        sample = {
            "image": image,
            "metadata": metadata,
            "filename": os.path.basename(image_path)
        }
        
        # Apply transformation if provided (e.g., torchvision.transforms)
        if self.transform:
            sample["image"] = self.transform(sample["image"])
            
        return sample

def load_data():
    """
    Loads the CCPD dataset and returns train and validation DataLoaders.
    
    Returns:
        train_loader, val_loader: DataLoaders for training and validation.
    """
    dataset_path = "/content/LPR-Project/dataset/ccpd-preprocess/CCPD2019/ccpd_base"
    # Limit dataset to a maximum of 4000 images (adjust as needed)
    dataset = CCPDDataset(dataset_path, max_images=4000)
    
    # Optionally, split dataset into 80% train and 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    return train_loader, val_loader

# Quick testing of load_data (uncomment the following block to test)
# if __name__ == "__main__":
#     train_loader, val_loader = load_data()
#     print("Train batches:", len(train_loader))
#     print("Validation batches:", len(val_loader))
#     for batch in train_loader:
#         print(batch["filename"])
#         break
