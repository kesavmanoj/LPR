import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split

def parse_ccpd_filename(filename):
    """
    Parses a CCPD image filename and extracts metadata.
    
    Expected filename format (example):
    "0166283524904-89_89-195,493_428,576-434,568_196,572_195,501_433,497-0_0_26_12_24_33_31-108-36.jpg"
    
    Fields:
      1. Area: Area ratio of license plate area to the entire picture area.
      2. Tilt degree: Horizontal and vertical tilt degrees (separated by '_').
      3. Bounding box coordinates: Coordinates of left-up and right-bottom vertices, e.g., "195,493_428,576"
      4. Four vertices locations: (x,y) pairs for four vertices (starting from the right-bottom), e.g., "434,568_196,572_195,501_433,497"
      5. License plate number: Indices for province, alphabet, and alphanumeric (e.g., "0_0_26_12_24_33_31")
      6. Brightness: Brightness of the license plate region.
      7. Blurriness: Blurriness of the license plate region.
    """
    # Remove the file extension.
    base = os.path.splitext(filename)[0]
    fields = base.split('-')
    if len(fields) != 7:
        raise ValueError(f"Expected 7 fields in filename, got {len(fields)}: {filename}")
    
    # Field 1: Area (convert to float and divide by 100)
    area_str = fields[0]
    area_ratio = float(area_str) / 100.0
    
    # Field 2: Tilt degree (expects two numbers separated by '_')
    tilt_parts = fields[1].split('_')
    if len(tilt_parts) != 2:
        raise ValueError(f"Expected 2 tilt values, got {len(tilt_parts)}: {fields[1]}")
    tilt_degree = tuple(map(int, tilt_parts))
    
    # Field 3: Bounding box coordinates
    # Replace commas with underscores so that "195,493_428,576" becomes "195_493_428_576"
    bbox_str = fields[2].replace(',', '_')
    bbox_parts = bbox_str.split('_')
    if len(bbox_parts) != 4:
        raise ValueError(f"Expected 4 bounding box values, got {len(bbox_parts)}: {bbox_str}")
    bbox = tuple(map(int, bbox_parts))
    
    # Field 4: Four vertices locations
    vertices_str = fields[3].replace(',', '_')
    vertices_parts = vertices_str.split('_')
    if len(vertices_parts) != 8:
        raise ValueError(f"Expected 8 vertex values, got {len(vertices_parts)}: {vertices_str}")
    vertices = tuple(map(int, vertices_parts))
    
    # Field 5: License plate number indices
    plate_indices = list(map(int, fields[4].split('_')))
    if len(plate_indices) != 7:
        raise ValueError(f"Expected 7 indices for license plate, got {len(plate_indices)}: {fields[4]}")
    provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
    alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
    ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
    plate_number = provinces[plate_indices[0]] + alphabets[plate_indices[1]] + ''.join(ads[i] for i in plate_indices[2:])
    
    # Field 6: Brightness
    brightness = int(fields[5])
    
    # Field 7: Blurriness
    blurriness = int(fields[6])
    
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
    PyTorch Dataset for CCPD images.
    
    Args:
        root_dir (str): Directory containing the CCPD images.
        transform (callable, optional): Transformations to apply to images.
        max_images (int, optional): Limit number of images for debugging.
    """
    def __init__(self, root_dir, transform=None, max_images=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.endswith('.jpg')
        ]
        # Optionally limit the number of images loaded
        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # Read image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        # Convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Parse metadata from the filename
        metadata = parse_ccpd_filename(os.path.basename(image_path))
        
        sample = {
            "image": image,
            "metadata": metadata,
            "filename": os.path.basename(image_path)
        }
        
        if self.transform:
            sample["image"] = self.transform(sample["image"])
            
        return sample

def custom_collate_fn(batch):
    """
    Custom collate function that keeps each sample as a dictionary.
    The DataLoader will return a list of sample dictionaries.
    """
    return batch

def load_data():
    """
    Loads the CCPD dataset and returns DataLoaders for training and validation.
    """
    dataset_path = "/content/LPR-Project/dataset/ccpd-preprocess/CCPD2019/ccpd_base"
    dataset = CCPDDataset(dataset_path, max_images=4000)
    
    # Split the dataset: 80% train, 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)
    
    return train_loader, val_loader

# Quick testing of load_data and parsing on a single batch
if __name__ == "__main__":
    train_loader, val_loader = load_data()
    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    for batch in train_loader:
        # Print filenames
        print("Filenames in batch:", [sample["filename"] for sample in batch])
        # Print metadata for the first sample in the batch
        print("Metadata for first sample:")
        for key, value in batch[0]["metadata"].items():
            print(f"  {key}: {value}")
        break
