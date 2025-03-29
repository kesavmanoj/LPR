import csv
import cv2
import numpy as np
import os

def scale_coordinates(corner_list, scale_x, scale_y):
    """
    Given a list of 8 numbers (for 4 (x,y) pairs),
    scale x coordinates by scale_x and y coordinates by scale_y.
    Returns a list of scaled coordinates.
    """
    scaled = []
    for i, v in enumerate(corner_list):
        if i % 2 == 0:
            scaled.append(float(v) * scale_x)
        else:
            scaled.append(float(v) * scale_y)
    return scaled

def process_csv(csv_filename, output_dir, 
                target_size=(256, 64), original_size=(720, 1160)):
    """
    Reads the CSV file containing filename, image_path, actual and predicted corners.
    Scales up the corner coordinates from target_size to original_size,
    draws two polygons on the original image (actual in red, predicted in blue),
    and saves the resulting image to output_dir.
    
    Args:
      csv_filename (str): Path to the CSV file.
      output_dir (str): Directory to save processed images.
      target_size (tuple): (width, height) that was used for training.
      original_size (tuple): (width, height) of the original images.
    """
    # Calculate scaling factors from target_size to original_size
    scale_x = original_size[0] / target_size[0]  # width scaling
    scale_y = original_size[1] / target_size[1]  # height scaling

    os.makedirs(output_dir, exist_ok=True)

    with open(csv_filename, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            filename = row["filename"]
            image_path = row["image_path"]
            # Parse corner strings (assumes values are comma-separated)
            actual_str = row["actual_corners"]
            predicted_str = row["predicted_corners"]

            # Convert strings to list of floats
            actual_corners = [float(x.strip()) for x in actual_str.split(",")]
            predicted_corners = [float(x.strip()) for x in predicted_str.split(",")]

            # Scale up the coordinates to the original image size
            actual_scaled = scale_coordinates(actual_corners, scale_x, scale_y)
            predicted_scaled = scale_coordinates(predicted_corners, scale_x, scale_y)

            # Reshape the 8 values into (4, 2) arrays for polygon drawing
            actual_pts = np.array(actual_scaled, dtype=np.int32).reshape((4, 2))
            predicted_pts = np.array(predicted_scaled, dtype=np.int32).reshape((4, 2))

            # Load the original image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Image not found: {image_path}")
                continue

            # Draw actual corners in red (BGR: (0, 0, 255)) and predicted corners in blue (BGR: (255, 0, 0))
            cv2.polylines(img, [actual_pts], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.polylines(img, [predicted_pts], isClosed=True, color=(255, 0, 0), thickness=2)

            # Save the processed image
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, img)
            print(f"Processed {filename} -> saved to {save_path}")

if __name__ == "__main__":
    # Update the CSV filename if necessary
    csv_filename = "corner_results.csv"
    output_dir = "scaled_output"
    process_csv(csv_filename, output_dir)
