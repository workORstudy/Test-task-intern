import torch
import numpy as np
import cv2
import sys
import os

# Add the root directory of the project to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../'))
sys.path.append(project_root)

from models.superpoint import load_model

# Paths
MODEL_WEIGHTS_PATH = './Task2_CVis_Sentinel2/models/superpoint_v1.pth'  # Path to the pre-trained model weights
PROCESSED_DATA_PATH = './Task2_CVis_Sentinel2/data/processed/'  # Path to the directory containing preprocessed data
OUTPUT_PATH = './Task2_CVis_Sentinel2/results/'  # Path to save the results (keypoints and descriptors)

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def detect_keypoints_and_descriptors(model, image):
    """
    Detects keypoints and descriptors in an image using the SuperPoint model.
    Args:
        model (nn.Module): The loaded SuperPoint model.
        image (ndarray): Input grayscale image of shape (H, W).
    Returns:
        keypoints (ndarray): Array of detected keypoints in the format (x, y).
        descriptors (ndarray): Array of corresponding descriptors with shape (N, D).
    """
    # Normalize the image to the range [0, 1]
    image = image / 255.0
    # Convert the image to a 4D tensor and move it to the appropriate device
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Inference: Obtain keypoints and descriptors
    with torch.no_grad():
        keypoints, descriptors = model(image_tensor)
    
    # Convert keypoints and descriptors from tensors to NumPy arrays
    keypoints = keypoints.squeeze().cpu().numpy()  # Remove extra dimensions
    descriptors = descriptors.squeeze().cpu().numpy()

    return keypoints, descriptors

def process_images(model, input_file, output_path):
    """
    Processes images stored in a .npy file and saves their keypoints and descriptors.
    Args:
        model (nn.Module): The loaded SuperPoint model.
        input_file (str): Path to the .npy file containing grayscale images.
        output_path (str): Path to save the results.
    """
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Load the .npy file containing multiple images
    images = np.load(input_file)
    for i, image in enumerate(images):
        # Detect keypoints and descriptors for the current image
        keypoints, descriptors = detect_keypoints_and_descriptors(model, image[0])  # Assuming (B, C, H, W) format

        # Save the keypoints and descriptors for the current image
        np.save(os.path.join(output_path, f'image_{i}_keypoints.npy'), keypoints)
        np.save(os.path.join(output_path, f'image_{i}_descriptors.npy'), descriptors)
        print(f"Processed image {i}")

if __name__ == "__main__":
    print("Loading SuperPoint model...")
    # Load the pre-trained SuperPoint model
    model = load_model(MODEL_WEIGHTS_PATH, device=device)
    print("SuperPoint model successfully loaded!")

    print("Processing images...")
    # Process the test images and save results
    process_images(model, os.path.join(PROCESSED_DATA_PATH, 'test_grayscale.npy'), OUTPUT_PATH)
    print("Processing completed!")
