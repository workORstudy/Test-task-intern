# data_preprocessing.py
# Purpose: Preprocessing Sentinel-2 satellite images from a Kaggle dataset, manually extracted

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Paths
RAW_DATA_PATH = r"C:\Users\work_\Downloads\archive"  # Folder with extracted raw files
# RAW_DATA_PATH = './data/raw/'  # Alternative relative path for raw data
PROCESSED_IMAGES_PATH = './Task2_CVis_Sentinel2/data/processed/images/'  # Folder for processed PNG images
PROCESSED_DATA_PATH = './Task2_CVis_Sentinel2/data/processed/'  # Folder for saving .npy files

def find_tci_images(data_path):
    """
    Recursively searches for all files containing 'TCI' in the name within subdirectories.
    TCI files are the RGB visualization images in Sentinel-2 datasets.
    """
    tci_files = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if "TCI" in file and file.endswith('.jp2'):  # JP2 is the format used by Sentinel-2
                tci_files.append(os.path.join(root, file))
    return tci_files

def preprocess_image(image, target_size=(256, 256)):
    """
    Resizes and normalizes an image.
    Args:
        image (ndarray): Input image to be processed.
        target_size (tuple): Desired output size of the image (width, height).
    Returns:
        ndarray: Processed and normalized image.
    """
    resized_img = cv2.resize(image, target_size)  # Resize the image to the target size
    normalized_img = resized_img / 255.0  # Normalize pixel values to the range [0, 1]
    return normalized_img

def convert_jp2_to_png(jp2_file, output_file):
    """
    Converts a JP2 image to PNG format (OpenCV supports JP2 via OpenJPEG).
    Args:
        jp2_file (str): Path to the JP2 file.
        output_file (str): Path to save the converted PNG file.
    Returns:
        ndarray: Loaded image as a NumPy array.
    """
    image = cv2.imread(jp2_file, cv2.IMREAD_UNCHANGED)  # Load JP2 file
    if image is None:
        raise ValueError(f"Failed to load file: {jp2_file}")
    cv2.imwrite(output_file, image)  # Save the image in PNG format
    return image

def prepare_dataset(data_path, processed_images_path, target_size=(256, 256)):
    """
    Loads TCI images, processes them, and saves them as PNG files.
    Args:
        data_path (str): Path to the raw data.
        processed_images_path (str): Path to save processed PNG images.
        target_size (tuple): Desired size for resizing images.
    """
    tci_files = find_tci_images(data_path)  # Find all TCI files
    images = []
    os.makedirs(processed_images_path, exist_ok=True)  # Ensure the output directory exists

    for jp2_file in tci_files:
        try:
            print(f"Processing file: {jp2_file}")
            # Convert JP2 to PNG with a fixed name
            base_name = os.path.basename(jp2_file).replace('.jp2', '.png')  # Change extension
            png_path = os.path.join(processed_images_path, base_name)
            
            # Convert JP2 → PNG
            image = convert_jp2_to_png(jp2_file, png_path)
            
            # Preprocess the image for the model
            processed_img = preprocess_image(image, target_size)
            images.append(processed_img)  # Add processed image to the list
        except Exception as e:
            print(f"Skipped file {jp2_file} due to error: {e}")

    images = np.array(images)  # Convert the list of images to a NumPy array
    np.save(os.path.join(PROCESSED_DATA_PATH, 'all_images.npy'), images)  # Save as .npy file
    print(f"Processed data saved to file: {os.path.join(PROCESSED_DATA_PATH, 'all_images.npy')}")

def split_and_save_data(data_path, processed_path):
    """
    Splits data into train/validation/test sets and saves them as .npy files.
    Args:
        data_path (str): Path to the processed .npy file with all images.
        processed_path (str): Directory to save train, validation, and test datasets.
    """
    images = np.load(data_path)  # Load processed images from .npy file
    train, test = train_test_split(images, test_size=0.2, random_state=42)  # Split into train and test
    train, val = train_test_split(train, test_size=0.25, random_state=42)  # Further split train into train/val

    # Ensure the output directory exists
    os.makedirs(processed_path, exist_ok=True)

    # Save the splits to .npy files
    np.save(os.path.join(processed_path, 'train.npy'), train)
    np.save(os.path.join(processed_path, 'val.npy'), val)
    np.save(os.path.join(processed_path, 'test.npy'), test)

    print("Data split into train, val, and test.")

if __name__ == "__main__":
    # Check if the raw data directory exists
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Directory {RAW_DATA_PATH} not found. Please ensure the data is extracted.")

    # Process images
    os.makedirs(PROCESSED_IMAGES_PATH, exist_ok=True)
    prepare_dataset(RAW_DATA_PATH, PROCESSED_IMAGES_PATH)

    # Split data into train/val/test
    split_and_save_data(
        os.path.join(PROCESSED_DATA_PATH, 'all_images.npy'),
        PROCESSED_DATA_PATH
    )
