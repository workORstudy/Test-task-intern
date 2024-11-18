import numpy as np
import cv2
import os

def convert_to_grayscale(data_path, save_path):
    """
    Converts RGB image data to grayscale and saves it in a new file.
    Args:
        data_path (str): Path to the input .npy file containing RGB images.
        save_path (str): Path to save the resulting grayscale images as a .npy file.
    """
    # Load the RGB image data from the .npy file
    data = np.load(data_path)
    print(f"Data shape in file {data_path}: {data.shape}, data type: {data.dtype}")

    # List to store grayscale images
    grayscale_data = []

    for i in range(data.shape[0]):  # Iterate through all images in the dataset
        # Each image has the format (H, W, C) with 3 color channels
        image = data[i]  # Shape: (256, 256, 3)

        # Ensure data type is uint8 for OpenCV compatibility
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)  # Normalize to [0, 255] and convert to uint8

        # Convert the RGB image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        grayscale_data.append(gray_image)

    # Convert the list of grayscale images to a NumPy array and reshape to (B, C, H, W)
    # B: Batch size (number of images), C: Channels (1 for grayscale), H: Height, W: Width
    grayscale_data = np.array(grayscale_data).reshape(-1, 1, data.shape[1], data.shape[2])

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the grayscale images to a .npy file
    np.save(save_path, grayscale_data)
    print(f"Grayscale data saved to: {save_path}")

# Call the function for each dataset: train, validation, and test
convert_to_grayscale('./Task2_CVis_Sentinel2/data/processed/train.npy', './Task2_CVis_Sentinel2/data/processed/train_grayscale.npy')
convert_to_grayscale('./Task2_CVis_Sentinel2/data/processed/val.npy', './Task2_CVis_Sentinel2/data/processed/val_grayscale.npy')
convert_to_grayscale('./Task2_CVis_Sentinel2/data/processed/test.npy', './Task2_CVis_Sentinel2/data/processed/test_grayscale.npy')


"""
Function Description:

Clear description of the function's purpose, input arguments, and output.
Explains that the function converts RGB images into grayscale and saves them.

Data Loading:

Mentioned that the input data is loaded as a NumPy array.
Prints the shape and data type of the input data for debugging purposes.

Data Normalization:

Explained the need to normalize and convert image data to uint8 for OpenCV compatibility.
Added a check to ensure the data type of the image is corrected if necessary.

Grayscale Conversion:

Detailed the conversion of each image from RGB to grayscale using OpenCV's cvtColor method.

Reshaping:

Explained reshaping the grayscale data to the required format (Batch, Channels, Height, Width).

Saving Process:

Ensured the directory for saving the processed data exists or is created.
Described saving the processed data back into a .npy file.

Function Calls:

Highlighted that the function is called for train, validation, and test datasets to ensure all subsets are converted to grayscale.
"""