import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

RESULTS_PATH = './Task2_CVis_Sentinel2/results/'

def compute_matches(descriptors1, descriptors2, threshold=0.8):
    """
    Computes the number of matches between descriptors of two images.
    Args:
        descriptors1 (ndarray): Descriptors for the first image, shape (N1, D, ...).
        descriptors2 (ndarray): Descriptors for the second image, shape (N2, D, ...).
        threshold (float): Similarity threshold to consider a match valid.
    Returns:
        int: The number of valid matches.
    """
    # Flatten descriptors into 2D arrays
    descriptors1 = descriptors1.reshape(descriptors1.shape[0], -1)  # Shape: (N1, D)
    descriptors2 = descriptors2.reshape(descriptors2.shape[0], -1)  # Shape: (N2, D)
    
    # Calculate cosine similarity between descriptors
    similarity = cosine_similarity(descriptors1, descriptors2)
    
    # Find the best match for each descriptor in descriptors1
    matches = np.argmax(similarity, axis=1)
    
    # Count valid matches that exceed the similarity threshold
    correct_matches = sum(similarity[i, match] > threshold for i, match in enumerate(matches))
    return correct_matches

def evaluate_results(results_path):
    """
    Evaluates the results by comparing descriptors between all pairs of images.
    Args:
        results_path (str): Path to the folder containing the results 
                            (keypoints and descriptors files).
    Returns:
        dict: A dictionary containing the match counts for each pair of images.
    """
    # Collect all keypoints and descriptors files
    keypoints_files = [f for f in os.listdir(results_path) if f.endswith('_keypoints.npy')]
    descriptors_files = [f for f in os.listdir(results_path) if f.endswith('_descriptors.npy')]
    
    # Sort files to ensure correct pairwise comparison
    keypoints_files.sort()
    descriptors_files.sort()
    
    scores = {}
    for i in range(len(descriptors_files) - 1):
        # Load descriptors for the current and next image
        descriptors1 = np.load(os.path.join(results_path, descriptors_files[i]))
        descriptors2 = np.load(os.path.join(results_path, descriptors_files[i + 1]))
        
        # Compute matches between the descriptors
        matches = compute_matches(descriptors1, descriptors2)
        
        # Create a name for the pair and store the match count
        pair_name = f"{descriptors_files[i]} <-> {descriptors_files[i + 1]}"
        scores[pair_name] = matches
        print(f"Matches between {descriptors_files[i]} and {descriptors_files[i + 1]}: {matches}")

    return scores

if __name__ == "__main__":
    print("Evaluating results...")
    # Run evaluation on all pairs of images in the results directory
    scores = evaluate_results(RESULTS_PATH)
    
    # Print detailed evaluation results
    print("\nEvaluation Results:")
    for pair, matches in scores.items():
        print(f"{pair}: {matches} matches")
    
    # Calculate and display the average number of matches
    avg_matches = np.mean(list(scores.values()))
    print(f"\nAverage number of matches: {avg_matches}")
