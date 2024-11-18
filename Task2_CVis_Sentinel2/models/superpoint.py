import torch
import torch.nn as nn

class SuperPoint(nn.Module):
    """
    Adapted SuperPoint model to match the provided weights.
    """
    def __init__(self):
        super(SuperPoint, self).__init__()
        # Encoder layers
        self.conv1a = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # First convolutional layer
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # Second convolutional layer
        self.conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # Third convolutional layer
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # Fourth convolutional layer
        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Fifth convolutional layer
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)  # Sixth convolutional layer
        self.conv4a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)  # Seventh convolutional layer
        self.conv4b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)  # Eighth convolutional layer

        # Additional layers for the intermediate representation
        self.convDa = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # Expands to 256 channels
        self.convDb = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)  # Refines to 256 channels

        # Keypoints and descriptors layers
        self.convPa = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # Keypoints heatmap generation
        self.convPb = nn.Conv2d(256, 65, kernel_size=1, stride=1, padding=0)  # Descriptors extraction

    def forward(self, x):
        # Encoder pass
        x = torch.relu(self.conv1a(x))  # Apply ReLU after first convolution
        x = torch.relu(self.conv1b(x))  # Apply ReLU after second convolution
        x = nn.MaxPool2d(2, 2)(x)  # Downsample the feature map
        x = torch.relu(self.conv2a(x))  # Apply ReLU after third convolution
        x = torch.relu(self.conv2b(x))  # Apply ReLU after fourth convolution
        x = nn.MaxPool2d(2, 2)(x)  # Downsample again
        x = torch.relu(self.conv3a(x))  # Apply ReLU after fifth convolution
        x = torch.relu(self.conv3b(x))  # Apply ReLU after sixth convolution
        x = nn.MaxPool2d(2, 2)(x)  # Downsample further
        x = torch.relu(self.conv4a(x))  # Apply ReLU after seventh convolution
        x = torch.relu(self.conv4b(x))  # Apply ReLU after eighth convolution
        x = torch.relu(self.convDa(x))  # Expand the feature map
        x = torch.relu(self.convDb(x))  # Refine the feature map

        # Generate keypoints heatmap
        keypoints = self.convPa(x)
        keypoints = torch.softmax(keypoints, dim=1)  # Normalize keypoints across channels

        # Generate descriptors
        descriptors = self.convPb(x)
        return keypoints, descriptors


def adapt_weights(state_dict):
    """
    Adapts weights in the state_dict to match the architecture of the model.
    This handles discrepancies between the original model weights and the current model's architecture.
    """
    adapted_state_dict = {}
    for key, value in state_dict.items():
        if key == "convPa.weight":
            # Expand the input channels of `convPa.weight` from 128 to 256 by duplicating weights
            adapted_state_dict[key] = torch.cat([value, value.clone()], dim=1)
        elif key == "convPa.bias":
            # Bias for `convPa` does not require modification
            adapted_state_dict[key] = value
        else:
            # Keep other weights unchanged
            adapted_state_dict[key] = value
    return adapted_state_dict


def load_model(weights_path, device='cpu'):
    """
    Loads the SuperPoint model with adapted weights.
    Args:
        weights_path (str): Path to the pre-trained model weights.
        device (str): The device to load the model on ('cpu' or 'cuda').
    Returns:
        nn.Module: Loaded and adapted SuperPoint model.
    """
    # Initialize the SuperPoint model
    model = SuperPoint()
    # Load the state_dict from the weights file
    state_dict = torch.load(weights_path, map_location=device)
    
    # Adapt the state_dict to fit the current architecture
    state_dict = adapt_weights(state_dict)
    
    # Load the adapted state_dict into the model
    model.load_state_dict(state_dict)
    model.to(device)  # Move the model to the specified device
    model.eval()  # Set the model to evaluation mode
    return model
