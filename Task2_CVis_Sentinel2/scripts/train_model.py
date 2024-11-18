import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Додавання шляху до кореневої папки проекту
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../'))
sys.path.append(project_root)

from models.superpoint import load_model

# Шляхи
PROCESSED_DATA_PATH = './Task2_CVis_Sentinel2/data/processed/'
MODEL_WEIGHTS_PATH = './Task2_CVis_Sentinel2/models/superpoint_v1.pth'
CHECKPOINT_PATH = './Task2_CVis_Sentinel2/models/checkpoints/'

# Налаштування пристрою
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Датасет
class SatelliteDataset(Dataset):
    """
    Датасет для ч/б супутникових зображень.
    """
    def __init__(self, data_path):
        self.data = np.load(data_path)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        image = self.data[idx]
        image = image / 255.0  # Нормалізація до [0, 1]
        return torch.tensor(image, dtype=torch.float32)  # (1, H, W)

# Гіперпараметри
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001

# Підготовка даних
train_loader = DataLoader(SatelliteDataset(os.path.join(PROCESSED_DATA_PATH, 'train_grayscale.npy')), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(SatelliteDataset(os.path.join(PROCESSED_DATA_PATH, 'val_grayscale.npy')), batch_size=BATCH_SIZE)

# Завантаження моделі
print("Завантаження моделі SuperPoint...")
model = load_model(MODEL_WEIGHTS_PATH, device=device)
print("Модель SuperPoint успішно завантажена!")

# Визначення функцій втрат
images_criterion = nn.MSELoss()  # Втрати для порівняння зображень
keypoints_criterion = nn.CrossEntropyLoss()  # Втрати для ключових точок (можна замінити залежно від задачі)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_model(model, train_loader, val_loader, epochs, checkpoint_path):
    """
    Функція для тренування моделі.
    """
    os.makedirs(checkpoint_path, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images in train_loader:
            images = images.to(device)
            optimizer.zero_grad()

            # Прямий прохід
            keypoints, descriptors = model(images)

            # Масштабування цільового зображення до розміру виходу моделі
            images_resized = F.interpolate(images, size=keypoints.shape[2:], mode='bilinear', align_corners=False)

            # Втрати
            loss = images_criterion(keypoints, images_resized)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                keypoints, descriptors = model(images)

                # Масштабування цільового зображення
                images_resized = F.interpolate(images, size=keypoints.shape[2:], mode='bilinear', align_corners=False)
                val_loss += images_criterion(keypoints, images_resized).item()

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Збереження чекпойнта
        checkpoint_file = os.path.join(checkpoint_path, f'epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Чекпойнт збережено: {checkpoint_file}")

    # Збереження фінальної моделі
    final_model_path = os.path.join(checkpoint_path, 'superpoint_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Фінальна модель збережена: {final_model_path}")

if __name__ == "__main__":
    train_model(model, train_loader, val_loader, EPOCHS, CHECKPOINT_PATH)
