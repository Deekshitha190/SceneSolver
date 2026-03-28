'''
from google.colab import drive
drive.mount('/content/drive')

import zipfile

# Update with your actual path if different
zip_path = '/content/drive/MyDrive/Split Dataset.zip'
extract_path = '/content/dataset/'  # You can change this if you like

# Unzip
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Files extracted to:", extract_path)
'''

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models.video import r3d_18, R3D_18_Weights
from sklearn.utils.class_weight import compute_class_weight
import random

# Reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Custom Dataset
class CrimeVideoDataset(Dataset):
    def __init__(self, root_dir, class_to_idx, frames_per_clip=64, is_train=True):
        self.root_dir = root_dir
        self.class_to_idx = class_to_idx
        self.frames_per_clip = frames_per_clip
        self.samples = []
        self.is_train = is_train

        for class_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            for scenario_name in sorted(os.listdir(class_path)):
                scenario_path = os.path.join(class_path, scenario_name)
                frame_files = sorted([f for f in os.listdir(scenario_path) if f.endswith('.jpg')])
                if len(frame_files) >= self.frames_per_clip:
                    self.samples.append((scenario_path, class_name))

        base_transforms = [
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
        ]

        if self.is_train:
            base_transforms += [transforms.RandomHorizontalFlip(p=0.5)]

        base_transforms += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]

        self.transform = transforms.Compose(base_transforms)

    def sharpen(self, image):
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scenario_path, class_name = self.samples[idx]
        frame_files = sorted([f for f in os.listdir(scenario_path) if f.endswith('.jpg')])
        total_frames = len(frame_files)

        interval = total_frames // self.frames_per_clip
        sampled_indices = [i * interval for i in range(self.frames_per_clip)]
        sampled_indices = [min(i, total_frames - 1) for i in sampled_indices]

        frames = []
        for i in sampled_indices:
            img_path = os.path.join(scenario_path, frame_files[i])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.sharpen(img)
            img = self.transform(img)
            frames.append(img)

        clip = torch.stack(frames, dim=1)  # Shape: (3, T, H, W)
        label = self.class_to_idx[class_name]
        return clip, label

# Paths and class mapping
train_path = '/content/dataset/Split Dataset/train'
val_path = '/content/dataset/Split Dataset/val'
test_path = '/content/dataset/Split Dataset/test'
class_to_idx = {'Fighting': 0, 'Robbery': 1, 'Explosion': 2, 'Shoplifting': 3, 'Non-Crime': 4}

# Dataset & DataLoader
train_dataset = CrimeVideoDataset(train_path, class_to_idx, is_train=True)
val_dataset = CrimeVideoDataset(val_path, class_to_idx, is_train=False)
test_dataset = CrimeVideoDataset(test_path, class_to_idx, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

# Compute class weights
train_labels = [class_to_idx[class_name] for _, class_name in train_dataset.samples]
train_labels = np.array(train_labels)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
weights_tensor = torch.tensor(class_weights, dtype=torch.float)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_tensor = weights_tensor.to(device)

# Model setup
model = r3d_18(weights=R3D_18_Weights.DEFAULT)
for param in model.stem.parameters():
    param.requires_grad = False
for param in model.layer1.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Dropout(0.6),
    nn.Linear(model.fc.in_features, 5)
)
model = model.to(device)

# Loss, Optimizer, Scheduler
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                 patience=2, factor=0.5)

# Training Loop
epochs = 20
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
best_val_acc = 0.0

for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for clips, labels in train_loader:
        clips, labels = clips.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for clips, labels in val_loader:
            clips, labels = clips.to(device), labels.to(device)
            outputs = model(clips)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    scheduler.step(val_loss)

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% "
          f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
    print(f"Current Learning Rate: {scheduler.optimizer.param_groups[0]['lr']:.6f}")
    print()

    '''
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), '3D_ResNet.pth')
        print("✅ Best model saved.")
    '''
    # Save at Epoch 19 (index 18)
    if epoch == 18:
        torch.save(model.state_dict(), '3D_ResNet_Epoch_19.pth')
        print("📁 Model saved at Epoch 19.")

    # Save at Epoch 20 (index 19)
    if epoch == 19:
        torch.save(model.state_dict(), '3D_ResNet_Epoch_20.pth')
        print("📁 Final model saved at Epoch 20.")
