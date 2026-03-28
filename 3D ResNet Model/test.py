import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.video import r3d_18, R3D_18_Weights

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define class mappings
class_to_idx = {'Fighting': 0, 'Robbery': 1, 'Explosion': 2, 'Shoplifting': 3, 'Non-Crime': 4}
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Load the model architecture
model = r3d_18(weights=R3D_18_Weights.DEFAULT)
for param in model.stem.parameters():
    param.requires_grad = False
for param in model.layer1.parameters():
    param.requires_grad = False
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.6),
    torch.nn.Linear(model.fc.in_features, len(class_to_idx))
)
model = model.to(device)

# Updated transform: Resize to 128x128 directly (no CenterCrop)
base_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),  # Direct resize
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Sharpening filter
def sharpen(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# Inference function for a single video folder
def predict_single_sequence(sequence_path):
    frame_files = sorted([f for f in os.listdir(sequence_path) if f.endswith('.jpg')])
    total_frames = len(frame_files)

    frames_per_clip = 64
    interval = total_frames // frames_per_clip
    sampled_indices = [i * interval for i in range(frames_per_clip)]
    sampled_indices = [min(i, total_frames - 1) for i in sampled_indices]

    frames = []
    for i in sampled_indices:
        img_path = os.path.join(sequence_path, frame_files[i])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = sharpen(img)
        img = base_transforms(img)
        frames.append(img)

    clip = torch.stack(frames, dim=1)  # Shape: (3, T, H, W)
    clip = clip.unsqueeze(0).to(device)  # Add batch dimension

    model.eval()
    with torch.no_grad():
        output = model(clip)
        _, predicted = torch.max(output, 1)
        pred_class_name = idx_to_class[predicted.item()]
        print(f"Predicted Class: {pred_class_name}")
        return pred_class_name

# -------------------------------
# Example usage on one test sample

input_sequence = '/content/dataset/Split Dataset/test/Explosion/video_25'

# Load and predict using model from Epoch 19
model.load_state_dict(torch.load('3D_ResNet_Epoch_19.pth', map_location=device))
print("🔍 Prediction using model from Epoch 19:")
predict_single_sequence(input_sequence)

# Load and predict using model from Epoch 20
model.load_state_dict(torch.load('3D_ResNet_Epoch_20.pth', map_location=device))
print("🔍 Prediction using model from Epoch 20:")
predict_single_sequence(input_sequence)