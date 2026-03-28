import cv2
import torch
import os
import numpy as np
from torchvision import transforms

# --- Constants ---
NUM_FRAMES = 64
IMG_SIZE = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Frame extractor ---
def extract_frames(video_path, num_frames=NUM_FRAMES):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < num_frames:
        raise ValueError("Video too short")

    indices = np.linspace(0, total-1, num_frames).astype(int)
    frames = []

    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    return frames  # list of np arrays

# --- Frame preprocess ---
def preprocess_frames_for_resnet(frames):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    tensors = [transform(frame) for frame in frames]
    clip = torch.stack(tensors, dim=1)  # shape: (3, T, H, W)
    return clip.unsqueeze(0).to(device)  # shape: (1, 3, T, H, W)

def preprocess_frames_for_transformer(frames):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    tensors = [transform(frame) for frame in frames]
    clip = torch.stack(tensors, dim=0)  # (T, 3, H, W)
    clip = clip.permute(1, 0, 2, 3)     # (3, T, H, W)
    return clip.unsqueeze(0).to(device)  # (1, 3, T, H, W)
