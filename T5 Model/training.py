'''
from google.colab import drive
drive.mount('/content/drive')

import zipfile
import os

# Set up dataset path
zip_path = '/content/drive/MyDrive/Split Dataset.zip'
extract_dir = '/content/split_dataset/'
# Unzip the dataset
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
    
pip install rouge-score bert-score
'''

# ===================== Imports =====================
import os
import json
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import ImageFilter, Image
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics import classification_report, precision_recall_fscore_support
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt

# ===================== Preprocessing Functions =====================
def apply_sharpen(img):
    return img.filter(ImageFilter.Kernel(
        size=(3, 3),
        kernel=(0, -1, 0, -1, 5, -1, 0, -1, 0),
        scale=None
    ))

# ===================== Dataset =====================
class CrimeSummaryDataset(Dataset):
    def __init__(self, video_dir, summary_json, class_to_idx, tokenizer, max_length=64, is_train=True):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.class_to_idx = class_to_idx
        self.is_train = is_train

        with open(summary_json, 'r') as f:
            summaries = json.load(f)

        for entry in summaries:
            video_id = entry['video_id']
            class_name = entry['class']
            summary = entry['summary']
            video_path = os.path.join(video_dir, class_name, video_id)
            if os.path.isdir(video_path):
                self.samples.append((video_path, class_name, summary))

        if self.is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((128, 128)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomRotation(10),
                transforms.Lambda(apply_sharpen),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, class_name, summary = self.samples[idx]
        frames = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg')])
        interval = len(frames) // 64
        indices = [min(i * interval, len(frames)-1) for i in range(64)]

        clip = []
        for i in indices:
            img = Image.open(os.path.join(video_path, frames[i])).convert("RGB")
            img = self.transform(np.array(img))
            clip.append(img)

        clip = torch.stack(clip, dim=1)  # (3, T, H, W)

        input_text = f"class: {class_name} summarize:"
        input_ids = self.tokenizer(input_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt").input_ids.squeeze(0)
        label_ids = self.tokenizer(summary, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt").input_ids.squeeze(0)

        return clip, input_ids, label_ids, summary

# ===================== Visual Feature Projector =====================
class VisualFeatureProjector(nn.Module):
    def __init__(self, input_dim=512, output_dim=512):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# ===================== Setup =====================
from torchvision.models.video import r3d_18, R3D_18_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

resnet = r3d_18(weights=R3D_18_Weights.DEFAULT)
resnet.fc = nn.Identity()
resnet.eval().to(device)

projector = VisualFeatureProjector().to(device)
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

params = list(projector.parameters()) + list(t5_model.parameters())
optimizer = torch.optim.AdamW(params, lr=2e-4)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# ===================== Dataset =====================
class_to_idx = {'Fighting': 0, 'Robbery': 1, 'Explosion': 2, 'Shoplifting': 3, 'Non-Crime': 4}
train_dataset = CrimeSummaryDataset("/content/split_dataset/Split Dataset/train", "/content/split_dataset/Split Dataset/train_summaries.json", class_to_idx, tokenizer, is_train=True)
val_dataset = CrimeSummaryDataset("/content/split_dataset/Split Dataset/val", "/content/split_dataset/Split Dataset/val_summaries.json", class_to_idx, tokenizer, is_train=False)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# ===================== Training =====================
epochs = 50
sum_train_losses, sum_val_losses = [], []
sum_train_accuracies, sum_val_accuracies = [], []
bleu_scores, rouge_scores = [], []

for epoch in range(epochs):
    projector.train()
    t5_model.train()

    running_loss = 0.0
    correct_tokens, total_tokens = 0, 0

    for clips, input_ids, labels, _ in tqdm(train_loader):
        clips, input_ids, labels = clips.to(device), input_ids.to(device), labels.to(device)
        with torch.no_grad():
            feats = resnet(clips)
        enc_feats = projector(feats).unsqueeze(1)

        out = t5_model(input_ids=input_ids, encoder_outputs=(enc_feats,), labels=labels)
        loss = out.loss
        logits = out.logits

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        pred_tokens = torch.argmax(logits, dim=-1)
        mask = labels != tokenizer.pad_token_id
        correct_tokens += ((pred_tokens == labels) & mask).sum().item()
        total_tokens += mask.sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct_tokens / total_tokens
    sum_train_losses.append(train_loss)
    sum_train_accuracies.append(train_acc)

    # ===================== Validation =====================
    projector.eval()
    t5_model.eval()
    val_loss = 0.0
    bleu_total, rouge_total, count = 0.0, 0.0, 0
    correct_tokens, total_tokens = 0, 0

    with torch.no_grad():
        for clips, input_ids, labels, summaries in val_loader:
            clips, input_ids, labels = clips.to(device), input_ids.to(device), labels.to(device)
            feats = resnet(clips)
            enc_feats = projector(feats).unsqueeze(1)

            out = t5_model(input_ids=input_ids, encoder_outputs=(enc_feats,), labels=labels)
            val_loss += out.loss.item()
            logits = out.logits

            pred_tokens = torch.argmax(logits, dim=-1)
            mask = labels != tokenizer.pad_token_id
            correct_tokens += ((pred_tokens == labels) & mask).sum().item()
            total_tokens += mask.sum().item()

            generated_ids = t5_model.generate(input_ids=input_ids, encoder_outputs=(enc_feats,), max_length=64)
            pred_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for pred, gt in zip(pred_texts, summaries):
                bleu_total += sentence_bleu([gt.split()], pred.split(), smoothing_function=SmoothingFunction().method1)
                rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True).score(gt, pred)
                rouge_total += rouge['rougeL'].fmeasure
                count += 1

    val_loss /= len(val_loader)
    val_acc = correct_tokens / total_tokens
    bleu_score = bleu_total / count
    rouge_score = rouge_total / count

    sum_val_losses.append(val_loss)
    sum_val_accuracies.append(val_acc)
    bleu_scores.append(bleu_score)
    rouge_scores.append(rouge_score)

    print(f"\nEpoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
    print(f"BLEU: {bleu_score:.4f} | ROUGE-L: {rouge_score:.4f}")

# ===================== Save Final Models =====================
if epoch == 44:
    torch.save(t5_model.state_dict(), "T5_Decoder_Epoch_45.pth")
    torch.save(projector.state_dict(), "VisualFeatureProjector_Epoch45.pth")
    print("💾 Saved models at epoch 45.")

if epoch == epochs - 1:
    torch.save(t5_model.state_dict(), "T5_Decoder_Epoch_50.pth")
    torch.save(projector.state_dict(), "VisualFeatureProjector_Epoch50.pth")
    print("💾 Saved final models at epoch 50.")
