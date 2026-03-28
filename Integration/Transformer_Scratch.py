# Transformer Model (Scratch)
 
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
'''

# =================== Imports ===================
# import os
# import json
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from collections import Counter

# # =================== Tokenizer & Vocabulary ===================
# class SimpleTokenizer:
#     def __init__(self, summaries, min_freq=1):
#         counter = Counter()
#         for summary in summaries:
#             counter.update(summary.lower().split())
#         self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
#         for word, freq in counter.items():
#             if freq >= min_freq:
#                 self.word2idx[word] = len(self.word2idx)
#         self.idx2word = {idx: word for word, idx in self.word2idx.items()}

#     def encode(self, text, max_len=20):
#         tokens = text.lower().split()
#         encoded = [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in tokens]
#         encoded = [self.word2idx["<SOS>"]] + encoded + [self.word2idx["<EOS>"]]
#         if len(encoded) < max_len:
#             encoded += [self.word2idx["<PAD>"]] * (max_len - len(encoded))
#         else:
#             encoded = encoded[:max_len]
#         return encoded

#     def decode(self, tokens):
#         words = []
#         for token in tokens:
#             word = self.idx2word.get(token, "<UNK>")
#             if word == "<EOS>":
#                 break
#             if word != "<PAD>" and word != "<SOS>":
#                 words.append(word)
#         return ' '.join(words)

# # =================== Dataset ===================
# class CrimeSummaryDataset(Dataset):
#     def __init__(self, data_dir, summary_json, tokenizer, max_len=20):
#         self.data_dir = data_dir
#         self.max_len = max_len
#         self.tokenizer = tokenizer

#         with open(summary_json, 'r') as f:
#             self.annotations = json.load(f)

#         self.samples = []
#         for ann in self.annotations:
#             class_folder = ann["class"]
#             video_id = ann["video_id"]
#             summary = ann["summary"]
#             frames_path = os.path.join(self.data_dir, class_folder, video_id)
#             self.samples.append((frames_path, summary))

#         self.transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((128, 128)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5, 0.5, 0.5],
#                                   std=[0.5, 0.5, 0.5])
#         ])

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         frames_path, summary = self.samples[idx]
#         frames = sorted(os.listdir(frames_path))[:64]
#         images = []
#         for frame in frames:
#             img_path = os.path.join(frames_path, frame)
#             img = cv2.imread(img_path)
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = self.transform(img)
#             images.append(img)
#         images = torch.stack(images)  # (64, 3, 128, 128)
#         images = images.permute(1, 0, 2, 3)  # (3, 64, 128, 128)
#         encoded_summary = self.tokenizer.encode(summary, max_len=self.max_len)
#         return images, torch.tensor(encoded_summary)

# # =================== 3D CNN Encoder ===================
# class VideoEncoder(nn.Module):
#     def __init__(self, embed_dim):
#         super(VideoEncoder, self).__init__()
#         self.conv1 = nn.Conv3d(3, 32, kernel_size=3, padding=1)
#         self.pool1 = nn.MaxPool3d(2)
#         self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
#         self.pool2 = nn.MaxPool3d(2)
#         self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
#         self.pool3 = nn.AdaptiveAvgPool3d(1)
#         self.fc = nn.Linear(128, embed_dim)

#     def forward(self, x):
#         x = F.relu(self.pool1(self.conv1(x)))
#         x = F.relu(self.pool2(self.conv2(x)))
#         x = F.relu(self.pool3(self.conv3(x)))
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x  # (B, embed_dim)

# # =================== Full Transformer Model ===================
# class VideoSummaryTransformer(nn.Module):
#     def __init__(self, embed_dim, vocab_size, num_heads, hidden_dim, num_layers):
#         super(VideoSummaryTransformer, self).__init__()
#         self.encoder = VideoEncoder(embed_dim)
#         self.embed = nn.Embedding(vocab_size, embed_dim)
#         self.pos_encoder = nn.Parameter(torch.randn(1, 20, embed_dim))
#         decoder_layer = nn.TransformerDecoderLayer(
#             d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True
#         )
#         self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
#         self.fc_out = nn.Linear(embed_dim, vocab_size)

#     def forward(self, video, captions):
#         batch_size = video.shape[0]
#         video_embed = self.encoder(video)
#         video_embed = video_embed.unsqueeze(1)

#         captions_embed = self.embed(captions) + self.pos_encoder[:, :captions.size(1), :]
#         tgt_mask = nn.Transformer.generate_square_subsequent_mask(captions.size(1)).to(video.device)

#         output = self.transformer_decoder(
#             tgt=captions_embed,
#             memory=video_embed,
#             tgt_mask=tgt_mask
#         )
#         return self.fc_out(output)

# # =================== Accuracy Function ===================
# def calculate_accuracy(preds, targets, pad_idx):
#     preds = preds.argmax(-1)
#     correct = (preds == targets).float()
#     mask = (targets != pad_idx).float()
#     acc = (correct * mask).sum() / mask.sum()
#     return acc.item() * 100

# # =================== Training and Validation ===================
# def train(model, loader, optimizer, criterion, vocab):
#     model.train()
#     total_loss, total_acc = 0, 0

#     for images, captions in loader:
#         images, captions = images.to(device), captions.to(device)
#         outputs = model(images, captions[:, :-1])
#         loss = criterion(outputs.reshape(-1, outputs.shape[-1]), captions[:, 1:].reshape(-1))

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         total_acc += calculate_accuracy(outputs, captions[:, 1:], vocab["<PAD>"])

#     return total_loss / len(loader), total_acc / len(loader)

# def validate(model, loader, criterion, vocab):
#     model.eval()
#     total_loss, total_acc = 0, 0

#     with torch.no_grad():
#         for images, captions in loader:
#             images, captions = images.to(device), captions.to(device)
#             outputs = model(images, captions[:, :-1])
#             loss = criterion(outputs.reshape(-1, outputs.shape[-1]), captions[:, 1:].reshape(-1))

#             total_loss += loss.item()
#             total_acc += calculate_accuracy(outputs, captions[:, 1:], vocab["<PAD>"])

#     return total_loss / len(loader), total_acc / len(loader)

# # =================== Inference Function ===================
# def generate_summary(model, video, tokenizer, max_len=20):
#     model.eval()
#     with torch.no_grad():
#         video = video.to(device)
#         video_embed = model.encoder(video).unsqueeze(1)

#         output_indices = [tokenizer.word2idx["<SOS>"]]
#         for _ in range(max_len):
#             tgt = torch.tensor(output_indices).unsqueeze(0).to(device)
#             tgt_embed = model.embed(tgt) + model.pos_encoder[:, :tgt.size(1), :]
#             tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(device)
#             output = model.transformer_decoder(tgt_embed, video_embed, tgt_mask=tgt_mask)
#             next_token_logits = model.fc_out(output[:, -1, :])
#             next_token = next_token_logits.argmax(-1).item()

#             if next_token == tokenizer.word2idx["<EOS>"]:
#                 break
#             output_indices.append(next_token)

#         return tokenizer.decode(output_indices[1:])

# # =================== Main ===================
# import cv2  # put this here so everything works

# # Set paths
# train_dir = r"C:\Users\lala deepika\Desktop\final_dataset\Datasets\Split Dataset\train"
# val_dir = r"C:\Users\lala deepika\Desktop\final_dataset\Datasets\Split Dataset\train\val"
# test_dir = r"C:\Users\lala deepika\Desktop\final_dataset\Datasets\Split Dataset\test"

# train_json = r"C:\Users\lala deepika\Desktop\final_dataset\Datasets\Split Dataset\train_summaries.json"
# val_json = r"C:\Users\lala deepika\Desktop\final_dataset\Datasets\Split Dataset\val_summaries.json"
# test_json = r"C:\Users\lala deepika\Desktop\final_dataset\Datasets\Split Dataset\test_summaries.json"

# # Build tokenizer from train summaries
# with open(train_json, 'r') as f:
#     summaries = [ann['summary'] for ann in json.load(f)]
# tokenizer = SimpleTokenizer(summaries)
# vocab = tokenizer.word2idx

# # Create datasets & dataloaders
# train_dataset = CrimeSummaryDataset(train_dir, train_json, tokenizer)
# val_dataset = CrimeSummaryDataset(val_dir, val_json, tokenizer)
# test_dataset = CrimeSummaryDataset(test_dir, test_json, tokenizer)

# train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# # Model setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = VideoSummaryTransformer(embed_dim=256, vocab_size=len(vocab), num_heads=4, hidden_dim=512, num_layers=2).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# criterion = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])

# # Training loop
# best_val_acc = 0
# epochs = 30

# for epoch in range(epochs):
#     print(f"\nEpoch {epoch+1}/{epochs}")
#     train_loss, train_acc = train(model, train_loader, optimizer, criterion, vocab)
#     val_loss, val_acc = validate(model, val_loader, criterion, vocab)
#     print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         torch.save(model.state_dict(), "best_transformer.pth")

# # =================== Testing ===================
# print("\nLoading best model and evaluating on test set...")
# model.load_state_dict(torch.load("best_transformer.pth"))
# test_loss, test_acc = validate(model, test_loader, criterion, vocab)
# print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")

# # =================== Sample Generation ===================
# print("\n=== Sample Test Summaries ===")
# for images, _ in test_loader:
#     summary = generate_summary(model, images, tokenizer)
#     print("Generated Summary:", summary)
#     break  # remove break to generate for full test set

import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoEncoder(nn.Module):
    def __init__(self, embed_dim):
        super(VideoEncoder, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(2)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class VideoSummaryTransformer(nn.Module):
    def __init__(self, embed_dim, vocab_size, num_heads, hidden_dim, num_layers):
        super(VideoSummaryTransformer, self).__init__()
        self.encoder = VideoEncoder(embed_dim)
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, 20, embed_dim))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, video, captions):
        video_embed = self.encoder(video).unsqueeze(1)
        captions_embed = self.embed(captions) + self.pos_encoder[:, :captions.size(1), :]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(captions.size(1)).to(video.device)
        output = self.transformer_decoder(captions_embed, video_embed, tgt_mask=tgt_mask)
        return self.fc_out(output)

def generate_summary(model, video, tokenizer, max_len=20):
    model.eval()
    with torch.no_grad():
        video = video.to(next(model.parameters()).device)
        video_embed = model.encoder(video).unsqueeze(1)

        output_indices = [tokenizer.word2idx["<SOS>"]]
        for _ in range(max_len):
            tgt = torch.tensor(output_indices).unsqueeze(0).to(video.device)
            tgt_embed = model.embed(tgt) + model.pos_encoder[:, :tgt.size(1), :]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(video.device)
            output = model.transformer_decoder(tgt_embed, video_embed, tgt_mask=tgt_mask)
            next_token = model.fc_out(output[:, -1, :]).argmax(-1).item()
            if next_token == tokenizer.word2idx["<EOS>"]:
                break
            output_indices.append(next_token)

        return tokenizer.decode(output_indices[1:])

class SimpleTokenizer:
    def __init__(self, summaries, min_freq=1):
        from collections import Counter
        counter = Counter()
        for summary in summaries:
            counter.update(summary.lower().split())
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        for word, freq in counter.items():
            if freq >= min_freq:
                self.word2idx[word] = len(self.word2idx)
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def encode(self, text, max_len=20):
        tokens = text.lower().split()
        encoded = [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in tokens]
        encoded = [self.word2idx["<SOS>"]] + encoded + [self.word2idx["<EOS>"]]
        if len(encoded) < max_len:
            encoded += [self.word2idx["<PAD>"]] * (max_len - len(encoded))
        else:
            encoded = encoded[:max_len]
        return encoded

    def decode(self, tokens):
        words = []
        for token in tokens:
            word = self.idx2word.get(token, "<UNK>")
            if word == "<EOS>":
                break
            if word not in ["<PAD>", "<SOS>"]:
                words.append(word)
        return ' '.join(words)
