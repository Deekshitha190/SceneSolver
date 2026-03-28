import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torchvision.models.video import r3d_18, R3D_18_Weights
import torch.nn as nn

# ========== Configuration ==========
scenario_path = "/content/split_dataset/Split Dataset Shortened/test/Fighting/video_09"  # 👈 Change this
pred_class = "Fighting"  # 👈 Manually specify class
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Preprocessing ==========
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ========== Load Models ==========
# Load ResNet-3D encoder (no classification head)
resnet = r3d_18(weights=R3D_18_Weights.DEFAULT)
resnet.fc = nn.Identity()
resnet.eval().to(device)

# Load projector
class VisualFeatureProjector(nn.Module):
    def __init__(self, input_dim=512, output_dim=512):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.linear(x)

projector = VisualFeatureProjector()
projector.load_state_dict(torch.load("VisualFeatureProjector_Epoch50.pth", map_location=device))
projector.eval().to(device)

# Load T5 model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
t5_model.load_state_dict(torch.load("T5_Decoder_Epoch_50.pth", map_location=device))
t5_model.eval().to(device)

# ========== Load Frames ==========
frames = sorted([f for f in os.listdir(scenario_path) if f.endswith('.jpg')])
interval = len(frames) // 64
indices = [min(i * interval, len(frames)-1) for i in range(64)]

clip = []
for i in indices:
    img = Image.open(os.path.join(scenario_path, frames[i])).convert("RGB")
    img = transform(np.array(img))
    clip.append(img)

clip = torch.stack(clip, dim=1).unsqueeze(0).to(device)  # (1, 3, T, H, W)

# ========== Generate Summary ==========
input_text = f"class: {pred_class} summarize:"
input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=64).input_ids.to(device)

with torch.no_grad():
    visual_feats = resnet(clip)                    # (1, 512)
    enc_feats = projector(visual_feats).unsqueeze(1)  # (1, 1, 512)
    generated_ids = t5_model.generate(input_ids=input_ids, encoder_outputs=(enc_feats,), max_length=64)
    summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# ========== Output ==========
print(f"📂 Scenario: {scenario_path}")
print(f"📝 Generated Summary:\n{summary}")