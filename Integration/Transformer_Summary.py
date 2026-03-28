import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torchvision.models.video import r3d_18, R3D_18_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Visual feature projector
class VisualFeatureProjector(nn.Module):
    def __init__(self, input_dim=512, output_dim=512):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# Load all components once
tokenizer = T5Tokenizer.from_pretrained("t5-small")

resnet = r3d_18(weights=R3D_18_Weights.DEFAULT)
resnet.fc = nn.Identity()
resnet.to(device)
resnet.eval()

projector = VisualFeatureProjector().to(device)
projector.load_state_dict(torch.load("VisualFeatureProjector_Epoch50.pth", map_location=device))
projector.eval()

t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
t5_model.load_state_dict(torch.load("T5_Decoder_Epoch_50.pth", map_location=device))
t5_model.eval()

def generate_summary(frames_tensor, predicted_class):
    """
    frames_tensor: (1, 3, 64, 128, 128)
    predicted_class: class name string
    """
    with torch.no_grad():
        feats = resnet(frames_tensor.to(device))
        enc_feats = projector(feats).unsqueeze(1)

        input_text = f"class: {predicted_class} summarize:"
        input_ids = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=64).input_ids.to(device)

        generated_ids = t5_model.generate(input_ids=input_ids, encoder_outputs=(enc_feats,), max_length=64)
        summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return summary

