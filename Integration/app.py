# from flask import Flask, render_template, request, redirect, url_for
from pymongo import MongoClient
import os
import torch
import torchvision.transforms as transforms
import cv2
from werkzeug.utils import secure_filename
from torchvision.models.video import r3d_18, R3D_18_Weights
from Transformer_summary import generate_summary  # Updated import
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- MongoDB Setup ----------------
client = MongoClient("mongodb://localhost:27017/")
db = client["scene_solver"]
users_collection = db["users"]

# ---------------- Device & Model Setup ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3D CNN for classification
cnn_model = r3d_18(weights=R3D_18_Weights.DEFAULT)
cnn_model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.6),
    torch.nn.Linear(cnn_model.fc.in_features, 5)
)
cnn_model.load_state_dict(torch.load("Best_Model_1.pth", map_location=device))
cnn_model.to(device)
cnn_model.eval()

class_to_label = {0: 'Fighting', 1: 'Robbery', 2: 'Explosion', 3: 'Shoplifting', 4: 'Non-Crime'}

# ---------------- Helper Functions ----------------
def extract_frames_for_inference(video_path, num_frames=64):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // num_frames)
    frames = []
    count = 0
    while len(frames) < num_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        count += 1
    cap.release()
    return frames

def preprocess_frames(frames):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((144, 144)),
        transforms.CenterCrop((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    processed = [transform(f) for f in frames]
    clip = torch.stack(processed, dim=1)  # Shape: (3, T, H, W)
    return clip

def predict_crime_and_summary(video_path):
    frames = extract_frames_for_inference(video_path)
    frames_tensor = preprocess_frames(frames).unsqueeze(0).to(device)  # (1, 3, 64, 128, 128)

    with torch.no_grad():
        crime_output = cnn_model(frames_tensor)
        predicted_idx = torch.argmax(crime_output).item()
        predicted_class = class_to_label[predicted_idx]

    summary = ""
    if predicted_class != 'Non-Crime':
        summary = generate_summary(frames_tensor, predicted_class)

    return predicted_class, summary

# ---------------- Routes ----------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'video' not in request.files:
            return "No file uploaded"
        file = request.files['video']
        if file.filename == '':
            return "No selected file"
        filename = secure_filename(file.filename)
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(video_path)

        predicted_class, summary = predict_crime_and_summary(video_path)
        return render_template('result.html', filename=filename, predicted_class=predicted_class, summary=summary)
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if users_collection.find_one({'username': username}):
            return render_template('signup.html', error="User already exists.")
        users_collection.insert_one({'username': username, 'password': password})
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users_collection.find_one({'username': username, 'password': password})
        if user:
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Invalid credentials.")
    return render_template('login.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return os.path.join(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
