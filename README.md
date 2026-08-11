# Scene Solver – AI Crime Scene Analysis

Scene Solver is an AI-powered crime scene analysis system that analyzes crime-related videos, classifies the detected activity, and generates a concise summary of the scene.

The project combines **3D CNN-based video classification** with **T5 Transformer-based text summarization** and integrates the models into a Flask web application.

## Key Features

- 🎥 Upload and analyze crime-related videos
- 🧠 Classify videos into:
  - Fighting
  - Robbery
  - Explosion
  - Shoplifting
  - Non-Crime
- 📝 Generate automated scene summaries using T5
- 🔍 Extract and preprocess 64 video frames
- 🌐 Flask-based web application
- 🔐 User signup and login
- 🗄️ MongoDB-based user authentication

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Video Classification | 3D ResNet-18 / R3D-18 |
| Text Summarization | T5 Transformer |
| Computer Vision | OpenCV |
| Web Framework | Flask |
| Database | MongoDB |
| Frontend | HTML, CSS |
| Libraries | TorchVision, Transformers |

## System Workflow

```text
Input Video
     ↓
Frame Extraction (64 Frames)
     ↓
Frame Preprocessing
     ↓
3D ResNet-18
     ↓
Crime Classification
     ↓
T5 Transformer
     ↓
Scene Summary
     ↓
Flask Web Application

## Project Structure
SceneSolver/
│
├── 3D ResNet Model/
│   ├── performance_metrics.py
│   ├── test.py
│   └── training.py
│
├── T5 Model/
│   ├── performance_metrics.py
│   ├── test.py
│   └── training.py
│
├── Integration/
│   ├── 3D_ResNet.py
│   ├── Transformer_Scratch.py
│   ├── Transformer_Summary.py
│   ├── app.py
│   ├── helper.py
│   ├── Tokenizer_Vocab.json
│   ├── home.html
│   ├── index.html
│   ├── login.html
│   ├── result.html
│   ├── signup.html
│   └── CSS files
│
└── .gitignore
