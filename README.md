# Spot DeepFake: Advanced Deepfake Detection System

## Overview

Spot DeepFake is a professional-grade AI system for detecting AI-generated deepfake images and videos with 95.67% accuracy using EfficientNetB0 transfer learning.

## Features

- Real-time image and video analysis
- 95.67% validation accuracy
- Professional, clean UI (no gradients/emojis)
- EfficientNetB0 architecture with custom classifier
- Free deployment on Streamlit Cloud
- Complete documentation and reproducibility

## Technology Stack

- **Model:** PyTorch EfficientNetB0
- **Framework:** Streamlit
- **Training:** Google Colab (Tesla T4 GPU)
- **Deployment:** Streamlit Cloud
- **Language:** Python 3.9+

## Performance Metrics

| Metric | Value |
|--------|-------|
| Validation Accuracy | 95.67% |
| Precision | 97.00% |
| Recall | 94.25% |
| F1-Score | 0.9560 |
| Model Size | ~18 MB |
| Inference Time | 50-100ms |

## Installation & Setup

### Local Setup

1. Clone repository:
```bash
git clone https://github.com/YOUR_USERNAME/Spot_DeepFake.git
cd Spot_DeepFake


## Performance Metrics

| Metric | Value |
|--------|-------|
| Validation Accuracy | 96-98% |
| Precision (Fake) | 95-97% |
| Recall (Fake) | 94-96% |
| F1-Score | 0.96+ |
| Inference Time | 50-100ms |
| Model Size | 18MB |

## Project Structure

Spot_DeepFake/
├── app/
│ └── spot_deepfake_app.py # Streamlit web application
├── src/
│ ├── model.py # Model architecture
│ ├── train.py # Training script
│ └── preprocess.py # Data preprocessing
├── utils/
│ └── dataset_utils.py # Dataset loading utilities
├── models/
│ └── spot_deepfake_efficientnetb0.pth # Trained model weights
├── notebooks/
│ └── Spot_DeepFake_Training.ipynb # Colab training notebook
├── data/
│ └── .gitkeep # Placeholder (actual data on Drive)
├── reports/
│ └── training_report.txt # Training metrics
├── .streamlit/
│ └── config.toml # Streamlit configuration
├── requirements.txt # Python dependencies
├── .gitignore # Git ignore rules
└── README.md # This file
