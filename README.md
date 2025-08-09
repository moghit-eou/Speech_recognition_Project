# 🗣️ Audio Spoken Digits Classification (MLP + Flask + Docker)

This project implements a **spoken digit classification system** using a **Multi-Layer Perceptron (MLP)** neural network trained on spectrograms of audio recordings.  
The model is deployed via **Flask** and containerized with **Docker**, and can be hosted on platforms like **Hugging Face Spaces**.

---

## 📌 Project Overview

- **Goal:** Classify spoken digits (`0` to `9`) from audio recordings.
- **Feature Extraction:** Generate **spectrograms** from audio to obtain numerical representations.
- **Model Type:** Multi-Layer Perceptron (MLP) — *No CNN used*.
- **Frameworks:** `TensorFlow/Keras` for training, `Flask` for serving predictions.
- **Training Environment:** **Google Colab GPU** (NVIDIA Tesla T4).
- **Deployment:** Containerized with **Docker** and deployed on **Hugging Face Spaces**.
- **Model Size:** ~200 MB (handled using Git LFS).

---

## 📂 Repository Structure
.
├── Flask/ # Flask app with prediction endpoint & UI
│ ├── static/ # CSS, JS, and static files
│ ├── templates/ # HTML templates for web UI
│ ├── app.py # Main Flask application
│ ├── predict.py # Prediction logic
│ ├── test.py # Local testing script
│ ├── requirements.txt # Python dependencies
│ ├── words.npz # Vocabulary/label encoder
│ ├── recording.wav # Example input audio
│ ├── last_model_e20_acc78_.keras # Trained MLP model
│ └── ...
├── data/ # Dataset (if provided)
├── models_last/ # Model storage
├── audio_spectrogram.ipynb # Notebook for spectrogram generation
├── model_evaluation.ipynb # Notebook for metrics & plots
├── full_project.ipynb # Complete training + evaluation pipeline
├── Dockerfile # Docker build file
├── .gitattributes # Git LFS settings
└── README.md # This file

