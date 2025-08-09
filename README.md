# 🗣️ Audio Spoken Digits Classification (MLP + Flask + Docker)

This project implements a **spoken digit classification system** using a **Multi-Layer Perceptron (MLP)** neural network trained on spectrograms of audio recordings. The model is deployed via **Flask** and containerized with **Docker**.

---

## 📌 Project Overview

- **Goal:** Classify spoken digits (`0` to `9`) from audio recordings
- **Feature Extraction:** Generate **spectrograms** from audio to obtain numerical representations
- **Model Type:** Multi-Layer Perceptron (MLP) — *No CNN used*
- **Frameworks:** `TensorFlow/Keras` for training, `Flask` for serving predictions
- **Training Environment:** **Google Colab GPU** (NVIDIA Tesla T4)
- **Model Size:** ~200 MB (handled using Git LFS)

---

## 📂 Repository Structure

```
.
├── Flask/                           # Flask app with prediction endpoint & UI
│   ├── static/                      # CSS, JS, and static files
│   ├── templates/                   # HTML templates for web UI
│   ├── app.py                       # Main Flask application
│   ├── predict.py                   # Prediction logic
│   ├── requirements.txt             # Python dependencies
│   ├── words.npz                    # Vocabulary/label encoder
│   ├── last_model_e20_acc78_.keras  # Trained MLP model
│   └── ...
├── Dockerfile                       # Docker build file
├── .gitattributes                   # Git LFS settings
└── README.md                        # This file
```

---

## 🧠 Model Architecture

### **MLP Design (Best Performing After Extensive Testing)**

This architecture was selected as the **optimal configuration** after experimenting with multiple different architectures, layer configurations, and hyperparameters:

- **Input Layer:** Flattened spectrogram features
- **Hidden Layer 1:** 512 neurons (ReLU + Dropout 0.3)
- **Hidden Layer 2:** 256 neurons (ReLU + Dropout 0.3)  
- **Hidden Layer 3:** 128 neurons (ReLU + Dropout 0.2)
- **Output Layer:** 10 neurons (Softmax) for digits 0–9

**Training Configuration:**
- **Optimizer:** Adam (lr=0.001)
- **Loss:** Categorical Crossentropy
- **Batch Size:** 32
- **Epochs:** 20
- **Environment:** Google Colab GPU (NVIDIA Tesla T4)

---

## 📊 Model Performance

### **Classification Results**

| Digit | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.80      | 0.84   | 0.82     | 333     |
| 1     | 0.75      | 0.81   | 0.78     | 340     |
| 2     | 0.82      | 0.86   | 0.84     | 340     |
| 3     | 0.76      | 0.73   | 0.74     | 340     |
| 4     | 0.83      | 0.84   | 0.85     | 340     |
| 5     | 0.86      | 0.84   | 0.85     | 340     |
| 6     | 0.76      | 0.74   | 0.75     | 340     |
| 7     | 0.72      | 0.81   | 0.76     | 340     |
| 8     | 0.85      | 0.76   | 0.80     | 340     |
| 9     | 0.85      | 0.76   | 0.80     | 340     |
| **Overall Accuracy** | **-** | **-** | **0.79** | **3393** |

**Key Metrics:**
- **Accuracy:** 79%
- **AUC Score:** 0.95–0.97 across classes
- **Inference Time:** ~50ms per prediction

---

## 🚀 Running the Project

### **Option 1: Using Docker (Recommended)**

#### **1. Build Docker Image**
```bash
docker build -t audio-digits .
```

#### **2. Run Container**
```bash
docker run -p 7860:7860 audio-digits
```

#### **3. Access Application**
Open your browser and navigate to:
```
http://localhost:7860
```

---

### **Option 2: Using Python 3.10 + Requirements**

#### **1. Install Python 3.10**
Ensure you have Python 3.10 installed on your system.

#### **2. Install Dependencies**
```bash
cd Flask/
pip install -r requirements.txt
```

#### **3. Install System Dependencies (Linux/macOS)**
```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y ffmpeg portaudio19-dev

# macOS (with Homebrew)
brew install ffmpeg portaudio
```

#### **4. Run Flask Application**
```bash
python app.py
```

#### **5. Access Application**
Open your browser and navigate to:
```
http://localhost:7860
```

---

## 🌐 Try the Model Online

**🔗 Live Demo:** [Hugging Face Space](YOUR_HF_SPACE_LINK_HERE)

Use the model directly in your browser without any local setup!

---

## 🛠️ Technical Details

### **Audio Processing Pipeline**
1. **Input:** WAV/MP3 audio file
2. **Resampling:** Convert to 22,050 Hz using Librosa
3. **Spectrogram Generation:** STFT → Amplitude to dB conversion
4. **Normalization:** Z-score normalization
5. **Prediction:** Forward pass through trained MLP

### **System Requirements**
- **Python:** 3.10
- **Memory:** ~500MB RAM during inference
- **Storage:** ~200MB for model files
- **Audio Formats:** WAV, MP3, OGG supported

---

## 🔧 Troubleshooting

### **Common Issues**

**Model Loading Error:**
- Ensure model file exists in Flask/ directory
- Check Git LFS is properly configured for .keras files

**Audio Processing Error:**
- Install ffmpeg system dependency
- Verify audio file format is supported

**Docker Build Fails:**
- Ensure Docker has sufficient memory (>4GB recommended)
- Check internet connection for dependency downloads

---

## 📋 Dependencies

### **Core Python Libraries**
- TensorFlow 2.13.0
- Librosa 0.10.1  
- Flask 2.3.3
- NumPy 1.24.3
- Matplotlib 3.7.2
- Scikit-learn 1.3.0

### **System Dependencies**
- FFmpeg (audio processing)
- PortAudio (audio I/O)

---

## 📈 Performance Notes

- **Best Performing Digits:** 5, 2, 4 (F1-Score > 0.84)
- **Challenging Digits:** 3, 6, 7 (confused due to phonetic similarity)
- **Model Size:** Optimized for web deployment (~200MB)
- **Inference Speed:** Real-time capable (~50ms per prediction)

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make your changes
4. Test thoroughly
5. Submit pull request

---

## 📜 License

MIT License © 2025

---

**⭐ Star this project if it helped you!**
