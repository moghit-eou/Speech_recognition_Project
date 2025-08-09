# 🗣️ Audio Spoken Digits Classification (MLP + Flask + Docker)

- This project implements a **spoken digit classification system** using a **Multi-Layer Perceptron (MLP)** neural network trained on spectrograms of audio recordings. The model is deployed via **Flask** and containerized with **Docker**.
- To read the full documenation of this project made by  people contributed to this project (FRENCH) , here's the [Link](https://drive.google.com/file/d/1xJvf5XY12HHe-Xpn4nqw97dxgLUKCEp2/view)

---
## 🌐 Try the Model Online 

**🔗 Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/moghit/Audio_Classification)

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
├── Flask/                                       # Flask app with prediction endpoint & UI
│ ├── static/                                    # CSS, JS, and static files
│ ├── templates/                                 # HTML templates for web UI
│ ├── app.py                                     # Main Flask application
│ ├── predict.py                                 # Prediction logic
│ ├── test.py                                    # Local testing script
│ ├── requirements.txt                           # Python dependencies
│ ├── words.npz                                  # Vocabulary/label encoder
│ ├── recording.wav                              # Example input audio
│ ├── last_model_e20_acc78_.keras                # Trained MLP model
│ └── ...
├── data/                                        # Dataset (if provided)
├── models_last/                                 # Model storage
├── .ipynb_checkpoints/                          # Jupyter checkpoints
├── .gitattributes                               # Git LFS settings
├── .gitignore                                   # Git ignore rules
├── README.md                                    # Project documentation
├── audio_spctrogram.ipynb                       # Notebook for spectrogram generation
├── full_project.ipynb                           # Complete training + evaluation pipeline
└── model_evaluation.ipynb                       # Metrics & plots
```

---

## 🎵 Spectrograms: The Key to Audio Classification

### **What are Spectrograms?**

A **spectrogram** is a visual representation of audio that shows how the frequency content of a signal changes over time. Think of it as a "musical fingerprint" that transforms sound waves into images that machines can understand. Link above to documntation to know more

**Why Spectrograms Work for Digit Classification:**
- Each spoken digit has unique **frequency patterns**
- Different people saying the same digit share similar **spectral signatures**
- Time-frequency representation captures both **temporal** and **tonal** characteristics

### **Spectrogram Visualization Examples**

![Spectrogram](https://i.ibb.co/svytstk1/image.png)
![Spectrogram 3D](https://i.ibb.co/rRPP9s4s/image.png)



### **Technical Spectrogram Details**
- **Sample Rate:** 22,050 Hz (audio resampling)
- **Window Function:** Hann window for STFT
- **Frequency Range:** 0 to ~11 kHz
- **Time Resolution:** Variable based on audio length
- **Color Scale:** Amplitude in decibels (dB)

---

## 🧠 Model Architecture

### **MLP Design (Best Performing After Extensive Testing)**

This architecture was selected as the **optimal configuration** after experimenting ( A LOT ) multiple different architectures, layer configurations, and hyperparameters:

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


**Key Metrics:**
- **Accuracy:** 79%
- **AUC Score:** 0.95–0.97 across classes
- **Inference Time:** ~50ms per prediction
### 📈 Performance Notes

- **Best Performing Digits:** 5, 2, 4 (F1-Score > 0.84)
- **Challenging Digits:** 3, 6, 7 (confused due to phonetic similarity)
- **Model Size:** Optimized for web deployment (~200MB)
- **Inference Speed:** Real-time capable (~50ms per prediction
---
---

## 🚀 Running the Project

clone the repository in huggingFace [Link](https://huggingface.co/spaces/moghit/Audio_Classification/tree/main)
you may have to use Git Large File Storage (LFS)

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
clone the repository in huggingFace [Link](https://huggingface.co/spaces/moghit/Audio_Classification/tree/main)

#### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```
#### **3. Run Flask Application**
```bash
python app.py
```

#### **5. Access Application**
Open your browser and navigate to:
```
http://localhost:5000 
```



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



## 📜 License

MIT License © 2025

---

**⭐ Star this project if it helped you!**
