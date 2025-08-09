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

