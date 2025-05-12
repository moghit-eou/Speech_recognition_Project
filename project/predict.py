import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF logging
os.environ["TF_ENABLE_CPU_OPTIMIZATION"] = "0"  # Suppress CPU optimization messages
import tensorflow as tf
import sounddevice as sd
import numpy as np
import librosa
import scipy.io.wavfile as wav

# Paramètres
duration = 1  # Durée de l'enregistrement (en secondes)
sr = 22050  # Taux d'échantillonnage
n_fft = 2048
hop_length = 512
fixed_length = 10000  # Taille des vecteurs 1D

try:
    # Charger le modèle entraîné avec des options spécifiques
    model = tf.keras.models.load_model("model2.keras")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

data_path = "../../animals_sound"
# Liste des mots utilisés dans l'entraînement (à adapter selon ton dataset)
words = os.listdir(data_path)


print("the list of words are \nword = {}".format(words))

def record_audio(duration, sr):
    print("==========================================🎤 Enregistrement en cours...==========================================")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype=np.float32)
    sd.wait()
    print("✅ Enregistrement terminé.")
    print("writting the file")
    wav.write("ouput.wav" , sr , audio)
    print("the file is written")
    return audio.flatten()

def extract_features(signal, sr):
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(stft)
    spectrogram_flat = spectrogram.flatten()

    # Ajuster la taille
    if len(spectrogram_flat) > fixed_length:
        spectrogram_flat = spectrogram_flat[:fixed_length]
    else:
        spectrogram_flat = np.pad(spectrogram_flat, (0, fixed_length - len(spectrogram_flat)))

    return np.array([spectrogram_flat])  # Ajouter une dimension pour le modèle


audio_test_path = "C:/Users/lenovo/Desktop/Speech_recognition_Project/project/ouput.wav"
#audio_signal , _ = librosa.load(audio_test_path  , sr = sr)

# 🎤 Enregistrer la voix

print ("========= into function record_audio =========")
audio_signal = record_audio(duration, sr) # real_time processing



# 🔍 Extraire les caractéristiques
X_test = extract_features(audio_signal, sr)

# 🤖 Faire la prédiction
prediction = model.predict(X_test)
predicted_word = words[np.argmax(prediction)]  # Trouver le mot le plus probable

print(f"🗣️ Mot prédit : {predicted_word}")
