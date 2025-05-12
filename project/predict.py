import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF logging
os.environ["TF_ENABLE_CPU_OPTIMIZATION"] = "0"  # Suppress CPU optimization messages

import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import os

with np.load("../data/words.npz") as data :
    words = data["y"]

model = tf.keras.models.load_model("../models_last/last_model_e20_acc78_.keras")


duration = 1  
sr = 22050  
n_fft = 2048
hop_length = 512
fixed_length = 45100  


def record_audio(duration, sr):
    print ("")
    print("🎤 Enregistrement en cours...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype=np.float32)
    sd.wait()
    print("✅ Enregistrement terminé.")
    return audio.flatten()

def extract_features(signal, sr):
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(stft)
    spectrogram_flat = spectrogram.flatten()

    if len(spectrogram_flat) > fixed_length:
        spectrogram_flat = spectrogram_flat[:fixed_length]
    else:
        spectrogram_flat = np.pad(spectrogram_flat, (0, fixed_length - len(spectrogram_flat)))

    return np.array([spectrogram_flat]) 

def report(prediction):
    test = []
    values = prediction[0]
    for i in range ( 10 ) : 
        value = np.round ( values[i] * 100 , 2 )
        value = round(float(value) , 2 )
        word = str(words[i])
        test.append(( value , word ))
    values_sorted = sorted(test , key = lambda item : item[0] * -1 )

    print ("prediction from high to low probability")
    print()
    for i in range ( 10 ) : 
        print (f"{values_sorted[i][1]} is {values_sorted[i][0]} % ")


    print ( "so the prediction word is " , values_sorted[0][1] )
    return values_sorted[0][1]
        
    
print("===========================================================================================================")
audio_signal = record_audio(duration, sr)
X_test = extract_features(audio_signal, sr)
prediction = model.predict(X_test)
word = report(prediction)