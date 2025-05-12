import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import librosa
import json

# Chemin du dossier principal
data_path = "C:/Users/lenovo/Desktop/animals_sound"

# Paramètres
hop_length = 512
sr = 22050 * 2
n_fft = 2048



X = [] 
y = []

# Parcourir les sous-dossiers (mots)
for word in os.listdir(data_path):
    word_path = os.path.join(data_path, word)
    
    if os.path.isdir(word_path):  
        print(f"Traitement du mot : {word}")

        for file in os.listdir(word_path):
            if file.endswith(".wav"):
                file_path = os.path.join(word_path, file)

                # Charger le fichier audio
                signal, sample_rate = librosa.load(file_path, sr=sr)

                spectrogram = librosa.feature.melspectrogram(y = signal , sr = sample_rate ) 
                spectrogram = librosa.power_to_db(spectrogram, ref = np.max )

                spectrogram = spectrogram.flatten()

                X.append(spectrogram)
                y.append(word)




print("\n=================== Loading Data into  file =======================\n")


np.savez("data_melspectrogram.npz" , X = X , y = y )

print("the data is goooooooooooooooooooooooooooooooooooD")

