import os
import numpy as np
import librosa
import json

# Chemin du dossier principal
data_path = "C:/Users/lenovo/Desktop/animals_sound"

# Paramètres
hop_length = 512
n_fft = 2048
sr = 22050  
fixed_length = 10000  # Longueur fixe des vecteurs 1D

# Stockage des données
X = []
y = []

data = {
  "X" : [],
  "y" : []
}

# Parcourir les sous-dossiers (mots)
for word in os.listdir(data_path):
    word_path = os.path.join(data_path, word)
    
    if os.path.isdir(word_path):  
        print(f"Traitement du mot : {word}")

        for file in os.listdir(word_path):
            if file.endswith(".wav"):
                file_path = os.path.join(word_path, file)

                # Charger le fichier audio
                signal, _ = librosa.load(file_path, sr=sr)

                # STFT - Spectrogramme
                stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
                spectrogram = np.abs(stft)
                spectrogram_flat = spectrogram.flatten()  # Transformation en vecteur 1D

                # Ajuster la taille du vecteur
                if len(spectrogram_flat) > fixed_length:
                    spectrogram_flat = spectrogram_flat[:fixed_length]  # Tronquer
                else:
                    spectrogram_flat = np.pad(spectrogram_flat, (0, fixed_length - len(spectrogram_flat)))  # Ajouter des zéros

                # Stocker les données
                X.append(spectrogram_flat)
                y.append(word)
                print("{} is added".format(word))
                data["X"].append(spectrogram_flat.tolist())
                data["y"].append(word)




print("\n=================== Loading Data into json file =======================\n")

json_path = "../../input_data.json"
with open(json_path , "w") as fp :
    json.dump(data , fp , indent= 4)


print("the data is goooooooooooooooooooooooooooooooooooD")

