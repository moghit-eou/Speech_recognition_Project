import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import json 
import numpy as np

data_path_json = "C:/Users/lenovo/Desktop/input_data.json"
with open(data_path_json , "r" ) as fp :
    data = json.load(fp)

X = np.array(data["X"])
y = np.array(data["y"])


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  

# Séparation en entraînement et test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Construction du modèle MLP
model = keras.Sequential([
    keras.layers.Input(shape=(X.shape[1],)),  # Entrée avec shape (fixed_length,)
    
    keras.layers.Dense(512, activation='sigmoid'),
    keras.layers.Dropout(0.3),  # Régularisation pour éviter l'overfitting
    
    keras.layers.Dense(256, activation='sigmoid'),
    keras.layers.Dropout(0.3),
    
    keras.layers.Dense(128, activation='sigmoid'),
    
    keras.layers.Dense(len(set(y)), activation='softmax')  # Sortie avec softmax pour classification
])

# Compilation du modèle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Affichage du résumé
model.summary()

# Entraînement du modèle
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Évaluation du modèle
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Sauvegarde du modèle
model.save("model.keras")


