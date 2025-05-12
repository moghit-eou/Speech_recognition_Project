import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import json 
import numpy as np

print("waiting for loading")

data = np.load("../digits_data.npz")

print("the data loaded successfully ")

X = np.array(data["first"])
y = np.array(data["second"])

print("taking X and y ")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  

print("splitting the data ")
# Séparation en entraînement et test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Construction du modèle MLP
model = keras.Sequential([
    keras.layers.Input(shape=(X.shape[1],)),  # Entrée avec shape (fixed_length,)
    
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.3),  # Régularisation pour éviter l'overfitting
    
    keras.layers.Dense(256, activation='relu'),
    
    keras.layers.Dropout(0.3),
    
    keras.layers.Dense(128, activation='relu'),
    
 
    keras.layers.Dense(len(set(y)), activation='softmax')  # Sortie avec softmax pour classification
])
print("model compilation")
# Compilation du modèle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Affichage du résumé
model.summary()

print("the training")
# Entraînement du modèle
history = model.fit(X_train, y_train, epochs=15, batch_size=100, validation_data=(X_test, y_test))

# Évaluation du modèle
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc * 100:.2f}%")

print("PLOTIIIIIIIIING")
def plot_history(history):
    
    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

plot_history(history)
# Sauvegarde du modèle
model.save("model.keras")


