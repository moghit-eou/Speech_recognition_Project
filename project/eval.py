import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import os


model = tf.keras.models.load_model("../models/model_2.keras")
test_loss, test_acc = model.evaluate(X_test, y_test)
