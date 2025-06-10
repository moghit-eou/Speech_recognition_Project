from flask import Flask, render_template, request, jsonify
import os
from predict import model, words, extract_features, sr
import librosa
import numpy as np
import time

app = Flask(__name__)

# Mapping des mots vers les chiffres
word_to_digit = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
}

def report_with_probabilities(prediction):
    """
    Fonction modifiée pour retourner le meilleur résultat ET toutes les probabilités
    """
    test = []
    values = prediction[0]
    for i in range(10): 
        value = np.round(values[i] * 100, 2)
        value = round(float(value), 2)
        word = str(words[i])
        test.append((value, word))
    
    # Trier par probabilité décroissante
    values_sorted = sorted(test, key=lambda item: item[0], reverse=True)
    
    # Retourner le meilleur résultat et toutes les probabilités
    best_prediction = values_sorted[0][1]
    all_probabilities = [{"word": word, "probability": prob} for prob, word in values_sorted]
    
    return best_prediction, all_probabilities

def safe_remove_file(file_path, max_retries=3, delay=0.5):
    """Tente de supprimer un fichier de manière sécurisée avec plusieurs essais"""
    for i in range(max_retries):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
        except PermissionError:
            if i < max_retries - 1:  # Ne pas attendre au dernier essai
                time.sleep(delay)
    return False

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict')
def predict_page():
    return render_template("predict.html")

@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file found', 'success': False}), 400
    
    # Utiliser un nom de fichier unique pour éviter les conflits
    unique_id = str(int(time.time()))
    audio_path = f'recording_{unique_id}.wav'
    
    try:
        # Sauvegarde du fichier
        audio_file = request.files['audio']
        audio_file.save(audio_path)
        
        # Chargement de l'audio avec librosa (compatible avec predict.py)
        audio_signal, _ = librosa.load(audio_path, sr=sr)
        
        # Utilisation des fonctions importées de predict.py
        X_test = extract_features(audio_signal, sr)
        prediction = model.predict(X_test)
        
        # Utilisation de notre fonction modifiée pour obtenir toutes les probabilités
        predicted_word, all_probabilities = report_with_probabilities(prediction)
        
        # Conversion en chiffre
        predicted_digit = word_to_digit.get(predicted_word.lower(), '?')
        
        return jsonify({
            'success': True,
            'prediction': predicted_word,
            'digit': predicted_digit,
            'display': f"{predicted_word.capitalize()} ({predicted_digit})",
            'probabilities': all_probabilities
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
        
    finally:
        # Nettoyage garantie dans tous les cas
        safe_remove_file(audio_path)

if __name__ == '__main__':
    app.run(debug=True)