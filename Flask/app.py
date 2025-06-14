from flask import Flask, render_template, request, jsonify
import os
from predict import model, words, extract_features, sr
import librosa
import numpy as np
# this web app was designed by achraf
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
    
    # Sauvegarde du fichier
    audio_file = request.files['audio']
    audio_path = 'recording.wav'
    audio_file.save(audio_path)
    
    try:
        # Chargement de l'audio avec librosa (compatible avec predict.py)
        audio_signal, _ = librosa.load(audio_path, sr=sr)
        
        # Utilisation des fonctions importées de predict.py
        X_test = extract_features(audio_signal, sr)
        prediction = model.predict(X_test)
        
        # Utilisation de notre fonction modifiée pour obtenir toutes les probabilités
        predicted_word, all_probabilities = report_with_probabilities(prediction)
        
        # Conversion en chiffre
        predicted_digit = word_to_digit.get(predicted_word.lower(), '?')
        
        # Nettoyage
        os.remove(audio_path)
        
        return jsonify({
            'success': True,
            'prediction': predicted_word,
            'digit': predicted_digit,
            'display': f"{predicted_word.capitalize()} ({predicted_digit})",
            'probabilities': all_probabilities
        })
        
    except Exception as e:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
