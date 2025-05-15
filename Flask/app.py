from flask import Flask,  render_template , request
import sys , os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../project')))
import predict

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return 'No audio file found', 400
    
    audio_file = request.files['audio']
    audio_file.save('recording.wav') 
    signal = predict.load_audio(audio_file)
    X_test = predict.extract_features(signal , sr = 22050)  
    prediction = predict.model.predict(X_test)
    word = predict.report(prediction)
    return word, 200



if __name__ == '__main__':

    app.run(debug = True)