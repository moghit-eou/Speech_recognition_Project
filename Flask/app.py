from flask import Flask,  render_template , request
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
    
    return 'Audio received and saved', 200



if __name__ == '__main__':

    app.run(debug = True)