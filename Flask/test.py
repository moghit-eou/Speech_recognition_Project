import sys , os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../project')))
import predict


signal = predict.load_audio("recording.wav")
X_test = predict.extract_features(signal , sr = 22050)  
prediction = predict.model.predict(X_test)
word = predict.report(prediction)

print ( word )
print ( "end of the program" )