from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model dan scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Mengambil nilai dari form input
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    # Scaling data
    final_features = scaler.transform(final_features)
    
    # Prediksi menggunakan model
    prediction = model.predict(final_features)
    
    # Menentukan hasil
    if prediction[0] == 1:
        output = "Positif Diabetes"
    else:
        output = "Negatif Diabetes"
        
    return render_template('index.html', prediction_text='Hasil Prediksi: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)