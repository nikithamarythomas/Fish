from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load the model, scaler, and label encoder
model = joblib.load('model/fish_species_model.pkl')
scaler = joblib.load('model/scaler.pkl')
le = joblib.load('model/label_encoder.pkl')

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [data['length1'], data['length2'], data['length3'], data['height'], data['width'], data['weight']]
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)
    prediction = model.predict(features)
    species = le.inverse_transform(prediction)[0]
    return jsonify({'species': species})

if __name__ == '__main__':
    app.run(debug=True)
