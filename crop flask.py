from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

# Load the trained model and label encoder
with open("crop_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("label_encoder.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from frontend

@app.route('/')
def home():
    return "Crop Recommendation API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        data = request.get_json()
        print("Received Data:", data)  # Debugging log

        # Validate received data
        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing one or more required fields'}), 400

        # Extract feature values
        features = np.array([
            data['N'], data['P'], data['K'], 
            data['temperature'], data['humidity'], 
            data['ph'], data['rainfall']
        ]).reshape(1, -1)

        # Predict crop
        prediction = model.predict(features)
        crop_name = label_encoder.inverse_transform(prediction)[0]
        print("Predicted Crop:", crop_name)  # Debugging log

        return jsonify({'recommended_crop': crop_name})
    
    except Exception as e:
        print("Error:", str(e))  # Debugging log
        return jsonify({'error': str(e)}), 500

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
