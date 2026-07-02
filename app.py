import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Paths to saved models
MODEL_PATH = os.path.join('models', 'RandomForest.pkl')
SCALER_PATH = os.path.join('models', 'scaler.pkl')
ENCODER_PATH = os.path.join('models', 'label_encoder.pkl')

# Load the models
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        features = [
            float(data['sepal_length']),
            float(data['sepal_width']),
            float(data['petal_length']),
            float(data['petal_width'])
        ]
        
        # Reshape and scale
        features_arr = np.array([features])
        features_scaled = scaler.transform(features_arr)
        
        # Predict
        prediction_idx = model.predict(features_scaled)[0]
        prediction_label = label_encoder.inverse_transform([prediction_idx])[0]
        
        return jsonify({
            'success': True,
            'prediction': prediction_label
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=False,
        use_reloader=False
    )

