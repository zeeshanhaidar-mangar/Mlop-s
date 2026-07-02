import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

# Load models
model = joblib.load(os.path.join(BASE_DIR, 'models', 'RandomForest.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'models', 'scaler.pkl'))
label_encoder = joblib.load(os.path.join(BASE_DIR, 'models', 'label_encoder.pkl'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        features = np.array([[
            float(data['sepal_length']),
            float(data['sepal_width']),
            float(data['petal_length']),
            float(data['petal_width'])
        ]])

        features_scaled = scaler.transform(features)

        pred = model.predict(features_scaled)[0]
        label = label_encoder.inverse_transform([pred])[0]

        return jsonify({
            "success": True,
            "prediction": label
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

# IMPORTANT FOR RENDER
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
