import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

model = joblib.load(os.path.join(BASE_DIR, 'models', 'RandomForest.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'models', 'scaler.pkl'))
label_encoder = joblib.load(os.path.join(BASE_DIR, 'models', 'label_encoder.pkl'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
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

    return jsonify({"success": True, "prediction": label})


# 🔥 FIX: prevent duplicate execution + avoid port conflict
if __name__ == "__main__":
    import sys

    # If Streamlit is running it, DON'T start Flask server
    if "streamlit" not in sys.modules:
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
