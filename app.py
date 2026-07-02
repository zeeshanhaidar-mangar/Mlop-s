import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

# -----------------------------
# Load models safely
# -----------------------------
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'RandomForest.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'label_encoder.pkl')

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print("Models loaded successfully.")
except Exception as e:
    print("Model loading failed:", e)

# -----------------------------
# Routes
# -----------------------------
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

        # scale
        features_scaled = scaler.transform(features)

        # predict
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


# -----------------------------
# SAFE STARTUP (IMPORTANT)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False   # 🔥 FIXES duplicate Flask runs
    )
