import os
import joblib
import numpy as np
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models
model = joblib.load(os.path.join(BASE_DIR, 'models', 'RandomForest.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'models', 'scaler.pkl'))
label_encoder = joblib.load(os.path.join(BASE_DIR, 'models', 'label_encoder.pkl'))

# UI
st.set_page_config(page_title="Iris Classifier", layout="centered")

st.title("🌸 Iris Classification App")

st.write("Enter flower measurements below:")

# Inputs
sepal_length = st.number_input("Sepal Length", min_value=0.0)
sepal_width = st.number_input("Sepal Width", min_value=0.0)
petal_length = st.number_input("Petal Length", min_value=0.0)
petal_width = st.number_input("Petal Width", min_value=0.0)

# Predict
if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    label = label_encoder.inverse_transform([prediction])[0]

    st.success(f"Prediction: {label}")
