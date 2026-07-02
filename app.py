import os
import streamlit as st

st.title("🔥 Debug Mode App")

st.write("Step 1: App started successfully")

BASE_DIR = os.path.dirname(__file__)
st.write("Step 2: Base directory loaded")

try:
    import joblib
    import numpy as np

    st.write("Step 3: Libraries loaded")

    model = joblib.load(os.path.join(BASE_DIR, "models", "RandomForest.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(BASE_DIR, "models", "label_encoder.pkl"))

    st.success("Step 4: Models loaded successfully")

except Exception as e:
    st.error(f"ERROR DURING LOAD: {e}")
