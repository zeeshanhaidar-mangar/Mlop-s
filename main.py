import sys
from data_loader import load_data_from_db
from preprocessing import preprocess_data
from model_training import train_and_evaluate
import joblib
import os
from config import MODELS_DIR

def run_pipeline():
    print("Starting MLOps Pipeline...")

    # 1. Load Data
    print("\n[Step 1] Loading data from XAMPP DB...")
    df = load_data_from_db()
    if df is None:
        print("Pipeline failed at data loading stage.")
        return

    # 2. Preprocess Data
    print("\n[Step 2] Preprocessing data...")
    X_train, X_test, y_train, y_test, le, scaler = preprocess_data(df)

    # Save the label encoder and scaler for future inference
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    print(f"Saved LabelEncoder and Scaler to {MODELS_DIR}")

    # 3. Train and Evaluate
    print("\n[Step 3] Training models...")
    results = train_and_evaluate(X_train, X_test, y_train, y_test, le)

    print("\n" + "="*40)
    print("Pipeline Execution Completed Successfully!")
    print("="*40)
    for res in results:
        print(f"- {res['name']}: {res['accuracy']:.4f} accuracy")
    print("="*40)
    print(f"Check 'metrics.txt' for detailed reports.")

if __name__ == "__main__":
    run_pipeline()
