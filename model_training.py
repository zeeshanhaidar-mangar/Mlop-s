import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from config import MODELS_DIR, METRICS_FILE

def train_and_evaluate(X_train, X_test, y_train, y_test, label_encoder):
    """
    Trains 3 models, saves them, and calculates metrics.
    """
    models = {
        "LogisticRegression": LogisticRegression(max_iter=200),
        "RandomForest": RandomForestClassifier(n_estimators=100),
        "SVM": SVC(probability=True)
    }

    results = []

    with open(METRICS_FILE, "w") as f:
        f.write("Iris Classification Model Performance\n")
        f.write("="*40 + "\n\n")

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

        print(f"{name} Accuracy: {acc:.4f}")

        # Save Metrics to file
        with open(METRICS_FILE, "a") as f:
            f.write(f"Model: {name}\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + "\n")
            f.write("Classification Report:\n")
            f.write(cr + "\n")
            f.write("-" * 40 + "\n\n")

        # Save Model to .pkl
        model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(model, model_path)
        print(f"Saved model to {model_path}")

        results.append({"name": name, "accuracy": acc, "path": model_path})

    return results

if __name__ == "__main__":
    pass
