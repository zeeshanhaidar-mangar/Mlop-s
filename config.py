import os

# Database Configuration (XAMPP Default)
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "mlops",
    "table": "iris"
}

# Model Configuration
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Metrics Configuration
METRICS_FILE = "metrics.txt"
