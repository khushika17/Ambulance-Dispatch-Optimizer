"""Project paths and configuration."""

from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent

DATA_DIR = BACKEND_DIR / "data"
MODELS_DIR = BACKEND_DIR / "models"
STATIC_DIR = BACKEND_DIR / "static"
IMAGES_DIR = STATIC_DIR / "images"

DATA_CSV = DATA_DIR / "ambulance_data.csv"
MODEL_FILE = MODELS_DIR / "ambulance_dispatch_model.pkl"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.pkl"
METRICS_FILE = MODELS_DIR / "metrics.json"

for d in (DATA_DIR, MODELS_DIR, STATIC_DIR, IMAGES_DIR):
    d.mkdir(parents=True, exist_ok=True)
