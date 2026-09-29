"""Flask API for the Ambulance Dispatch Optimizer."""

import json
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.config import (  # noqa: E402
    DATA_CSV,
    IMAGES_DIR,
    METRICS_FILE,
    MODEL_FILE,
    PREPROCESSOR_FILE,
)
from src.prediction_service import REQUIRED_FIELDS, AmbulanceDispatchService  # noqa: E402

app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIR),
    static_url_path="",
)
CORS(app)

_service: AmbulanceDispatchService | None = None


def get_service() -> AmbulanceDispatchService:
    global _service
    if _service is None:
        if not MODEL_FILE.exists() or not PREPROCESSOR_FILE.exists():
            raise FileNotFoundError(
                "Model artifacts not found. Run `python backend/train.py` first."
            )
        _service = AmbulanceDispatchService()
    return _service


def reset_service() -> None:
    global _service
    _service = None


@app.route("/")
def serve_index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/static/images/<path:filename>")
def serve_image(filename: str):
    return send_from_directory(str(IMAGES_DIR), filename)


@app.route("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": MODEL_FILE.exists(),
            "preprocessor_loaded": PREPROCESSOR_FILE.exists(),
        }
    )


@app.route("/api/model-info")
def model_info():
    if not METRICS_FILE.exists():
        return jsonify({"error": "Metrics not available. Train the model first."}), 404
    payload = json.loads(METRICS_FILE.read_text())
    images = []
    if IMAGES_DIR.exists():
        for img in sorted(IMAGES_DIR.glob("*.png")):
            images.append(
                {"name": img.stem.replace("_", " ").title(), "url": f"/static/images/{img.name}"}
            )
    payload["images"] = images
    payload["dataset_path"] = str(DATA_CSV)
    return jsonify(payload)


@app.route("/api/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Expected JSON body."}), 400

    call_data = request.get_json()
    missing = [f for f in REQUIRED_FIELDS if f not in call_data]
    if missing:
        return jsonify({"error": "Missing fields", "missing": missing}), 400

    try:
        service = get_service()
        prediction = service.predict_dispatch_quality(call_data)
        optimization = service.optimize_dispatch(call_data)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return jsonify(
        {
            "prediction": prediction,
            "optimization_tips": optimization["optimization_tips"],
            "input": call_data,
        }
    )


@app.route("/api/retrain", methods=["POST"])
def retrain():
    body = request.get_json(silent=True) or {}
    n_samples = int(body.get("n_samples", 5000))
    n_samples = max(500, min(n_samples, 50000))

    try:
        from train import run_pipeline  # noqa: WPS433
        run_pipeline(n_samples=n_samples)
        reset_service()
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Retraining failed: {exc}"}), 500

    return jsonify({"status": "ok", "n_samples": n_samples})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
