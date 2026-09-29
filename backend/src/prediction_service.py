"""Prediction service for real-time ambulance dispatch optimization."""

from datetime import datetime

import joblib
import pandas as pd

from .config import MODEL_FILE, PREPROCESSOR_FILE
from .preprocessing import AmbulanceDataPreprocessor


REQUIRED_FIELDS = (
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "latitude",
    "longitude",
    "call_type",
    "severity_level",
    "available_ambulances",
    "hospitals_nearby",
    "weather_condition",
    "temperature",
    "wind_speed",
    "traffic_level",
    "distance_to_hospital",
    "population_density",
    "ambulance_type",
)


class AmbulanceDispatchService:
    """Service for real-time ambulance dispatch predictions."""

    def __init__(self, model_path=None, preprocessor_path=None):
        self.model = joblib.load(model_path or MODEL_FILE)
        self.preprocessor = AmbulanceDataPreprocessor.load_preprocessor(
            preprocessor_path or PREPROCESSOR_FILE
        )

    def prepare_input(self, call_data: dict) -> pd.DataFrame:
        df = pd.DataFrame([call_data])

        if "timestamp" not in df.columns:
            df["timestamp"] = datetime.now()

        ts = pd.to_datetime(df["timestamp"])
        df["hour"] = df.get("hour", ts.dt.hour)
        df["day_of_week"] = df.get("day_of_week", ts.dt.weekday)
        df["month"] = df.get("month", ts.dt.month)
        df["is_weekend"] = df.get("is_weekend", (ts.dt.weekday >= 5).astype(int))
        df = df.drop(columns=["timestamp"])

        X_processed, _ = self.preprocessor.preprocess(df, fit=False)
        return X_processed

    def predict_dispatch_quality(self, call_data: dict) -> dict:
        X = self.prepare_input(call_data)
        prediction = int(self.model.predict(X)[0])
        probability = self.model.predict_proba(X)[0]
        return {
            "optimal_dispatch": prediction,
            "confidence": float(probability[prediction]),
            "probability_optimal": float(probability[1]),
            "probability_suboptimal": float(probability[0]),
            "recommendation": (
                "Dispatch immediately - conditions favor a fast response."
                if prediction == 1
                else "Consider alternative dispatch strategy - response time at risk."
            ),
        }

    def batch_predict(self, call_data_list):
        return [self.predict_dispatch_quality(c) for c in call_data_list]

    def optimize_dispatch(self, call_data: dict) -> dict:
        prediction = self.predict_dispatch_quality(call_data)
        tips: list[str] = []

        if call_data.get("traffic_level", 0) > 3:
            tips.append(
                "High traffic detected. Consider alternative routes or prioritize closer hospitals."
            )
        if call_data.get("severity_level", 0) >= 4:
            tips.append(
                "High severity case. Dispatch advanced or cardiac life support ambulance."
            )
        if call_data.get("weather_condition", "Clear") in ("Snowy", "Foggy"):
            tips.append("Poor weather. Allow extra response time and use caution.")
        if call_data.get("distance_to_hospital", 0) > 10:
            tips.append(
                "Patient is far from a hospital. Consider helicopter ambulance if available."
            )
        if call_data.get("available_ambulances", 99) < 3:
            tips.append("Low ambulance availability. Prepare for potential delays.")
        if call_data.get("hospitals_nearby", 99) < 2:
            tips.append("Few hospitals nearby - coordinate with regional dispatch.")
        if not tips:
            tips.append("Conditions are favorable - proceed with standard dispatch.")

        return {"primary_prediction": prediction, "optimization_tips": tips}


def example_prediction():
    service = AmbulanceDispatchService()
    sample = {
        "hour": 14,
        "day_of_week": 2,
        "month": 5,
        "is_weekend": 0,
        "latitude": 40.7,
        "longitude": -73.9,
        "call_type": "Cardiac",
        "severity_level": 4,
        "available_ambulances": 5,
        "hospitals_nearby": 3,
        "weather_condition": "Clear",
        "temperature": 22,
        "wind_speed": 15,
        "traffic_level": 3,
        "distance_to_hospital": 3.5,
        "population_density": 3000,
        "ambulance_type": "Advanced",
    }
    print(service.predict_dispatch_quality(sample))
    print(service.optimize_dispatch(sample))


if __name__ == "__main__":
    example_prediction()
