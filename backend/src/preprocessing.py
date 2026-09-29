"""Data preprocessing and feature engineering for ambulance dispatch."""

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import PREPROCESSOR_FILE


class AmbulanceDataPreprocessor:
    """Handle data preprocessing and feature engineering."""

    CATEGORICAL_DEFAULTS = {
        "call_type": "Medical",
        "weather_condition": "Clear",
        "ambulance_type": "Basic",
    }

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.feature_names: list[str] | None = None

    def _safe_label_transform(self, encoder: LabelEncoder, series: pd.Series) -> np.ndarray:
        """Transform with fallback to first known class for unseen labels."""
        known = set(encoder.classes_)
        fallback = encoder.classes_[0]
        cleaned = series.apply(lambda v: v if v in known else fallback)
        return encoder.transform(cleaned)

    def preprocess(self, df: pd.DataFrame, fit: bool = True, test_size: float = 0.2,
                   random_state: int = 42):
        """Run the full preprocessing pipeline."""
        df = df.copy()
        df = df.drop("timestamp", axis=1, errors="ignore")

        y = df["optimal_dispatch"] if "optimal_dispatch" in df.columns else None
        X = df.drop(["optimal_dispatch", "response_time"], axis=1, errors="ignore")

        # Encode categorical columns
        if fit:
            categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
            for col in categorical_cols:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
        else:
            for col, enc in self.label_encoders.items():
                if col not in X.columns:
                    X[col] = self.CATEGORICAL_DEFAULTS.get(col, enc.classes_[0])
                X[col] = self._safe_label_transform(enc, X[col].astype(str))

        X = self._engineer_features(X)

        if fit:
            self.feature_names = X.columns.tolist()
        else:
            # Align column order/availability with training set
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.feature_names]

        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)

        if fit:
            return train_test_split(
                X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
            )
        return X_scaled, y

    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create derived features."""
        X["is_peak_hours"] = X["hour"].apply(lambda x: 1 if 8 <= x <= 18 else 0)
        X["is_night"] = X["hour"].apply(lambda x: 1 if x >= 22 or x <= 6 else 0)
        X["severity_traffic"] = X["severity_level"] * X["traffic_level"]
        X["distance_traffic"] = X["distance_to_hospital"] * X["traffic_level"]
        X["distance_bin"] = pd.cut(
            X["distance_to_hospital"],
            bins=[-0.01, 2, 5, 10, 1000],
            labels=[0, 1, 2, 3],
        ).astype(int)
        X["high_density"] = (X["population_density"] > 2000).astype(int)
        X["ambulance_availability_ratio"] = X["available_ambulances"] / X[
            "hospitals_nearby"
        ].replace(0, 1)
        X["weather_severity"] = X["temperature"].apply(
            lambda t: 1 if t < 0 or t > 30 else 0
        )
        X["resource_constraints"] = (5 - X["available_ambulances"]) + (
            5 - X["hospitals_nearby"]
        )
        return X

    def save_preprocessor(self, filename=None) -> None:
        target = filename or PREPROCESSOR_FILE
        joblib.dump(self, target)
        print(f"Preprocessor saved to {target}")

    @staticmethod
    def load_preprocessor(filename=None) -> "AmbulanceDataPreprocessor":
        target = filename or PREPROCESSOR_FILE
        return joblib.load(target)


if __name__ == "__main__":
    from .data_generation import generate_ambulance_data

    df = generate_ambulance_data(n_samples=5000)
    pre = AmbulanceDataPreprocessor()
    X_train, X_test, y_train, y_test = pre.preprocess(df, fit=True)
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
    pre.save_preprocessor()
