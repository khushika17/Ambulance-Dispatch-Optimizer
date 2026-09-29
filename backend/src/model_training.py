"""Train and evaluate ML models for ambulance dispatch optimization."""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import METRICS_FILE, MODEL_FILE

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False


class AmbulanceDispatchModel:
    """Wrap a single classifier with training/eval helpers."""

    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model = self._create_model()
        self.metrics: dict = {}

    def _create_model(self):
        if self.model_type == "xgboost" and XGB_AVAILABLE:
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
            )
        if self.model_type == "lightgbm" and LGB_AVAILABLE:
            return lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=7,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=150,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1,
            )
        if self.model_type == "gradient_boost":
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=7,
                learning_rate=0.1,
                random_state=42,
            )
        return LogisticRegression(max_iter=1000, random_state=42)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is not None and y_val is not None and self.model_type in (
            "xgboost",
            "lightgbm",
        ):
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        else:
            self.model.fit(X_train, y_train)
        print(f"{self.model_type} trained.")

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test) -> dict:
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]
        self.metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        return self.metrics

    def feature_importance(self, feature_names) -> pd.DataFrame | None:
        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importance = np.abs(self.model.coef_[0])
        else:
            return None
        return pd.DataFrame(
            {"feature": feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

    def save(self, filename=None):
        target = filename or MODEL_FILE
        joblib.dump(self.model, target)
        print(f"Model saved to {target}")

    @staticmethod
    def load(filename=None):
        target = filename or MODEL_FILE
        return joblib.load(target)


def get_available_model_types() -> list[str]:
    types = ["random_forest", "gradient_boost"]
    if XGB_AVAILABLE:
        types.insert(0, "xgboost")
    if LGB_AVAILABLE:
        types.append("lightgbm")
    return types


def train_multiple_models(X_train, X_test, y_train, y_test, feature_names) -> dict:
    """Train several classifiers and return their results."""
    results = {}
    for model_type in get_available_model_types():
        print(f"\n{'=' * 50}\nTraining {model_type}\n{'=' * 50}")
        model = AmbulanceDispatchModel(model_type=model_type)
        model.train(X_train, y_train, X_test, y_test)
        metrics = model.evaluate(X_test, y_test)
        results[model_type] = {"model": model, "metrics": metrics}
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            print(f"  {key}: {metrics[key]:.4f}")
    return results


def select_best_model(results: dict) -> tuple[str, AmbulanceDispatchModel]:
    """Pick the model with the highest ROC-AUC."""
    best_name = max(results, key=lambda k: results[k]["metrics"]["roc_auc"])
    return best_name, results[best_name]["model"]


def save_metrics(metrics_by_model: dict, best_name: str, feature_names) -> None:
    serializable = {
        name: {k: v for k, v in res["metrics"].items()}
        for name, res in metrics_by_model.items()
    }
    payload = {
        "best_model": best_name,
        "feature_names": list(feature_names),
        "models": serializable,
    }
    METRICS_FILE.write_text(json.dumps(payload, indent=2))
    print(f"Metrics written to {METRICS_FILE}")


if __name__ == "__main__":
    from .data_generation import generate_ambulance_data
    from .preprocessing import AmbulanceDataPreprocessor

    df = generate_ambulance_data(n_samples=5000)
    pre = AmbulanceDataPreprocessor()
    X_train, X_test, y_train, y_test = pre.preprocess(df, fit=True)
    results = train_multiple_models(
        X_train, X_test, y_train, y_test, pre.feature_names
    )
    best_name, best_model = select_best_model(results)
    print(f"\nBest model: {best_name}")
    best_model.save()
    pre.save_preprocessor()
    save_metrics(results, best_name, pre.feature_names)
