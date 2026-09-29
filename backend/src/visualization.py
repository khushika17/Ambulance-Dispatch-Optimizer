"""Visualization utilities for ambulance dispatch analysis."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve

from .config import IMAGES_DIR


class DispatchVisualizer:
    """Create and save dispatch visualizations under backend/static/images."""

    @staticmethod
    def _save(fig, name: str) -> str:
        path = IMAGES_DIR / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")
        return name

    @staticmethod
    def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15) -> str:
        fig, ax = plt.subplots(figsize=(10, 8))
        top = importance_df.head(top_n).iloc[::-1]
        ax.barh(top["feature"], top["importance"], color="#0d6efd")
        ax.set_xlabel("Importance Score")
        ax.set_title(f"Top {top_n} Most Important Features")
        return DispatchVisualizer._save(fig, "feature_importance.png")

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, labels=("Suboptimal", "Optimal")) -> str:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=list(labels),
            yticklabels=list(labels),
            ax=ax,
        )
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        return DispatchVisualizer._save(fig, "confusion_matrix.png")

    @staticmethod
    def plot_roc_curve(y_true, y_pred_proba) -> str:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fpr, tpr, color="#0d6efd", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        return DispatchVisualizer._save(fig, "roc_curve.png")

    @staticmethod
    def plot_response_time_distribution(df: pd.DataFrame) -> str:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(df["response_time"], bins=30, color="#0d6efd", edgecolor="black")
        axes[0].axvline(8, color="red", linestyle="--", label="8 min threshold")
        axes[0].set_title("Response Time Distribution")
        axes[0].set_xlabel("Minutes")
        axes[0].legend()

        optimal = df.loc[df["optimal_dispatch"] == 1, "response_time"]
        sub = df.loc[df["optimal_dispatch"] == 0, "response_time"]
        axes[1].hist(optimal, bins=30, alpha=0.6, label="Optimal", color="#198754")
        axes[1].hist(sub, bins=30, alpha=0.6, label="Suboptimal", color="#dc3545")
        axes[1].set_title("Response Time by Dispatch Quality")
        axes[1].set_xlabel("Minutes")
        axes[1].legend()
        return DispatchVisualizer._save(fig, "response_time_distribution.png")

    @staticmethod
    def plot_heatmap_correlations(df: pd.DataFrame) -> str:
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Feature Correlation Matrix")
        return DispatchVisualizer._save(fig, "correlation_heatmap.png")

    @staticmethod
    def plot_model_comparison(results: dict) -> str:
        rows = []
        for name, res in results.items():
            m = res["metrics"]
            rows.append(
                {
                    "Model": name,
                    "Accuracy": m["accuracy"],
                    "Precision": m["precision"],
                    "Recall": m["recall"],
                    "F1-Score": m["f1"],
                    "ROC-AUC": m["roc_auc"],
                }
            )
        metrics_df = pd.DataFrame(rows).set_index("Model")

        fig, ax = plt.subplots(figsize=(12, 6))
        metrics_df.plot(kind="bar", ax=ax, width=0.8)
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right")
        plt.xticks(rotation=20)
        return DispatchVisualizer._save(fig, "model_comparison.png")
