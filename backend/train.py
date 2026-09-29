"""End-to-end training pipeline.

Run from the project root:
    python backend/train.py
"""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.data_generation import generate_ambulance_data, save_dataset  # noqa: E402
from src.model_training import (  # noqa: E402
    save_metrics,
    select_best_model,
    train_multiple_models,
)
from src.preprocessing import AmbulanceDataPreprocessor  # noqa: E402
from src.visualization import DispatchVisualizer  # noqa: E402


def run_pipeline(n_samples: int = 5000) -> None:
    print("\n" + "=" * 70)
    print("AMBULANCE DISPATCH OPTIMIZER - ML PIPELINE")
    print("=" * 70)

    print("\n[1/7] Generating synthetic dataset...")
    df = generate_ambulance_data(n_samples=n_samples)
    save_dataset(df)
    print(f"  rows={len(df)}  optimal_dispatch_rate={df['optimal_dispatch'].mean():.2%}")

    print("\n[2/7] Preprocessing...")
    pre = AmbulanceDataPreprocessor()
    X_train, X_test, y_train, y_test = pre.preprocess(df, fit=True)
    pre.save_preprocessor()
    print(f"  train={X_train.shape}  test={X_test.shape}  features={X_train.shape[1]}")

    print("\n[3/7] Training models...")
    results = train_multiple_models(X_train, X_test, y_train, y_test, pre.feature_names)

    print("\n[4/7] Selecting best model...")
    best_name, best_model = select_best_model(results)
    print(f"  best={best_name}  roc_auc={results[best_name]['metrics']['roc_auc']:.4f}")
    best_model.save()

    print("\n[5/7] Persisting metrics...")
    save_metrics(results, best_name, pre.feature_names)

    print("\n[6/7] Feature importance...")
    importance_df = best_model.feature_importance(pre.feature_names)
    if importance_df is not None:
        print(importance_df.head(10).to_string(index=False))

    print("\n[7/7] Generating visualizations...")
    viz = DispatchVisualizer()
    if importance_df is not None:
        viz.plot_feature_importance(importance_df)
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    viz.plot_confusion_matrix(y_test, y_pred)
    viz.plot_roc_curve(y_test, y_proba)
    viz.plot_response_time_distribution(df)
    viz.plot_heatmap_correlations(df)
    viz.plot_model_comparison(results)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
