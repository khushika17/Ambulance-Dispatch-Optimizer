"""Generate synthetic ambulance dispatch data for model training."""

from datetime import datetime, timedelta
import random

import numpy as np
import pandas as pd

from .config import DATA_CSV


def generate_ambulance_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic ambulance dispatch dataset."""
    np.random.seed(seed)
    random.seed(seed)

    start_date = datetime(2023, 1, 1)
    minute_offsets = np.random.uniform(0, 365 * 24 * 60, n_samples).astype(int)
    timestamps = [start_date + timedelta(minutes=int(m)) for m in minute_offsets]

    data = {
        "timestamp": timestamps,
        "hour": [ts.hour for ts in timestamps],
        "day_of_week": [ts.weekday() for ts in timestamps],
        "month": [ts.month for ts in timestamps],
        "is_weekend": [1 if ts.weekday() >= 5 else 0 for ts in timestamps],
        "latitude": np.random.uniform(40.5, 40.9, n_samples),
        "longitude": np.random.uniform(-74.0, -73.7, n_samples),
        "call_type": np.random.choice(
            ["Medical", "Trauma", "Cardiac", "Respiratory", "Other"], n_samples
        ),
        "severity_level": np.random.choice([1, 2, 3, 4, 5], n_samples),
        "available_ambulances": np.random.randint(2, 15, n_samples),
        "hospitals_nearby": np.random.randint(1, 8, n_samples),
        "weather_condition": np.random.choice(
            ["Clear", "Rainy", "Foggy", "Snowy"], n_samples
        ),
        "temperature": np.random.uniform(-10, 35, n_samples),
        "wind_speed": np.random.uniform(0, 50, n_samples),
        "traffic_level": np.random.randint(0, 6, n_samples),
        "distance_to_hospital": np.random.uniform(0.5, 15, n_samples),
        "population_density": np.random.uniform(100, 5000, n_samples),
        "ambulance_type": np.random.choice(
            ["Basic", "Advanced", "Cardiac"], n_samples, p=[0.4, 0.35, 0.25]
        ),
    }

    df = pd.DataFrame(data)

    weather_penalty = df["weather_condition"].map(
        {"Clear": 0, "Rainy": 1.2, "Foggy": 2.0, "Snowy": 3.0}
    )
    df["response_time"] = (
        2.5
        + df["severity_level"] * 0.3
        + df["distance_to_hospital"] * 0.45
        + df["traffic_level"] * 0.8
        + df["hour"].apply(lambda x: 1.5 if 8 <= x <= 18 else 0.5)
        + weather_penalty
        + np.random.normal(0, 1.5, n_samples)
    )
    df["response_time"] = df["response_time"].clip(lower=2)
    df["optimal_dispatch"] = (df["response_time"] < 8).astype(int)

    return df


def save_dataset(df: pd.DataFrame, filename=None) -> pd.DataFrame:
    """Save generated dataset to CSV."""
    target = filename or DATA_CSV
    df.to_csv(target, index=False)
    print(f"Dataset saved to {target}")
    return df


if __name__ == "__main__":
    df = generate_ambulance_data(n_samples=5000)
    save_dataset(df)
    print("\nDataset Overview:")
    print(df.head())
    print(f"\nDataset shape: {df.shape}")
    print(f"\nOptimal dispatch distribution:\n{df['optimal_dispatch'].value_counts()}")
