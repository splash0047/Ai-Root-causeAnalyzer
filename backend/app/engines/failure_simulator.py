"""
AI Root Cause Analyzer - Controlled Failure Simulation
Generates intentionally corrupted datasets to validate the RCA engine
can correctly identify the injected root cause.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import json


class FailureSimulator:
    """
    Controlled experiment runner that injects specific failure modes
    into clean data to stress-test the RCA engine.
    """

    def __init__(self, baseline_data_path: Optional[Path] = None,
                 training_stats_path: Optional[Path] = None):
        self.baseline_data = None
        self.training_stats = {}
        self.feature_cols = []

        if baseline_data_path and baseline_data_path.exists():
            self.baseline_data = pd.read_csv(baseline_data_path)

        if training_stats_path and training_stats_path.exists():
            with open(training_stats_path, "r") as f:
                self.training_stats = json.load(f)
                self.feature_cols = self.training_stats.get("feature_columns", [])

    def inject_feature_noise(self, feature: str, noise_factor: float = 3.0,
                              n_samples: int = 500, seed: int = 99) -> pd.DataFrame:
        """
        Inject noise by randomizing a specific feature's distribution.
        The noise_factor controls how far from normal the distribution shifts.
        """
        np.random.seed(seed)
        data = self._get_sample(n_samples)
        stats = self.training_stats.get("feature_stats", {}).get(feature, {})

        if not stats:
            raise ValueError(f"No training stats found for feature: {feature}")

        # Shift the mean by noise_factor standard deviations
        shifted_mean = stats["mean"] + noise_factor * stats["std"]
        data[feature] = np.random.normal(shifted_mean, stats["std"], len(data))

        return data

    def drop_feature(self, feature: str, drop_mode: str = "zero",
                      n_samples: int = 500) -> pd.DataFrame:
        """
        Simulate a critical feature being corrupted.
        Modes: 'zero' (set to 0), 'null' (set to NaN), 'constant' (single value)
        """
        data = self._get_sample(n_samples)

        if drop_mode == "zero":
            data[feature] = 0
        elif drop_mode == "null":
            data[feature] = np.nan
        elif drop_mode == "constant":
            data[feature] = data[feature].median()

        return data

    def skew_distribution(self, feature: str, skew_direction: str = "high",
                           percentile: float = 0.85, n_samples: int = 500,
                           seed: int = 99) -> pd.DataFrame:
        """
        Artificially push a feature's distribution to one extreme.
        """
        np.random.seed(seed)
        data = self._get_sample(n_samples)
        stats = self.training_stats.get("feature_stats", {}).get(feature, {})

        if skew_direction == "high":
            target_value = stats.get("q75", stats.get("mean", 0)) * 1.5
            data[feature] = np.random.normal(target_value, stats.get("std", 1) * 0.3, len(data))
        else:
            target_value = stats.get("q25", stats.get("mean", 0)) * 0.5
            data[feature] = np.random.normal(target_value, stats.get("std", 1) * 0.3, len(data))

        return data

    def inject_interaction_drift(self, feature1: str, feature2: str,
                                  n_samples: int = 500, seed: int = 99) -> pd.DataFrame:
        """
        Create a correlated shift in two features simultaneously
        to test interaction detection.
        """
        np.random.seed(seed)
        data = self._get_sample(n_samples)
        stats1 = self.training_stats.get("feature_stats", {}).get(feature1, {})
        stats2 = self.training_stats.get("feature_stats", {}).get(feature2, {})

        if not stats1 or not stats2:
            raise ValueError("Training stats missing for interaction features")

        # Shift both features in opposite extremes
        data[feature1] = np.random.normal(
            stats1["mean"] + 2 * stats1["std"], stats1["std"] * 0.5, len(data)
        )
        data[feature2] = np.random.normal(
            stats2["mean"] - 2 * stats2["std"], stats2["std"] * 0.5, len(data)
        )

        return data

    def inject_concept_drift(self, n_samples: int = 500, seed: int = 99) -> pd.DataFrame:
        """
        Simulate concept drift: keep feature distributions the same,
        but flip the target labels (the underlying relationship changes).
        """
        np.random.seed(seed)
        data = self._get_sample(n_samples)

        # Flip a percentage of labels to simulate concept shift
        flip_mask = np.random.random(len(data)) < 0.3
        if "default" in data.columns:
            data.loc[flip_mask, "default"] = 1 - data.loc[flip_mask, "default"]

        return data

    def inject_missing_values(self, features: list, missing_rate: float = 0.3,
                               n_samples: int = 500, seed: int = 99) -> pd.DataFrame:
        """Randomly set values to NaN across multiple features."""
        np.random.seed(seed)
        data = self._get_sample(n_samples)

        for feature in features:
            if feature in data.columns:
                mask = np.random.random(len(data)) < missing_rate
                data.loc[mask, feature] = np.nan

        return data

    def _get_sample(self, n_samples: int) -> pd.DataFrame:
        """Get a clean sample from baseline data."""
        if self.baseline_data is None:
            raise ValueError("No baseline data loaded")

        return self.baseline_data.sample(
            n=min(n_samples, len(self.baseline_data)),
            random_state=42
        ).reset_index(drop=True).copy()
