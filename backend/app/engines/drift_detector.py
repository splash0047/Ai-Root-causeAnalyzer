"""
AI Root Cause Analyzer - Drift Detection Engine
Statistical comparison between training baseline and live data
using KS-test and Population Stability Index (PSI).
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Optional
import json
from pathlib import Path


class DriftDetector:
    """
    Detects data drift per feature using:
    - Kolmogorov-Smirnov (KS) test for distribution shift
    - Population Stability Index (PSI) for stability measurement
    - Prediction distribution shift (for concept drift detection)
    - Feature-target relationship mutation check
    """

    def __init__(self, training_stats_path: Optional[Path] = None,
                 baseline_data_path: Optional[Path] = None,
                 p_value_threshold: float = 0.05):
        self.p_value_threshold = p_value_threshold
        self.training_stats = {}
        self.baseline_data = None

        if training_stats_path and training_stats_path.exists():
            with open(training_stats_path, "r") as f:
                self.training_stats = json.load(f)

        if baseline_data_path and baseline_data_path.exists():
            self.baseline_data = pd.read_csv(baseline_data_path)

    def detect(self, live_data: pd.DataFrame,
               predictions: Optional[np.ndarray] = None,
               actuals: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Run comprehensive drift detection and return a detailed report.
        """
        feature_cols = self.training_stats.get("feature_columns", [])
        available_cols = [c for c in feature_cols if c in live_data.columns]

        report = {
            "drift_detected": False,
            "drifted_features": [],
            "feature_drift_scores": {},
            "prediction_drift": None,
            "concept_drift_signals": None,
            "overall_drift_severity": "none",
        }

        if self.baseline_data is None or len(available_cols) == 0:
            return report

        # 1. Per-feature KS-test and PSI
        for col in available_cols:
            baseline_col = self.baseline_data[col].dropna()
            live_col = live_data[col].dropna()

            if len(live_col) < 10:
                continue

            # KS Test
            ks_stat, ks_pvalue = stats.ks_2samp(baseline_col, live_col)

            # PSI
            psi_score = self._calculate_psi(baseline_col.values, live_col.values)

            is_drifted = ks_pvalue < self.p_value_threshold

            report["feature_drift_scores"][col] = {
                "ks_statistic": round(float(ks_stat), 4),
                "ks_pvalue": round(float(ks_pvalue), 6),
                "psi": round(float(psi_score), 4),
                "is_drifted": is_drifted,
                "drift_magnitude": round(float(ks_stat), 4),
            }

            if is_drifted:
                report["drifted_features"].append(col)

        # 2. Prediction distribution shift (concept drift signal)
        if predictions is not None and self.baseline_data is not None:
            report["prediction_drift"] = self._check_prediction_drift(predictions)

        # 3. Feature-target relationship mutation
        if actuals is not None and len(available_cols) > 0:
            report["concept_drift_signals"] = self._check_concept_drift(
                live_data, actuals, available_cols
            )

        # Set global flags
        report["drift_detected"] = len(report["drifted_features"]) > 0
        drifted_count = len(report["drifted_features"])
        total_features = len(available_cols)

        if drifted_count == 0:
            report["overall_drift_severity"] = "none"
        elif drifted_count / total_features < 0.3:
            report["overall_drift_severity"] = "low"
        elif drifted_count / total_features < 0.6:
            report["overall_drift_severity"] = "medium"
        else:
            report["overall_drift_severity"] = "high"

        return report

    def _calculate_psi(self, baseline: np.ndarray, live: np.ndarray,
                       n_bins: int = 10) -> float:
        """Calculate Population Stability Index between two distributions."""
        # Create bins from baseline
        bins = np.percentile(baseline, np.linspace(0, 100, n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf
        bins = np.unique(bins)

        baseline_counts = np.histogram(baseline, bins=bins)[0]
        live_counts = np.histogram(live, bins=bins)[0]

        # Add small epsilon to avoid log(0)
        eps = 1e-6
        baseline_pct = (baseline_counts + eps) / (len(baseline) + eps * len(bins))
        live_pct = (live_counts + eps) / (len(live) + eps * len(bins))

        psi = np.sum((live_pct - baseline_pct) * np.log(live_pct / baseline_pct))
        return float(psi)

    def _check_prediction_drift(self, predictions: np.ndarray) -> Dict[str, Any]:
        """Check if prediction distribution has shifted from baseline."""
        baseline_preds = self.baseline_data.get("default")
        if baseline_preds is None:
            return {"detected": False}

        baseline_default_rate = float(baseline_preds.mean())
        live_default_rate = float(np.mean(predictions))
        shift = abs(live_default_rate - baseline_default_rate)

        return {
            "detected": shift > 0.10,
            "baseline_rate": round(baseline_default_rate, 4),
            "live_rate": round(live_default_rate, 4),
            "absolute_shift": round(shift, 4),
        }

    def _check_concept_drift(self, live_data: pd.DataFrame,
                              actuals: np.ndarray,
                              feature_cols: List[str]) -> Dict[str, Any]:
        """
        Check if feature-target relationships have changed.
        Compares correlation coefficients between baseline and live data.
        """
        if self.baseline_data is None:
            return {"detected": False}

        baseline_target = self.baseline_data.get("default")
        if baseline_target is None:
            return {"detected": False}

        correlation_shifts = {}
        significant_shifts = 0

        for col in feature_cols:
            if col not in self.baseline_data.columns or col not in live_data.columns:
                continue

            try:
                baseline_corr = float(self.baseline_data[col].corr(baseline_target))
                live_corr = float(live_data[col].corr(pd.Series(actuals)))
                shift = abs(live_corr - baseline_corr)

                correlation_shifts[col] = {
                    "baseline_correlation": round(baseline_corr, 4),
                    "live_correlation": round(live_corr, 4),
                    "shift": round(shift, 4),
                    "significant": shift > 0.15,
                }

                if shift > 0.15:
                    significant_shifts += 1
            except Exception:
                continue

        return {
            "detected": significant_shifts > 0,
            "significant_feature_count": significant_shifts,
            "details": correlation_shifts,
        }
