"""
AI Root Cause Analyzer - Data Integrity Checker Engine
Validates incoming data for missing values, duplicates, schema mismatches,
and out-of-range anomalies.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List, Optional


class DataIntegrityChecker:
    """
    Checks data integrity by detecting:
    - Missing values per feature
    - Duplicate records
    - Schema mismatches against baseline
    - Out-of-range values
    """

    def __init__(self, training_stats_path: Optional[Path] = None):
        self.training_stats = {}
        if training_stats_path and training_stats_path.exists():
            with open(training_stats_path, "r") as f:
                self.training_stats = json.load(f)

    def check(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run all integrity checks and return a comprehensive JSON report."""
        report = {
            "total_records": len(data),
            "issues_found": False,
            "missing_values": self._check_missing(data),
            "duplicates": self._check_duplicates(data),
            "schema_mismatch": self._check_schema(data),
            "out_of_range": self._check_range(data),
        }

        # Set global flag
        report["issues_found"] = (
            report["missing_values"]["total_missing"] > 0
            or report["duplicates"]["count"] > 0
            or report["schema_mismatch"]["has_mismatch"]
            or report["out_of_range"]["total_violations"] > 0
        )

        report["issue_summary"] = self._summarize(report)
        return report

    def _check_missing(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect missing values per feature."""
        missing = data.isnull().sum()
        missing_pct = (missing / len(data) * 100).round(2)
        details = {
            col: {"count": int(missing[col]), "percentage": float(missing_pct[col])}
            for col in data.columns if missing[col] > 0
        }
        return {
            "total_missing": int(missing.sum()),
            "affected_features": list(details.keys()),
            "details": details,
        }

    def _check_duplicates(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect duplicate records."""
        dup_count = int(data.duplicated().sum())
        return {
            "count": dup_count,
            "percentage": round(dup_count / len(data) * 100, 2) if len(data) > 0 else 0,
        }

    def _check_schema(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check if incoming data matches expected baseline schema."""
        expected_cols = self.training_stats.get("feature_columns", [])
        if not expected_cols:
            return {"has_mismatch": False, "missing_columns": [], "extra_columns": []}

        incoming = set(data.columns)
        expected = set(expected_cols)

        missing_cols = list(expected - incoming)
        extra_cols = list(incoming - expected)

        return {
            "has_mismatch": len(missing_cols) > 0 or len(extra_cols) > 0,
            "missing_columns": missing_cols,
            "extra_columns": extra_cols,
        }

    def _check_range(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect values outside the training distribution bounds."""
        feature_stats = self.training_stats.get("feature_stats", {})
        violations = {}
        total_violations = 0

        for col, stats in feature_stats.items():
            if col not in data.columns:
                continue
            col_data = data[col].dropna()
            if len(col_data) == 0:
                continue

            # Use IQR-extended bounds (1.5x beyond training range)
            train_range = stats["max"] - stats["min"]
            lower_bound = stats["min"] - 0.5 * train_range
            upper_bound = stats["max"] + 0.5 * train_range

            below = int((col_data < lower_bound).sum())
            above = int((col_data > upper_bound).sum())

            if below > 0 or above > 0:
                violations[col] = {
                    "below_range": below,
                    "above_range": above,
                    "expected_range": [round(lower_bound, 2), round(upper_bound, 2)],
                }
                total_violations += below + above

        return {
            "total_violations": total_violations,
            "details": violations,
        }

    def _summarize(self, report: Dict) -> List[str]:
        """Generate human-readable summary of issues."""
        issues = []
        if report["missing_values"]["total_missing"] > 0:
            affected = ", ".join(report["missing_values"]["affected_features"])
            issues.append(f"Missing values detected in: {affected}")
        if report["duplicates"]["count"] > 0:
            issues.append(f"{report['duplicates']['count']} duplicate records found")
        if report["schema_mismatch"]["has_mismatch"]:
            if report["schema_mismatch"]["missing_columns"]:
                issues.append(f"Missing columns: {report['schema_mismatch']['missing_columns']}")
        if report["out_of_range"]["total_violations"] > 0:
            issues.append(f"Out-of-range values in: {list(report['out_of_range']['details'].keys())}")
        return issues
