"""
AI Root Cause Analyzer — Ablation Study Runner
Systematically evaluates the incremental value of each RCA component
by running controlled simulations across 4 configurations:
  1. Baseline (Drift Only)
  2. + SHAP Feature Importance
  3. + Counterfactual Validation
  4. + Full Pipeline (Memory + LLM)
"""
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from pathlib import Path

from app.config import settings
from app.engines.integrity_checker import DataIntegrityChecker
from app.engines.drift_detector import DriftDetector
from app.engines.rca_engine import RCAEngine
from app.engines.failure_simulator import FailureSimulator


class AblationRunner:
    """Runs ablation studies across multiple failure scenarios and RCA configs."""

    FAILURE_SCENARIOS = [
        {"type": "noise", "feature": "credit_score", "noise_factor": 3.0, "expected": "credit_score"},
        {"type": "noise", "feature": "income", "noise_factor": 4.0, "expected": "income"},
        {"type": "noise", "feature": "debt_to_income", "noise_factor": 3.0, "expected": "debt_to_income"},
        {"type": "drop", "feature": "credit_score", "expected": "credit_score"},
        {"type": "drop", "feature": "loan_amount", "expected": "loan_amount"},
        {"type": "skew", "feature": "income", "expected": "income"},
        {"type": "skew", "feature": "employment_years", "expected": "employment_years"},
        {"type": "interaction", "feature": "credit_score", "feature2": "income", "expected": "credit_score"},
        {"type": "interaction", "feature": "debt_to_income", "feature2": "loan_amount", "expected": "debt_to_income"},
        {"type": "concept", "expected": "concept"},
        {"type": "missing", "feature": "credit_score", "expected": "credit_score"},
        {"type": "missing", "feature": "income", "expected": "income"},
    ]

    def __init__(self):
        self.simulator = FailureSimulator(
            baseline_data_path=settings.BASELINE_DATA_PATH,
            training_stats_path=settings.TRAINING_STATS_PATH,
        )
        self.integrity_checker = DataIntegrityChecker(
            training_stats_path=settings.TRAINING_STATS_PATH,
        )
        self.drift_detector = DriftDetector(
            training_stats_path=settings.TRAINING_STATS_PATH,
            baseline_data_path=settings.BASELINE_DATA_PATH,
        )
        self.rca_engine = RCAEngine()

        import joblib
        self.model = None
        if settings.BASELINE_MODEL_PATH.exists():
            self.model = joblib.load(settings.BASELINE_MODEL_PATH)

    def run(self, n_samples: int = 500) -> Dict[str, Any]:
        """Run the full ablation study and return structured results."""
        if self.model is None:
            return {"error": "Model not loaded"}

        configs = [
            {"name": "Drift Only", "mode": "lightweight", "use_shap": False, "use_counterfactual": False},
            {"name": "+ SHAP", "mode": "deep_shap_only", "use_shap": True, "use_counterfactual": False},
            {"name": "+ Counterfactuals", "mode": "deep", "use_shap": True, "use_counterfactual": True},
            {"name": "Full Pipeline", "mode": "deep", "use_shap": True, "use_counterfactual": True},
        ]

        results = {"configs": [], "summary": {}, "scenarios": len(self.FAILURE_SCENARIOS)}

        for config in configs:
            config_result = self._run_config(config, n_samples)
            results["configs"].append(config_result)

        # Summary
        results["summary"] = {
            "best_config": max(results["configs"], key=lambda c: c["accuracy"])["name"],
            "accuracy_progression": [c["accuracy"] for c in results["configs"]],
            "avg_time_progression": [c["avg_time_ms"] for c in results["configs"]],
        }

        return results

    def _run_config(self, config: Dict, n_samples: int) -> Dict[str, Any]:
        """Run all scenarios under a single configuration."""
        correct = 0
        total = 0
        times = []
        scenario_results = []

        for scenario in self.FAILURE_SCENARIOS:
            try:
                data = self._generate_failure(scenario, n_samples)
                feature_cols = self.rca_engine.feature_cols
                available = [c for c in feature_cols if c in data.columns]
                X = data[available].fillna(0)
                predictions = self.model.predict_proba(X)[:, 1]
                actuals = data["default"].values if "default" in data.columns else None

                integrity = self.integrity_checker.check(data[available])
                drift = self.drift_detector.detect(data[available], predictions, actuals)

                start = time.time()

                mode = "lightweight" if config["mode"] == "lightweight" else "deep"
                rca = self.rca_engine.analyze(
                    live_data=data, drift_report=drift,
                    integrity_report=integrity,
                    predictions=predictions, actuals=actuals,
                    mode=mode,
                )

                elapsed_ms = round((time.time() - start) * 1000, 1)
                times.append(elapsed_ms)

                # Check if the expected feature is in the top-3 ranked
                hit = self._check_hit(rca, scenario["expected"])
                if hit:
                    correct += 1
                total += 1

                scenario_results.append({
                    "scenario": f"{scenario['type']}:{scenario.get('feature', 'all')}",
                    "expected": scenario["expected"],
                    "detected": rca["root_cause"],
                    "top_feature": rca["ranked_features"][0]["feature"] if rca["ranked_features"] else "none",
                    "hit": hit,
                    "confidence": rca["confidence_score"],
                    "time_ms": elapsed_ms,
                })
            except Exception as e:
                total += 1
                scenario_results.append({
                    "scenario": f"{scenario['type']}:{scenario.get('feature', 'all')}",
                    "error": str(e),
                    "hit": False,
                })

        accuracy = correct / total if total > 0 else 0
        avg_time = round(np.mean(times), 1) if times else 0

        return {
            "name": config["name"],
            "mode": config["mode"],
            "accuracy": round(accuracy, 4),
            "correct": correct,
            "total": total,
            "avg_time_ms": avg_time,
            "scenarios": scenario_results,
        }

    def _generate_failure(self, scenario: Dict, n_samples: int) -> pd.DataFrame:
        """Generate a failure dataset based on the scenario config."""
        t = scenario["type"]
        f = scenario.get("feature", "credit_score")

        if t == "noise":
            return self.simulator.inject_feature_noise(f, scenario.get("noise_factor", 3.0), n_samples)
        elif t == "drop":
            return self.simulator.drop_feature(f, "zero", n_samples)
        elif t == "skew":
            return self.simulator.skew_distribution(f, n_samples=n_samples)
        elif t == "interaction":
            return self.simulator.inject_interaction_drift(f, scenario["feature2"], n_samples)
        elif t == "concept":
            return self.simulator.inject_concept_drift(n_samples)
        elif t == "missing":
            return self.simulator.inject_missing_values([f], n_samples=n_samples)
        else:
            raise ValueError(f"Unknown failure type: {t}")

    def _check_hit(self, rca: Dict, expected: str) -> bool:
        """Check if the expected root cause was detected in top-3 features."""
        if expected == "concept":
            return "concept" in rca.get("root_cause", "").lower() or "overfitting" in rca.get("root_cause", "").lower()

        ranked = rca.get("ranked_features", [])
        top_features = [f["feature"].lower() for f in ranked[:3]]
        return expected.lower() in " ".join(top_features)
