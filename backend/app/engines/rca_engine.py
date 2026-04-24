"""
AI Root Cause Analyzer - RCA Decision Engine
Advanced multi-signal root cause analysis with:
- Bounded counterfactual causality testing
- Feature interaction detection (top-k bounded)
- Normalized confidence scoring
- Uncertainty handling
- Failure severity classification
- Ranked root cause output
"""
import numpy as np
import pandas as pd
import shap
import joblib
from itertools import combinations
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import time
import time

from app.config import settings


class RCAEngine:
    """
    Core Root Cause Analysis engine implementing multi-signal
    diagnostic reasoning with bounded counterfactual validation.
    """

    def __init__(self):
        self.model = None
        self.training_stats = {}
        self.feature_cols = []

        # Dynamic weights (evolve via feedback loop)
        self.weights = {
            "drift": settings.WEIGHT_DRIFT,
            "correlation": settings.WEIGHT_CORRELATION,
            "shap": settings.WEIGHT_SHAP,
            "memory": settings.WEIGHT_MEMORY,
        }

        self._load_model()
        self._load_training_stats()

    def _load_model(self):
        """Load the baseline model."""
        if settings.BASELINE_MODEL_PATH.exists():
            self.model = joblib.load(settings.BASELINE_MODEL_PATH)

    def _load_training_stats(self):
        """Load training statistics for bounded counterfactuals."""
        if settings.TRAINING_STATS_PATH.exists():
            with open(settings.TRAINING_STATS_PATH, "r") as f:
                self.training_stats = json.load(f)
                self.feature_cols = self.training_stats.get("feature_columns", [])

    def analyze(self, live_data: pd.DataFrame,
                drift_report: Dict[str, Any],
                integrity_report: Dict[str, Any],
                predictions: np.ndarray,
                actuals: Optional[np.ndarray] = None,
                mode: str = "deep",
                memory_match_score: float = 0.0) -> Dict[str, Any]:
        """
        Run the full RCA diagnostic pipeline.

        Args:
            live_data: Current batch of feature data
            drift_report: Output from DriftDetector
            integrity_report: Output from DataIntegrityChecker
            predictions: Model predictions for live data
            actuals: Ground truth labels (if available)
            mode: 'lightweight' or 'deep'
            memory_match_score: Similarity score from vector DB (0-1)

        Returns:
            Comprehensive RCA result with ranked causes, confidence, and reasoning chain.
        """
        reasoning_chain = []
        ranked_causes = []
        latency_breakdown = {}

        # ─── Step 1: Integrity Signal ──────────────────────
        t0 = time.perf_counter()
        integrity_signal = self._assess_integrity(integrity_report)
        latency_breakdown["Integrity"] = round((time.perf_counter() - t0) * 1000, 2)
        reasoning_chain.append({
            "step": "Data Integrity Check",
            "result": integrity_signal["summary"],
            "score": integrity_signal["score"],
        })

        # ─── Step 2: Drift Signal ──────────────────────────
        t0 = time.perf_counter()
        drift_signal = self._assess_drift(drift_report)
        latency_breakdown["Drift"] = round((time.perf_counter() - t0) * 1000, 2)
        reasoning_chain.append({
            "step": "Drift Detection",
            "result": drift_signal["summary"],
            "score": drift_signal["score"],
            "drifted_features": drift_signal.get("drifted_features", []),
        })

        # ─── Step 3: Performance Signal ────────────────────
        t0 = time.perf_counter()
        perf_signal = self._assess_performance(predictions, actuals)
        latency_breakdown["Performance"] = round((time.perf_counter() - t0) * 1000, 2)
        reasoning_chain.append({
            "step": "Performance Assessment",
            "result": perf_signal["summary"],
            "score": perf_signal["score"],
        })

        # ─── Step 4: SHAP Analysis (Deep mode only) ───────
        t0 = time.perf_counter()
        shap_signal = {"score": 0, "top_features": [], "summary": "Skipped (lightweight mode)"}
        if mode == "deep" and self.model is not None:
            shap_signal = self._analyze_shap(live_data)
            reasoning_chain.append({
                "step": "SHAP Feature Impact",
                "result": shap_signal["summary"],
                "score": shap_signal["score"],
                "top_features": shap_signal["top_features"],
            })
        latency_breakdown["SHAP"] = round((time.perf_counter() - t0) * 1000, 2)

        # ─── Step 5: Counterfactual Causality (Deep mode) ──
        t0 = time.perf_counter()
        counterfactual_signal = {"validated_causes": [], "summary": "Skipped (lightweight mode)"}
        if mode == "deep" and self.model is not None:
            suspect_features = self._get_suspect_features(drift_signal, shap_signal)
            counterfactual_signal = self._run_counterfactuals(live_data, predictions, suspect_features)
            reasoning_chain.append({
                "step": "Counterfactual Validation",
                "result": counterfactual_signal["summary"],
                "validated_causes": counterfactual_signal["validated_causes"],
            })
        latency_breakdown["Counterfactual"] = round((time.perf_counter() - t0) * 1000, 2)

        # ─── Step 6: Feature Interaction Testing (Deep) ────
        t0 = time.perf_counter()
        interaction_signal = {"interactions": [], "summary": "Skipped (lightweight mode)"}
        if mode == "deep" and self.model is not None:
            interaction_signal = self._test_interactions(live_data, predictions, shap_signal)
            reasoning_chain.append({
                "step": "Interaction Testing",
                "result": interaction_signal["summary"],
                "interactions": interaction_signal["interactions"],
            })
        latency_breakdown["Interaction"] = round((time.perf_counter() - t0) * 1000, 2)

        # ─── Step 7: Multi-Signal Diagnosis ────────────────
        t0 = time.perf_counter()
        diagnosis = self._diagnose(
            integrity_signal, drift_signal, perf_signal,
            shap_signal, counterfactual_signal, interaction_signal
        )
        latency_breakdown["Diagnosis"] = round((time.perf_counter() - t0) * 1000, 2)

        # ─── Step 8: Confidence Scoring ────────────────────
        confidence, confidence_components = self._calculate_confidence(
            drift_signal, perf_signal, shap_signal, memory_match_score
        )

        # ─── Step 9: Severity Classification ───────────────
        severity = self._classify_severity(perf_signal, drift_signal)

        # ─── Step 10: Uncertainty Check ────────────────────
        is_uncertain = confidence < settings.RCA_CONFIDENCE_THRESHOLD

        # Build ranked causes list
        ranked_causes = self._build_ranked_causes(
            drift_signal, shap_signal, counterfactual_signal, interaction_signal
        )

        return {
            "root_cause": diagnosis["root_cause"],
            "root_cause_detail": diagnosis["detail"],
            "confidence_score": round(confidence, 4),
            "confidence_components": confidence_components,
            "severity": severity,
            "is_uncertain": is_uncertain,
            "ranked_features": ranked_causes,
            "reasoning_chain": reasoning_chain,
            "latency_breakdown": latency_breakdown,
            "rca_mode": mode,
            "rca_logic_version": settings.RCA_LOGIC_VERSION,
            "model_version": settings.MODEL_VERSION,
        }

    # ─── Signal Assessment Methods ─────────────────────────

    def _assess_integrity(self, report: Dict) -> Dict[str, Any]:
        """Score data integrity issues."""
        if not report.get("issues_found", False):
            return {"score": 0.0, "summary": "No integrity issues detected"}

        missing_pct = report["missing_values"]["total_missing"] / max(report["total_records"], 1)
        dup_pct = report["duplicates"]["count"] / max(report["total_records"], 1)
        schema_issue = 1.0 if report["schema_mismatch"]["has_mismatch"] else 0.0

        score = min(1.0, missing_pct * 2 + dup_pct + schema_issue * 0.5)
        summary = "; ".join(report.get("issue_summary", ["Issues detected"]))
        return {"score": round(score, 4), "summary": summary}

    def _assess_drift(self, report: Dict) -> Dict[str, Any]:
        """Score drift severity across features."""
        if not report.get("drift_detected", False):
            return {"score": 0.0, "summary": "No drift detected", "drifted_features": []}

        drifted = report.get("drifted_features", [])
        scores = report.get("feature_drift_scores", {})

        # Average KS statistic of drifted features
        drift_magnitudes = [
            scores[f]["drift_magnitude"] for f in drifted if f in scores
        ]
        avg_magnitude = np.mean(drift_magnitudes) if drift_magnitudes else 0.0

        return {
            "score": round(float(avg_magnitude), 4),
            "summary": f"Drift detected in {len(drifted)} features: {', '.join(drifted)}",
            "drifted_features": drifted,
            "per_feature_magnitude": {f: scores[f]["drift_magnitude"] for f in drifted if f in scores},
        }

    def _assess_performance(self, predictions: np.ndarray,
                             actuals: Optional[np.ndarray]) -> Dict[str, Any]:
        """Score performance degradation."""
        if actuals is None:
            return {"score": 0.0, "summary": "No ground truth available", "accuracy_drop": 0.0}

        from sklearn.metrics import accuracy_score
        baseline_acc = self.training_stats.get("accuracy", 0.85)
        live_acc = accuracy_score(actuals, (predictions > 0.5).astype(int))
        drop = max(0, baseline_acc - live_acc)

        return {
            "score": round(min(1.0, drop * 3), 4),
            "summary": f"Accuracy: {live_acc:.4f} (baseline: {baseline_acc:.4f}, drop: {drop:.4f})",
            "accuracy_drop": round(drop, 4),
            "live_accuracy": round(live_acc, 4),
            "baseline_accuracy": round(baseline_acc, 4),
        }

    def _analyze_shap(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run SHAP analysis to identify impactful features."""
        try:
            feature_data = data[self.feature_cols] if self.feature_cols else data
            # Use a sample for performance
            sample = feature_data.sample(min(200, len(feature_data)), random_state=42)

            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(sample)

            # Mean absolute SHAP per feature
            mean_shap = np.abs(shap_values).mean(axis=0)
            feature_importance = dict(zip(sample.columns, mean_shap.round(4)))

            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = [{"feature": f, "shap_impact": float(v)} for f, v in sorted_features[:5]]

            max_impact = max(mean_shap) if len(mean_shap) > 0 else 0
            return {
                "score": round(float(max_impact), 4),
                "top_features": top_features,
                "all_features": feature_importance,
                "summary": f"Top impactful feature: {sorted_features[0][0]} ({sorted_features[0][1]:.4f})" if sorted_features else "No features analyzed",
            }
        except Exception as e:
            return {"score": 0.0, "top_features": [], "all_features": {}, "summary": f"SHAP analysis failed: {str(e)}"}

    # ─── Causality Engine ──────────────────────────────────

    def _get_suspect_features(self, drift_signal: Dict, shap_signal: Dict) -> List[str]:
        """Identify the top-k suspect features for counterfactual testing."""
        suspects = set()

        # Add drifted features
        for f in drift_signal.get("drifted_features", []):
            suspects.add(f)

        # Add SHAP top features
        for item in shap_signal.get("top_features", []):
            suspects.add(item["feature"])

        # Bound to top-k
        k = settings.RCA_MAX_INTERACTION_K
        return list(suspects)[:k]

    def _run_counterfactuals(self, data: pd.DataFrame,
                              predictions: np.ndarray,
                              suspect_features: List[str]) -> Dict[str, Any]:
        """
        Bounded counterfactual testing: Revert suspect features
        to their training-baseline values and check if prediction changes.
        Uses training distribution medians as realistic reset points.
        """
        feature_stats = self.training_stats.get("feature_stats", {})
        feature_data = data[self.feature_cols].copy() if self.feature_cols else data.copy()
        validated = []

        # Sample failed predictions
        failed_mask = predictions > 0.5  # Predicted default
        failed_indices = np.where(failed_mask)[0]
        if len(failed_indices) == 0:
            return {"validated_causes": [], "summary": "No failed predictions to analyze"}

        sample_idx = failed_indices[:min(100, len(failed_indices))]
        sample_data = feature_data.iloc[sample_idx].copy()
        original_preds = self.model.predict(sample_data)

        for feature in suspect_features:
            if feature not in feature_stats or feature not in sample_data.columns:
                continue

            # Bounded reset: use the MEDIAN from training distribution
            baseline_value = feature_stats[feature]["median"]

            modified = sample_data.copy()
            modified[feature] = baseline_value
            new_preds = self.model.predict(modified)

            # Count how many predictions flipped
            flipped = int((original_preds != new_preds).sum())
            flip_rate = flipped / len(sample_idx) if len(sample_idx) > 0 else 0

            if flip_rate > 0.1:  # At least 10% predictions change
                validated.append({
                    "feature": feature,
                    "flip_rate": round(flip_rate, 4),
                    "flipped_count": flipped,
                    "total_tested": len(sample_idx),
                    "baseline_value_used": round(baseline_value, 4),
                    "causality_confirmed": True,
                })

        summary = f"Validated {len(validated)} causal features out of {len(suspect_features)} suspects"
        return {"validated_causes": validated, "summary": summary}

    def _test_interactions(self, data: pd.DataFrame,
                            predictions: np.ndarray,
                            shap_signal: Dict) -> Dict[str, Any]:
        """
        Test bounded pairwise feature interactions.
        Only tests combinations of top-k features to prevent combinatorial explosion.
        """
        top_features = [f["feature"] for f in shap_signal.get("top_features", [])[:settings.RCA_MAX_INTERACTION_K]]

        if len(top_features) < 2 or self.model is None:
            return {"interactions": [], "summary": "Not enough features for interaction testing"}

        feature_stats = self.training_stats.get("feature_stats", {})
        feature_data = data[self.feature_cols].copy() if self.feature_cols else data.copy()

        failed_mask = predictions > 0.5
        failed_indices = np.where(failed_mask)[0]
        if len(failed_indices) == 0:
            return {"interactions": [], "summary": "No failed predictions to analyze"}

        sample_idx = failed_indices[:min(50, len(failed_indices))]
        sample_data = feature_data.iloc[sample_idx].copy()
        original_preds = self.model.predict(sample_data)

        interactions = []
        for f1, f2 in combinations(top_features, 2):
            if f1 not in feature_stats or f2 not in feature_stats:
                continue

            modified = sample_data.copy()
            modified[f1] = feature_stats[f1]["median"]
            modified[f2] = feature_stats[f2]["median"]
            new_preds = self.model.predict(modified)

            paired_flip = int((original_preds != new_preds).sum())
            paired_rate = paired_flip / len(sample_idx) if len(sample_idx) > 0 else 0

            # Compare to individual flips (did pairing increase the effect?)
            mod_f1 = sample_data.copy()
            mod_f1[f1] = feature_stats[f1]["median"]
            f1_flip = int((original_preds != self.model.predict(mod_f1)).sum()) / len(sample_idx)

            mod_f2 = sample_data.copy()
            mod_f2[f2] = feature_stats[f2]["median"]
            f2_flip = int((original_preds != self.model.predict(mod_f2)).sum()) / len(sample_idx)

            interaction_boost = paired_rate - max(f1_flip, f2_flip)

            if interaction_boost > 0.05:
                interactions.append({
                    "features": [f1, f2],
                    "paired_flip_rate": round(paired_rate, 4),
                    "individual_max_flip_rate": round(max(f1_flip, f2_flip), 4),
                    "interaction_boost": round(interaction_boost, 4),
                })

        summary = f"Found {len(interactions)} significant interactions among {len(top_features)} features"
        return {"interactions": interactions, "summary": summary}

    # ─── Diagnosis & Scoring ───────────────────────────────

    def _diagnose(self, integrity, drift, perf, shap_sig, counterfactual, interaction) -> Dict:
        """Multi-signal root cause diagnosis."""
        has_integrity_issues = integrity["score"] > 0.3
        has_drift = drift["score"] > 0.2
        has_perf_drop = perf["score"] > 0.2
        has_interaction = len(interaction.get("interactions", [])) > 0
        has_causal = len(counterfactual.get("validated_causes", [])) > 0

        # Multi-signal reasoning (not IF-ELSE)
        if has_integrity_issues and has_perf_drop:
            return {
                "root_cause": "Data Integrity Issue Causing Model Failure",
                "detail": f"Data quality problems ({integrity['summary']}) are degrading predictions.",
            }
        elif has_drift and has_perf_drop and has_causal:
            top_causal = counterfactual["validated_causes"][0]["feature"] if counterfactual["validated_causes"] else "unknown"
            return {
                "root_cause": "Data Drift Causing Model Failure",
                "detail": f"Feature '{top_causal}' has drifted and causally confirmed to flip predictions.",
            }
        elif has_interaction and has_perf_drop:
            pair = interaction["interactions"][0]["features"]
            return {
                "root_cause": "Feature Interaction Shift",
                "detail": f"Interaction between '{pair[0]}' and '{pair[1]}' is causing compound failure.",
            }
        elif not has_drift and has_perf_drop:
            return {
                "root_cause": "Concept Drift or Model Overfitting",
                "detail": "Performance dropped without data distribution drift. Feature-target relationships may have changed.",
            }
        elif has_drift and not has_perf_drop:
            return {
                "root_cause": "Data Drift (No Performance Impact Yet)",
                "detail": f"Drift detected in {', '.join(drift.get('drifted_features', []))} but model is still performing within bounds.",
            }
        else:
            return {
                "root_cause": "No Significant Issue Detected",
                "detail": "All signals within normal parameters.",
            }

    def _calculate_confidence(self, drift, perf, shap_sig, memory_boost: float) -> tuple:
        """
        Normalized confidence score with explicit mathematical defensibility.
        All components are min-max normalized to [0,1] before weighted aggregation.
        """
        drift_norm = self._normalize(drift["score"], 0, 1)
        corr_norm = self._normalize(perf["score"], 0, 1)
        shap_norm = self._normalize(shap_sig["score"], 0, 2)
        mem_norm = min(1.0, memory_boost)

        components = {
            "drift_signal": round(self.weights["drift"] * drift_norm, 4),
            "correlation": round(self.weights["correlation"] * corr_norm, 4),
            "shap_validation": round(self.weights["shap"] * shap_norm, 4),
            "memory_match": round(self.weights["memory"] * mem_norm, 4)
        }
        
        confidence = sum(components.values())
        return min(1.0, max(0.0, confidence)), components

    def _normalize(self, value: float, low: float, high: float) -> float:
        """Min-max normalize a value to [0, 1]."""
        if high == low:
            return 0.0
        return max(0.0, min(1.0, (value - low) / (high - low)))

    def _classify_severity(self, perf: Dict, drift: Dict) -> Dict[str, Any]:
        """Dynamic severity classification."""
        acc_drop = perf.get("accuracy_drop", 0)
        
        # Affected ratio calculation
        drifted_features = len(drift.get("drifted_features", []))
        total_features = max(len(self.feature_cols) if self.feature_cols else 1, drifted_features)
        affected_ratio = drifted_features / total_features if total_features > 0 else 0
        business_weight = 0.5 # Default business importance

        score = (0.5 * acc_drop) + (0.3 * affected_ratio) + (0.2 * business_weight)
        
        if acc_drop > settings.SEVERITY_HIGH_THRESHOLD or score > 0.4:
            level = "high"
        elif acc_drop > settings.SEVERITY_MEDIUM_THRESHOLD or score > 0.2:
            level = "medium"
        else:
            level = "low"
            
        return {"level": level, "score": round(score, 4)}

    def _build_ranked_causes(self, drift, shap_sig, counterfactual, interaction) -> List[Dict]:
        """Build ranked list of root causes sorted by impact."""
        ranked = []

        # From counterfactual validated features
        for cause in counterfactual.get("validated_causes", []):
            ranked.append({
                "feature": cause["feature"],
                "impact": cause["flip_rate"],
                "source": "counterfactual",
                "causality_confirmed": True,
            })

        # From SHAP (non-duplicated)
        existing = {r["feature"] for r in ranked}
        for feat in shap_sig.get("top_features", []):
            if feat["feature"] not in existing:
                ranked.append({
                    "feature": feat["feature"],
                    "impact": feat["shap_impact"],
                    "source": "shap",
                    "causality_confirmed": False,
                })

        # From interactions
        for inter in interaction.get("interactions", []):
            ranked.append({
                "feature": f"{inter['features'][0]} × {inter['features'][1]}",
                "impact": inter["interaction_boost"],
                "source": "interaction",
                "causality_confirmed": False,
            })

        ranked.sort(key=lambda x: x["impact"], reverse=True)
        return ranked

    def update_weights(self, adjustment: Dict[str, float], reason: str = "") -> Dict[str, Any]:
        """
        Update diagnostic weights from feedback loop.
        Bounded to prevent corrupting the system.
        """
        max_delta = settings.FEEDBACK_MAX_WEIGHT_DELTA
        previous = dict(self.weights)

        for key, delta in adjustment.items():
            if key in self.weights:
                clamped_delta = max(-max_delta, min(max_delta, delta))
                self.weights[key] = max(0.05, min(0.60, self.weights[key] + clamped_delta))

        # Re-normalize weights to sum to 1.0
        total = sum(self.weights.values())
        self.weights = {k: round(v / total, 4) for k, v in self.weights.items()}

        return {
            "previous_weights": previous,
            "updated_weights": dict(self.weights),
            "adjustment_applied": adjustment,
            "reason": reason,
        }
