"""The ablation switches must execute different diagnostic paths."""
import numpy as np
import pandas as pd

from app.engines.rca_engine import RCAEngine


def test_diagnostic_stages_are_independently_switchable(monkeypatch):
    engine = RCAEngine()
    calls = []
    monkeypatch.setattr(engine, "_analyze_shap", lambda data: (
        calls.append("shap") or {"score": 0.0, "top_features": [], "summary": "stub"}))
    monkeypatch.setattr(engine, "_run_counterfactuals", lambda *args: (
        calls.append("sensitivity") or {"validated_causes": [], "summary": "stub"}))
    monkeypatch.setattr(engine, "_test_interactions", lambda *args: (
        calls.append("interactions") or {"interactions": [], "summary": "stub"}))
    data = pd.DataFrame([{name: 0 for name in engine.feature_cols}])
    integrity = {"issues_found": False}
    drift = {"drift_detected": False}
    configs = [
        ({"use_shap": False, "use_counterfactual": False, "use_interactions": False}, []),
        ({"use_shap": True, "use_counterfactual": False, "use_interactions": False}, ["shap"]),
        ({"use_shap": True, "use_counterfactual": True, "use_interactions": False}, ["shap", "sensitivity"]),
        ({"use_shap": True, "use_counterfactual": True, "use_interactions": True}, ["shap", "sensitivity", "interactions"]),
    ]
    for switches, expected in configs:
        calls.clear()
        engine.analyze(data, drift, integrity, np.array([0.1]), mode="deep", **switches)
        assert calls == expected
