"""
AI Root Cause Analyzer — Comprehensive Test Suite
Tests all 8 API endpoints with structured assertions.
Run with: py -3.13 -m pytest tests/test_api.py -v
"""
import pytest
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# ═══════════════════════════════════════════════════════════
# 1. Health Check
# ═══════════════════════════════════════════════════════════

class TestHealth:
    def test_health_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_required_fields(self):
        r = client.get("/health")
        data = r.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "rca_logic_version" in data
        assert "model_version" in data

    def test_health_status_healthy(self):
        r = client.get("/health")
        assert r.json()["status"] == "healthy"

    def test_health_model_loaded(self):
        r = client.get("/health")
        assert r.json()["model_loaded"] is True

    def test_health_has_trace_header(self):
        r = client.get("/health")
        assert "x-trace-id" in r.headers
        assert "x-response-time" in r.headers


# ═══════════════════════════════════════════════════════════
# 2. Metrics
# ═══════════════════════════════════════════════════════════

class TestMetrics:
    def test_metrics_returns_200(self):
        r = client.get("/metrics?window_hours=24")
        assert r.status_code == 200

    def test_metrics_has_window_hours(self):
        r = client.get("/metrics?window_hours=24")
        data = r.json()
        assert "window_hours" in data
        assert "total_predictions" in data

    def test_metrics_different_windows(self):
        for hours in [1, 6, 12, 24]:
            r = client.get(f"/metrics?window_hours={hours}")
            assert r.status_code == 200


# ═══════════════════════════════════════════════════════════
# 3. Simulation (core test — exercises full RCA pipeline)
# ═══════════════════════════════════════════════════════════

class TestSimulation:
    def test_noise_simulation(self):
        r = client.post("/simulate", json={
            "failure_type": "noise",
            "feature": "credit_score",
            "n_samples": 100,
            "noise_factor": 3.0,
            "run_rca": True,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["simulation_type"] == "noise"
        assert data["n_samples"] == 100

    def test_noise_has_rca_result(self):
        r = client.post("/simulate", json={
            "failure_type": "noise",
            "feature": "credit_score",
            "n_samples": 100,
            "noise_factor": 3.0,
            "run_rca": True,
        })
        data = r.json()
        assert "rca_result" in data
        rca = data["rca_result"]
        assert "root_cause" in rca
        assert "confidence_score" in rca
        assert "severity" in rca
        assert "ranked_features" in rca
        assert "reasoning_chain" in rca

    def test_drop_simulation(self):
        r = client.post("/simulate", json={
            "failure_type": "drop",
            "feature": "income",
            "n_samples": 100,
            "run_rca": True,
        })
        assert r.status_code == 200
        assert "rca_result" in r.json()

    def test_skew_simulation(self):
        r = client.post("/simulate", json={
            "failure_type": "skew",
            "feature": "loan_amount",
            "n_samples": 100,
            "run_rca": True,
        })
        assert r.status_code == 200

    def test_interaction_simulation(self):
        r = client.post("/simulate", json={
            "failure_type": "interaction",
            "feature": "credit_score",
            "feature2": "income",
            "n_samples": 100,
            "run_rca": True,
        })
        assert r.status_code == 200

    def test_concept_drift_simulation(self):
        r = client.post("/simulate", json={
            "failure_type": "concept",
            "n_samples": 100,
            "run_rca": True,
        })
        assert r.status_code == 200

    def test_missing_values_simulation(self):
        r = client.post("/simulate", json={
            "failure_type": "missing",
            "feature": "credit_score",
            "n_samples": 100,
            "run_rca": True,
        })
        # Missing values may cause 200 or 400 depending on data state
        assert r.status_code in [200, 400]

    def test_invalid_failure_type(self):
        r = client.post("/simulate", json={
            "failure_type": "invalid_type",
            "n_samples": 100,
        })
        assert r.status_code == 400

    def test_simulation_without_rca(self):
        r = client.post("/simulate", json={
            "failure_type": "noise",
            "feature": "credit_score",
            "n_samples": 50,
            "run_rca": False,
        })
        assert r.status_code == 200
        data = r.json()
        assert "rca_result" not in data
        assert "data_preview" in data

    def test_severity_is_valid(self):
        r = client.post("/simulate", json={
            "failure_type": "noise",
            "feature": "credit_score",
            "n_samples": 100,
            "noise_factor": 5.0,
            "run_rca": True,
        })
        severity = r.json()["rca_result"]["severity"]
        assert severity in ["low", "medium", "high"]

    def test_confidence_in_range(self):
        r = client.post("/simulate", json={
            "failure_type": "noise",
            "feature": "credit_score",
            "n_samples": 100,
            "run_rca": True,
        })
        conf = r.json()["rca_result"]["confidence_score"]
        assert 0.0 <= conf <= 1.0


# ═══════════════════════════════════════════════════════════
# 4. RCA History
# ═══════════════════════════════════════════════════════════

class TestRCAHistory:
    def test_history_returns_200(self):
        r = client.get("/rca/history?limit=10")
        assert r.status_code == 200

    def test_history_has_results_key(self):
        r = client.get("/rca/history?limit=10")
        assert "results" in r.json()

    def test_history_respects_limit(self):
        r = client.get("/rca/history?limit=5")
        results = r.json()["results"]
        assert len(results) <= 5

    def test_history_severity_filter(self):
        for sev in ["high", "medium", "low"]:
            r = client.get(f"/rca/history?limit=10&severity={sev}")
            assert r.status_code == 200


# ═══════════════════════════════════════════════════════════
# 5. Data Ingestion
# ═══════════════════════════════════════════════════════════

class TestIngest:
    def _sample_records(self):
        return [
            {"credit_score": 700, "income": 50000, "loan_amount": 20000,
             "debt_to_income": 0.3, "employment_years": 5, "age": 35,
             "num_credit_lines": 3, "has_mortgage": 1, "loan_purpose_encoded": 2},
            {"credit_score": 600, "income": 40000, "loan_amount": 15000,
             "debt_to_income": 0.4, "employment_years": 2, "age": 28,
             "num_credit_lines": 2, "has_mortgage": 0, "loan_purpose_encoded": 1},
        ]

    def test_ingest_returns_200(self):
        r = client.post("/ingest", json={
            "records": self._sample_records(),
            "actuals": [0, 1],
        })
        assert r.status_code == 200

    def test_ingest_has_status(self):
        r = client.post("/ingest", json={
            "records": self._sample_records(),
            "actuals": [0, 1],
        })
        data = r.json()
        assert data["status"] == "success"
        assert data["records_ingested"] == 2

    def test_ingest_has_anomaly_flags(self):
        r = client.post("/ingest", json={
            "records": self._sample_records(),
        })
        data = r.json()
        assert "anomaly_flags" in data


# ═══════════════════════════════════════════════════════════
# 6. Full RCA
# ═══════════════════════════════════════════════════════════

class TestRCA:
    def _sample_records(self):
        return [
            {"credit_score": 700, "income": 50000, "loan_amount": 20000,
             "debt_to_income": 0.3, "employment_years": 5, "age": 35,
             "num_credit_lines": 3, "has_mortgage": 1, "loan_purpose_encoded": 2},
        ] * 20  # Need enough records for SHAP

    def test_rca_lightweight_mode(self):
        r = client.post("/rca", json={
            "records": self._sample_records(),
            "mode": "lightweight",
        })
        assert r.status_code == 200
        data = r.json()
        assert "rca_id" in data
        assert "result" in data

    def test_rca_deep_mode(self):
        r = client.post("/rca", json={
            "records": self._sample_records(),
            "mode": "deep",
        })
        assert r.status_code == 200
        data = r.json()
        result = data["result"]
        assert "root_cause" in result
        assert "confidence_score" in result
        assert "ranked_features" in result

    def test_rca_returns_integrity_report(self):
        r = client.post("/rca", json={
            "records": self._sample_records(),
            "mode": "lightweight",
        })
        assert r.status_code == 200
        assert "integrity_report" in r.json()

    def test_rca_returns_drift_report(self):
        r = client.post("/rca", json={
            "records": self._sample_records(),
            "mode": "lightweight",
        })
        assert r.status_code == 200
        assert "drift_report" in r.json()

    def test_rca_empty_records_fails(self):
        r = client.post("/rca", json={
            "records": [{"unknown_col": 123}],
            "mode": "lightweight",
        })
        assert r.status_code == 400


# ═══════════════════════════════════════════════════════════
# 7. Feedback
# ═══════════════════════════════════════════════════════════

class TestFeedback:
    def _create_rca(self):
        """Helper to create an RCA and return its ID."""
        records = [
            {"credit_score": 700, "income": 50000, "loan_amount": 20000,
             "debt_to_income": 0.3, "employment_years": 5, "age": 35,
             "num_credit_lines": 3, "has_mortgage": 1, "loan_purpose_encoded": 2},
        ] * 10
        r = client.post("/rca", json={"records": records, "mode": "lightweight"})
        assert r.status_code == 200, f"RCA creation failed: {r.json()}"
        return r.json()["rca_id"]

    def test_feedback_accurate(self):
        rca_id = self._create_rca()
        r = client.post("/feedback", json={
            "rca_id": rca_id,
            "feedback": "accurate",
            "notes": "Test feedback",
        })
        assert r.status_code == 200
        assert r.json()["status"] == "feedback_recorded"

    def test_feedback_rejected(self):
        rca_id = self._create_rca()
        r = client.post("/feedback", json={
            "rca_id": rca_id,
            "feedback": "rejected",
        })
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════
# 8. Ablation
# ═══════════════════════════════════════════════════════════

class TestAblation:
    def test_ablation_endpoint_exists(self):
        """Ablation runs 48 tests so we just verify the endpoint responds."""
        r = client.post("/ablation?n_samples=50")
        assert r.status_code == 200

    def test_ablation_has_configs(self):
        r = client.post("/ablation?n_samples=50")
        data = r.json()
        assert "configs" in data
        assert len(data["configs"]) == 4

    def test_ablation_has_summary(self):
        r = client.post("/ablation?n_samples=50")
        data = r.json()
        assert "summary" in data
        assert "best_config" in data["summary"]
        assert "accuracy_progression" in data["summary"]
