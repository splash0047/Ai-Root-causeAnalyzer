"""Stored memory should only be reported successful after the write succeeds."""
from app.engines.vector_memory import VectorMemory


def test_case_storage_reports_failed_upsert(monkeypatch):
    memory = VectorMemory()
    memory._available = True
    result = {"root_cause": "feature drift", "severity": "low", "ranked_features": []}

    def fail(*args):
        raise RuntimeError("index write unavailable")

    monkeypatch.setattr(memory, "_upsert_with_inference", fail)
    assert memory.store_case(1, result, {}) is False


def test_case_storage_reports_successful_upsert(monkeypatch):
    memory = VectorMemory()
    memory._available = True
    writes = []
    monkeypatch.setattr(memory, "_upsert_with_inference", lambda *args: writes.append(args))
    assert memory.store_case(1, {"root_cause": "feature drift", "severity": "low"}, {}) is True
    assert len(writes) == 1
