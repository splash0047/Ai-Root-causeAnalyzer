"""
AI Root Cause Analyzer - FastAPI Main Application
Provides REST API endpoints for data ingestion, RCA execution,
metrics monitoring, RCA history retrieval, and feedback.
"""
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import joblib
import json

from app.config import settings
from app.database import get_db, PredictionLog, RCALog, WeightEvolutionLog, Base, engine
from app.engines.integrity_checker import DataIntegrityChecker
from app.engines.drift_detector import DriftDetector
from app.engines.rca_engine import RCAEngine
from app.engines.failure_simulator import FailureSimulator
from app.engines.llm_reasoner import LLMReasoner
from app.engines.vector_memory import VectorMemory

# ─── FastAPI App ───────────────────────────────────────────

app = FastAPI(
    title="AI Root Cause Analyzer",
    description="Advanced ML Model Monitoring & Root Cause Analysis System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Engine Initialization ─────────────────────────────────

# Auto-create tables on startup
Base.metadata.create_all(bind=engine)

integrity_checker = DataIntegrityChecker(
    training_stats_path=settings.TRAINING_STATS_PATH
)
drift_detector = DriftDetector(
    training_stats_path=settings.TRAINING_STATS_PATH,
    baseline_data_path=settings.BASELINE_DATA_PATH,
)
rca_engine = RCAEngine()
failure_simulator = FailureSimulator(
    baseline_data_path=settings.BASELINE_DATA_PATH,
    training_stats_path=settings.TRAINING_STATS_PATH,
)
llm_reasoner = LLMReasoner(
    gemini_api_key=settings.GEMINI_API_KEY,
    openai_api_key=settings.OPENAI_API_KEY,
)
vector_memory = VectorMemory(
    api_key=settings.PINECONE_API_KEY,
    index_name=settings.PINECONE_INDEX_NAME,
)

# Load model
model = None
if settings.BASELINE_MODEL_PATH.exists():
    model = joblib.load(settings.BASELINE_MODEL_PATH)


# ─── Pydantic Schemas ─────────────────────────────────────

class IngestRequest(BaseModel):
    """Schema for data ingestion endpoint."""
    records: List[Dict[str, Any]] = Field(..., description="List of feature dictionaries")
    actuals: Optional[List[float]] = Field(None, description="Ground truth labels if available")
    batch_id: Optional[str] = Field(None, description="Batch identifier")


class IngestResponse(BaseModel):
    status: str
    records_ingested: int
    batch_id: Optional[str]
    anomaly_flags: int


class RCARequest(BaseModel):
    """Schema for RCA execution endpoint."""
    records: List[Dict[str, Any]] = Field(..., description="Dataset snapshot for RCA")
    actuals: Optional[List[float]] = Field(None, description="Ground truth labels")
    mode: str = Field("deep", description="RCA mode: 'lightweight' or 'deep'")


class FeedbackRequest(BaseModel):
    """Schema for user feedback on RCA results."""
    rca_id: int = Field(..., description="ID of the RCA log entry")
    feedback: str = Field(..., description="'accurate' or 'rejected'")
    notes: Optional[str] = Field(None, description="Optional correction notes")


class SimulationRequest(BaseModel):
    """Schema for triggering a controlled failure simulation."""
    failure_type: str = Field(..., description="Type: noise, drop, skew, interaction, concept, missing")
    feature: Optional[str] = Field(None, description="Target feature(s)")
    feature2: Optional[str] = Field(None, description="Second feature for interaction tests")
    n_samples: int = Field(500, description="Number of samples")
    noise_factor: float = Field(3.0, description="Noise magnitude")
    run_rca: bool = Field(True, description="Automatically run RCA on simulated data")


# ─── Endpoints ─────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "AI Root Cause Analyzer",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": ["/ingest", "/rca", "/metrics", "/rca/history", "/feedback", "/simulate"],
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest_data(request: IngestRequest, db: Session = Depends(get_db)):
    """
    Ingest prediction data with adaptive anomaly scoring.
    High anomaly scores are flagged for priority RCA processing.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run model training first.")

    df = pd.DataFrame(request.records)
    feature_cols = rca_engine.feature_cols

    # Ensure features exist
    available_cols = [c for c in feature_cols if c in df.columns]
    if not available_cols:
        raise HTTPException(status_code=400, detail=f"No expected features found. Expected: {feature_cols}")

    # Generate predictions
    X = df[available_cols].fillna(0)
    predictions = model.predict_proba(X)[:, 1]

    # Adaptive anomaly scoring (higher = more anomalous)
    baseline_rate = rca_engine.training_stats.get("default_rate", 0.3)
    anomaly_scores = np.abs(predictions - baseline_rate)

    anomaly_flags = 0
    for i, record in enumerate(request.records):
        actual = request.actuals[i] if request.actuals and i < len(request.actuals) else None
        a_score = float(anomaly_scores[i])

        if a_score > 0.3:
            anomaly_flags += 1

        log = PredictionLog(
            input_features=record,
            prediction=float(predictions[i]),
            actual=actual,
            model_version=settings.MODEL_VERSION,
            anomaly_score=a_score,
            batch_id=request.batch_id
        )
        db.add(log)

    db.commit()

    return IngestResponse(
        status="success",
        records_ingested=len(request.records),
        batch_id=request.batch_id,
        anomaly_flags=anomaly_flags,
    )


@app.post("/rca")
async def run_rca(request: RCARequest, db: Session = Depends(get_db)):
    """
    Execute full Root Cause Analysis on a dataset snapshot.
    Supports lightweight and deep modes for cost control.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    df = pd.DataFrame(request.records)
    feature_cols = rca_engine.feature_cols
    available_cols = [c for c in feature_cols if c in df.columns]

    if not available_cols:
        raise HTTPException(status_code=400, detail=f"No expected features found.")

    X = df[available_cols].fillna(0)
    predictions = model.predict_proba(X)[:, 1]
    actuals = np.array(request.actuals) if request.actuals else None

    # Step 1: Integrity Check
    integrity_report = integrity_checker.check(df[available_cols])

    # Step 2: Drift Detection
    drift_report = drift_detector.detect(df[available_cols], predictions, actuals)

    # Step 3: Vector Memory Search (find similar past cases)
    memory_result = vector_memory.search_similar(rca_result={}, drift_report=drift_report)
    memory_match_score = memory_result.get("match_score", 0.0)

    # Step 4: RCA Analysis (with memory boost)
    rca_result = rca_engine.analyze(
        live_data=df,
        drift_report=drift_report,
        integrity_report=integrity_report,
        predictions=predictions,
        actuals=actuals,
        mode=request.mode,
        memory_match_score=memory_match_score,
    )

    # Step 5: LLM Reasoning (generate explanation & fix)
    llm_output = llm_reasoner.generate_explanation(
        rca_result=rca_result,
        drift_report=drift_report,
        integrity_report=integrity_report,
    )
    rca_result["llm_explanation"] = llm_output

    # Store RCA result
    rca_log = RCALog(
        root_cause=rca_result["root_cause"],
        confidence_score=rca_result["confidence_score"],
        severity=rca_result["severity"],
        ranked_features=rca_result["ranked_features"],
        explanation=llm_output.get("explanation", rca_result["root_cause_detail"]),
        suggested_fix=llm_output.get("suggested_fix", ""),
        reasoning_chain=rca_result["reasoning_chain"],
        rca_mode=rca_result["rca_mode"],
        rca_logic_version=rca_result["rca_logic_version"],
        model_version=rca_result["model_version"],
        is_uncertain=rca_result["is_uncertain"],
    )
    db.add(rca_log)
    db.commit()
    db.refresh(rca_log)

    # Step 6: Store case in vector memory for future pattern matching
    vector_memory.store_case(
        rca_id=rca_log.id,
        rca_result=rca_result,
        drift_report=drift_report,
    )

    return {
        "rca_id": rca_log.id,
        "result": rca_result,
        "integrity_report": integrity_report,
        "drift_report": drift_report,
        "similar_cases": memory_result.get("similar_cases", []),
        "llm_explanation": llm_output,
    }


@app.get("/metrics")
async def get_metrics(window_hours: int = 24, db: Session = Depends(get_db)):
    """
    Get model performance metrics with rolling window tracking.
    """
    cutoff = datetime.utcnow() - timedelta(hours=window_hours)

    logs = db.query(PredictionLog).filter(
        PredictionLog.timestamp >= cutoff,
        PredictionLog.actual.isnot(None)
    ).all()

    if not logs:
        return {
            "window_hours": window_hours,
            "total_predictions": 0,
            "message": "No predictions with ground truth in this window",
        }

    predictions = [log.prediction for log in logs]
    actuals = [log.actual for log in logs]

    from sklearn.metrics import accuracy_score, f1_score
    pred_labels = [1 if p > 0.5 else 0 for p in predictions]

    accuracy = accuracy_score(actuals, pred_labels)
    f1 = f1_score(actuals, pred_labels, zero_division=0)
    avg_anomaly = np.mean([log.anomaly_score for log in logs])

    baseline_acc = rca_engine.training_stats.get("accuracy", 0.85)

    return {
        "window_hours": window_hours,
        "total_predictions": len(logs),
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "baseline_accuracy": round(baseline_acc, 4),
        "accuracy_drop": round(max(0, baseline_acc - accuracy), 4),
        "avg_anomaly_score": round(avg_anomaly, 4),
        "high_anomaly_count": sum(1 for log in logs if log.anomaly_score > 0.3),
    }


@app.get("/rca/history")
async def get_rca_history(limit: int = 20, severity: Optional[str] = None,
                           db: Session = Depends(get_db)):
    """Get historical RCA results with optional severity filtering."""
    query = db.query(RCALog).order_by(RCALog.timestamp.desc())

    if severity:
        query = query.filter(RCALog.severity == severity)

    logs = query.limit(limit).all()

    return {
        "total": len(logs),
        "results": [
            {
                "id": log.id,
                "timestamp": log.timestamp.isoformat() if log.timestamp else None,
                "root_cause": log.root_cause,
                "confidence_score": log.confidence_score,
                "severity": log.severity,
                "is_uncertain": log.is_uncertain,
                "ranked_features": log.ranked_features,
                "explanation": log.explanation,
                "rca_mode": log.rca_mode,
                "user_feedback": log.user_feedback,
            }
            for log in logs
        ],
    }


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest, db: Session = Depends(get_db)):
    """
    Submit user feedback on an RCA result.
    Active feedback: adjusts diagnostic weights (bounded).
    """
    rca_log = db.query(RCALog).filter(RCALog.id == request.rca_id).first()
    if not rca_log:
        raise HTTPException(status_code=404, detail="RCA record not found")

    rca_log.user_feedback = request.feedback
    rca_log.feedback_notes = request.notes

    weight_update = None
    if request.feedback == "rejected":
        # Active behavioral adjustment: decrease dominant signal weight
        adjustment = {"drift": -0.02, "shap": -0.01}
        weight_update = rca_engine.update_weights(adjustment, reason=f"User rejected RCA #{request.rca_id}")

        evolution = WeightEvolutionLog(
            rca_log_id=request.rca_id,
            previous_weights=weight_update["previous_weights"],
            updated_weights=weight_update["updated_weights"],
            adjustment_reason=weight_update["reason"],
        )
        db.add(evolution)

    db.commit()

    return {
        "status": "feedback_recorded",
        "rca_id": request.rca_id,
        "feedback": request.feedback,
        "weight_update": weight_update,
    }


@app.post("/simulate")
async def run_simulation(request: SimulationRequest, db: Session = Depends(get_db)):
    """
    Run a controlled failure simulation and optionally execute RCA on the result.
    This proves the system can detect injected failures.
    """
    try:
        if request.failure_type == "noise":
            data = failure_simulator.inject_feature_noise(
                feature=request.feature, noise_factor=request.noise_factor,
                n_samples=request.n_samples
            )
        elif request.failure_type == "drop":
            data = failure_simulator.drop_feature(
                feature=request.feature, drop_mode="zero",
                n_samples=request.n_samples
            )
        elif request.failure_type == "skew":
            data = failure_simulator.skew_distribution(
                feature=request.feature, n_samples=request.n_samples
            )
        elif request.failure_type == "interaction":
            data = failure_simulator.inject_interaction_drift(
                feature1=request.feature, feature2=request.feature2,
                n_samples=request.n_samples
            )
        elif request.failure_type == "concept":
            data = failure_simulator.inject_concept_drift(n_samples=request.n_samples)
        elif request.failure_type == "missing":
            features = [request.feature] if request.feature else ["credit_score", "income"]
            data = failure_simulator.inject_missing_values(
                features=features, n_samples=request.n_samples
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown failure type: {request.failure_type}")

        result = {
            "simulation_type": request.failure_type,
            "target_feature": request.feature,
            "n_samples": len(data),
            "data_preview": data.head(5).to_dict(orient="records"),
        }

        # Optionally run RCA on simulated data
        if request.run_rca and model is not None:
            feature_cols = rca_engine.feature_cols
            available_cols = [c for c in feature_cols if c in data.columns]
            X = data[available_cols].fillna(0)
            predictions = model.predict_proba(X)[:, 1]
            actuals = data["default"].values if "default" in data.columns else None

            integrity_report = integrity_checker.check(data[available_cols])
            drift_report = drift_detector.detect(data[available_cols], predictions, actuals)
            rca_result = rca_engine.analyze(
                live_data=data, drift_report=drift_report,
                integrity_report=integrity_report,
                predictions=predictions, actuals=actuals,
                mode="deep",
            )

            result["rca_result"] = rca_result
            result["integrity_report"] = integrity_report
            result["drift_report"] = {
                "drift_detected": drift_report["drift_detected"],
                "drifted_features": drift_report["drifted_features"],
                "overall_drift_severity": drift_report["overall_drift_severity"],
            }

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ablation")
async def run_ablation(n_samples: int = 500):
    """
    Run a comprehensive ablation study across 12 failure scenarios
    and 4 RCA configurations. Proves each component's incremental value.
    """
    from app.engines.ablation_runner import AblationRunner
    runner = AblationRunner()
    results = runner.run(n_samples=n_samples)
    return results


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "rca_logic_version": settings.RCA_LOGIC_VERSION,
        "model_version": settings.MODEL_VERSION,
        "vector_memory": vector_memory.is_available,
        "llm_provider": "gemini" if settings.GEMINI_API_KEY else ("openai" if settings.OPENAI_API_KEY else "none"),
    }
