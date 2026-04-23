"""
AI Root Cause Analyzer - Database Layer
SQLAlchemy async-compatible engine and session management.
"""
from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, Text, DateTime, JSON
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import func

from app.config import settings

# Use SQLite fallback if Postgres is unavailable
import os
_db_url = settings.DATABASE_URL
if _db_url.startswith("postgresql"):
    try:
        _test_engine = create_engine(_db_url, pool_pre_ping=True)
        with _test_engine.connect() as conn:
            conn.execute(__import__('sqlalchemy').text("SELECT 1"))
        engine = create_engine(_db_url, pool_pre_ping=True, pool_size=10)
    except Exception:
        _sqlite_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rca_local.db")
        _db_url = f"sqlite:///{_sqlite_path}"
        engine = create_engine(_db_url, connect_args={"check_same_thread": False})
        print(f"[INFO] Postgres unavailable. Using SQLite fallback: {_sqlite_path}")
else:
    engine = create_engine(_db_url, pool_pre_ping=True, pool_size=10)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ─── ORM Models ────────────────────────────────────────────

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    input_features = Column(JSON, nullable=False)
    prediction = Column(Float, nullable=False)
    actual = Column(Float, nullable=True)
    model_version = Column(String(50), default=settings.MODEL_VERSION)
    anomaly_score = Column(Float, default=0.0)
    batch_id = Column(String(100), nullable=True)


class RCALog(Base):
    __tablename__ = "rca_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    root_cause = Column(String(200), nullable=False)
    confidence_score = Column(Float, nullable=False)
    severity = Column(String(20), default="medium")
    ranked_features = Column(JSON, nullable=True)
    explanation = Column(Text, nullable=True)
    suggested_fix = Column(Text, nullable=True)
    reasoning_chain = Column(JSON, nullable=True)
    rca_mode = Column(String(20), default="deep")
    rca_logic_version = Column(String(50), default=settings.RCA_LOGIC_VERSION)
    model_version = Column(String(50), default=settings.MODEL_VERSION)
    dataset_snapshot_id = Column(String(100), nullable=True)
    user_feedback = Column(String(20), nullable=True)
    feedback_notes = Column(Text, nullable=True)
    is_uncertain = Column(Boolean, default=False)


class WeightEvolutionLog(Base):
    __tablename__ = "weight_evolution_log"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    rca_log_id = Column(Integer, nullable=True)
    previous_weights = Column(JSON, nullable=False)
    updated_weights = Column(JSON, nullable=False)
    adjustment_reason = Column(Text, nullable=True)


# ─── Session Dependency ────────────────────────────────────

def get_db():
    """FastAPI dependency for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
