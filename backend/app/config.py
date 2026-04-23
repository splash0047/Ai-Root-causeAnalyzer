"""
AI Root Cause Analyzer - Configuration Module
Centralized settings with environment variable loading.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file (prefer .env with real keys, fallback to .env.example)
_backend_dir = Path(__file__).parent.parent
_env_path = _backend_dir / ".env"
if not _env_path.exists():
    _env_path = _backend_dir / ".env.example"
load_dotenv(dotenv_path=_env_path if _env_path.exists() else None)


class Settings:
    # --- Database ---
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://rca_user:rca_secret_2024@localhost:5432/rca_db")

    # --- Redis ---
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # --- LLM Providers ---
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # --- Pinecone ---
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "rca-cases")

    # --- RCA Engine Configuration ---
    RCA_CONFIDENCE_THRESHOLD: float = float(os.getenv("RCA_CONFIDENCE_THRESHOLD", "0.65"))
    RCA_MAX_INTERACTION_K: int = int(os.getenv("RCA_MAX_INTERACTION_K", "3"))
    FEEDBACK_MAX_WEIGHT_DELTA: float = float(os.getenv("FEEDBACK_MAX_WEIGHT_DELTA", "0.05"))

    # --- RCA Versioning ---
    RCA_LOGIC_VERSION: str = "v1.0"
    MODEL_VERSION: str = "v1.0"

    # --- Drift Thresholds ---
    DRIFT_PVALUE_THRESHOLD: float = 0.05
    SEVERITY_HIGH_THRESHOLD: float = 0.20
    SEVERITY_MEDIUM_THRESHOLD: float = 0.05

    # --- Baseline Scoring Weights ---
    WEIGHT_DRIFT: float = 0.30
    WEIGHT_CORRELATION: float = 0.25
    WEIGHT_SHAP: float = 0.30
    WEIGHT_MEMORY: float = 0.15

    # --- Model Paths ---
    MODEL_DIR: Path = Path(__file__).parent.parent.parent / "model"
    BASELINE_MODEL_PATH: Path = MODEL_DIR / "baseline_model.joblib"
    BASELINE_DATA_PATH: Path = MODEL_DIR / "baseline_data.csv"
    TRAINING_STATS_PATH: Path = MODEL_DIR / "training_stats.json"


settings = Settings()
