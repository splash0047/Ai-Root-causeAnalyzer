# AI Root Cause Analyzer (RCA) - Project Implementation Context

This document serves as a comprehensive technical context file for LLMs to understand the architecture, implementation details, and design decisions behind the AI Root Cause Analyzer project.

---

## 1. High-Level Overview
The AI Root Cause Analyzer is a production-grade observability platform for Machine Learning systems. It goes beyond standard anomaly detection (e.g., "The model is drifting") to perform full causal diagnostics (e.g., "The model is drifting because 'credit_score' dropped by 30% due to an upstream pipeline failure, which caused a 15% drop in accuracy").

It uses a hybrid execution model:
1. **Statistical Engines**: Hard math (SHAP, KS-Tests, Counterfactuals).
2. **Generative AI Reasoner**: LLMs (Gemini/OpenAI) to translate math into human-readable RCA reports.
3. **Semantic Memory (RAG)**: Pinecone vector database to recall past incidents.

---

## 2. Tech Stack
- **Backend Framework**: Python 3.13, FastAPI, Pydantic
- **Database**: SQLite (Local development via `rca_local.db`) / PostgreSQL (Production ready via SQLAlchemy)
- **Vector Database**: Pinecone (for RAG memory)
- **Machine Learning**: XGBoost (Baseline Model), SHAP (Explainability), Pandas, NumPy, Scikit-learn
- **Frontend**: React 18, Vite, TailwindCSS (Vanilla CSS logic), Recharts
- **Testing**: Pytest with FastAPI `TestClient`

---

## 3. Backend Architecture & Engines (`backend/app/engines/`)

The core intelligence of the app lives in modular "Engines":

### A. Data Integrity Checker (`integrity_checker.py`)
- **Purpose**: Checks raw data quality before advanced math runs.
- **Implementation**: Computes missing value percentages, type mismatches, and out-of-bounds metrics based on training statistics.

### B. Drift Detector (`drift_detector.py`)
- **Purpose**: Detects statistical shifts in the data distribution.
- **Implementation**: Uses Two-Sample Kolmogorov-Smirnov (KS) tests for numerical features and Population Stability Index (PSI) for categorical features. Detects "Concept Drift" by tracking accuracy decay over time.

### C. Failure Simulator (`failure_simulator.py`)
- **Purpose**: Injects controlled anomalies to test the observability system.
- **Implementation**: Simulates 6 failure modes:
  1. `noise`: Injects Gaussian noise into a specific feature.
  2. `drop`: Zeroes out a feature.
  3. `skew`: Shifts the distribution mean of a feature.
  4. `missing`: Replaces feature values with NaNs.
  5. `interaction`: Destroys the correlation between two previously correlated features.
  6. `concept`: Flips target labels to simulate real-world concept drift without changing input features.

### D. RCA Engine (`rca_engine.py`)
- **Purpose**: The core deterministic brain that aggregates signals.
- **Implementation**: 
  - **SHAP Analysis**: Computes SHAP values to find which features drove predictions.
  - **Bounded Counterfactuals**: Takes the top anomalous feature, perturbs it back to its historical baseline, and checks if the prediction flips back to normal. If it does, true causality is proven.
  - **Interaction Detection**: Looks for joint-distribution failures.
  - **Confidence Scoring**: Synthesizes Integrity + Drift + SHAP + Counterfactuals into a 0.0 to 1.0 confidence score.

### E. LLM Reasoner (`llm_reasoner.py`)
- **Purpose**: Translates the JSON dictionary from the RCA Engine into English.
- **Implementation**: Uses `google-genai` (Gemini) or `openai` SDK. Takes the statistical report, drift report, and integrity report and asks the LLM to output a JSON containing `explanation` and `suggested_fix`.

### F. Vector Memory (`vector_memory.py`)
- **Purpose**: Gives the system historical context (RAG).
- **Implementation**: Uses the Pinecone Python SDK. Every time a new RCA is completed, it serializes the drift report and SHAP profile into a text string, generates an embedding (via LLM), and stores it. On new anomalies, it queries Pinecone for the top 3 most similar past cases.

### G. Ablation Runner (`ablation_runner.py`)
- **Purpose**: A scientific validation harness.
- **Implementation**: Runs a matrix of 12 failure scenarios against 4 engine configurations (Baseline, Intermediate, Advanced, Full). Tracks "Hit Rate" to prove mathematically that adding counterfactuals and RAG memory improves diagnostic accuracy.

---

## 4. API Endpoints (`backend/app/main.py`)

1. `POST /ingest`: Ingests live predictions. Calculates adaptive anomaly scores (difference between prediction prob and baseline rate). Stores in `PredictionLog` DB table.
2. `POST /rca`: Runs the full 6-step RCA pipeline (Integrity -> Drift -> Memory Search -> Multi-Signal RCA -> LLM Reasoning -> Vector Storage). Has `lightweight` (fast) and `deep` (thorough) modes.
3. `POST /simulate`: Uses `failure_simulator` to generate bad data, then optionally runs `/rca` on it.
4. `POST /ablation`: Runs the ablation study.
5. `GET /metrics`: Returns rolling 24-hour metrics (accuracy, F1, anomaly counts).
6. `GET /rca/history`: Retrieves past RCA reports from the database.
7. `POST /feedback`: Allows humans to rate an RCA as "accurate" or "rejected".
8. `GET /health`: Checks system status.

**Note on Serialization**: The API includes a custom `_safe()` recursive serializer to convert `numpy.bool_`, `numpy.int64`, and `NaN` values into standard Python primitives (or `None`) to prevent JSON serialization crashes.

---

## 5. Database Schema (`backend/app/database.py`)
- `PredictionLog`: Stores individual predictions, inputs, and anomaly scores.
- `RCALog`: Stores completed root cause analyses, reasoning chains, LLM explanations, confidence scores, and human feedback.
- `WeightEvolutionLog`: Tracks how the internal engine weights (e.g., weighting SHAP vs. Drift) evolve based on human feedback.

---

## 6. Frontend Architecture (`frontend/`)
- Built with React + Vite.
- Styled with TailwindCSS (mostly custom CSS in `index.css` via glassmorphism design tokens).
- **Core Components**:
  - `Dashboard.jsx`: Top-level metrics and a quick list of recent high-severity anomalies.
  - `Simulator.jsx`: UI to trigger the 6 failure injection modes and see immediate RCA results.
  - `RCADetail.jsx`: Deep-dive view of a single RCA. Shows the "Reasoning Chain" progression bar, feature importance tables, LLM-generated explanations, and feedback buttons.
  - `RCAHistory.jsx`: Data table of past cases.
  - `Ablation.jsx`: Visualizes the results of the ablation study using Recharts (progress bars for accuracy improvements across configurations).

---

## 7. Model Details (`model/`)
- The baseline model is an `XGBClassifier` trained on a synthetic credit scoring dataset (`baseline_data.csv`).
- It is saved via `joblib` (`baseline_model.joblib`).
- Training statistics (feature means, std devs, baseline default rates) are saved in `training_stats.json` for the backend engines to use as a reference point for drift and integrity checks.

---

## 8. Testing (`backend/tests/test_api.py`)
- Comprehensive test suite using `pytest`.
- Fully mocks API requests via FastAPI's `TestClient`.
- Covers all 6 simulation modes and validates schema responses, specifically ensuring `NaN` outputs from math operations don't break JSON encoding.
