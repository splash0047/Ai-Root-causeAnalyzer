# 🔍 AI Root Cause Analyzer (RCA)

An industry-grade AI-powered Root Cause Analysis system for ML model monitoring. Goes beyond basic anomaly detection to provide **actionable, causal diagnostics** with multi-signal reasoning.

## 🏗️ Architecture

```
┌──────────────┐     ┌──────────────────────────────────────────────────┐
│  React UI    │────▶│  FastAPI Backend (8 endpoints)                   │
│  (Vite)      │     │  ├── /ingest      → Data ingestion + anomaly    │
│  Port: 5173  │     │  ├── /rca         → Full RCA pipeline           │
│              │     │  ├── /metrics     → Model performance metrics   │
│  Dashboard   │     │  ├── /rca/history → Historical RCA results      │
│  RCA History │     │  ├── /feedback    → Human feedback loop         │
│  Simulator   │     │  ├── /simulate   → Controlled failure injection │
│  Ablation    │     │  ├── /ablation   → Ablation study runner        │
│              │     │  └── /health     → System health check          │
└──────────────┘     └─────────┬────────────────┬──────────────────────┘
                               │                │
                    ┌──────────▼──────┐  ┌──────▼──────────┐
                    │  RCA Engine     │  │  LLM Reasoner   │
                    │  ├── SHAP       │  │  (Gemini/OpenAI)│
                    │  ├── Counter-   │  └──────┬──────────┘
                    │  │   factuals   │         │
                    │  ├── Drift      │  ┌──────▼──────────┐
                    │  ├── Integrity  │  │  Pinecone Vector│
                    │  └── Multi-sig  │  │  Memory (RAG)   │
                    └─────────────────┘  └─────────────────┘
```

## 📖 Project Motivation & Deep Dive

Traditional ML observability platforms often stop at anomaly detection (e.g., "Your model drifted at 2:00 PM"). This leaves data scientists with the manual, time-consuming task of digging through data to find the root cause. 

**This project solves the "Why?"** 
By combining statistical rigor (SHAP, KS-tests) with causal inference (Bounded Counterfactuals) and Generative AI, this system automatically diagnoses the exact failure mode and provides a human-readable remediation plan.

### The 6-Signal RCA Engine (How it Works)
Instead of relying on a single metric, the engine aggregates 6 distinct diagnostic signals to build a high-confidence causal chain:
1. **Data Integrity**: Hard rules for missingness, type mismatches, and out-of-bounds values.
2. **Distributional Drift**: Tracks covariate shift using two-sample Kolmogorov-Smirnov (KS) tests and Population Stability Index (PSI).
3. **Concept Drift**: Monitors target/prediction relationship decay over time.
4. **SHAP (Feature Importance)**: Identifies which features mathematically drove the anomalous predictions in the current batch.
5. **Bounded Counterfactuals**: The engine perturbs the top SHAP features within historical bounds to see if reversing the drift "fixes" the prediction. If yes, causality is established.
6. **Interaction Effects**: Detects hidden joint-distribution failures (e.g., Feature A is fine, Feature B is fine, but their combination is anomalous).

### RAG-Powered Diagnostic Memory
The system features a **Retrieval-Augmented Generation (RAG)** loop using Pinecone. Every time a root cause is confirmed, its statistical fingerprint (drift vectors, SHAP profiles) is embedded and stored. When a new anomaly occurs, the system queries Pinecone for semantically similar past incidents, allowing the LLM to say: *"This looks identical to the data pipeline outage we had last Tuesday."*

### The Ablation Study (Proving it Works)
To rigorously prove the engine's efficacy, we built an automated validation harness (`/ablation`). It runs 12 distinct failure scenarios (e.g., 30% noise on a critical feature, 100% missing values on a secondary feature) across 4 engine configurations:
- **Baseline**: SHAP only
- **Intermediate**: SHAP + Drift
- **Advanced**: SHAP + Drift + Counterfactuals + Interactions
- **Full**: All signals + Vector Memory

*The study definitively proves that multi-signal reasoning with counterfactuals significantly reduces false positives compared to standard SHAP analysis.*

## ✨ Key Features

### Backend (Python/FastAPI)
- **6-Signal RCA Engine**: SHAP importance, counterfactual validation, drift detection, integrity checks, interaction testing, multi-signal aggregation
- **Bounded Counterfactual Causality**: Validates causal claims by testing if reversing a feature drift restores predictions
- **LLM Reasoning** (Phase 4): Gemini/OpenAI generates human-readable explanations and suggested fixes
- **Pinecone Vector Memory** (Phase 4): Stores RCA cases as vectors for pattern matching across incidents
- **Failure Simulator**: 6 failure injection types (noise, drop, skew, interaction, concept drift, missing values)
- **Ablation Study** (Phase 6): Systematic validation across 12 scenarios × 4 configurations
- **Confidence Scoring**: Multi-factor confidence with memory boost and uncertainty flagging
- **Active Feedback Loop**: Human feedback adjusts RCA weight evolution over time

### Frontend (React/Vite)
- **Premium Dark UI**: Glassmorphism cards, gradient accents, Inter + JetBrains Mono typography
- **Dashboard**: Real-time metrics (accuracy, F1, accuracy drop, anomaly count) with time window selector
- **RCA History**: Filterable table with severity badges, confidence bars, feedback status
- **Failure Simulator**: Interactive controls for all 6 failure types with inline RCA results
- **RCA Detail**: Reasoning chain visualization, ranked features with causal badges, feedback buttons
- **Ablation Study**: Accuracy progression bars, per-scenario hit/miss detail table

## 🚀 Quick Start

### Prerequisites
- Python 3.13+ with pip
- Node.js 18+ with npm
- (Optional) PostgreSQL, Redis, Pinecone API key, Gemini API key

### Backend
```bash
cd backend
cp .env.example .env    # Add your API keys
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev             # Opens at http://localhost:5173
```

### Docker (Full Stack)
```bash
docker-compose up --build
```

## 📁 Project Structure

```
RCA/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app with 8 endpoints
│   │   ├── config.py            # Settings from .env
│   │   ├── database.py          # SQLAlchemy models (Postgres/SQLite)
│   │   └── engines/
│   │       ├── rca_engine.py       # Core 6-signal RCA analysis
│   │       ├── drift_detector.py   # Statistical drift detection
│   │       ├── integrity_checker.py # Data quality validation
│   │       ├── failure_simulator.py # 6 failure injection types
│   │       ├── llm_reasoner.py     # Gemini/OpenAI explanations
│   │       ├── vector_memory.py    # Pinecone case memory
│   │       └── ablation_runner.py  # Ablation study engine
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Shell with sidebar navigation
│   │   ├── api.js               # Centralized API client
│   │   ├── index.css            # Design system tokens
│   │   ├── App.css              # Component styles
│   │   └── components/
│   │       ├── Dashboard.jsx       # Metrics + recent RCA
│   │       ├── RCAHistory.jsx      # Filterable history table
│   │       ├── Simulator.jsx       # Failure injection UI
│   │       ├── RCADetail.jsx       # Reasoning chain + features
│   │       └── Ablation.jsx        # Ablation study results
│   └── package.json
├── model/
│   └── train_baseline.py       # XGBoost training script
├── infra/
│   └── init.sql                # PostgreSQL schema
├── docker-compose.yml
└── README.md
```

## 🔬 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/ingest` | Ingest data batch with anomaly detection |
| POST | `/rca` | Run full RCA analysis (lightweight/deep) |
| GET | `/metrics` | Model performance metrics |
| GET | `/rca/history` | Historical RCA results |
| POST | `/feedback` | Submit human feedback on RCA |
| POST | `/simulate` | Inject controlled failures |
| POST | `/ablation` | Run ablation study |
| GET | `/health` | System health check |

## 📊 RCA Pipeline (6 Steps)

1. **Integrity Check** → Missing values, range violations, type errors
2. **Drift Detection** → KS test, PSI, concept drift
3. **Vector Memory Search** → Find similar past cases in Pinecone
4. **Multi-Signal RCA** → SHAP + counterfactuals + interactions + drift
5. **LLM Reasoning** → Generate explanation + suggested fix
6. **Case Storage** → Store in vector memory for future matching

## 🧠 What This Project Actually Does (Simple)

Most ML monitoring tools tell you:
> "Your model performance dropped."

This system tells you:
> "Your model failed because the `credit_score` feature dropped by 30%, which caused incorrect predictions for users aged 40–60. This is likely due to an upstream data pipeline issue."

In short:
- Detects model issues ✅
- Finds the exact cause ✅
- Proves causality (not just correlation) ✅
- Suggests fixes automatically ✅

This turns ML debugging from manual guesswork into an automated diagnostic process.

## 🔄 End-to-End Flow (Real Scenario)

1. Data is sent to `/ingest`
2. System computes anomaly score
3. If anomaly detected → triggers RCA
4. RCA Engine runs:
   - Data Integrity Check
   - Drift Detection
   - SHAP Feature Analysis
   - Counterfactual Validation
5. System searches Pinecone for similar past failures
6. LLM generates explanation + suggested fix
7. Result is stored and shown in dashboard

Example Output:
- Root Cause: Drift in `income`
- Confidence: 0.87
- Explanation: Income distribution shifted causing misclassification
- Suggested Fix: Retrain model on updated distribution
## 🧪 How to Use the System (Step-by-Step)

### Step 1: Start Backend & Frontend
Run backend and frontend as described in Quick Start.

---

### Step 2: Generate Data
Use simulator:

POST /simulate

Example:
- Inject noise into `income`
- Drop `credit_score`
- Create concept drift

---

### Step 3: Run RCA
POST /rca

Options:
- `lightweight`: fast, low-cost analysis
- `deep`: full RCA with counterfactuals + memory + LLM

---

### Step 4: View Results
Go to UI:
- Dashboard → metrics
- RCA History → past failures
- RCA Detail → full reasoning chain

---

### Step 5: Provide Feedback
Mark RCA as:
- Accurate ✅
- Rejected ❌

This improves future analysis.

---

### Step 6: Run Ablation Study
POST /ablation

This validates:
- how each component improves accuracy

## ⚙️ Implementation Details

### RCA Engine Internals
- Uses SHAP to identify important features
- Applies KS-Test and PSI for drift detection
- Performs bounded counterfactual testing:
  - Modifies feature within valid range
  - Checks if prediction changes
  - Confirms causal relationship

### Confidence Score
Computed from:
- Drift magnitude
- Feature importance (SHAP)
- Counterfactual success
- Historical similarity (vector memory)

### Vector Memory (RAG)
- Stores past RCA cases
- Retrieves similar incidents
- Helps improve reasoning and confidence

### LLM Reasoning
- Converts structured RCA output into:
  - Explanation
  - Suggested fix

  ## ⚠️ Limitations

- Causality is model-based, not true causal inference
- Performance depends on quality of training data
- Vector memory improves over time but starts cold
- Counterfactuals may not capture complex real-world dependencies
- System currently optimized for structured/tabular data

Future improvements:
- Better causal modeling
- Streaming architecture
- Advanced feature interaction handling

## 📝 License

MIT
