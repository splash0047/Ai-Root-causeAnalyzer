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

## 📝 License

MIT
