# 🔍 AI Root Cause Analyzer (RCA)

An industry-grade AI-powered Root Cause Analysis system for ML model monitoring. Goes beyond basic anomaly detection to provide **actionable, causal diagnostics** with multi-signal reasoning.

## 🏗️ Architecture

```
RCA/
├── backend/          # FastAPI application
│   ├── app/
│   │   ├── main.py           # API endpoints
│   │   ├── config.py         # Centralized settings
│   │   ├── database.py       # SQLAlchemy ORM
│   │   └── engines/
│   │       ├── rca_engine.py         # Core RCA diagnostic engine
│   │       ├── drift_detector.py     # KS-test & PSI drift detection
│   │       ├── integrity_checker.py  # Data quality validation
│   │       └── failure_simulator.py  # Controlled failure injection
│   ├── requirements.txt
│   └── .env.example
├── model/            # Baseline model & training
│   └── train_baseline.py
├── infra/            # Infrastructure
│   └── init.sql
└── docker-compose.yml
```

## 🚀 Key Features

- **Multi-Signal RCA Engine**: Combines drift detection, SHAP analysis, bounded counterfactual causality, and feature interaction testing
- **Bounded Counterfactual Causality**: Reverts suspect features to training-baseline values to confirm causal links
- **Feature Interaction Detection**: Top-k bounded pairwise interaction testing
- **Adaptive Anomaly Scoring**: Flags high-anomaly predictions for priority RCA
- **Active Feedback Loop**: User feedback adjusts diagnostic weights (bounded ±5% max delta)
- **Controlled Failure Simulation**: 6 failure types (noise, drop, skew, interaction, concept drift, missing values)
- **Tiered Execution Modes**: Lightweight vs Deep RCA for cost control

## 🛠️ Setup

### Prerequisites
- Python 3.10+
- Docker & Docker Compose

### 1. Start Infrastructure
```bash
docker-compose up -d
```

### 2. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 3. Train Baseline Model
```bash
cd model
python train_baseline.py
```

### 4. Run the API Server
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

### 5. Access API Docs
Open [http://localhost:8000/docs](http://localhost:8000/docs)

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ingest` | Ingest prediction data with anomaly scoring |
| `POST` | `/rca` | Execute Root Cause Analysis |
| `GET` | `/metrics` | Model performance metrics (rolling window) |
| `GET` | `/rca/history` | Historical RCA results |
| `POST` | `/feedback` | Submit feedback on RCA results |
| `POST` | `/simulate` | Run controlled failure simulation |
| `GET` | `/health` | System health check |

## 📄 License

MIT
