# AI Root Cause Analyzer

A research prototype for investigating failures of a **synthetic loan default classifier**. It accepts a batch of tabular records, compares their features with a training baseline, checks data quality, calculates model attributions, and ranks possible explanations. The React UI exposes a simulator, history, and a controlled ablation exercise.

**Status:** portfolio demo. The training data and injected failures are synthetic. Diagnostic scores have not been calibrated against independent incident labels. This is not a production monitoring or lending decision system.

## What is implemented

| Component | Behavior | Boundary |
| --- | --- | --- |
| Model | XGBoost trained on 5,000 generated loan records | No external credit dataset or field validation |
| Integrity and drift | Missing, duplicate, schema and range checks; KS-based distribution comparisons | Batch checks, not streaming monitoring |
| RCA | SHAP feature ranking, median feature substitutions, feature-pair tests | Substitution reveals model sensitivity, not real-world causality |
| Explanations | Gemini 2.0 Flash with GPT-4o-mini fallback | Without credentials, returns deterministic text |
| Case memory | Optional Pinecone integrated-inference index | Disabled without configured credentials and index |
| Persistence | SQLAlchemy with local SQLite by default | No user accounts or tenancy |
| UI | React 19, Vite 8, Recharts | Illustrations in `docs/screenshots` are SVG mockups |

The request path is: `POST /rca` → schema validation → model predictions → integrity and drift reports → optional Pinecone lookup → RCA scoring → optional LLM explanation → SQL log → optional case storage. The API also provides synthetic failure injection, feedback, and a staged ablation runner.

The weighted `confidence_score` is a **diagnostic score**. It is not a calibrated probability that a root-cause assertion is correct.

## Run locally

**Prerequisites:** Python 3.10 (the CI version), Node 22+, npm. The project trains a small deterministic synthetic model before starting the API.

```bash
git clone https://github.com/splash0047/Ai-Root-causeAnalyzer.git
cd Ai-Root-causeAnalyzer
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
python model/train_baseline.py
cd backend
uvicorn app.main:app --reload --port 8000
```

In another terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open `http://localhost:5173`, `http://localhost:8000/docs`, or `http://localhost:8000/health`. A working API health response has `model_loaded: true`; without the generated model it reports `degraded`.

Alternatively, `docker compose up --build` builds the synthetic baseline into the backend image and serves the frontend at `http://localhost:5173`. The Compose demo persists its SQLite database in the `rca_data` volume. It has no Postgres or Redis service.

### Configuration

Defaults allow the local synthetic demo. Copy `backend/.env.example` to `backend/.env` only if you need to set optional services or a different database. Do not commit real keys.

| Variable | Use |
| --- | --- |
| `DATABASE_URL` | SQLAlchemy URL; defaults to local SQLite |
| `CORS_ORIGINS` | Comma-separated frontend origins; defaults to localhost:5173 |
| `APP_ENV` | Set to `production` to fail when a configured Postgres connection is unavailable instead of silently falling back to SQLite |
| `GEMINI_API_KEY`, `OPENAI_API_KEY` | Optional explanation providers |
| `PINECONE_API_KEY`, `PINECONE_INDEX_NAME` | Optional integrated-inference index configured with a `diagnosis_text` source field |
| `VITE_API_BASE_URL` | Frontend build-time API URL; defaults to localhost:8000 |

The API currently has no authentication. Do not expose it to untrusted users; put access controls and request limits at a gateway before using a publicly reachable deployment. The GitHub Pages workflow builds the frontend, but it needs a reachable API URL for interactive use. A successful deployment workflow does not establish backend runtime availability.

## Example: investigate a batch

Send `POST /rca` with `records` containing exactly the nine numeric model features: `age`, `income`, `credit_score`, `loan_amount`, `employment_years`, `num_credit_lines`, `debt_to_income`, `has_mortgage`, `loan_purpose_encoded`. `actuals` is optional but, if supplied, must contain one binary label per record. The maximum batch size is 1,000. Select `mode: "lightweight"` for integrity and drift only, or `"deep"` for SHAP and model sensitivity checks. See `/docs` for request schemas and response examples.

| Endpoint | Purpose |
| --- | --- |
| `POST /ingest` | Store predictions and optional labels |
| `POST /rca` | Analyze and store a batch diagnosis |
| `GET /metrics` | Model metrics over ingested records with labels |
| `GET /rca/history` | Recent diagnoses |
| `POST /feedback` | Mark a diagnosis accurate or rejected |
| `POST /simulate` | Generate one of six synthetic failure types |
| `POST /simulate/fix` | Compare model outputs under a local feature substitution, optionally against supplied labels |
| `POST /ablation` | Compare four engine configurations on twelve injected failures |
| `GET /eval/metrics/eval` | Feedback counts and agreement; unsupported metrics are unavailable |
| `GET /health` | Model and optional-service status |

## What the ablation measures

`POST /ablation?n_samples=50` uses twelve deterministic synthetic injections. For each configuration, a *hit* means the injected feature appears in the top three ranked features; for a label flip it checks the broad concept-drift diagnosis. The denominator is twelve scenarios. This is a **top-three synthetic hit rate**, not field RCA accuracy.

The configurations enable distinct stages: (1) integrity and drift, (2) SHAP, (3) median-substitution model sensitivity, (4) interaction tests. Pinecone and LLM explanations are **not** part of this comparison. The endpoint reports per-scenario results, errors, component flags, and engine-only elapsed time. No false-positive rate can be inferred because this suite contains no unmodified control incidents. Do not compare its timings with full request latency.

Run and save the complete report from the backend directory:

```bash
python -m scripts.run_synthetic_ablation --n-samples 50 --output ../docs/evaluation/my-ablation.json
```

An [example raw report](docs/evaluation/synthetic-ablation-example.json) records package versions, input hashes, all 48 scenario/configuration outcomes, and the execution environment. In that Python 3.12 run, the four configurations hit **10/12, 9/12, 9/12, and 9/12** scenarios respectively. The added stages did not improve this metric. Results may differ with dependency versions, and the generated baseline includes the same distribution used by the simulator. These results are not external validation.

## Verification

```bash
python model/train_baseline.py
cd backend
python -m pytest tests/ -v
cd ../frontend
npm ci
npm run lint
npm run build
```

GitHub Actions performs model training, backend tests, and a frontend build. Tests cover API happy paths and selected validation failures. Optional external provider integrations and a deployed production stack require separate verification.

## Known limitations

- No independent incident corpus, verified root-cause labels, measured time savings, or calibrated RCA confidence.
- An observed model prediction flip under a synthetic feature substitution is **not** proof of a causal relationship.
- Optional LLM and memory services can fail or be disabled without preventing the statistical demo from running.
- The implementation uses blocking model and provider work in request handlers; large-scale serving, authentication, schema migrations, and long-running job orchestration remain future engineering work.
- SVGs under `docs/screenshots` illustrate intended screens; they are not captured from a deployed session.

For a credible next evaluation, collect held-out failure and no-failure incidents with independently reviewed causes, freeze the scoring protocol, compare distinct engine configurations, and report counts, uncertainty, latency distributions, and negative results.
