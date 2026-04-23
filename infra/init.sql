-- =============================================================
-- AI Root Cause Analyzer - Database Schema Initialization
-- =============================================================

-- Prediction Logs: Tracks every inference event
CREATE TABLE IF NOT EXISTS prediction_logs (
    id              SERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    input_features  JSONB NOT NULL,
    prediction      FLOAT NOT NULL,
    actual          FLOAT,
    model_version   VARCHAR(50) NOT NULL DEFAULT 'v1.0',
    anomaly_score   FLOAT DEFAULT 0.0,
    batch_id        VARCHAR(100)
);

-- RCA Logs: Tracks every RCA diagnosis
CREATE TABLE IF NOT EXISTS rca_logs (
    id                  SERIAL PRIMARY KEY,
    timestamp           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    root_cause          VARCHAR(200) NOT NULL,
    confidence_score    FLOAT NOT NULL,
    severity            VARCHAR(20) NOT NULL DEFAULT 'medium',
    ranked_features     JSONB,
    explanation         TEXT,
    suggested_fix       TEXT,
    reasoning_chain     JSONB,
    rca_mode            VARCHAR(20) NOT NULL DEFAULT 'deep',
    rca_logic_version   VARCHAR(50) NOT NULL DEFAULT 'v1.0',
    model_version       VARCHAR(50) NOT NULL DEFAULT 'v1.0',
    dataset_snapshot_id VARCHAR(100),
    user_feedback       VARCHAR(20),
    feedback_notes      TEXT,
    is_uncertain        BOOLEAN DEFAULT FALSE
);

-- Weight Evolution Log: Tracks dynamic weight adjustments from feedback
CREATE TABLE IF NOT EXISTS weight_evolution_log (
    id              SERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    rca_log_id      INTEGER REFERENCES rca_logs(id),
    previous_weights JSONB NOT NULL,
    updated_weights  JSONB NOT NULL,
    adjustment_reason TEXT
);

-- Indexes for performance
CREATE INDEX idx_prediction_logs_timestamp ON prediction_logs(timestamp);
CREATE INDEX idx_prediction_logs_batch ON prediction_logs(batch_id);
CREATE INDEX idx_rca_logs_timestamp ON rca_logs(timestamp);
CREATE INDEX idx_rca_logs_severity ON rca_logs(severity);
CREATE INDEX idx_rca_logs_feedback ON rca_logs(user_feedback);
