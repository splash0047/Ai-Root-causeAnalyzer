"""
AI Root Cause Analyzer - Baseline Model Training
Trains an XGBoost classifier on a synthetic Loan Default dataset.
Stores the model, baseline data statistics, and training reference data.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier
import joblib

# ─── Output paths ──────────────────────────────────────────
MODEL_DIR = Path(__file__).parent
MODEL_PATH = MODEL_DIR / "baseline_model.joblib"
BASELINE_DATA_PATH = MODEL_DIR / "baseline_data.csv"
TRAINING_STATS_PATH = MODEL_DIR / "training_stats.json"


def generate_loan_dataset(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic Loan Default dataset.
    Features mirror real-world credit risk variables.
    """
    np.random.seed(seed)

    data = pd.DataFrame({
        "age": np.random.randint(21, 70, n_samples),
        "income": np.random.lognormal(mean=10.8, sigma=0.5, size=n_samples).astype(int),
        "credit_score": np.random.normal(680, 80, n_samples).clip(300, 850).astype(int),
        "loan_amount": np.random.lognormal(mean=10, sigma=0.7, size=n_samples).astype(int),
        "employment_years": np.random.exponential(scale=5, size=n_samples).clip(0, 40).round(1),
        "num_credit_lines": np.random.poisson(lam=4, size=n_samples),
        "debt_to_income": np.random.uniform(0.05, 0.8, n_samples).round(3),
        "has_mortgage": np.random.binomial(1, 0.4, n_samples),
        "loan_purpose": np.random.choice(
            ["debt_consolidation", "home_improvement", "business", "education", "medical"],
            n_samples,
            p=[0.35, 0.25, 0.15, 0.15, 0.10]
        ),
    })

    # Encode categorical
    data["loan_purpose_encoded"] = data["loan_purpose"].map({
        "debt_consolidation": 0, "home_improvement": 1,
        "business": 2, "education": 3, "medical": 4
    })

    # Generate realistic default probability based on features
    default_prob = (
        -0.02 * data["credit_score"] / 850
        + 0.3 * data["debt_to_income"]
        + 0.1 * (data["loan_amount"] / data["income"]).clip(0, 5)
        - 0.01 * data["employment_years"]
        + 0.05 * (data["age"] < 30).astype(int)
        + np.random.normal(0, 0.1, n_samples)
    )

    # Normalize to probability range
    default_prob = (default_prob - default_prob.min()) / (default_prob.max() - default_prob.min())
    data["default"] = (default_prob > 0.55).astype(int)

    return data


def train_baseline_model():
    """Train and save the baseline XGBoost model with full statistics."""
    print("=" * 60)
    print("  AI RCA - Baseline Model Training")
    print("=" * 60)

    # Generate dataset
    print("\n[1/4] Generating synthetic Loan Default dataset...")
    data = generate_loan_dataset(n_samples=5000)
    print(f"      Dataset shape: {data.shape}")
    print(f"      Default rate: {data['default'].mean():.2%}")

    # Feature columns (excluding target and raw categorical)
    feature_cols = [
        "age", "income", "credit_score", "loan_amount",
        "employment_years", "num_credit_lines", "debt_to_income",
        "has_mortgage", "loan_purpose_encoded"
    ]

    X = data[feature_cols]
    y = data["default"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train XGBoost
    print("\n[2/4] Training XGBoost classifier...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n[3/4] Evaluation Results:")
    print(f"      Accuracy: {accuracy:.4f}")
    print(f"      F1 Score: {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"\n[4/4] Model saved to: {MODEL_PATH}")

    # Save baseline data for drift comparison
    data.to_csv(BASELINE_DATA_PATH, index=False)
    print(f"      Baseline data saved to: {BASELINE_DATA_PATH}")

    # Save training statistics (distributions for drift detection)
    training_stats = {
        "feature_columns": feature_cols,
        "sample_count": len(X_train),
        "default_rate": float(y_train.mean()),
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "feature_stats": {}
    }

    for col in feature_cols:
        col_data = X_train[col]
        training_stats["feature_stats"][col] = {
            "mean": float(col_data.mean()),
            "std": float(col_data.std()),
            "min": float(col_data.min()),
            "max": float(col_data.max()),
            "median": float(col_data.median()),
            "q25": float(col_data.quantile(0.25)),
            "q75": float(col_data.quantile(0.75)),
        }

    with open(TRAINING_STATS_PATH, "w") as f:
        json.dump(training_stats, f, indent=2)
    print(f"      Training stats saved to: {TRAINING_STATS_PATH}")

    print("\n" + "=" * 60)
    print("  Baseline model training complete!")
    print("=" * 60)

    return model, data, training_stats


if __name__ == "__main__":
    train_baseline_model()
