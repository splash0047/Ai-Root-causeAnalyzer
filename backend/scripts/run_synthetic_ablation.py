"""Write a reproducible, clearly scoped synthetic diagnosis benchmark report."""
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from app.config import settings
from app.engines.ablation_runner import AblationRunner


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 10 <= args.n_samples <= 1000:
        parser.error("--n-samples must be between 10 and 1000")
    if not settings.BASELINE_MODEL_PATH.exists() or not settings.BASELINE_DATA_PATH.exists():
        parser.error("Train the synthetic baseline first: python model/train_baseline.py")

    result = AblationRunner().run(n_samples=args.n_samples)
    report = {
        "scope": "Synthetic in-distribution injection; 12 expected feature/concept diagnoses; no control incidents",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "xgboost", "shap")},
        },
        "inputs": {
            "n_samples_per_scenario": args.n_samples,
            "model_sha256": sha256(settings.BASELINE_MODEL_PATH),
            "baseline_sha256": sha256(settings.BASELINE_DATA_PATH),
            "training_stats_sha256": sha256(settings.TRAINING_STATS_PATH),
            "generator_seed": 42,
            "simulator_seed": 99,
        },
        "result": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote synthetic evaluation to {args.output}")


if __name__ == "__main__":
    main()
