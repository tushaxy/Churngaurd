"""Inference and batch scoring engine for ChurnGuard."""

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.churn.config import settings
from src.churn.explain import get_risk_band
from src.churn.utils import get_logger, load_json

logger = get_logger(__name__)


def load_model_pipeline() -> Pipeline:
    """Loads fitted scikit-learn model pipeline from disk.

    Returns:
        Fitted Pipeline instance.
    """
    model_path = Path(settings.paths.model_file)
    if not model_path.exists():
        alt_path = Path(__file__).resolve().parent.parent.parent / settings.paths.model_file
        if alt_path.exists():
            model_path = alt_path
        else:
            raise FileNotFoundError(f"Model file not found at: {model_path}")
    return joblib.load(model_path)


def load_optimal_threshold() -> float:
    """Loads calibrated optimal decision threshold from threshold.json.

    Returns:
        Optimal threshold float (default 0.50 if file not found).
    """
    threshold_path = Path(settings.paths.threshold_file)
    if not threshold_path.exists():
        alt_path = Path(__file__).resolve().parent.parent.parent / settings.paths.threshold_file
        if alt_path.exists():
            threshold_path = alt_path
        else:
            return 0.50

    try:
        data = load_json(threshold_path)
        return float(data.get("optimal_threshold", 0.50))
    except Exception:
        return 0.50


def score_customers(
    df: pd.DataFrame,
    pipeline: Pipeline | None = None,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Scores customer records with churn probability, risk band, and revenue at risk.

    Args:
        df: DataFrame containing customer features.
        pipeline: Optional pre-loaded pipeline model.
        threshold: Optional decision threshold override.

    Returns:
        DataFrame enriched with prediction outputs.
    """
    from src.churn.train import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

    model = pipeline or load_model_pipeline()
    tau = threshold if threshold is not None else load_optimal_threshold()

    scored = df.copy()

    # Ensure required feature columns exist
    missing_cols = [c for c in NUMERICAL_FEATURES + CATEGORICAL_FEATURES if c not in scored.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns for scoring: {missing_cols}")

    feature_matrix = scored[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    probs = model.predict_proba(feature_matrix)[:, 1]

    scored["churn_probability"] = np.round(probs, 4)
    scored["churn_prediction"] = (probs >= tau).astype(int)
    scored["risk_band"] = [get_risk_band(p) for p in probs]

    monetary_series = scored["monetary"] if "monetary" in scored.columns else 0.0
    scored["revenue_at_risk"] = np.round(monetary_series * scored["churn_probability"], 2)
    scored["scored_at"] = datetime.now().isoformat()

    return scored
