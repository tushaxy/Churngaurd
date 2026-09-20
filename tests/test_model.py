"""Unit tests for machine learning pipeline, threshold optimization, and SHAP explainability."""

import numpy as np

from src.churn.explain import get_risk_band
from src.churn.train import build_preprocessor, optimize_revenue_at_risk_threshold


def test_get_risk_band() -> None:
    """Verifies churn probability maps to standard risk tiers."""
    assert get_risk_band(0.15) == "Low"
    assert get_risk_band(0.45) == "Medium"
    assert get_risk_band(0.72) == "High"
    assert get_risk_band(0.92) == "Critical"


def test_optimize_revenue_at_risk_threshold() -> None:
    """Verifies threshold optimization algorithm searches and minimizes target ratio difference."""
    n_samples = 200
    np.random.seed(42)
    y_test = np.random.binomial(1, 0.20, size=n_samples)
    test_probs = np.random.uniform(0.05, 0.95, size=n_samples)
    test_monetary = np.random.uniform(50.0, 500.0, size=n_samples)

    best_thresh, best_metrics, tradeoff_curve = optimize_revenue_at_risk_threshold(
        y_test=y_test,
        test_probs=test_probs,
        test_monetary=test_monetary,
        target_ratio=0.07,
    )

    assert 0.10 <= best_thresh <= 0.95
    assert "revenue_at_risk" in best_metrics
    assert "revenue_at_risk_ratio" in best_metrics
    assert len(tradeoff_curve) > 50


def test_build_preprocessor() -> None:
    """Verifies ColumnTransformer builds without errors."""
    preprocessor = build_preprocessor()
    transformers = [t[0] for t in preprocessor.transformers]
    assert "num" in transformers
    assert "cat" in transformers


def test_explain_customer_prediction() -> None:
    """Verifies that explain_customer_prediction produces complete payload."""
    from src.churn.explain import explain_customer_prediction

    sample = {
        "frequency": 8,
        "monetary": 450.0,
        "avg_order_value": 56.25,
        "tenure_days": 240,
        "orders_per_month": 1.01,
        "return_rate": 0.02,
        "rfm_score": 444,
        "segment": "Loyal Customers",
        "kmeans_cluster": "High-Value Loyalists",
    }
    result = explain_customer_prediction(sample)
    assert "churn_probability" in result
    assert "risk_band" in result
    assert "revenue_at_risk" in result
    assert "recommended_action" in result
    assert "top_drivers" in result
    assert isinstance(result["top_drivers"], list)
