"""Unit tests for pipeline scoring and SQLite database persistence."""

import sqlite3
from pathlib import Path

import pandas as pd

from src.churn.config import settings
from src.churn.predict import load_optimal_threshold, score_customers


def test_load_optimal_threshold() -> None:
    """Verifies that optimal decision threshold loads properly."""
    threshold = load_optimal_threshold()
    assert isinstance(threshold, float)
    assert 0.10 <= threshold <= 0.95


def test_score_customers() -> None:
    """Verifies batch scoring function on customer feature sample."""
    sample_df = pd.DataFrame(
        [
            {
                "CustomerID": 99901,
                "frequency": 12,
                "monetary": 1250.0,
                "avg_order_value": 104.17,
                "tenure_days": 400,
                "orders_per_month": 0.91,
                "return_rate": 0.01,
                "rfm_score": 555,
                "segment": "Champions",
                "kmeans_cluster": "High-Value Loyalists",
            },
            {
                "CustomerID": 99902,
                "frequency": 1,
                "monetary": 25.0,
                "avg_order_value": 25.0,
                "tenure_days": 180,
                "orders_per_month": 0.17,
                "return_rate": 0.0,
                "rfm_score": 111,
                "segment": "Lost",
                "kmeans_cluster": "Dormant / High Churn Risk",
            },
        ]
    )

    scored = score_customers(sample_df)
    assert "churn_probability" in scored.columns
    assert "churn_prediction" in scored.columns
    assert "risk_band" in scored.columns
    assert "revenue_at_risk" in scored.columns
    assert len(scored) == 2


def test_sqlite_persistence() -> None:
    """Verifies that SQLite database contains customers and predictions tables with valid records."""
    db_path = Path(settings.paths.sqlite_path)
    assert db_path.exists(), "SQLite database file does not exist."

    conn = sqlite3.connect(db_path)
    try:
        customers_count = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        predictions_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

        assert customers_count > 0, "Customers table is empty."
        assert predictions_count > 0, "Predictions table is empty."
        assert (
            customers_count == predictions_count
        ), "Counts between customers and predictions mismatch."
    finally:
        conn.close()


def test_score_customers_missing_columns_auto_imputed() -> None:
    """Verifies score_customers handles raw batches missing engineered features."""
    raw_batch = pd.DataFrame(
        [
            {"CustomerID": 90001, "frequency": 5, "monetary": 500.0},
            {"CustomerID": 90002, "frequency": 1, "monetary": 30.0},
        ]
    )
    scored = score_customers(raw_batch)
    assert len(scored) == 2
    assert "churn_probability" in scored.columns
    assert "risk_band" in scored.columns
    assert "revenue_at_risk" in scored.columns
    assert (scored["churn_probability"] >= 0.0).all()
