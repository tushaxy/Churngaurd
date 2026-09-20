"""Unit tests for RFM calculations, segmentation mapping, and Pareto analysis."""

from datetime import datetime

import pandas as pd
import pytest

from src.churn.rfm import assign_rfm_segment, compute_rfm_features
from src.churn.segmentation import compute_pareto_revenue_share


@pytest.fixture
def controlled_purchases() -> pd.DataFrame:
    """Fixture providing known transaction records across distinct customer personas."""
    return pd.DataFrame(
        [
            # Customer 1: Champion persona (very recent, high frequency, high monetary)
            {
                "Invoice": "INV-101",
                "CustomerID": 1,
                "InvoiceDate": "2024-12-30 10:00:00",
                "TotalSpend": 100.0,
                "Country": "United Kingdom",
            },
            {
                "Invoice": "INV-102",
                "CustomerID": 1,
                "InvoiceDate": "2024-12-20 10:00:00",
                "TotalSpend": 150.0,
                "Country": "United Kingdom",
            },
            {
                "Invoice": "INV-103",
                "CustomerID": 1,
                "InvoiceDate": "2024-12-10 10:00:00",
                "TotalSpend": 250.0,
                "Country": "United Kingdom",
            },
            # Customer 2: Churned high-value customer (recency > 90 days)
            {
                "Invoice": "INV-201",
                "CustomerID": 2,
                "InvoiceDate": "2024-05-01 10:00:00",
                "TotalSpend": 500.0,
                "Country": "France",
            },
            # Customer 3: One-off small purchase long ago (Lost persona)
            {
                "Invoice": "INV-301",
                "CustomerID": 3,
                "InvoiceDate": "2024-01-01 10:00:00",
                "TotalSpend": 25.0,
                "Country": "Germany",
            },
            # Customer 4: New recent customer (1 recent order)
            {
                "Invoice": "INV-401",
                "CustomerID": 4,
                "InvoiceDate": "2024-12-28 10:00:00",
                "TotalSpend": 45.0,
                "Country": "United Kingdom",
            },
            # Customer 5: Steady repeat customer
            {
                "Invoice": "INV-501",
                "CustomerID": 5,
                "InvoiceDate": "2024-12-01 10:00:00",
                "TotalSpend": 80.0,
                "Country": "United Kingdom",
            },
        ]
    )


def test_rfm_metric_calculations(controlled_purchases: pd.DataFrame) -> None:
    """Verifies accuracy of recency, frequency, monetary, and derived AOV math."""
    ref_date = datetime(2024, 12, 31, 23, 59, 59)
    rfm = compute_rfm_features(controlled_purchases, snapshot_date=ref_date)

    # Customer 1
    c1 = rfm[rfm["CustomerID"] == 1].iloc[0]
    assert c1["frequency"] == 3
    assert c1["monetary"] == 500.0  # 100 + 150 + 250
    assert c1["avg_order_value"] == round(500.0 / 3, 2)
    assert c1["recency"] == 1  # 2024-12-31 minus 2024-12-30
    assert c1["churned"] == 0

    # Customer 2 (Churned: last purchase May 1, 2024 > 90 days before Dec 31)
    c2 = rfm[rfm["CustomerID"] == 2].iloc[0]
    assert c2["recency"] > 90
    assert c2["churned"] == 1
    assert c2["frequency"] == 1
    assert c2["monetary"] == 500.0


def test_segment_mapping_rules() -> None:
    """Verifies that quintile scores map to correct segment labels."""
    assert assign_rfm_segment(5, 5, 5) == "Champions"
    assert assign_rfm_segment(4, 4, 4) == "Champions"
    assert assign_rfm_segment(1, 5, 5) == "Can't Lose Them"
    assert assign_rfm_segment(4, 5, 3) == "Loyal Customers"
    assert assign_rfm_segment(1, 4, 3) == "At Risk"
    assert assign_rfm_segment(5, 2, 3) == "Potential Loyalist"
    assert assign_rfm_segment(5, 1, 2) == "New Customers"
    assert assign_rfm_segment(3, 1, 1) == "Promising"
    assert assign_rfm_segment(2, 2, 2) == "Need Attention"
    assert assign_rfm_segment(1, 1, 1) == "Lost"


def test_pareto_revenue_share_math() -> None:
    """Verifies Pareto share computation on a controlled distribution."""
    # 10 customers: top 2 have 65% of total wealth ($650 out of $1000)
    spends = [400.0, 250.0, 50.0, 50.0, 50.0, 40.0, 40.0, 40.0, 40.0, 40.0]
    df = pd.DataFrame({"CustomerID": range(1, 11), "monetary": spends})

    metrics = compute_pareto_revenue_share(df)
    assert metrics["total_customers"] == 10
    assert metrics["total_revenue"] == 1000.0
    assert metrics["top_20_count"] == 2
    assert metrics["top_20_revenue"] == 650.0
    assert metrics["top_20_revenue_share"] == 0.65
