"""RFM feature engineering and rule-based customer segmentation module."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.churn.config import settings
from src.churn.utils import get_logger

logger = get_logger(__name__)


def assign_rfm_segment(r: int, f: int, m: int) -> str:
    """Maps RFM quintile scores (1-5) to industry standard customer segments.

    Args:
        r: Recency score (1=least recent, 5=most recent).
        f: Frequency score (1=least frequent, 5=most frequent).
        m: Monetary score (1=least spend, 5=most spend).

    Returns:
        Segment name string.
    """
    # 1. Champions: bought recently, buy often and spend the most
    if r in (4, 5) and f in (4, 5) and m in (4, 5):
        return "Champions"

    # 2. Can't Lose Them: made biggest purchases and often, but haven't returned recently
    if r == 1 and f in (4, 5) and m in (4, 5):
        return "Can't Lose Them"

    # 3. Loyal Customers: buy regularly with consistent spending
    if r in (3, 4, 5) and f in (3, 4, 5):
        return "Loyal Customers"

    # 4. At Risk: past high-value customers showing significant signs of churn
    if r in (1, 2) and (f in (3, 4, 5) or m in (4, 5)):
        return "At Risk"

    # 5. Potential Loyalist: recent shoppers with moderate purchase frequency
    if r in (4, 5) and f in (2, 3):
        return "Potential Loyalist"

    # 6. New Customers: high recency with low frequency
    if r in (4, 5) and f == 1:
        return "New Customers"

    # 7. Promising: recent buyers with initial engagement
    if r in (3, 4) and f <= 2:
        return "Promising"

    # 8. Need Attention: mid-range recency, frequency, and spend
    if r in (2, 3) and f in (2, 3):
        return "Need Attention"

    # 9. Lost: lowest recency, lowest frequency, and lowest spend
    if r == 1 and f == 1 and m == 1:
        return "Lost"

    # 10. Hibernating: low recency and frequency
    if r in (1, 2) and f in (1, 2):
        return "Hibernating"

    return "Need Attention"


def compute_rfm_features(
    purchases_df: pd.DataFrame,
    cancellations_df: pd.DataFrame | None = None,
    snapshot_date: datetime | None = None,
) -> pd.DataFrame:
    """Computes RFM metrics and derived behavioural features per customer.

    Args:
        purchases_df: Cleaned positive purchase transactions.
        cancellations_df: Optional DataFrame of cancelled transactions.
        snapshot_date: Analysis reference date. Defaults to max(InvoiceDate) + 1 day.

    Returns:
        Customer-level DataFrame containing RFM features, derived attributes, and segments.
    """
    logger.info("Computing RFM features for %d transactions...", len(purchases_df))

    # Ensure datetime format
    purchases = purchases_df.copy()
    purchases["InvoiceDate"] = pd.to_datetime(purchases["InvoiceDate"])

    if snapshot_date is None:
        ref_date = purchases["InvoiceDate"].max() + pd.Timedelta(days=1)
    else:
        ref_date = pd.to_datetime(snapshot_date)

    logger.info("RFM Analysis Snapshot Date: %s", ref_date.strftime("%Y-%m-%d"))

    # Base RFM Aggregations
    rfm = (
        purchases.groupby("CustomerID")
        .agg(
            recency_days=("InvoiceDate", lambda x: (ref_date - x.max()).days),
            frequency=("Invoice", "nunique"),
            monetary=("TotalSpend", "sum"),
            first_purchase=("InvoiceDate", "min"),
            last_purchase=("InvoiceDate", "max"),
            country=("Country", "first"),
        )
        .reset_index()
    )

    # Derived Features
    rfm["recency"] = rfm["recency_days"]
    rfm["avg_order_value"] = np.round(rfm["monetary"] / np.maximum(rfm["frequency"], 1), 2)
    rfm["tenure_days"] = (ref_date - rfm["first_purchase"]).dt.days
    rfm["orders_per_month"] = np.round(
        rfm["frequency"] / (np.maximum(rfm["tenure_days"], 15) / 30.44), 2
    )

    # Calculate cancellation/return rate per customer
    if cancellations_df is not None and not cancellations_df.empty:
        cancels = cancellations_df.copy()
        cancel_counts = cancels.groupby("CustomerID")["Invoice"].nunique().rename("cancellations")
        rfm = rfm.merge(cancel_counts, on="CustomerID", how="left")
        rfm["cancellations"] = rfm["cancellations"].fillna(0).astype(int)
        rfm["return_rate"] = np.round(
            rfm["cancellations"] / (rfm["frequency"] + rfm["cancellations"]), 4
        )
    else:
        rfm["cancellations"] = 0
        rfm["return_rate"] = 0.0

    # Ensure monetary spend is non-negative and round
    rfm["monetary"] = np.round(np.maximum(rfm["monetary"], settings.preprocessing.min_monetary), 2)

    # Quintile RFM Scoring (1 to 5)
    # Recency: lower days -> higher score 5 (use rank to handle identical values smoothly)
    rfm["r_score"] = pd.qcut(
        rfm["recency"].rank(method="first"), q=5, labels=[5, 4, 3, 2, 1]
    ).astype(int)
    rfm["f_score"] = pd.qcut(
        rfm["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]
    ).astype(int)
    rfm["m_score"] = pd.qcut(
        rfm["monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    # Combined RFM Score (e.g. 555 for Champions)
    rfm["rfm_score"] = rfm["r_score"] * 100 + rfm["f_score"] * 10 + rfm["m_score"]
    rfm["rfm_code"] = (
        rfm["r_score"].astype(str) + rfm["f_score"].astype(str) + rfm["m_score"].astype(str)
    )

    # Assign named segment
    rfm["segment"] = [
        assign_rfm_segment(r, f, m)
        for r, f, m in zip(rfm["r_score"], rfm["f_score"], rfm["m_score"], strict=False)
    ]

    # Assign Ground Truth Churn label based on configurable churn window (recency > 90 days)
    churn_window = settings.business_targets.churn_window_days
    rfm["churned"] = (rfm["recency"] > churn_window).astype(int)

    overall_churn_rate = rfm["churned"].mean()
    logger.info(
        "RFM computation completed for %d customers. Overall churn rate: %.2f%% (%d churned).",
        len(rfm),
        overall_churn_rate * 100,
        rfm["churned"].sum(),
    )

    return rfm


def run_rfm_pipeline() -> pd.DataFrame:
    """Loads cleaned transactions and computes RFM customer master table.

    Returns:
        Customer master DataFrame.
    """
    parquet_path = Path(settings.paths.processed_transactions_file)
    if not parquet_path.exists():
        from src.churn.preprocessing import run_preprocessing_pipeline

        purchases_df, cancellations_df = run_preprocessing_pipeline()
    else:
        purchases_df = pd.read_parquet(parquet_path)
        cancellations_df = None

    rfm_df = compute_rfm_features(purchases_df, cancellations_df)
    return rfm_df


if __name__ == "__main__":
    run_rfm_pipeline()
