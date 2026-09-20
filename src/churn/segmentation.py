"""K-Means clustering and Pareto revenue analysis module."""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.churn.config import settings
from src.churn.utils import get_logger

logger = get_logger(__name__)


def evaluate_kmeans_clusters(
    scaled_features: np.ndarray,
    k_min: int = 3,
    k_max: int = 7,
) -> dict[int, dict[str, float]]:
    """Evaluates K-Means clustering across a range of k using Inertia and Silhouette scores.

    Args:
        scaled_features: Scaled log-transformed feature matrix.
        k_min: Minimum number of clusters to test.
        k_max: Maximum number of clusters to test.

    Returns:
        Dictionary mapping each k to its inertia and silhouette score.
    """
    logger.info("Evaluating K-Means clustering for k in [%d, %d]...", k_min, k_max)
    eval_results = {}

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=settings.project.seed, n_init=10)
        labels = kmeans.fit_predict(scaled_features)
        sil = float(silhouette_score(scaled_features, labels, sample_size=3000, random_state=42))
        inertia = float(kmeans.inertia_)
        eval_results[k] = {"inertia": inertia, "silhouette": sil}
        logger.info("k=%d: Inertia=%.2f, Silhouette=%.4f", k, inertia, sil)

    return eval_results


def fit_kmeans_clustering(
    rfm_df: pd.DataFrame,
    n_clusters: int | None = None,
) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    """Applies log-transformation, standard scaling, and K-Means clustering to RFM features.

    Args:
        rfm_df: DataFrame with recency, frequency, and monetary attributes.
        n_clusters: Number of clusters (defaults to config setting).

    Returns:
        Tuple of (enriched_rfm_df, fitted_kmeans, fitted_scaler).
    """
    k = n_clusters or settings.rfm.optimal_k
    logger.info("Fitting K-Means model with optimal k=%d...", k)

    df = rfm_df.copy()

    # Log-transform to normalize heavy-tailed distributions
    log_features = np.log1p(df[["recency", "frequency", "monetary"]].values)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(log_features)

    kmeans = KMeans(n_clusters=k, random_state=settings.project.seed, n_init=10)
    df["cluster_id"] = kmeans.fit_predict(scaled_features)

    # Characterize clusters by centroid values to generate intuitive cluster names
    cluster_means = df.groupby("cluster_id")[["recency", "frequency", "monetary"]].mean()

    # Assign meaningful labels:
    # High monetary & high frequency -> VIP Core
    # Low recency & moderate frequency -> Emerging Active
    # Moderate recency & moderate monetary -> Nurture
    # High recency & low frequency -> At-Risk / Dormant
    cluster_labels = {}
    sorted_by_monetary = cluster_means.sort_values("monetary", ascending=False).index.tolist()

    if len(sorted_by_monetary) >= 4:
        cluster_labels[sorted_by_monetary[0]] = "High-Value Loyalists"
        cluster_labels[sorted_by_monetary[1]] = "Engaged Frequent Buyers"
        cluster_labels[sorted_by_monetary[2]] = "Occasional / New Shoppers"
        cluster_labels[sorted_by_monetary[3]] = "Dormant / High Churn Risk"
    else:
        for idx, c_id in enumerate(sorted_by_monetary):
            cluster_labels[c_id] = f"Cluster {idx + 1}"

    df["kmeans_cluster"] = df["cluster_id"].map(cluster_labels)
    logger.info(
        "K-Means clustering complete. Cluster distribution:\n%s",
        df["kmeans_cluster"].value_counts(),
    )

    return df, kmeans, scaler


def compute_pareto_revenue_share(df: pd.DataFrame) -> dict[str, float]:
    """Calculates empirical Pareto revenue concentration metrics.

    Args:
        df: Customer master table containing 'monetary' spend per customer.

    Returns:
        Dictionary with empirical Pareto statistics.
    """
    total_customers = len(df)
    total_revenue = float(df["monetary"].sum())

    sorted_df = df.sort_values(by="monetary", ascending=False).reset_index(drop=True)
    sorted_df["cumulative_revenue"] = sorted_df["monetary"].cumsum()
    sorted_df["cumulative_share"] = sorted_df["cumulative_revenue"] / total_revenue
    sorted_df["customer_rank"] = sorted_df.index + 1
    sorted_df["customer_percentile"] = sorted_df["customer_rank"] / total_customers

    top_20_count = int(np.ceil(0.20 * total_customers))
    top_20_revenue = float(sorted_df.iloc[:top_20_count]["monetary"].sum())
    top_20_share = top_20_revenue / total_revenue

    top_10_count = int(np.ceil(0.10 * total_customers))
    top_10_share = float(sorted_df.iloc[:top_10_count]["monetary"].sum()) / total_revenue

    logger.info(
        "Pareto Analysis Result: Top 20%% of customers (%d of %d) generate $%.2f out of $%.2f (%.2f%% of total revenue).",
        top_20_count,
        total_customers,
        top_20_revenue,
        total_revenue,
        top_20_share * 100,
    )

    return {
        "total_customers": total_customers,
        "total_revenue": total_revenue,
        "top_20_count": top_20_count,
        "top_20_revenue": top_20_revenue,
        "top_20_revenue_share": round(top_20_share, 4),
        "top_10_revenue_share": round(top_10_share, 4),
        "target_pareto_share": settings.business_targets.target_pareto_revenue_share,
    }


def save_customer_master(df: pd.DataFrame) -> None:
    """Persists customer master table to SQLite, CSV, and Parquet.

    Args:
        df: Customer master DataFrame.
    """
    # 1. Save to CSV and Parquet
    csv_path = Path(settings.paths.customer_master_file)
    parquet_path = Path(settings.paths.customer_master_parquet)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_parquet(parquet_path, index=False)
    logger.info("Saved customer master to %s and %s", csv_path, parquet_path)

    # 2. Save to SQLite database
    sqlite_path = Path(settings.paths.sqlite_path)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(sqlite_path)
    try:
        # Format datetimes as ISO strings for SQLite
        export_df = df.copy()
        if "first_purchase" in export_df.columns:
            export_df["first_purchase"] = export_df["first_purchase"].astype(str)
        if "last_purchase" in export_df.columns:
            export_df["last_purchase"] = export_df["last_purchase"].astype(str)

        export_df.to_sql("customers", conn, if_exists="replace", index=False)
        logger.info("Persisted customer master table to SQLite: %s (table: customers)", sqlite_path)
    finally:
        conn.close()


def run_segmentation_pipeline() -> tuple[pd.DataFrame, dict[str, float]]:
    """Executes full RFM and K-Means segmentation pipeline.

    Returns:
        Tuple of (customer_master_df, pareto_metrics).
    """
    from src.churn.rfm import run_rfm_pipeline

    rfm_df = run_rfm_pipeline()
    customer_master, _, _ = fit_kmeans_clustering(rfm_df)
    pareto_metrics = compute_pareto_revenue_share(customer_master)
    save_customer_master(customer_master)
    return customer_master, pareto_metrics


if __name__ == "__main__":
    run_segmentation_pipeline()
