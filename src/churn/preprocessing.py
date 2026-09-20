"""Data preprocessing and cleaning module for ChurnGuard transactions.

Cleans raw transactions: handles missing CustomerIDs, isolates and filters cancellations,
removes duplicates, applies IQR-based outlier boundaries, and derives transaction-level metrics.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.churn.config import settings
from src.churn.utils import get_logger

logger = get_logger(__name__)


def clean_transactions(
    df: pd.DataFrame,
    iqr_factor: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cleans raw transaction DataFrame and separates cancellations for return rate tracking.

    Args:
        df: Raw transactions DataFrame.
        iqr_factor: Multiplier for IQR outlier thresholding. Defaults to config setting.

    Returns:
        Tuple of (cleaned_purchases_df, cancellations_df).
    """
    initial_rows = len(df)
    logger.info("Starting preprocessing on %d raw transaction records...", initial_rows)

    factor = iqr_factor if iqr_factor is not None else settings.preprocessing.iqr_factor

    # Normalize column names
    df = df.copy()
    col_map = {c: c.strip() for c in df.columns}
    df.rename(columns=col_map, inplace=True)

    # 1. Drop rows with missing CustomerID
    missing_cust = df["CustomerID"].isna().sum()
    df = df.dropna(subset=["CustomerID"]).copy()
    df["CustomerID"] = df["CustomerID"].astype(int)
    logger.info(
        "Filtered %d records with missing CustomerID (remaining: %d).", missing_cust, len(df)
    )

    # 2. Parse InvoiceDate
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    invalid_dates = df["InvoiceDate"].isna().sum()
    if invalid_dates > 0:
        df = df.dropna(subset=["InvoiceDate"]).copy()
        logger.info("Filtered %d records with unparseable InvoiceDate.", invalid_dates)

    # 3. Deduplicate exact duplicate rows
    duplicates = df.duplicated().sum()
    df = df.drop_duplicates().copy()
    logger.info("Removed %d duplicate rows (remaining: %d).", duplicates, len(df))

    # 4. Standardize types
    df["Invoice"] = df["Invoice"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["Description"] = df["Description"].fillna("UNKNOWN").astype(str).str.strip()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0).astype(int)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0.0).astype(float)

    # Filter invalid non-positive prices
    valid_price_mask = df["Price"] > 0
    df = df[valid_price_mask].copy()

    # 5. Separate purchases and cancellations
    # Cancellations are marked with Quantity < 0 or Invoice starting with 'C'
    cancellation_mask = (df["Quantity"] < 0) | (df["Invoice"].str.startswith("C", na=False))
    cancellations_df = df[cancellation_mask].copy()
    purchases_df = df[~cancellation_mask].copy()

    logger.info(
        "Separated %d cancellations (%.2f%%) and %d positive purchases.",
        len(cancellations_df),
        (len(cancellations_df) / len(df)) * 100 if len(df) > 0 else 0,
        len(purchases_df),
    )

    # 6. Outlier Handling on purchases using IQR
    # Compute IQR bounds on Quantity and Price
    q25_qty, q75_qty = purchases_df["Quantity"].quantile(0.25), purchases_df["Quantity"].quantile(
        0.75
    )
    iqr_qty = q75_qty - q25_qty
    upper_qty = q75_qty + (factor * iqr_qty)

    q25_prc, q75_prc = purchases_df["Price"].quantile(0.25), purchases_df["Price"].quantile(0.75)
    iqr_prc = q75_prc - q25_prc
    upper_prc = q75_prc + (factor * iqr_prc)

    qty_outliers = (purchases_df["Quantity"] > upper_qty).sum()
    prc_outliers = (purchases_df["Price"] > upper_prc).sum()

    # Clip extreme values to upper bound to preserve revenue signal without skewing distributions
    purchases_df["Quantity"] = purchases_df["Quantity"].clip(upper=int(upper_qty))
    purchases_df["Price"] = purchases_df["Price"].clip(upper=float(upper_prc))

    logger.info(
        "IQR Outlier boundary: Quantity upper=%d (%d clipped), Price upper=%.2f (%d clipped).",
        upper_qty,
        qty_outliers,
        upper_prc,
        prc_outliers,
    )

    # 7. Compute TotalSpend
    purchases_df["TotalSpend"] = np.round(purchases_df["Quantity"] * purchases_df["Price"], 2)
    cancellations_df["TotalSpend"] = np.round(
        cancellations_df["Quantity"] * cancellations_df["Price"], 2
    )

    logger.info(
        "Preprocessing complete: %d valid purchases ready for RFM analysis.",
        len(purchases_df),
    )
    return purchases_df, cancellations_df


def run_preprocessing_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Executes the data cleaning pipeline and persists processed outputs.

    Returns:
        Tuple of (purchases_df, cancellations_df).
    """
    raw_path = Path(settings.paths.raw_transactions_file)
    if not raw_path.exists():
        from src.churn.data_ingestion import ingest_and_save_raw_data

        raw_df = ingest_and_save_raw_data()
    else:
        logger.info("Loading raw data from %s...", raw_path)
        raw_df = pd.read_csv(raw_path, encoding="utf-8-sig")

    purchases_df, cancellations_df = clean_transactions(raw_df)

    # Save to data/processed
    processed_dir = Path(settings.paths.processed_data_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = Path(settings.paths.processed_transactions_file)
    sample_csv_path = Path(settings.paths.processed_transactions_sample)

    purchases_df.to_parquet(parquet_path, index=False)
    logger.info("Saved full cleaned dataset to %s (%d rows)", parquet_path, len(purchases_df))

    # Save small representative sample (500 rows) for version control
    sample_df = purchases_df.head(500)
    sample_df.to_csv(sample_csv_path, index=False, encoding="utf-8-sig")
    logger.info("Saved representative sample to %s", sample_csv_path)

    return purchases_df, cancellations_df


if __name__ == "__main__":
    run_preprocessing_pipeline()
