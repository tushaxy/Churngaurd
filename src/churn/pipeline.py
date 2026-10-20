"""Command-line pipeline orchestrator for ChurnGuard."""

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.churn.config import settings
from src.churn.predict import score_customers
from src.churn.segmentation import run_segmentation_pipeline
from src.churn.train import train_churn_model
from src.churn.utils import get_logger

logger = get_logger(__name__)


def run_training_step() -> dict:
    """Executes model training, tuning, and SHAP explainer generation."""
    logger.info(">>> Starting Model Training Step...")
    metrics = train_churn_model()
    logger.info(
        ">>> Model Training Complete. ROC-AUC: %.4f, PR-AUC: %.4f",
        metrics["roc_auc"],
        metrics["pr_auc"],
    )
    return metrics


def run_scoring_step() -> pd.DataFrame:
    """Scores all customers in SQLite database and persists to predictions table."""
    logger.info(">>> Starting Batch Customer Scoring Step...")

    sqlite_path = Path(settings.paths.sqlite_path)
    conn = sqlite3.connect(sqlite_path)

    try:
        customers_df = pd.read_sql("SELECT * FROM customers", conn)
        logger.info("Loaded %d customers from SQLite database for scoring.", len(customers_df))

        scored_df = score_customers(customers_df)

        # Columns for predictions table
        prediction_cols = [
            "CustomerID",
            "churn_probability",
            "churn_prediction",
            "risk_band",
            "revenue_at_risk",
            "scored_at",
        ]
        pred_table = scored_df[prediction_cols].copy()

        # Write to SQLite predictions table
        pred_table.to_sql("predictions", conn, if_exists="replace", index=False)
        logger.info(
            "Upserted %d prediction records into SQLite table 'predictions'.", len(pred_table)
        )

        # Save scored customer dataset
        scored_csv = Path("data/processed/scored_customers.csv")
        scored_df.to_csv(scored_csv, index=False, encoding="utf-8-sig")
        logger.info("Saved full scored customer master to %s", scored_csv)

        return scored_df
    finally:
        conn.close()


def run_all_steps() -> None:
    """Executes end-to-end data pipeline, modeling, scoring, and exports."""
    start_time = datetime.now()
    logger.info("=================================================================")
    logger.info(">>> CHURNGUARD: STARTING REPRODUCIBLE END-TO-END PIPELINE <<<")
    logger.info("=================================================================")

    # Step 1: Segmentation (runs ingestion, preprocessing, RFM, K-Means, Pareto)
    logger.info("[1/3] Executing Data Ingestion, Cleaning & RFM Segmentation...")
    customer_master, pareto_metrics = run_segmentation_pipeline()
    logger.info("Customer Master Table established: %d customers.", len(customer_master))

    # Step 2: Model Training & Threshold Optimization
    logger.info("[2/3] Training Tuned Random Forest & Optimizing Decision Threshold...")
    metrics = run_training_step()

    # Step 3: Batch Scoring & Database Seeding
    logger.info("[3/3] Scoring Entire Customer Base & Seeding Predictions Table...")
    scored_df = run_scoring_step()

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("=================================================================")
    logger.info(">>> PIPELINE EXECUTION SUCCEEDED in %.1f seconds! <<<", elapsed)
    logger.info("Key Empirical Results:")
    logger.info("  - Customers: %d", len(scored_df))
    logger.info(
        "  - Top 20%% Revenue Share: %.2f%% (Target: %.2f%%)",
        pareto_metrics["top_20_revenue_share"] * 100,
        pareto_metrics["target_pareto_share"] * 100,
    )
    logger.info(
        "  - Revenue at Risk: $%.2f (%.2f%% of revenue, Target: ~7.0%%)",
        metrics["revenue_at_risk_at_optimal"],
        metrics["revenue_at_risk_pct_at_optimal"],
    )
    logger.info("  - ROC-AUC: %.4f | PR-AUC: %.4f", metrics["roc_auc"], metrics["pr_auc"])
    logger.info("=================================================================")


def run_pipeline(command: str = "run-all") -> None:
    """Convenience dispatcher for running pipeline stages programmatically."""
    if command == "train":
        run_training_step()
    elif command == "score":
        run_scoring_step()
    elif command == "run-all":
        run_all_steps()
    else:
        raise ValueError(f"Unknown pipeline command: {command}")


def main() -> None:
    """CLI entrypoint dispatcher."""
    parser = argparse.ArgumentParser(description="ChurnGuard Pipeline CLI")
    parser.add_argument(
        "command",
        choices=["train", "score", "run-all"],
        help="Pipeline command to execute: 'train', 'score', or 'run-all'",
    )

    args = parser.parse_args()
    run_pipeline(args.command)


if __name__ == "__main__":
    main()
