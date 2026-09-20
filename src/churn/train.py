"""Machine learning training, hyperparameter tuning, evaluation, and threshold optimization."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.churn.config import settings
from src.churn.utils import get_logger, save_json, set_random_seed

logger = get_logger(__name__)

# Features used for classification (excluding raw snapshot recency to prevent target leakage)
NUMERICAL_FEATURES = [
    "frequency",
    "monetary",
    "avg_order_value",
    "tenure_days",
    "orders_per_month",
    "return_rate",
    "rfm_score",
]

CATEGORICAL_FEATURES = ["segment", "kmeans_cluster"]


def build_preprocessor() -> ColumnTransformer:
    """Constructs scikit-learn column transformer for numeric scaling and categorical encoding.

    Returns:
        Configured ColumnTransformer instance.
    """
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def compare_imbalance_strategies(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
) -> str:
    """Evaluates class_weight='balanced' against SMOTE using 5-fold CV to select highest PR-AUC.

    Args:
        X_train: Training feature DataFrame.
        y_train: Training ground truth churn labels.
        preprocessor: Fitted or unfitted feature ColumnTransformer.

    Returns:
        Winning strategy identifier: 'balanced' or 'smote'.
    """
    logger.info("Comparing class imbalance strategies ('balanced' vs 'smote') via 5-fold CV...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=settings.project.seed)

    # Strategy 1: class_weight='balanced'
    balanced_scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        X_tr_proc = preprocessor.fit_transform(X_tr)
        X_val_proc = preprocessor.transform(X_val)

        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            class_weight="balanced",
            random_state=settings.project.seed,
            n_jobs=-1,
        )
        clf.fit(X_tr_proc, y_tr)
        val_probs = clf.predict_proba(X_val_proc)[:, 1]
        balanced_scores.append(average_precision_score(y_val, val_probs))

    mean_balanced_pr_auc = float(np.mean(balanced_scores))

    # Strategy 2: SMOTE oversampling
    smote_scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        X_tr_proc = preprocessor.fit_transform(X_tr)
        X_val_proc = preprocessor.transform(X_val)

        smote = SMOTE(random_state=settings.project.seed)
        X_tr_res, y_tr_res = smote.fit_resample(X_tr_proc, y_tr)

        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=settings.project.seed,
            n_jobs=-1,
        )
        clf.fit(X_tr_res, y_tr_res)
        val_probs = clf.predict_proba(X_val_proc)[:, 1]
        smote_scores.append(average_precision_score(y_val, val_probs))

    mean_smote_pr_auc = float(np.mean(smote_scores))

    logger.info(
        "5-Fold CV PR-AUC — class_weight='balanced': %.4f | SMOTE: %.4f",
        mean_balanced_pr_auc,
        mean_smote_pr_auc,
    )

    chosen = "balanced" if mean_balanced_pr_auc >= mean_smote_pr_auc else "smote"
    logger.info("Selected imbalance strategy: %s", chosen)
    return chosen


def tune_random_forest(
    X_train: np.ndarray,
    y_train: pd.Series,
    class_weight: str | None = "balanced",
) -> RandomForestClassifier:
    """Tunes RandomForestClassifier hyperparameters using RandomizedSearchCV.

    Args:
        X_train: Preprocessed training features.
        y_train: Training labels.
        class_weight: Weighting strategy ('balanced' or None).

    Returns:
        Best fitted RandomForestClassifier model.
    """
    logger.info("Executing RandomizedSearchCV over 40 hyperparameter iterations...")

    base_rf = RandomForestClassifier(
        class_weight=class_weight,
        random_state=settings.project.seed,
        n_jobs=-1,
    )

    param_distributions = {
        "n_estimators": [100, 150, 200, 250],
        "max_depth": [6, 8, 10, 12, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.8],
    }

    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_distributions,
        n_iter=40,
        scoring="average_precision",
        cv=5,
        random_state=settings.project.seed,
        n_jobs=-1,
        verbose=0,
    )

    search.fit(X_train, y_train)
    logger.info(
        "Hyperparameter tuning completed. Best PR-AUC: %.4f with params: %s",
        search.best_score_,
        search.best_params_,
    )
    return search.best_estimator_


def optimize_revenue_at_risk_threshold(
    y_test: np.ndarray,
    test_probs: np.ndarray,
    test_monetary: np.ndarray,
    target_ratio: float = 0.07,
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    """Grid-searches decision probability thresholds to match empirical Revenue at Risk to target ~7%.

    Revenue at risk is defined as: sum(monetary * churn_probability) for customers above threshold.

    Args:
        y_test: Ground truth binary churn labels on test set.
        test_probs: Predicted churn probabilities.
        test_monetary: Monetary values of test customers.
        target_ratio: Target revenue at risk fraction of total revenue (default 0.07).

    Returns:
        Tuple of (optimal_threshold, chosen_metrics, tradeoff_curve_data).
    """
    logger.info(
        "Optimizing decision threshold to achieve target ~%.1f%% Revenue at Risk...",
        target_ratio * 100,
    )
    total_test_rev = float(np.sum(test_monetary))

    threshold_candidates = np.linspace(0.10, 0.95, 86)
    tradeoff_records = []

    best_diff = float("inf")
    best_thresh = 0.50
    best_record = {}

    for t in threshold_candidates:
        preds = (test_probs >= t).astype(int)
        flagged_mask = preds == 1
        n_flagged = int(np.sum(flagged_mask))

        if n_flagged > 0:
            rar_amount = float(np.sum(test_monetary[flagged_mask] * test_probs[flagged_mask]))
        else:
            rar_amount = 0.0

        rar_ratio = rar_amount / total_test_rev if total_test_rev > 0 else 0.0

        p = float(precision_score(y_test, preds, zero_division=0))
        r = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))

        record = {
            "threshold": round(float(t), 3),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "revenue_at_risk": round(rar_amount, 2),
            "revenue_at_risk_ratio": round(rar_ratio, 4),
            "revenue_at_risk_pct": round(rar_ratio * 100, 2),
            "flagged_customers": n_flagged,
        }
        tradeoff_records.append(record)

        diff = abs(rar_ratio - target_ratio)
        if diff < best_diff:
            best_diff = diff
            best_thresh = float(t)
            best_record = record

    logger.info(
        "Optimal Threshold found: %.3f -> Revenue at Risk: $%.2f (%.2f%% of revenue, target %.2f%%). Precision=%.3f, Recall=%.3f, F1=%.3f.",
        best_thresh,
        best_record["revenue_at_risk"],
        best_record["revenue_at_risk_pct"],
        target_ratio * 100,
        best_record["precision"],
        best_record["recall"],
        best_record["f1"],
    )

    return best_thresh, best_record, tradeoff_records


def train_churn_model() -> dict[str, Any]:
    """Executes the full end-to-end model training, evaluation, and serialization pipeline.

    Returns:
        Dictionary containing test performance metrics, threshold optimization, and paths.
    """
    set_random_seed(settings.project.seed)

    # 1. Load data
    data_path = Path(settings.paths.customer_master_file)
    if not data_path.exists():
        from src.churn.segmentation import run_segmentation_pipeline

        df, _ = run_segmentation_pipeline()
    else:
        df = pd.read_csv(data_path)

    logger.info("Loaded %d customer records for model training.", len(df))

    # Features and Target
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[settings.model.target_column].copy()
    monetary_values = df["monetary"].values

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test, m_train, m_test = train_test_split(
        X,
        y,
        monetary_values,
        test_size=settings.model.test_size,
        random_state=settings.project.seed,
        stratify=y,
    )

    logger.info(
        "Dataset split: %d train samples (%d churned), %d test samples (%d churned).",
        len(X_train),
        y_train.sum(),
        len(X_test),
        y_test.sum(),
    )

    # 2. Preprocessor construction and imbalance strategy selection
    preprocessor = build_preprocessor()
    strategy = compare_imbalance_strategies(X_train, y_train, preprocessor)

    # Fit preprocessor on full training split
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Apply SMOTE if chosen, otherwise class_weight='balanced'
    if strategy == "smote":
        smote = SMOTE(random_state=settings.project.seed)
        X_train_proc, y_train = smote.fit_resample(X_train_proc, y_train)
        rf_model = tune_random_forest(X_train_proc, y_train, class_weight=None)
    else:
        rf_model = tune_random_forest(X_train_proc, y_train, class_weight="balanced")

    # 3. Model Evaluation & Calibration
    raw_test_probs = rf_model.predict_proba(X_test_proc)[:, 1]
    brier = float(brier_score_loss(y_test, raw_test_probs))
    logger.info("Raw Model Brier Score: %.4f", brier)

    # Calibrate probabilities if Brier score > 0.12
    if brier > 0.12:
        logger.info("Calibrating model probabilities using CalibratedClassifierCV...")
        calibrated_model = CalibratedClassifierCV(estimator=rf_model, method="sigmoid", cv="prefit")
        calibrated_model.fit(X_train_proc, y_train)
        final_model = calibrated_model
        test_probs = final_model.predict_proba(X_test_proc)[:, 1]
        brier_final = float(brier_score_loss(y_test, test_probs))
        logger.info("Calibrated Model Brier Score: %.4f", brier_final)
    else:
        final_model = rf_model
        test_probs = raw_test_probs
        brier_final = brier

    # Full End-to-End Pipeline
    full_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", final_model)])

    # Standard Metrics
    roc_auc = float(roc_auc_score(y_test, test_probs))
    pr_auc = float(average_precision_score(y_test, test_probs))

    # Compute ROC and PR curve points
    fpr, tpr, roc_thresh = roc_curve(y_test, test_probs)
    pr_precision, pr_recall, pr_thresh = precision_recall_curve(y_test, test_probs)
    prob_true, prob_pred = calibration_curve(y_test, test_probs, n_bins=10)

    # 4. Threshold Optimization for ~7% Revenue at Risk
    optimal_thresh, chosen_metrics, tradeoff_curve = optimize_revenue_at_risk_threshold(
        y_test.values,
        test_probs,
        m_test,
        target_ratio=settings.business_targets.target_revenue_at_risk_ratio,
    )

    # Final predictions at optimal threshold
    final_preds = (test_probs >= optimal_thresh).astype(int)
    cm = confusion_matrix(y_test, final_preds).tolist()

    # 5. Compile and serialize artifacts
    metrics_summary = {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier_final, 4),
        "optimal_threshold": optimal_thresh,
        "precision_at_optimal": chosen_metrics["precision"],
        "recall_at_optimal": chosen_metrics["recall"],
        "f1_at_optimal": chosen_metrics["f1"],
        "revenue_at_risk_at_optimal": chosen_metrics["revenue_at_risk"],
        "revenue_at_risk_pct_at_optimal": chosen_metrics["revenue_at_risk_pct"],
        "target_revenue_at_risk_ratio": settings.business_targets.target_revenue_at_risk_ratio,
        "confusion_matrix": {
            "tn": cm[0][0],
            "fp": cm[0][1],
            "fn": cm[1][0],
            "tp": cm[1][1],
        },
        "imbalance_strategy": strategy,
        "curve_points": {
            "roc": {
                "fpr": [round(float(x), 4) for x in fpr[:: max(1, len(fpr) // 50)]],
                "tpr": [round(float(x), 4) for x in tpr[:: max(1, len(tpr) // 50)]],
            },
            "pr": {
                "precision": [
                    round(float(x), 4) for x in pr_precision[:: max(1, len(pr_precision) // 50)]
                ],
                "recall": [round(float(x), 4) for x in pr_recall[:: max(1, len(pr_recall) // 50)]],
            },
            "calibration": {
                "prob_pred": [round(float(x), 4) for x in prob_pred],
                "prob_true": [round(float(x), 4) for x in prob_true],
            },
        },
    }

    # Save model pipeline
    model_path = Path(settings.paths.model_file)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(full_pipeline, model_path)
    logger.info("Saved fitted model pipeline to %s", model_path)

    # Save metrics JSON
    metrics_path = Path(settings.paths.metrics_file)
    save_json(metrics_summary, metrics_path)
    logger.info("Saved model performance metrics to %s", metrics_path)

    # Save threshold tradeoff curve
    threshold_path = Path(settings.paths.threshold_file)
    threshold_payload = {
        "optimal_threshold": optimal_thresh,
        "chosen_metrics": chosen_metrics,
        "tradeoff_curve": tradeoff_curve,
    }
    save_json(threshold_payload, threshold_path)
    logger.info("Saved threshold tradeoff data to %s", threshold_path)

    # 6. Fit and Save SHAP Explainer
    from src.churn.explain import fit_and_save_shap_explainer

    fit_and_save_shap_explainer(full_pipeline, X_train)

    return metrics_summary


if __name__ == "__main__":
    train_churn_model()
