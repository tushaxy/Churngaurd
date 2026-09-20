"""Dependency injection and resource providers for FastAPI."""

import sqlite3
from collections.abc import Generator
from pathlib import Path

import joblib
import shap
from sklearn.pipeline import Pipeline

from src.churn.config import settings
from src.churn.utils import get_logger

logger = get_logger(__name__)

_PIPELINE: Pipeline | None = None
_EXPLAINER: shap.TreeExplainer | None = None


def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """Provides a SQLite database connection with row factory enabled."""
    db_path = Path(settings.paths.sqlite_path)
    if not db_path.exists():
        alt_path = Path(__file__).resolve().parent.parent / settings.paths.sqlite_path
        if alt_path.exists():
            db_path = alt_path

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def get_model_pipeline() -> Pipeline:
    """Provides cached singleton model pipeline."""
    global _PIPELINE
    if _PIPELINE is None:
        model_path = Path(settings.paths.model_file)
        if not model_path.exists():
            alt_path = Path(__file__).resolve().parent.parent / settings.paths.model_file
            if alt_path.exists():
                model_path = alt_path
            else:
                raise FileNotFoundError(f"Model file not found at: {model_path}")
        logger.info("Loading model pipeline from %s", model_path)
        _PIPELINE = joblib.load(model_path)
    return _PIPELINE


def get_shap_explainer() -> shap.TreeExplainer | None:
    """Provides cached singleton SHAP TreeExplainer."""
    global _EXPLAINER
    if _EXPLAINER is None:
        explainer_path = Path(settings.paths.explainer_file)
        if not explainer_path.exists():
            alt_path = Path(__file__).resolve().parent.parent / settings.paths.explainer_file
            if alt_path.exists():
                explainer_path = alt_path
            else:
                return None
        logger.info("Loading SHAP explainer from %s", explainer_path)
        _EXPLAINER = joblib.load(explainer_path)
    return _EXPLAINER
