"""Pytest session configuration and fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.churn.config import settings


@pytest.fixture(scope="session", autouse=True)
def ensure_database_exists() -> None:
    """Ensures SQLite database exists before running any integration tests."""
    db_path = Path(settings.paths.sqlite_path)
    if not db_path.exists():
        from src.churn.pipeline import run_pipeline

        run_pipeline("run-all")
