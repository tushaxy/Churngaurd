"""Utility functions for logging, configuration loading, and serialization."""

import io
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

_LOGGERS: dict[str, logging.Logger] = {}


def read_csv_robust(file_or_path: Any, **kwargs: Any) -> pd.DataFrame:
    """Reads a CSV file, path, or buffer using multi-encoding detection to prevent decode errors.

    Specifically resolves 'utf-8 codec can't decode byte 0x96' and other Windows / Excel
    encoding anomalies by trying ['utf-8', 'utf-8-sig', 'cp1252', 'latin1', 'iso-8859-1']
    and falling back to replacement characters if needed.

    Args:
        file_or_path: File path (str or Path), Streamlit UploadedFile, or bytes buffer.
        **kwargs: Additional keyword arguments passed to pd.read_csv.

    Returns:
        Loaded pandas DataFrame.
    """
    if isinstance(file_or_path, (str, Path)):
        path = Path(file_or_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_or_path}")
        with open(path, "rb") as f:
            content_bytes = f.read()
    elif hasattr(file_or_path, "getvalue"):
        content_bytes = file_or_path.getvalue()
    elif hasattr(file_or_path, "read"):
        content_bytes = file_or_path.read()
        if hasattr(file_or_path, "seek"):
            file_or_path.seek(0)
    elif isinstance(file_or_path, (bytes, bytearray)):
        content_bytes = bytes(file_or_path)
    else:
        return pd.read_csv(file_or_path, **kwargs)

    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1", "iso-8859-1"]
    for enc in encodings:
        try:
            decoded = content_bytes.decode(enc)
            return pd.read_csv(io.StringIO(decoded), **kwargs)
        except (UnicodeDecodeError, LookupError):
            continue

    decoded = content_bytes.decode("utf-8", errors="replace")
    return pd.read_csv(io.StringIO(decoded), **kwargs)


def get_logger(name: str = "churnguard", level: str | None = None) -> logging.Logger:
    """Configures and returns a structured logger.

    Args:
        name: Name of the logger, typically __name__.
        level: Optional log level override (e.g. 'DEBUG', 'INFO').

    Returns:
        Configured logging.Logger instance.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    log_level = level or os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    _LOGGERS[name] = logger
    return logger


def load_yaml_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Loads YAML configuration safely into a dictionary.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        # Look in workspace root if relative path
        alt_path = Path(__file__).resolve().parent.parent.parent / config_path
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def save_json(data: Any, file_path: str | Path) -> None:
    """Serializes data to a formatted JSON file.

    Args:
        data: Python dictionary or serializable object.
        file_path: Destination path for the JSON file.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(file_path: str | Path) -> dict[str, Any]:
    """Loads and deserializes data from a JSON file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Deserialized dictionary.
    """
    path = Path(file_path)
    if not path.exists():
        alt_path = Path(__file__).resolve().parent.parent.parent / file_path
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"JSON file not found at: {file_path}")

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def set_random_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility across random, numpy, and python hash.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
