"""Shared utility functions, Plotly visual theming, and data loaders for Streamlit."""

import os
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

from src.churn.config import settings
from src.churn.explain import explain_customer_prediction
from src.churn.predict import load_optimal_threshold
from src.churn.utils import get_logger, load_json

logger = get_logger(__name__)

API_BASE_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

# Unified Brand Color Palette
COLOR_PRIMARY = "#4F46E5"  # Indigo
COLOR_TEAL = "#14B8A6"  # Teal
COLOR_EMERALD = "#10B981"  # Emerald
COLOR_AMBER = "#F59E0B"  # Amber
COLOR_CORAL = "#EF4444"  # Coral
COLOR_PURPLE = "#8B5CF6"  # Purple
COLOR_SLATE = "#64748B"  # Slate Gray
COLOR_DARK = "#0F172A"  # Deep Slate

SEGMENT_COLORS = {
    "Champions": "#4F46E5",
    "Loyal Customers": "#14B8A6",
    "Potential Loyalist": "#10B981",
    "New Customers": "#3B82F6",
    "Promising": "#8B5CF6",
    "Need Attention": "#F59E0B",
    "At Risk": "#EF4444",
    "Can't Lose Them": "#991B1B",
    "Hibernating": "#94A3B8",
    "Lost": "#64748B",
}

RISK_COLORS = {
    "Low": "#10B981",
    "Medium": "#F59E0B",
    "High": "#EF4444",
    "Critical": "#7F1D1D",
}


def inject_custom_css() -> None:
    """Reads and injects styles.css into the active Streamlit page."""
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def get_plotly_layout(
    title: str | None = None,
    height: int = 420,
    show_legend: bool = True,
) -> dict[str, Any]:
    """Generates a cohesive, publication-grade Plotly layout template."""
    return {
        "title": {
            "text": title or "",
            "font": {
                "family": "Plus Jakarta Sans, sans-serif",
                "size": 16,
                "color": "#0F172A",
                "weight": 700,
            },
            "x": 0.02,
            "y": 0.96,
        },
        "height": height,
        "paper_bgcolor": "#FFFFFF",
        "plot_bgcolor": "#FFFFFF",
        "font": {"family": "Plus Jakarta Sans, sans-serif", "color": "#334155", "size": 12},
        "margin": {"l": 40, "r": 30, "t": 50, "b": 40},
        "legend": (
            {
                "orientation": "h",
                "yanchor": "bottom",
                "y": -0.22,
                "xanchor": "center",
                "x": 0.5,
                "font": {"size": 11},
            }
            if show_legend
            else {"visible": False}
        ),
        "xaxis": {
            "showgrid": True,
            "gridcolor": "#F1F5F9",
            "zeroline": False,
            "linecolor": "#E2E8F0",
            "tickfont": {"color": "#64748B", "size": 11},
        },
        "yaxis": {
            "showgrid": True,
            "gridcolor": "#F1F5F9",
            "zeroline": False,
            "linecolor": "#E2E8F0",
            "tickfont": {"color": "#64748B", "size": 11},
        },
        "hoverlabel": {
            "bgcolor": "#0F172A",
            "font_size": 12,
            "font_family": "Plus Jakarta Sans, sans-serif",
            "font_color": "#FFFFFF",
        },
    }


@st.cache_data(ttl=600)
def load_customer_master_data() -> pd.DataFrame:
    """Loads customer records joined with predictions from SQLite, with CSV fallback.

    Returns:
        Consolidated customer DataFrame.
    """
    db_path = Path(settings.paths.sqlite_path)
    if not db_path.exists():
        alt_path = Path(__file__).resolve().parent.parent / settings.paths.sqlite_path
        if alt_path.exists():
            db_path = alt_path

    if db_path.exists():
        try:
            conn = sqlite3.connect(db_path)
            query = """
                SELECT
                    c.*,
                    p.churn_probability,
                    p.churn_prediction,
                    p.risk_band,
                    p.revenue_at_risk
                FROM customers c
                LEFT JOIN predictions p ON c.CustomerID = p.CustomerID
            """
            df = pd.read_sql(query, conn)
            conn.close()
            if not df.empty:
                return df
        except Exception as e:
            logger.warning("Error reading from SQLite: %s. Falling back to CSV...", e)

    # Fallback to scored CSV
    csv_path = Path("data/processed/scored_customers.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)

    # Fallback to customer master CSV
    master_csv = Path(settings.paths.customer_master_file)
    if master_csv.exists():
        df = pd.read_csv(master_csv)
        # Compute default predictions if table was not yet scored
        df["churn_probability"] = 0.5
        df["churn_prediction"] = df["churned"]
        df["risk_band"] = "Medium"
        df["revenue_at_risk"] = df["monetary"] * 0.5
        return df

    return pd.DataFrame()


@st.cache_data(ttl=600)
def load_metrics() -> dict[str, Any]:
    """Loads model performance metrics dictionary from models/metrics.json."""
    metrics_path = Path(settings.paths.metrics_file)
    if not metrics_path.exists():
        alt = Path(__file__).resolve().parent.parent / settings.paths.metrics_file
        if alt.exists():
            metrics_path = alt

    if metrics_path.exists():
        return load_json(metrics_path)
    return {}


@st.cache_data(ttl=600)
def load_threshold_info() -> dict[str, Any]:
    """Loads decision threshold optimization data from models/threshold.json."""
    th_path = Path(settings.paths.threshold_file)
    if not th_path.exists():
        alt = Path(__file__).resolve().parent.parent / settings.paths.threshold_file
        if alt.exists():
            th_path = alt

    if th_path.exists():
        return load_json(th_path)
    return {}


def check_api_health() -> bool:
    """Pings FastAPI /health endpoint to check if backend is running."""
    try:
        res = requests.get(f"{API_BASE_URL}/health", timeout=1.5)
        return res.status_code == 200
    except Exception:
        return False


def request_prediction_api(customer_dict: dict[str, Any]) -> dict[str, Any]:
    """Requests real-time churn prediction from FastAPI, with seamless local fallback.

    Args:
        customer_dict: Customer attributes.

    Returns:
        Prediction and SHAP explanation payload.
    """
    try:
        res = requests.post(f"{API_BASE_URL}/predict", json=customer_dict, timeout=3.0)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass

    # Graceful Local Fallback: use local joblib model and SHAP engine directly
    res = explain_customer_prediction(customer_dict)
    tau = load_optimal_threshold()
    res["churn_prediction"] = int(res["churn_probability"] >= tau)
    res["CustomerID"] = customer_dict.get("CustomerID")
    return res


def render_metric_card(
    title: str,
    value: str,
    delta: str | None = None,
    delta_type: str = "neutral",
) -> str:
    """Renders HTML string for a styled metric card."""
    delta_html = ""
    if delta:
        delta_html = f'<div class="metric-delta {delta_type}">{delta}</div>'

    return f"""
    <div class="metric-card">
        <div class="metric-label">{title}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """


def render_sidebar_status() -> None:
    """Renders persistent status and system information in Streamlit sidebar."""
    with st.sidebar:
        st.markdown("### 🛡️ ChurnGuard System")
        is_live = check_api_health()
        if is_live:
            st.markdown(
                '<div style="background:#064E3B;color:#6EE7B7;padding:6px 12px;border-radius:8px;font-size:0.8rem;font-weight:600;margin-bottom:12px;display:flex;align-items:center;gap:6px;">'
                '<span style="height:8px;width:8px;background:#10B981;border-radius:50%;display:inline-block;"></span>'
                "FastAPI Backend: CONNECTED (Port 8000)</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:#1E293B;color:#94A3B8;padding:6px 12px;border-radius:8px;font-size:0.8rem;font-weight:600;margin-bottom:12px;display:flex;align-items:center;gap:6px;">'
                '<span style="height:8px;width:8px;background:#F59E0B;border-radius:50%;display:inline-block;"></span>'
                "Engine: Direct In-Memory Mode</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.caption("v1.0.0 | Production Portfolio Standard")
