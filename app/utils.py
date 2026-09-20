"""Shared utility functions, Plotly visual theming, and data loaders for Streamlit."""

from __future__ import annotations

import io
import os
from pathlib import Path
import sqlite3
from typing import Any

import pandas as pd
import requests
import streamlit as st

from src.churn.config import settings
from src.churn.explain import explain_customer_prediction
from src.churn.predict import load_optimal_threshold
from src.churn.utils import get_logger, load_json

logger = get_logger(__name__)


def read_csv_robust(file_or_path: Any, **kwargs: Any) -> pd.DataFrame:
    """Reads a CSV file, path, or buffer using multi-encoding detection to prevent decode errors.

    Specifically resolves 'utf-8 codec can't decode byte 0x96' and other Windows / Excel
    encoding anomalies by trying ['utf-8', 'utf-8-sig', 'cp1252', 'latin1', 'iso-8859-1']
    and falling back to replacement characters if needed.
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

API_BASE_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

# Enterprise Color Palette (Slate, Sapphire, Emerald, Amber, Crimson)
COLOR_PRIMARY = "#2563EB"  # Sapphire Blue
COLOR_NAVY = "#1E3A8A"  # Deep Navy
COLOR_EMERALD = "#059669"  # Retained / Low Risk
COLOR_AMBER = "#D97706"  # Watch / Medium Risk
COLOR_CORAL = "#E11D48"  # Churn / High Risk
COLOR_PURPLE = "#7C3AED"  # Purple
COLOR_SLATE = "#64748B"  # Slate Gray
COLOR_DARK = "#0F172A"  # Obsidian Slate

SEGMENT_COLORS = {
    "Champions": "#1E40AF",
    "Loyal Customers": "#0284C7",
    "Potential Loyalist": "#0D9488",
    "New Customers": "#2563EB",
    "Promising": "#7C3AED",
    "Need Attention": "#D97706",
    "At Risk": "#DC2626",
    "Can't Lose Them": "#991B1B",
    "Hibernating": "#64748B",
    "Lost": "#475569",
}

RISK_COLORS = {
    "Low": "#059669",
    "Medium": "#D97706",
    "High": "#DC2626",
    "Critical": "#991B1B",
}


def inject_custom_css() -> None:
    """Reads and injects styles.css into the active Streamlit page."""
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_header_command_bar(
    breadcrumb: str,
    title: str,
    subtitle: str,
    model_tag: str = "RF-SMOTE v1.0.0",
) -> None:
    """Renders a high-end enterprise SaaS command bar header."""
    is_live = check_api_health()
    api_status_html = (
        '<span class="badge-meta active"><span style="color:#10B981;">●</span> API Online</span>'
        if is_live
        else '<span class="badge-meta"><span style="color:#F59E0B;">●</span> Local Engine</span>'
    )

    st.markdown(
        f"""
        <div class="command-bar">
            <div class="command-bar-title-wrap">
                <div class="command-bar-breadcrumb">{breadcrumb}</div>
                <h1 class="command-bar-title">{title}</h1>
                <p class="command-bar-subtitle">{subtitle}</p>
            </div>
            <div class="command-bar-meta">
                {api_status_html}
                <span class="badge-meta">{model_tag}</span>
                <span class="badge-meta">τ = 0.900</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(
    label: str,
    value: str,
    footer_text: str = "",
    pill_text: str = "",
    pill_type: str = "neutral",
    tag: str = "",
) -> str:
    """Renders a polished, enterprise-grade KPI metric card."""
    pill_html = f'<span class="metric-pill {pill_type}">{pill_text}</span>' if pill_text else ""
    tag_html = f'<span class="metric-tag">{tag}</span>' if tag else ""

    return f"""
    <div class="metric-card">
        <div class="metric-header">
            <span class="metric-label">{label}</span>
            {tag_html}
        </div>
        <div class="metric-value">{value}</div>
        <div class="metric-footer">
            {pill_html}
            <span>{footer_text}</span>
        </div>
    </div>
    """


def get_plotly_layout(
    title: str | None = None,
    height: int = 380,
    show_legend: bool = True,
) -> dict[str, Any]:
    """Generates an ultra-clean, institutional Plotly layout template."""
    return {
        "title": (
            {
                "text": title,
                "font": {
                    "family": "Inter, -apple-system, sans-serif",
                    "size": 14,
                    "color": "#0F172A",
                    "weight": 700,
                },
                "x": 0.01,
                "y": 0.96,
            }
            if title
            else None
        ),
        "height": height,
        "paper_bgcolor": "#FFFFFF",
        "plot_bgcolor": "#FFFFFF",
        "font": {
            "family": "Inter, -apple-system, sans-serif",
            "color": "#475569",
            "size": 11,
        },
        "margin": {"l": 45, "r": 25, "t": 35 if title else 20, "b": 35},
        "legend": (
            {
                "orientation": "h",
                "yanchor": "bottom",
                "y": -0.22,
                "xanchor": "center",
                "x": 0.5,
                "font": {"size": 11, "color": "#475569"},
            }
            if show_legend
            else {"visible": False}
        ),
        "xaxis": {
            "showgrid": True,
            "gridcolor": "#F1F5F9",
            "gridwidth": 1,
            "zeroline": False,
            "linecolor": "#E2E8F0",
            "tickfont": {"color": "#64748B", "size": 10},
        },
        "yaxis": {
            "showgrid": True,
            "gridcolor": "#F1F5F9",
            "gridwidth": 1,
            "zeroline": False,
            "linecolor": "#E2E8F0",
            "tickfont": {"color": "#64748B", "size": 10},
        },
        "hoverlabel": {
            "bgcolor": "#0F172A",
            "font_size": 12,
            "font_family": "Inter, -apple-system, sans-serif",
            "font_color": "#FFFFFF",
            "bordercolor": "#1E293B",
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
        return read_csv_robust(csv_path)

    # Fallback to customer master CSV
    master_csv = Path(settings.paths.customer_master_file)
    if master_csv.exists():
        df = read_csv_robust(master_csv)
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

    # Graceful Local Fallback
    res = explain_customer_prediction(customer_dict)
    tau = load_optimal_threshold()
    res["churn_prediction"] = int(res["churn_probability"] >= tau)
    res["CustomerID"] = customer_dict.get("CustomerID")
    return res


def render_sidebar_status() -> None:
    """Renders sleek enterprise telemetry metadata in Streamlit sidebar."""
    with st.sidebar:
        is_live = check_api_health()

        status_tag = (
            '<div class="telemetry-pill live"><span class="telemetry-dot"></span> API CONNECTED</div>'
            if is_live
            else '<div class="telemetry-pill" style="background:#1E293B;color:#94A3B8;">● LOCAL MODE</div>'
        )

        st.markdown(
            f"""
            <div class="sidebar-telemetry">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span style="font-size:0.75rem; font-weight:700; color:#F1F5F9; letter-spacing:0.04em;">SYSTEM TELEMETRY</span>
                    {status_tag}
                </div>
                <div class="telemetry-row">
                    <span>Host API</span>
                    <span class="telemetry-val">Port 8000</span>
                </div>
                <div class="telemetry-row">
                    <span>Database</span>
                    <span class="telemetry-val">SQLite (7,941)</span>
                </div>
                <div class="telemetry-row">
                    <span>Threshold</span>
                    <span class="telemetry-val">τ = 0.900</span>
                </div>
                <div class="telemetry-row">
                    <span>Engine</span>
                    <span class="telemetry-val">RF + TreeSHAP</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
