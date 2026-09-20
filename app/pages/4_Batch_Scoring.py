"""Batch Customer Churn Scoring Portal & CSV Pipeline."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import streamlit as st

from app.utils import (
    inject_custom_css,
    read_csv_robust,
    render_header_command_bar,
    render_metric_card,
    render_sidebar_status,
)
from src.churn.predict import score_customers

inject_custom_css()
render_sidebar_status()

# 1. Executive Command Bar Header
render_header_command_bar(
    breadcrumb="DECISIONING / BATCH PIPELINE",
    title="Batch Scoring Engine & Capital Exposure",
    subtitle="Process bulk customer account exports for high-throughput churn scoring, risk tiering, and CRM activation.",
)

# 2. Template Download & Format Guide
col_t1, col_t2 = st.columns([2, 1])

with col_t1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-header-row">
            <span class="section-header">Batch Ingestion Specifications</span>
            <span class="badge-meta">Schema Tolerant</span>
        </div>
        <div class="section-subtitle">Supports standard CSV exports from Shopify, Stripe, Salesforce, Snowflake, and Excel</div>
        <div style="font-size: 0.82rem; color: #475569; line-height: 1.6;">
            <b>Supported Encodings:</b> UTF-8, UTF-8-BOM, CP1252 (Windows ANSI), Latin-1, ISO-8859-1.<br>
            <b>Core Attributes:</b> <code>CustomerID</code>, <code>frequency</code>, <code>monetary</code>, <code>tenure_days</code>.<br>
            <b>Engineered Fields:</b> <code>avg_order_value</code>, <code>orders_per_month</code>, <code>return_rate</code>, <code>segment</code>, <code>kmeans_cluster</code> (auto-calculated if omitted).
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_t2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-header-row">
            <span class="section-header">Sample Ingestion Template</span>
            <span class="badge-meta">CSV</span>
        </div>
        <div class="section-subtitle">Download reference dataset formatted for batch scoring</div>
        """,
        unsafe_allow_html=True,
    )

    sample_template = pd.DataFrame(
        [
            {
                "CustomerID": 88001,
                "frequency": 14,
                "monetary": 1450.00,
                "avg_order_value": 103.57,
                "tenure_days": 420,
                "orders_per_month": 1.01,
                "return_rate": 0.02,
                "rfm_score": 555,
                "segment": "Champions",
                "kmeans_cluster": "High-Value Loyalists",
            },
            {
                "CustomerID": 88002,
                "frequency": 2,
                "monetary": 65.50,
                "avg_order_value": 32.75,
                "tenure_days": 180,
                "orders_per_month": 0.34,
                "return_rate": 0.00,
                "rfm_score": 222,
                "segment": "Hibernating",
                "kmeans_cluster": "Dormant / High Churn Risk",
            },
            {
                "CustomerID": 88003,
                "frequency": 6,
                "monetary": 420.00,
                "avg_order_value": 70.00,
                "tenure_days": 290,
                "orders_per_month": 0.63,
                "return_rate": 0.04,
                "rfm_score": 444,
                "segment": "Loyal Customers",
                "kmeans_cluster": "Engaged Frequent Buyers",
            },
        ]
    )
    csv_template_bytes = sample_template.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Sample CSV Template",
        data=csv_template_bytes,
        file_name="churnguard_batch_template.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)


def normalize_uploaded_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Normalizes column names and auto-calculates missing engineered features."""
    df_norm = df_raw.copy()

    # Column aliases mapping to canonical names
    aliases = {
        "customer_id": "CustomerID",
        "customerid": "CustomerID",
        "client_id": "CustomerID",
        "id": "CustomerID",
        "account_id": "CustomerID",
        "spend": "monetary",
        "total_spend": "monetary",
        "revenue": "monetary",
        "sales": "monetary",
        "gmv": "monetary",
        "orders": "frequency",
        "order_count": "frequency",
        "total_orders": "frequency",
        "purchases": "frequency",
        "recency_days": "recency",
        "days_inactive": "recency",
        "tenure": "tenure_days",
        "account_age": "tenure_days",
        "aov": "avg_order_value",
        "average_order_value": "avg_order_value",
        "velocity": "orders_per_month",
        "monthly_orders": "orders_per_month",
        "returns": "cancellations",
        "refund_rate": "return_rate",
        "rfm": "rfm_score",
        "cluster": "kmeans_cluster",
    }

    rename_map = {}
    for col in df_norm.columns:
        clean_key = str(col).strip().lower().replace(" ", "_").replace("-", "_")
        if clean_key in aliases:
            rename_map[col] = aliases[clean_key]
        elif clean_key in [
            "frequency",
            "monetary",
            "recency",
            "tenure_days",
            "avg_order_value",
            "orders_per_month",
            "cancellations",
            "return_rate",
            "rfm_score",
            "segment",
            "kmeans_cluster",
        ]:
            rename_map[col] = clean_key

    df_norm = df_norm.rename(columns=rename_map)

    # Auto-impute or calculate core attributes
    if "CustomerID" not in df_norm.columns:
        df_norm["CustomerID"] = np.arange(10001, 10001 + len(df_norm))

    if "frequency" not in df_norm.columns:
        df_norm["frequency"] = 1
    else:
        df_norm["frequency"] = (
            pd.to_numeric(df_norm["frequency"], errors="coerce").fillna(1).astype(int)
        )

    if "monetary" not in df_norm.columns:
        df_norm["monetary"] = 100.0
    else:
        df_norm["monetary"] = pd.to_numeric(df_norm["monetary"], errors="coerce").fillna(100.0)

    if "tenure_days" not in df_norm.columns:
        df_norm["tenure_days"] = 180
    else:
        df_norm["tenure_days"] = (
            pd.to_numeric(df_norm["tenure_days"], errors="coerce").fillna(180).astype(int)
        )

    if "avg_order_value" not in df_norm.columns:
        df_norm["avg_order_value"] = np.round(
            df_norm["monetary"] / np.maximum(df_norm["frequency"], 1), 2
        )
    else:
        df_norm["avg_order_value"] = pd.to_numeric(
            df_norm["avg_order_value"], errors="coerce"
        ).fillna(df_norm["monetary"] / np.maximum(df_norm["frequency"], 1))

    if "orders_per_month" not in df_norm.columns:
        df_norm["orders_per_month"] = np.round(
            df_norm["frequency"] / np.maximum(df_norm["tenure_days"] / 30.4, 1.0), 3
        )
    else:
        df_norm["orders_per_month"] = pd.to_numeric(
            df_norm["orders_per_month"], errors="coerce"
        ).fillna(0.5)

    if "return_rate" not in df_norm.columns:
        df_norm["return_rate"] = 0.0
    else:
        df_norm["return_rate"] = pd.to_numeric(df_norm["return_rate"], errors="coerce").fillna(0.0)

    if "rfm_score" not in df_norm.columns:
        df_norm["rfm_score"] = 333
    else:
        df_norm["rfm_score"] = (
            pd.to_numeric(df_norm["rfm_score"], errors="coerce").fillna(333).astype(int)
        )

    if "segment" not in df_norm.columns:
        df_norm["segment"] = "Core Retail"

    if "kmeans_cluster" not in df_norm.columns:
        df_norm["kmeans_cluster"] = "Core Retail"

    return df_norm


# 3. File Upload Zone
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="section-header-row">
        <span class="section-header">Batch File Ingestion</span>
        <span class="badge-meta">CSV Auto-Decoder</span>
    </div>
    <div class="section-subtitle">Select or drop customer account files (UTF-8, Windows CP1252, or Latin-1)</div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    label_visibility="collapsed",
    help="Upload customer dataset. Automatically decodes multi-region character sets.",
)

if uploaded_file is not None:
    try:
        raw_df = read_csv_robust(uploaded_file)
        input_df = normalize_uploaded_columns(raw_df)

        st.markdown(
            f"""
            <div style="background:#ECFDF5; border:1px solid #A7F3D0; border-radius:8px; padding:10px 14px; margin-top:10px; font-size:0.85rem; color:#065F46; font-weight:600;">
                ✓ Successfully parsed {len(input_df):,} customer records from <code>{uploaded_file.name}</code>. Multi-encoding decoder resolved all character sets.
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Inspect Ingested Data Preview (First 5 Rows)", expanded=False):
            st.dataframe(input_df.head(5), use_container_width=True)

        if st.button("Execute Batch Churn Scoring Pipeline", use_container_width=True):
            progress_bar = st.progress(0, text="Initializing ML Pipeline & Column Transformers...")
            time.sleep(0.15)
            progress_bar.progress(35, text="Applying standard scaling and one-hot encoding...")
            time.sleep(0.15)
            progress_bar.progress(
                70, text="Evaluating tree ensemble probabilities and revenue exposure..."
            )

            scored_df = score_customers(input_df)

            progress_bar.progress(100, text="Batch scoring complete.")
            time.sleep(0.2)
            progress_bar.empty()

            # Batch Summary KPI Cards
            tot_scored = len(scored_df)
            tot_rar = scored_df["revenue_at_risk"].sum()
            high_risk_count = (scored_df["risk_band"] == "High").sum()
            crit_risk_count = (scored_df["risk_band"] == "Critical").sum()

            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            b_col1, b_col2, b_col3, b_col4 = st.columns(4)
            with b_col1:
                st.markdown(
                    render_metric_card(
                        "Scored Accounts",
                        f"{tot_scored:,}",
                        "Profiles Scored",
                        "100% Complete",
                        "neutral",
                        "Batch",
                    ),
                    unsafe_allow_html=True,
                )
            with b_col2:
                st.markdown(
                    render_metric_card(
                        "Capital at Risk",
                        f"${tot_rar:,.2f}",
                        "Total Exposure",
                        "Risk Weighted",
                        "negative",
                        "Revenue",
                    ),
                    unsafe_allow_html=True,
                )
            with b_col3:
                st.markdown(
                    render_metric_card(
                        "High Risk Cohort",
                        f"{high_risk_count:,}",
                        f"{(high_risk_count/tot_scored)*100:.1f}% of batch",
                        "Elevated",
                        "negative",
                        "Watch",
                    ),
                    unsafe_allow_html=True,
                )
            with b_col4:
                st.markdown(
                    render_metric_card(
                        "Critical Risk Cohort",
                        f"{crit_risk_count:,}",
                        f"{(crit_risk_count/tot_scored)*100:.1f}% of batch",
                        "Urgent",
                        "negative",
                        "Action",
                    ),
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

            # Results Table
            st.markdown(
                """
                <div class="section-header-row">
                    <span class="section-header">Batch Scoring Results Matrix</span>
                    <span class="badge-meta">Enriched Outputs</span>
                </div>
                <div class="section-subtitle">Accounts ranked by predicted churn probability and dollar capital exposure</div>
                """,
                unsafe_allow_html=True,
            )

            display_cols = [
                "CustomerID",
                "segment",
                "monetary",
                "frequency",
                "churn_probability",
                "risk_band",
                "revenue_at_risk",
            ]
            valid_cols = [c for c in display_cols if c in scored_df.columns]
            table_view = (
                scored_df[valid_cols].sort_values("churn_probability", ascending=False).copy()
            )

            table_view["monetary"] = table_view["monetary"].apply(lambda x: f"${x:,.2f}")
            table_view["churn_probability"] = table_view["churn_probability"].apply(
                lambda x: f"{x*100:.1f}%"
            )
            table_view["revenue_at_risk"] = table_view["revenue_at_risk"].apply(
                lambda x: f"${x:,.2f}"
            )

            st.dataframe(table_view, use_container_width=True, hide_index=True)

            # Download Enriched CSV
            scored_csv = scored_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Export Scored Batch Results (CSV)",
                data=scored_csv,
                file_name=f"churnguard_scored_batch_{int(time.time())}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    except Exception as e:
        st.markdown(
            f"""
            <div style="background:#FEF2F2; border:1px solid #FCA5A5; border-radius:8px; padding:14px; font-size:0.88rem; color:#991B1B;">
                <b>Ingestion Notice:</b> Unable to parse CSV payload. Ensure the file contains valid comma-separated values.<br>
                <span style="font-size:0.78rem; color:#B91C1C;">Diagnostic Details: {e}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("</div>", unsafe_allow_html=True)
