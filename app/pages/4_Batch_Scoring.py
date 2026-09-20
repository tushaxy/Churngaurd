"""Batch Customer Churn Scoring Portal & CSV Pipeline."""

import time

import pandas as pd
import streamlit as st

from app.utils import (
    inject_custom_css,
    render_metric_card,
    render_sidebar_status,
)
from src.churn.predict import score_customers

inject_custom_css()
render_sidebar_status()

st.markdown(
    """
    <div style="margin-bottom: 1.5rem;">
        <h1 style="font-size: 2.2rem; font-weight: 800; color: #0F172A; margin-bottom: 0.25rem;">
            Batch Churn Scoring Engine
        </h1>
        <p style="color: #64748B; font-size: 1.05rem;">
            Upload bulk customer transaction profiles for high-throughput batch scoring, risk tiering, and CSV export.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# 1. Template Download & Instructions
col_t1, col_t2 = st.columns([2, 1])

with col_t1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-header">Batch File Specifications</div>', unsafe_allow_html=True
    )
    st.markdown(
        """
        Your CSV file should include the following customer features:
        `CustomerID`, `frequency`, `monetary`, `avg_order_value`, `tenure_days`, `orders_per_month`, `return_rate`, `rfm_score`, `segment`, `kmeans_cluster`.
        Missing columns will fail schema validation.
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_t2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Sample CSV Template</div>', unsafe_allow_html=True)

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
        label="📥 Download Template CSV",
        data=csv_template_bytes,
        file_name="churnguard_batch_template.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

# 2. File Upload Zone
uploaded_file = st.file_uploader(
    "Drag and drop customer CSV file to score",
    type=["csv"],
    help="Upload a CSV with required RFM and behavioral metrics.",
)

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
        st.success(f"Successfully loaded {len(input_df):,} rows from '{uploaded_file.name}'.")

        # Validate required columns
        required_cols = [
            "frequency",
            "monetary",
            "avg_order_value",
            "tenure_days",
            "orders_per_month",
            "return_rate",
            "rfm_score",
            "segment",
            "kmeans_cluster",
        ]
        missing = [c for c in required_cols if c not in input_df.columns]

        if missing:
            st.error(f"Validation Error: Uploaded file is missing required columns: {missing}")
            st.stop()

        with st.expander("Preview Uploaded Data (First 5 Rows)", expanded=False):
            st.dataframe(input_df.head(5), use_container_width=True)

        if st.button("🚀 Execute Batch Scoring Pipeline", use_container_width=True):
            progress_bar = st.progress(0, text="Initializing ML Pipeline...")
            time.sleep(0.2)
            progress_bar.progress(30, text="Applying feature transformations...")
            time.sleep(0.2)
            progress_bar.progress(70, text="Predicting churn probabilities & risk bands...")

            scored_df = score_customers(input_df)

            progress_bar.progress(100, text="Batch scoring complete!")
            time.sleep(0.3)
            progress_bar.empty()

            st.balloons()

            # Batch Summary KPI Cards
            tot_scored = len(scored_df)
            tot_rar = scored_df["revenue_at_risk"].sum()
            high_risk_count = (scored_df["risk_band"] == "High").sum()
            crit_risk_count = (scored_df["risk_band"] == "Critical").sum()

            b_col1, b_col2, b_col3, b_col4 = st.columns(4)
            with b_col1:
                st.markdown(
                    render_metric_card(
                        "Total Scored", f"{tot_scored:,}", "Profiles Processed", "neutral"
                    ),
                    unsafe_allow_html=True,
                )
            with b_col2:
                st.markdown(
                    render_metric_card(
                        "Revenue at Risk", f"${tot_rar:,.2f}", "Total Value at Risk", "negative"
                    ),
                    unsafe_allow_html=True,
                )
            with b_col3:
                st.markdown(
                    render_metric_card(
                        "High Risk Accounts",
                        f"{high_risk_count:,}",
                        f"{(high_risk_count/tot_scored)*100:.1f}% of batch",
                        "negative",
                    ),
                    unsafe_allow_html=True,
                )
            with b_col4:
                st.markdown(
                    render_metric_card(
                        "Critical Risk Accounts",
                        f"{crit_risk_count:,}",
                        f"{(crit_risk_count/tot_scored)*100:.1f}% of batch",
                        "negative",
                    ),
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

            # Results Table with styling
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-header">Batch Prediction Results</div>', unsafe_allow_html=True
            )
            st.markdown(
                '<div class="section-subtitle">Sorted by individual revenue at risk (highest first)</div>',
                unsafe_allow_html=True,
            )

            display_cols = [
                "CustomerID",
                "segment",
                "churn_probability",
                "risk_band",
                "monetary",
                "revenue_at_risk",
                "frequency",
                "avg_order_value",
            ]
            final_display = scored_df[display_cols].sort_values("revenue_at_risk", ascending=False)

            # Conditional color styling
            styled_table = final_display.style.background_gradient(
                subset=["churn_probability"], cmap="YlOrRd", vmin=0.0, vmax=1.0
            ).format(
                {
                    "churn_probability": "{:.2%}",
                    "monetary": "${:,.2f}",
                    "revenue_at_risk": "${:,.2f}",
                    "avg_order_value": "${:,.2f}",
                }
            )
            st.dataframe(styled_table, use_container_width=True, height=380)

            # Download Scored CSV
            scored_csv = scored_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Scored Batch Results (CSV)",
                data=scored_csv,
                file_name=f"scored_batch_{int(time.time())}.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error processing CSV: {e}")
