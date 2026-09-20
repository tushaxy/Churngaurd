"""ChurnGuard Enterprise: Customer Segmentation & Churn Intelligence Platform.

Main application entrypoint configuring multipage st.navigation and enterprise design tokens.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="ChurnGuard Enterprise — Retention Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enterprise multipage navigation grouped by functional domains
pages = {
    "REVENUE INTELLIGENCE": [
        st.Page(
            "pages/1_Executive_Overview.py",
            title="Executive Overview",
            icon=":material/dashboard:",
            default=True,
        ),
        st.Page(
            "pages/2_Customer_Segmentation.py",
            title="Customer Segmentation",
            icon=":material/pie_chart:",
        ),
        st.Page(
            "pages/6_Business_Insights.py",
            title="Business Insights & ROI",
            icon=":material/insights:",
        ),
    ],
    "DECISIONING & ML OPS": [
        st.Page(
            "pages/3_Real_Time_Prediction.py",
            title="Real-Time Scoring",
            icon=":material/bolt:",
        ),
        st.Page(
            "pages/4_Batch_Scoring.py",
            title="Batch Scoring Engine",
            icon=":material/upload_file:",
        ),
        st.Page(
            "pages/5_Model_Performance.py",
            title="Model Diagnostics",
            icon=":material/monitoring:",
        ),
    ],
}

pg = st.navigation(pages)
pg.run()
