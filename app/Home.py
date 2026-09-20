"""ChurnGuard: Customer Segmentation & Churn Prediction System.

Main application entrypoint configuring multipage st.navigation and global design tokens.
"""

import streamlit as st

st.set_page_config(
    page_title="ChurnGuard — Retention Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define multipage navigation using st.Page
pages = [
    st.Page(
        "pages/1_Executive_Overview.py",
        title="Executive Overview",
        icon="🏠",
        default=True,
    ),
    st.Page(
        "pages/2_Customer_Segmentation.py",
        title="Customer Segmentation",
        icon="👥",
    ),
    st.Page(
        "pages/3_Real_Time_Prediction.py",
        title="Real-Time Prediction",
        icon="🎯",
    ),
    st.Page(
        "pages/4_Batch_Scoring.py",
        title="Batch Scoring",
        icon="📦",
    ),
    st.Page(
        "pages/5_Model_Performance.py",
        title="Model Performance",
        icon="📈",
    ),
    st.Page(
        "pages/6_Business_Insights.py",
        title="Business Insights",
        icon="💼",
    ),
]

pg = st.navigation(pages)
pg.run()
