"""Real-Time Customer Churn Scoring & SHAP Explainability Dashboard."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app.utils import (
    RISK_COLORS,
    inject_custom_css,
    load_customer_master_data,
    render_header_command_bar,
    render_sidebar_status,
    request_prediction_api,
)

inject_custom_css()
render_sidebar_status()

# 1. Executive Command Bar Header
render_header_command_bar(
    breadcrumb="DECISIONING / REAL-TIME INFERENCE",
    title="Real-Time Scoring & SHAP Diagnostics",
    subtitle="Evaluate churn probabilities, analyze local SHAP feature attributions, and trigger prescriptive retention protocols.",
)

df = load_customer_master_data()

# 2. Input Controls: Lookup Existing vs Manual Simulation
st.markdown('<div class="section-card">', unsafe_allow_html=True)
tab_lookup, tab_manual = st.tabs(["Existing Account Lookup", "Custom Parameter Simulation"])

selected_data = {}

with tab_lookup:
    if not df.empty:
        col_s1, col_s2 = st.columns([1.5, 2])
        with col_s1:
            customer_list = df["CustomerID"].tolist()
            default_index = 5 if len(customer_list) > 10 else 0

            selected_id = st.selectbox(
                "Select Account ID from Database",
                options=customer_list,
                index=default_index,
                format_func=lambda x: f"Account #{x}",
            )

        row = df[df["CustomerID"] == selected_id].iloc[0]

        with col_s2:
            st.markdown(
                f"""
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:10px 16px; margin-top:20px;">
                    <div style="display:flex; justify-content:space-between; font-size:0.82rem; margin-bottom:4px;">
                        <span><b>Segment:</b> {row['segment']}</span>
                        <span><b>Cluster:</b> {row['kmeans_cluster']}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#64748B;">
                        <span>Spend: <b>${row['monetary']:,.2f}</b></span>
                        <span>Orders: <b>{row['frequency']}</b></span>
                        <span>Recency: <b>{row['recency']} days</b></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        selected_data = row.to_dict()

with tab_manual:
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        sim_recency = st.slider("Recency (Days Inactive)", 1, 365, 45)
        sim_frequency = st.number_input(
            "Lifetime Order Frequency", min_value=1, max_value=250, value=6
        )
        sim_monetary = st.number_input(
            "Total Monetary Spend ($)", min_value=1.0, max_value=50000.0, value=350.0
        )

    with col_m2:
        sim_tenure = st.slider("Customer Tenure (Days)", 15, 730, 240)
        sim_orders_month = st.slider("Order Velocity (Orders / Month)", 0.05, 10.0, 0.75, step=0.05)
        sim_returns = st.slider("Return Rate (%)", 0.0, 50.0, 2.0, step=0.5) / 100.0

    with col_m3:
        sim_rfm = st.number_input("RFM Score (e.g. 444)", min_value=111, max_value=555, value=344)
        sim_segment = st.selectbox(
            "Customer Segment",
            options=[
                "Champions",
                "Loyal Customers",
                "Potential Loyalist",
                "New Customers",
                "Promising",
                "Need Attention",
                "At Risk",
                "Can't Lose Them",
                "Hibernating",
                "Lost",
            ],
            index=1,
        )
        sim_cluster = st.selectbox(
            "Structural Cluster",
            options=[
                "High-Value Loyalists",
                "Core Retail Customers",
                "Dormant / High Churn Risk",
                "Engaged Frequent Buyers",
            ],
            index=1,
        )

    if st.button("Run Simulation with Parameters", use_container_width=True):
        selected_data = {
            "CustomerID": 99999,
            "recency": sim_recency,
            "recency_days": sim_recency,
            "frequency": sim_frequency,
            "monetary": sim_monetary,
            "avg_order_value": round(sim_monetary / max(sim_frequency, 1), 2),
            "tenure_days": sim_tenure,
            "orders_per_month": sim_orders_month,
            "cancellations": int(round(sim_frequency * sim_returns)),
            "return_rate": sim_returns,
            "rfm_score": sim_rfm,
            "segment": sim_segment,
            "kmeans_cluster": sim_cluster,
        }

st.markdown("</div>", unsafe_allow_html=True)

# 3. Trigger Prediction & Render Diagnosis
if selected_data:
    with st.spinner("Executing model inference & TreeSHAP kernel..."):
        prediction = request_prediction_api(selected_data)

    p_churn = float(prediction["churn_probability"])
    p_pct = float(prediction.get("churn_percentage", round(p_churn * 100, 1)))
    risk_band = prediction["risk_band"]
    band_color = RISK_COLORS.get(risk_band, "#2563EB")
    cust_rar = float(prediction["revenue_at_risk"])
    rec_action = prediction["recommended_action"]
    drivers = prediction.get("top_drivers", [])

    # Results Layout
    col_gauge, col_details = st.columns([1.1, 1.9])

    with col_gauge:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="section-header-row">
                <span class="section-header">Churn Probability Score</span>
                <span class="badge-risk badge-{risk_band.lower()}">{risk_band} Risk</span>
            </div>
            <div class="section-subtitle">Model confidence calibrated at threshold τ = 0.900</div>
            """,
            unsafe_allow_html=True,
        )

        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=p_pct,
                domain={"x": [0, 1], "y": [0, 1]},
                number={
                    "suffix": "%",
                    "font": {
                        "family": "Inter, sans-serif",
                        "size": 42,
                        "color": "#0F172A",
                        "weight": 800,
                    },
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 1,
                        "tickcolor": "#94A3B8",
                        "tickfont": {"size": 10},
                    },
                    "bar": {"color": band_color, "thickness": 0.28},
                    "bgcolor": "#F8FAFC",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 30], "color": "#ECFDF5"},
                        {"range": [30, 60], "color": "#FFFBEB"},
                        {"range": [60, 85], "color": "#FFF1F2"},
                        {"range": [85, 100], "color": "#FEF2F2"},
                    ],
                    "threshold": {
                        "line": {"color": "#0F172A", "width": 3},
                        "thickness": 0.75,
                        "value": 90.0,
                    },
                },
            )
        )
        fig_gauge.update_layout(
            height=260,
            margin={"l": 20, "r": 20, "t": 20, "b": 10},
            paper_bgcolor="#FFFFFF",
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(
            f"""
            <div style="background:#F8FAFC; border-radius:8px; padding:10px 14px; border:1px solid #E2E8F0;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:0.75rem; color:#64748B; font-weight:700; text-transform:uppercase; letter-spacing:0.04em;">Account Capital at Risk</span>
                    <span style="font-size:1.1rem; color:#DC2626; font-weight:800; font-variant-numeric:tabular-nums;">${cust_rar:,.2f}</span>
                </div>
                <div style="font-size:0.75rem; color:#94A3B8; margin-top:3px;">
                    Spend (${selected_data.get('monetary', 0):,.2f}) × P(Churn: {p_churn:.3f})
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_details:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="section-header-row">
                <span class="section-header">SHAP Feature Influence Attribution</span>
                <span class="badge-meta">TreeSHAP Local Drivers</span>
            </div>
            <div class="section-subtitle">Diverging behavioral attributions: Red increases churn risk, Green reinforces retention</div>
            """,
            unsafe_allow_html=True,
        )

        if drivers:
            feat_names = [d["display_name"] for d in drivers][::-1]
            shap_vals = [d["shap_value"] for d in drivers][::-1]
            colors = ["#DC2626" if val > 0 else "#059669" for val in shap_vals]

            fig_shap = go.Figure()
            fig_shap.add_trace(
                go.Bar(
                    y=feat_names,
                    x=shap_vals,
                    orientation="h",
                    marker={"color": colors, "line": {"width": 0}},
                    hovertemplate="<b>%{y}</b><br>SHAP Contribution: %{x:+.4f}<extra></extra>",
                )
            )
            fig_shap.update_layout(
                height=220,
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                margin={"l": 10, "r": 20, "t": 10, "b": 25},
                font={"family": "Inter, sans-serif", "size": 11, "color": "#334155"},
                xaxis={
                    "title": "SHAP Log-Odds Contribution",
                    "zeroline": True,
                    "zerolinecolor": "#0F172A",
                    "zerolinewidth": 1.5,
                    "gridcolor": "#F1F5F9",
                    "tickfont": {"size": 10},
                },
                yaxis={"showgrid": False, "tickfont": {"size": 11}},
            )
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.info("SHAP driver breakdown unavailable for this profile.")

        # Prescriptive Playbook Box
        st.markdown(
            f"""
            <div class="playbook-card">
                <div class="playbook-header">Prescriptive Retention Playbook ({selected_data.get('segment')})</div>
                <div class="playbook-body">{rec_action}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
