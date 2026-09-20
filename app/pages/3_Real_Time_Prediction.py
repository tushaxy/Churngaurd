"""Real-Time Customer Churn Scoring & SHAP Explainability Dashboard."""

import plotly.graph_objects as go
import streamlit as st

from app.utils import (
    RISK_COLORS,
    inject_custom_css,
    load_customer_master_data,
    render_sidebar_status,
    request_prediction_api,
)

inject_custom_css()
render_sidebar_status()

st.markdown(
    """
    <div style="margin-bottom: 1.5rem;">
        <h1 style="font-size: 2.2rem; font-weight: 800; color: #0F172A; margin-bottom: 0.25rem;">
            Real-Time Churn Scoring & Explainability
        </h1>
        <p style="color: #64748B; font-size: 1.05rem;">
            Lookup an existing customer profile or simulate custom metrics to diagnose churn risk with SHAP explainability.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

df = load_customer_master_data()

# 1. Input Controls: Lookup Existing vs Manual Simulation
tab_lookup, tab_manual = st.tabs(["🔍 Lookup Existing Customer", "⚙️ Manual Feature Simulation"])

selected_data = {}

with tab_lookup:
    if not df.empty:
        col_s1, col_s2 = st.columns([1.5, 2])
        with col_s1:
            customer_list = df["CustomerID"].tolist()
            # Default to an interesting customer
            default_index = 0
            if len(customer_list) > 10:
                # pick a customer with moderate recency
                default_index = 5

            selected_id = st.selectbox(
                "Select CustomerID from Database",
                options=customer_list,
                index=default_index,
                format_func=lambda x: f"Customer #{x}",
            )

        row = df[df["CustomerID"] == selected_id].iloc[0]

        with col_s2:
            st.markdown(
                f"""
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:12px 18px; margin-top:24px;">
                    <div style="display:flex; justify-content:space-between; font-size:0.9rem; margin-bottom:4px;">
                        <span><b>Segment:</b> {row['segment']}</span>
                        <span><b>Cluster:</b> {row['kmeans_cluster']}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.9rem; color:#64748B;">
                        <span>Spend: ${row['monetary']:,.2f}</span>
                        <span>Orders: {row['frequency']}</span>
                        <span>Recency: {row['recency']} days</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        selected_data = row.to_dict()

with tab_manual:
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        sim_recency = st.slider("Recency (Days since last purchase)", 1, 365, 45)
        sim_frequency = st.number_input("Frequency (# Orders)", min_value=1, max_value=250, value=6)
        sim_monetary = st.number_input(
            "Monetary Total Spend ($)", min_value=1.0, max_value=50000.0, value=350.0
        )

    with col_m2:
        sim_tenure = st.slider("Customer Tenure (Days)", 15, 730, 240)
        sim_orders_month = st.slider("Orders per Month", 0.05, 10.0, 0.75, step=0.05)
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
            "K-Means Cluster",
            options=[
                "High-Value Loyalists",
                "Engaged Frequent Buyers",
                "Occasional / New Shoppers",
                "Dormant / High Churn Risk",
            ],
            index=1,
        )

    if st.button("Simulate Custom Profile"):
        selected_data = {
            "CustomerID": 99999,
            "recency": sim_recency,
            "frequency": sim_frequency,
            "monetary": sim_monetary,
            "avg_order_value": round(sim_monetary / sim_frequency, 2),
            "tenure_days": sim_tenure,
            "orders_per_month": sim_orders_month,
            "return_rate": sim_returns,
            "rfm_score": sim_rfm,
            "segment": sim_segment,
            "kmeans_cluster": sim_cluster,
        }

st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

# 2. Trigger Prediction & Render Diagnosis
if selected_data:
    with st.spinner("Analyzing customer behavioral signals with tuned model & SHAP..."):
        prediction = request_prediction_api(selected_data)

    p_churn = prediction["churn_probability"]
    p_pct = prediction.get("churn_percentage", round(p_churn * 100, 1))
    risk_band = prediction["risk_band"]
    band_color = RISK_COLORS.get(risk_band, "#4F46E5")
    cust_rar = prediction["revenue_at_risk"]
    rec_action = prediction["recommended_action"]
    drivers = prediction.get("top_drivers", [])

    # Results Layout
    col_gauge, col_details = st.columns([1, 1.5])

    with col_gauge:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                <span class="section-header">Churn Risk Score</span>
                <span class="badge-risk badge-{risk_band.lower()}">{risk_band.upper()} RISK</span>
            </div>
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
                        "family": "Plus Jakarta Sans",
                        "size": 48,
                        "color": "#0F172A",
                        "weight": 800,
                    },
                },
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
                    "bar": {"color": band_color, "thickness": 0.3},
                    "bgcolor": "#F1F5F9",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 30], "color": "#ECFDF5"},
                        {"range": [30, 60], "color": "#FFFBEB"},
                        {"range": [60, 85], "color": "#FEF2F2"},
                        {"range": [85, 100], "color": "#FEE2E2"},
                    ],
                    "threshold": {
                        "line": {"color": "#0F172A", "width": 4},
                        "thickness": 0.8,
                        "value": 90.0,
                    },
                },
            )
        )
        fig_gauge.update_layout(
            height=280,
            margin={"l": 25, "r": 25, "t": 30, "b": 10},
            paper_bgcolor="#FFFFFF",
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(
            f"""
            <div style="background:#F8FAFC; border-radius:10px; padding:12px 16px; margin-top:10px; border:1px solid #E2E8F0;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:0.85rem; color:#64748B; font-weight:600;">INDIVIDUAL REVENUE AT RISK</span>
                    <span style="font-size:1.15rem; color:#EF4444; font-weight:800;">${cust_rar:,.2f}</span>
                </div>
                <div style="font-size:0.8rem; color:#94A3B8; margin-top:4px;">
                    Calculated as Monetary (${selected_data.get('monetary', 0):,.2f}) × P(Churn: {p_churn:.2f})
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_details:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-header">SHAP Feature Influence Breakdown</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-subtitle">Diverging drivers pushing towards churn (Red) vs. retention (Green)</div>',
            unsafe_allow_html=True,
        )

        if drivers:
            # Build horizontal diverging bar chart
            feat_names = [d["display_name"] for d in drivers][::-1]
            shap_vals = [d["shap_value"] for d in drivers][::-1]
            colors = ["#EF4444" if val > 0 else "#10B981" for val in shap_vals]

            fig_shap = go.Figure()
            fig_shap.add_trace(
                go.Bar(
                    y=feat_names,
                    x=shap_vals,
                    orientation="h",
                    marker={"color": colors, "line": {"width": 0}},
                    hovertemplate="<b>%{y}</b><br>SHAP Impact: %{x:+.4f}<extra></extra>",
                )
            )
            fig_shap.update_layout(
                height=260,
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                margin={"l": 10, "r": 20, "t": 10, "b": 30},
                font={"family": "Plus Jakarta Sans", "size": 11, "color": "#334155"},
                xaxis={
                    "title": "SHAP Value (Impact on Model Log-Odds)",
                    "zeroline": True,
                    "zerolinecolor": "#0F172A",
                    "zerolinewidth": 2,
                    "gridcolor": "#F1F5F9",
                },
                yaxis={"showgrid": False},
            )
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.info("SHAP driver breakdown unavailable for this profile.")

        # Prescriptive Playbook Box
        st.markdown(
            f"""
            <div class="playbook-box">
                <div class="playbook-title">🎯 Prescriptive Retention Playbook ({selected_data.get('segment')})</div>
                <div class="playbook-text">{rec_action}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
