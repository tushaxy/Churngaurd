"""Strategic Business Insights, Pareto Concentration & What-If Revenue Simulator."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.utils import (
    get_plotly_layout,
    inject_custom_css,
    load_customer_master_data,
    render_header_command_bar,
    render_metric_card,
    render_sidebar_status,
)

inject_custom_css()
render_sidebar_status()

# 1. Executive Command Bar Header
render_header_command_bar(
    breadcrumb="REVENUE INTELLIGENCE / ROI SIMULATOR",
    title="Strategic Business Insights & Retention Simulator",
    subtitle="Validate Pareto revenue concentration, evaluate cohort financial exposure, and model retention ROI.",
)

df = load_customer_master_data()

if df.empty:
    st.error("Dataset not found. Please run pipeline to seed database.")
    st.stop()

total_customers = len(df)
total_revenue = float(df["monetary"].sum())

# 2. Pareto Concentration Analysis
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="section-header-row">
        <span class="section-header">Pareto Revenue Concentration (80/20 Analysis)</span>
        <span class="badge-meta">Cumulative Distribution</span>
    </div>
    <div class="section-subtitle">Empirical cumulative revenue distribution across descending ranked customer accounts</div>
    """,
    unsafe_allow_html=True,
)

# Sort customers by monetary descending
sorted_df = df.sort_values("monetary", ascending=False).reset_index(drop=True)
sorted_df["cumulative_revenue"] = sorted_df["monetary"].cumsum()
sorted_df["cumulative_share"] = (sorted_df["cumulative_revenue"] / total_revenue) * 100
sorted_df["customer_percentile"] = ((sorted_df.index + 1) / total_customers) * 100

top_20_count = int(np.ceil(0.20 * total_customers))
top_20_rev = sorted_df.iloc[:top_20_count]["monetary"].sum()
top_20_share = (top_20_rev / total_revenue) * 100

top_10_count = int(np.ceil(0.10 * total_customers))
top_10_share = (sorted_df.iloc[:top_10_count]["monetary"].sum() / total_revenue) * 100

# Executive Callout Card
st.markdown(
    f"""
    <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-left:4px solid #2563EB; border-radius:8px; padding:14px 18px; margin-bottom:16px;">
        <div style="font-size:0.75rem; font-weight:700; color:#2563EB; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px;">
            Institutional Revenue Concentration Benchmark
        </div>
        <div style="font-size:1.05rem; font-weight:800; color:#0F172A; margin-bottom:4px;">
            Top 20% of Accounts ({top_20_count:,} customers) Generate {top_20_share:.2f}% (${top_20_rev:,.2f}) of Total GMV
        </div>
        <div style="font-size:0.84rem; color:#475569; line-height:1.5;">
            The top 10% ({top_10_count:,} accounts) command <b>{top_10_share:.2f}%</b> of total portfolio revenue.
            This heavy-tailed distribution underscores that churn mitigation must aggressively protect top-tier relationships.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Pareto Plot
step = max(1, len(sorted_df) // 100)
pareto_sampled = sorted_df.iloc[::step]

fig_pareto = go.Figure()

# Cumulative Share Line
fig_pareto.add_trace(
    go.Scatter(
        x=pareto_sampled["customer_percentile"],
        y=pareto_sampled["cumulative_share"],
        name="Cumulative Revenue %",
        mode="lines",
        line={"color": "#2563EB", "width": 2.5},
        hovertemplate="Top %{x:.1f}% of Accounts<br>Generate %{y:.1f}% of GMV<extra></extra>",
    )
)

# 45-degree reference line
fig_pareto.add_trace(
    go.Scatter(
        x=[0, 100],
        y=[0, 100],
        name="Equal Distribution (1:1)",
        mode="lines",
        line={"color": "#CBD5E1", "width": 1.5, "dash": "dash"},
    )
)

# Marker at 20%
fig_pareto.add_trace(
    go.Scatter(
        x=[20.0],
        y=[top_20_share],
        mode="markers+text",
        name="Top 20% Anchor",
        marker={"size": 8, "color": "#DC2626"},
        text=[f"20% : {top_20_share:.1f}%"],
        textposition="top left",
        textfont={"family": "Inter", "size": 11, "color": "#0F172A"},
    )
)

pareto_layout = get_plotly_layout(height=340, show_legend=True)
pareto_layout["xaxis"]["title"] = "Cumulative Percentage of Customer Accounts (%)"
pareto_layout["yaxis"]["title"] = "Cumulative Percentage of Total GMV (%)"
fig_pareto.update_layout(pareto_layout)
st.plotly_chart(fig_pareto, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# 3. Interactive What-If Retention Simulator
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="section-header-row">
        <span class="section-header">What-If Retention ROI Simulator</span>
        <span class="badge-meta">Financial Modeling</span>
    </div>
    <div class="section-subtitle">Simulate capital recovered by deploying targeted intervention plays across vulnerable cohorts</div>
    """,
    unsafe_allow_html=True,
)

sim_c1, sim_c2 = st.columns([1.2, 1.8])

with sim_c1:
    target_segment = st.selectbox(
        "Target Customer Cohort",
        options=["At Risk", "Can't Lose Them", "Need Attention", "All High-Risk Accounts"],
        index=0,
    )

    reduction_pct = st.slider(
        "Projected Churn Reduction (%)",
        min_value=5,
        max_value=60,
        value=20,
        step=5,
        help="Anticipated success rate of intervention playbooks (e.g. VIP concierge outreach or reactivation vouchers).",
    )

    intervention_cost_per_cust = st.number_input(
        "Intervention Cost per Customer ($)",
        min_value=0.0,
        max_value=100.0,
        value=15.0,
        step=5.0,
        help="Direct outreach, customer success labor, or voucher discount cost per account.",
    )

with sim_c2:
    if target_segment == "All High-Risk Accounts":
        target_pool = df[df["risk_band"].isin(["High", "Critical"])].copy()
    else:
        target_pool = df[df["segment"] == target_segment].copy()

    total_pool_cust = len(target_pool)
    pool_rar = target_pool["revenue_at_risk"].sum()
    accounts_saved = int(round(total_pool_cust * (reduction_pct / 100.0)))
    recovered_revenue = pool_rar * (reduction_pct / 100.0)
    campaign_cost = accounts_saved * intervention_cost_per_cust
    net_roi_dollars = max(0.0, recovered_revenue - campaign_cost)
    roi_multiple = (recovered_revenue / campaign_cost) if campaign_cost > 0 else 0.0

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.markdown(
            render_metric_card(
                "Recovered Capital",
                f"${recovered_revenue:,.0f}",
                f"+{reduction_pct}% Churn Reduction",
                "Gross Recovery",
                "positive",
                "ROI",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            render_metric_card(
                "Accounts Retained",
                f"{accounts_saved:,}",
                f"From {total_pool_cust:,} at risk",
                "Saved Base",
                "positive",
                "Accounts",
            ),
            unsafe_allow_html=True,
        )

    with res_col2:
        st.markdown(
            render_metric_card(
                "Net Financial Return",
                f"${net_roi_dollars:,.0f}",
                f"After ${campaign_cost:,.0f} spend",
                "Net Profit",
                "positive",
                "Capital",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            render_metric_card(
                "Campaign ROI Multiple",
                f"{roi_multiple:.1f}x",
                "Gross Return on Spend",
                "Capital Efficiency",
                "positive",
                "Multiple",
            ),
            unsafe_allow_html=True,
        )

st.markdown("</div>", unsafe_allow_html=True)
