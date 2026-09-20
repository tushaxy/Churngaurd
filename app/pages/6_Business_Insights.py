"""Strategic Business Insights, Pareto Concentration & What-If Revenue Simulator."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.utils import (
    COLOR_CORAL,
    COLOR_PRIMARY,
    SEGMENT_COLORS,
    get_plotly_layout,
    inject_custom_css,
    load_customer_master_data,
    render_metric_card,
    render_sidebar_status,
)

inject_custom_css()
render_sidebar_status()

st.markdown(
    """
    <div style="margin-bottom: 1.5rem;">
        <h1 style="font-size: 2.2rem; font-weight: 800; color: #0F172A; margin-bottom: 0.25rem;">
            Strategic Business Insights & ROI Simulator
        </h1>
        <p style="color: #64748B; font-size: 1.05rem;">
            Verify Pareto revenue concentration, analyze segment vulnerability, and model the financial ROI of targeted retention plays.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

df = load_customer_master_data()

if df.empty:
    st.error("Dataset not found. Please run `make pipeline` first.")
    st.stop()

total_customers = len(df)
total_revenue = float(df["monetary"].sum())

# 1. Pareto Concentration Analysis
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown(
    '<div class="section-header">Pareto Revenue Concentration (80/20 Analysis)</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-subtitle">Empirical cumulative revenue distribution across ranked customer accounts</div>',
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

# Callout banner
st.markdown(
    f"""
    <div style="background:linear-gradient(90deg, #EEF2FF 0%, #E0E7FF 100%); border-left:5px solid #4F46E5; border-radius:10px; padding:16px 20px; margin-bottom:18px;">
        <div style="font-size:1.15rem; font-weight:800; color:#1E1B4B; margin-bottom:4px;">
            💎 Empirical Pareto Finding: Top 20% of Customers Drive {top_20_share:.2f}% of Revenue
        </div>
        <div style="font-size:0.92rem; color:#3730A3; line-height:1.5;">
            Out of <b>{total_customers:,}</b> accounts, the top <b>{top_20_count:,}</b> customers account for <b>${top_20_rev:,.2f}</b> in spend.
            The top 10% ({top_10_count:,} accounts) command <b>{top_10_share:.2f}%</b> of total revenue.
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
        line={"color": COLOR_PRIMARY, "width": 3},
        hovertemplate="Top %{x:.1f}% of Customers<br>Generate %{y:.1f}% of Revenue<extra></extra>",
    )
)

# 45-degree reference line (Equal distribution)
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
        marker={"size": 10, "color": COLOR_CORAL},
        text=[f"20% → {top_20_share:.1f}%"],
        textposition="top left",
        textfont={"size": 12, "color": COLOR_CORAL, "family": "Plus Jakarta Sans"},
    )
)

p_layout = get_plotly_layout(height=340, show_legend=True)
p_layout["xaxis"]["title"] = "Cumulative % of Customers (Ranked by Total Spend)"
p_layout["yaxis"]["title"] = "Cumulative % of Total Revenue"
fig_pareto.update_layout(p_layout)
st.plotly_chart(fig_pareto, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# 2. Revenue at Risk by Segment Breakdown
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown(
    '<div class="section-header">Revenue at Risk Distribution Across Segments</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-subtitle">Identifying which customer tiers contain the highest aggregate dollar exposure</div>',
    unsafe_allow_html=True,
)

seg_rar = (
    df.groupby("segment")
    .agg(
        total_revenue=("monetary", "sum"),
        revenue_at_risk=("revenue_at_risk", "sum"),
        churned_accounts=("churn_prediction", "sum"),
        total_accounts=("CustomerID", "count"),
    )
    .reset_index()
    .sort_values("revenue_at_risk", ascending=False)
)

col_bar, col_table = st.columns([1.5, 1])

with col_bar:
    fig_seg_rar = go.Figure()
    fig_seg_rar.add_trace(
        go.Bar(
            x=seg_rar["segment"],
            y=seg_rar["revenue_at_risk"],
            marker={"color": [SEGMENT_COLORS.get(s, COLOR_PRIMARY) for s in seg_rar["segment"]]},
            hovertemplate="<b>%{x}</b><br>Revenue at Risk: $%{y:,.2f}<extra></extra>",
        )
    )
    bar_layout = get_plotly_layout(height=320, show_legend=False)
    bar_layout["xaxis"]["title"] = ""
    bar_layout["xaxis"]["tickangle"] = -35
    bar_layout["yaxis"]["title"] = "Revenue at Risk ($)"
    fig_seg_rar.update_layout(bar_layout)
    st.plotly_chart(fig_seg_rar, use_container_width=True)

with col_table:
    seg_rar_display = seg_rar.copy()
    seg_rar_display["Risk %"] = (
        seg_rar_display["revenue_at_risk"] / seg_rar_display["total_revenue"]
    ) * 100
    seg_rar_display = seg_rar_display.rename(
        columns={
            "segment": "Segment",
            "revenue_at_risk": "Rev at Risk ($)",
            "total_revenue": "Total Rev ($)",
            "churned_accounts": "At-Risk Accts",
        }
    )
    st.dataframe(
        seg_rar_display[["Segment", "Rev at Risk ($)", "Risk %", "At-Risk Accts"]].style.format(
            {
                "Rev at Risk ($)": "${:,.0f}",
                "Risk %": "{:.1f}%",
                "At-Risk Accts": "{:,}",
            }
        ),
        use_container_width=True,
        height=300,
    )

st.markdown("</div>", unsafe_allow_html=True)

# 3. Interactive What-If Simulator
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown(
    '<div class="section-header">Interactive What-If Retention ROI Simulator</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-subtitle">Simulate the revenue recovered by reducing churn across vulnerable customer segments</div>',
    unsafe_allow_html=True,
)

sim_c1, sim_c2 = st.columns([1.2, 1.8])

with sim_c1:
    target_segment = st.selectbox(
        "Select Target Customer Segment",
        options=["At Risk", "Can't Lose Them", "Need Attention", "All High-Risk Accounts"],
        index=0,
    )

    reduction_pct = st.slider(
        "Projected Churn Reduction (%)",
        min_value=5,
        max_value=60,
        value=20,
        step=5,
        help="Expected success rate of targeted retention playbooks (e.g. VIP concierge outreach or win-back campaigns).",
    )

    intervention_cost_per_cust = st.number_input(
        "Estimated Cost per Intervention ($)",
        min_value=0.0,
        max_value=100.0,
        value=15.0,
        step=5.0,
        help="Marketing / voucher / direct outreach cost allocated per customer.",
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
                "Recovered Revenue",
                f"${recovered_revenue:,.0f}",
                f"+{reduction_pct}% Churn Reduction",
                "positive",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            render_metric_card(
                "Accounts Saved",
                f"{accounts_saved:,}",
                f"From {total_pool_cust:,} at risk",
                "positive",
            ),
            unsafe_allow_html=True,
        )

    with res_col2:
        st.markdown(
            render_metric_card(
                "Net Financial Return",
                f"${net_roi_dollars:,.0f}",
                f"After ${campaign_cost:,.0f} spend",
                "positive",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            render_metric_card(
                "Campaign ROI Multiple", f"{roi_multiple:.1f}x", "Gross Return on Spend", "positive"
            ),
            unsafe_allow_html=True,
        )

st.markdown("</div>", unsafe_allow_html=True)
