"""Customer Segmentation Explorer & RFM Strategy Playbook."""

from __future__ import annotations

import numpy as np
import plotly.express as px
import streamlit as st

from app.utils import (
    SEGMENT_COLORS,
    get_plotly_layout,
    inject_custom_css,
    load_customer_master_data,
    render_header_command_bar,
    render_sidebar_status,
)

inject_custom_css()
render_sidebar_status()

# 1. Executive Command Bar Header
render_header_command_bar(
    breadcrumb="REVENUE INTELLIGENCE / SEGMENTATION",
    title="Customer Segmentation & Behavioral Matrix",
    subtitle="Multidimensional RFM clustering, behavioral distribution surfaces, and prescriptive account retention playbooks.",
)

df = load_customer_master_data()

if df.empty:
    st.error("No customer records found. Please run pipeline to seed database.")
    st.stop()

# 2. Filtering & Export Controls
st.markdown('<div class="section-card">', unsafe_allow_html=True)
filter_col1, filter_col2, filter_col3 = st.columns([1.5, 1.5, 1])

with filter_col1:
    selected_segments = st.multiselect(
        "Filter by RFM Segment",
        options=list(df["segment"].unique()),
        default=list(df["segment"].unique()),
    )

with filter_col2:
    selected_clusters = st.multiselect(
        "Filter by Structural K-Means Cluster",
        options=list(df["kmeans_cluster"].unique()),
        default=list(df["kmeans_cluster"].unique()),
    )

with filter_col3:
    st.markdown("<div style='height: 1.65rem;'></div>", unsafe_allow_html=True)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Export Filtered Accounts (CSV)",
        data=csv_bytes,
        file_name="churnguard_customer_master.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown("</div>", unsafe_allow_html=True)

# Apply filters
filtered_df = df[
    df["segment"].isin(selected_segments) & df["kmeans_cluster"].isin(selected_clusters)
].copy()

if filtered_df.empty:
    st.warning("No customers match the active filters. Adjust your segment or cluster selections.")
    st.stop()

# 3. 2D & 3D Interactive Visualizations
col_vis1, col_vis2 = st.columns(2)

with col_vis1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-header-row">
            <span class="section-header">Recency vs. Monetary Distribution</span>
            <span class="badge-meta">Bubble Size = Frequency</span>
        </div>
        <div class="section-subtitle">Account spend elasticity plotted against days since last purchase</div>
        """,
        unsafe_allow_html=True,
    )

    # Sample if large for snappy rendering
    sample_size = min(1500, len(filtered_df))
    scatter_sample = filtered_df.sample(n=sample_size, random_state=42)

    fig_2d = px.scatter(
        scatter_sample,
        x="recency",
        y="monetary",
        color="segment",
        color_discrete_map=SEGMENT_COLORS,
        size="frequency",
        size_max=22,
        hover_name="CustomerID",
        hover_data={
            "recency": True,
            "monetary": ":$.2f",
            "frequency": True,
            "segment": True,
            "churn_probability": ":.2%",
        },
        labels={
            "recency": "Recency (Days Inactive)",
            "monetary": "Monetary Value (Spend $)",
            "segment": "RFM Tier",
        },
    )

    layout_2d = get_plotly_layout(height=380, show_legend=False)
    layout_2d["yaxis"]["type"] = "log"
    layout_2d["yaxis"]["title"] = "Monetary Value (Log Scale $)"
    layout_2d["xaxis"]["title"] = "Days Since Last Transaction"
    fig_2d.update_layout(layout_2d)
    st.plotly_chart(fig_2d, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_vis2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-header-row">
            <span class="section-header">Multidimensional RFM Space</span>
            <span class="badge-meta">Recency × Frequency × Spend</span>
        </div>
        <div class="section-subtitle">3D topological clustering showing spatial separation between cohorts</div>
        """,
        unsafe_allow_html=True,
    )

    sample_3d = filtered_df.sample(n=min(1000, len(filtered_df)), random_state=42)
    fig_3d = px.scatter_3d(
        sample_3d,
        x="recency",
        y="frequency",
        z=np.log1p(sample_3d["monetary"]),
        color="segment",
        color_discrete_map=SEGMENT_COLORS,
        hover_name="CustomerID",
        labels={
            "recency": "Recency (Days)",
            "frequency": "Frequency",
            "z": "Log Spend ($)",
        },
    )
    fig_3d.update_layout(
        height=380,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        paper_bgcolor="#FFFFFF",
        scene={
            "xaxis": {"backgroundcolor": "#F8FAFC", "gridcolor": "#E2E8F0"},
            "yaxis": {"backgroundcolor": "#F8FAFC", "gridcolor": "#E2E8F0"},
            "zaxis": {"backgroundcolor": "#F8FAFC", "gridcolor": "#E2E8F0"},
        },
        showlegend=False,
    )
    st.plotly_chart(fig_3d, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# 4. Treemap View
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="section-header-row">
        <span class="section-header">Hierarchical Customer Base Treemap</span>
        <span class="badge-meta">Segment Volume & Revenue Density</span>
    </div>
    <div class="section-subtitle">Cluster allocation and proportional Gross Merchandise Value breakdown</div>
    """,
    unsafe_allow_html=True,
)

treemap_df = (
    filtered_df.groupby(["kmeans_cluster", "segment"])
    .agg(
        account_count=("CustomerID", "count"),
        total_spend=("monetary", "sum"),
        mean_churn=("churn_probability", "mean"),
    )
    .reset_index()
)

fig_tree = px.treemap(
    treemap_df,
    path=["kmeans_cluster", "segment"],
    values="total_spend",
    color="mean_churn",
    color_continuous_scale="Blues",
    hover_data={"account_count": True, "total_spend": ":$,.0f", "mean_churn": ":.1%"},
    labels={"total_spend": "Total Spend", "mean_churn": "Avg Churn Prob"},
)
tree_layout = get_plotly_layout(height=360, show_legend=False)
fig_tree.update_layout(tree_layout)
st.plotly_chart(fig_tree, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# 5. Segment Strategy Playbooks
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="section-header-row">
        <span class="section-header">Segment Strategy Playbooks & Retention Protocols</span>
        <span class="badge-meta">Institutional Guidelines</span>
    </div>
    <div class="section-subtitle">Empirically backed prescriptive intervention playbooks for revenue retention teams</div>
    """,
    unsafe_allow_html=True,
)

SEGMENT_STRATEGIES = [
    (
        "Champions",
        "VIP White-Glove Treatment",
        "Reward loyalty, provide early beta access to new catalog lines, and recruit as brand advocates. Do not discount heavily.",
        "High Priority",
        "positive",
    ),
    (
        "Loyal Customers",
        "Cross-Sell & Automated Replenishment",
        "Offer relevant higher-value bundles and automated replenishment reminders based on their past purchase cadence.",
        "Expansion Candidate",
        "positive",
    ),
    (
        "Potential Loyalist",
        "Engagement & Loyalty Nurture",
        "Recommend top-rated items, invite to tier-1 rewards program, and offer a second-order complimentary perk.",
        "Growth Target",
        "neutral",
    ),
    (
        "New Customers",
        "Onboarding & Adoption Cadence",
        "Deliver proactive setup assistance, satisfaction check-ins, and a 14-day re-engagement offer to build purchasing habit.",
        "Onboarding",
        "neutral",
    ),
    (
        "Promising",
        "Incentivize Basket Frequency",
        "Provide limited-time free shipping thresholds and personalized category discovery digests.",
        "Conversion Target",
        "neutral",
    ),
    (
        "Need Attention",
        "Reactivation Re-engagement",
        "Deploy time-sensitive 15% incentive vouchers and highlight new arrivals in historical browsing categories.",
        "Monitoring",
        "negative",
    ),
    (
        "At Risk",
        "Multi-Channel Win-Back Campaign",
        "Execute 25% targeted win-back sequence across primary and secondary channels. Deploy customer feedback surveys.",
        "Immediate Intervention",
        "negative",
    ),
    (
        "Can't Lose Them",
        "Executive Concierge Outreach",
        "Direct outreach from Customer Relations Leadership + $100 service credit to protect mission-critical lifetime accounts.",
        "Critical Capital Risk",
        "negative",
    ),
    (
        "Hibernating",
        "Automated Catalog Clearance",
        "Re-activate with clearance discount events or seasonal blowout promotions. Transition to low-frequency email cadence.",
        "Low-Effort Outreach",
        "neutral",
    ),
    (
        "Lost",
        "Automated Exit Diagnostics",
        "Deploy single-question exit survey to identify structural service or product gaps. Avoid aggressive ad spend.",
        "De-prioritized",
        "neutral",
    ),
]

cols_per_row = 2
for i in range(0, len(SEGMENT_STRATEGIES), cols_per_row):
    r_cols = st.columns(cols_per_row)
    for j in range(cols_per_row):
        idx = i + j
        if idx < len(SEGMENT_STRATEGIES):
            name, play, desc, priority, pill_type = SEGMENT_STRATEGIES[idx]
            color = SEGMENT_COLORS.get(name, "#2563EB")

            # Segment metrics
            seg_data = df[df["segment"] == name]
            count = len(seg_data)
            avg_val = seg_data["monetary"].mean() if count > 0 else 0.0
            seg_churn = (seg_data["churned"].mean() * 100) if count > 0 else 0.0

            with r_cols[j]:
                st.markdown(
                    f"""
                    <div style="background:#FFFFFF; border: 1px solid #E2E8F0; border-top: 3px solid {color}; border-radius: 8px; padding: 1.1rem; margin-bottom: 1rem; box-shadow: 0 1px 2px rgba(0,0,0,0.03);">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 0.4rem;">
                            <span style="font-size: 1rem; font-weight: 800; color: #0F172A;">{name}</span>
                            <span class="metric-pill {pill_type}">{priority}</span>
                        </div>
                        <div style="display:flex; gap:14px; font-size:0.78rem; color:#64748B; margin-bottom:0.6rem;">
                            <span>Accounts: <b style="color:#0F172A;">{count:,}</b></span>
                            <span>Avg Spend: <b style="color:#0F172A;">${avg_val:,.2f}</b></span>
                            <span>Churn Rate: <b style="color:#0F172A;">{seg_churn:.1f}%</b></span>
                        </div>
                        <div style="font-size: 0.85rem; font-weight: 700; color: {color}; margin-bottom: 0.35rem;">{play}</div>
                        <div style="font-size: 0.82rem; color: #475569; line-height: 1.45;">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

st.markdown("</div>", unsafe_allow_html=True)
