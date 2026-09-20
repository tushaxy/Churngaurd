"""Customer Segmentation Explorer & RFM Strategy Playbook."""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.utils import (
    SEGMENT_COLORS,
    get_plotly_layout,
    inject_custom_css,
    load_customer_master_data,
    render_sidebar_status,
)

inject_custom_css()
render_sidebar_status()

st.markdown(
    """
    <div style="margin-bottom: 1.5rem;">
        <h1 style="font-size: 2.2rem; font-weight: 800; color: #0F172A; margin-bottom: 0.25rem;">
            Customer Segmentation & Behavioral Matrix
        </h1>
        <p style="color: #64748B; font-size: 1.05rem;">
            Explore behavioral clusters, analyze multidimensional RFM distributions, and access prescriptive segment playbooks.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

df = load_customer_master_data()

if df.empty:
    st.error("No customer records found. Please run `make pipeline` first.")
    st.stop()

# 1. Filters Row
filter_col1, filter_col2, filter_col3 = st.columns([1.5, 1.5, 1])
with filter_col1:
    selected_segments = st.multiselect(
        "Filter by Segment",
        options=list(df["segment"].unique()),
        default=list(df["segment"].unique()),
    )

with filter_col2:
    selected_clusters = st.multiselect(
        "Filter by K-Means Cluster",
        options=list(df["kmeans_cluster"].unique()),
        default=list(df["kmeans_cluster"].unique()),
    )

with filter_col3:
    st.markdown("<div style='height: 1.8rem;'></div>", unsafe_allow_html=True)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Master CSV",
        data=csv_bytes,
        file_name="customer_master.csv",
        mime="text/csv",
        use_container_width=True,
    )

# Filter dataframe
filtered_df = df[
    (df["segment"].isin(selected_segments)) & (df["kmeans_cluster"].isin(selected_clusters))
].copy()

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

# 2. Charts: 2D RxF Scatter & Treemap
col_scat, col_tree = st.columns([1.3, 1])

with col_scat:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-header">Recency vs. Frequency (Bubble Size = Spend)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">Visualizing customer lifetime value and activity decay</div>',
        unsafe_allow_html=True,
    )

    # Subsample 1500 points for snappy rendering in browser
    plot_sample = filtered_df.sample(min(1500, len(filtered_df)), random_state=42)

    fig_2d = px.scatter(
        plot_sample,
        x="recency",
        y="frequency",
        size="monetary",
        color="segment",
        color_discrete_map=SEGMENT_COLORS,
        hover_data=["CustomerID", "avg_order_value", "tenure_days"],
        size_max=35,
        opacity=0.75,
    )
    layout_2d = get_plotly_layout(height=420, show_legend=False)
    layout_2d["xaxis"]["title"] = "Recency (Days Since Last Order)"
    layout_2d["yaxis"]["title"] = "Frequency (Total Distinct Orders)"
    fig_2d.update_layout(layout_2d)
    st.plotly_chart(fig_2d, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_tree:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-header">Revenue Concentration Treemap</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-subtitle">Share of monetary volume by segment & cluster</div>',
        unsafe_allow_html=True,
    )

    tree_agg = (
        filtered_df.groupby(["segment", "kmeans_cluster"])
        .agg(monetary=("monetary", "sum"), count=("CustomerID", "count"))
        .reset_index()
    )

    fig_tree = px.treemap(
        tree_agg,
        path=[px.Constant("All Revenue"), "segment", "kmeans_cluster"],
        values="monetary",
        color="segment",
        color_discrete_map=SEGMENT_COLORS,
    )
    tree_layout = get_plotly_layout(height=420, show_legend=False)
    tree_layout["margin"] = {"l": 10, "r": 10, "t": 10, "b": 10}
    fig_tree.update_layout(tree_layout)
    st.plotly_chart(fig_tree, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# 3. 3D RFM Interactive Scatter Plot
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown(
    '<div class="section-header">3D RFM Manifold Representation</div>', unsafe_allow_html=True
)
st.markdown(
    '<div class="section-subtitle">Rotate and zoom through normalized Log(R), Log(F), and Log(M) space</div>',
    unsafe_allow_html=True,
)

sample_3d = filtered_df.sample(min(1200, len(filtered_df)), random_state=42)

fig_3d = go.Figure(
    data=[
        go.Scatter3d(
            x=np.log1p(sample_3d["recency"]),
            y=np.log1p(sample_3d["frequency"]),
            z=np.log1p(sample_3d["monetary"]),
            mode="markers",
            marker={
                "size": 4,
                "color": [SEGMENT_COLORS.get(s, "#64748B") for s in sample_3d["segment"]],
                "opacity": 0.8,
            },
            text=[
                f"Customer {cid}<br>Segment: {seg}<br>Recency: {r}d<br>Orders: {f}<br>Spend: ${m:,.2f}"
                for cid, seg, r, f, m in zip(
                    sample_3d["CustomerID"],
                    sample_3d["segment"],
                    sample_3d["recency"],
                    sample_3d["frequency"],
                    sample_3d["monetary"],
                    strict=False,
                )
            ],
            hoverinfo="text",
        )
    ]
)

fig_3d.update_layout(
    height=480,
    margin={"l": 0, "r": 0, "t": 0, "b": 0},
    paper_bgcolor="#FFFFFF",
    scene={
        "xaxis": {"title": "Log(Recency)", "backgroundcolor": "#FFFFFF", "gridcolor": "#F1F5F9"},
        "yaxis": {"title": "Log(Frequency)", "backgroundcolor": "#FFFFFF", "gridcolor": "#F1F5F9"},
        "zaxis": {"title": "Log(Monetary)", "backgroundcolor": "#FFFFFF", "gridcolor": "#F1F5F9"},
        "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.2}},
    },
)
st.plotly_chart(fig_3d, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# 4. Segment Strategy Playbook Cards
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown(
    '<div class="section-header">Segment Playbooks & Operational Strategies</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-subtitle">Recommended play and resource allocation for each customer tier</div>',
    unsafe_allow_html=True,
)

SEGMENT_STRATEGIES = [
    (
        "Champions",
        "VIP White-Glove Treatment",
        "Reward loyalty, provide early beta access to new catalog lines, and recruit as brand advocates. Do not discount heavily.",
        "⭐ High Priority",
    ),
    (
        "Loyal Customers",
        "Cross-Sell & Upsell",
        "Offer relevant higher-value bundles and automated replenishment reminders based on their past purchase cadence.",
        "⚡ High Impact",
    ),
    (
        "Potential Loyalist",
        "Engagement Nurture",
        "Recommend popular items, invite to community rewards program, and offer a second-order perk code.",
        "🌱 Growth Opportunity",
    ),
    (
        "New Customers",
        "Onboarding Excellence",
        "Send proactive setup support, satisfaction check-ins, and a 14-day re-engagement offer to build habit.",
        "🎯 Retention Driver",
    ),
    (
        "Promising",
        "Incentivize Frequency",
        "Provide limited-time free shipping thresholds and personalized category discovery digests.",
        "💡 Conversion Target",
    ),
    (
        "Need Attention",
        "Reactivation Discounts",
        "Deploy time-sensitive 15% discount vouchers and highlight new arrivals in categories they browsed.",
        "⚠️ Watch List",
    ),
    (
        "At Risk",
        "Multi-Channel Win-Back",
        "Aggressive 25% 'We Miss You' campaign via email + SMS. Offer direct feedback surveys to resolve grievances.",
        "🚨 Immediate Intervention",
    ),
    (
        "Can't Lose Them",
        "Executive Intervention",
        "Direct outreach from Customer Relations Director + $100 apology/retention credit to save high lifetime value accounts.",
        "🔥 Highest Financial Risk",
    ),
    (
        "Hibernating",
        "Catalog Clearance Deals",
        "Re-activate with steep clearance discounts or seasonal blowout promotions. Transition to low-frequency email cadence.",
        "💤 Low Effort Reactivation",
    ),
    (
        "Lost",
        "Automated Exit Survey",
        "Send a 1-question exit survey to identify product or service gaps. Avoid spending high customer acquisition cost.",
        "📉 De-prioritize",
    ),
]

cols_per_row = 2
for i in range(0, len(SEGMENT_STRATEGIES), cols_per_row):
    r_cols = st.columns(cols_per_row)
    for j in range(cols_per_row):
        idx = i + j
        if idx < len(SEGMENT_STRATEGIES):
            name, play, desc, priority = SEGMENT_STRATEGIES[idx]
            color = SEGMENT_COLORS.get(name, "#64748B")
            with r_cols[j]:
                st.markdown(
                    f"""
                    <div style="background:#F8FAFC; border-left: 5px solid {color}; border-radius: 10px; padding: 1.1rem; margin-bottom: 1rem;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 0.35rem;">
                            <span style="font-size: 1.1rem; font-weight: 700; color: #0F172A;">{name}</span>
                            <span style="font-size: 0.75rem; font-weight: 700; background: #E2E8F0; padding: 2px 8px; border-radius: 4px; color: #475569;">{priority}</span>
                        </div>
                        <div style="font-size: 0.9rem; font-weight: 600; color: {color}; margin-bottom: 0.35rem;">{play}</div>
                        <div style="font-size: 0.85rem; color: #64748B; line-height: 1.4;">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

st.markdown("</div>", unsafe_allow_html=True)
