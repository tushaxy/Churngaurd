# ChurnGuard — Power BI Dashboard Rebuild Guide

A comprehensive, step-by-step guide to rebuilding the complete **ChurnGuard Customer Segmentation & Churn Prevention Dashboard** in Microsoft Power BI in ~30 minutes.

---

## Prerequisites
1. **Power BI Desktop** (free download from Microsoft).
2. The 4 export files in `powerbi/exports/`:
   - `CustomerSummary.csv`
   - `SegmentSummary.csv`
   - `MonthlyRevenue.csv`
   - `ModelMetrics.csv`
3. The custom theme file: `powerbi/theme.json`.
4. DAX formulas reference: `powerbi/DAX_Measures.md`.

---

## Phase 1: Data Ingestion & Transformation (5 Minutes)

1. **Launch Power BI Desktop** and choose **Get data** > **Text/CSV**.
2. Navigate to `powerbi/exports/` and import all 4 CSV files:
   - Select `CustomerSummary.csv` > **Transform Data** (opens Power Query).
   - In Power Query, select **New Source** > **CSV** and load `SegmentSummary.csv`, `MonthlyRevenue.csv`, and `ModelMetrics.csv`.
3. **Verify Data Types**:
   - `CustomerSummary`:
     - `CustomerID`: Integer (Text or Whole Number)
     - `monetary`, `avg_order_value`, `revenue_at_risk`, `churn_probability`: Decimal Number
     - `frequency`, `recency`, `tenure_days`, `r_score`, `f_score`, `m_score`, `rfm_score`: Whole Number
     - `segment`, `kmeans_cluster`, `risk_band`: Text
   - `MonthlyRevenue`:
     - `Date`: Date
     - `Revenue`, `RevenueAtRisk`: Decimal Number
4. Click **Close & Apply**.

---

## Phase 2: Theme & Relationships (3 Minutes)

1. **Apply Brand Theme**:
   - Navigate to the **View** ribbon tab > **Themes** dropdown > **Browse for themes...**.
   - Select `powerbi/theme.json`. This automatically synchronizes visual styling, borders, and color hexes (`#4F46E5`, `#14B8A6`, `#10B981`, `#F59E0B`, `#EF4444`).
2. **Configure Data Model Relationships** (Model View):
   - Power BI will automatically connect `CustomerSummary[segment]` to `SegmentSummary[Segment]` (1-to-many relationship).
   - Ensure the relationship cross-filter direction is set to **Single** or **Both**.

---

## Phase 3: Create Key DAX Measures (7 Minutes)

1. In the **Home** ribbon, select **Enter Data** and create a blank table named `_Measures`.
2. Right-click `_Measures` and select **New measure** for each measure defined in `DAX_Measures.md`:
   - `Total Revenue = SUM(CustomerSummary[monetary])`
   - `Total Customers = DISTINCTCOUNT(CustomerSummary[CustomerID])`
   - `Average Order Value = DIVIDE([Total Revenue], SUM(CustomerSummary[frequency]), 0)`
   - `Revenue at Risk = SUM(CustomerSummary[revenue_at_risk])`
   - `% Revenue at Risk = DIVIDE([Revenue at Risk], [Total Revenue], 0)`
   - `Churn Rate % = DIVIDE(CALCULATE(COUNTROWS(CustomerSummary), CustomerSummary[churn_prediction] = 1), COUNTROWS(CustomerSummary), 0)`
   - `Top 20% Revenue Share = ...` (see DAX_Measures.md)
   - `Projected Recovered Revenue = ...` (see DAX_Measures.md)

---

## Phase 4: Build Page 1 — Executive Overview (8 Minutes)

1. **Rename Page 1** to `Executive Overview`.
2. **Header Title**:
   - Insert a Text Box: `ChurnGuard | Executive Retention Intelligence` (Bold, 20pt, Color `#0F172A`).
3. **KPI Card Row** (Top Banner):
   - Insert 5 **New Card** visuals:
     - Card 1: `[Total Customers]`
     - Card 2: `[Total Revenue]` ($)
     - Card 3: `[Churn Rate %]` (%)
     - Card 4: `[% Revenue at Risk]` (%)
     - Card 5: `[Average Order Value]` ($)
4. **Pareto Combo Chart** (Bottom Left):
   - Visual: **Line and Clustered Column Chart**.
   - X-Axis: `CustomerSummary[CustomerID]` (sorted descending by `monetary`).
   - Column Y-Axis: `CustomerSummary[monetary]`.
   - Line Y-Axis: Cumulative revenue share percentage.
   - Title: `Pareto Revenue Concentration (Top 20% Accounts Command 62.5% of Revenue)`.
5. **Monthly Trajectory Line Chart** (Bottom Right):
   - Visual: **Line Chart**.
   - X-Axis: `MonthlyRevenue[Date]` (Month/Year).
   - Y-Axis: `SUM(MonthlyRevenue[Revenue])` and `SUM(MonthlyRevenue[RevenueAtRisk])`.
   - Line Colors: Indigo `#4F46E5` for Revenue, Red `#EF4444` for Revenue at Risk.

---

## Phase 5: Build Page 2 — Customer Segmentation (5 Minutes)

1. **Add Page 2** and rename to `Customer Segmentation`.
2. **Slicers Panel** (Left Sidebar):
   - Slicer 1: `CustomerSummary[segment]` (Tile or Dropdown).
   - Slicer 2: `CustomerSummary[kmeans_cluster]` (Vertical List).
   - Slicer 3: `CustomerSummary[risk_band]` (Tile).
3. **Revenue Treemap** (Top Center):
   - Visual: **Treemap**.
   - Category: `CustomerSummary[segment]`.
   - Values: `[Total Revenue]`.
   - Tooltip: `[Total Customers]`, `[Revenue at Risk]`.
4. **Recency vs. Frequency Bubble Scatter** (Bottom):
   - Visual: **Scatter Chart**.
   - X-Axis: `CustomerSummary[recency]` (Average).
   - Y-Axis: `CustomerSummary[frequency]` (Average).
   - Size: `CustomerSummary[monetary]`.
   - Legend: `CustomerSummary[segment]`.

---

## Phase 6: Build Page 3 — Risk Register & Root Cause Tree (5 Minutes)

1. **Add Page 3** and rename to `Risk Register & Root Cause`.
2. **High-Risk Customer Table Visual** (Left):
   - Visual: **Table**.
   - Columns: `CustomerID`, `segment`, `risk_band`, `churn_probability`, `monetary`, `revenue_at_risk`, `return_rate`.
   - Filter on Visual: `risk_band` IN `("High", "Critical")`.
   - Sort By: `revenue_at_risk` Descending.
   - **Conditional Formatting**:
     - Right-click `churn_probability` > **Conditional formatting** > **Background color** > Gradient from Light Yellow `#FEF3C7` to Dark Red `#EF4444`.
3. **Decomposition Tree / Key Influencers** (Right):
   - Visual: **Decomposition Tree**.
   - Analyze: `[Revenue at Risk]`.
   - Explain By: `CustomerSummary[segment]`, `CustomerSummary[risk_band]`, `CustomerSummary[kmeans_cluster]`.
   - This allows stakeholders to drill down from total company risk directly into specific segments and risk tiers.

---

## Phase 7: Verification & Publishing

1. Select **File** > **Save As** > `ChurnGuard_Dashboard.pbix`.
2. Test Slicers:
   - Click "At Risk" in the segment slicer: verify that all KPI cards and customer risk lists filter dynamically.
   - Click "Critical" in risk band: verify that individual high-risk accounts and their dollar exposures highlight instantly.
3. Publish to Power BI Service (optional) for sharing with stakeholders.
