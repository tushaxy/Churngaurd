# ChurnGuard — Power BI DAX Measures Reference

This document contains copy-paste ready DAX formulas tailored for the **ChurnGuard Power BI Dashboard**. All measures reference tables from `powerbi/exports/` (`CustomerSummary`, `SegmentSummary`, `MonthlyRevenue`, `ModelMetrics`).

---

## 1. Core Financial & Volume KPIs

### Total Revenue
```dax
Total Revenue = 
SUM(CustomerSummary[monetary])
```
*Format: Currency ($#,##0)*

### Total Customers
```dax
Total Customers = 
DISTINCTCOUNT(CustomerSummary[CustomerID])
```
*Format: Whole Number (#,##0)*

### Average Order Value (AOV)
```dax
Average Order Value = 
DIVIDE([Total Revenue], SUM(CustomerSummary[frequency]), 0)
```
*Format: Currency ($#,##0.00)*

---

## 2. Churn & Risk Intelligence

### Churn Rate %
```dax
Churn Rate % = 
DIVIDE(
    CALCULATE(COUNTROWS(CustomerSummary), CustomerSummary[churn_prediction] = 1),
    COUNTROWS(CustomerSummary),
    0
)
```
*Format: Percentage (0.0%)*

### Actual Ground Truth Churn Rate %
```dax
Actual Churn Rate % = 
DIVIDE(
    CALCULATE(COUNTROWS(CustomerSummary), CustomerSummary[actual_churned] = 1),
    COUNTROWS(CustomerSummary),
    0
)
```
*Format: Percentage (0.0%)*

### Revenue at Risk ($)
```dax
Revenue at Risk = 
SUM(CustomerSummary[revenue_at_risk])
```
*Format: Currency ($#,##0)*

### % Revenue at Risk
```dax
% Revenue at Risk = 
DIVIDE([Revenue at Risk], [Total Revenue], 0)
```
*Format: Percentage (0.00%)*

---

## 3. Pareto & Concentration Measures

### Top 20% Customer Revenue
```dax
Top 20% Revenue = 
VAR TotalCust = COUNTROWS(CustomerSummary)
VAR Top20Count = ROUNDUP(TotalCust * 0.20, 0)
VAR Top20Table = 
    TOPN(
        Top20Count,
        CustomerSummary,
        CustomerSummary[monetary],
        DESC
    )
RETURN
    CALCULATE(
        SUM(CustomerSummary[monetary]),
        Top20Table
    )
```
*Format: Currency ($#,##0)*

### Top 20% Revenue Share %
```dax
Top 20% Revenue Share = 
DIVIDE([Top 20% Revenue], [Total Revenue], 0)
```
*Format: Percentage (0.0%)*

---

## 4. Risk Tier Cohort Counts

### Critical Risk Customers
```dax
Critical Risk Customers = 
CALCULATE(
    COUNTROWS(CustomerSummary),
    CustomerSummary[risk_band] = "Critical"
)
```
*Format: Whole Number (#,##0)*

### High Risk Customers
```dax
High Risk Customers = 
CALCULATE(
    COUNTROWS(CustomerSummary),
    CustomerSummary[risk_band] = "High"
)
```
*Format: Whole Number (#,##0)*

### Medium Risk Customers
```dax
Medium Risk Customers = 
CALCULATE(
    COUNTROWS(CustomerSummary),
    CustomerSummary[risk_band] = "Medium"
)
```
*Format: Whole Number (#,##0)*

### Low Risk Customers
```dax
Low Risk Customers = 
CALCULATE(
    COUNTROWS(CustomerSummary),
    CustomerSummary[risk_band] = "Low"
)
```
*Format: Whole Number (#,##0)*

---

## 5. Decision & ROI Simulation

### Projected Recovered Revenue
```dax
Projected Recovered Revenue = 
VAR ChurnReductionRate = 0.20 -- 20% target reduction
VAR TargetRiskRevenue = 
    CALCULATE(
        [Revenue at Risk],
        CustomerSummary[risk_band] IN {"High", "Critical"}
    )
RETURN
    TargetRiskRevenue * ChurnReductionRate
```
*Format: Currency ($#,##0)*
