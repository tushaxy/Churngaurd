# ChurnGuard: Model Card

## Model Overview
- **Model Name:** ChurnGuard Random Forest Classifier
- **Model Architecture:** Scikit-Learn `RandomForestClassifier` with probability calibration and decision-threshold optimization.
- **Model Version:** 1.0.0 (Production Release)
- **Date:** September 2026
- **Model Type:** Supervised Binary Classification with Post-Hoc Explainability (Tree SHAP).
- **Primary Artifacts:**
  - `models/model.pkl` (Trained Random Forest ensemble)
  - `models/shap_explainer.pkl` (Serialized SHAP TreeExplainer)
  - `models/threshold.json` (Full 100-step threshold-revenue tradeoff surface)
  - `models/metrics.json` (Out-of-sample performance evaluation)

---

## Intended Use
- **Primary Intended Use:** Identify retail and e-commerce customers at elevated risk of permanent defection (inactivity > 90 days), quantify associated revenue exposure, and trigger proactive retention plays.
- **Target Users:** Retention Marketers, Customer Success Managers, Head of Growth, and Chief Revenue Officers.
- **Out-of-Scope Uses:**
  - Credit scoring, loan underwriting, or financial solvency assessment.
  - Performance management or surveillance of sales personnel.
  - Short-lifecycle consumer apps (e.g. daily active mobile games) without recalibration of the 90-day inactivity window.

---

## Feature Architecture & Leakage Safeguards

### Engineered Features
1. `frequency` (int): Total count of fulfilled purchase orders over customer lifetime.
2. `monetary` (float): Lifetime Gross Merchandise Value (GMV) in USD.
3. `avg_order_value` (float): Average basket size (`monetary / frequency`).
4. `tenure_days` (int): Days between account creation (first recorded order) and snapshot reference date.
5. `orders_per_month` (float): Purchase velocity metric normalized across tenure.
6. `cancellations` (int): Total volume of cancelled transactions.
7. `return_rate` (float): Ratio of cancelled items to total items purchased.

### Explicit Target Leakage Elimination
> **Methodological Safeguard:**  
> The business ground truth for customer churn is defined as a period of inactivity exceeding 90 days (`recency_days > 90`). Directly feeding snapshot `recency` or `recency_days` into the feature matrix creates a 100% deterministic target leak ($P(\text{churn} \mid \text{recency} > 90) = 1.0$), reducing the model to a trivial `if/else` threshold rule incapable of predicting churn *before* it occurs.
> 
> ChurnGuard strictly isolates the classification feature space from current snapshot recency. The model learns exclusively from behavioral rhythms, purchasing velocity, order consistency, basket elasticity, and dispute rates.

---

## Training & Resampling Strategy

- **Dataset Size:** 7,941 unique customer profiles aggregated from 120,000+ transaction lines.
- **Imbalance Handling:** The natural churn rate is ~16.1% (Class 0: 6,664 accounts; Class 1: 1,277 accounts; ratio 5.22:1).
- **Resampling Protocol:** Evaluated both Synthetic Minority Over-sampling Technique (`SMOTE(k_neighbors=5, random_state=42)`) and Cost-Sensitive Learning (`class_weight='balanced'`) under 5-Fold Stratified Cross-Validation.
  - **Winning Strategy:** SMOTE achieved superior mean PR-AUC and Brier calibration score without distorting prediction probabilities.
- **Hyperparameter Tuning:** 40 iterations of `RandomizedSearchCV` across `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features`.

---

## Quantitative Evaluation (Out-of-Sample Test Set)

The evaluation holdout set consists of 1,589 previously unseen customer accounts (20% stratified split).

| Evaluation Metric | Measured Value | Benchmark / Target | Status |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | **0.9934** | $\ge 0.8500$ | Exceeded |
| **PR-AUC (Average Precision)** | **0.9612** | $\ge 0.8000$ | Exceeded |
| **Brier Calibration Score** | **0.0229** | $\le 0.1000$ | Well-Calibrated |
| **Decision Threshold ($\tau^*$)** | **0.900** | Revenue-Optimized | Selected |
| **Precision @ $\tau^*$** | **91.30%** | $\ge 80.0\%$ | High Precision |
| **Recall @ $\tau^*$** | **81.71%** | $\ge 75.0\%$ | High Coverage |
| **F1-Score @ $\tau^*$** | **0.8624** | $\ge 0.8000$ | Robust Balance |
| **Revenue at Risk Ratio** | **6.94%** | **~7.00% Target** | **Within 0.06%** |
| **Total Test Revenue at Risk** | **$50,996.37** | Holdout Total: $734,815 | Quantified |

### Confusion Matrix ($\tau = 0.90$)
```
                   Predicted Retained (0)    Predicted Churn (1)
Actual Retained (0):        1,312                    20   (FP)
Actual Churn (1):              47 (FN)              210   (TP)
```
- **False Positive Rate:** 1.50% (avoids wasting retention budgets on loyal customers).
- **False Negative Rate:** 18.29% (acceptable tradeoff to isolate the highest-confidence ~7% revenue exposure).

---

## Explainability (SHAP TreeExplainer)

Every prediction is paired with exact local feature attributions via Tree SHAP:
1. **Primary Global Driver:** `orders_per_month` (Purchase velocity decay is the single strongest indicator of impending churn).
2. **Secondary Global Driver:** `frequency` (Lower lifetime order count increases churn susceptibility).
3. **Tertiary Global Driver:** `avg_order_value` (Customers with smaller average baskets churn at higher rates).
4. **Prescriptive Action Mapping:** Attributions are ingested by the automated retention engine to assign context-specific interventions:
   - *Velocity Drop:* Personalized re-engagement sequence with urgency triggers.
   - *AOV Decay:* Bundled incentive discount (e.g. $20 off $100 order).
   - *Return Discontent:* Concierge support outreach with complimentary returns.

---

## Ethical Considerations & Fairness
- **Demographic Neutrality:** Features are derived exclusively from transactional purchase behaviors and time dynamics. Geographic metadata (`country`) is used solely for regional aggregation and is excluded from high-weight modeling splits.
- **Intervention Disparity:** To prevent discrimination against low-spend customers, the platform provides segmented playbooks so that even lower-tier at-risk cohorts receive automated reactivation offers rather than being abandoned.

---

## Operational Constraints & Retraining Triggers
1. **Inference Latency:** Target $< 15$ ms per single-customer payload on CPU. Production benchmark: ~8 ms.
2. **Batch Throughput:** Evaluates 10,000 customer records in $< 1.5$ seconds.
3. **Model Drift Monitoring:**
   - Trigger retraining if Population Stability Index (PSI) on `orders_per_month` exceeds $0.25$.
   - Recalibrate threshold $\tau^*$ if quarterly macro churn rate shifts by $> 3.5$ percentage points.
