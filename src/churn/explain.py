"""SHAP explainability module and prescriptive retention playbook engine."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.churn.config import settings
from src.churn.utils import get_logger

logger = get_logger(__name__)

# Feature name mapping for human-readable presentation
FEATURE_DISPLAY_NAMES = {
    "frequency": "Purchase Frequency",
    "monetary": "Total Monetary Spend",
    "avg_order_value": "Average Order Value",
    "tenure_days": "Customer Tenure (Days)",
    "orders_per_month": "Monthly Order Velocity",
    "return_rate": "Cancellation / Return Rate",
    "rfm_score": "RFM Combined Score",
}

RETENTION_PLAYBOOK = {
    "Champions": {
        "Critical": "Urgent VIP Intervention: Personal outreach from VP of Customer Success + bespoke loyalty tier upgrade + exclusive preview of new catalog.",
        "High": "VIP Concierge Retention: Dedicated account review + 20% annual anniversary gift credit + invite to VIP advisory board.",
        "Medium": "Loyalty Celebration: Personalized appreciation note + early-access to upcoming seasonal collections.",
        "Low": "Brand Advocate Nurture: Automatic entry into VIP perks program + zero-friction returns guarantee.",
    },
    "Loyal Customers": {
        "Critical": "Win-Back Sequence: Immediate high-touch email offering double loyalty points + 15% discount on past favorite categories.",
        "High": "Proactive Value Check: Survey asking for feedback on recent orders + custom bundled re-order discount.",
        "Medium": "Engagement Booster: Tailored replenishment reminders based on past order cycle + free shipping perk.",
        "Low": "Steady Engagement: Bi-weekly newsletter highlighting curated recommendations matching purchase history.",
    },
    "At Risk": {
        "Critical": "Emergency Win-Back Campaign: Aggressive 25% 'We Miss You' voucher valid for 14 days + direct feedback inquiry.",
        "High": "Targeted Re-activation: Multi-channel SMS/Email sequence featuring best-sellers in previously ordered categories.",
        "Medium": "Value Re-engagement: Highlight new arrivals and price-drop alerts on previously viewed items.",
        "Low": "Standard Retargeting: Automated reminder of unused loyalty points or perks.",
    },
    "Can't Lose Them": {
        "Critical": "Executive Phone Outreach: Direct call from Customer Relations executive + $100 store credit + VIP satisfaction guarantee.",
        "High": "Priority Concierge Offer: Exclusive 30% renewal credit + dedicated customer service liaison.",
        "Medium": "Special Come-Back Privilege: Free premium gift with next purchase + free priority delivery for 6 months.",
        "Low": "VIP Check-In: Quarterly VIP status update and concierge contact.",
    },
    "Potential Loyalist": {
        "Critical": "Loyalty Incentive: Welcome-back voucher + free gift on second purchase to rekindle relationship.",
        "High": "Category Expansion: Cross-sell recommendation sequence with a limited-time 15% incentive.",
        "Medium": "Product Education: Showcase high-satisfaction products and customer reviews.",
        "Low": "Next-Step Nurture: Loyalty program onboarding with bonus points on next order.",
    },
    "New Customers": {
        "Critical": "Post-Onboarding Recovery: Apology check-in to confirm satisfaction with first order + 20% second-order promo.",
        "High": "Second-Purchase Incentive: Limited-time 15% discount code expiring in 10 days.",
        "Medium": "Welcome Series: Step-by-step product guide + customer support contact.",
        "Low": "Standard Onboarding: Welcome sequence with curated starter products.",
    },
    "Need Attention": {
        "Critical": "Flash Revival Deal: 20% off entire order for 72 hours only.",
        "High": "Re-engagement Quiz: 'Find your next favorite item' quiz with instant discount reward.",
        "Medium": "Category Discount: 10% discount on customer's most purchased category.",
        "Low": "Seasonal Announcement: Notification of seasonal catalog launch.",
    },
    "Promising": {
        "Critical": "Re-activation Offer: 15% discount voucher + free shipping.",
        "High": "Product Discovery: Personalized email featuring top items in customer's price range.",
        "Medium": "Social Proof: Testimonials and top-rated items showcase.",
        "Low": "Browse Abandonment: Light touch reminder of popular items.",
    },
    "Hibernating": {
        "Critical": "Deep Discount Re-engagement: 25% off blowout coupon to test remaining customer interest.",
        "High": "Re-activation Promo: Flat $15 credit on any order over $50.",
        "Medium": "Clean-Up Ping: 'Are you still interested?' permission email with 10% incentive.",
        "Low": "Low-Frequency Digest: Monthly highlights newsletter.",
    },
    "Lost": {
        "Critical": "Exit Survey & Win-Back: Short 1-question feedback survey + 30% final re-activation discount.",
        "High": "Re-engagement Campaign: High-discount catalog clearance alert.",
        "Medium": "Semi-Annual Ping: Major annual sale announcement.",
        "Low": "Dormant Archive: Move to low-cadence quarterly brand awareness list.",
    },
}


def fit_and_save_shap_explainer(pipeline: Pipeline, X_train: pd.DataFrame) -> shap.TreeExplainer:
    """Extracts the underlying tree model and preprocessed feature matrix to fit TreeExplainer.

    Args:
        pipeline: Trained scikit-learn Pipeline (ColumnTransformer + Classifier).
        X_train: Raw training DataFrame.

    Returns:
        Fitted shap.TreeExplainer instance.
    """
    logger.info("Fitting SHAP TreeExplainer on trained model...")

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Extract base tree model if wrapped in CalibratedClassifierCV
    if hasattr(classifier, "calibrated_classifiers_"):
        tree_model = classifier.calibrated_classifiers_[0].estimator
    else:
        tree_model = classifier

    # Transform a representative background sample (150 rows) for TreeExplainer
    X_train_proc = preprocessor.transform(X_train)
    sample_background = X_train_proc[:150]

    explainer = shap.TreeExplainer(tree_model, data=sample_background)

    explainer_path = Path(settings.paths.explainer_file)
    explainer_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(explainer, explainer_path)
    logger.info("Serialized SHAP explainer to %s", explainer_path)

    return explainer


def load_shap_explainer() -> shap.TreeExplainer | None:
    """Loads fitted SHAP TreeExplainer from disk.

    Returns:
        Fitted explainer or None if not yet created.
    """
    explainer_path = Path(settings.paths.explainer_file)
    if not explainer_path.exists():
        alt_path = Path(__file__).resolve().parent.parent.parent / settings.paths.explainer_file
        if alt_path.exists():
            explainer_path = alt_path
        else:
            return None

    return joblib.load(explainer_path)


def get_risk_band(churn_probability: float) -> str:
    """Categorizes churn probability into business operational risk tiers.

    Args:
        churn_probability: Model predicted churn probability in [0, 1].

    Returns:
        One of 'Low', 'Medium', 'High', 'Critical'.
    """
    if churn_probability < 0.30:
        return "Low"
    elif churn_probability < 0.60:
        return "Medium"
    elif churn_probability < 0.85:
        return "High"
    else:
        return "Critical"


def explain_customer_prediction(
    customer_dict: dict[str, Any],
    pipeline: Pipeline | None = None,
    explainer: shap.TreeExplainer | None = None,
) -> dict[str, Any]:
    """Generates local SHAP explanation, risk band, revenue at risk, and prescriptive action.

    Args:
        customer_dict: Dictionary of customer attributes.
        pipeline: Optional pre-loaded pipeline model.
        explainer: Optional pre-loaded SHAP TreeExplainer.

    Returns:
        Comprehensive explanation dictionary.
    """
    # 1. Load pipeline if not supplied
    if pipeline is None:
        model_path = Path(settings.paths.model_file)
        if not model_path.exists():
            alt_path = Path(__file__).resolve().parent.parent.parent / settings.paths.model_file
            model_path = alt_path
        pipeline = joblib.load(model_path)

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Convert customer dict to DataFrame row matching expected features
    from src.churn.train import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

    expected_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    row_data = {col: [customer_dict.get(col, 0)] for col in expected_cols}
    input_df = pd.DataFrame(row_data)

    # 2. Predict churn probability
    test_proc = preprocessor.transform(input_df)
    churn_prob = float(classifier.predict_proba(test_proc)[0, 1])

    risk_band = get_risk_band(churn_prob)
    monetary = float(customer_dict.get("monetary", 0.0))
    revenue_at_risk = round(monetary * churn_prob, 2)

    # 3. Retrieve feature names after one-hot encoding
    try:
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    except Exception:
        cat_feature_names = []

    all_transformed_feature_names = NUMERICAL_FEATURES + cat_feature_names

    # 4. SHAP local explanation
    if explainer is None:
        explainer = load_shap_explainer()

    drivers = []
    feature_shap_dict = {}

    if explainer is not None:
        try:
            shap_values = explainer.shap_values(test_proc)
            # Handle binary classification shape: (samples, features, 2) or (samples, features) or list
            if isinstance(shap_values, list):
                # class 1 (churn) SHAP values
                sv = shap_values[1][0]
            elif shap_values.ndim == 3:
                sv = shap_values[0, :, 1]
            else:
                sv = shap_values[0]

            for idx, feat_name in enumerate(all_transformed_feature_names):
                if idx < len(sv):
                    val = float(sv[idx])
                    feature_shap_dict[feat_name] = round(val, 4)

            # Sort features by absolute SHAP impact
            sorted_shap = sorted(feature_shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)

            for feat, impact in sorted_shap[:6]:
                # Format friendly display name
                display_name = FEATURE_DISPLAY_NAMES.get(feat, feat.replace("_", " ").title())
                direction = "increases churn risk" if impact > 0 else "promotes retention"
                drivers.append(
                    {
                        "feature": feat,
                        "display_name": display_name,
                        "shap_value": impact,
                        "direction": direction,
                        "impact_type": "risk" if impact > 0 else "retention",
                    }
                )
        except Exception as e:
            logger.warning("Error calculating individual SHAP values: %s", e)

    # 5. Determine prescriptive recommended retention action
    segment = customer_dict.get("segment", "Need Attention")
    segment_playbook = RETENTION_PLAYBOOK.get(segment, RETENTION_PLAYBOOK["Need Attention"])
    recommended_action = segment_playbook.get(risk_band, "Deploy personalized retention offer.")

    return {
        "churn_probability": round(churn_prob, 4),
        "churn_percentage": round(churn_prob * 100, 1),
        "risk_band": risk_band,
        "revenue_at_risk": revenue_at_risk,
        "monetary": monetary,
        "segment": segment,
        "top_drivers": drivers,
        "shap_values": feature_shap_dict,
        "recommended_action": recommended_action,
    }
