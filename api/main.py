"""FastAPI application for ChurnGuard churn prediction and customer segmentation service."""

import sqlite3
from datetime import datetime

import shap
from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from sklearn.pipeline import Pipeline

from api.dependencies import get_db_connection, get_model_pipeline, get_shap_explainer
from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    BatchPredictSummary,
    CustomerFeatures,
    CustomerListResponse,
    CustomerRecord,
    HealthResponse,
    MetricsResponse,
    PredictResponse,
    SegmentSummaryItem,
    SegmentSummaryResponse,
)
from src.churn.config import settings
from src.churn.explain import explain_customer_prediction
from src.churn.predict import load_optimal_threshold
from src.churn.utils import get_logger, load_json

logger = get_logger(__name__)

app = FastAPI(
    title="ChurnGuard API",
    description="Production Machine Learning & Customer Segmentation Engine for Churn Prevention",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for Streamlit and external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="System Health & Diagnostic Check",
)
def health_check(
    db: sqlite3.Connection = Depends(get_db_connection),
    pipeline: Pipeline = Depends(get_model_pipeline),
) -> HealthResponse:
    """Verifies that the API, machine learning model, and SQLite database are operational."""
    db_ok = False
    try:
        cur = db.cursor()
        cur.execute("SELECT 1")
        db_ok = True
    except Exception as e:
        logger.error("Health check database error: %s", e)

    return HealthResponse(
        status="healthy" if (db_ok and pipeline is not None) else "degraded",
        model_loaded=pipeline is not None,
        database_connected=db_ok,
        timestamp=datetime.now().isoformat(),
    )


@app.get(
    "/customers",
    response_model=CustomerListResponse,
    tags=["Customers"],
    summary="List and Filter Customers with Pagination",
)
def get_customers(
    segment: str | None = Query(None, description="Filter by customer segment name"),
    risk_band: str | None = Query(
        None, description="Filter by risk band (Low, Medium, High, Critical)"
    ),
    search: str | None = Query(None, description="Search by CustomerID"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(25, ge=1, le=200, description="Items per page"),
    db: sqlite3.Connection = Depends(get_db_connection),
) -> CustomerListResponse:
    """Retrieves paginated customers joined with their latest churn predictions."""
    query = """
        SELECT
            c.*,
            p.churn_probability,
            p.churn_prediction,
            p.risk_band,
            p.revenue_at_risk
        FROM customers c
        LEFT JOIN predictions p ON c.CustomerID = p.CustomerID
        WHERE 1=1
    """
    params = []

    if segment:
        query += " AND c.segment = ?"
        params.append(segment)

    if risk_band:
        query += " AND p.risk_band = ?"
        params.append(risk_band)

    if search:
        query += " AND CAST(c.CustomerID AS TEXT) LIKE ?"
        params.append(f"%{search}%")

    count_query = f"SELECT COUNT(*) FROM ({query})"
    cur = db.cursor()
    total = cur.execute(count_query, params).fetchone()[0]

    offset = (page - 1) * page_size
    query += " ORDER BY c.monetary DESC LIMIT ? OFFSET ?"
    params.extend([page_size, offset])

    rows = cur.execute(query, params).fetchall()
    items = [CustomerRecord(**dict(r)) for r in rows]
    total_pages = (total + page_size - 1) // page_size if total > 0 else 1

    return CustomerListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@app.get(
    "/customers/{customer_id}",
    response_model=CustomerRecord,
    tags=["Customers"],
    summary="Retrieve Customer by ID",
)
def get_customer_by_id(
    customer_id: int,
    db: sqlite3.Connection = Depends(get_db_connection),
) -> CustomerRecord:
    """Fetches full customer attributes and predictions for a specific CustomerID."""
    query = """
        SELECT
            c.*,
            p.churn_probability,
            p.churn_prediction,
            p.risk_band,
            p.revenue_at_risk
        FROM customers c
        LEFT JOIN predictions p ON c.CustomerID = p.CustomerID
        WHERE c.CustomerID = ?
    """
    cur = db.cursor()
    row = cur.execute(query, (customer_id,)).fetchone()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer with ID {customer_id} not found",
        )

    return CustomerRecord(**dict(row))


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Prediction"],
    summary="Real-Time Churn Scoring & SHAP Drivers",
)
def predict_churn(
    customer: CustomerFeatures,
    pipeline: Pipeline = Depends(get_model_pipeline),
    explainer: shap.TreeExplainer | None = Depends(get_shap_explainer),
) -> PredictResponse:
    """Predicts churn probability, assigns risk band, calculates individual revenue at risk,

    and extracts top local SHAP drivers pushing toward churn vs retention.
    """
    cust_dict = customer.model_dump()
    result = explain_customer_prediction(cust_dict, pipeline=pipeline, explainer=explainer)

    threshold = load_optimal_threshold()
    churn_pred = int(result["churn_probability"] >= threshold)

    return PredictResponse(
        CustomerID=customer.CustomerID,
        churn_probability=result["churn_probability"],
        churn_percentage=result["churn_percentage"],
        churn_prediction=churn_pred,
        risk_band=result["risk_band"],
        revenue_at_risk=result["revenue_at_risk"],
        monetary=result["monetary"],
        segment=result["segment"],
        recommended_action=result["recommended_action"],
        top_drivers=result["top_drivers"],
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    tags=["Prediction"],
    summary="Batch Churn Scoring",
)
def predict_churn_batch(
    request: BatchPredictRequest,
    pipeline: Pipeline = Depends(get_model_pipeline),
    explainer: shap.TreeExplainer | None = Depends(get_shap_explainer),
) -> BatchPredictResponse:
    """Performs high-throughput batch scoring across multiple customer profiles."""
    if not request.customers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch request customer list cannot be empty",
        )

    predictions = []
    total_rar = 0.0
    high_count = 0
    critical_count = 0

    threshold = load_optimal_threshold()

    for cust in request.customers:
        cust_dict = cust.model_dump()
        res = explain_customer_prediction(cust_dict, pipeline=pipeline, explainer=explainer)
        churn_pred = int(res["churn_probability"] >= threshold)

        pred_item = PredictResponse(
            CustomerID=cust.CustomerID,
            churn_probability=res["churn_probability"],
            churn_percentage=res["churn_percentage"],
            churn_prediction=churn_pred,
            risk_band=res["risk_band"],
            revenue_at_risk=res["revenue_at_risk"],
            monetary=res["monetary"],
            segment=res["segment"],
            recommended_action=res["recommended_action"],
            top_drivers=res["top_drivers"][:3],  # return top 3 for batch efficiency
        )
        predictions.append(pred_item)
        total_rar += res["revenue_at_risk"]

        if res["risk_band"] == "High":
            high_count += 1
        elif res["risk_band"] == "Critical":
            critical_count += 1

    summary = BatchPredictSummary(
        total_customers=len(predictions),
        total_revenue_at_risk=round(total_rar, 2),
        high_risk_count=high_count,
        critical_risk_count=critical_count,
    )

    return BatchPredictResponse(predictions=predictions, summary=summary)


@app.get(
    "/segments/summary",
    response_model=SegmentSummaryResponse,
    tags=["Analytics"],
    summary="Aggregate Segment Performance & Revenue at Risk",
)
def get_segments_summary(
    db: sqlite3.Connection = Depends(get_db_connection),
) -> SegmentSummaryResponse:
    """Computes aggregated KPIs per customer segment."""
    query = """
        SELECT
            c.segment,
            COUNT(c.CustomerID) AS customer_count,
            ROUND(SUM(c.monetary), 2) AS total_revenue,
            ROUND(AVG(c.avg_order_value), 2) AS avg_order_value,
            ROUND(AVG(c.churned) * 100, 2) AS churn_rate_pct,
            ROUND(SUM(p.revenue_at_risk), 2) AS revenue_at_risk
        FROM customers c
        LEFT JOIN predictions p ON c.CustomerID = p.CustomerID
        GROUP BY c.segment
        ORDER BY total_revenue DESC
    """
    cur = db.cursor()
    rows = cur.execute(query).fetchall()

    total_revenue_all = sum(r["total_revenue"] for r in rows) if rows else 1.0
    total_cust_all = sum(r["customer_count"] for r in rows) if rows else 0

    # Strategic playbooks per segment
    segment_plays = {
        "Champions": "VIP Exclusive Loyalty Program & Product Co-Creation",
        "Loyal Customers": "Cross-Sell Bundles & Automated Replenishment",
        "Potential Loyalist": "Engagement Nurture & Second-Purchase Promo",
        "New Customers": "Onboarding Concierge & First 30-Day Check-in",
        "Promising": "Curated Recommendations & Free Shipping Thresholds",
        "Need Attention": "Limited-Time Category Discount Re-engagement",
        "At Risk": "High-Impact Multi-Channel Win-Back Sequence (25% off)",
        "Can't Lose Them": "Executive Concierge Phone Call & Account Review",
        "Hibernating": "Permission Marketing & Catalog Clearance Alert",
        "Lost": "Automated Exit Survey & Final Reactivation Code",
    }

    items = []
    for r in rows:
        seg = r["segment"]
        rev = float(r["total_revenue"] or 0.0)
        share = round((rev / total_revenue_all) * 100, 2)
        items.append(
            SegmentSummaryItem(
                segment=seg,
                customer_count=int(r["customer_count"]),
                total_revenue=rev,
                revenue_share_pct=share,
                avg_order_value=float(r["avg_order_value"] or 0.0),
                churn_rate_pct=float(r["churn_rate_pct"] or 0.0),
                revenue_at_risk=float(r["revenue_at_risk"] or 0.0),
                recommended_strategy=segment_plays.get(seg, "Deploy targeted offer"),
            )
        )

    return SegmentSummaryResponse(
        segments=items,
        total_customers=total_cust_all,
        total_revenue=round(total_revenue_all, 2),
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["Analytics"],
    summary="Model Validation & Performance Metrics",
)
def get_metrics() -> MetricsResponse:
    """Returns test set evaluation metrics, calibration, and optimal threshold parameters."""
    metrics_path = settings.paths.metrics_file
    try:
        data = load_json(metrics_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metrics not available: {e}",
        )

    return MetricsResponse(
        roc_auc=data["roc_auc"],
        pr_auc=data["pr_auc"],
        brier_score=data["brier_score"],
        optimal_threshold=data["optimal_threshold"],
        precision_at_optimal=data["precision_at_optimal"],
        recall_at_optimal=data["recall_at_optimal"],
        f1_at_optimal=data["f1_at_optimal"],
        revenue_at_risk_at_optimal=data["revenue_at_risk_at_optimal"],
        revenue_at_risk_pct_at_optimal=data["revenue_at_risk_pct_at_optimal"],
        target_revenue_at_risk_ratio=data["target_revenue_at_risk_ratio"],
        imbalance_strategy=data.get("imbalance_strategy", "smote"),
        confusion_matrix=data["confusion_matrix"],
        curve_points=data.get("curve_points", {}),
    )
