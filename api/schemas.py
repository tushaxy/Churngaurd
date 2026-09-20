"""Pydantic v2 schemas for ChurnGuard API request and response models."""

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["healthy"])
    model_loaded: bool = Field(..., examples=[True])
    database_connected: bool = Field(..., examples=[True])
    timestamp: str = Field(...)


class CustomerFeatures(BaseModel):
    CustomerID: int | None = Field(None, examples=[10001])
    recency: float | None = Field(35.0, description="Recency in days", examples=[35.0])
    frequency: int = Field(..., ge=1, description="Total distinct orders", examples=[6])
    monetary: float = Field(..., ge=0.0, description="Total lifetime spend ($)", examples=[420.50])
    avg_order_value: float = Field(
        ..., ge=0.0, description="Average spend per order", examples=[70.08]
    )
    tenure_days: int = Field(..., ge=1, description="Days as a customer", examples=[365])
    orders_per_month: float = Field(..., ge=0.0, description="Purchase cadence", examples=[0.50])
    return_rate: float = Field(
        0.0, ge=0.0, le=1.0, description="Ratio of cancellations", examples=[0.03]
    )
    rfm_score: int = Field(
        ..., ge=111, le=555, description="Combined RFM quintile score", examples=[444]
    )
    segment: str = Field(..., description="Customer segment", examples=["Loyal Customers"])
    kmeans_cluster: str = Field(
        ..., description="K-Means cluster label", examples=["High-Value Loyalists"]
    )


class DriverDetail(BaseModel):
    feature: str = Field(..., examples=["orders_per_month"])
    display_name: str = Field(..., examples=["Monthly Order Velocity"])
    shap_value: float = Field(..., examples=[-0.045])
    direction: str = Field(..., examples=["promotes retention"])
    impact_type: str = Field(..., examples=["retention"])


class PredictResponse(BaseModel):
    CustomerID: int | None = Field(None, examples=[10001])
    churn_probability: float = Field(..., ge=0.0, le=1.0, examples=[0.1245])
    churn_percentage: float = Field(..., examples=[12.5])
    churn_prediction: int = Field(..., examples=[0])
    risk_band: str = Field(..., examples=["Low"])
    revenue_at_risk: float = Field(..., examples=[52.35])
    monetary: float = Field(..., examples=[420.50])
    segment: str = Field(..., examples=["Loyal Customers"])
    recommended_action: str = Field(...)
    top_drivers: list[DriverDetail] = Field(default_factory=list)


class BatchPredictRequest(BaseModel):
    customers: list[CustomerFeatures]


class BatchPredictSummary(BaseModel):
    total_customers: int
    total_revenue_at_risk: float
    high_risk_count: int
    critical_risk_count: int


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    summary: BatchPredictSummary


class CustomerRecord(BaseModel):
    CustomerID: int
    recency: int
    frequency: int
    monetary: float
    avg_order_value: float
    tenure_days: int
    orders_per_month: float
    return_rate: float
    r_score: int
    f_score: int
    m_score: int
    rfm_score: int
    rfm_code: str
    segment: str
    kmeans_cluster: str
    churn_probability: float | None = None
    churn_prediction: int | None = None
    risk_band: str | None = None
    revenue_at_risk: float | None = None


class CustomerListResponse(BaseModel):
    items: list[CustomerRecord]
    total: int
    page: int
    page_size: int
    total_pages: int


class SegmentSummaryItem(BaseModel):
    segment: str
    customer_count: int
    total_revenue: float
    revenue_share_pct: float
    avg_order_value: float
    churn_rate_pct: float
    revenue_at_risk: float
    recommended_strategy: str


class SegmentSummaryResponse(BaseModel):
    segments: list[SegmentSummaryItem]
    total_customers: int
    total_revenue: float


class MetricsResponse(BaseModel):
    roc_auc: float
    pr_auc: float
    brier_score: float
    optimal_threshold: float
    precision_at_optimal: float
    recall_at_optimal: float
    f1_at_optimal: float
    revenue_at_risk_at_optimal: float
    revenue_at_risk_pct_at_optimal: float
    target_revenue_at_risk_ratio: float
    imbalance_strategy: str
    confusion_matrix: dict[str, int]
    curve_points: dict[str, Any]
