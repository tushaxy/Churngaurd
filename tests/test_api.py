"""Integration and unit tests for FastAPI backend endpoints."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_check() -> None:
    """Verifies that the /health endpoint returns 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["database_connected"] is True
    assert "timestamp" in data


def test_get_customers_paginated() -> None:
    """Verifies that /customers returns paginated customer records."""
    response = client.get("/customers?page=1&page_size=10")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) == 10
    assert data["total"] > 0
    assert data["page"] == 1
    assert data["page_size"] == 10


def test_get_customers_filtered_by_segment() -> None:
    """Verifies segment filtering in /customers."""
    response = client.get("/customers?segment=Champions&page_size=5")
    assert response.status_code == 200
    data = response.json()
    for item in data["items"]:
        assert item["segment"] == "Champions"


def test_get_customer_by_id_success() -> None:
    """Verifies retrieving a specific customer by ID."""
    # First get an existing customer ID
    list_res = client.get("/customers?page_size=1")
    assert list_res.status_code == 200
    cust_id = list_res.json()["items"][0]["CustomerID"]

    response = client.get(f"/customers/{cust_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["CustomerID"] == cust_id
    assert "churn_probability" in data
    assert "segment" in data


def test_get_customer_by_id_not_found() -> None:
    """Verifies that a non-existent customer ID returns 404."""
    response = client.get("/customers/999999999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_predict_churn_success() -> None:
    """Verifies real-time churn prediction endpoint with valid payload."""
    payload = {
        "CustomerID": 12345,
        "recency": 25.0,
        "frequency": 8,
        "monetary": 580.0,
        "avg_order_value": 72.5,
        "tenure_days": 300,
        "orders_per_month": 0.81,
        "return_rate": 0.02,
        "rfm_score": 544,
        "segment": "Loyal Customers",
        "kmeans_cluster": "High-Value Loyalists",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_band"] in ["Low", "Medium", "High", "Critical"]
    assert "revenue_at_risk" in data
    assert "recommended_action" in data
    assert "top_drivers" in data
    assert len(data["top_drivers"]) > 0


def test_predict_churn_validation_error() -> None:
    """Verifies that invalid input payload triggers 422 Unprocessable Entity."""
    # Missing required field 'frequency'
    invalid_payload = {
        "CustomerID": 12345,
        "monetary": 580.0,
        "segment": "Loyal Customers",
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_batch_predict_churn() -> None:
    """Verifies batch scoring endpoint."""
    batch_payload = {
        "customers": [
            {
                "CustomerID": 101,
                "frequency": 10,
                "monetary": 1200.0,
                "avg_order_value": 120.0,
                "tenure_days": 400,
                "orders_per_month": 0.76,
                "return_rate": 0.01,
                "rfm_score": 555,
                "segment": "Champions",
                "kmeans_cluster": "High-Value Loyalists",
            },
            {
                "CustomerID": 102,
                "frequency": 1,
                "monetary": 30.0,
                "avg_order_value": 30.0,
                "tenure_days": 90,
                "orders_per_month": 0.33,
                "return_rate": 0.0,
                "rfm_score": 111,
                "segment": "Lost",
                "kmeans_cluster": "Dormant / High Churn Risk",
            },
        ]
    }
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 2
    assert "summary" in data
    assert data["summary"]["total_customers"] == 2
    assert "total_revenue_at_risk" in data["summary"]


def test_get_segments_summary() -> None:
    """Verifies that /segments/summary returns aggregated metrics across segments."""
    response = client.get("/segments/summary")
    assert response.status_code == 200
    data = response.json()
    assert "segments" in data
    assert len(data["segments"]) > 0
    assert data["total_customers"] > 0
    assert data["total_revenue"] > 0
    for seg in data["segments"]:
        assert "segment" in seg
        assert "customer_count" in seg
        assert "revenue_share_pct" in seg


def test_get_metrics() -> None:
    """Verifies that /metrics returns test metrics and optimal threshold."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "roc_auc" in data
    assert "pr_auc" in data
    assert "optimal_threshold" in data
    assert "confusion_matrix" in data
    assert data["roc_auc"] > 0.80
