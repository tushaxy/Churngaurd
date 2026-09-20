"""Unit tests for the data preprocessing and cleaning pipeline."""

import pandas as pd
import pytest

from src.churn.preprocessing import clean_transactions


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Fixture providing a controlled raw dataset with edge cases."""
    records = [
        # Normal purchases (customer 10001)
        {
            "Invoice": "536365",
            "StockCode": "85123A",
            "Description": "WHITE HEART HOLDER",
            "Quantity": 6,
            "InvoiceDate": "2024-01-10 10:00:00",
            "Price": 2.55,
            "CustomerID": 10001,
            "Country": "United Kingdom",
        },
        # Missing CustomerID -> must be dropped
        {
            "Invoice": "536366",
            "StockCode": "71053",
            "Description": "LANTERN",
            "Quantity": 2,
            "InvoiceDate": "2024-01-10 10:30:00",
            "Price": 3.39,
            "CustomerID": None,
            "Country": "United Kingdom",
        },
        # Exact duplicate of row 1 -> must be deduplicated
        {
            "Invoice": "536365",
            "StockCode": "85123A",
            "Description": "WHITE HEART HOLDER",
            "Quantity": 6,
            "InvoiceDate": "2024-01-10 10:00:00",
            "Price": 2.55,
            "CustomerID": 10001,
            "Country": "United Kingdom",
        },
        # Cancellation invoice (prefix 'C' and negative quantity)
        {
            "Invoice": "C536367",
            "StockCode": "85123A",
            "Description": "WHITE HEART HOLDER",
            "Quantity": -2,
            "InvoiceDate": "2024-01-11 12:00:00",
            "Price": 2.55,
            "CustomerID": 10001,
            "Country": "United Kingdom",
        },
        # Normal purchases across customer 10002
        {
            "Invoice": "536369",
            "StockCode": "22423",
            "Description": "REGENCY CAKESTAND",
            "Quantity": 2,
            "InvoiceDate": "2024-01-13 15:00:00",
            "Price": 12.75,
            "CustomerID": 10002,
            "Country": "France",
        },
        # Invalid price (0.00) -> must be filtered
        {
            "Invoice": "536370",
            "StockCode": "22423",
            "Description": "REGENCY CAKESTAND",
            "Quantity": 1,
            "InvoiceDate": "2024-01-14 16:00:00",
            "Price": 0.00,
            "CustomerID": 10003,
            "Country": "Germany",
        },
    ]

    # Add 10 normal small quantity rows to establish standard IQR baseline
    for i in range(10):
        records.append(
            {
                "Invoice": f"53638{i}",
                "StockCode": "21730",
                "Description": "GLASS STAR",
                "Quantity": (i % 4) + 1,
                "InvoiceDate": f"2024-01-{15+i:02d} 11:00:00",
                "Price": 4.25,
                "CustomerID": 10004 + (i % 3),
                "Country": "United Kingdom",
            }
        )

    # Extreme quantity outlier
    records.append(
        {
            "Invoice": "536368",
            "StockCode": "22423",
            "Description": "REGENCY CAKESTAND",
            "Quantity": 5000,
            "InvoiceDate": "2024-01-12 14:00:00",
            "Price": 12.75,
            "CustomerID": 10002,
            "Country": "France",
        }
    )

    return pd.DataFrame(records)


def test_missing_customer_dropped(sample_raw_data: pd.DataFrame) -> None:
    """Verifies records with missing CustomerID are eliminated."""
    purchases, cancellations = clean_transactions(sample_raw_data)
    assert 10001 in purchases["CustomerID"].values
    assert 10002 in purchases["CustomerID"].values
    assert purchases["CustomerID"].isna().sum() == 0
    assert cancellations["CustomerID"].isna().sum() == 0


def test_cancellations_separated(sample_raw_data: pd.DataFrame) -> None:
    """Verifies cancelled orders are accurately separated into cancellations dataframe."""
    purchases, cancellations = clean_transactions(sample_raw_data)
    assert len(cancellations) == 1
    assert cancellations.iloc[0]["Invoice"] == "C536367"
    assert cancellations.iloc[0]["Quantity"] == -2
    assert (purchases["Quantity"] > 0).all()


def test_deduplication(sample_raw_data: pd.DataFrame) -> None:
    """Verifies identical duplicate rows are removed."""
    purchases, _ = clean_transactions(sample_raw_data)
    matches = purchases[(purchases["CustomerID"] == 10001) & (purchases["Invoice"] == "536365")]
    assert len(matches) == 1


def test_outlier_clipping(sample_raw_data: pd.DataFrame) -> None:
    """Verifies extreme values are clipped to IQR boundary."""
    purchases, _ = clean_transactions(sample_raw_data, iqr_factor=1.5)
    # The extreme quantity of 5000 should be clipped well below 5000
    assert purchases["Quantity"].max() < 5000


def test_total_spend_calculated(sample_raw_data: pd.DataFrame) -> None:
    """Verifies TotalSpend calculation."""
    purchases, cancellations = clean_transactions(sample_raw_data)
    row = purchases[purchases["Invoice"] == "536365"].iloc[0]
    expected_spend = round(row["Quantity"] * row["Price"], 2)
    assert row["TotalSpend"] == expected_spend


def test_read_csv_robust_with_byte_0x96() -> None:
    """Verifies read_csv_robust gracefully decodes CP1252 byte 0x96 without raising UnicodeDecodeError."""
    from src.churn.utils import read_csv_robust

    # Byte 0x96 is invalid in utf-8, but valid in cp1252 (en-dash '–')
    raw_bytes = b"CustomerID,segment,monetary\n10001,Champions \x96 VIP,1250.50\n"
    df = read_csv_robust(raw_bytes)
    assert len(df) == 1
    assert df.iloc[0]["CustomerID"] == 10001
    assert "Champions" in df.iloc[0]["segment"]
    assert df.iloc[0]["monetary"] == 1250.50
