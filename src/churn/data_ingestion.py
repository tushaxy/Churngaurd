"""Data ingestion module for ChurnGuard.

Loads UCI Online Retail II dataset if available in data/raw/, otherwise generates
a realistic synthetic e-commerce transaction dataset with calibrated Pareto spend,
seasonality, cancellations, and recency-correlated churn behavior.
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

from src.churn.config import settings
from src.churn.utils import get_logger, set_random_seed

logger = get_logger(__name__)

PRODUCT_CATALOG = [
    ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2.55),
    ("71053", "WHITE METAL LANTERN", 3.39),
    ("84406B", "CREAM CUPID HEARTS COAT HANGER", 2.75),
    ("84029G", "KNITTED UNION FLAG HOT WATER BOTTLE", 3.75),
    ("84029E", "RED WOOLLY HOTTIE WHITE HEART.", 3.75),
    ("22752", "SET 7 BABUSHKA NESTING BOXES", 7.65),
    ("21730", "GLASS STAR FROSTED T-LIGHT HOLDER", 4.25),
    ("22633", "HAND WARMER UNION JACK", 1.85),
    ("22632", "HAND WARMER RED RETROSPOT", 1.85),
    ("84879", "ASSORTED COLOUR BIRD ORNAMENT", 1.69),
    ("22745", "POPPY'S PLAYHOUSE BEDROOM ", 2.10),
    ("22748", "POPPY'S PLAYHOUSE KITCHEN", 2.10),
    ("22749", "FELTCRAFT PRINCESS CHARLOTTE DOLL", 3.75),
    ("22310", "IVORY KNITTED MUG COSY ", 1.65),
    ("84969", "BOX OF 6 ASSORTED COLOUR TEASPOONS", 4.25),
    ("22423", "REGENCY CAKESTAND 3 TIER", 12.75),
    ("22720", "SET OF 3 CAKE TINS PANTRY DESIGN ", 4.95),
    ("22722", "SET OF 6 SPICE TINS PANTRY DESIGN", 3.95),
    ("21754", "HOME BUILDING BLOCK WORD", 5.95),
    ("21755", "LOVE BUILDING BLOCK WORD", 5.95),
    ("21777", "RECIPE BOX WITH METAL HEART", 7.95),
    ("48187", "DOORMAT NEW ENGLAND", 6.75),
    ("22960", "JAM MAKING SET WITH JARS", 4.25),
    ("22910", "PAPER CHAIN KIT VINTAGE CHRISTMAS", 2.55),
    ("22629", "SPACEBOY LUNCH BOX ", 1.95),
    ("22630", "DOLLY GIRL LUNCH BOX", 1.95),
    ("22621", "TRADITIONAL KNITTING NANCY", 1.45),
    ("22457", "NATURAL SLATE HEART CHALKBOARD ", 2.95),
    ("22469", "HEART OF WICKER SMALL", 1.45),
    ("22470", "HEART OF WICKER LARGE", 2.55),
    ("22197", "POPCORN HOLDER", 0.85),
    ("85099B", "JUMBO BAG RED RETROSPOT", 2.08),
    ("21931", "JUMBO STORAGE BAG SUKI", 2.08),
    ("85099C", "JUMBO BAG STRAWBERRY", 2.08),
    ("20725", "LUNCH BAG RED RETROSPOT", 1.65),
    ("20727", "LUNCH BAG BLACK SKULL.", 1.65),
    ("20728", "LUNCH BAG CARS BLUE", 1.65),
    ("22382", "LUNCH BAG SPACEBOY DESIGN ", 1.65),
    ("22383", "LUNCH BAG SUKI DESIGN ", 1.65),
    ("22384", "LUNCH BAG PINK POLKADOT", 1.65),
    ("20726", "LUNCH BAG WOODLAND", 1.65),
    ("22386", "JUMBO BAG PINK POLKADOT", 2.08),
    ("22411", "JUMBO SHOPPER VINTAGE RED PAISLEY", 2.08),
    ("22138", "BAKING SET 9 PIECE RETROSPOT ", 4.95),
    ("22139", "RETROSPOT TEA SET CERAMIC 11 PC ", 9.95),
    ("21212", "PACK OF 72 RETROSPOT CAKE CASES", 0.55),
    ("84991", "60 TEATIME FAIRY CAKE CASES", 0.55),
    ("21977", "PACK OF 60 PINK PAISLEY CAKE CASES", 0.55),
    ("84992", "72 SWEETHEART FAIRY CAKE CASES", 0.55),
    ("22086", "PAPER CHAIN KIT 50'S CHRISTMAS ", 2.55),
    ("21523", "DOORMAT FANCY FONT HOME SWEET HOME", 6.75),
    ("21524", "DOORMAT SPOTTY HOME SWEET HOME", 6.75),
    ("22699", "ROSES REGENCY TEACUP AND SAUCER ", 2.95),
    ("22697", "GREEN REGENCY TEACUP AND SAUCER", 2.95),
    ("22698", "PINK REGENCY TEACUP AND SAUCER", 2.95),
    ("22961", "JAM VALUE PACK OF 20", 1.45),
    ("22962", "JAM JAR WALL CLOCK", 8.15),
    ("22963", "JAM JAR LIGHT", 16.95),
    ("23084", "RABBIT NIGHT LIGHT", 2.08),
    ("POST", "POSTAGE", 18.00),
]

COUNTRIES = [
    ("United Kingdom", 0.88),
    ("Germany", 0.035),
    ("France", 0.030),
    ("EIRE", 0.020),
    ("Spain", 0.010),
    ("Netherlands", 0.008),
    ("Belgium", 0.005),
    ("Switzerland", 0.005),
    ("Portugal", 0.004),
    ("Australia", 0.003),
]


def load_raw_dataset(raw_dir: str | None = None) -> pd.DataFrame | None:
    """Attempts to discover and load an existing raw dataset from disk.

    Args:
        raw_dir: Optional override for the raw data directory.

    Returns:
        DataFrame if file found and loaded, else None.
    """
    directory = Path(raw_dir or settings.paths.raw_data_dir)
    if not directory.exists():
        return None

    # Check for Online Retail files
    candidates = list(directory.glob("online_retail*.csv")) + list(
        directory.glob("online_retail*.xlsx")
    )
    if not candidates:
        return None

    target = candidates[0]
    logger.info("Found existing raw dataset: %s", target)
    if target.suffix.lower() == ".csv":
        return pd.read_csv(target, encoding="utf-8-sig")
    return pd.read_excel(target)


def generate_synthetic_transactions(
    n_customers: int | None = None,
    n_transactions: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generates a realistic synthetic e-commerce transaction dataset.

    Calibrated so that:
    1. Top ~20% of customers drive ~65% of total revenue (Pareto alpha=1.267).
    2. Recency > 90 days defines customer churn (~25-27% churn rate).
    3. Realistic seasonality, cancellations (~5.5%), and transaction clusters.

    Args:
        n_customers: Number of distinct customers (default 8,000).
        n_transactions: Target number of transaction rows (default 250,000).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame of synthetic retail transactions.
    """
    n_cust = n_customers or settings.synthetic_data.n_customers
    n_trans = n_transactions or settings.synthetic_data.n_transactions
    rng_seed = seed or settings.synthetic_data.random_state
    set_random_seed(rng_seed)
    fake = Faker()
    fake.seed_instance(rng_seed)

    logger.info(
        "Generating synthetic dataset: %d customers, target %d transactions...",
        n_cust,
        n_trans,
    )

    # 1. Customer master setup
    customer_ids = np.arange(10001, 10001 + n_cust)

    # Assign country per customer
    country_names = [c[0] for c in COUNTRIES]
    country_probs = [c[1] for c in COUNTRIES]
    country_probs = np.array(country_probs) / sum(country_probs)
    customer_countries = np.random.choice(country_names, size=n_cust, p=country_probs)

    # 2. Spend allocation: lognormal with sigma=1.208 calibrates top 20% to ~65% revenue share
    target_spends = np.random.lognormal(mean=5.5, sigma=1.208, size=n_cust)
    target_spends = np.clip(target_spends, 15.0, 45000.0)

    # 3. Churn behavior assignment (~26.5% base churn rate)
    is_churned = np.random.binomial(1, 0.265, size=n_cust).astype(bool)

    # Date range: End date = 2024-12-31, 730 days of history
    end_date = datetime(2024, 12, 31, 23, 59, 59)
    total_days = 730

    # Churned customers have last purchase > 90 days ago (92 to 450 days)
    churn_last_purchase_offset = np.random.randint(92, 450, size=n_cust)
    # Active customers have last purchase <= 89 days ago (1 to 88 days)
    active_last_purchase_offset = np.random.randint(1, 89, size=n_cust)
    last_purchase_offset = np.where(
        is_churned, churn_last_purchase_offset, active_last_purchase_offset
    )

    # Customer tenure (days before last purchase)
    customer_tenure_days = np.random.randint(30, total_days - last_purchase_offset + 1, size=n_cust)
    first_purchase_offset = last_purchase_offset + customer_tenure_days

    # Order count per customer: proportional to spend
    orders_per_cust = np.maximum(1, np.round((target_spends / 36.0) ** 0.82).astype(int))
    orders_per_cust = np.clip(orders_per_cust, 1, 95)

    # 4. Generate Invoices and line items
    logger.info("Synthesizing orders and item baskets across %d customers...", n_cust)
    records = []
    invoice_counter = 536365
    n_catalog = len(PRODUCT_CATALOG)

    for i in range(n_cust):
        c_id = int(customer_ids[i])
        c_country = str(customer_countries[i])
        n_orders = int(orders_per_cust[i])
        f_offset = int(first_purchase_offset[i])
        l_offset = int(last_purchase_offset[i])
        cust_target_spend = float(target_spends[i])

        # Allocate target spend across orders
        if n_orders == 1:
            order_target_spends = [cust_target_spend]
            order_offsets = [l_offset]
        else:
            proportions = np.random.dirichlet(np.ones(n_orders))
            order_target_spends = cust_target_spend * proportions
            # Guarantee that l_offset is the exact latest order
            intermediate_offsets = np.random.randint(l_offset + 1, f_offset + 1, size=n_orders - 1)
            order_offsets = np.concatenate([[l_offset], intermediate_offsets])
            order_offsets = np.sort(
                order_offsets
            )  # ascending (l_offset is smallest, i.e. most recent)

        for o_idx in range(n_orders):
            offset = int(order_offsets[o_idx])
            order_spend = max(5.0, float(order_target_spends[o_idx]))

            order_date = end_date - timedelta(
                days=offset,
                hours=int(np.random.randint(8, 19)),
                minutes=int(np.random.randint(0, 60)),
                seconds=int(np.random.randint(0, 60)),
            )

            invoice_str = str(invoice_counter)
            invoice_counter += 1

            # Is this invoice cancelled? (~5.2% probability)
            is_cancelled = np.random.rand() < settings.synthetic_data.cancellation_rate
            if is_cancelled:
                invoice_str = f"C{invoice_str}"

            # Number of distinct items in basket (1 to 6 items)
            basket_size = int(
                np.random.choice([1, 2, 3, 4, 5, 6], p=[0.20, 0.30, 0.25, 0.15, 0.07, 0.03])
            )
            item_indices = np.random.choice(n_catalog, size=basket_size, replace=False)

            # Spend per item in basket
            if basket_size == 1:
                item_spends = [order_spend]
            else:
                item_spends = order_spend * np.random.dirichlet(np.ones(basket_size))

            for b_idx in range(basket_size):
                stock_code, desc, base_price = PRODUCT_CATALOG[item_indices[b_idx]]
                price = round(base_price * float(np.random.uniform(0.96, 1.04)), 2)

                item_spend = max(price, float(item_spends[b_idx]))
                qty = max(1, int(round(item_spend / price)))

                if is_cancelled:
                    qty = -abs(qty)

                records.append(
                    (
                        invoice_str,
                        stock_code,
                        desc,
                        qty,
                        order_date.strftime("%Y-%m-%d %H:%M:%S"),
                        price,
                        c_id,
                        c_country,
                    )
                )

    df = pd.DataFrame(
        records,
        columns=[
            "Invoice",
            "StockCode",
            "Description",
            "Quantity",
            "InvoiceDate",
            "Price",
            "CustomerID",
            "Country",
        ],
    )

    logger.info("Synthetic transactions generated: %d rows.", len(df))
    return df


def ingest_and_save_raw_data() -> pd.DataFrame:
    """Ingests dataset (loading existing or synthesizing) and saves to data/raw.

    Returns:
        Raw transactions DataFrame.
    """
    raw_file = Path(settings.paths.raw_transactions_file)
    raw_file.parent.mkdir(parents=True, exist_ok=True)

    df = load_raw_dataset()
    if df is None:
        logger.info(
            "No raw data found in %s. Generating synthetic data...", settings.paths.raw_data_dir
        )
        df = generate_synthetic_transactions()
        df.to_csv(raw_file, index=False, encoding="utf-8-sig")
        logger.info("Saved raw synthetic transactions to %s", raw_file)
    else:
        logger.info("Loaded %d raw transaction rows.", len(df))

    return df


if __name__ == "__main__":
    ingest_and_save_raw_data()
