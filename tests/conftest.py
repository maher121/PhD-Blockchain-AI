"""Shared V0.2 test fixtures (small synthetic DataCo-like tables).

The fixtures are deliberately crafted to mirror the real DataCo columns the
V0.2 pipeline needs, including missing values, an outcome-adjacent column,
order-id grouping and a target flag. They do NOT claim to reproduce the
real dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

_FIXTURE_COLUMNS = [
    "row_id",
    "Type",
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Late_delivery_risk",
    "Category Name",
    "Department Name",
    "Market",
    "Order Id",
    "Order Item Id",
    "Order Item Quantity",
    "Order Item Product Price",
    "Product Price",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Profit Ratio",
    "Order Item Total",
    "Sales",
    "Order Customer Id",
    "Product Card Id",
    "order date (DateOrders)",
    "shipping date (DateOrders)",
    "Order Status",
    "Delivery Status",
    "Order Region",
    "Shipping Mode",
    "Customer Segment",
    "Product Name",
]


def make_dataco_like_frame(n_orders: int = 8, rows_per_order: int = 4, seed: int = 42) -> pd.DataFrame:
    """Deterministic small table shaped like the DataCo schema."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    row_id = 0
    for order_idx in range(n_orders):
        order_id = f"ORD-{1000 + order_idx}"
        customer = f"CUST-{100 + (order_idx % 4)}"
        n_items = rows_per_order
        order_total = 0.0
        for item_idx in range(n_items):
            quantity = float(rng.integers(1, 6))
            unit_price = float(rng.uniform(10.0, 120.0))
            price = unit_price * rng.uniform(0.95, 1.05)
            discount_rate = float(rng.choice([0.0, 0.1, 0.25]))
            discount = unit_price * discount_rate
            line_total = quantity * unit_price
            order_total += line_total
            scheduled = float(int(rng.integers(1, 5)))
            real = scheduled + float(rng.choice([0.0, 1.0, -1.0]))
            late = int(real > scheduled)
            rows.append(
                {
                    "row_id": row_id,
                    "Order Id": order_id,
                    "Order Item Id": f"ITEM-{row_id + 5000}",
                    "Order Customer Id": customer,
                    "Product Card Id": f"PROD-{1000 + (row_id % 5)}",
                    "Type": rng.choice(["DEBIT", "CASH", "PAYMENT", "TRANSFER"]),
                    "Market": rng.choice(["Europe", "USCA", "LATAM"]),
                    "Shipping Mode": rng.choice(["Standard Class", "First Class", "Second Class"]),
                    "Customer Segment": rng.choice(["Consumer", "Corporate", "Home Office"]),
                    "Department Name": rng.choice(
                        ["Technology", "Apparel", "Fitness", "Book Shop"]
                    ),
                    "Category Name": f"cat-{rng.integers(1, 6)}",
                    "Order Region": rng.choice(["West", "Central"]),
                    "Order Status": rng.choice(["COMPLETE", "PENDING", "PROCESSING"]),
                    "Delivery Status": (
                        "Late delivery" if late else "Shipping on time"
                    ),
                    "Product Name": f"Product {row_id % 7}",
                    "order date (DateOrders)": pd.Timestamp("2015-01-01 12:00:00")
                    + pd.Timedelta(days=order_idx, hours=item_idx),
                    "shipping date (DateOrders)": pd.Timestamp("2015-01-03 12:00:00")
                    + pd.Timedelta(days=order_idx, hours=item_idx),
                    "Days for shipping (real)": real,
                    "Days for shipment (scheduled)": scheduled,
                    "Order Item Quantity": quantity,
                    "Order Item Product Price": unit_price,
                    "Product Price": price,
                    "Order Item Discount": discount,
                    "Order Item Discount Rate": discount_rate,
                    "Order Item Profit Ratio": float(rng.uniform(0.2, 0.6)),
                    "Order Item Total": line_total,
                    "Sales": line_total - discount,
                    "Late_delivery_risk": late,
                }
            )
            row_id += 1
    df = pd.DataFrame(rows, columns=_FIXTURE_COLUMNS)
    # Inject a couple of quality issues so preprocessing logic is exercised:
    df.loc[1, "Product Price"] = np.nan  # missing numeric (imputed)
    df.loc[2, "Order Item Quantity"] = -5.0  # impossible negative (reported)
    df.loc[3, "Late_delivery_risk"] = np.nan  # missing target (dropped at transform)
    df.loc[4, "Market"] = np.nan  # missing categorical (mode/imputed "")
    return df


@pytest.fixture(scope="session")
def dataco_like_frame() -> pd.DataFrame:
    return make_dataco_like_frame()


def write_dataco_like_csv(directory, **kwargs) -> "Path":
    """Write a DataCo-like fixture CSV into ``directory`` and return its path."""
    import pathlib

    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frame = make_dataco_like_frame(**kwargs)
    path = directory / "DataCoSupplyChainDataset.csv"
    frame.drop(columns=["row_id"]).to_csv(path, index=False, encoding="latin1")
    return path