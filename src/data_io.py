# src/data_io.py

import pandas as pd
from typing import Dict, Set


import pandas as pd
from typing import Dict, Set

def load_transactions_csv(path: str) -> pd.DataFrame:
    """
    Handles a few formats:

    1) Long format, one item column (possibly comma-separated):
        transaction_id, item
        1, "milk"
        1, "bread"

    2) Long format, items column with comma-separated lists:
        transaction_id, items
        1, "milk, bread, eggs"

    3) Wide format, multiple item columns:
        transaction_id, item1, item2, item3, ...

    We always normalize to a long DataFrame with cols: ['tid', 'item'].
    """
    df = pd.read_csv(path)

    # Normalize column names for detection
    cols_lower = [c.lower() for c in df.columns]

    # Try to detect a "transaction id" column
    tid_col = None
    for c in df.columns:
        cl = c.lower()
        if cl in ["transaction_id", "tid", "id"]:
            tid_col = c
            break
    if tid_col is None:
        # Fallback: assume first column is transaction id
        tid_col = df.columns[0]

    # Try to detect a single "item(s)" column
    item_col = None
    for c in df.columns:
        cl = c.lower()
        if cl in ["item", "items", "product", "products"]:
            item_col = c
            break

    # Case A: we have a single item/items column (possibly comma-separated)
    if item_col is not None and len(df.columns) == 2:
        long_rows = []
        for _, row in df.iterrows():
            tid = row[tid_col]
            cell = row[item_col]

            if pd.isna(cell):
                continue

            # Split by comma in case it's "milk, bread, eggs"
            raw = str(cell)
            for token in raw.split(","):
                val = token.strip()
                if val != "":
                    long_rows.append({"tid": tid, "item": val})

        return pd.DataFrame(long_rows)

    # Case B: our old "long format" with (transaction_id, item) and extra columns
    if {"transaction_id", "item"}.issubset(set(cols_lower)):
        # Map back to actual column names
        trans_col = df.columns[cols_lower.index("transaction_id")]
        item_col  = df.columns[cols_lower.index("item")]

        long_rows = []
        for _, row in df.iterrows():
            tid = row[trans_col]
            cell = row[item_col]
            if pd.isna(cell):
                continue
            raw = str(cell)
            for token in raw.split(","):
                val = token.strip()
                if val != "":
                    long_rows.append({"tid": tid, "item": val})
        return pd.DataFrame(long_rows)

    # Case C: wide format – first col is tid, rest are item columns
    cols = list(df.columns)
    tid_col = cols[0]
    item_cols = cols[1:]

    long_rows = []
    for _, row in df.iterrows():
        tid = row[tid_col]
        for c in item_cols:
            val = str(row[c]).strip()
            if val != "" and val.lower() != "nan":
                long_rows.append({"tid": tid, "item": val})

    return pd.DataFrame(long_rows)


def load_products_csv(path: str) -> pd.DataFrame:
    """
    Expected columns: product_id, name (or similar).
    We normalize to ['product_id', 'name'].
    """
    df = pd.read_csv(path)

    # Ensure product_id column exists
    if "product_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "product_id"})

    # Try to find a name-like column
    name_col = None
    for c in df.columns:
        if c.lower() in ["name", "product_name", "item_name"]:
            name_col = c
            break
    if name_col is None:
        # fallback: second column
        name_col = df.columns[1]

    df = df[["product_id", name_col]].rename(columns={name_col: "name"})
    return df


def df_to_transactions(df: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    Convert a long df (tid, item) into dict: tid -> set(items)
    """
    transactions: Dict[str, Set[str]] = {}
    for tid, group in df.groupby("tid"):
        transactions[str(tid)] = set(group["item"].astype(str))
    return transactions


def basic_stats(transactions: Dict[str, Set[str]]) -> dict:
    """
    Simple statistics about the transaction DB:
      - number of transactions
      - total items (counting duplicates)
      - unique items
    """
    all_items = set()
    total_items = 0
    for items in transactions.values():
        all_items |= items
        total_items += len(items)

    return {
        "transaction_count": len(transactions),
        "total_items": total_items,
        "unique_items": len(all_items),
    }
