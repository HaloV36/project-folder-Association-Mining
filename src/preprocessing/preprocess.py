# src/preprocessing/preprocess.py

from typing import Dict, Set, Tuple
import re

TransactionDB = Dict[str, Set[str]]


def standardize_item_name(name: str) -> str:
    """
    Standardize item names:
    - convert to string
    - strip leading/trailing whitespace
    - collapse multiple spaces
    - lowercase
    """
    return re.sub(r"\s+", " ", str(name).strip()).lower()


def preprocess_transactions(
    transactions: TransactionDB,
    valid_products: Set[str],
) -> Tuple[TransactionDB, str]:
    """
    Apply preprocessing steps:
      - standardize item names
      - if valid_products is non-empty, drop items not in valid_products
      - remove empty transactions
      - remove single-item transactions
      - duplicates are automatically removed by using sets

    Returns:
      cleaned_transactions: dict[tid] -> set(items)
      report: multi-line string describing what was done
    """
    before_count = len(transactions)

    empty_transactions = 0
    single_item_transactions = 0
    invalid_items_count = 0
    removed_transactions = 0
    total_cleaned_items = 0

    cleaned: TransactionDB = {}

    for tid, items in transactions.items():
        # 1) standardize names
        std_items = {standardize_item_name(i) for i in items}

        # 2) filter invalid products if we have a list
        if valid_products:
            valid_items = {i for i in std_items if i in valid_products}
            invalid_items_count += len(std_items) - len(valid_items)
        else:
            valid_items = std_items

        # 3) remove empty and single-item transactions
        if len(valid_items) == 0:
            empty_transactions += 1
            removed_transactions += 1
            continue

        if len(valid_items) == 1:
            single_item_transactions += 1
            removed_transactions += 1
            continue

        cleaned[tid] = valid_items
        total_cleaned_items += len(valid_items)

    after_count = len(cleaned)
    if cleaned:
        all_clean_items = set().union(*cleaned.values())
    else:
        all_clean_items = set()

    report_lines = [
        "Preprocessing Report:",
        "---------------------",
        "Before Cleaning:",
        f"- Total transactions: {before_count}",
        f"- Empty transactions (or became empty): {empty_transactions}",
        f"- Single-item transactions: {single_item_transactions}",
        f"- Invalid items removed: {invalid_items_count}",
        "",
        "After Cleaning:",
        f"- Valid transactions: {after_count}",
        f"- Total items (after cleaning): {total_cleaned_items}",
        f"- Unique products: {len(all_clean_items)}",
        f"- Transactions removed: {removed_transactions}",
    ]

    report = "\n".join(report_lines)
    return cleaned, report
