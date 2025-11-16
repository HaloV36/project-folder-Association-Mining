# src/algorithms/eclat.py

from typing import Dict, Set, List, Tuple, FrozenSet
import time
import math
from itertools import combinations

TransactionDB = Dict[str, Set[str]]
Itemset = FrozenSet[str]
TIDSet = Set[str]


def build_vertical_representation(
    transactions: TransactionDB
) -> Dict[str, TIDSet]:
    """
    Build vertical representation:
      item -> set of transaction IDs (TID-set)
    """
    vertical: Dict[str, TIDSet] = {}
    for tid, items in transactions.items():
        for item in items:
            if item not in vertical:
                vertical[item] = set()
            vertical[item].add(tid)
    return vertical


def _eclat_recursive(
    prefix: Itemset,
    items_tids: List[Tuple[str, TIDSet]],
    min_support_count: int,
    results: Dict[Itemset, float],
    total_transactions: int,
):
    """
    Recursive depth-first Eclat.
    """
    while items_tids:
        item, tids = items_tids.pop()
        new_itemset = prefix | frozenset([item])
        support_count = len(tids)

        if support_count >= min_support_count:
            support = support_count / total_transactions
            results[new_itemset] = support

            new_items_tids: List[Tuple[str, TIDSet]] = []
            for other_item, other_tids in items_tids:
                inter = tids & other_tids
                if len(inter) >= min_support_count:
                    new_items_tids.append((other_item, inter))

            _eclat_recursive(
                new_itemset,
                new_items_tids,
                min_support_count,
                results,
                total_transactions,
            )


def eclat(
    transactions: TransactionDB,
    min_support: float
) -> Tuple[Dict[Itemset, float], float]:
    """
    Eclat algorithm (vertical format).
    Returns:
      frequent_itemsets: dict[itemset] = support (relative)
      elapsed_time: runtime in seconds
    """
    start = time.time()
    vertical = build_vertical_representation(transactions)
    n_trans = len(transactions)
    min_sup_count = max(1, math.ceil(min_support * n_trans))


    items_tids = [(item, tids) for item, tids in vertical.items()]
    results: Dict[Itemset, float] = {}
    _eclat_recursive(frozenset(), items_tids, min_sup_count, results, n_trans)
    elapsed = time.time() - start
    return results, elapsed


def generate_association_rules_eclat(
    frequent_itemsets: Dict[Itemset, float],
    min_confidence: float
) -> List[dict]:
    """
    Generate association rules from Eclat's frequent itemsets.
    Uses the same logic as Apriori's rule generation.
    """
    from algorithms.apriori import generate_association_rules
    return generate_association_rules(frequent_itemsets, min_confidence)
