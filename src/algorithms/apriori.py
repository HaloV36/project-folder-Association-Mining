# src/algorithms/apriori.py

from typing import Dict, Set, List, Tuple, FrozenSet
from itertools import combinations
import time

TransactionDB = Dict[str, Set[str]]
Itemset = FrozenSet[str]


def _generate_candidates(prev_frequents: List[Itemset], k: int) -> List[Itemset]:
    """
    Generate candidate k-itemsets from (k-1)-itemsets (Apriori join and prune step).
    """
    candidates = set()
    n = len(prev_frequents)
    for i in range(n):
        for j in range(i + 1, n):
            union = prev_frequents[i] | prev_frequents[j]
            if len(union) == k:
                # prune: all (k-1)-subsets must be frequent
                all_subsets_frequent = True
                for subset in combinations(union, k - 1):
                    if frozenset(subset) not in prev_frequents:
                        all_subsets_frequent = False
                        break
                if all_subsets_frequent:
                    candidates.add(union)
    return list(candidates)


def apriori(
    transactions: TransactionDB,
    min_support: float
) -> Tuple[Dict[Itemset, float], float]:
    """
    Apriori algorithm (horizontal format).
    Returns:
      frequent_itemsets: dict[itemset] = support (relative)
      elapsed_time: runtime in seconds
    """
    start = time.time()

    # Count 1-itemsets
    item_counts: Dict[str, int] = {}
    for items in transactions.values():
        for i in items:
            item_counts[i] = item_counts.get(i, 0) + 1

    n_trans = len(transactions)
    frequents: Dict[Itemset, float] = {}
    Lk: List[Itemset] = []

    # L1
    for item, count in item_counts.items():
        sup = count / n_trans
        if sup >= min_support:
            fs = frozenset([item])
            frequents[fs] = sup
            Lk.append(fs)

    # Lk for k >= 2
    k = 2
    while Lk:
        Ck = _generate_candidates(Lk, k)
        support_counts: Dict[Itemset, int] = {c: 0 for c in Ck}

        for items in transactions.values():
            for c in Ck:
                if c.issubset(items):
                    support_counts[c] += 1

        Lk = []
        for c, count in support_counts.items():
            sup = count / n_trans
            if sup >= min_support:
                frequents[c] = sup
                Lk.append(c)

        k += 1

    elapsed = time.time() - start
    return frequents, elapsed


def generate_association_rules(
    frequent_itemsets: Dict[Itemset, float],
    min_confidence: float
) -> List[dict]:
    """
    Generate association rules X -> Y from frequent itemsets.

    Returns a list of dicts with keys:
      - antecedent (frozenset)
      - consequent (frozenset)
      - support (float)
      - confidence (float)
      - lift (float)
    """
    rules: List[dict] = []

    for itemset, sup_xy in frequent_itemsets.items():
        if len(itemset) < 2:
            continue

        items = list(itemset)
        for r in range(1, len(items)):
            for antecedent_tuple in combinations(items, r):
                antecedent = frozenset(antecedent_tuple)
                consequent = itemset - antecedent

                sup_x = frequent_itemsets.get(antecedent, 0.0)
                sup_y = frequent_itemsets.get(consequent, 0.0)
                if sup_x == 0 or sup_y == 0:
                    continue

                conf = sup_xy / sup_x
                if conf >= min_confidence:
                    lift = conf / sup_y
                    rules.append(
                        {
                            "antecedent": antecedent,
                            "consequent": consequent,
                            "support": sup_xy,
                            "confidence": conf,
                            "lift": lift,
                        }
                    )
    return rules
