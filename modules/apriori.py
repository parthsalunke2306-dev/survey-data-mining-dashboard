"""
Apriori Algorithm: Association Rule Mining & Market Basket Analysis
Discovers frequent itemsets and association rules from multi-select survey responses.
"""
from itertools import combinations
import pandas as pd
from typing import List, Dict, Tuple, Any

def extract_transactions(df: pd.DataFrame, column: str) -> List[List[str]]:
    """Extracts comma-separated multi-select survey items into transaction baskets."""
    transactions = []
    if column not in df.columns:
        return transactions
    for items_str in df[column].dropna():
        # Split by comma and clean whitespace
        basket = [item.strip() for item in str(items_str).split(",") if item.strip()]
        if basket:
            transactions.append(basket)
    return transactions

def run_apriori(transactions: List[List[str]], min_support: float = 0.15, min_confidence: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pure Python Apriori Implementation:
    Computes Support, Confidence, and Lift for survey baskets.
    
    Support(A) = Count(A) / Total_Transactions
    Confidence(A -> B) = Support(A U B) / Support(A)
    Lift(A -> B) = Confidence(A -> B) / Support(B)
    """
    N = len(transactions)
    if N == 0:
        return pd.DataFrame(), pd.DataFrame()

    # Step 1: Count single item frequencies (L1)
    item_counts = {}
    for t in transactions:
        for item in set(t):
            item_counts[item] = item_counts.get(item, 0) + 1

    frequent_items = {}
    for item, count in item_counts.items():
        sup = count / N
        if sup >= min_support:
            frequent_items[frozenset([item])] = sup

    # Step 2: Generate 2-itemsets and 3-itemsets (Lk)
    all_frequent_itemsets = dict(frequent_items)
    current_itemsets = list(frequent_items.keys())
    k = 2

    while current_itemsets and k <= 3:
        # Candidate generation
        candidates = set()
        for i in range(len(current_itemsets)):
            for j in range(i + 1, len(current_itemsets)):
                union = current_itemsets[i] | current_itemsets[j]
                if len(union) == k:
                    candidates.add(union)

        # Count support for candidates
        cand_counts = {c: 0 for c in candidates}
        for t in transactions:
            t_set = set(t)
            for c in candidates:
                if c.issubset(t_set):
                    cand_counts[c] += 1

        # Filter by min_support
        current_itemsets = []
        for c, count in cand_counts.items():
            sup = count / N
            if sup >= min_support:
                all_frequent_itemsets[c] = sup
                current_itemsets.append(c)
        k += 1

    # Format frequent itemsets table
    itemsets_records = []
    for itemset, sup in all_frequent_itemsets.items():
        itemsets_records.append({
            "Itemset": ", ".join(sorted(list(itemset))),
            "Size": len(itemset),
            "Support": round(sup, 4),
            "Support_Pct": f"{sup * 100:.1f}%",
            "Count": int(round(sup * N))
        })
    itemsets_df = pd.DataFrame(itemsets_records).sort_values(by="Support", ascending=False).reset_index(drop=True)

    # Step 3: Generate Association Rules
    rules_records = []
    for itemset, sup_AB in all_frequent_itemsets.items():
        if len(itemset) > 1:
            for r in range(1, len(itemset)):
                for antecedent in combinations(itemset, r):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    sup_A = all_frequent_itemsets.get(antecedent, 0)
                    sup_B = all_frequent_itemsets.get(consequent, 0)

                    if sup_A > 0 and sup_B > 0:
                        conf = sup_AB / sup_A
                        lift = conf / sup_B
                        if conf >= min_confidence:
                            rules_records.append({
                                "Antecedent (IF)": ", ".join(sorted(list(antecedent))),
                                "Consequent (THEN)": ", ".join(sorted(list(consequent))),
                                "Support": round(sup_AB, 4),
                                "Confidence": round(conf, 4),
                                "Lift": round(lift, 4),
                                "Rule": f"IF {{{', '.join(sorted(list(antecedent)))}}} THEN {{{', '.join(sorted(list(consequent)))}}}"
                            })

    rules_df = pd.DataFrame(rules_records)
    if not rules_df.empty:
        rules_df = rules_df.sort_values(by=["Lift", "Confidence"], ascending=False).reset_index(drop=True)

    return itemsets_df, rules_df
