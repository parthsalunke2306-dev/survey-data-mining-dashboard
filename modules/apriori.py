"""
Apriori Algorithm: Association Rule Mining & Behavioral Basket Analysis
Discovers frequent itemsets and association rules from multi-select and behavioral survey indicators.
Computes Support, Confidence, and Lift.
"""
from itertools import combinations
import pandas as pd
from typing import List, Dict, Tuple, Any
from modules.data_processor import parse_multiselect

def extract_transactions(df: pd.DataFrame, basket_type: str = "spending") -> List[List[str]]:
    """
    Extracts transaction baskets based on the requested basket type:
    - 'spending': Recent spending multi-select categories in the last 7 days.
    - 'assets': Preferred asset classes multi-select over the next 5 years.
    - 'behavior': Discrete behavioral traits (tracking rigor, emergency buffer, research, SIP intention, confidence, planning).
    """
    transactions = []
    if len(df) == 0:
        return transactions

    if basket_type == "spending":
        col = "Recent_Spending"
        if col in df.columns:
            for val in df[col].dropna():
                items = parse_multiselect(val)
                if items:
                    transactions.append(items)

    elif basket_type == "assets":
        col = "Asset_Interests"
        if col in df.columns:
            for val in df[col].dropna():
                items = parse_multiselect(val)
                if items:
                    transactions.append(items)

    elif basket_type == "behavior":
        for _, row in df.iterrows():
            b = []
            # 1. Expense Tracking
            if row.get("Tracking_Method") in ["Mobile App", "Spreadsheet", "Pen and Paper"]:
                b.append("Active Tracking")
            else:
                b.append("Informal/Mental Tracking")
            # 2. Emergency Savings
            if str(row.get("Has_Emergency_Fund", "")).lower() == "yes":
                b.append("Has Emergency Fund")
            else:
                b.append("No Emergency Cushion")
            # 3. Financial Research
            if row.get("Research_Frequency") in ["Daily", "Weekly"]:
                b.append("Active Market Research")
            else:
                b.append("Passive/Rare Research")
            # 4. Investment Automation (SIP)
            if str(row.get("Plan_Automated_Invest", "")).lower() == "yes":
                b.append("Plans SIP Automation")
            else:
                b.append("No SIP Plan")
            # 5. Financial Confidence
            if pd.to_numeric(row.get("Financial_Confidence"), errors="coerce") >= 4:
                b.append("High Confidence")
            else:
                b.append("Moderate/Low Confidence")
            # 6. Actionable Plan
            if pd.to_numeric(row.get("Wealth_Plan_Readiness"), errors="coerce") >= 4:
                b.append("Has Actionable Plan")
            else:
                b.append("No Actionable Plan")
            # 7. Demat / Brokerage ownership
            if str(row.get("Has_Investment_Account", "")).lower() == "yes":
                b.append("Owns Demat/Brokerage")

            transactions.append(b)

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
        candidates = set()
        for i in range(len(current_itemsets)):
            for j in range(i + 1, len(current_itemsets)):
                union = current_itemsets[i] | current_itemsets[j]
                if len(union) == k:
                    candidates.add(union)

        cand_counts = {c: 0 for c in candidates}
        for t in transactions:
            t_set = set(t)
            for c in candidates:
                if c.issubset(t_set):
                    cand_counts[c] += 1

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
                        if conf >= min_confidence:
                            lift = conf / sup_B
                            ant_str = ", ".join(sorted(list(antecedent)))
                            con_str = ", ".join(sorted(list(consequent)))
                            rules_records.append({
                                "Antecedent": ant_str,
                                "Consequent": con_str,
                                "Support": round(sup_AB, 4),
                                "Support_Pct": f"{sup_AB * 100:.1f}%",
                                "Confidence": round(conf, 4),
                                "Confidence_Pct": f"{conf * 100:.1f}%",
                                "Lift": round(lift, 3),
                                "Rule": f"IF {{{ant_str}}} THEN {{{con_str}}}"
                            })

    rules_df = pd.DataFrame(rules_records)
    if not rules_df.empty:
        rules_df = rules_df.sort_values(by=["Lift", "Confidence"], ascending=False).reset_index(drop=True)
    else:
        rules_df = pd.DataFrame(columns=["Antecedent", "Consequent", "Support", "Support_Pct", "Confidence", "Confidence_Pct", "Lift", "Rule"])

    return itemsets_df, rules_df
