"""
Apriori Algorithm: Association Rule Mining Module
==================================================
VIVA EXPLANATION GUIDE FOR PROFESSOR:
1. What this code does:
   Performs Market Basket Analysis to discover behavioral association rules.
   Example: "IF a student tracks expenses actively -> THEN they also plan automated SIPs" with high confidence.
2. Python Libraries used:
   - mlxtend: Standard academic library for Apriori and association rule mining.
   - pandas: For transaction table structuring.
3. Built-in functions used:
   - mlxtend.preprocessing.TransactionEncoder(): Converts list of transaction baskets into a one-hot boolean DataFrame.
   - mlxtend.frequent_patterns.apriori(): Discovers all frequent itemsets exceeding the minimum support threshold.
   - mlxtend.frequent_patterns.association_rules(): Generates (Antecedent -> Consequent) rules with Support, Confidence, and Lift metrics.
4. Why these built-ins are used:
   Eliminates complex manual candidate-generation loops by using optimized vectorized boolean matrix operations.
5. Output produced:
   Two DataFrames: frequent itemsets and ranked association rules.
"""

import pandas as pd
from typing import List, Tuple
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from modules.data_processor import parse_multiselect


def extract_transactions(df: pd.DataFrame, basket_type: str = "behavior") -> List[List[str]]:
    """
    Extracts transaction item baskets based on selected survey domain.
    """
    transactions = []
    if len(df) == 0:
        return transactions

    if basket_type == "spending" and "Recent_Spending" in df.columns:
        # Multi-select discretionary categories in the prior 7 days
        for val in df["Recent_Spending"].dropna():
            items = parse_multiselect(val)
            if items:
                transactions.append(items)

    elif basket_type == "assets" and "Asset_Interests" in df.columns:
        # Preferred asset classes over the next 5 years
        for val in df["Asset_Interests"].dropna():
            items = parse_multiselect(val)
            if items:
                transactions.append(items)

    elif basket_type == "behavior":
        # Discretized behavioral traits
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


def run_apriori(
    transactions: List[List[str]],
    min_support: float = 0.15,
    min_confidence: float = 0.5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes Apriori association rule mining using mlxtend.
    
    Mathematical Metrics:
    - Support(A) = Count(A) / N
    - Confidence(A -> B) = Support(A ∪ B) / Support(A)
    - Lift(A -> B) = Confidence(A -> B) / Support(B)
    """
    N = len(transactions)
    if N == 0:
        return pd.DataFrame(), pd.DataFrame()

    # Step 1: One-hot encode transactions using TransactionEncoder
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)

    # Step 2: Mine Frequent Itemsets with built-in apriori()
    frequent_itemsets = apriori(df_trans, min_support=min_support, use_colnames=True)
    if len(frequent_itemsets) == 0:
        return pd.DataFrame(), pd.DataFrame()

    # Format Itemsets DataFrame
    itemset_records = []
    for _, row in frequent_itemsets.iterrows():
        items_list = sorted(list(row["itemsets"]))
        sup = float(row["support"])
        itemset_records.append({
            "Itemset": ", ".join(items_list),
            "Items": items_list,
            "Support": round(sup, 4),
            "Support_Pct": f"{sup * 100:.1f}%",
            "Count": int(round(sup * N))
        })
    itemsets_df = pd.DataFrame(itemset_records).sort_values("Support", ascending=False)

    # Step 3: Mine Association Rules with built-in association_rules()
    rules_raw = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )
    if len(rules_raw) == 0:
        return itemsets_df, pd.DataFrame()

    # Format Rules DataFrame
    rule_records = []
    for _, row in rules_raw.iterrows():
        ant_list = sorted(list(row["antecedents"]))
        con_list = sorted(list(row["consequents"]))
        sup = float(row["support"])
        conf = float(row["confidence"])
        lift = float(row["lift"])

        rule_records.append({
            "Antecedent": ", ".join(ant_list),
            "Consequent": ", ".join(con_list),
            "Support": round(sup, 4),
            "Support_Pct": f"{sup * 100:.1f}%",
            "Confidence": round(conf, 4),
            "Confidence_Pct": f"{conf * 100:.1f}%",
            "Lift": round(lift, 2)
        })

    rules_df = pd.DataFrame(rule_records).sort_values(by=["Lift", "Confidence"], ascending=False)
    return itemsets_df, rules_df
