"""
Pure Python CART Decision Tree (Gini Impurity) Module
Zero external C-extension DLL dependencies (immune to Windows AppLocker).
Implements Gini Impurity: Gini(S) = 1 - SUM(p_i^2)
"""
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

def gini_impurity(series: pd.Series) -> float:
    """Computes Gini Impurity Gini(S) = 1 - SUM(p_i^2)."""
    total = len(series)
    if total == 0:
        return 0.0
    counts = series.value_counts()
    gini = 1.0
    for c in counts:
        p = c / total
        gini -= p ** 2
    return gini

def gini_gain(df: pd.DataFrame, feature: str, target: str) -> float:
    """Computes Reduction in Gini Impurity."""
    total = len(df)
    if total == 0:
        return 0.0
    base_gini = gini_impurity(df[target])
    weighted_gini = 0.0
    for _, subset in df.groupby(feature):
        p = len(subset) / total
        weighted_gini += p * gini_impurity(subset[target])
    return base_gini - weighted_gini

class CARTNode:
    def __init__(self, is_leaf: bool = False, prediction: Any = None, feature: Optional[str] = None, samples: int = 0):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature = feature
        self.samples = samples
        self.children: Dict[str, 'CARTNode'] = {}

def train_cart_tree(df: pd.DataFrame, features: List[str], target: str, max_depth: int = 4, test_size: float = 0.25) -> Dict[str, Any]:
    """
    Pure Python & Pandas CART Decision Tree using Gini Impurity.
    """
    data = df[features + [target]].dropna()
    classes = sorted(list(data[target].unique()))

    # Stratified Train/Test split in pure Python
    np.random.seed(42)
    train_indices = []
    test_indices = []
    for c in classes:
        c_idxs = data.index[data[target] == c].tolist()
        np.random.shuffle(c_idxs)
        split = int(len(c_idxs) * (1 - test_size))
        train_indices.extend(c_idxs[:split])
        test_indices.extend(c_idxs[split:])

    train_df = data.loc[train_indices]
    test_df = data.loc[test_indices]

    def build_tree(current_df, current_features, depth):
        if len(current_df) == 0:
            return CARTNode(is_leaf=True, prediction="Unknown", samples=0)
        majority = current_df[target].mode()[0]
        cur_gini = gini_impurity(current_df[target])

        if cur_gini == 0 or len(current_features) == 0 or depth >= max_depth:
            return CARTNode(is_leaf=True, prediction=majority, samples=len(current_df))

        best_feat = max(current_features, key=lambda f: gini_gain(current_df, f, target))
        if gini_gain(current_df, best_feat, target) <= 1e-4:
            return CARTNode(is_leaf=True, prediction=majority, samples=len(current_df))

        node = CARTNode(is_leaf=False, feature=best_feat, samples=len(current_df))
        rem_features = [f for f in current_features if f != best_feat]

        for val, subset in current_df.groupby(best_feat):
            node.children[str(val)] = build_tree(subset, rem_features, depth + 1)
        return node

    root = build_tree(train_df, features, depth=0)

    # Predict test samples
    def predict_one(sample):
        curr = root
        while curr and not curr.is_leaf:
            val = str(sample.get(curr.feature, ""))
            if val in curr.children:
                curr = curr.children[val]
            elif curr.children:
                curr = list(curr.children.values())[0]
            else:
                break
        return curr.prediction if curr else "Unknown"

    preds = [predict_one(row) for _, row in test_df.iterrows()]
    actuals = test_df[target].tolist()
    acc = sum(1 for a, p in zip(actuals, preds) if a == p) / len(actuals) if actuals else 0.0

    # Confusion matrix
    cm = []
    for actual in classes:
        row = []
        for predicted in classes:
            row.append(sum(1 for a, p in zip(actuals, preds) if a == actual and p == predicted))
        cm.append(row)

    # Extract rules
    rules = []
    def traverse(node, conds):
        if node.is_leaf:
            c_str = " AND ".join(conds) if conds else "ALWAYS"
            rules.append(f"IF {c_str} THEN {target} = '{node.prediction}' [n={node.samples}]")
            return
        for branch, child in node.children.items():
            traverse(child, conds + [f"({node.feature} == '{branch}')"])
    traverse(root, [])

    # Calculate Gini Gain Feature Importances
    gains = {f: gini_gain(train_df, f, target) for f in features}
    tot_gain = sum(gains.values()) or 1.0
    importances = {f: round(g / tot_gain, 4) for f, g in sorted(gains.items(), key=lambda x: x[1], reverse=True)}

    return {
        "accuracy": round(acc, 4),
        "confusion_matrix": cm,
        "labels": [str(c) for c in classes],
        "feature_importances": importances,
        "rules_text": "\n".join(rules[:12]),
        "train_samples": len(train_df),
        "test_samples": len(test_df)
    }
