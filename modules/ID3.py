"""
ID3 Algorithm (Iterative Dichotomiser 3)
Builds multi-way decision trees using Shannon's Entropy and Information Gain.
"""
import math
import pandas as pd
from typing import Dict, List, Any, Optional

def entropy(series: pd.Series) -> float:
    """Computes Shannon Entropy H(S) = - SUM(p_i * log2(p_i))."""
    total = len(series)
    if total == 0:
        return 0.0
    counts = series.value_counts()
    ent = 0.0
    for c in counts:
        p = c / total
        if p > 0:
            ent -= p * math.log2(p)
    return ent

def information_gain(df: pd.DataFrame, feature: str, target: str) -> float:
    """Computes Information Gain IG(S, A) = H(S) - SUM((|S_v|/|S|) * H(S_v))."""
    total = len(df)
    if total == 0:
        return 0.0
    base_ent = entropy(df[target])
    weighted_ent = 0.0
    for val, subset in df.groupby(feature):
        p = len(subset) / total
        weighted_ent += p * entropy(subset[target])
    return base_ent - weighted_ent

class ID3Node:
    def __init__(self, is_leaf: bool = False, prediction: Any = None, feature: Optional[str] = None, samples: int = 0):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature = feature
        self.samples = samples
        self.children: Dict[str, 'ID3Node'] = {}

class ID3Classifier:
    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth
        self.root: Optional[ID3Node] = None
        self.target: Optional[str] = None
        self.features: List[str] = []

    def fit(self, df: pd.DataFrame, features: List[str], target: str):
        self.target = target
        self.features = features
        self.root = self._build(df, features, depth=0)
        return self

    def _build(self, df: pd.DataFrame, features: List[str], depth: int) -> ID3Node:
        if len(df) == 0:
            return ID3Node(is_leaf=True, prediction="Unknown", samples=0)
        
        majority = df[self.target].mode()[0]
        cur_ent = entropy(df[self.target])

        if cur_ent == 0 or len(features) == 0 or depth >= self.max_depth:
            return ID3Node(is_leaf=True, prediction=majority, samples=len(df))

        best_feat = max(features, key=lambda f: information_gain(df, f, self.target))
        if information_gain(df, best_feat, self.target) <= 1e-4:
            return ID3Node(is_leaf=True, prediction=majority, samples=len(df))

        node = ID3Node(is_leaf=False, feature=best_feat, samples=len(df))
        rem_features = [f for f in features if f != best_feat]

        for val, subset in df.groupby(best_feat):
            node.children[str(val)] = self._build(subset, rem_features, depth + 1)
        return node

    def predict_one(self, sample: pd.Series) -> Any:
        curr = self.root
        while curr and not curr.is_leaf:
            val = str(sample.get(curr.feature, ""))
            if val in curr.children:
                curr = curr.children[val]
            else:
                if curr.children:
                    curr = list(curr.children.values())[0]
                else:
                    break
        return curr.prediction if curr else "Unknown"

    def predict(self, df: pd.DataFrame) -> List[Any]:
        return [self.predict_one(row) for _, row in df.iterrows()]
