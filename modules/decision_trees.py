"""
Decision Tree Mining Module (ID3 & J48 / C4.5)
==============================================
VIVA EXPLANATION GUIDE FOR PROFESSOR:
1. What this code does:
   Demonstrates decision tree induction and mathematical splitting metrics:
   - ID3 (Iterative Dichotomiser 3): Uses Shannon Entropy and Information Gain.
   - J48 / C4.5: Uses Gain Ratio (Information Gain / Split Info) to prevent bias towards attributes with many distinct values.
2. Formulas implemented:
   - Shannon Entropy: H(S) = - Σ (p_i * log2(p_i))
   - Information Gain: IG(S, A) = H(S) - Σ (|S_v| / |S|) * H(S_v)
   - Split Information: SplitInfo(S, A) = - Σ (|S_v| / |S|) * log2(|S_v| / |S|)
   - Gain Ratio: GR(S, A) = IG(S, A) / SplitInfo(S, A)
3. Python Libraries used:
   - math: For log2 calculations.
   - pandas: For group-by subsetting and value counts.
4. Output produced:
   A comparative feature gain table and recursive tree nodes for Plotly tree hierarchy visualization.
"""
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

def calculate_entropy(target_series: pd.Series) -> float:
    """
    Computes Shannon Entropy H(S):
    H(S) = - SUM(p_i * log2(p_i))
    """
    total = len(target_series)
    if total == 0:
        return 0.0
    value_counts = target_series.value_counts()
    entropy = 0.0
    for count in value_counts:
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

def calculate_information_gain(df: pd.DataFrame, feature: str, target: str) -> Tuple[float, float, Dict[str, float]]:
    """
    Computes Information Gain IG(S, A) = H(S) - SUM((|S_v|/|S|) * H(S_v))
    Returns: (info_gain, base_entropy, subset_entropies)
    """
    total_len = len(df)
    if total_len == 0:
        return 0.0, 0.0, {}
    
    base_entropy = calculate_entropy(df[target])
    weighted_entropy = 0.0
    subset_entropies = {}
    
    for val, subset in df.groupby(feature):
        sub_len = len(subset)
        sub_ent = calculate_entropy(subset[target])
        subset_entropies[str(val)] = sub_ent
        weighted_entropy += (sub_len / total_len) * sub_ent
        
    info_gain = base_entropy - weighted_entropy
    return info_gain, base_entropy, subset_entropies

def calculate_split_info(df: pd.DataFrame, feature: str) -> float:
    """
    Computes Split Information for C4.5:
    SplitInfo(S, A) = - SUM((|S_v|/|S|) * log2(|S_v|/|S|))
    """
    total_len = len(df)
    if total_len == 0:
        return 0.0
    split_info = 0.0
    for _, subset in df.groupby(feature):
        p = len(subset) / total_len
        if p > 0:
            split_info -= p * math.log2(p)
    return split_info

def calculate_gain_ratio(df: pd.DataFrame, feature: str, target: str) -> Tuple[float, float, float]:
    """
    Computes C4.5 / J48 Gain Ratio = Information Gain / Split Information
    Returns: (gain_ratio, info_gain, split_info)
    """
    info_gain, _, _ = calculate_information_gain(df, feature, target)
    split_info = calculate_split_info(df, feature)
    if split_info == 0:
        return 0.0, info_gain, split_info
    gain_ratio = info_gain / split_info
    return gain_ratio, info_gain, split_info

def get_feature_gain_table(df: pd.DataFrame, features: List[str], target: str) -> pd.DataFrame:
    """
    Generates a full educational/academic comparison table for ID3 (Info Gain) vs J48 (Gain Ratio).
    """
    base_ent = calculate_entropy(df[target])
    records = []
    for feat in features:
        if feat == target:
            continue
        # Check if continuous or categorical
        is_num = pd.api.types.is_numeric_dtype(df[feat]) and df[feat].nunique() > 5
        if is_num:
            # Simple threshold for comparison
            median_val = df[feat].median()
            temp_feat = df[feat].apply(lambda x: f"> {median_val:.1f}" if x > median_val else f"<= {median_val:.1f}")
            temp_df = df[[target]].copy()
            temp_df["temp_feat"] = temp_feat
            ig, _, _ = calculate_information_gain(temp_df, "temp_feat", target)
            si = calculate_split_info(temp_df, "temp_feat")
        else:
            ig, _, _ = calculate_information_gain(df, feat, target)
            si = calculate_split_info(df, feat)
            
        gr = ig / si if si > 0 else 0.0
        records.append({
            "Feature": feat,
            "Type": "Numeric" if is_num else "Categorical",
            "Base Entropy": round(base_ent, 4),
            "Info Gain (ID3)": round(ig, 4),
            "Split Info": round(si, 4),
            "Gain Ratio (J48/C4.5)": round(gr, 4)
        })
    res_df = pd.DataFrame(records).sort_values(by="Gain Ratio (J48/C4.5)", ascending=False).reset_index(drop=True)
    return res_df


class TreeNode:
    """Represents a node in ID3 or C4.5 Decision Tree"""
    def __init__(self, is_leaf: bool = False, prediction: Any = None, 
                 feature: Optional[str] = None, split_info: Optional[Dict] = None,
                 samples: int = 0, entropy: float = 0.0):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature = feature
        self.split_info = split_info or {}
        self.samples = samples
        self.entropy = entropy
        self.children: Dict[str, 'TreeNode'] = {}

    def to_dict(self) -> Dict:
        if self.is_leaf:
            return {
                "name": f"Predict: {self.prediction} (n={self.samples})",
                "is_leaf": True,
                "prediction": self.prediction,
                "samples": self.samples,
                "entropy": round(self.entropy, 3)
            }
        return {
            "name": f"{self.feature}?",
            "is_leaf": False,
            "feature": self.feature,
            "samples": self.samples,
            "entropy": round(self.entropy, 3),
            "children": [
                {"branch": str(branch), **child.to_dict()}
                for branch, child in self.children.items()
            ]
        }


class DecisionTreeMiner:
    """
    Interpretable Decision Tree Miner supporting ID3 and C4.5 / J48 logic.
    """
    def __init__(self, algorithm: str = "J48", max_depth: int = 4, min_samples_split: int = 4):
        self.algorithm = algorithm.upper()  # 'ID3' or 'J48' (C4.5)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: Optional[TreeNode] = None
        self.target: Optional[str] = None
        self.features: List[str] = []

    def fit(self, df: pd.DataFrame, features: List[str], target: str):
        self.target = target
        self.features = features
        self.root = self._build_tree(df, features, depth=0)
        return self

    def _build_tree(self, df: pd.DataFrame, features: List[str], depth: int) -> TreeNode:
        samples_count = len(df)
        if samples_count == 0:
            return TreeNode(is_leaf=True, prediction="Unknown", samples=0)

        majority_class = df[self.target].mode()[0]
        current_entropy = calculate_entropy(df[self.target])

        # Stopping criteria (Pure node, no features left, max depth reached, or too few samples)
        if (current_entropy == 0 or len(features) == 0 or 
            depth >= self.max_depth or samples_count < self.min_samples_split):
            return TreeNode(is_leaf=True, prediction=majority_class, 
                            samples=samples_count, entropy=current_entropy)

        # Select best splitting feature
        best_feature = None
        best_score = -1.0
        
        for feat in features:
            if self.algorithm == "ID3":
                score, _, _ = calculate_information_gain(df, feat, self.target)
            else:  # J48 / C4.5 uses Gain Ratio
                score, _, _ = calculate_gain_ratio(df, feat, self.target)

            if score > best_score:
                best_score = score
                best_feature = feat

        # If no significant gain can be made, create leaf
        if best_feature is None or best_score <= 1e-4:
            return TreeNode(is_leaf=True, prediction=majority_class, 
                            samples=samples_count, entropy=current_entropy)

        node = TreeNode(is_leaf=False, feature=best_feature, samples=samples_count, entropy=current_entropy)
        remaining_features = [f for f in features if f != best_feature]

        # Branch for each category of the selected feature
        for val, subset in df.groupby(best_feature):
            if len(subset) == 0:
                node.children[str(val)] = TreeNode(is_leaf=True, prediction=majority_class, samples=0)
            else:
                node.children[str(val)] = self._build_tree(subset, remaining_features, depth + 1)

        return node

    def predict_one(self, sample: pd.Series) -> Any:
        node = self.root
        while node and not node.is_leaf:
            val = str(sample.get(node.feature, ""))
            if val in node.children:
                node = node.children[val]
            else:
                # Default to first child or leaf prediction
                if node.children:
                    node = list(node.children.values())[0]
                else:
                    break
        return node.prediction if node else "Unknown"

    def predict(self, df: pd.DataFrame) -> List[Any]:
        return [self.predict_one(row) for _, row in df.iterrows()]

    def extract_rules(self) -> List[str]:
        """Traverses the tree to generate human-readable IF-THEN rules."""
        rules = []
        def _traverse(node: TreeNode, current_conditions: List[str]):
            if node.is_leaf:
                condition_str = " AND ".join(current_conditions) if current_conditions else "ALWAYS"
                rules.append(f"IF {condition_str} THEN {self.target} = '{node.prediction}' [Confidence: n={node.samples}]")
                return
            for branch, child in node.children.items():
                _traverse(child, current_conditions + [f"({node.feature} == '{branch}')"])

        if self.root:
            _traverse(self.root, [])
        return rules
