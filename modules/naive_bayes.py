"""
Pure Python Naive Bayes Classifier Module
Zero external C-extension DLL dependencies (immune to Windows AppLocker).
Implements Gaussian & Categorical Naive Bayes using Bayes' Theorem:
P(C|X) = (P(C) * PROD(P(x_i|C))) / P(X)
"""
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Any

def run_naive_bayes_classification(df: pd.DataFrame, target: str = "Has_Emergency_Fund") -> Dict[str, Any]:
    """
    Pure Python & NumPy implementation of Naive Bayes with Gaussian continuous likelihoods
    and Laplace-smoothed categorical likelihoods.
    """
    num_features = [
        "Peer_Pressure_Spend", "Stress_Spend", "Financial_Confidence",
        "Lifestyle_Upgrade_Spend", "Wealth_Plan_Readiness"
    ]
    num_features = [c for c in num_features if c in df.columns]

    cat_features = ["Monthly_Budget", "Tracking_Method", "Research_Frequency"]
    cat_features = [c for c in cat_features if c in df.columns]

    all_features = num_features + cat_features

    # Filter data
    data = df[all_features + [target]].dropna()
    N = len(data)

    # Train/Test Split (75% / 25%) in pure Python
    np.random.seed(42)
    classes = sorted(list(data[target].unique()))
    train_indices = []
    test_indices = []

    for c in classes:
        c_idxs = data.index[data[target] == c].tolist()
        np.random.shuffle(c_idxs)
        split = int(len(c_idxs) * 0.75)
        train_indices.extend(c_idxs[:split])
        test_indices.extend(c_idxs[split:])

    train_df = data.loc[train_indices]
    test_df = data.loc[test_indices]

    # 1. Class Priors P(C)
    priors = {}
    for c in classes:
        priors[c] = len(train_df[train_df[target] == c]) / len(train_df)

    # 2. Gaussian Parameters for Numerical Features (Mean & Variance per class)
    gaussian_params = {}
    for f in num_features:
        gaussian_params[f] = {}
        for c in classes:
            subset = train_df[train_df[target] == c][f].astype(float)
            mean = float(subset.mean()) if len(subset) > 0 else 3.0
            var = float(subset.var()) if len(subset) > 1 and subset.var() > 1e-4 else 1.0
            gaussian_params[f][c] = (mean, var)

    def gaussian_pdf(x, mean, var):
        denom = math.sqrt(2 * math.pi * var)
        num = math.exp(-((x - mean) ** 2) / (2 * var))
        return max(num / denom, 1e-6)

    # 3. Predict on Test Set
    preds = []
    actuals = test_df[target].tolist()

    for _, row in test_df.iterrows():
        posteriors = {}
        for c in classes:
            log_prob = math.log(priors[c])
            # Numeric likelihoods
            for f in num_features:
                val = float(row[f])
                mean, var = gaussian_params[f][c]
                prob = gaussian_pdf(val, mean, var)
                log_prob += math.log(prob)
            # Categorical likelihoods with Laplace smoothing
            for f in cat_features:
                val = str(row[f])
                subset = train_df[train_df[target] == c]
                matching = len(subset[subset[f] == val])
                total_c = len(subset)
                num_distinct = max(train_df[f].nunique(), 1)
                cond_prob = (matching + 1) / (total_c + num_distinct)
                log_prob += math.log(cond_prob)
            posteriors[c] = log_prob
        # Argmax
        pred_class = max(posteriors.keys(), key=lambda k: posteriors[k])
        preds.append(pred_class)

    # 4. Accuracy & Confusion Matrix
    correct = sum(1 for a, p in zip(actuals, preds) if a == p)
    acc = correct / len(actuals) if actuals else 0.0

    cm = []
    for actual in classes:
        row = []
        for predicted in classes:
            row.append(sum(1 for a, p in zip(actuals, preds) if a == actual and p == predicted))
        cm.append(row)

    report = {}
    for i, lbl in enumerate(classes):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(len(classes)) if r != i)
        fn = sum(cm[i][c] for c in range(len(classes)) if c != i)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        report[lbl] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1-score": round(f1, 3),
            "support": sum(cm[i])
        }

    return {
        "accuracy": round(acc, 4),
        "class_priors": {str(c): round(p, 4) for c, p in priors.items()},
        "confusion_matrix": cm,
        "labels": [str(c) for c in classes],
        "classification_report": report,
        "n_train": len(train_df),
        "n_test": len(test_df)
    }
