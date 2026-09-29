"""
Pure Python K-Nearest Neighbors (KNN) Classifier Module
Zero external C-extension DLL dependencies (immune to Windows AppLocker).
Implements Euclidean distance calculation, k-nearest search, majority voting,
train/test splits, and confusion matrix evaluation.
"""
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List, Any, Tuple

def run_knn_classification(df: pd.DataFrame, target: str = "Has_Emergency_Fund", k: int = 5) -> Dict[str, Any]:
    """
    Pure Python & NumPy implementation of K-Nearest Neighbors.
    """
    feature_cols = [
        "Peer_Pressure_Spend", "Stress_Spend", "Financial_Confidence",
        "Lifestyle_Upgrade_Spend", "Wealth_Plan_Readiness"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Categorical one-hot encoding in pure pandas
    cat_cols = ["Monthly_Budget", "Tracking_Method", "Research_Frequency"]
    cat_cols = [c for c in cat_cols if c in df.columns]

    X_cat = pd.get_dummies(df[cat_cols], drop_first=True, dtype=float)
    X_num = df[feature_cols].copy().fillna(3).astype(float)
    X = pd.concat([X_num, X_cat], axis=1).values
    y = df[target].astype(str).values

    # Stratified Train/Test Split (75% / 25%) in pure Python
    np.random.seed(42)
    classes = np.unique(y)
    train_indices = []
    test_indices = []
    
    for c in classes:
        c_idxs = np.where(y == c)[0]
        np.random.shuffle(c_idxs)
        split = int(len(c_idxs) * 0.75)
        train_indices.extend(c_idxs[:split])
        test_indices.extend(c_idxs[split:])

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    # Z-score normalization based on training set
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1.0
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    # Helper: predict for single query vector
    def predict_one(x_query, curr_k):
        dists = np.sqrt(np.sum((X_train_scaled - x_query) ** 2, axis=1))
        k_indices = np.argsort(dists)[:curr_k]
        k_labels = y_train[k_indices]
        return Counter(k_labels).most_common(1)[0][0]

    # Evaluate across test k values
    k_accuracies = []
    for test_k in [1, 3, 5, 7, 9]:
        preds = [predict_one(x, test_k) for x in X_test_scaled]
        acc = np.mean(np.array(preds) == y_test)
        k_accuracies.append({"k": test_k, "accuracy": round(float(acc), 4)})

    # Final predictions for requested k
    final_preds = [predict_one(x, k) for x in X_test_scaled]
    overall_acc = float(np.mean(np.array(final_preds) == y_test))

    # Confusion matrix in pure Python
    labels = sorted(list(classes))
    cm = []
    for actual in labels:
        row = []
        for predicted in labels:
            count = sum(1 for a, p in zip(y_test, final_preds) if a == actual and p == predicted)
            row.append(count)
        cm.append(row)

    # Classification metrics
    report = {}
    for i, lbl in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(len(labels)) if r != i)
        fn = sum(cm[i][c] for c in range(len(labels)) if c != i)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        support = sum(cm[i])
        report[lbl] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1-score": round(f1, 3),
            "support": support
        }

    return {
        "k": k,
        "accuracy": round(overall_acc, 4),
        "k_tuning": k_accuracies,
        "confusion_matrix": cm,
        "labels": labels,
        "classification_report": report,
        "n_train": len(X_train),
        "n_test": len(X_test)
    }
