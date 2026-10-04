"""
CART Decision Tree (Gini Impurity) Module
==========================================
VIVA EXPLANATION GUIDE FOR PROFESSOR:
1. What this code does:
   Builds a Classification and Regression Tree (CART) to predict student emergency fund status.
   Splits branches using the Gini Impurity metric: Gini(S) = 1 - Σ(p_i^2).
2. Python Libraries used:
   - scikit-learn (sklearn): Standard machine learning library.
   - pandas & numpy: For tabular data encoding.
3. Built-in functions used:
   - sklearn.tree.DecisionTreeClassifier(criterion="gini"): Fits the CART tree model.
   - sklearn.tree.export_text(): Generates readable text-based IF-THEN rules directly from the tree.
   - sklearn.model_selection.train_test_split(): Stratified train/test splitting.
   - sklearn.metrics.accuracy_score(), confusion_matrix(): Standard classification evaluation.
4. Why these built-ins are used:
   Replaces manual recursive node-splitting loops with Scikit-learn's optimized tree builder.
5. Output produced:
   Dictionary with model accuracy, confusion matrix, labels, feature importances, and rules text.
"""

import pandas as pd
from typing import Dict, List, Any

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, confusion_matrix


def train_cart_tree(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    max_depth: int = 4,
    test_size: float = 0.25
) -> Dict[str, Any]:
    """
    Trains and evaluates a CART Decision Tree using Scikit-Learn.
    """
    data = df[features + [target]].dropna()
    if len(data) == 0:
        return {}

    # One-hot encode categorical features for Scikit-Learn compatibility
    X = pd.get_dummies(data[features], drop_first=True, dtype=float)
    y = data[target].astype(str)
    classes = sorted(list(y.unique()))

    # Train / Test Split using Scikit-Learn
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Fit CART Decision Tree with Gini Impurity
    clf = DecisionTreeClassifier(criterion="gini", max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    # Evaluation metrics
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds, labels=classes)

    # Built-in Feature Importances
    importances = {
        col: round(float(imp), 4)
        for col, imp in sorted(zip(X.columns, clf.feature_importances_), key=lambda x: x[1], reverse=True)
    }

    # Extract human-readable decision rules using Scikit-Learn export_text
    rules_text = export_text(clf, feature_names=list(X.columns))

    return {
        "accuracy": round(float(acc), 4),
        "confusion_matrix": cm.tolist(),
        "labels": [str(c) for c in classes],
        "feature_importances": importances,
        "rules_text": rules_text[:1000],
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }
