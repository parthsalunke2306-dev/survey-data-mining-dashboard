"""
Naive Bayes Classifier Module
==============================
VIVA EXPLANATION GUIDE FOR PROFESSOR:
1. What this code does:
   Predicts student outcomes (e.g., Has Emergency Fund) using Bayes' Theorem:
   P(Class | Features) = P(Class) * P(Features | Class) / P(Features)
2. Python Libraries used:
   - scikit-learn (sklearn): Standard machine learning library.
   - pandas & numpy: For data handling.
3. Built-in functions used:
   - sklearn.naive_bayes.GaussianNB(): Fits a Gaussian Naive Bayes model assuming normal distribution for features.
   - sklearn.model_selection.train_test_split(): Stratified train/test splitting.
   - sklearn.metrics.accuracy_score(), confusion_matrix(), classification_report(): Standard performance metrics.
4. Why these built-ins are used:
   Replaces manual Gaussian probability density formula calculation with Scikit-learn's optimized classifier.
5. Output produced:
   Dictionary containing overall accuracy, class priors, confusion matrix, and classification report.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def run_naive_bayes_classification(df: pd.DataFrame, target: str = "Has_Emergency_Fund") -> Dict[str, Any]:
    """
    Trains and evaluates a Gaussian Naive Bayes Classifier using Scikit-Learn.
    """
    if len(df) == 0 or target not in df.columns:
        return {}

    num_features = [
        "Peer_Pressure_Spend", "Stress_Spend", "Financial_Confidence",
        "Lifestyle_Upgrade_Spend", "Wealth_Plan_Readiness"
    ]
    num_features = [c for c in num_features if c in df.columns]

    cat_features = ["Monthly_Budget", "Tracking_Method", "Research_Frequency"]
    cat_features = [c for c in cat_features if c in df.columns]

    # One-hot encode categorical features using Pandas get_dummies
    X_cat = pd.get_dummies(df[cat_features], drop_first=True, dtype=float)
    X_num = df[num_features].copy().fillna(3).astype(float)
    X = pd.concat([X_num, X_cat], axis=1)
    y = df[target].astype(str)

    # Train / Test Split using Scikit-Learn
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Fit Gaussian Naive Bayes model
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    preds = nb.predict(X_test)

    # Performance Evaluation using built-in Scikit-Learn metrics
    overall_acc = float(accuracy_score(y_test, preds))
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)

    classes = sorted(list(np.unique(y)))
    priors = {cls: round(float(prior), 4) for cls, prior in zip(nb.classes_, nb.class_prior_)}

    return {
        "overall_accuracy": round(overall_acc, 4),
        "classes": classes,
        "priors": priors,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "feature_names": list(X.columns)
    }
