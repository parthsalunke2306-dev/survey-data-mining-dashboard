"""
K-Nearest Neighbors (KNN) Classifier Module
===========================================
VIVA EXPLANATION GUIDE FOR PROFESSOR:
1. What this code does:
   Classifies students (e.g. predicting whether they have an Emergency Fund)
   based on the financial habits of their k-nearest peers in feature space.
2. Python Libraries used:
   - scikit-learn (sklearn): Standard machine learning library.
   - pandas & numpy: For data preprocessing.
3. Built-in functions used:
   - sklearn.model_selection.train_test_split(): Splits data into 75% training and 25% testing sets.
   - sklearn.preprocessing.StandardScaler(): Normalizes features so Euclidean distances are not biased by scale.
   - sklearn.neighbors.KNeighborsClassifier(): Fits the KNN model using majority voting of k nearest neighbors.
   - sklearn.metrics.accuracy_score(), confusion_matrix(), classification_report(): Standard classification evaluation metrics.
4. Why these built-ins are used:
   Avoids manual Euclidean distance calculation loops and provides academically standard model evaluation.
5. Output produced:
   Dictionary containing overall accuracy, k-accuracy curve, confusion matrix, and classification report.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def run_knn_classification(df: pd.DataFrame, target: str = "Has_Emergency_Fund", k: int = 5) -> Dict[str, Any]:
    """
    Trains and evaluates a K-Nearest Neighbors Classifier using Scikit-Learn.
    """
    if len(df) == 0 or target not in df.columns:
        return {}

    feature_cols = [
        "Peer_Pressure_Spend", "Stress_Spend", "Financial_Confidence",
        "Lifestyle_Upgrade_Spend", "Wealth_Plan_Readiness"
    ]
    num_features = [c for c in feature_cols if c in df.columns]

    cat_cols = ["Monthly_Budget", "Tracking_Method", "Research_Frequency"]
    cat_features = [c for c in cat_cols if c in df.columns]

    # One-hot encode categorical features using Pandas get_dummies
    X_cat = pd.get_dummies(df[cat_features], drop_first=True, dtype=float)
    X_num = df[num_features].copy().fillna(3).astype(float)
    X = pd.concat([X_num, X_cat], axis=1)
    y = df[target].astype(str)

    # Train / Test Split using Scikit-Learn
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Feature Scaling using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Evaluate across test k values for hyperparameter comparison
    k_accuracies = []
    for test_k in [1, 3, 5, 7, 9]:
        knn_eval = KNeighborsClassifier(n_neighbors=test_k)
        knn_eval.fit(X_train_scaled, y_train)
        preds_eval = knn_eval.predict(X_test_scaled)
        acc_eval = accuracy_score(y_test, preds_eval)
        k_accuracies.append({"k": test_k, "accuracy": round(float(acc_eval), 4)})

    # Fit final KNN model with requested k
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    final_preds = knn.predict(X_test_scaled)

    # Performance Evaluation using built-in Scikit-Learn metrics
    overall_acc = float(accuracy_score(y_test, final_preds))
    cm = confusion_matrix(y_test, final_preds)
    report = classification_report(y_test, final_preds, output_dict=True, zero_division=0)

    classes = sorted(list(np.unique(y)))

    return {
        "k": k,
        "overall_accuracy": round(overall_acc, 4),
        "k_accuracies": k_accuracies,
        "classes": classes,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "feature_names": list(X.columns)
    }
