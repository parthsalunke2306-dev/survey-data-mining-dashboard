"""
K-Means Clustering & PCA Module for Student Behavioral Segmentation
===================================================================
VIVA EXPLANATION GUIDE FOR PROFESSOR:
1. What this code does:
   Groups university students into distinct financial behavioral personas (archetypes)
   such as "Disciplined Future Planners" or "Aspirational but Unprepared".
2. Python Libraries used:
   - scikit-learn (sklearn): Standard industry machine learning library.
   - pandas & numpy: For data frames and array operations.
3. Built-in functions used:
   - StandardScaler().fit_transform(X): Normalizes features so each variable has mean=0 and std=1.
   - KMeans(n_clusters=k).fit_predict(X): Executes K-Means clustering algorithm to assign cluster labels.
   - silhouette_score(X, labels): Calculates clustering quality metric (higher means tighter, separated clusters).
   - PCA(n_components=2).fit_transform(X): Principal Component Analysis to reduce 6 features down to 2D for visualization.
4. Why these built-ins are used:
   To use clean, standard, proven Scikit-learn algorithms rather than writing manual distance loops.
5. Output produced:
   Dictionary with cluster profiles, silhouette score, elbow evaluation, and 2D PCA scatter points.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any

# Standard Scikit-Learn imports
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def run_kmeans_segmentation(df: pd.DataFrame, n_clusters: int = 3, max_iter: int = 300) -> Dict[str, Any]:
    """
    Executes K-Means Clustering on 6 student financial metrics using Scikit-Learn.
    """
    if len(df) == 0:
        return {"n_clusters": n_clusters, "profiles": [], "elbow_curve": [], "scatter_points": []}

    # Step 1: Feature Extraction using Pandas mapping
    track_map = {"Mobile App": 3.0, "Spreadsheet": 3.0, "Pen and Paper": 2.0, "Mental Math": 1.0, "I don't track it": 0.0}
    s_track = df["Tracking_Method"].map(lambda x: track_map.get(str(x).strip(), 0.0)).values
    s_em = (df["Has_Emergency_Fund"].astype(str).str.strip().str.lower() == "yes").astype(float).values
    s_conf = pd.to_numeric(df["Financial_Confidence"], errors="coerce").fillna(3.0).values
    s_plan = pd.to_numeric(df["Wealth_Plan_Readiness"], errors="coerce").fillna(3.0).values
    res_map = {"Daily": 3.0, "Weekly": 2.0, "Monthly": 1.0, "Rarely / Never": 0.0}
    s_res = df["Research_Frequency"].map(lambda x: res_map.get(str(x).strip(), 0.0)).values
    s_sip = (df["Plan_Automated_Invest"].astype(str).str.strip().str.lower() == "yes").astype(float).values

    # Combine into feature matrix (131 rows, 6 columns)
    X_raw = np.column_stack([s_track, s_em, s_conf, s_plan, s_res, s_sip])
    N = len(X_raw)

    # Step 2: Feature Standardization using Scikit-Learn StandardScaler
    # Formula: z = (x - u) / s
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Step 3: Elbow Evaluation across k = 2 to 5
    elbow_data = []
    for k_eval in range(2, 6):
        km_eval = KMeans(n_clusters=k_eval, random_state=42, n_init=10)
        labels_eval = km_eval.fit_predict(X_scaled)
        sil_eval = float(silhouette_score(X_scaled, labels_eval))
        elbow_data.append({
            "k": k_eval,
            "inertia": round(float(km_eval.inertia_), 2),
            "silhouette": round(sil_eval, 4)
        })

    # Step 4: Fit Final KMeans model with requested k
    k = max(2, min(n_clusters, 5))
    kmeans_model = KMeans(n_clusters=k, max_iter=max_iter, random_state=42, n_init=10)
    final_labels = kmeans_model.fit_predict(X_scaled)
    current_silhouette = float(silhouette_score(X_scaled, final_labels))

    # Step 5: 2D Dimensionality Reduction using Scikit-Learn PCA
    pca_model = PCA(n_components=2, random_state=42)
    pca_coords = pca_model.fit_transform(X_scaled)
    var_explained = pca_model.explained_variance_ratio_

    # Step 6: Compute Cluster Profiles using Pandas groupby
    feature_labels = ["Tracking", "Emergency_Fund", "Confidence", "Planning", "Research", "SIP_Intention"]
    raw_df_means = pd.DataFrame(X_raw, columns=feature_labels)
    raw_df_means["Cluster"] = final_labels

    cluster_group = raw_df_means.groupby("Cluster").mean()

    profiles = []
    for c in range(k):
        sub_count = int((final_labels == c).sum())
        pct = (sub_count / N) * 100
        row_means = cluster_group.loc[c] if c in cluster_group.index else pd.Series(0, index=feature_labels)

        avg_track = float(row_means["Tracking"])
        avg_em = float(row_means["Emergency_Fund"])
        avg_conf = float(row_means["Confidence"])
        avg_plan = float(row_means["Planning"])
        avg_res = float(row_means["Research"])
        avg_sip = float(row_means["SIP_Intention"])

        # Determine empirically grounded persona label
        if avg_em >= 0.60 and avg_sip >= 0.70:
            archetype = "Disciplined Future Planners"
            desc = "High emergency reserve coverage, active research habits, and high commitment to future SIP investment automation."
            color = "#10B981"
        elif avg_sip >= 0.70 and avg_em < 0.40:
            archetype = "Aspirational but Unprepared"
            desc = "High optimism and automation intent, but vulnerable liquidity cushions and low daily budgeting discipline."
            color = "#F59E0B"
        elif avg_sip < 0.50:
            archetype = "Cautious Traditional Non-Investors"
            desc = "Conservative stance, lower research frequency, and hesitant towards market-linked automated investments."
            color = "#3B82F6"
        else:
            archetype = "Developing Financial Students"
            desc = "Moderate discipline across tracking, emerging savings habits, and growing interest in investment tools."
            color = "#8B5CF6"

        profiles.append({
            "Cluster_ID": c,
            "Cluster_Name": f"Cluster {c+1}: {archetype}",
            "Archetype": archetype,
            "Description": desc,
            "Color": color,
            "Count": sub_count,
            "Percentage": f"{pct:.1f}%",
            "Avg_Tracking": round(avg_track, 2),
            "Emergency_Fund_Rate": f"{avg_em * 100:.1f}%",
            "Avg_Confidence": round(avg_conf, 2),
            "Avg_Planning": round(avg_plan, 2),
            "Avg_Research": round(avg_res, 2),
            "SIP_Intention_Rate": f"{avg_sip * 100:.1f}%"
        })

    # Step 7: Build Scatter Points for Plotly visualization
    scatter_points = []
    for i in range(N):
        scatter_points.append({
            "x": round(float(pca_coords[i, 0]), 3),
            "y": round(float(pca_coords[i, 1]), 3),
            "Cluster": f"Cluster {final_labels[i]+1}",
            "ID": str(df.iloc[i].get("Respondent_ID", f"RESP_{i+1:03d}")),
            "Year": str(df.iloc[i].get("Academic_Year", "")),
            "Major": str(df.iloc[i].get("Stream_Major", ""))
        })

    return {
        "n_clusters": k,
        "silhouette_score": round(current_silhouette, 4),
        "elbow_curve": elbow_data,
        "profiles": profiles,
        "scatter_points": scatter_points,
        "pca_variance_ratio": [round(float(v), 4) for v in var_explained]
    }
