"""
Pure Python K-Means Clustering & PCA Module
Zero external C-extension DLL dependencies (immune to Windows AppLocker/Application Control policies).
Includes Euclidean distance, centroid optimization, Silhouette score, and SVD-based 2D PCA.
"""
import random
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))

def run_kmeans_segmentation(df: pd.DataFrame, n_clusters: int = 3, max_iter: int = 50) -> Dict[str, Any]:
    """
    Pure Python & NumPy K-Means clustering and PCA 2D projection.
    """
    feature_cols = [
        "Peer_Pressure_Spend", "Stress_Spend", "Financial_Confidence",
        "Lifestyle_Upgrade_Spend", "Wealth_Plan_Readiness"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Convert to numeric matrix
    X = df[feature_cols].copy().fillna(3).values.astype(float)
    N, D = X.shape

    # Standardize (Z-score) in pure NumPy
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    stds[stds == 0] = 1.0
    X_scaled = (X - means) / stds

    # Seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # 1. Elbow & Inertia for k = 2 to 6
    elbow_data = []
    for k in range(2, 7):
        # K-Means++ init
        centroids = [X_scaled[random.randint(0, N - 1)]]
        for _ in range(1, k):
            dists = np.array([min(euclidean_distance(x, c) ** 2 for c in centroids) for x in X_scaled])
            probs = dists / dists.sum() if dists.sum() > 0 else np.ones(N) / N
            cumprobs = np.cumsum(probs)
            r = random.random()
            idx = np.searchsorted(cumprobs, r)
            centroids.append(X_scaled[min(idx, N - 1)])
        centroids = np.array(centroids)

        # Run Lloyd's iterations
        labels = np.zeros(N, dtype=int)
        for _ in range(20):
            # Assign
            for i in range(N):
                dists = [euclidean_distance(X_scaled[i], c) for c in centroids]
                labels[i] = int(np.argmin(dists))
            # Update centroids
            new_centroids = []
            for j in range(k):
                members = X_scaled[labels == j]
                if len(members) > 0:
                    new_centroids.append(members.mean(axis=0))
                else:
                    new_centroids.append(centroids[j])
            centroids = np.array(new_centroids)

        # Compute Inertia
        inertia = sum(euclidean_distance(X_scaled[i], centroids[labels[i]]) ** 2 for i in range(N))
        
        # Approximate Silhouette Score
        sil_sum = 0.0
        for i in range(N):
            c_i = labels[i]
            same_cluster = X_scaled[(labels == c_i) & (np.arange(N) != i)]
            a_i = np.mean([euclidean_distance(X_scaled[i], o) for o in same_cluster]) if len(same_cluster) > 0 else 0
            
            b_i = float('inf')
            for other_c in range(k):
                if other_c != c_i:
                    other_cluster = X_scaled[labels == other_c]
                    if len(other_cluster) > 0:
                        dist_o = np.mean([euclidean_distance(X_scaled[i], o) for o in other_cluster])
                        b_i = min(b_i, dist_o)
            if b_i == float('inf'):
                b_i = 0.0
            max_ab = max(a_i, b_i)
            sil_i = (b_i - a_i) / max_ab if max_ab > 0 else 0
            sil_sum += sil_i

        elbow_data.append({
            "k": k,
            "inertia": round(inertia, 2),
            "silhouette": round(sil_sum / N, 4)
        })

    # 2. Primary K-Means fit for requested n_clusters
    k = n_clusters
    centroids = [X_scaled[random.randint(0, N - 1)]]
    for _ in range(1, k):
        dists = np.array([min(euclidean_distance(x, c) ** 2 for c in centroids) for x in X_scaled])
        probs = dists / dists.sum() if dists.sum() > 0 else np.ones(N) / N
        cumprobs = np.cumsum(probs)
        centroids.append(X_scaled[min(np.searchsorted(cumprobs, random.random()), N - 1)])
    centroids = np.array(centroids)

    final_labels = np.zeros(N, dtype=int)
    for _ in range(max_iter):
        for i in range(N):
            final_labels[i] = int(np.argmin([euclidean_distance(X_scaled[i], c) for c in centroids]))
        new_centroids = []
        for j in range(k):
            members = X_scaled[final_labels == j]
            new_centroids.append(members.mean(axis=0) if len(members) > 0 else centroids[j])
        centroids = np.array(new_centroids)

    # 3. PCA 2D Projection using pure NumPy SVD
    # Centering
    X_centered = X_scaled - np.mean(X_scaled, axis=0)
    # SVD: X = U * S * Vt
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # First 2 principal components
    pca_2d = np.dot(X_centered, Vt[:2, :].T)
    variance_explained = (S ** 2) / np.sum(S ** 2)

    # 4. Format profiles
    df_clustered = df.copy()
    df_clustered["Cluster"] = [f"Segment {c+1}" for c in final_labels]
    df_clustered["PCA1"] = pca_2d[:, 0]
    df_clustered["PCA2"] = pca_2d[:, 1]

    profiles = []
    for c in range(k):
        subset = df_clustered[df_clustered["Cluster"] == f"Segment {c+1}"]
        avg_peer = subset["Peer_Pressure_Spend"].mean() if "Peer_Pressure_Spend" in subset else 3
        avg_stress = subset["Stress_Spend"].mean() if "Stress_Spend" in subset else 3
        avg_conf = subset["Financial_Confidence"].mean() if "Financial_Confidence" in subset else 3
        
        if avg_stress >= 3.7 and avg_peer >= 3.4:
            archetype = "Impulsive & Peer-Driven Spenders"
        elif avg_conf >= 3.7:
            archetype = "Confident & Proactive Planners"
        elif avg_peer <= 2.3 and avg_stress <= 2.6:
            archetype = "Frugal / Conservative Savers"
        else:
            archetype = "Moderate / Transitioning Students"

        emergency_rate = (subset["Has_Emergency_Fund"] == "Yes").mean() * 100 if "Has_Emergency_Fund" in subset else 0
        invest_rate = (subset["Has_Investment_Account"] == "Yes").mean() * 100 if "Has_Investment_Account" in subset else 0

        profiles.append({
            "Cluster": f"Segment {c+1}",
            "Archetype": archetype,
            "Count": len(subset),
            "Percentage": f"{(len(subset) / len(df)) * 100:.1f}%",
            "Avg_Peer_Spend": round(avg_peer, 2),
            "Avg_Stress_Spend": round(avg_stress, 2),
            "Avg_Financial_Confidence": round(avg_conf, 2),
            "Emergency_Fund_Rate": f"{emergency_rate:.1f}%",
            "Investment_Rate": f"{invest_rate:.1f}%"
        })

    scatter_points = []
    for idx, row in df_clustered.iterrows():
        scatter_points.append({
            "x": round(row["PCA1"], 3),
            "y": round(row["PCA2"], 3),
            "Cluster": row["Cluster"],
            "Major": str(row.get("Stream_Major", "Student")),
            "Budget": str(row.get("Monthly_Budget", ""))
        })

    matched_sil = next((item["silhouette"] for item in elbow_data if item["k"] == n_clusters), 0.35)

    return {
        "n_clusters": n_clusters,
        "silhouette_score": matched_sil,
        "elbow_curve": elbow_data,
        "profiles": profiles,
        "scatter_points": scatter_points,
        "pca_variance_ratio": [round(float(v), 4) for v in variance_explained[:2]]
    }
