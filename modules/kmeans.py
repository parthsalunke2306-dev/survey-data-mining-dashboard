"""
Pure Python & NumPy K-Means Clustering & PCA Module
Specialized for Student Financial Behaviour Segmentation.
Zero external C-extension DLL dependencies (immune to Windows AppLocker and serverless Lambda restrictions).
"""
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Any

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))

def run_kmeans_segmentation(df: pd.DataFrame, n_clusters: int = 3, max_iter: int = 50) -> Dict[str, Any]:
    """
    K-Means clustering across 6 core financial indicators:
    1. Tracking behavior (0-3 scale)
    2. Emergency fund availability (0 or 1)
    3. Financial confidence (1-5 Likert)
    4. Wealth plan readiness (1-5 Likert)
    5. Research frequency (0-3 scale)
    6. Investment automation intention (0 or 1)
    """
    if len(df) == 0:
        return {"n_clusters": n_clusters, "profiles": [], "elbow_curve": [], "scatter_points": []}

    # 1. Feature Extraction & Encoding
    track_map = {"Mobile App": 3.0, "Spreadsheet": 3.0, "Pen and Paper": 2.0, "Mental Math": 1.0, "I don't track it": 0.0}
    s_track = df["Tracking_Method"].map(lambda x: track_map.get(str(x).strip(), 0.0)).values.astype(float)
    s_em = df["Has_Emergency_Fund"].apply(lambda x: 1.0 if str(x).strip().lower() == "yes" else 0.0).values
    s_conf = pd.to_numeric(df["Financial_Confidence"], errors="coerce").fillna(3.0).values.astype(float)
    s_plan = pd.to_numeric(df["Wealth_Plan_Readiness"], errors="coerce").fillna(3.0).values.astype(float)
    res_map = {"Daily": 3.0, "Weekly": 2.0, "Monthly": 1.0, "Rarely / Never": 0.0}
    s_res = df["Research_Frequency"].map(lambda x: res_map.get(str(x).strip(), 0.0)).values.astype(float)
    s_sip = df["Plan_Automated_Invest"].apply(lambda x: 1.0 if str(x).strip().lower() == "yes" else 0.0).values

    X_raw = np.column_stack([s_track, s_em, s_conf, s_plan, s_res, s_sip])
    N, D = X_raw.shape

    # Standardize (Z-score)
    means = np.mean(X_raw, axis=0)
    stds = np.std(X_raw, axis=0)
    stds[stds == 0] = 1.0
    X_scaled = (X_raw - means) / stds

    np.random.seed(42)
    random.seed(42)

    # 2. Elbow & Inertia for k = 2 to 5
    elbow_data = []
    for k_eval in range(2, 6):
        # K-Means++ initialization
        centroids = [X_scaled[random.randint(0, N - 1)]]
        for _ in range(1, k_eval):
            dists = np.array([min(euclidean_distance(x, c) ** 2 for c in centroids) for x in X_scaled])
            probs = dists / dists.sum() if dists.sum() > 0 else np.ones(N) / N
            cumprobs = np.cumsum(probs)
            idx = np.searchsorted(cumprobs, random.random())
            centroids.append(X_scaled[min(idx, N - 1)])
        centroids = np.array(centroids)

        labels = np.zeros(N, dtype=int)
        for _ in range(25):
            dists = np.linalg.norm(X_scaled[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(dists, axis=1)
            new_centroids = np.array([
                X_scaled[labels == j].mean(axis=0) if (labels == j).sum() > 0 else centroids[j]
                for j in range(k_eval)
            ])
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        # Inertia
        inertia = sum(euclidean_distance(X_scaled[i], centroids[labels[i]]) ** 2 for i in range(N))
        
        # Approximate Silhouette Score
        sil_sum = 0.0
        for i in range(N):
            c_i = labels[i]
            same_cluster = X_scaled[(labels == c_i) & (np.arange(N) != i)]
            a_i = np.mean([euclidean_distance(X_scaled[i], o) for o in same_cluster]) if len(same_cluster) > 0 else 0
            
            b_i = float('inf')
            for other_c in range(k_eval):
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
            "k": k_eval,
            "inertia": round(float(inertia), 2),
            "silhouette": round(float(sil_sum / N), 4)
        })

    # 3. Fit requested k
    k = max(2, min(n_clusters, 5))
    centroids = [X_scaled[random.randint(0, N - 1)]]
    for _ in range(1, k):
        dists = np.array([min(euclidean_distance(x, c) ** 2 for c in centroids) for x in X_scaled])
        probs = dists / dists.sum() if dists.sum() > 0 else np.ones(N) / N
        cumprobs = np.cumsum(probs)
        idx = np.searchsorted(cumprobs, random.random())
        centroids.append(X_scaled[min(idx, N - 1)])
    centroids = np.array(centroids)

    final_labels = np.zeros(N, dtype=int)
    for _ in range(max_iter):
        dists = np.linalg.norm(X_scaled[:, np.newaxis] - centroids, axis=2)
        final_labels = np.argmin(dists, axis=1)
        new_centroids = np.array([
            X_scaled[final_labels == j].mean(axis=0) if (final_labels == j).sum() > 0 else centroids[j]
            for j in range(k)
        ])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    # 4. Pure NumPy PCA 2D Projection for visualization
    X_centered = X_scaled - np.mean(X_scaled, axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    pca_2d = np.dot(X_centered, Vt[:2, :].T)
    variance_explained = (S ** 2) / np.sum(S ** 2)

    # 5. Build Cluster Profiles with actual data-driven descriptive labels
    profiles = []
    feature_labels = ["Tracking", "Emergency_Fund", "Confidence", "Planning", "Research", "SIP_Intention"]
    raw_df_means = pd.DataFrame(X_raw, columns=feature_labels)
    raw_df_means["Cluster"] = final_labels

    cluster_group = raw_df_means.groupby("Cluster").mean()

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

        # Determine empirically grounded label
        if avg_em >= 0.65 and avg_sip >= 0.7:
            archetype = "Disciplined Future Planners"
            desc = "High emergency reserve coverage, active research habits, and high commitment to future SIP investment automation."
            color = "#047857"
        elif avg_sip >= 0.7 and avg_em < 0.4:
            archetype = "Aspirational but Unprepared"
            desc = "High optimism and automation intent, but vulnerable liquidity cushions and low daily budgeting discipline."
            color = "#334155"
        elif avg_sip < 0.5:
            archetype = "Cautious Traditional Non-Investors"
            desc = "Conservative stance, lower research frequency, and hesitant towards market-linked automated investments."
            color = "#475569"
        else:
            archetype = "Developing Financial Students"
            desc = "Moderate discipline across tracking, emerging savings habits, and growing interest in investment tools."
            color = "#059669"

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

    scatter_points = []
    for i in range(N):
        scatter_points.append({
            "x": round(float(pca_2d[i, 0]), 3),
            "y": round(float(pca_2d[i, 1]), 3),
            "Cluster": f"Cluster {final_labels[i]+1}",
            "ID": str(df.iloc[i].get("Respondent_ID", f"RESP_{i+1:03d}")),
            "Year": str(df.iloc[i].get("Academic_Year", "")),
            "Major": str(df.iloc[i].get("Stream_Major", ""))
        })

    matched_sil = next((item["silhouette"] for item in elbow_data if item["k"] == k), 0.35)

    return {
        "n_clusters": k,
        "silhouette_score": matched_sil,
        "elbow_curve": elbow_data,
        "profiles": profiles,
        "scatter_points": scatter_points,
        "pca_variance_ratio": [round(float(v), 4) for v in variance_explained[:2]]
    }
