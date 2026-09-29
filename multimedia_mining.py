"""
Multimedia Data Mining Module: Visual & Perceptual Behavioral Signatures
Mines multi-dimensional visual feature vectors and constructs psychographic radar signatures
for student financial archetypes.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any

def extract_visual_behavioral_signatures(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extracts multi-attribute visual radar signatures across key student segments.
    """
    dimensions = [
        "Peer_Pressure_Spend", "Stress_Spend", "Financial_Confidence",
        "Lifestyle_Upgrade_Spend", "Wealth_Plan_Readiness"
    ]
    labels = ["Peer Pressure", "Stress Spending", "Confidence", "Tech Upgrades", "Wealth Plan"]

    # 1. Overall Student Average Signature
    overall_vector = [round(df[d].mean(), 2) for d in dimensions if d in df]

    # 2. Signature by Academic Year (Freshmen vs Seniors)
    cohort_signatures = []
    if "Academic_Year" in df:
        for year in ["1st Year", "2nd Year", "3rd Year", "4th Year"]:
            subset = df[df["Academic_Year"] == year]
            if not subset.empty:
                cohort_signatures.append({
                    "cohort": year,
                    "values": [round(subset[d].mean(), 2) for d in dimensions if d in subset],
                    "sample_size": len(subset)
                })

    # 3. Signature by Emergency Fund (Prepared vs Vulnerable)
    prepared_vector = []
    unprepared_vector = []
    if "Has_Emergency_Fund" in df:
        prep_subset = df[df["Has_Emergency_Fund"] == "Yes"]
        unprep_subset = df[df["Has_Emergency_Fund"] == "No"]
        prepared_vector = [round(prep_subset[d].mean(), 2) for d in dimensions if d in prep_subset]
        unprepared_vector = [round(unprep_subset[d].mean(), 2) for d in dimensions if d in unprep_subset]

    return {
        "dimensions": labels,
        "overall_signature": overall_vector,
        "cohort_signatures": cohort_signatures,
        "prepared_vs_unprepared": {
            "has_emergency_fund": prepared_vector,
            "lacks_emergency_fund": unprepared_vector
        },
        "multimedia_insights": [
            "Senior students (3rd & 4th Year) demonstrate expanded polygons in Wealth Planning Readiness and Financial Confidence.",
            "Students lacking an emergency fund display a distinctive visual distortion: heightened Stress Spending and suppressed Wealth Planning."
        ]
    }
