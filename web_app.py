"""
Pure Python Full-Stack Web Application for Survey Data Mining & Analytics
Built with FastAPI, Plotly, Pandas, and Scikit-Learn
"""
import os
import glob
import json
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from modules.data_processor import clean_survey_data
from modules.decision_trees import (
    calculate_entropy, get_feature_gain_table, DecisionTreeMiner
)
from modules.visualizations import (
    plot_distribution, plot_likert_summary, plot_crosstab_heatmap,
    plot_sunburst_hierarchy, plot_feature_gain_comparison,
    plot_interactive_tree_structure
)
from modules.apriori import extract_transactions, run_apriori
from modules.kmeans import run_kmeans_segmentation
from modules.knn import run_knn_classification
from modules.naive_bayes import run_naive_bayes_classification
from modules.decision_tree import train_cart_tree
from modules.text_mining import analyze_survey_text
from modules.spatial_mining import run_spatial_mining
from modules.web_mining import run_web_mining
from modules.multimedia_mining import extract_visual_behavioral_signatures

# Initialize App & Templates
app = FastAPI(title="FinPulse Survey Mining Portal")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

templates = Jinja2Templates(directory=TEMPLATES_DIR)
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Load and clean dataset once in memory
DATA_DIR = os.path.join(BASE_DIR, "data")
csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
if csv_files:
    raw_df = pd.read_csv(csv_files[0])
    clean_df = clean_survey_data(raw_df)
else:
    raw_df = pd.DataFrame()
    clean_df = pd.DataFrame()

DATA_DICTIONARY = [
    {"name": "Academic_Year", "type": "Ordinal Categorical", "description": "Current stage of degree (1st Year to Post-Grad)."},
    {"name": "Stream_Major", "type": "Nominal Categorical", "description": "Academic field: Data Science, IT & Eng, Commerce, Sciences."},
    {"name": "Living_Situation", "type": "Nominal Categorical", "description": "Living with parents, renting private PG, or campus hostel."},
    {"name": "Commute_Mode", "type": "Nominal Categorical", "description": "Public transit, auto/cab, personal bike/car, walking."},
    {"name": "Monthly_Budget", "type": "Ordinal Categorical", "description": "Monthly discretionary money managed (<₹2k to >₹10k)."},
    {"name": "Tracking_Method", "type": "Nominal Categorical", "description": "Method of tracking expenses (Mental Math, App, Sheet)."},
    {"name": "Has_Emergency_Fund", "type": "Binary Target", "description": "Maintains 1-month liquid emergency reserve (Yes/No)."},
    {"name": "Has_Investment_Account", "type": "Binary Target", "description": "Holds Demat, brokerage or crypto wallet (Yes/No)."},
    {"name": "Plan_Automated_Invest", "type": "Binary Target", "description": "Plans to automate SIP investments post-graduation (Yes/No)."},
    {"name": "Peer_Pressure_Spend", "type": "Likert Scale (1-5)", "description": "Frequency of spending to conform to peer activities."},
    {"name": "Stress_Spend", "type": "Likert Scale (1-5)", "description": "Likelihood of spending on food/entertainment when stressed."},
    {"name": "Financial_Confidence", "type": "Likert Scale (1-5)", "description": "Subjective confidence in managing personal funds post-grad."}
]


# Pydantic Request Models
class MineRequest(BaseModel):
    target: str = "Has_Emergency_Fund"
    algorithm: str = "J48"
    max_depth: int = 3
    features: Optional[List[str]] = None

class PredictRequest(BaseModel):
    sample: Dict[str, Any]
    target: str = "Has_Emergency_Fund"


# ----------------- WEB ROUTES -----------------

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    """Serves the primary web dashboard interface."""
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/api/overview")
async def get_overview():
    """Provides survey records and data dictionary."""
    records = clean_df.replace({np.nan: ""}).to_dict(orient="records")
    return JSONResponse({
        "total_records": len(clean_df),
        "dictionary": DATA_DICTIONARY,
        "records": records[:100]  # preview records
    })

@app.get("/api/chart/distribution")
async def get_distribution_chart(feature: str = "Academic_Year"):
    if feature not in clean_df.columns:
        feature = "Academic_Year"
    fig = plot_distribution(clean_df, feature)
    return JSONResponse(json.loads(fig.to_json()))

@app.get("/api/chart/likert")
async def get_likert_chart():
    likert_cols = [
        "Peer_Pressure_Spend", "Stress_Spend", "Financial_Confidence",
        "Lifestyle_Upgrade_Spend", "Wealth_Plan_Readiness"
    ]
    fig = plot_likert_summary(clean_df, likert_cols)
    return JSONResponse(json.loads(fig.to_json()))

@app.get("/api/chart/crosstab")
async def get_crosstab_chart(x: str = "Monthly_Budget", y: str = "Has_Emergency_Fund"):
    fig = plot_crosstab_heatmap(clean_df, x, y)
    return JSONResponse(json.loads(fig.to_json()))

@app.get("/api/chart/sunburst")
async def get_sunburst_chart():
    fig = plot_sunburst_hierarchy(clean_df, ["Stream_Major", "Monthly_Budget", "Has_Emergency_Fund"])
    return JSONResponse(json.loads(fig.to_json()))


@app.post("/api/mine")
async def mine_decision_tree(req: MineRequest):
    """Executes ID3 or J48 decision tree mining in pure Python."""
    target = req.target
    if target not in clean_df.columns:
        target = "Has_Emergency_Fund"

    available_features = [
        "Academic_Year", "Stream_Major", "Living_Situation", "Commute_Mode",
        "Funding_Source", "Monthly_Budget", "Tracking_Method", "Research_Frequency",
        "Peer_Pressure_Spend", "Stress_Spend", "Financial_Confidence",
        "Lifestyle_Upgrade_Spend", "Wealth_Plan_Readiness", "Investment_Obstacle"
    ]
    features = req.features or [f for f in available_features if f in clean_df.columns and f != target][:7]

    # Calculate Information Gain / Gain Ratio table
    gain_table = get_feature_gain_table(clean_df, features, target)
    gain_fig = plot_feature_gain_comparison(gain_table)

    # Fit Decision Tree Miner
    miner = DecisionTreeMiner(algorithm=req.algorithm, max_depth=req.max_depth, min_samples_split=4)
    miner.fit(clean_df, features, target)

    # Build interactive tree visualization
    tree_dict = miner.root.to_dict()
    tree_fig = plot_interactive_tree_structure(tree_dict)
    rules = miner.extract_rules()

    return JSONResponse({
        "gain_table": gain_table.to_dict(orient="records"),
        "gain_fig": json.loads(gain_fig.to_json()),
        "tree_fig": json.loads(tree_fig.to_json()),
        "rules": rules
    })


@app.post("/api/predict")
async def predict_student_persona(req: PredictRequest):
    """Predicts outcome for a hypothetical student persona using mined rules."""
    target = req.target
    sample_series = pd.Series(req.sample)

    features = [
        "Academic_Year", "Stream_Major", "Living_Situation", "Monthly_Budget",
        "Tracking_Method", "Research_Frequency", "Peer_Pressure_Spend",
        "Stress_Spend", "Financial_Confidence"
    ]
    features = [f for f in features if f in clean_df.columns and f != target]

    miner = DecisionTreeMiner(algorithm="J48", max_depth=3, min_samples_split=4)
    miner.fit(clean_df, features, target)
    prediction = miner.predict_one(sample_series)

    # Financial advice logic
    tracking = req.sample.get("Tracking_Method", "")
    stress = req.sample.get("Stress_Spend", 3)
    peer = req.sample.get("Peer_Pressure_Spend", 3)

    advice_points = []
    if tracking in ["Mental Math", "I don't track it"]:
        advice_points.append("Replace mental accounting with a digital tracking tool (e.g., Notion, Excel, or expense tracking apps) to convert loose change into structured savings.")
    if stress >= 4:
        advice_points.append("Implement a 48-hour delay rule for exam-related emotional food/lifestyle spending to curb impulse outlays.")
    if peer >= 4:
        advice_points.append("Establish a dedicated weekly social budget ceiling to participate in peer activities without depleting primary reserves.")
    if not advice_points:
        advice_points.append("Maintain disciplined budgeting and explore opening a zero-maintenance Demat account for long-term index SIPs.")

    return JSONResponse({
        "prediction": str(prediction),
        "advice": " ".join(advice_points)
    })


# ----------------- EXTENDED DATA MINING ENDPOINTS -----------------

@app.get("/api/mining/apriori")
async def get_apriori_rules(basket: str = "spending", min_support: float = 0.15, min_confidence: float = 0.45):
    """Executes Apriori association rule mining on multi-select survey baskets."""
    col = "Recent_Spending" if basket == "spending" else "Asset_Interests"
    transactions = extract_transactions(clean_df, col)
    itemsets_df, rules_df = run_apriori(transactions, min_support=min_support, min_confidence=min_confidence)
    return JSONResponse({
        "basket_type": basket,
        "total_baskets": len(transactions),
        "frequent_itemsets": itemsets_df.to_dict(orient="records"),
        "rules": rules_df.to_dict(orient="records")
    })

@app.get("/api/mining/kmeans")
async def get_kmeans_clusters(k: int = 3):
    """Executes K-Means clustering and returns PCA 2D projections & archetypes."""
    res = run_kmeans_segmentation(clean_df, n_clusters=k)
    return JSONResponse(res)

@app.get("/api/mining/knn")
async def get_knn_classification(target: str = "Has_Emergency_Fund", k: int = 5):
    """Executes K-Nearest Neighbors classifier."""
    if target not in clean_df.columns:
        target = "Has_Emergency_Fund"
    res = run_knn_classification(clean_df, target=target, k=k)
    return JSONResponse(res)

@app.get("/api/mining/naive_bayes")
async def get_naive_bayes_classification(target: str = "Has_Emergency_Fund"):
    """Executes Naive Bayes classifier."""
    if target not in clean_df.columns:
        target = "Has_Emergency_Fund"
    res = run_naive_bayes_classification(clean_df, target=target)
    return JSONResponse(res)

@app.get("/api/mining/cart")
async def get_cart_tree(target: str = "Has_Emergency_Fund", depth: int = 4):
    """Executes CART Decision Tree (Gini Impurity)."""
    if target not in clean_df.columns:
        target = "Has_Emergency_Fund"
    features = [
        "Academic_Year", "Stream_Major", "Living_Situation", "Monthly_Budget",
        "Tracking_Method", "Research_Frequency", "Peer_Pressure_Spend", "Stress_Spend"
    ]
    features = [f for f in features if f in clean_df.columns and f != target]
    res = train_cart_tree(clean_df, features=features, target=target, max_depth=depth)
    res.pop("model", None)
    return JSONResponse(res)

@app.get("/api/mining/text")
async def get_text_mining():
    """Extracts NLP topics and word frequency from student comments."""
    res = analyze_survey_text(clean_df)
    return JSONResponse(res)

@app.get("/api/mining/spatial")
async def get_spatial_mining():
    """Extracts spatial distance zones and commute correlations."""
    res = run_spatial_mining(clean_df)
    return JSONResponse(res)

@app.get("/api/mining/web")
async def get_web_mining():
    """Extracts digital research habits and FinTech adoption metrics."""
    res = run_web_mining(clean_df)
    return JSONResponse(res)

@app.get("/api/mining/multimedia")
async def get_multimedia_mining():
    """Extracts perceptual visual radar signatures across student cohorts."""
    res = extract_visual_behavioral_signatures(clean_df)
    return JSONResponse(res)


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("  FinPulse Survey Analytics & Data Mining Web Portal")
    print("  Server launching on: http://localhost:8000")
    print("="*70 + "\n")
    uvicorn.run("web_app:app", host="127.0.0.1", port=8000, reload=True)
