"""
Pure Python Full-Stack Web Application for Survey Data Mining & Analytics
Project: "Behavioral Insights into Financial Planning Among College Students"
FastAPI backend with Plotly, Pandas, NumPy, and Scikit-Learn.
"""
import os
import glob
import io
import json
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request, Query, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from modules.data_processor import (
    clean_survey_data, get_executive_kpis, get_key_findings, parse_multiselect, calculate_fdi
)
import modules.visualizations as viz
from modules.decision_trees import (
    calculate_entropy, get_feature_gain_table, DecisionTreeMiner
)
from modules.apriori import extract_transactions, run_apriori
from modules.kmeans import run_kmeans_segmentation
from modules.knn import run_knn_classification
from modules.naive_bayes import run_naive_bayes_classification
from modules.decision_tree import train_cart_tree
try:
    from modules.text_mining import analyze_survey_text
    from modules.spatial_mining import run_spatial_mining
    from modules.web_mining import run_web_mining
    from modules.multimedia_mining import extract_visual_behavioral_signatures
except ImportError as e:
    print(f"Warning: Extended mining module missing dependency: {e}")

# Initialize App & Directories
app = FastAPI(title="Student Financial Behaviour Analytics Dashboard")

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
    {"name": "Respondent_ID", "type": "Identifier", "description": "Anonymized survey respondent code (RESP_001 to RESP_131)."},
    {"name": "Academic_Year", "type": "Ordinal Categorical", "description": "Academic stage: 1st Year, 2nd Year, 3rd Year, 4th Year, Post-Grad."},
    {"name": "Stream_Major", "type": "Nominal Categorical", "description": "Academic stream: Data Science & AI, Engineering & IT, Commerce, Sciences."},
    {"name": "Living_Situation", "type": "Nominal Categorical", "description": "Living with parents, renting private PG, or campus hostel."},
    {"name": "Commute_Mode", "type": "Nominal Categorical", "description": "Public transit, cab/auto, personal vehicle, walking/campus."},
    {"name": "Monthly_Budget", "type": "Ordinal Categorical", "description": "Discretionary money managed monthly (<₹2k to >₹10k)."},
    {"name": "Tracking_Method", "type": "Nominal Categorical", "description": "Budget tracking mechanism (App, Sheet, Pen/Paper, Mental Math, None)."},
    {"name": "Has_Emergency_Fund", "type": "Binary Indicator", "description": "Possesses a 1-month liquid expense buffer (Yes/No)."},
    {"name": "Has_Investment_Account", "type": "Binary Indicator", "description": "Holds active Demat, brokerage, or crypto account (Yes/No)."},
    {"name": "Research_Frequency", "type": "Ordinal Categorical", "description": "Frequency of personal finance research (Daily, Weekly, Monthly, Rarely)."},
    {"name": "Recent_Spending", "type": "Multi-Select Categorical", "description": "Discretionary categories spent on in prior 7 days."},
    {"name": "Peer_Pressure_Spend", "type": "Likert Scale (1-5)", "description": "Tendency to spend to match peer social activities."},
    {"name": "Stress_Spend", "type": "Likert Scale (1-5)", "description": "Likelihood of spending on food/entertainment when stressed."},
    {"name": "Financial_Confidence", "type": "Likert Scale (1-5)", "description": "Subjective confidence in managing personal funds post-graduation."},
    {"name": "Lifestyle_Upgrade_Spend", "type": "Likert Scale (1-5)", "description": "Tendency to upgrade tech, gear, or style when new models launch."},
    {"name": "Wealth_Plan_Readiness", "type": "Likert Scale (1-5)", "description": "Possession of an actionable 3-year debt and wealth creation roadmap."},
    {"name": "Salary_Alone_Enough", "type": "Binary Indicator", "description": "Belief that traditional employment salary alone will suffice."},
    {"name": "Plan_Automated_Invest", "type": "Binary Indicator", "description": "Intention to automate investments (SIPs) once securing full-time job."},
    {"name": "Asset_Interests", "type": "Multi-Select Categorical", "description": "Preferred asset classes over 5 years (Mutual Funds, Stocks, Gold, Real Estate, Crypto)."},
    {"name": "Portfolio_Review_Freq", "type": "Ordinal Categorical", "description": "Envisioned portfolio review frequency (Monthly, Quarterly, Annually)."},
    {"name": "Investment_Obstacle", "type": "Nominal Categorical", "description": "Primary impediment to investing (Volatility fear, Capital, Education)."},
    {"name": "Philosophy_Active_vs_Passive", "type": "Nominal Categorical", "description": "Philosophical preference: Active income, Passive indexing, or Frugality."},
    {"name": "FDI_Score", "type": "Composite Index (0-100)", "description": "Financial Discipline Index based on Tracking, Liquidity, Planning & Research."}
]

# Pydantic Models
class MineRequest(BaseModel):
    target: str = "Has_Emergency_Fund"
    algorithm: str = "J48"
    max_depth: int = 3
    features: Optional[List[str]] = None

class PredictRequest(BaseModel):
    sample: Dict[str, Any]
    target: str = "Has_Emergency_Fund"


import functools

_df_cache = {}

def get_filtered_df(
    academic_year: Optional[str] = "All",
    stream_major: Optional[str] = "All",
    living_situation: Optional[str] = "All"
) -> pd.DataFrame:
    cache_key = (academic_year, stream_major, living_situation)
    if cache_key in _df_cache:
        return _df_cache[cache_key]
        
    df_f = clean_df.copy()
    if academic_year and academic_year != "All" and "Academic_Year" in df_f.columns:
        df_f = df_f[df_f["Academic_Year"] == academic_year]
    if stream_major and stream_major != "All" and "Stream_Major" in df_f.columns:
        df_f = df_f[df_f["Stream_Major"] == stream_major]
    if living_situation and living_situation != "All" and "Living_Situation" in df_f.columns:
        df_f = df_f[df_f["Living_Situation"] == living_situation]
        
    _df_cache[cache_key] = df_f
    return df_f


# ----------------- PRIMARY WEB & API ROUTES -----------------

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    """Serves the primary web dashboard interface with cache-busting headers."""
    response = templates.TemplateResponse(request=request, name="index.html")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.get("/api/filter-options")
async def get_filter_options():
    """Returns available unique categories for dynamic UI dropdowns."""
    academic_years = ["All"] + sorted([y for y in clean_df["Academic_Year"].dropna().unique() if y])
    streams = ["All"] + sorted([s for s in clean_df["Stream_Major"].dropna().unique() if s])
    living = ["All"] + sorted([l for l in clean_df["Living_Situation"].dropna().unique() if l])

    return JSONResponse({
        "academic_years": academic_years,
        "streams": streams,
        "living_situations": living,
        "total_records": len(clean_df)
    })

_dashboard_cache = {}

@app.get("/api/dashboard-data")
async def get_dashboard_data(
    academic_year: Optional[str] = "All",
    stream_major: Optional[str] = "All",
    living_situation: Optional[str] = "All"
):
    """
    Primary endpoint returning dynamic KPIs, structured module chart JSONs,
    and automatic insights for the selected filters.
    """
    cache_key = (academic_year, stream_major, living_situation)
    if cache_key in _dashboard_cache:
        return JSONResponse(_dashboard_cache[cache_key])

    df = get_filtered_df(academic_year, stream_major, living_situation)
    n = len(df)
    
    # 1. Executive KPIs & Narrative
    kpis = get_executive_kpis(df)
    
    # 2. Key Findings & Academic Conclusions
    findings_data = get_key_findings(df)

    # 3. Module 2: Demographics Charts
    demo_charts = {}
    if n > 0:
        demo_charts["academic_year_donut"] = json.loads(viz.plot_donut_chart(df, "Academic_Year", "Academic Year Distribution").to_json())
        demo_charts["stream_bar"] = json.loads(viz.plot_bar_chart(df, "Stream_Major", "Academic Stream / Major Breakdown", horizontal=True).to_json())
        demo_charts["living_donut"] = json.loads(viz.plot_donut_chart(df, "Living_Situation", "Living Situation Distribution").to_json())
        demo_charts["commute_bar"] = json.loads(viz.plot_bar_chart(df, "Commute_Mode", "Campus Commuting Mode", horizontal=True).to_json())
        demo_charts["academic_budget_stacked"] = json.loads(viz.plot_stacked_academic_budget(df).to_json())

    # 4. Module 3: Income & Discipline Charts
    income_charts = {}
    if n > 0:
        income_charts["funding_donut"] = json.loads(viz.plot_donut_chart(df, "Funding_Source", "Primary Monthly Funding Source").to_json())
        budget_order = ["Under ₹2,000", "₹2,000 - ₹5,000", "₹5,000 - ₹10,000", "Above ₹10,000"]
        income_charts["budget_col"] = json.loads(viz.plot_bar_chart(df, "Monthly_Budget", "Monthly Discretionary Money Managed", order=budget_order).to_json())
        income_charts["tracking_bar"] = json.loads(viz.plot_bar_chart(df, "Tracking_Method", "Expense Tracking Method", horizontal=True).to_json())
        income_charts["emergency_donut"] = json.loads(viz.plot_donut_chart(df, "Has_Emergency_Fund", "Emergency Fund Availability (1-Month Cushion)").to_json())
        income_charts["demat_donut"] = json.loads(viz.plot_donut_chart(df, "Has_Investment_Account", "Demat / Brokerage / Crypto Ownership").to_json())
        res_order = ["Rarely / Never", "Monthly", "Weekly", "Daily"]
        income_charts["research_bar"] = json.loads(viz.plot_bar_chart(df, "Research_Frequency", "Financial Research Frequency", order=res_order).to_json())
        income_charts["comp_budget_emergency"] = json.loads(viz.plot_comparative_rate(df, "Monthly_Budget", "Has_Emergency_Fund", "Emergency Fund Coverage by Monthly Budget Tier").to_json())
        income_charts["comp_tracking_emergency"] = json.loads(viz.plot_comparative_rate(df, "Tracking_Method", "Has_Emergency_Fund", "Emergency Fund Coverage by Expense Tracking Method").to_json())
        income_charts["comp_research_confidence"] = json.loads(viz.plot_research_vs_confidence(df).to_json())

    # 5. Module 4: Spending Behaviour Charts
    spending_charts = {}
    if n > 0:
        spending_counts = {}
        for s in df["Recent_Spending"].dropna():
            for item in parse_multiselect(s):
                spending_counts[item] = spending_counts.get(item, 0) + 1
        spending_charts["recent_spending_bar"] = json.loads(viz.plot_multiselect_breakdown(
            spending_counts, n, "Most Common Spending Categories (Prior 7 Days)"
        ).to_json())
        spending_charts["peer_vs_stress_bubble"] = json.loads(viz.plot_peer_vs_stress_correlation(df).to_json())

    # 6. Module 5: Mindset & Planning Charts
    mindset_charts = {}
    if n > 0:
        likert_cols = [
            "Peer_Pressure_Spend", "Stress_Spend", "Financial_Confidence",
            "Lifestyle_Upgrade_Spend", "Wealth_Plan_Readiness"
        ]
        mindset_charts["likert_summary"] = json.loads(viz.plot_likert_diverging(df, likert_cols).to_json())
        mindset_charts["conf_plan_matrix"] = json.loads(viz.plot_confidence_planning_matrix(df).to_json())

    # 7. Module 6: Investment Readiness Charts
    invest_charts = {}
    if n > 0:
        invest_charts["salary_alone_donut"] = json.loads(viz.plot_donut_chart(df, "Salary_Alone_Enough", "Is Traditional Salary Enough for Long-Term Goals?").to_json())
        invest_charts["sip_donut"] = json.loads(viz.plot_donut_chart(df, "Plan_Automated_Invest", "Intention to Automate Future SIP Investments").to_json())
        
        asset_counts = {}
        for a in df["Asset_Interests"].dropna():
            for item in parse_multiselect(a):
                asset_counts[item] = asset_counts.get(item, 0) + 1
        invest_charts["asset_bar"] = json.loads(viz.plot_multiselect_breakdown(
            asset_counts, n, "Preferred Asset Classes Over Next 5 Years"
        ).to_json())

        invest_charts["portfolio_review_bar"] = json.loads(viz.plot_bar_chart(df, "Portfolio_Review_Freq", "Envisioned Portfolio Review Frequency").to_json())
        invest_charts["obstacle_bar"] = json.loads(viz.plot_bar_chart(df, "Investment_Obstacle", "Primary Obstacles to Starting Investment Journey", horizontal=True).to_json())
        invest_charts["philosophy_bar"] = json.loads(viz.plot_bar_chart(df, "Philosophy_Active_vs_Passive", "Financial Independence Strategic Approach", horizontal=True).to_json())

    # 8. FDI Distribution & Summary Statistics
    fdi_chart = {}
    fdi_stats = {
        "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0,
        "low_pct": 0.0, "mod_pct": 0.0, "high_pct": 0.0,
        "low_count": 0, "mod_count": 0, "high_count": 0
    }
    if n > 0 and "FDI_Score" in df.columns:
        fdi_chart = json.loads(viz.plot_fdi_distribution(df).to_json())
        scores = df["FDI_Score"].dropna()
        if len(scores) > 0:
            fdi_stats = {
                "mean": round(float(scores.mean()), 1),
                "median": round(float(scores.median()), 1),
                "min": round(float(scores.min()), 1),
                "max": round(float(scores.max()), 1),
                "low_pct": round(float((scores <= 35).mean() * 100), 1),
                "mod_pct": round(float(((scores > 35) & (scores <= 65)).mean() * 100), 1),
                "high_pct": round(float((scores > 65).mean() * 100), 1),
                "low_count": int((scores <= 35).sum()),
                "mod_count": int(((scores > 35) & (scores <= 65)).sum()),
                "high_count": int((scores > 65).sum()),
            }

    res = {
        "sample_size": n,
        "total_cohort": len(clean_df),
        "kpis": kpis,
        "fdi_chart": fdi_chart,
        "fdi_stats": fdi_stats,
        "demographics_charts": demo_charts,
        "income_charts": income_charts,
        "spending_charts": spending_charts,
        "mindset_charts": mindset_charts,
        "invest_charts": invest_charts,
        "findings": findings_data["findings"],
        "conclusions": findings_data["conclusions"]
    }
    _dashboard_cache[cache_key] = res
    return JSONResponse(res)

# ----------------- MODULE 7: DATA MINING ENDPOINTS -----------------

_correlation_cache = None

@app.get("/api/mining/correlation")
async def get_correlation_matrix():
    """Computes Spearman rank correlation matrix across ordinal/numeric behavioral dimensions."""
    global _correlation_cache
    if _correlation_cache:
        return JSONResponse(_correlation_cache)
        
    res_map = {"Daily": 3, "Weekly": 2, "Monthly": 1, "Rarely / Never": 0}
    df_calc = clean_df.copy()
    df_calc["Research_Freq_Num"] = df_calc["Research_Frequency"].map(lambda x: res_map.get(str(x).strip(), 0))

    corr_cols = [
        "Peer_Pressure_Spend", "Stress_Spend", "Financial_Confidence",
        "Lifestyle_Upgrade_Spend", "Wealth_Plan_Readiness", "Research_Freq_Num"
    ]
    # Pure Pandas/NumPy Spearman correlation (rank Pearson correlation) without scipy dependency
    corr_matrix = df_calc[corr_cols].rank().corr().round(3)
    fig = viz.plot_spearman_heatmap(corr_matrix)

    _correlation_cache = {
        "columns": corr_cols,
        "matrix": corr_matrix.to_dict(),
        "heatmap": json.loads(fig.to_json())
    }
    return JSONResponse(_correlation_cache)

_kmeans_cache = {}

@app.get("/api/mining/kmeans")
async def get_kmeans_clusters(k: int = 3):
    """Executes K-Means behavioral clustering and returns profiles, silhouette score, and 2D PCA."""
    if k in _kmeans_cache:
        return JSONResponse(_kmeans_cache[k])
    res = run_kmeans_segmentation(clean_df, n_clusters=k)
    _kmeans_cache[k] = res
    return JSONResponse(res)

_apriori_cache = {}

@app.get("/api/mining/apriori")
async def get_apriori_rules(basket: str = "behavior", min_support: float = 0.15, min_confidence: float = 0.5):
    """Executes Apriori association rule mining on behavioral traits or multi-select items."""
    cache_key = (basket, min_support, min_confidence)
    if cache_key in _apriori_cache:
        return JSONResponse(_apriori_cache[cache_key])
        
    transactions = extract_transactions(clean_df, basket_type=basket)
    itemsets_df, rules_df = run_apriori(transactions, min_support=min_support, min_confidence=min_confidence)
    res = {
        "basket_type": basket,
        "total_baskets": len(transactions),
        "frequent_itemsets": itemsets_df.to_dict(orient="records"),
        "rules": rules_df.to_dict(orient="records")
    }
    _apriori_cache[cache_key] = res
    return JSONResponse(res)

_dt_cache = {}

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
    
    cache_key = (target, req.algorithm, req.max_depth, tuple(features))
    if cache_key in _dt_cache:
        return JSONResponse(_dt_cache[cache_key])

    gain_table = get_feature_gain_table(clean_df, features, target)
    gain_fig = viz.plot_feature_gain_comparison(gain_table)

    miner = DecisionTreeMiner(algorithm=req.algorithm, max_depth=req.max_depth, min_samples_split=4)
    miner.fit(clean_df, features, target)

    tree_dict = miner.root.to_dict()
    tree_fig = viz.plot_interactive_tree_structure(tree_dict)
    rules = miner.extract_rules()

    res = {
        "gain_table": gain_table.to_dict(orient="records"),
        "gain_fig": json.loads(gain_fig.to_json()),
        "tree_fig": json.loads(tree_fig.to_json()),
        "rules": rules
    }
    _dt_cache[cache_key] = res
    return JSONResponse(res)

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

    # Personalized feedback logic
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

# Extended Mining Algorithms
@app.get("/api/mining/knn")
async def get_knn_classification(target: str = "Has_Emergency_Fund", k: int = 5):
    res = run_knn_classification(clean_df, target=target, k=k)
    return JSONResponse(res)

@app.get("/api/mining/naive_bayes")
async def get_naive_bayes_classification(target: str = "Has_Emergency_Fund"):
    res = run_naive_bayes_classification(clean_df, target=target)
    return JSONResponse(res)

@app.get("/api/mining/cart")
async def get_cart_tree(target: str = "Has_Emergency_Fund", depth: int = 4):
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
    res = analyze_survey_text(clean_df)
    return JSONResponse(res)

@app.get("/api/mining/spatial")
async def get_spatial_mining():
    res = run_spatial_mining(clean_df)
    return JSONResponse(res)

@app.get("/api/mining/web")
async def get_web_mining():
    res = run_web_mining(clean_df)
    return JSONResponse(res)

@app.get("/api/mining/multimedia")
async def get_multimedia_mining():
    res = extract_visual_behavioral_signatures(clean_df)
    return JSONResponse(res)

# ----------------- DATA DOWNLOAD ENDPOINTS -----------------

@app.get("/api/download/cleaned-data")
async def download_cleaned_data():
    """Streams the cleaned and anonymized CSV dataset (PII removed)."""
    stream = io.StringIO()
    clean_df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=Student_Financial_Habits_Cleaned_Dataset.csv"
    return response

@app.get("/api/download/summary")
async def download_summary_report():
    """Generates and downloads a structured markdown/text research summary report."""
    kpis = get_executive_kpis(clean_df)
    findings = get_key_findings(clean_df)
    
    report = f"""================================================================================
FIELD RESEARCH ANALYTICS REPORT: STUDENT FINANCIAL HABITS & SPENDING BEHAVIOR
Empirical Study: Behavioral Insights into Financial Planning Among College Students
================================================================================

1. EXECUTIVE OVERVIEW (SAMPLE SIZE N = {kpis['total_respondents']})
--------------------------------------------------------------------------------
- Average Financial Confidence (1-5): {kpis['avg_confidence']} / 5.0
- Emergency Fund Coverage (1-Month Cushion): {kpis['emergency_fund_pct']}%
- Structured Expense Tracking Rate: {kpis['tracking_rate_pct']}%
- SIP Investment Automation Intention: {kpis['sip_intention_pct']}%
- Actionable 3-Year Wealth Plan Readiness: {kpis['actionable_plan_pct']}%
- Cohort Mean Financial Discipline Index (FDI): {kpis['avg_fdi_score']} / 100

Executive Narrative:
{kpis['narrative_summary']}

2. KEY RESEARCH FINDINGS
--------------------------------------------------------------------------------
"""
    for f in findings["findings"]:
        report += f"\n* {f['title']} [{f['stat']}]:\n  {f['desc']}\n"

    report += """
3. ACADEMIC RESEARCH CONCLUSIONS
--------------------------------------------------------------------------------
A. Financial Management Practices:
""" + findings["conclusions"].get("management_practices", "") + """

B. Spending Behaviour & Psychology:
""" + findings["conclusions"].get("spending_behaviour", "") + """

C. Confidence & Preparedness Divergence:
""" + findings["conclusions"].get("confidence_and_preparedness", "") + """

D. Investment Preferences & Barriers:
""" + findings["conclusions"].get("investment_preferences", "") + """

E. Student Behavioural Segmentation:
""" + findings["conclusions"].get("behavioral_segments", "") + """

F. Campus Policy & Workshop Recommendations:
""" + findings["conclusions"].get("campus_recommendations", "") + """

================================================================================
Generated by: Student Financial Behaviour Analytics Dashboard
Methodology: Descriptive Statistics, Spearman Rank Correlation, K-Means Clustering, Apriori Association Rules
================================================================================
"""
    response = StreamingResponse(iter([report]), media_type="text/plain")
    response.headers["Content-Disposition"] = "attachment; filename=Student_Financial_Habits_Research_Summary.txt"
    return response


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("  Student Financial Behaviour Analytics Dashboard")
    print("  Server launching on: http://localhost:8000")
    print("="*70 + "\n")
    uvicorn.run("web_app:app", host="127.0.0.1", port=8000, reload=True)
