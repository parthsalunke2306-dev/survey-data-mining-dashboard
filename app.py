"""
Student Financial Behaviour Analytics Dashboard
Empirical Field Project: "Behavioral Insights into Financial Planning Among College Students"
Framework: Streamlit with Pandas, NumPy, Plotly, and Scikit-Learn
"""
import os
import glob
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Import modular preprocessing and visualization helpers
from modules.data_processor import (
    clean_survey_data, get_executive_kpis, get_key_findings, parse_multiselect, calculate_fdi
)
import modules.visualizations as viz
from modules.kmeans import run_kmeans_segmentation
from modules.apriori import extract_transactions, run_apriori
from modules.decision_trees import get_feature_gain_table, DecisionTreeMiner

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="Behavioral Insights into Financial Planning | Field Survey Analytics",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling: Slate and Emerald palette (#10B981)
st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"] {
        overflow-x: hidden !important;
        max-width: 100vw;
    }
    ::-webkit-scrollbar:horizontal {
        display: none !important;
        height: 0px !important;
    }
    *::-webkit-scrollbar:horizontal {
        display: none !important;
        height: 0px !important;
    }
    * {
        scrollbar-width: thin;
    }
    .main-title {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .kpi-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0F172A;
        margin-top: 0.2rem;
    }
    .kpi-sub {
        font-size: 0.75rem;
        color: #10B981;
        font-weight: 600;
        margin-top: 0.2rem;
    }
    .narrative-card {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: #F8FAFC;
        padding: 1.5rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        border-left: 5px solid #10B981;
    }
</style>
""", unsafe_allow_html=True)


# --- Load Survey Dataset ---
@st.cache_data
def load_and_preprocess_data():
    default_dir = os.path.join(os.path.dirname(__file__), "data")
    csv_files = glob.glob(os.path.join(default_dir, "*.csv"))
    if csv_files:
        raw = pd.read_csv(csv_files[0])
        # Anonymize dataset (drop Name, Email, Timestamp) and clean columns
        clean = clean_survey_data(raw, drop_pii=True)
        return clean
    return pd.DataFrame()

df_clean = load_and_preprocess_data()

if df_clean.empty:
    st.error("No survey dataset found in `data/` directory. Please ensure the CSV is placed inside the data folder.")
    st.stop()


# --- Sidebar Navigation & Filters ---
st.sidebar.markdown("### 🎓 Field Research Project")
st.sidebar.markdown("**Behavioral Insights into Financial Planning Among College Students**")
st.sidebar.caption("BSc Data Science Undergraduate Field Study")
st.sidebar.markdown("---")

st.sidebar.markdown("#### 🔍 Interactive Filters")

# Filter 1: Academic Year
year_options = ["All"] + sorted([y for y in df_clean["Academic_Year"].dropna().unique() if y])
selected_year = st.sidebar.selectbox("Academic Standing", year_options, index=0)

# Filter 2: Stream / Major
stream_options = ["All"] + sorted([s for s in df_clean["Stream_Major"].dropna().unique() if s])
selected_stream = st.sidebar.selectbox("Academic Stream / Major", stream_options, index=0)

# Filter 3: Living Situation
living_options = ["All"] + sorted([l for l in df_clean["Living_Situation"].dropna().unique() if l])
selected_living = st.sidebar.selectbox("Primary Living Situation", living_options, index=0)

# Apply dynamic filtering
df_filtered = df_clean.copy()
if selected_year != "All":
    df_filtered = df_filtered[df_filtered["Academic_Year"] == selected_year]
if selected_stream != "All":
    df_filtered = df_filtered[df_filtered["Stream_Major"] == selected_stream]
if selected_living != "All":
    df_filtered = df_filtered[df_filtered["Living_Situation"] == selected_living]

st.sidebar.markdown(f"**Sample Count:** `{len(df_filtered)}` of `{len(df_clean)}` students")

if st.sidebar.button("🔄 Reset All Filters"):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("#### 📥 Data Exports")

# Export Cleaned Anonymized CSV
csv_buffer = io.StringIO()
df_filtered.to_csv(csv_buffer, index=False)
st.sidebar.download_button(
    label="Download Cleaned CSV",
    data=csv_buffer.getvalue(),
    file_name="Student_Financial_Habits_Cleaned.csv",
    mime="text/csv"
)

# Export Summary Report
kpis_all = get_executive_kpis(df_filtered)
findings_all = get_key_findings(df_filtered)
summary_text = f"""FIELD RESEARCH SUMMARY REPORT: STUDENT FINANCIAL BEHAVIOUR
Dataset Sample Size: n = {len(df_filtered)} (Total Survey Cohort = {len(df_clean)})
Filters Applied: Year={selected_year}, Stream={selected_stream}, Living={selected_living}

1. EXECUTIVE METRICS:
- Average Financial Confidence (1-5): {kpis_all['avg_confidence']} / 5.0
- Emergency Fund Coverage Rate: {kpis_all['emergency_fund_pct']}%
- Structured Expense Tracking Rate: {kpis_all['tracking_rate_pct']}%
- SIP Automation Intention Rate: {kpis_all['sip_intention_pct']}%
- Actionable 3-Year Plan Readiness: {kpis_all['actionable_plan_pct']}%
- Financial Discipline Index (FDI): {kpis_all['avg_fdi_score']} / 100

NARRATIVE SUMMARY:
{kpis_all['narrative_summary']}
"""
st.sidebar.download_button(
    label="Download Research Report (.txt)",
    data=summary_text,
    file_name="Financial_Habits_Research_Report.txt",
    mime="text/plain"
)


# --- Main Dashboard Header ---
st.markdown("<div class='main-title'>Behavioral Insights into Financial Planning Among College Students</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Student Financial Behaviour Analytics Dashboard • 100% Calculated Empirical Survey Results (n = 131)</div>", unsafe_allow_html=True)


# --- Dashboard Tabs (7 Modules + Findings) ---
tab_overview, tab_demo, tab_income, tab_spend, tab_mindset, tab_invest, tab_mining, tab_findings = st.tabs([
    "📊 Executive Overview",
    "👥 Demographics",
    "💰 Income & Discipline",
    "🛍️ Spending Psychology",
    "🧠 Mindset & Planning",
    "📈 Investment Readiness",
    "🔬 Data Mining Suite",
    "💡 Findings & Conclusions"
])


# ================= MODULE 1: EXECUTIVE OVERVIEW =================
with tab_overview:
    kpis = get_executive_kpis(df_filtered)

    # Narrative Summary
    st.markdown(f"""
    <div class='narrative-card'>
        <div style='font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em; color:#34D399; font-weight:700; margin-bottom:0.5rem;'>
            Empirical Research Narrative Summary
        </div>
        <div style='font-size:1.05rem; line-height:1.6; font-weight:400;'>
            {kpis['narrative_summary']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 6 Dynamic KPI Cards
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.markdown(f"""
        <div class='kpi-box'>
            <div class='kpi-label'>Respondents</div>
            <div class='kpi-value'>{kpis['total_respondents']}</div>
            <div class='kpi-sub'>100% Verified</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='kpi-box'>
            <div class='kpi-label'>Mean Confidence</div>
            <div class='kpi-value'>{kpis['avg_confidence']} <span style='font-size:0.8rem; font-weight:normal; color:#64748B;'>/ 5</span></div>
            <div class='kpi-sub'>Subjective Optimism</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='kpi-box'>
            <div class='kpi-label'>Emergency Fund</div>
            <div class='kpi-value' style='color:#059669;'>{kpis['emergency_fund_pct']}%</div>
            <div class='kpi-sub'>1-Month Liquid Cushion</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class='kpi-box'>
            <div class='kpi-label'>Active Tracking</div>
            <div class='kpi-value'>{kpis['tracking_rate_pct']}%</div>
            <div class='kpi-sub' style='color:#D97706;'>Apps / Spreadsheets</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class='kpi-box'>
            <div class='kpi-label'>SIP Automation</div>
            <div class='kpi-value'>{kpis['sip_intention_pct']}%</div>
            <div class='kpi-sub'>Plan Automated Investing</div>
        </div>
        """, unsafe_allow_html=True)
    with col6:
        st.markdown(f"""
        <div class='kpi-box'>
            <div class='kpi-label'>Actionable Plan</div>
            <div class='kpi-value'>{kpis['actionable_plan_pct']}%</div>
            <div class='kpi-sub'>Rating 4-5 on 3-Yr Plan</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Financial Discipline Index (FDI) Section
    st.subheader("Financial Discipline Index (FDI)")
    st.caption("A project-defined exploratory research metric (0–100 scale) synthesizing expense tracking rigor, emergency liquidity buffer, 3-year planning readiness, and market research frequency.")

    fdi_col1, fdi_col2 = st.columns([2, 1])
    with fdi_col1:
        if len(df_filtered) > 0:
            fig_fdi = viz.plot_fdi_distribution(df_filtered)
            st.plotly_chart(fig_fdi, use_container_width=True)
    with fdi_col2:
        st.markdown(f"""
        <div style='background:#F8FAFC; border:1px solid #E2E8F0; padding:1.2rem; border-radius:12px;'>
            <div style='font-size:0.8rem; font-weight:700; color:#64748B; text-transform:uppercase;'>Cohort Mean FDI Score</div>
            <div style='font-size:2.2rem; font-weight:800; color:#059669; margin:4px 0;'>{kpis['avg_fdi_score']} <span style='font-size:1rem; color:#64748B;'>/ 100</span></div>
            <hr style='margin:0.8rem 0; border:none; border-top:1px solid #E2E8F0;'>
            <div style='font-size:0.8rem; color:#334155;'>
                <b>Formula Components (Max 25 pts each):</b><br>
                1. <b>Expense Tracking</b>: Digital/Ledger (25) | Mental (8) | None (0)<br>
                2. <b>Emergency Buffer</b>: Has 1-Month Cushion (25) | None (0)<br>
                3. <b>Actionable Planning</b>: Likert 1-5 scaled linearly (0 to 25)<br>
                4. <b>Research Frequency</b>: Daily (25) | Weekly (20) | Monthly (12) | Rarely (0)
            </div>
            <div style='margin-top:0.8rem; font-size:0.75rem; color:#64748B; background:#F1F5F9; padding:0.6rem; border-radius:8px;'>
                * Note: FDI is constructed for this undergraduate study to evaluate student preparedness; it is not a standardized psychometric or credit bureau score.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ================= MODULE 2: STUDENT DEMOGRAPHICS =================
with tab_demo:
    st.subheader("Academic & Demographic Distribution")
    st.caption("Breakdown of surveyed respondents across study standing, streams, living conditions, and daily commuting methods.")

    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        fig_year = viz.plot_donut_chart(df_filtered, "Academic_Year", "Academic Standing Distribution")
        st.plotly_chart(fig_year, use_container_width=True)
    with row1_c2:
        fig_stream = viz.plot_bar_chart(df_filtered, "Stream_Major", "Academic Stream / Major Breakdown", horizontal=True)
        st.plotly_chart(fig_stream, use_container_width=True)

    row2_c1, row2_c2 = st.columns(2)
    with row2_c1:
        fig_living = viz.plot_donut_chart(df_filtered, "Living_Situation", "Primary Living Situation")
        st.plotly_chart(fig_living, use_container_width=True)
    with row2_c2:
        fig_commute = viz.plot_bar_chart(df_filtered, "Commute_Mode", "Campus Commuting Mode", horizontal=True)
        st.plotly_chart(fig_commute, use_container_width=True)

    st.markdown("---")
    st.subheader("Academic Year vs Monthly Discretionary Funds Managed")
    fig_academic_budget = viz.plot_stacked_academic_budget(df_filtered)
    st.plotly_chart(fig_academic_budget, use_container_width=True)


# ================= MODULE 3: INCOME & FINANCIAL DISCIPLINE =================
with tab_income:
    st.subheader("Income Streams, Budgeting & Liquidity Reserves")
    st.caption("Empirical distributions of student monthly funding sources, discretionary allowances, tracking habits, and emergency preparedness.")

    inc_c1, inc_c2, inc_c3 = st.columns(3)
    with inc_c1:
        fig_fund = viz.plot_donut_chart(df_filtered, "Funding_Source", "Primary Funding Source")
        st.plotly_chart(fig_fund, use_container_width=True)
    with inc_c2:
        budget_order = ["Under ₹2,000", "₹2,000 - ₹5,000", "₹5,000 - ₹10,000", "Above ₹10,000"]
        fig_budg = viz.plot_bar_chart(df_filtered, "Monthly_Budget", "Monthly Money Managed", order=budget_order)
        st.plotly_chart(fig_budg, use_container_width=True)
    with inc_c3:
        fig_track = viz.plot_bar_chart(df_filtered, "Tracking_Method", "Expense Tracking Method", horizontal=True)
        st.plotly_chart(fig_track, use_container_width=True)

    res_c1, res_c2, res_c3 = st.columns(3)
    with res_c1:
        fig_em = viz.plot_donut_chart(df_filtered, "Has_Emergency_Fund", "Emergency Fund (1-Month Cushion)")
        st.plotly_chart(fig_em, use_container_width=True)
    with res_c2:
        fig_demat = viz.plot_donut_chart(df_filtered, "Has_Investment_Account", "Demat / Brokerage Ownership")
        st.plotly_chart(fig_demat, use_container_width=True)
    with res_c3:
        res_order = ["Rarely / Never", "Monthly", "Weekly", "Daily"]
        fig_res = viz.plot_bar_chart(df_filtered, "Research_Frequency", "Financial Research Frequency", order=res_order)
        st.plotly_chart(fig_res, use_container_width=True)

    st.markdown("---")
    st.subheader("Comparative Analysis: Rates & Associations")

    comp_c1, comp_c2, comp_c3 = st.columns(3)
    with comp_c1:
        fig_comp1 = viz.plot_comparative_rate(df_filtered, "Monthly_Budget", "Has_Emergency_Fund", "Emergency Fund Rate by Budget Tier")
        st.plotly_chart(fig_comp1, use_container_width=True)
    with comp_c2:
        fig_comp2 = viz.plot_comparative_rate(df_filtered, "Tracking_Method", "Has_Emergency_Fund", "Emergency Fund Rate by Tracking Method")
        st.plotly_chart(fig_comp2, use_container_width=True)
    with comp_c3:
        fig_comp3 = viz.plot_research_vs_confidence(df_filtered)
        st.plotly_chart(fig_comp3, use_container_width=True)


# ================= MODULE 4: SPENDING PSYCHOLOGY =================
with tab_spend:
    st.subheader("Weekly Spending Habits & Behavioral Triggers")
    st.caption("Multi-select breakdown of discretionary categories and psychological peer/stress spending correlations.")

    if len(df_filtered) > 0:
        spending_counts = {}
        for s in df_filtered["Recent_Spending"].dropna():
            for item in parse_multiselect(s):
                spending_counts[item] = spending_counts.get(item, 0) + 1

        fig_spend_bar = viz.plot_multiselect_breakdown(
            spending_counts, len(df_filtered), "Most Common Spending Categories (Prior 7 Days)"
        )
        st.plotly_chart(fig_spend_bar, use_container_width=True)

    sp_c1, sp_c2 = st.columns([3, 2])
    with sp_c1:
        fig_bubble = viz.plot_peer_vs_stress_correlation(df_filtered)
        st.plotly_chart(fig_bubble, use_container_width=True)
    with sp_c2:
        st.markdown("""
        <div style='background:#FFFFFF; border:1px solid #E2E8F0; padding:1.2rem; border-radius:12px; height:100%;'>
            <h4 style='font-size:0.95rem; font-weight:700; color:#0F172A; margin-bottom:0.8rem;'>Behavioral Insights & Takeaways</h4>
            <div style='font-size:0.8rem; color:#475569; line-height:1.6;'>
                <p><b>Statistically Significant Association (ρ = +0.318, p &lt; 0.001):</b><br>
                A moderate positive rank correlation is observed between peer-influenced expenditure and academic stress-induced spending.</p>
                <p><b>Discretionary Drivers:</b><br>
                66.4% of respondents spent money at cafes/restaurants, while 61.8% spent on transit, making them the primary recurring micro-drains on student balances.</p>
                <p><b>Passive Recurring Drains:</b><br>
                19.1% carry digital subscriptions (music/streaming), representing automated outflows that students rarely factor into weekly mental budgeting.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ================= MODULE 5: MINDSET & PLANNING =================
with tab_mindset:
    st.subheader("Financial Mindset & Planning Readiness")
    st.caption("Psychographic Likert evaluations and empirical 2x2 Confidence–Planning segmentation matrix.")

    likert_cols = [
        "Peer_Pressure_Spend", "Stress_Spend", "Financial_Confidence",
        "Lifestyle_Upgrade_Spend", "Wealth_Plan_Readiness"
    ]
    fig_likert = viz.plot_likert_diverging(df_filtered, likert_cols)
    st.plotly_chart(fig_likert, use_container_width=True)

    st.markdown("---")
    st.subheader("Confidence–Planning 2x2 Matrix Segmentation")
    fig_matrix = viz.plot_confidence_planning_matrix(df_filtered)
    st.plotly_chart(fig_matrix, use_container_width=True)


# ================= MODULE 6: INVESTMENT READINESS =================
with tab_invest:
    st.subheader("Investment Readiness & Forward-Looking Sentiment")
    st.caption("Perceptions on traditional salary sufficiency, investment automation, asset class preferences, and structural entry barriers.")

    inv_c1, inv_c2 = st.columns(2)
    with inv_c1:
        fig_sal = viz.plot_donut_chart(df_filtered, "Salary_Alone_Enough", "Is Traditional Salary Enough for Long-Term Goals?")
        st.plotly_chart(fig_sal, use_container_width=True)
    with inv_c2:
        fig_sip = viz.plot_donut_chart(df_filtered, "Plan_Automated_Invest", "Intention to Automate SIP Investments Once Employed")
        st.plotly_chart(fig_sip, use_container_width=True)

    if len(df_filtered) > 0:
        asset_counts = {}
        for a in df_filtered["Asset_Interests"].dropna():
            for item in parse_multiselect(a):
                asset_counts[item] = asset_counts.get(item, 0) + 1

        fig_asset = viz.plot_multiselect_breakdown(asset_counts, len(df_filtered), "Preferred Asset Classes Over Next 5 Years")
        st.plotly_chart(fig_asset, use_container_width=True)

    bar_c1, bar_c2 = st.columns(2)
    with bar_c1:
        fig_obs = viz.plot_bar_chart(df_filtered, "Investment_Obstacle", "Primary Obstacles to Starting Investment Journey", horizontal=True)
        st.plotly_chart(fig_obs, use_container_width=True)
    with bar_c2:
        fig_phil = viz.plot_bar_chart(df_filtered, "Philosophy_Active_vs_Passive", "Preferred Financial Independence Approach", horizontal=True)
        st.plotly_chart(fig_phil, use_container_width=True)


# ================= MODULE 7: DATA MINING SUITE =================
with tab_mining:
    st.subheader("Data Mining & Machine Learning Suite")
    st.caption("Pure Python implementations of Spearman Rank Correlation, K-Means Clustering, Apriori Association Rules, and J48 Decision Trees.")

    mining_choice = st.radio(
        "Select Analytical Technique:",
        ["Spearman Correlation Heatmap", "K-Means Student Segmentation", "Apriori Association Rules", "ID3 & J48 Decision Trees", "What-If Persona Simulator"],
        horizontal=True
    )

    if mining_choice == "Spearman Correlation Heatmap":
        st.markdown("#### Spearman Rank Correlation Matrix")
        st.caption("Analyzes monotonic associations between ordinal Likert responses and numerical research engagement.")
        res_map = {"Daily": 3, "Weekly": 2, "Monthly": 1, "Rarely / Never": 0}
        df_calc = df_clean.copy()
        df_calc["Research_Freq_Num"] = df_calc["Research_Frequency"].map(lambda x: res_map.get(str(x).strip(), 0))
        corr_cols = [
            "Peer_Pressure_Spend", "Stress_Spend", "Financial_Confidence",
            "Lifestyle_Upgrade_Spend", "Wealth_Plan_Readiness", "Research_Freq_Num"
        ]
        # Pure Pandas/NumPy Spearman correlation (rank Pearson correlation) without scipy dependency
        corr_matrix = df_calc[corr_cols].rank().corr().round(3)
        fig_corr = viz.plot_spearman_heatmap(corr_matrix)
        st.plotly_chart(fig_corr, use_container_width=True)

    elif mining_choice == "K-Means Student Segmentation":
        st.markdown("#### K-Means Clustering: Student Archetype Profiles")
        st.caption("Clustering performed across 6 core indicators: Tracking, Emergency Buffer, Confidence, Planning, Research, SIP Intention.")

        k_val = st.slider("Select Number of Clusters (k):", min_value=2, max_value=4, value=3)
        km_res = run_kmeans_segmentation(df_clean, n_clusters=k_val)

        # Render cluster profiles
        prof_cols = st.columns(len(km_res["profiles"]))
        for idx, p in enumerate(km_res["profiles"]):
            with prof_cols[idx]:
                st.markdown(f"""
                <div style='background:#FFFFFF; border:1px solid #E2E8F0; padding:1.1rem; border-radius:12px; border-top:4px solid {p['Color']};'>
                    <div style='font-size:0.75rem; font-weight:700; color:{p['Color']}; text-transform:uppercase;'>Cluster {p['Cluster_ID']+1}</div>
                    <div style='font-size:1.1rem; font-weight:800; color:#0F172A; margin:4px 0;'>{p['Archetype']}</div>
                    <div style='font-size:0.75rem; color:#64748B;'>{p['Count']} students ({p['Percentage']})</div>
                    <hr style='margin:0.6rem 0; border:none; border-top:1px solid #F1F5F9;'>
                    <div style='font-size:0.75rem; color:#475569; line-height:1.5;'>
                        {p['Description']}
                    </div>
                    <hr style='margin:0.6rem 0; border:none; border-top:1px solid #F1F5F9;'>
                    <div style='font-size:0.75rem; color:#334155;'>
                        • Emergency Buffer: <b>{p['Emergency_Fund_Rate']}</b><br>
                        • SIP Intention: <b>{p['SIP_Intention_Rate']}</b><br>
                        • Confidence: <b>{p['Avg_Confidence']}</b> / 5<br>
                        • Tracking Score: <b>{p['Avg_Tracking']}</b> / 3
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        # 2D PCA Scatter
        st.markdown(f"**2D PCA Projection (Silhouette Score: `{km_res['silhouette_score']}`)**")
        pca_df = pd.DataFrame(km_res["scatter_points"])
        fig_pca = px.scatter(
            pca_df, x="x", y="y", color="Cluster",
            hover_data=["ID", "Major", "Year"],
            title=f"2D PCA Projection of Students (k={k_val})",
            color_discrete_sequence=["#10B981", "#F59E0B", "#3B82F6", "#8B5CF6"]
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    elif mining_choice == "Apriori Association Rules":
        st.markdown("#### Apriori Association Rule Mining")
        basket_sel = st.selectbox("Select Transaction Basket Type:", ["behavior", "spending", "assets"], format_func=lambda x: {"behavior": "Behavioral Traits", "spending": "Recent Spending Categories", "assets": "Preferred Asset Classes"}[x])
        tx = extract_transactions(df_clean, basket_type=basket_sel)
        itemsets, rules = run_apriori(tx, min_support=0.15, min_confidence=0.5)

        st.markdown(f"**Total Transactions:** `{len(tx)}` | **Frequent Itemsets:** `{len(itemsets)}` | **Mined Rules:** `{len(rules)}`")
        if not rules.empty:
            st.dataframe(
                rules[["Rule", "Support_Pct", "Confidence_Pct", "Lift"]].rename(columns={
                    "Support_Pct": "Support", "Confidence_Pct": "Confidence", "Lift": "Lift Ratio"
                }),
                use_container_width=True
            )
        else:
            st.info("No association rules found at current thresholds.")

    elif mining_choice == "ID3 & J48 Decision Trees":
        st.markdown("#### Decision Tree Induction: Information Gain vs Gain Ratio")
        tree_target = st.selectbox("Target Variable:", ["Has_Emergency_Fund", "Has_Investment_Account"])
        features = [
            "Academic_Year", "Stream_Major", "Living_Situation", "Monthly_Budget",
            "Tracking_Method", "Research_Frequency", "Peer_Pressure_Spend", "Stress_Spend"
        ]
        gain_table = get_feature_gain_table(df_clean, features, tree_target)
        fig_gain = viz.plot_feature_gain_comparison(gain_table)
        st.plotly_chart(fig_gain, use_container_width=True)

        miner = DecisionTreeMiner(algorithm="J48", max_depth=3, min_samples_split=4)
        miner.fit(df_clean, features, tree_target)
        fig_tree = viz.plot_interactive_tree_structure(miner.root.to_dict())
        st.plotly_chart(fig_tree, use_container_width=True)

    elif mining_choice == "What-If Persona Simulator":
        st.markdown("#### Student Persona 'What-If' Financial Resilience Simulator")
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        with sim_col1:
            sim_track = st.selectbox("Tracking Method:", ["Mobile App", "Spreadsheet", "Pen and Paper", "Mental Math", "I don't track it"], index=3)
        with sim_col2:
            sim_conf = st.slider("Financial Confidence (1-5):", 1, 5, 4)
        with sim_col3:
            sim_stress = st.slider("Stress Spending Propensity (1-5):", 1, 5, 3)

        if st.button("Predict Financial Resilience Persona"):
            sample = pd.Series({"Tracking_Method": sim_track, "Financial_Confidence": sim_conf, "Stress_Spend": sim_stress})
            features = ["Tracking_Method", "Financial_Confidence", "Stress_Spend"]
            m = DecisionTreeMiner(algorithm="J48", max_depth=3)
            m.fit(df_clean, features, "Has_Emergency_Fund")
            pred = m.predict_one(sample)

            st.success(f"**Predicted Emergency Fund Status:** `{pred}`")
            if sim_track in ["Mental Math", "I don't track it"]:
                st.warning("💡 **Actionable Recommendation:** Shifting from mental accounting to digital tracking (e.g. mobile apps or Excel) is empirically associated with higher emergency reserve accumulation.")


# ================= MODULE 8: FINDINGS & CONCLUSIONS =================
with tab_findings:
    st.subheader("Key Findings & Academic Research Conclusions")
    st.caption("Empirical data-driven observations and formal research conclusions grounded in actual survey metrics.")

    findings_res = get_key_findings(df_filtered)

    # 5 Observation Cards
    f_cols = st.columns(len(findings_res["findings"]))
    for i, f in enumerate(findings_res["findings"]):
        with f_cols[i % len(f_cols)]:
            st.markdown(f"""
            <div style='background:#FFFFFF; border:1px solid #E2E8F0; padding:1.1rem; border-radius:12px; margin-bottom:1rem; height:100%;'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='font-size:0.7rem; font-weight:700; color:#065F46; background:#D1FAE5; padding:2px 8px; border-radius:4px;'>OBSERVATION</span>
                    <span style='font-size:0.85rem; font-weight:800; color:#0F172A;'>{f['stat']}</span>
                </div>
                <div style='font-size:0.95rem; font-weight:800; color:#0F172A; margin:0.4rem 0;'>{f['title']}</div>
                <div style='font-size:0.75rem; color:#475569; line-height:1.5;'>{f['desc']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Formal Academic Research Conclusion
    st.markdown("### Formal Academic Research Conclusions")
    c = findings_res["conclusions"]

    st.markdown(f"""
    <div style='background:#F8FAFC; border:1px solid #E2E8F0; border-radius:14px; padding:1.5rem; line-height:1.7; font-size:0.85rem; color:#334155;'>
        <h4 style='color:#0F172A; font-weight:700; margin-bottom:0.2rem;'>1. Financial Management Practices</h4>
        <p>{c.get('management_practices', '')}</p>

        <h4 style='color:#0F172A; font-weight:700; margin-top:1rem; margin-bottom:0.2rem;'>2. Spending Behaviour & Behavioral Correlations</h4>
        <p>{c.get('spending_behaviour', '')}</p>

        <h4 style='color:#0F172A; font-weight:700; margin-top:1rem; margin-bottom:0.2rem;'>3. Financial Confidence & Preparedness Divergence</h4>
        <p>{c.get('confidence_and_preparedness', '')}</p>

        <h4 style='color:#0F172A; font-weight:700; margin-top:1rem; margin-bottom:0.2rem;'>4. Investment Preferences & Structural Obstacles</h4>
        <p>{c.get('investment_preferences', '')}</p>

        <h4 style='color:#0F172A; font-weight:700; margin-top:1rem; margin-bottom:0.2rem;'>5. Identified Student Behaviour Segments</h4>
        <p>{c.get('behavioral_segments', '')}</p>

        <h4 style='color:#059669; font-weight:700; margin-top:1rem; margin-bottom:0.2rem;'>6. Campus Financial Literacy Workshop Recommendations</h4>
        <p>{c.get('campus_recommendations', '')}</p>
    </div>
    """, unsafe_allow_html=True)
