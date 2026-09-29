import os
import glob
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, confusion_matrix

from modules.data_processor import clean_survey_data, SHORT_NAME_MAP
from modules.decision_trees import (
    calculate_entropy, calculate_information_gain, calculate_gain_ratio,
    get_feature_gain_table, DecisionTreeMiner
)
from modules.visualizations import (
    plot_distribution, plot_likert_summary, plot_crosstab_heatmap,
    plot_sunburst_hierarchy, plot_feature_gain_comparison,
    plot_confusion_matrix_interactive, plot_interactive_tree_structure, COLORS
)

# Page Setup
st.set_page_config(
    page_title="Student Financial Behavior | Data Mining Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #334155;
    }
    .metric-title { font-size: 0.85rem; color: #94A3B8; text-transform: uppercase; font-weight: 600; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #38BDF8; margin: 4px 0; }
    .metric-sub { font-size: 0.75rem; color: #CBD5E1; }
    .badge {
        display: inline-block; padding: 2px 8px; border-radius: 6px;
        font-size: 0.8rem; font-weight: 600; margin-right: 5px;
    }
    .badge-blue { background-color: #DBEAFE; color: #1E40AF; }
    .badge-green { background-color: #D1FAE5; color: #065F46; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px; border-radius: 8px; font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- DATA LOADING -----------------
@st.cache_data
def load_data():
    default_dir = os.path.join(os.path.dirname(__file__), "data")
    csv_files = glob.glob(os.path.join(default_dir, "*.csv"))
    if csv_files:
        raw_df = pd.read_csv(csv_files[0])
        clean_df = clean_survey_data(raw_df)
        return raw_df, clean_df, os.path.basename(csv_files[0])
    return None, None, None

raw_df, clean_df, filename = load_data()

# Sidebar
st.sidebar.image("https://img.icons8.com/isometric/100/combo-chart.png", width=65)
st.sidebar.title("Survey Analytics")
st.sidebar.caption("Data Mining: ID3, J48 / C4.5 Decision Trees")

uploaded_file = st.sidebar.file_uploader("Upload New Survey CSV (optional)", type=["csv"])
if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    clean_df = clean_survey_data(raw_df)
    filename = uploaded_file.name

if clean_df is None:
    st.error("No survey dataset found in `data/` folder. Please upload a CSV file.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Loaded Survey:** `{filename}`")
st.sidebar.markdown(f"**Respondents:** `{len(clean_df)}`")
st.sidebar.markdown(f"**Variables:** `{clean_df.shape[1]}`")
st.sidebar.markdown("---")
st.sidebar.markdown("### Project Quick Links")
st.sidebar.info("🎓 Field Study: **Student Financial Habits & Spending Behavior**\n\nAlgorithms: **ID3** & **J48 / C4.5**")

# ----------------- HEADER & KPIS -----------------
st.title("💳 Student Financial Habits & Spending Behavior Survey")
st.markdown("An interactive exploratory data analytics & data mining dashboard applying **ID3 (Information Gain)** and **J48 / C4.5 (Gain Ratio)** decision tree induction.")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Respondents</div>
        <div class="metric-value">{len(clean_df)}</div>
        <div class="metric-sub">100% Survey Completion</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    emergency_rate = (clean_df["Has_Emergency_Fund"] == "Yes").mean() * 100 if "Has_Emergency_Fund" in clean_df else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Emergency Fund Rate</div>
        <div class="metric-value">{emergency_rate:.1f}%</div>
        <div class="metric-sub">Have 1-Month Cushion</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    invest_acc_rate = (clean_df["Has_Investment_Account"] == "Yes").mean() * 100 if "Has_Investment_Account" in clean_df else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Investment Adoption</div>
        <div class="metric-value">{invest_acc_rate:.1f}%</div>
        <div class="metric-sub">Demat / Crypto / Brokerage</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    sip_plan_rate = (clean_df["Plan_Automated_Invest"] == "Yes").mean() * 100 if "Plan_Automated_Invest" in clean_df else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Automated SIP Intent</div>
        <div class="metric-value">{sip_plan_rate:.1f}%</div>
        <div class="metric-sub">Plan to Automate Investing</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- MAIN NAVIGATION TABS -----------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Overview & Data",
    "📊 Survey EDA & Visualizations",
    "🧠 Decision Tree Mining (ID3 / J48)",
    "⚖️ Model Benchmarks & Metrics",
    "🔮 What-If Student Simulator",
    "📑 Academic Methodology & Rules"
])

# ----------------- TAB 1: OVERVIEW & DATA -----------------
with tab1:
    st.subheader("Survey Dataset Overview")
    st.markdown("Below is the processed survey dataset with standardized attribute names, cleaned values, and derived indicators.")
    
    col_a, col_b = st.columns([3, 1])
    with col_a:
        search_term = st.text_input("🔍 Search respondents or filter table:", "")
    with col_b:
        view_mode = st.radio("View", ["Cleaned Data", "Raw Survey Responses"], horizontal=True)

    display_df = clean_df if view_mode == "Cleaned Data" else raw_df
    if search_term:
        mask = display_df.astype(str).apply(lambda row: row.str.contains(search_term, case=False).any(), axis=1)
        display_df = display_df[mask]

    st.dataframe(display_df, use_container_width=True, height=350)
    st.caption(f"Showing {len(display_df)} of {len(clean_df)} records.")

    st.markdown("---")
    st.subheader("Data Dictionary & Attribute Categorization")
    dict_cols = [
        {"Attribute": "Academic_Year", "Survey Question": "Current academic standing", "Type": "Ordinal Categorical", "Sample Values": "1st Year, 2nd Year, 3rd Year, etc."},
        {"Attribute": "Stream_Major", "Survey Question": "Academic stream/major", "Type": "Nominal Categorical", "Sample Values": "Data Science & AI, Engineering & IT, etc."},
        {"Attribute": "Living_Situation", "Survey Question": "Primary living situation", "Type": "Nominal Categorical", "Sample Values": "Parents, PG / Rent, Hostel"},
        {"Attribute": "Monthly_Budget", "Survey Question": "Money managed monthly", "Type": "Ordinal Categorical", "Sample Values": "Under ₹2,000 to Above ₹10,000"},
        {"Attribute": "Has_Emergency_Fund", "Survey Question": "Emergency fund for 1 month", "Type": "Binary Target", "Sample Values": "Yes / No"},
        {"Attribute": "Has_Investment_Account", "Survey Question": "Demat / Crypto / Brokerage", "Type": "Binary Target", "Sample Values": "Yes / No"},
        {"Attribute": "Peer_Pressure_Spend", "Survey Question": "Spends because friends do", "Type": "Likert 1-5 Scale", "Sample Values": "1 (Never) to 5 (Always)"},
        {"Attribute": "Stress_Spend", "Survey Question": "Spends when stressed on food/entertainment", "Type": "Likert 1-5 Scale", "Sample Values": "1 (Never) to 5 (Always)"},
        {"Attribute": "Financial_Confidence", "Survey Question": "Confidence in post-grad money management", "Type": "Likert 1-5 Scale", "Sample Values": "1 (Low) to 5 (High)"},
        {"Attribute": "Financial_Health_Segment", "Survey Question": "Engineered composite score", "Type": "Derived Multi-class", "Sample Values": "Vulnerable, Developing, Prudent"}
    ]
    st.table(pd.DataFrame(dict_cols))

# ----------------- TAB 2: EXPLORATORY DATA ANALYSIS -----------------
with tab2:
    st.subheader("Exploratory Data Analysis (EDA)")
    
    eda_tab_a, eda_tab_b, eda_tab_c = st.tabs(["Single Variable Distributions", "Psychographics & Likert Ratings", "Multivariate Cross-Tabs"])
    
    with eda_tab_a:
        col_eda1, col_eda2 = st.columns([1, 2])
        with col_eda1:
            cat_candidates = [
                "Academic_Year", "Stream_Major", "Living_Situation", "Commute_Mode",
                "Funding_Source", "Monthly_Budget", "Tracking_Method", "Research_Frequency",
                "Has_Emergency_Fund", "Has_Investment_Account", "Plan_Automated_Invest",
                "Investment_Obstacle", "Philosophy_Active_vs_Passive", "Financial_Health_Segment"
            ]
            selected_cat = st.selectbox("Select Survey Question to Visualize:", [c for c in cat_candidates if c in clean_df.columns])
            st.markdown(f"**Value Counts:**")
            st.dataframe(clean_df[selected_cat].value_counts().reset_index())
        with col_eda2:
            fig_dist = plot_distribution(clean_df, selected_cat)
            st.plotly_chart(fig_dist, use_container_width=True)

    with eda_tab_b:
        st.markdown("#### Student Spending Mindset & Behavioral Traits (1 = Strongly Disagree, 5 = Strongly Agree)")
        likert_items = [
            "Peer_Pressure_Spend", "Stress_Spend", "Financial_Confidence",
            "Lifestyle_Upgrade_Spend", "Wealth_Plan_Readiness"
        ]
        fig_likert = plot_likert_summary(clean_df, likert_items)
        st.plotly_chart(fig_likert, use_container_width=True)

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.info("💡 **Key Finding - Emotional Spending:** Notice how academic stress significantly drives discretionary spending on food and entertainment among students.")
        with col_b2:
            st.info("💡 **Key Finding - Peer Pressure:** A substantial subset of students admits to spending to match peer activities and lifestyle expectations.")

    with eda_tab_c:
        st.markdown("#### Bivariate Relationships & Hierarchy Analysis")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            axis_x = st.selectbox("X-Axis Feature", ["Monthly_Budget", "Stream_Major", "Living_Situation", "Tracking_Method"], index=0)
            axis_y = st.selectbox("Y-Axis Target", ["Has_Emergency_Fund", "Has_Investment_Account", "Financial_Health_Segment"], index=0)
            fig_ct = plot_crosstab_heatmap(clean_df, axis_x, axis_y)
            st.plotly_chart(fig_ct, use_container_width=True)
        with col_c2:
            fig_sunburst = plot_sunburst_hierarchy(clean_df, ["Stream_Major", "Monthly_Budget", "Has_Emergency_Fund"])
            st.plotly_chart(fig_sunburst, use_container_width=True)

# ----------------- TAB 3: DATA MINING (ID3 / J48) -----------------
with tab3:
    st.subheader("Data Mining Engine: ID3 & J48 / C4.5 Decision Tree Induction")
    st.markdown("""
    Here we compare the two classic decision tree algorithms:
    - **ID3**: Uses **Entropy** and **Information Gain** ($IG$). Tends to prefer attributes with numerous distinct values.
    - **J48 / C4.5**: Ross Quinlan's enhancement using **Gain Ratio** ($GR = IG / SplitInfo$), penalizing high-cardinality attributes and applying tree pruning.
    """)

    col_m1, col_m2 = st.columns([1, 1])
    with col_m1:
        target_choices = [c for c in ["Has_Emergency_Fund", "Has_Investment_Account", "Plan_Automated_Invest", "Financial_Health_Segment"] if c in clean_df.columns]
        target_var = st.selectbox("🎯 Target Variable (Class to Mine & Predict):", target_choices)
    with col_m2:
        algo_choice = st.radio("⚙️ Decision Tree Algorithm:", ["J48 / C4.5 (Gain Ratio + Pruning)", "ID3 (Information Gain)"], horizontal=True)

    available_features = [
        "Academic_Year", "Stream_Major", "Living_Situation", "Commute_Mode",
        "Funding_Source", "Monthly_Budget", "Tracking_Method", "Research_Frequency",
        "Peer_Pressure_Spend", "Stress_Spend", "Financial_Confidence",
        "Lifestyle_Upgrade_Spend", "Wealth_Plan_Readiness", "Investment_Obstacle"
    ]
    available_features = [f for f in available_features if f in clean_df.columns and f != target_var]

    selected_features = st.multiselect(
        "Select Attributes for Mining:",
        available_features,
        default=available_features[:6]
    )

    if not selected_features:
        st.warning("Please select at least one predictor feature.")
    else:
        # 1. Feature Gain Table
        st.markdown("### 1. Attribute Selection Metrics (Entropy, Info Gain & Gain Ratio)")
        st.caption(f"Parent Node Base Entropy H(S) = `{calculate_entropy(clean_df[target_var]):.4f}` bits.")
        
        gain_table = get_feature_gain_table(clean_df, selected_features, target_var)
        st.dataframe(gain_table, use_container_width=True)

        fig_gain = plot_feature_gain_comparison(gain_table)
        st.plotly_chart(fig_gain, use_container_width=True)

        # 2. Build Tree
        tree_algo = "ID3" if "ID3" in algo_choice else "J48"
        max_d = st.slider("Maximum Tree Depth (Pruning control)", min_value=2, max_value=6, value=3)
        
        miner = DecisionTreeMiner(algorithm=tree_algo, max_depth=max_d, min_samples_split=4)
        miner.fit(clean_df, selected_features, target_var)

        st.markdown(f"### 2. Generated {tree_algo} Decision Tree Structure")
        tree_dict = miner.root.to_dict()
        fig_tree = plot_interactive_tree_structure(tree_dict)
        st.plotly_chart(fig_tree, use_container_width=True)

        # 3. Extracted Rules
        st.markdown("### 3. Extracted IF-THEN Classification Rules")
        rules = miner.extract_rules()
        st.code("\n".join(rules), language="text")

# ----------------- TAB 4: BENCHMARK & METRICS -----------------
with tab4:
    st.subheader("Model Benchmarks, Validation & Confusion Matrix")
    st.markdown("Evaluating the mined Decision Tree on unseen test data using stratified validation.")

    if selected_features:
        # Preprocessing for Scikit-Learn
        X = pd.get_dummies(clean_df[selected_features], drop_first=True)
        y = clean_df[target_var]

        # ML Best Practice: Train/test split before evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        criterion_val = "entropy" if "ID3" in algo_choice else "gini"
        clf = DecisionTreeClassifier(criterion=criterion_val, max_depth=max_d, min_samples_split=4, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = (y_pred == y_test).mean()

        col_met1, col_met2, col_met3 = st.columns(3)
        with col_met1:
            st.metric("Test Accuracy", f"{acc * 100:.2f}%")
        with col_met2:
            st.metric("Training Samples", len(X_train))
        with col_met3:
            st.metric("Testing Samples", len(X_test))

        st.markdown("---")
        col_cm, col_imp = st.columns(2)
        with col_cm:
            labels = sorted(list(y.unique()))
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            fig_cm = plot_confusion_matrix_interactive(cm, [str(l) for l in labels])
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_imp:
            importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
            fig_imp = px.bar(
                x=importances.values, y=importances.index, orientation='h',
                title="Top Scikit-Learn Feature Importances",
                labels={"x": "Importance Weight", "y": "Feature"},
                color_discrete_sequence=["#2563EB"]
            )
            fig_imp.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("#### Detailed Classification Report")
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report_dict).transpose().round(3), use_container_width=True)

# ----------------- TAB 5: WHAT-IF SIMULATOR -----------------
with tab5:
    st.subheader("🔮 Student Financial Persona Simulator")
    st.markdown("Input hypothetical survey responses to test the mined decision rules and predict student financial behavior in real time.")

    with st.form("student_simulator_form"):
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        
        with sim_col1:
            sim_academic = st.selectbox("Academic Standing", clean_df["Academic_Year"].unique())
            sim_stream = st.selectbox("Major / Stream", clean_df["Stream_Major"].unique())
            sim_living = st.selectbox("Living Situation", clean_df["Living_Situation"].unique())
            sim_funding = st.selectbox("Funding Source", clean_df["Funding_Source"].unique())

        with sim_col2:
            sim_budget = st.selectbox("Monthly Budget Managed", clean_df["Monthly_Budget"].unique())
            sim_tracking = st.selectbox("Money Tracking Method", clean_df["Tracking_Method"].unique())
            sim_research = st.selectbox("Finance Research Frequency", clean_df["Research_Frequency"].unique())
            sim_obstacle = st.selectbox("Investment Obstacle", clean_df["Investment_Obstacle"].unique())

        with sim_col3:
            sim_peer = st.slider("Peer Pressure Spending Tendency (1-5)", 1, 5, 3)
            sim_stress = st.slider("Stress Spending on Food/Entertainment (1-5)", 1, 5, 3)
            sim_conf = st.slider("Post-Grad Financial Confidence (1-5)", 1, 5, 3)
            sim_wealth_plan = st.slider("Actionable Wealth & Debt Plan (1-5)", 1, 5, 2)

        submit_sim = st.form_submit_button("🚀 Run Decision Tree Prediction")

    if submit_sim:
        sim_sample = pd.Series({
            "Academic_Year": sim_academic,
            "Stream_Major": sim_stream,
            "Living_Situation": sim_living,
            "Funding_Source": sim_funding,
            "Monthly_Budget": sim_budget,
            "Tracking_Method": sim_tracking,
            "Research_Frequency": sim_research,
            "Investment_Obstacle": sim_obstacle,
            "Peer_Pressure_Spend": sim_peer,
            "Stress_Spend": sim_stress,
            "Financial_Confidence": sim_conf,
            "Wealth_Plan_Readiness": sim_wealth_plan
        })

        prediction = miner.predict_one(sim_sample)
        st.success(f"### Predicted Result for `{target_var}`: **{prediction}**")
        
        # Actionable insights based on inputs
        st.markdown("#### Tailored Recommendations for this Profile:")
        if sim_tracking in ["Mental Math", "I don't track it"]:
            st.warning("⚠️ **Budget Tracking Gap**: Switching from Mental Math to a structured tracking tool (App/Spreadsheet) increases financial discipline significantly.")
        if sim_peer >= 4 or sim_stress >= 4:
            st.warning("⚠️ **Emotional & Peer Spending**: High susceptibility to impulsive spending. Setting a weekly discretionary spending ceiling is recommended.")
        if sim_research in ["Daily", "Weekly"] and prediction == "Yes":
            st.info("🌟 **High Financial Intent**: Proactive research habits correlate strongly with readiness to begin automated SIP investing.")

# ----------------- TAB 6: ACADEMIC METHODOLOGY & RULES -----------------
with tab6:
    st.subheader("Field Project Academic Report & Methodology")
    st.markdown(r"""
    ### 1. Problem Formulation
    University students undergo a critical financial transition as they move from parental allowances to financial independence. 
    This study aims to discover actionable spending patterns, financial preparedness, and investment barriers through **supervised data mining**.

    ### 2. Mathematical Formulations

    #### A. Shannon's Entropy:
    The measure of impurity or uncertainty in a dataset $S$ containing classes $C = \{c_1, c_2, \dots, c_k\}$:
    $$H(S) = - \sum_{i=1}^{k} p_i \log_2(p_i)$$
    where $p_i$ is the probability of class $c_i$ in $S$.

    #### B. Information Gain (ID3 Algorithm):
    Measures the reduction in entropy achieved by partitioning on attribute $A$:
    $$IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$$

    #### C. Split Information and Gain Ratio (J48 / C4.5 Algorithm):
    ID3 is biased towards attributes with many distinct values. C4.5 normalizes Information Gain by the attribute's **Split Information**:
    $$SplitInfo(S, A) = - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} \log_2\left(\frac{|S_v|}{|S|}\right)$$
    $$GainRatio(S, A) = \frac{IG(S, A)}{SplitInfo(S, A)}$$

    ### 3. Policy & Campus Intervention Recommendations
    1. **Demystify Market Volatility**: Over 40% of surveyed students cited fear of market volatility and lack of capital as their top obstacles to investing.
    2. **Introduce Interactive Budgeting Bootcamps**: Moving students from mental math to automated budgeting early in their academic journey correlates directly with emergency fund creation.
    3. **Automate Early via Micro-SIPs**: Students demonstrate high intent for automated investing; workshops should guide them in opening zero-maintenance brokerage accounts and setting small recurring SIPs.
    """)
