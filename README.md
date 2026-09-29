# 📊 Student Financial Habits & Spending Behavior - Data Mining Dashboard

An academic field research project and interactive web dashboard developed with **Python, Streamlit, Plotly, and Scikit-Learn**. 
This application provides in-depth exploratory data analysis (EDA) and implements supervised decision tree mining using **ID3 (Information Gain)** and **J48 / C4.5 (Gain Ratio)** algorithms.

---

## 🚀 Quick Start (1-Click Run)

To run the dashboard:
1. Double-click `run_dashboard.bat` inside `D:\Survey_Data_Mining_Dashboard\`
   **OR**
2. In PowerShell / Terminal, run:
   ```powershell
   cd D:\Survey_Data_Mining_Dashboard
   .\.venv\Scripts\activate
   streamlit run app.py
   ```
3. Open your browser at `http://localhost:8501`.

---

## 📂 Project Architecture

```
D:\Survey_Data_Mining_Dashboard\
├── data\                                # Survey data responses (.csv)
│   └── Student Financial Habits...csv   # Google Forms Survey Responses
├── modules\                             # Backend Data Mining & Charting Engines
│   ├── data_processor.py                # Cleaning, Likert scales, feature engineering
│   ├── decision_trees.py                # Pure ID3 & C4.5/J48 algorithms + Rule Extractor
│   └── visualizations.py                # Interactive Plotly figures & tree network
├── .venv\                               # Dedicated Python 3.14 virtual environment
├── app.py                               # Multi-tab interactive Streamlit web dashboard
├── run_dashboard.bat                    # 1-Click Windows execution script
├── requirements.txt                     # Pinned project dependencies
└── README.md                            # Project documentation & Academic report
```

---

## 🧠 Data Mining Algorithms Implemented

### 1. Shannon's Entropy
Measures data impurity and uncertainty in target variable $S$ across classes $C$:
$$H(S) = - \sum_{i=1}^{k} p_i \log_2(p_i)$$

### 2. ID3 (Iterative Dichotomiser 3)
Partitions attributes based on **Information Gain ($IG$)**:
$$IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$$
- Selects the attribute with the maximum Information Gain at each node.
- Works best with discrete, categorical survey questions.

### 3. J48 / C4.5 (Ross Quinlan's Algorithm)
Penalizes high-cardinality attributes by calculating **Split Information ($SplitInfo$)** and optimizing the **Gain Ratio ($GR$)**:
$$SplitInfo(S, A) = - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} \log_2\left(\frac{|S_v|}{|S|}\right)$$
$$GainRatio(S, A) = \frac{IG(S, A)}{SplitInfo(S, A)}$$
- Mitigates overfitting through depth constraints and sample pruning.
- Produces clean, actionable `IF-THEN` classification rules.

---

## 🖥️ Dashboard Features

1. **Overview & Data Dictionary**: KPI summary cards, raw vs. cleaned data explorer, and data dictionary.
2. **Exploratory Data Analysis (EDA)**: Single-variable distributions, 5-point Likert psychographic rating summaries, bivariate cross-tabs, and student profile sunburst charts.
3. **Data Mining Studio**: Step-by-step feature ranking table (Entropy, Information Gain, Split Info, Gain Ratio), interactive tree diagram, and rule extraction.
4. **Model Benchmarks & Metrics**: Train/Test split evaluation, confusion matrix heatmap, and Scikit-Learn feature importances.
5. **What-If Student Simulator**: Interactive input form to simulate and predict any student persona's financial behavior in real-time.
6. **Academic Report & Methodology**: Formulations and campus intervention strategies.
