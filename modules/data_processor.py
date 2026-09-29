"""
Data Preprocessing & Cleaning Module for Student Financial Habits Survey
"""
import re
import pandas as pd
import numpy as np

# Standardized short column names mapping for convenience and clean display
SHORT_NAME_MAP = {
    "Timestamp": "Timestamp",
    "Email address": "Email",
    "Name": "Name",
    "What is your current academic standing?": "Academic_Year",
    "What is your academic stream/major?": "Stream_Major",
    "What is your primary living situation?": "Living_Situation",
    "How do you typically commute to campus?": "Commute_Mode",
    "What is the primary source of your monthly funds?": "Funding_Source",
    "Approximately how much money do you manage monthly (excluding tuition/rent paid directly by parents)?": "Monthly_Budget",
    "How do you primarily track your money?": "Tracking_Method",
    "Do you currently have an emergency fund that can cover one month of expenses?": "Has_Emergency_Fund",
    "Have you already set up a demat account, crypto wallet, or brokerage account to prepare for future wealth building?": "Has_Investment_Account",
    "How often do you actively research or read about personal finance, market trends, or trading strategies?": "Research_Frequency",
    "Which of the following did you spend money on in the last 7 days?": "Recent_Spending",
    '"I frequently spend money because my friends are doing it (e.g., going to a cafe, buying a specific brand)."': "Peer_Pressure_Spend",
    '"When I feel stressed about academics or life, I am more likely to spend money on food or entertainment."': "Stress_Spend",
    '"I feel confident about managing my own money once I graduate and get a full-time job."': "Financial_Confidence",
    '"I like to invest in updating my tech, equipment, or personal style whenever newer options come out."': "Lifestyle_Upgrade_Spend",
    '"I have a clear, actionable plan to manage debt and grow my wealth within the first three years of graduating."': "Wealth_Plan_Readiness",
    "Do you believe relying solely on a traditional salary will be enough to achieve your long-term financial goals?": "Salary_Alone_Enough",
    "Do you plan to automate your future investments (e.g., Systematic Investment Plans) once you secure a full-time income?": "Plan_Automated_Invest",
    "Which of the following asset classes are you most interested in exploring over the next 5 years?": "Asset_Interests",
    "Once you begin earning a full-time income, how often do you foresee yourself reviewing and rebalancing your investment portfolio?": "Portfolio_Review_Freq",
    "What do you consider the most significant obstacle to starting your investment journey?": "Investment_Obstacle",
    "In your opinion, which approach is more critical for long-term financial independence?": "Philosophy_Active_vs_Passive",
    "What specific topics regarding personal finance, market analysis, or investing would you like to see covered in future workshops or campus events?": "Workshop_Interests"
}

def clean_survey_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw Google Forms survey responses:
    - Renames long survey questions to short, readable feature names.
    - Standardizes categorical responses (trims whitespace, unifies cases).
    - Cleans budget strings.
    - Converts Likert scales to numeric values (1 to 5).
    """
    df_clean = df.copy()
    
    # 1. Rename columns if matching
    rename_dict = {}
    for col in df_clean.columns:
        clean_col_key = col.strip()
        if clean_col_key in SHORT_NAME_MAP:
            rename_dict[col] = SHORT_NAME_MAP[clean_col_key]
        else:
            # Fuzzy match without quotes
            stripped_key = clean_col_key.replace('"', '').strip()
            matched = False
            for k, v in SHORT_NAME_MAP.items():
                if stripped_key in k.replace('"', '').strip():
                    rename_dict[col] = v
                    matched = True
                    break
            if not matched:
                # Fallback to a sanitized identifier
                sanitized = re.sub(r'[^\w\s]', '', col)[:30].strip().replace(' ', '_')
                rename_dict[col] = sanitized
                
    df_clean.rename(columns=rename_dict, inplace=True)
    
    # 2. Standardize Stream/Major
    if "Stream_Major" in df_clean.columns:
        df_clean["Stream_Major"] = df_clean["Stream_Major"].astype(str).str.strip().str.title()
        # Merge common variations
        stream_map = {
            "Data Science": "Data Science & AI",
            "Data Science ": "Data Science & AI",
            "Engineering & It": "Engineering & IT",
            "Engineering & It ": "Engineering & IT",
            "Commerce / Management / Finance": "Commerce & Management",
            "Commerce / Management / Finance ": "Commerce & Management",
            "Science": "Pure & Applied Sciences",
            "Arts & Humanities": "Arts & Humanities",
        }
        df_clean["Stream_Major"] = df_clean["Stream_Major"].replace(stream_map)

    # 3. Clean and categorize Monthly Budget
    if "Monthly_Budget" in df_clean.columns:
        df_clean["Monthly_Budget"] = df_clean["Monthly_Budget"].astype(str).str.strip()

    # 4. Clean Likert columns to numeric integers
    likert_cols = [
        "Peer_Pressure_Spend", "Stress_Spend", "Financial_Confidence",
        "Lifestyle_Upgrade_Spend", "Wealth_Plan_Readiness"
    ]
    for col in likert_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(3).astype(int)

    # 5. Clean Binary flags
    binary_cols = ["Has_Emergency_Fund", "Has_Investment_Account", "Salary_Alone_Enough", "Plan_Automated_Invest"]
    for col in binary_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.capitalize()
            df_clean[col] = df_clean[col].apply(lambda x: "Yes" if "yes" in x.lower() else ("No" if "no" in x.lower() else x))

    # 6. Synthesize High-Value Derived Target: Financial Literacy & Discipline Level
    # Low / Medium / High based on readiness, tracking, emergency fund, and investment account
    if all(c in df_clean.columns for c in ["Has_Emergency_Fund", "Has_Investment_Account", "Financial_Confidence"]):
        score = (
            (df_clean["Has_Emergency_Fund"] == "Yes").astype(int) * 2 +
            (df_clean["Has_Investment_Account"] == "Yes").astype(int) * 2 +
            (df_clean["Financial_Confidence"] >= 4).astype(int) * 2 +
            (df_clean["Stress_Spend"] <= 2).astype(int) * 1 +
            (df_clean["Peer_Pressure_Spend"] <= 2).astype(int) * 1
        )
        df_clean["Financial_Health_Segment"] = pd.cut(
            score, bins=[-1, 2, 5, 10], labels=["Vulnerable / Beginner", "Developing", "Prudent / Advanced"]
        )

    # Fill any remaining NaNs (e.g. optional open-ended text questions) with empty string
    df_clean = df_clean.fillna("")

    return df_clean
