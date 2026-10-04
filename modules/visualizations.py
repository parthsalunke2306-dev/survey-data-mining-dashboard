"""
Visualization Module using Plotly for Field Survey Analytics & Data Mining
Styled with a clean slate/white/soft-neutral palette with restrained emerald/green accents (#10B981).
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Restrained, elegant professional palette: slate, emerald/green, subtle blues and grays
THEME_COLORS = ["#10B981", "#3B82F6", "#64748B", "#F59E0B", "#8B5CF6", "#06B6D4", "#EF4444", "#94A3B8"]
EMERALD = "#10B981"
SLATE_DARK = "#1E293B"
SLATE_LIGHT = "#F8FAFC"
BORDER_GRAY = "#E2E8F0"

def get_base_layout(title: str = "") -> dict:
    """Standardized clean styling dictionary for Plotly charts."""
    return dict(
        title=dict(text=title, font=dict(family="Plus Jakarta Sans, sans-serif", size=14, color=SLATE_DARK, weight=600)),
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#334155", size=12),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(t=45, b=35, l=45, r=30),
        xaxis=dict(gridcolor="#F1F5F9", zeroline=False),
        yaxis=dict(gridcolor="#F1F5F9", zeroline=False),
    )

def plot_donut_chart(df: pd.DataFrame, column: str, title: str = None) -> go.Figure:
    """Generates an elegant Donut chart for categorical dimensions."""
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, "Count"]
    fig = go.Figure(data=[
        go.Pie(
            labels=counts[column].tolist(),
            values=counts["Count"].tolist(),
            hole=0.55,
            marker=dict(colors=THEME_COLORS),
            textinfo="label+percent",
            hoverinfo="label+value+percent"
        )
    ])
    layout = get_base_layout(title or f"Distribution of {column.replace('_', ' ')}")
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig

def plot_bar_chart(df: pd.DataFrame, column: str, title: str = None, horizontal: bool = False, order: List[str] = None) -> go.Figure:
    """Generates a clean bar chart with count labels."""
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, "Count"]
    if order:
        counts[column] = pd.Categorical(counts[column], categories=order, ordered=True)
        counts = counts.sort_values(column).dropna()

    if horizontal:
        fig = go.Figure(go.Bar(
            y=counts[column].tolist(), x=counts["Count"].tolist(),
            orientation="h",
            marker=dict(color=EMERALD),
            text=counts["Count"].tolist(),
            textposition="outside"
        ))
        layout = get_base_layout(title or f"{column.replace('_', ' ')} Breakdown")
        layout["xaxis"]["title"] = "Respondents"
    else:
        fig = go.Figure(go.Bar(
            x=counts[column].tolist(), y=counts["Count"].tolist(),
            marker=dict(color=THEME_COLORS[1]),
            text=counts["Count"].tolist(),
            textposition="outside"
        ))
        layout = get_base_layout(title or f"{column.replace('_', ' ')} Breakdown")
        layout["yaxis"]["title"] = "Respondents"

    fig.update_layout(**layout)
    return fig

def plot_stacked_academic_budget(df: pd.DataFrame) -> go.Figure:
    """Stacked bar chart of Academic Year vs Monthly Budget."""
    budget_order = ["Under ₹2,000", "₹2,000 - ₹5,000", "₹5,000 - ₹10,000", "Above ₹10,000"]
    ct = pd.crosstab(df["Academic_Year"], df["Monthly_Budget"])
    # Reindex columns to natural budget order
    cols = [c for c in budget_order if c in ct.columns]
    ct = ct[cols]

    fig = go.Figure()
    palette = ["#94A3B8", "#38BDF8", "#10B981", "#059669"]
    for i, col in enumerate(cols):
        fig.add_trace(go.Bar(
            name=col,
            x=ct.index.tolist(),
            y=ct[col].tolist(),
            marker_color=palette[i % len(palette)]
        ))
    layout = get_base_layout("Academic Year vs Monthly Budget Managed")
    layout["barmode"] = "stack"
    layout["yaxis"]["title"] = "Respondents"
    layout["legend"] = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    fig.update_layout(**layout)
    return fig

def plot_multiselect_breakdown(counts_dict: Dict[str, int], total_n: int, title: str, xlabel: str = "Respondents") -> go.Figure:
    """Horizontal bar chart for multi-select questions with percentage annotation."""
    sorted_items = sorted(counts_dict.items(), key=lambda x: x[1])
    labels = [str(k) for k, v in sorted_items]
    vals = [int(v) for k, v in sorted_items]
    pcts = [f"{v/total_n*100:.1f}% ({v})" for v in vals]

    fig = go.Figure(go.Bar(
        y=labels, x=vals,
        orientation="h",
        marker=dict(color=EMERALD),
        text=pcts,
        textposition="outside"
    ))
    layout = get_base_layout(title)
    layout["xaxis"]["title"] = xlabel
    layout["margin"]["r"] = 60
    # Add footnote note that percentages can exceed 100%
    layout["annotations"] = [dict(
        x=0.5, y=-0.22, xref="paper", yref="paper",
        text="* Note: Percentages exceed 100% collectively as respondents could select multiple categories.",
        showarrow=False, font=dict(size=10, color="#64748B")
    )]
    fig.update_layout(**layout)
    return fig

def plot_comparative_rate(df: pd.DataFrame, group_col: str, target_col: str, title: str) -> go.Figure:
    """Grouped percentage rate comparison."""
    rate_df = df.groupby(group_col)[target_col].apply(
        lambda s: (s == "Yes").mean() * 100
    ).reset_index()
    rate_df.columns = [group_col, "Emergency_Fund_Rate"]

    fig = go.Figure(go.Bar(
        x=rate_df[group_col].tolist(),
        y=[float(round(r, 1)) for r in rate_df["Emergency_Fund_Rate"]],
        marker_color="#3B82F6",
        text=[f"{r:.1f}%" for r in rate_df["Emergency_Fund_Rate"]],
        textposition="outside"
    ))
    layout = get_base_layout(title)
    layout["yaxis"]["title"] = "Emergency Fund Coverage (%)"
    layout["yaxis"]["range"] = [0, 100]
    fig.update_layout(**layout)
    return fig

def plot_research_vs_confidence(df: pd.DataFrame) -> go.Figure:
    """Bar chart of average financial confidence score across research frequency tiers."""
    order = ["Rarely / Never", "Monthly", "Weekly", "Daily"]
    agg = df.groupby("Research_Frequency")["Financial_Confidence"].agg(["mean", "count"]).reset_index()
    agg["Research_Frequency"] = pd.Categorical(agg["Research_Frequency"], categories=order, ordered=True)
    agg = agg.sort_values("Research_Frequency").dropna()

    fig = go.Figure(go.Bar(
        x=agg["Research_Frequency"].tolist(),
        y=[float(round(m, 2)) for m in agg["mean"]],
        marker_color=EMERALD,
        text=[f"{m:.2f} / 5 (n={int(c)})" for m, c in zip(agg["mean"], agg["count"])],
        textposition="outside"
    ))
    layout = get_base_layout("Financial Research Frequency vs Average Confidence")
    layout["yaxis"]["title"] = "Mean Confidence (1-5)"
    layout["yaxis"]["range"] = [0, 5.5]
    fig.update_layout(**layout)
    return fig

def plot_peer_vs_stress_correlation(df: pd.DataFrame) -> go.Figure:
    """2D frequency bubble chart comparing Peer Pressure Spending vs Stress Spending."""
    ct = pd.crosstab(df["Peer_Pressure_Spend"], df["Stress_Spend"]).reset_index()
    melted = ct.melt(id_vars="Peer_Pressure_Spend", var_name="Stress_Spend", value_name="Count")
    melted = melted[melted["Count"] > 0]

    # Calculate Spearman correlation via ranks (pure Pandas/NumPy, no external scipy needed)
    rho = float(df["Peer_Pressure_Spend"].rank().corr(df["Stress_Spend"].rank()))

    fig = go.Figure(go.Scatter(
        x=melted["Stress_Spend"].tolist(),
        y=melted["Peer_Pressure_Spend"].tolist(),
        mode="markers+text",
        marker=dict(
            size=[int(c * 5 + 10) for c in melted["Count"]],
            color=melted["Count"].tolist(),
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Respondents")
        ),
        text=melted["Count"].tolist(),
        textposition="middle center",
        textfont=dict(color="white", size=10, weight="bold")
    ))
    layout = get_base_layout(f"Peer Pressure vs Emotional/Stress Spending (Spearman ρ = +{rho:.3f})")
    layout["xaxis"]["title"] = "Stress-Related Spending Rating (1 = Disagree, 5 = Agree)"
    layout["yaxis"]["title"] = "Peer-Influenced Spending Rating (1 = Disagree, 5 = Agree)"
    layout["xaxis"]["dtick"] = 1
    layout["yaxis"]["dtick"] = 1
    fig.update_layout(**layout)
    return fig

def plot_likert_diverging(df: pd.DataFrame, likert_columns: List[str]) -> go.Figure:
    """Diverging stacked bar chart for 1-5 Likert scales."""
    data = []
    labels_map = {
        "Peer_Pressure_Spend": "Peer-Influenced Spending",
        "Stress_Spend": "Academic Stress Spending",
        "Financial_Confidence": "Future Financial Confidence",
        "Lifestyle_Upgrade_Spend": "Tech/Style Upgrade Habit",
        "Wealth_Plan_Readiness": "Actionable 3-Year Wealth Plan"
    }
    for col in likert_columns:
        if col in df.columns:
            counts = df[col].value_counts(normalize=True).sort_index() * 100
            for rating in range(1, 6):
                pct = float(round(counts.get(rating, 0.0), 1))
                data.append({"Question": labels_map.get(col, col), "Rating": f"Score {rating}", "Percentage": pct})
    
    plot_df = pd.DataFrame(data)
    fig = px.bar(
        plot_df, y="Question", x="Percentage", color="Rating", orientation="h",
        color_discrete_sequence=["#EF4444", "#F97316", "#CBD5E1", "#34D399", "#10B981"],
        title="Mindset & Behavioral Likert Distribution (1 = Strongly Disagree to 5 = Strongly Agree)"
    )
    layout = get_base_layout("Mindset & Behavioral Likert Distribution")
    layout["barmode"] = "stack"
    layout["xaxis"]["title"] = "Percentage of Respondents (%)"
    layout["yaxis"]["title"] = ""
    layout["margin"]["l"] = 180
    fig.update_layout(**layout)
    return fig

def plot_confidence_planning_matrix(df: pd.DataFrame) -> go.Figure:
    """2x2 matrix bar chart for Confidence vs Planning Segmentation."""
    counts = df["Confidence_Planning_Segment"].value_counts().reset_index()
    counts.columns = ["Segment", "Count"]
    n = len(df)
    counts["Pct"] = (counts["Count"] / n * 100).round(1)

    colors = {
        "Prudent Strategists (High Conf + Action Plan)": "#10B981",
        "Overconfident Optimists (High Conf + No Action Plan)": "#F59E0B",
        "Cautious Planners (Low Conf + Action Plan)": "#3B82F6",
        "Unprepared / At-Risk (Low Conf + No Action Plan)": "#EF4444"
    }

    fig = go.Figure(go.Bar(
        y=counts["Segment"].tolist(),
        x=counts["Count"].tolist(),
        orientation="h",
        marker=dict(color=[colors.get(s, "#64748B") for s in counts["Segment"]]),
        text=[f"{c} students ({p}%)" for c, p in zip(counts["Count"], counts["Pct"])],
        textposition="outside"
    ))
    layout = get_base_layout("Confidence–Planning 2x2 Matrix Segmentation")
    layout["xaxis"]["title"] = "Respondents"
    layout["margin"]["l"] = 240
    layout["margin"]["r"] = 80
    fig.update_layout(**layout)
    return fig

def plot_fdi_distribution(df: pd.DataFrame) -> go.Figure:
    """Financial Discipline Index (FDI) distribution manually binned to avoid rendering issues."""
    fdi = [float(round(v, 1)) for v in df["FDI_Score"].dropna().tolist()] if (len(df) > 0 and "FDI_Score" in df.columns) else []
    mean_val = float(round(sum(fdi) / len(fdi), 1)) if fdi else 0.0

    fig = go.Figure()
    
    if fdi:
        import numpy as np
        counts, bins = np.histogram(fdi, bins=10, range=(0, 100))
        bin_centers = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
        
        fig.add_trace(go.Bar(
            x=[float(x) for x in bin_centers],
            y=[int(c) for c in counts],
            width=9.5,
            marker=dict(color="#10B981", line=dict(color="white", width=1.5)),
            name="FDI Score"
        ))
    
    fig.add_vline(
        x=mean_val, line_dash="dash", line_color="#1E293B", line_width=2,
        annotation_text=f"Mean: {mean_val:.1f}", annotation_position="top right",
        annotation_font=dict(size=11)
    )
    
    # No in-chart title: the card header already names the chart, and a long title gets clipped on phones.
    layout = get_base_layout("")
    layout["xaxis"] = dict(
        gridcolor="#F1F5F9",
        zeroline=False,
        range=[-5, 105],
        dtick=10,
        title=dict(text="FDI score (0–100)", font=dict(size=11), standoff=4),
        tickfont=dict(size=10)
    )
    y_top = (max(int(c) for c in counts) * 1.25) if fdi else 1  # headroom so the mean label doesn't overlap the bars
    layout["yaxis"] = dict(
        gridcolor="#F1F5F9",
        zeroline=False,
        range=[0, y_top],
        title=dict(text="Students", font=dict(size=11), standoff=4),
        tickfont=dict(size=10)
    )
    layout["margin"] = dict(t=12, b=38, l=42, r=12)
    fig.update_layout(**layout)
    return fig

def plot_spearman_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    """Interactive Spearman correlation heatmap."""
    clean_labels = [c.replace("_", " ") for c in corr_df.columns]
    fig = px.imshow(
        corr_df.values,
        x=clean_labels,
        y=clean_labels,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=".2f",
        title="Spearman Rank Correlation Matrix (Behavioral & Planning Variables)"
    )
    layout = get_base_layout("Spearman Rank Correlation Matrix")
    fig.update_layout(**layout)
    return fig

def plot_distribution(df: pd.DataFrame, column: str, title: str = None) -> go.Figure:
    """Backward compatibility wrapper."""
    return plot_bar_chart(df, column, title)

def plot_likert_summary(df: pd.DataFrame, likert_columns: List[str]) -> go.Figure:
    """Backward compatibility wrapper."""
    return plot_likert_diverging(df, likert_columns)

def plot_crosstab_heatmap(df: pd.DataFrame, col_x: str, col_y: str) -> go.Figure:
    ct = pd.crosstab(df[col_y], df[col_x])
    fig = px.imshow(
        ct, text_auto=True, aspect="auto",
        color_continuous_scale="Blues",
        title=f"Cross-Tabulation: {col_y} vs {col_x}"
    )
    layout = get_base_layout(f"Cross-Tabulation: {col_y} vs {col_x}")
    fig.update_layout(**layout)
    return fig

def plot_sunburst_hierarchy(df: pd.DataFrame, path_cols: List[str], target_metric: str = None) -> go.Figure:
    fig = px.sunburst(
        df, path=path_cols,
        title="Student Profile Hierarchy (Stream → Budget → Emergency Fund)",
        color_discrete_sequence=THEME_COLORS
    )
    fig.update_traces(textinfo="label+percent entry")
    layout = get_base_layout("Student Profile Hierarchy")
    fig.update_layout(**layout)
    return fig

def plot_feature_gain_comparison(gain_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=gain_df["Feature"], y=gain_df["Info Gain (ID3)"],
        name="ID3 (Information Gain)", marker_color="#3B82F6"
    ))
    fig.add_trace(go.Bar(
        x=gain_df["Feature"], y=gain_df["Gain Ratio (J48/C4.5)"],
        name="J48 / C4.5 (Gain Ratio)", marker_color=EMERALD
    ))
    layout = get_base_layout("Feature Splitting Power: ID3 Information Gain vs. J48 Gain Ratio")
    layout["barmode"] = "group"
    layout["xaxis"]["title"] = "Survey Feature"
    layout["yaxis"]["title"] = "Score"
    layout["legend"] = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    fig.update_layout(**layout)
    return fig

def plot_interactive_tree_structure(tree_dict: Dict) -> go.Figure:
    """Renders tree structure using Plotly scatter."""
    x_nodes, y_nodes, text_nodes, color_nodes = [], [], [], []
    edge_x, edge_y = [], []

    def _calc_layout(node, x=0.5, y=1.0, dx=0.45, dy=0.25):
        x_nodes.append(x)
        y_nodes.append(y)
        is_leaf = node.get("is_leaf", False)
        if is_leaf:
            text_nodes.append(f"<b>LEAF</b><br>{node.get('prediction')}<br>n={node.get('samples')}")
            color_nodes.append(EMERALD)
        else:
            text_nodes.append(f"<b>{node.get('feature')}</b><br>H={node.get('entropy', 0):.2f}<br>n={node.get('samples')}")
            color_nodes.append("#3B82F6")

        children = node.get("children", [])
        n_children = len(children)
        if n_children > 0:
            step = (dx * 2) / max(n_children - 1, 1) if n_children > 1 else 0
            start_x = x - dx if n_children > 1 else x
            for i, ch in enumerate(children):
                ch_x = start_x + i * step
                ch_y = y - dy
                edge_x.extend([x, ch_x, None])
                edge_y.extend([y, ch_y, None])
                _calc_layout(ch, ch_x, ch_y, dx * 0.45, dy)

    _calc_layout(tree_dict)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1.5, color="#94A3B8"), hoverinfo="none"
    ))
    fig.add_trace(go.Scatter(
        x=x_nodes, y=y_nodes, mode="markers+text",
        marker=dict(size=40, color=color_nodes, line=dict(width=2, color="#1E293B")),
        text=[t.split('<br>')[0].replace('<b>', '').replace('</b>', '') for t in text_nodes],
        textposition="top center", hovertext=text_nodes, hoverinfo="text"
    ))
    layout = get_base_layout("Decision Tree Hierarchy")
    layout["showlegend"] = False
    layout["xaxis"] = dict(showgrid=False, zeroline=False, showticklabels=False)
    layout["yaxis"] = dict(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_layout(**layout)
    return fig
