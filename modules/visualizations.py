"""
Visualization Module using Plotly for Field Survey Analytics & Decision Trees
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Professional Theme Color Palette
COLORS = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#06B6D4", "#EC4899", "#84CC16"]

def plot_distribution(df: pd.DataFrame, column: str, title: str = None) -> go.Figure:
    """Creates a clean, styled bar chart for categorical distributions."""
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, "Count"]
    fig = px.bar(
        counts, x=column, y="Count",
        color=column,
        color_discrete_sequence=COLORS,
        text="Count",
        title=title or f"Distribution of {column}"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, xaxis_title=column, yaxis_title="Respondents", margin=dict(t=40, b=40, l=40, r=40))
    return fig

def plot_likert_summary(df: pd.DataFrame, likert_columns: List[str]) -> go.Figure:
    """Creates a diverging / stacked bar chart for 1-5 Likert scale survey questions."""
    data = []
    for col in likert_columns:
        if col in df.columns:
            counts = df[col].value_counts(normalize=True).sort_index() * 100
            for rating in range(1, 6):
                pct = counts.get(rating, 0.0)
                data.append({"Question": col.replace('_', ' '), "Rating": f"Score {rating}", "Percentage": pct})
    
    plot_df = pd.DataFrame(data)
    fig = px.bar(
        plot_df, y="Question", x="Percentage", color="Rating", orientation="h",
        color_discrete_sequence=["#EF4444", "#F97316", "#FBBF24", "#34D399", "#10B981"],
        title="Psychographic & Spending Attitude Ratings (1 = Strongly Disagree, 5 = Strongly Agree)"
    )
    fig.update_layout(barmode="stack", xaxis_title="Percentage of Respondents (%)", yaxis_title="", margin=dict(l=150))
    return fig

def plot_crosstab_heatmap(df: pd.DataFrame, col_x: str, col_y: str) -> go.Figure:
    """Generates an interactive heatmap showing cross-tabulation between two survey dimensions."""
    ct = pd.crosstab(df[col_y], df[col_x])
    fig = px.imshow(
        ct, text_auto=True, aspect="auto",
        color_continuous_scale="Blues",
        title=f"Cross-Tabulation: {col_y} vs {col_x}"
    )
    fig.update_layout(xaxis_title=col_x, yaxis_title=col_y)
    return fig

def plot_sunburst_hierarchy(df: pd.DataFrame, path_cols: List[str], target_metric: str = None) -> go.Figure:
    """Generates an exploratory Sunburst hierarchy chart for student profiles."""
    fig = px.sunburst(
        df, path=path_cols,
        title="Student Profile Hierarchy (Stream $\\rightarrow$ Budget $\\rightarrow$ Emergency Fund)",
        color_discrete_sequence=COLORS
    )
    fig.update_traces(textinfo="label+percent entry")
    return fig

def plot_feature_gain_comparison(gain_df: pd.DataFrame) -> go.Figure:
    """Plots ID3 Information Gain vs J48 Gain Ratio side-by-side."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=gain_df["Feature"], y=gain_df["Info Gain (ID3)"],
        name="ID3 (Information Gain)", marker_color="#3B82F6"
    ))
    fig.add_trace(go.Bar(
        x=gain_df["Feature"], y=gain_df["Gain Ratio (J48/C4.5)"],
        name="J48 / C4.5 (Gain Ratio)", marker_color="#10B981"
    ))
    fig.update_layout(
        barmode="group",
        title="Feature Splitting Power: ID3 Information Gain vs. J48 Gain Ratio",
        xaxis_title="Survey Feature",
        yaxis_title="Score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_confusion_matrix_interactive(cm: np.ndarray, labels: List[str]) -> go.Figure:
    """Plots an interactive confusion matrix heatmap."""
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
        x=labels, y=labels,
        text_auto=True,
        color_continuous_scale="Greens",
        title="Decision Tree Confusion Matrix"
    )
    return fig

def plot_interactive_tree_structure(tree_dict: Dict) -> go.Figure:
    """
    Renders an interactive tree diagram using Plotly node-link coordinates,
    providing visual clarity without needing external Graphviz installation.
    """
    x_nodes = []
    y_nodes = []
    text_nodes = []
    color_nodes = []
    edge_x = []
    edge_y = []
    edge_text = []

    def _calc_layout(node, x=0.5, y=1.0, dx=0.45, dy=0.25):
        x_nodes.append(x)
        y_nodes.append(y)
        
        is_leaf = node.get("is_leaf", False)
        if is_leaf:
            text_nodes.append(f"<b>LEAF</b><br>{node.get('prediction')}<br>n={node.get('samples')}")
            color_nodes.append("#10B981")
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
    # Edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1.5, color="#94A3B8"),
        hoverinfo="none"
    ))
    # Nodes
    fig.add_trace(go.Scatter(
        x=x_nodes, y=y_nodes,
        mode="markers+text",
        marker=dict(size=45, color=color_nodes, line=dict(width=2, color="#1E293B")),
        text=[t.split('<br>')[0].replace('<b>', '').replace('</b>', '') for t in text_nodes],
        textposition="top center",
        hovertext=text_nodes,
        hoverinfo="text"
    ))

    fig.update_layout(
        showlegend=False,
        title="Interactive Decision Tree Hierarchy (Hover nodes for details)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(t=50, b=20, l=20, r=20),
        plot_bgcolor="#F8FAFC"
    )
    return fig
