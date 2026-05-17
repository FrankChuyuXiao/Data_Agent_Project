from __future__ import annotations

import pandas as pd
import plotly.express as px
from .llm_utils import call_llm_json

def build_visualization_prompt() -> str:
    return """
You are a visualization decision agent.

Given a user question and an execution result, decide whether a visualization is useful.
Visualization is useful for comparisons, rankings, distributions, grouped numeric values, and trends.
Visualization is not useful for single scalar values or text-only results.

Rules:
- You must use only the execution result, not the raw dataset.
- Return JSON only.
- Valid chart_type values: bar, line, scatter, histogram, pie, none.

Return JSON:
{
  "visualize": true,
  "chart_type": "bar",
  "x": "column_name or null",
  "y": "column_name or null",
  "reason": "short reason"
}
""".strip()


def llm_decide_visualization(question: str, result_df: pd.DataFrame) -> dict | None:
    prompt = build_visualization_prompt()
    sample = result_df.head(10).to_dict(orient="records")
    user_prompt = f"""
Question: {question}
Columns: {list(result_df.columns)}
Sample rows: {sample}
""".strip()
    return call_llm_json(prompt, user_prompt)


def fallback_decision(question: str, result_df: pd.DataFrame) -> dict:
    q = question.lower()
    if result_df.empty or len(result_df.columns) < 2:
        return {"visualize": False, "chart_type": "none", "x": None, "y": None, "reason": "Not enough data."}

    numeric_cols = list(result_df.select_dtypes(include="number").columns)
    non_numeric_cols = [c for c in result_df.columns if c not in numeric_cols]

    wants_plot = any(word in q for word in ["plot", "chart", "visualize", "bar", "graph"])
    comparison_like = len(result_df) > 1 and len(numeric_cols) >= 1

    if not wants_plot and not comparison_like:
        return {"visualize": False, "chart_type": "none", "x": None, "y": None, "reason": "Visualization is not necessary."}

    y = numeric_cols[0]
    x = non_numeric_cols[0] if non_numeric_cols else result_df.columns[0]

    if "hist" in q or "distribution" in q:
        return {"visualize": True, "chart_type": "histogram", "x": y, "y": None, "reason": "A distribution chart is appropriate."}
    if "line" in q or "trend" in q:
        return {"visualize": True, "chart_type": "line", "x": x, "y": y, "reason": "A line chart can show a trend."}
    if "scatter" in q and len(numeric_cols) >= 2:
        return {"visualize": True, "chart_type": "scatter", "x": numeric_cols[0], "y": numeric_cols[1], "reason": "A scatter plot compares two numeric columns."}

    return {"visualize": True, "chart_type": "bar", "x": x, "y": y, "reason": "A bar chart is useful for comparison."}



def create_visualization(question, execution_result):
    q = question.lower()
    df = execution_result.copy()

    decision = {
        "visualize": False,
        "chart_type": None,
        "x": None,
        "y": None,
        "reason": ""
    }

    if df is None or df.empty:
        decision["reason"] = "Empty result."
        return None, decision

    # Do not visualize scalar result
    if df.shape == (1, 1):
        decision["reason"] = "Scalar result does not need visualization."
        return None, decision

    # Special case: top expensive geographic areas
    if "median_house_value" in df.columns and (
        "expensive" in q or "top" in q or "geographic" in q or "area" in q
    ):
        plot_df = df.copy()

        plot_df["area_label"] = plot_df.index.astype(str)

        if "latitude" in plot_df.columns and "longitude" in plot_df.columns:
            plot_df["area_label"] = (
                "Lat "
                + plot_df["latitude"].round(2).astype(str)
                + ", Lon "
                + plot_df["longitude"].round(2).astype(str)
            )

        plot_df = plot_df.sort_values("median_house_value", ascending=False).head(5)

        fig = px.bar(
            plot_df,
            x="area_label",
            y="median_house_value",
            title="Top 5 Most Expensive Geographic Areas",
            hover_data=[
                col for col in [
                    "ocean_proximity",
                    "median_income",
                    "housing_median_age",
                    "population"
                ]
                if col in plot_df.columns
            ]
        )

        decision.update({
            "visualize": True,
            "chart_type": "bar",
            "x": "area_label",
            "y": "median_house_value",
            "reason": "Top geographic areas with numeric house values can be compared with a bar chart."
        })

        return fig.to_html(full_html=False, include_plotlyjs="cdn"), decision

    # General case: first categorical/object column + first numeric column
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    if categorical_cols and numeric_cols:
        x_col = categorical_cols[0]
        y_col = numeric_cols[0]

        fig = px.bar(
            df,
            x=x_col,
            y=y_col,
            title=question
        )

        decision.update({
            "visualize": True,
            "chart_type": "bar",
            "x": x_col,
            "y": y_col,
            "reason": "Categorical and numeric columns are suitable for a bar chart."
        })

        return fig.to_html(full_html=False, include_plotlyjs="cdn"), decision

    # General case: multiple numeric columns
    if len(numeric_cols) >= 2:
        x_col = numeric_cols[0]
        y_col = numeric_cols[-1]

        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            title=question
        )

        decision.update({
            "visualize": True,
            "chart_type": "scatter",
            "x": x_col,
            "y": y_col,
            "reason": "Multiple numeric columns are suitable for a scatter plot."
        })

        return fig.to_html(full_html=False, include_plotlyjs="cdn"), decision

    decision["reason"] = "No suitable columns for visualization."
    return None, decision
