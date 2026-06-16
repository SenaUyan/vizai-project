# C:\project\vizai_project\app.py
from __future__ import annotations

from io import StringIO

import pandas as pd
import streamlit as st

from agent import VizAIAgent


def read_uploaded_csv(uploaded_file) -> pd.DataFrame | None:
    raw_bytes = uploaded_file.getvalue()

    for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            return pd.read_csv(StringIO(raw_bytes.decode(encoding)))
        except Exception:
            continue

    return None


def describe_action(action: dict, target_column: str | None) -> str:
    action_type = action["type"]

    if action_type == "missing_report":
        return "Check missing values and surface data quality issues."

    if action_type == "numeric_histogram":
        column = action["column"]
        if target_column and column == target_column:
            return f"Analyze the distribution of the primary target '{column}'."
        return f"Analyze the distribution of numeric feature '{column}'."

    if action_type == "categorical_bar":
        column = action["column"]
        if target_column:
            return f"Inspect the category balance of '{column}' to compare it with the target '{target_column}'."
        return f"Inspect the category balance of '{column}'."

    if action_type == "scatter_plot":
        x_column = action["x_column"]
        y_column = action["y_column"]
        return f"Measure the numeric relationship between '{x_column}' and '{y_column}'."

    if action_type == "correlation_heatmap":
        columns = action["columns"]
        return f"Summarize pairwise relationships among key numeric columns: {', '.join(columns)}."

    if action_type == "boxplot_by_category":
        numeric_column = action["numeric_column"]
        categorical_column = action["categorical_column"]
        return f"Compare how '{numeric_column}' changes across categories of '{categorical_column}'."

    if action_type == "time_series":
        datetime_column = action["datetime_column"]
        numeric_column = action["numeric_column"]
        return f"Track how '{numeric_column}' changes over time using '{datetime_column}'."

    return f"Run analysis step: {action_type}"


def build_report(result: dict) -> str:
    profile = result["profile"]
    evaluation = result["evaluation"]
    target_column = result.get("target_column")

    lines: list[str] = [
        "VizAI Analysis Report",
        "",
        "Dataset Summary",
        f"- Rows: {profile.row_count}",
        f"- Columns: {profile.column_count}",
        f"- Usable numeric columns: {len(profile.usable_numeric_columns)}",
        f"- Usable categorical columns: {len(profile.usable_categorical_columns)}",
        f"- Datetime columns: {len(profile.datetime_columns)}",
        f"- Primary target: {target_column if target_column else 'Not selected'}",
        "",
    ]

    if profile.warnings:
        lines.append("Warnings")
        lines.extend([f"- {warning}" for warning in profile.warnings])
        lines.append("")

    if result["actions"]:
        lines.append("Why These Charts Were Chosen")
        for action in result["actions"]:
            lines.append(f"- {describe_action(action, target_column)}")
        lines.append("")

    if result["strong_correlations"]:
        lines.append("Strong Correlations")
        for col_a, col_b, value in result["strong_correlations"][:5]:
            lines.append(f"- {col_a} vs {col_b}: {value}")
        lines.append("")

    lines.append("Insights")
    lines.extend([f"- {insight}" for insight in result["insights"]])
    lines.append("")

    lines.append("Evaluation")
    lines.append(f"- Score: {evaluation['score']} / 100")
    lines.extend([f"- {reason}" for reason in evaluation["reasons"]])

    if evaluation["missed_criteria"]:
        lines.append("")
        lines.append("Missed Criteria")
        lines.extend([f"- {item}" for item in evaluation["missed_criteria"]])

    lines.append("")
    lines.append("Agent Log")
    lines.extend([f"- {log}" for log in result["logs"]])

    return "\n".join(lines)


st.set_page_config(page_title="VizAI", page_icon="📊", layout="wide")

st.title("📊 VizAI")
st.caption("Agentic Data Visualization and Insight Generation System")

st.markdown(
    """
Upload a CSV file. VizAI will inspect the dataset, select a primary analysis target when possible,
decide which analysis steps to run, generate focused visualizations, produce insights, and evaluate the analysis quality.
"""
)

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Start by uploading a CSV file.")
    st.stop()

dataframe = read_uploaded_csv(uploaded_file)

if dataframe is None:
    st.error("The CSV file could not be read with common encodings.")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(dataframe.head(10), use_container_width=True)

if st.button("Run VizAI Analysis", type="primary"):
    try:
        agent = VizAIAgent()
        result = agent.analyze(dataframe)
        profile = result["profile"]
        evaluation = result["evaluation"]
        target_column = result.get("target_column")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", profile.row_count)
        col2.metric("Columns", profile.column_count)
        col3.metric("Usable Numeric", len(profile.usable_numeric_columns))
        col4.metric("Usable Categorical", len(profile.usable_categorical_columns))

        st.subheader("Primary Target")
        if target_column:
            st.success(f"VizAI selected '{target_column}' as the main analysis target.")
        else:
            st.info("VizAI did not select a single primary target for this dataset.")

        if profile.warnings:
            st.subheader("Profiling Warnings")
            for warning in profile.warnings:
                st.warning(warning)

        st.subheader("Why These Charts Were Chosen")
        if result["actions"]:
            for index, action in enumerate(result["actions"], start=1):
                st.write(f"{index}. {describe_action(action, target_column)}")
        else:
            st.info("No analysis actions were selected.")

        with st.expander("Agent Decision Log", expanded=True):
            for log in result["logs"]:
                st.write(f"- {log}")

        with st.expander("Detected Column Types", expanded=False):
            st.write("**Usable Numeric Columns:**", profile.usable_numeric_columns or "None")
            st.write("**Usable Categorical Columns:**", profile.usable_categorical_columns or "None")
            st.write("**Datetime Columns:**", profile.datetime_columns or "None")
            st.write("**Identifier Columns:**", profile.identifier_columns or "None")
            st.write("**Constant Columns:**", profile.constant_columns or "None")
            st.write("**Mostly Missing Columns:**", profile.mostly_missing_columns or "None")
            st.write("**High Cardinality Columns:**", profile.high_cardinality_columns or "None")

        st.subheader("Missing Values")
        st.dataframe(result["missing_df"], use_container_width=True)

        st.subheader("Summary Statistics")
        if result["stats_df"].empty:
            st.info("No usable numeric columns found.")
        else:
            st.dataframe(result["stats_df"], use_container_width=True)

        st.subheader("Strong Correlations")
        if result["strong_correlations"]:
            correlation_df = pd.DataFrame(
                result["strong_correlations"],
                columns=["column_a", "column_b", "correlation"],
            )
            st.dataframe(correlation_df, use_container_width=True)
        else:
            st.info("No strong correlations were detected above the threshold.")

        st.subheader("Generated Charts")
        if not result["charts"]:
            st.info("No charts were generated for this dataset.")
        else:
            for chart in result["charts"]:
                st.markdown(f"**{chart['title']}**")
                st.pyplot(chart["figure"], use_container_width=True)

        st.subheader("Insights")
        for insight in result["insights"]:
            st.write(f"- {insight}")

        st.subheader("Evaluation")
        st.metric("Analysis Score", f"{evaluation['score']} / 100")
        st.write(
            f"Points: {evaluation['achieved_points']} / {evaluation['total_possible_points']}"
        )

        for reason in evaluation["reasons"]:
            st.write(f"- {reason}")

        if evaluation["missed_criteria"]:
            st.markdown("**Missed Criteria**")
            for item in evaluation["missed_criteria"]:
                st.write(f"- {item}")

        report_text = build_report(result)
        st.download_button(
            label="Download Analysis Report",
            data=report_text,
            file_name="vizai_analysis_report.txt",
            mime="text/plain",
        )

    except Exception as exc:
        st.error(f"An unexpected error occurred during analysis: {exc}")