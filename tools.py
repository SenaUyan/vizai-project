# tools.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class DatasetProfile:
    row_count: int
    column_count: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    identifier_columns: list[str]
    constant_columns: list[str]
    high_cardinality_columns: list[str]
    mostly_missing_columns: list[str]
    usable_numeric_columns: list[str]
    usable_categorical_columns: list[str]
    warnings: list[str]


def _make_unique_column_names(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    unique_columns: list[str] = []

    for column in columns:
        base = column.strip() or "unnamed_column"
        if base not in seen:
            seen[base] = 0
            unique_columns.append(base)
            continue

        seen[base] += 1
        unique_columns.append(f"{base}_{seen[base]}")

    return unique_columns


def safe_copy(df: pd.DataFrame) -> pd.DataFrame:
    copied = df.copy()
    copied.columns = _make_unique_column_names([str(column) for column in copied.columns])
    copied = copied.replace([np.inf, -np.inf], np.nan)
    return copied


def _is_datetime_candidate(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False

    as_text = non_null.astype(str).str.strip()
    if as_text.empty:
        return False

    separator_ratio = as_text.str.contains(r"[-/:T ]", regex=True).mean()
    letter_ratio = as_text.str.contains(r"[A-Za-z]", regex=True).mean()

    return bool(separator_ratio >= 0.5 or letter_ratio >= 0.3)


def try_parse_datetime_columns(
    df: pd.DataFrame,
    threshold: float = 0.8,
) -> tuple[pd.DataFrame, list[str]]:
    parsed_df = df.copy()
    datetime_columns: list[str] = []

    for column in parsed_df.columns:
        series = parsed_df[column]

        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_columns.append(column)
            continue

        if not (
            pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
        ):
            continue

        non_null = series.dropna()
        if len(non_null) < 3:
            continue

        if not _is_datetime_candidate(non_null):
            continue

        converted = pd.to_datetime(non_null, errors="coerce")
        success_ratio = converted.notna().mean()

        if success_ratio >= threshold:
            parsed_df[column] = pd.to_datetime(parsed_df[column], errors="coerce")
            datetime_columns.append(column)

    return parsed_df, datetime_columns


def _is_identifier_column(series: pd.Series, column_name: str, row_count: int) -> bool:
    non_null = series.dropna()
    if non_null.empty or row_count == 0:
        return False

    lowered_name = column_name.lower()
    name_hint = (
        lowered_name == "id"
        or lowered_name.endswith("_id")
        or lowered_name.startswith("id_")
        or "identifier" in lowered_name
        or "uuid" in lowered_name
        or "guid" in lowered_name
    )

    unique_ratio = non_null.nunique(dropna=True) / len(non_null)
    mostly_unique = unique_ratio >= 0.95 and len(non_null) >= min(10, row_count)

    if pd.api.types.is_numeric_dtype(series):
        numeric_non_null = pd.to_numeric(non_null, errors="coerce").dropna()
        is_integer_like = (
            not numeric_non_null.empty
            and np.allclose(numeric_non_null, numeric_non_null.round())
        )
        is_monotonic = (
            numeric_non_null.is_monotonic_increasing
            or numeric_non_null.is_monotonic_decreasing
        )
        if name_hint and mostly_unique:
            return True
        if mostly_unique and is_integer_like and is_monotonic:
            return True
        return False

    if name_hint and mostly_unique:
        return True

    return False


def infer_column_types(df: pd.DataFrame) -> DatasetProfile:
    working_df = safe_copy(df)
    working_df, datetime_columns = try_parse_datetime_columns(working_df)

    row_count = len(working_df)
    column_count = len(working_df.columns)

    raw_numeric_columns = [
        column
        for column in working_df.columns
        if pd.api.types.is_numeric_dtype(working_df[column]) and column not in datetime_columns
    ]

    raw_categorical_columns = [
        column
        for column in working_df.columns
        if column not in raw_numeric_columns and column not in datetime_columns
    ]

    identifier_columns = [
        column
        for column in working_df.columns
        if _is_identifier_column(working_df[column], column, row_count)
    ]

    constant_columns = [
        column
        for column in working_df.columns
        if working_df[column].dropna().nunique() <= 1 and not working_df[column].dropna().empty
    ]

    mostly_missing_columns = [
        column
        for column in working_df.columns
        if row_count > 0 and working_df[column].isna().mean() >= 0.6
    ]

    high_cardinality_columns = [
        column
        for column in raw_categorical_columns
        if working_df[column].nunique(dropna=True) > min(30, max(15, int(row_count * 0.3)))
    ]

    usable_numeric_columns = [
        column
        for column in raw_numeric_columns
        if column not in identifier_columns
        and column not in constant_columns
        and column not in mostly_missing_columns
    ]

    usable_categorical_columns = [
        column
        for column in raw_categorical_columns
        if column not in identifier_columns
        and column not in constant_columns
        and column not in mostly_missing_columns
    ]

    warnings: list[str] = []

    if identifier_columns:
        warnings.append(
            f"Excluded identifier-like columns: {', '.join(identifier_columns)}."
        )
    if constant_columns:
        warnings.append(
            f"Excluded constant columns: {', '.join(constant_columns)}."
        )
    if mostly_missing_columns:
        warnings.append(
            f"Columns with heavy missingness detected: {', '.join(mostly_missing_columns)}."
        )
    if high_cardinality_columns:
        warnings.append(
            f"High-cardinality categorical columns detected: {', '.join(high_cardinality_columns)}."
        )
    if not usable_numeric_columns and not usable_categorical_columns and not datetime_columns:
        warnings.append("No strongly usable columns were detected for automated analysis.")

    return DatasetProfile(
        row_count=row_count,
        column_count=column_count,
        numeric_columns=usable_numeric_columns,
        categorical_columns=usable_categorical_columns,
        datetime_columns=datetime_columns,
        identifier_columns=identifier_columns,
        constant_columns=constant_columns,
        high_cardinality_columns=high_cardinality_columns,
        mostly_missing_columns=mostly_missing_columns,
        usable_numeric_columns=usable_numeric_columns,
        usable_categorical_columns=usable_categorical_columns,
        warnings=warnings,
    )


def prepare_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, DatasetProfile]:
    prepared_df = safe_copy(df)
    prepared_df, _ = try_parse_datetime_columns(prepared_df)
    profile = infer_column_types(prepared_df)
    return prepared_df, profile


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["column", "missing_count", "missing_percent", "non_null_count"]
        )

    missing_count = df.isna().sum()
    missing_percent = (missing_count / len(df) * 100).round(2)
    non_null_count = df.notna().sum()

    result = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": missing_count.values,
            "missing_percent": missing_percent.values,
            "non_null_count": non_null_count.values,
        }
    )

    return result.sort_values(
        by=["missing_percent", "missing_count"],
        ascending=[False, False],
    ).reset_index(drop=True)


def summary_statistics(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    if not numeric_columns:
        return pd.DataFrame(
            columns=[
                "column",
                "mean",
                "median",
                "std",
                "min",
                "max",
                "skewness",
                "unique_values",
            ]
        )

    rows: list[dict[str, Any]] = []

    for column in numeric_columns:
        if column not in df.columns:
            continue

        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue

        rows.append(
            {
                "column": column,
                "mean": round(float(series.mean()), 4),
                "median": round(float(series.median()), 4),
                "std": round(float(series.std()), 4) if len(series) > 1 else 0.0,
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
                "skewness": round(float(series.skew()), 4) if len(series) > 2 else 0.0,
                "unique_values": int(series.nunique()),
            }
        )

    return pd.DataFrame(rows)


def find_strong_correlations(
    df: pd.DataFrame,
    numeric_columns: list[str],
    threshold: float = 0.6,
) -> list[tuple[str, str, float]]:
    valid_columns = [column for column in numeric_columns if column in df.columns]
    if len(valid_columns) < 2:
        return []

    corr = df[valid_columns].corr(numeric_only=True)
    if corr.empty:
        return []

    strong_pairs: list[tuple[str, str, float]] = []

    for i, col_a in enumerate(corr.columns):
        for j, col_b in enumerate(corr.columns):
            if j <= i:
                continue

            value = corr.loc[col_a, col_b]
            if pd.notna(value) and abs(float(value)) >= threshold:
                strong_pairs.append((col_a, col_b, round(float(value), 3)))

    strong_pairs.sort(key=lambda item: abs(item[2]), reverse=True)
    return strong_pairs


def _empty_plot(message: str, title: str = "No Data") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def create_histogram(df: pd.DataFrame, column: str) -> plt.Figure:
    if column not in df.columns:
        return _empty_plot(f"Column '{column}' was not found.", "Histogram")

    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return _empty_plot(f"Column '{column}' does not contain plottable numeric data.", "Histogram")

    fig, ax = plt.subplots(figsize=(8, 4))
    bin_count = min(20, max(5, int(np.sqrt(len(series))))) if len(series) else 10

    ax.hist(series, bins=bin_count)
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def create_categorical_bar(df: pd.DataFrame, column: str, top_n: int = 12) -> plt.Figure:
    if column not in df.columns:
        return _empty_plot(f"Column '{column}' was not found.", "Category Counts")

    counts = df[column].fillna("Missing").astype(str).value_counts().head(top_n)
    if counts.empty:
        return _empty_plot(f"Column '{column}' does not contain plottable categories.", "Category Counts")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.index, counts.values)
    ax.set_title(f"Top Categories in {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def create_correlation_heatmap(df: pd.DataFrame, columns: list[str]) -> plt.Figure:
    valid_columns = [column for column in columns if column in df.columns]
    if len(valid_columns) < 2:
        return _empty_plot("At least two numeric columns are required.", "Correlation Heatmap")

    corr = df[valid_columns].corr(numeric_only=True)
    if corr.empty:
        return _empty_plot("Correlation matrix could not be computed.", "Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(corr.values)

    ax.set_xticks(range(len(valid_columns)))
    ax.set_yticks(range(len(valid_columns)))
    ax.set_xticklabels(valid_columns, rotation=45, ha="right")
    ax.set_yticklabels(valid_columns)
    ax.set_title("Correlation Heatmap")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    for i in range(len(valid_columns)):
        for j in range(len(valid_columns)):
            value = corr.values[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    return fig


def create_boxplot_by_category(
    df: pd.DataFrame,
    numeric_column: str,
    categorical_column: str,
) -> plt.Figure:
    if numeric_column not in df.columns or categorical_column not in df.columns:
        return _empty_plot("Required columns for boxplot were not found.", "Boxplot")

    plot_df = df[[categorical_column, numeric_column]].copy()
    plot_df[numeric_column] = pd.to_numeric(plot_df[numeric_column], errors="coerce")
    plot_df = plot_df.dropna()

    if plot_df.empty:
        return _empty_plot("No valid rows available for boxplot generation.", "Boxplot")

    grouped_values: list[np.ndarray] = []
    labels: list[str] = []

    for category, group in plot_df.groupby(categorical_column):
        values = group[numeric_column].to_numpy()
        if len(values) > 0:
            grouped_values.append(values)
            labels.append(str(category))

    if not grouped_values:
        return _empty_plot("No valid category groups available for boxplot.", "Boxplot")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(grouped_values, tick_labels=labels)
    ax.set_title(f"{numeric_column} by {categorical_column}")
    ax.set_xlabel(categorical_column)
    ax.set_ylabel(numeric_column)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def create_scatter_plot(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
) -> plt.Figure:
    if x_column not in df.columns or y_column not in df.columns:
        return _empty_plot("Required columns for scatter plot were not found.", "Scatter Plot")

    plot_df = df[[x_column, y_column]].copy()
    plot_df[x_column] = pd.to_numeric(plot_df[x_column], errors="coerce")
    plot_df[y_column] = pd.to_numeric(plot_df[y_column], errors="coerce")
    plot_df = plot_df.dropna()

    if plot_df.empty:
        return _empty_plot("No valid rows available for scatter plot generation.", "Scatter Plot")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(plot_df[x_column], plot_df[y_column])
    ax.set_title(f"{y_column} vs {x_column}")
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    fig.tight_layout()
    return fig


def create_time_series_plot(
    df: pd.DataFrame,
    datetime_column: str,
    numeric_column: str,
) -> plt.Figure:
    if datetime_column not in df.columns or numeric_column not in df.columns:
        return _empty_plot("Required columns for time-series plot were not found.", "Time Series")

    plot_df = df[[datetime_column, numeric_column]].copy()
    plot_df[datetime_column] = pd.to_datetime(plot_df[datetime_column], errors="coerce")
    plot_df[numeric_column] = pd.to_numeric(plot_df[numeric_column], errors="coerce")
    plot_df = plot_df.dropna().sort_values(by=datetime_column)

    if plot_df.empty:
        return _empty_plot("No valid rows available for time-series plotting.", "Time Series")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(plot_df[datetime_column], plot_df[numeric_column])
    ax.set_title(f"{numeric_column} over {datetime_column}")
    ax.set_xlabel(datetime_column)
    ax.set_ylabel(numeric_column)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig