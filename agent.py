
from __future__ import annotations

from typing import Any

import pandas as pd

from tools import (
    DatasetProfile,
    analyze_missing_values,
    create_boxplot_by_category,
    create_categorical_bar,
    create_correlation_heatmap,
    create_histogram,
    create_scatter_plot,
    create_time_series_plot,
    find_strong_correlations,
    prepare_dataframe,
    summary_statistics,
)


class VizAIAgent:
    def __init__(self) -> None:
        self.logs: list[str] = []

    def log(self, message: str) -> None:
        self.logs.append(message)

    def _infer_primary_target(
        self,
        profile: DatasetProfile,
        stats_df: pd.DataFrame,
        strong_correlations: list[tuple[str, str, float]],
    ) -> str | None:
        candidates = profile.usable_numeric_columns
        if not candidates:
            return None

        hint_order = [
            "price",
            "revenue",
            "sales",
            "profit",
            "income",
            "cost",
            "amount",
            "value",
            "score",
            "rating",
            "target",
            "label",
        ]

        score_map: dict[str, float] = {column: 0.0 for column in candidates}

        for column in candidates:
            lowered = column.lower()

            for index, hint in enumerate(hint_order):
                if hint == lowered or lowered.endswith(f"_{hint}") or hint in lowered:
                    score_map[column] += 100 - index * 5

            if not stats_df.empty and column in stats_df["column"].values:
                row = stats_df.loc[stats_df["column"] == column].iloc[0]
                score_map[column] += float(row["std"]) * 0.001
                score_map[column] += float(row["unique_values"]) * 0.2
                score_map[column] += abs(float(row["skewness"])) * 2

        for col_a, col_b, corr_value in strong_correlations:
            if col_a in score_map:
                score_map[col_a] += abs(corr_value) * 10
            if col_b in score_map:
                score_map[col_b] += abs(corr_value) * 10

        return max(score_map, key=score_map.get)

    def _rank_numeric_columns(
        self,
        profile: DatasetProfile,
        stats_df: pd.DataFrame,
        target_column: str | None,
    ) -> list[str]:
        if stats_df.empty:
            columns = profile.usable_numeric_columns.copy()
        else:
            columns = (
                stats_df.sort_values(
                    by=["std", "unique_values"],
                    ascending=[False, False],
                )["column"]
                .astype(str)
                .tolist()
            )

        columns = [column for column in columns if column in profile.usable_numeric_columns]

        if target_column and target_column in columns:
            columns.remove(target_column)
            columns.insert(0, target_column)

        return columns

    def _categorical_signal_score(
        self,
        df: pd.DataFrame,
        categorical_column: str,
        target_column: str | None,
    ) -> float:
        unique_count = df[categorical_column].nunique(dropna=True)
        if unique_count < 2 or unique_count > 12:
            return -1.0

        lowered = categorical_column.lower()

        semantic_bonus = 0.0
        preferred_hints = {
            "city": 8.0,
            "region": 7.0,
            "location": 7.0,
            "district": 4.0,
            "rooms": 8.0,
            "room": 8.0,
            "bedroom": 7.0,
            "furnished": 9.0,
            "condition": 7.0,
            "property_type": 8.0,
            "type": 3.0,
            "department": 8.0,
            "gender": 6.0,
            "category": 6.0,
            "segment": 6.0,
        }

        for hint, bonus in preferred_hints.items():
            if hint == lowered or hint in lowered:
                semantic_bonus = max(semantic_bonus, bonus)

        balance_bonus = max(0.0, 12.0 - float(unique_count))

        target_spread_bonus = 0.0
        if target_column and target_column in df.columns:
            working = df[[categorical_column, target_column]].copy()
            working[target_column] = pd.to_numeric(working[target_column], errors="coerce")
            working = working.dropna()

            if not working.empty:
                grouped = working.groupby(categorical_column)[target_column].agg(["mean", "count"])
                grouped = grouped[grouped["count"] >= 2]

                if len(grouped) >= 2:
                    spread = float(grouped["mean"].max() - grouped["mean"].min())
                    target_std = float(working[target_column].std()) if len(working) > 1 else 0.0
                    if target_std > 0:
                        target_spread_bonus = min(20.0, (spread / target_std) * 10.0)

        top_count = (
            df[categorical_column]
            .fillna("Missing")
            .astype(str)
            .value_counts()
            .iloc[0]
        )

        dominance_penalty = 0.0
        if len(df) > 0 and (top_count / len(df)) > 0.8:
            dominance_penalty = 6.0

        return semantic_bonus + balance_bonus + target_spread_bonus - dominance_penalty

    def _rank_categorical_columns(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        target_column: str | None,
    ) -> list[str]:
        candidates = [
            column
            for column in profile.usable_categorical_columns
            if column not in profile.high_cardinality_columns
        ]

        scored = [
            (column, self._categorical_signal_score(df, column, target_column))
            for column in candidates
        ]
        scored = [item for item in scored if item[1] >= 0]
        scored.sort(key=lambda item: item[1], reverse=True)

        return [column for column, _ in scored]

    def _deduplicate_actions(
        self,
        actions: list[dict[str, Any]],
        max_actions: int = 7,
    ) -> list[dict[str, Any]]:
        unique_actions: list[dict[str, Any]] = []
        seen: set[str] = set()

        for action in actions:
            key = repr(action)
            if key in seen:
                continue
            seen.add(key)
            unique_actions.append(action)

        return unique_actions[:max_actions]

    def choose_actions(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        stats_df: pd.DataFrame,
        strong_correlations: list[tuple[str, str, float]],
        target_column: str | None,
    ) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = [{"type": "missing_report"}]

        ranked_numeric = self._rank_numeric_columns(profile, stats_df, target_column)
        ranked_categorical = self._rank_categorical_columns(df, profile, target_column)

        if target_column:
            actions.append({"type": "numeric_histogram", "column": target_column})
        elif ranked_numeric:
            actions.append({"type": "numeric_histogram", "column": ranked_numeric[0]})

        if ranked_categorical:
            actions.append({"type": "categorical_bar", "column": ranked_categorical[0]})

        if len(ranked_numeric) >= 2:
            if target_column:
                target_pairs = [
                    pair
                    for pair in strong_correlations
                    if target_column in {pair[0], pair[1]}
                ]

                if target_pairs:
                    col_a, col_b, _ = target_pairs[0]
                    x_column = col_a if col_a != target_column else col_b
                    y_column = target_column
                    actions.append(
                        {
                            "type": "scatter_plot",
                            "x_column": x_column,
                            "y_column": y_column,
                        }
                    )
                else:
                    partner = next(
                        (column for column in ranked_numeric if column != target_column),
                        None,
                    )
                    if partner:
                        actions.append(
                            {
                                "type": "scatter_plot",
                                "x_column": partner,
                                "y_column": target_column,
                            }
                        )
            elif strong_correlations:
                col_a, col_b, _ = strong_correlations[0]
                actions.append(
                    {
                        "type": "scatter_plot",
                        "x_column": col_a,
                        "y_column": col_b,
                    }
                )
            else:
                actions.append(
                    {
                        "type": "scatter_plot",
                        "x_column": ranked_numeric[0],
                        "y_column": ranked_numeric[1],
                    }
                )

        heatmap_columns = ranked_numeric[:6]
        if len(heatmap_columns) >= 3:
            actions.append(
                {
                    "type": "correlation_heatmap",
                    "columns": heatmap_columns,
                }
            )

        boxplot_candidates = [
            column
            for column in ranked_categorical
            if 2 <= df[column].nunique(dropna=True) <= 10
        ]
        if target_column and boxplot_candidates:
            actions.append(
                {
                    "type": "boxplot_by_category",
                    "numeric_column": target_column,
                    "categorical_column": boxplot_candidates[0],
                }
            )
        elif ranked_numeric and boxplot_candidates:
            actions.append(
                {
                    "type": "boxplot_by_category",
                    "numeric_column": ranked_numeric[0],
                    "categorical_column": boxplot_candidates[0],
                }
            )

        if profile.datetime_columns and target_column:
            actions.append(
                {
                    "type": "time_series",
                    "datetime_column": profile.datetime_columns[0],
                    "numeric_column": target_column,
                }
            )

        if not target_column and len(ranked_numeric) >= 2:
            actions.append({"type": "numeric_histogram", "column": ranked_numeric[1]})

        if not target_column and len(ranked_categorical) >= 2:
            actions.append({"type": "categorical_bar", "column": ranked_categorical[1]})

        return self._deduplicate_actions(actions)

    def _best_category_effect_insight(
        self,
        df: pd.DataFrame,
        target_column: str | None,
        ranked_categorical: list[str],
    ) -> str | None:
        if not target_column or target_column not in df.columns:
            return None

        for categorical_column in ranked_categorical:
            if not (2 <= df[categorical_column].nunique(dropna=True) <= 10):
                continue

            working = df[[categorical_column, target_column]].copy()
            working[target_column] = pd.to_numeric(working[target_column], errors="coerce")
            working = working.dropna()
            if working.empty:
                continue

            grouped_means = (
                working.groupby(categorical_column)[target_column]
                .mean()
                .sort_values(ascending=False)
            )
            if len(grouped_means) < 2:
                continue

            best_group = grouped_means.index[0]
            best_value = grouped_means.iloc[0]
            worst_group = grouped_means.index[-1]
            worst_value = grouped_means.iloc[-1]

            return (
                f"'{categorical_column}' appears informative for '{target_column}': "
                f"'{best_group}' has the highest average ({best_value:.2f}), "
                f"while '{worst_group}' has the lowest ({worst_value:.2f})."
            )

        return None

    def generate_insights(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        missing_df: pd.DataFrame,
        stats_df: pd.DataFrame,
        strong_correlations: list[tuple[str, str, float]],
        target_column: str | None,
    ) -> list[str]:
        insights: list[str] = []

        insights.append(
            f"The dataset contains {profile.row_count} rows and {profile.column_count} columns."
        )

        insights.append(
            f"VizAI detected {len(profile.usable_numeric_columns)} usable numeric, "
            f"{len(profile.usable_categorical_columns)} usable categorical, and "
            f"{len(profile.datetime_columns)} datetime columns."
        )

        if target_column:
            insights.append(
                f"The agent selected '{target_column}' as the primary analysis target."
            )

        excluded_parts: list[str] = []
        if profile.identifier_columns:
            excluded_parts.append(
                f"identifier-like columns ({', '.join(profile.identifier_columns)})"
            )
        if profile.constant_columns:
            excluded_parts.append(
                f"constant columns ({', '.join(profile.constant_columns)})"
            )
        if profile.mostly_missing_columns:
            excluded_parts.append(
                f"heavily missing columns ({', '.join(profile.mostly_missing_columns)})"
            )

        if excluded_parts:
            insights.append(
                "The agent excluded " + ", ".join(excluded_parts) + " from the main analysis."
            )

        if not missing_df.empty and (missing_df["missing_count"] > 0).any():
            top_missing = missing_df.loc[missing_df["missing_percent"].idxmax()]
            insights.append(
                f"The most incomplete column is '{top_missing['column']}' with "
                f"{top_missing['missing_percent']}% missing values."
            )
        else:
            insights.append("No missing values were detected in the dataset.")

        if strong_correlations:
            col_a, col_b, corr_value = strong_correlations[0]
            insights.append(
                f"The strongest numeric relationship is between '{col_a}' and '{col_b}' "
                f"with correlation {corr_value}."
            )

        if not stats_df.empty:
            most_variable_row = stats_df.loc[stats_df["std"].idxmax()]
            insights.append(
                f"'{most_variable_row['column']}' shows the highest variation "
                f"(std={most_variable_row['std']})."
            )

            most_skewed_row = stats_df.iloc[stats_df["skewness"].abs().argmax()]
            insights.append(
                f"'{most_skewed_row['column']}' has the strongest skewness "
                f"({most_skewed_row['skewness']})."
            )

        ranked_categorical = self._rank_categorical_columns(df, profile, target_column)
        category_effect = self._best_category_effect_insight(df, target_column, ranked_categorical)
        if category_effect:
            insights.append(category_effect)
        elif ranked_categorical:
            counts = df[ranked_categorical[0]].fillna("Missing").astype(str).value_counts()
            if not counts.empty:
                insights.append(
                    f"In '{ranked_categorical[0]}', the most frequent category is "
                    f"'{counts.index[0]}' with {int(counts.iloc[0])} rows."
                )

        if profile.high_cardinality_columns:
            insights.append(
                f"High-cardinality categorical columns such as "
                f"{', '.join(profile.high_cardinality_columns[:2])} were detected and limited "
                f"to avoid misleading charts."
            )

        if not profile.usable_numeric_columns and not profile.usable_categorical_columns:
            insights.append(
                "The dataset has limited structure for automated chart generation, "
                "so the analysis mainly focuses on data quality signals."
            )

        return insights[:8]

    def evaluate_analysis(
        self,
        profile: DatasetProfile,
        actions: list[dict[str, Any]],
        insights: list[str],
        missing_df: pd.DataFrame,
        chart_count: int,
        target_column: str | None,
    ) -> dict[str, Any]:
        criteria: list[tuple[str, bool, int]] = []

        has_missing_report = any(action["type"] == "missing_report" for action in actions)
        has_histogram = any(action["type"] == "numeric_histogram" for action in actions)
        has_bar = any(action["type"] == "categorical_bar" for action in actions)
        has_scatter = any(action["type"] == "scatter_plot" for action in actions)
        has_heatmap = any(action["type"] == "correlation_heatmap" for action in actions)
        has_boxplot = any(action["type"] == "boxplot_by_category" for action in actions)
        has_time_series = any(action["type"] == "time_series" for action in actions)

        target_focused_actions = 0
        for action in actions:
            if target_column is None:
                continue

            if action["type"] == "numeric_histogram" and action.get("column") == target_column:
                target_focused_actions += 1
            elif action["type"] == "scatter_plot" and target_column in {
                action.get("x_column"),
                action.get("y_column"),
            }:
                target_focused_actions += 1
            elif (
                action["type"] == "boxplot_by_category"
                and action.get("numeric_column") == target_column
            ):
                target_focused_actions += 1
            elif action["type"] == "time_series" and action.get("numeric_column") == target_column:
                target_focused_actions += 1

        missing_detected = not missing_df.empty and (missing_df["missing_count"] > 0).any()

        if chart_count == 0:
            chart_set_is_focused = False
        elif profile.usable_numeric_columns or profile.usable_categorical_columns:
            chart_set_is_focused = 3 <= chart_count <= 6
        else:
            chart_set_is_focused = chart_count <= 2

        criteria.append(("Dataset profiling completed.", True, 15))
        criteria.append(("Missing-value analysis included.", has_missing_report, 10))
        criteria.append(
            (
                "A primary analysis target was identified or safely skipped.",
                bool(target_column) or not profile.usable_numeric_columns,
                15,
            )
        )

        if profile.usable_numeric_columns:
            criteria.append(("Focused numeric analysis included.", has_histogram, 10))

        if len(profile.usable_numeric_columns) >= 2:
            criteria.append(
                (
                    "Numeric relationship analysis included.",
                    has_scatter or has_heatmap,
                    15,
                )
            )

        if profile.usable_categorical_columns:
            criteria.append(("Categorical analysis included.", has_bar, 10))

        if target_column and profile.usable_categorical_columns:
            criteria.append(("Target-vs-category comparison included.", has_boxplot, 10))

        if profile.datetime_columns and target_column:
            criteria.append(("Time-based analysis included when applicable.", has_time_series, 5))

        criteria.append(
            (
                "The analysis stayed focused instead of generating too many charts.",
                chart_set_is_focused,
                5,
            )
        )

        criteria.append(
            (
                "The chosen charts are centered on the main target.",
                target_focused_actions >= 2 or not target_column,
                5,
            )
        )

        criteria.append(("Multiple meaningful insights were generated.", len(insights) >= 6, 5))

        total_weight = sum(weight for _, _, weight in criteria)
        achieved_weight = sum(weight for _, passed, weight in criteria if passed)
        base_score = int(round((achieved_weight / total_weight) * 100)) if total_weight else 100

        penalties: list[tuple[str, int]] = []

        if profile.constant_columns:
            penalties.append(
                (
                    f"Penalty applied: constant columns reduce analysis reliability ({', '.join(profile.constant_columns)}).",
                    3,
                )
            )

        if profile.high_cardinality_columns:
            penalties.append(
                (
                    "Penalty applied: high-cardinality categorical columns limit category-based analysis quality.",
                    4,
                )
            )

        if profile.mostly_missing_columns:
            penalties.append(
                (
                    "Penalty applied: heavily missing columns reduce data reliability.",
                    min(8, 3 + len(profile.mostly_missing_columns)),
                )
            )

        if missing_detected:
            worst_missing = float(missing_df["missing_percent"].max())
            if worst_missing >= 50:
                penalties.append(
                    ("Penalty applied: at least one column has over 50% missing values.", 6)
                )
            elif worst_missing >= 20:
                penalties.append(
                    ("Penalty applied: at least one column has over 20% missing values.", 3)
                )

        if target_column and target_column in missing_df["column"].values:
            target_missing = float(
                missing_df.loc[missing_df["column"] == target_column, "missing_percent"].iloc[0]
            )
            if target_missing >= 10:
                penalties.append(
                    (
                        f"Penalty applied: the primary target '{target_column}' contains notable missing values.",
                        5,
                    )
                )

        if chart_count > 6:
            penalties.append(
                ("Penalty applied: too many charts were generated for a focused analysis.", 4)
            )

        penalty_points = sum(points for _, points in penalties)
        final_score = max(0, min(100, base_score - penalty_points))

        reasons = [label for label, passed, _ in criteria if passed]
        missed = [label for label, passed, _ in criteria if not passed]
        missed.extend(label for label, _ in penalties)

        return {
            "score": final_score,
            "reasons": reasons,
            "missed_criteria": missed,
            "achieved_points": final_score,
            "total_possible_points": 100,
        }

    def analyze(self, df: pd.DataFrame) -> dict[str, Any]:
        self.logs = []

        if df.empty or len(df.columns) == 0:
            self.log("An empty dataset was uploaded.")
            empty_profile = DatasetProfile(
                row_count=len(df),
                column_count=len(df.columns),
                numeric_columns=[],
                categorical_columns=[],
                datetime_columns=[],
                identifier_columns=[],
                constant_columns=[],
                high_cardinality_columns=[],
                mostly_missing_columns=[],
                usable_numeric_columns=[],
                usable_categorical_columns=[],
                warnings=["The uploaded dataset is empty."],
            )
            empty_missing = analyze_missing_values(df)
            empty_stats = summary_statistics(df, [])
            return {
                "profile": empty_profile,
                "missing_df": empty_missing,
                "stats_df": empty_stats,
                "actions": [],
                "charts": [],
                "insights": ["The uploaded dataset is empty, so no automated analysis was performed."],
                "evaluation": {
                    "score": 0,
                    "reasons": ["The dataset was empty."],
                    "missed_criteria": [],
                    "achieved_points": 0,
                    "total_possible_points": 100,
                },
                "logs": self.logs,
                "strong_correlations": [],
                "target_column": None,
            }

        prepared_df, profile = prepare_dataframe(df)
        self.log("Dataset loaded and sanitized.")
        self.log("Column profiling completed.")

        for warning in profile.warnings:
            self.log(f"Profile warning: {warning}")

        missing_df = analyze_missing_values(prepared_df)
        self.log("Missing-value analysis completed.")

        stats_df = summary_statistics(prepared_df, profile.usable_numeric_columns)
        self.log("Summary statistics generated.")

        strong_correlations = find_strong_correlations(prepared_df, profile.usable_numeric_columns)
        if strong_correlations:
            strongest = strong_correlations[0]
            self.log(
                f"Strong correlation detected: {strongest[0]} vs {strongest[1]} ({strongest[2]})."
            )
        else:
            self.log("No strong numeric correlations passed the threshold.")

        target_column = self._infer_primary_target(profile, stats_df, strong_correlations)
        if target_column:
            self.log(f"Primary target selected: {target_column}.")
        else:
            self.log("No primary target was selected.")

        actions = self.choose_actions(
            prepared_df,
            profile,
            stats_df,
            strong_correlations,
            target_column,
        )
        self.log(f"Agent selected {len(actions)} analysis actions.")

        charts: list[dict[str, Any]] = []

        for action in actions:
            action_type = action["type"]

            if action_type == "missing_report":
                self.log("Prepared missing-value report.")
                continue

            if action_type == "numeric_histogram":
                column = action["column"]
                charts.append(
                    {
                        "title": f"Histogram: {column}",
                        "figure": create_histogram(prepared_df, column),
                    }
                )
                self.log(f"Generated histogram for '{column}'.")
                continue

            if action_type == "categorical_bar":
                column = action["column"]
                charts.append(
                    {
                        "title": f"Category Counts: {column}",
                        "figure": create_categorical_bar(prepared_df, column),
                    }
                )
                self.log(f"Generated category count chart for '{column}'.")
                continue

            if action_type == "correlation_heatmap":
                columns = action["columns"]
                charts.append(
                    {
                        "title": "Correlation Heatmap",
                        "figure": create_correlation_heatmap(prepared_df, columns),
                    }
                )
                self.log("Generated correlation heatmap.")
                continue

            if action_type == "scatter_plot":
                x_column = action["x_column"]
                y_column = action["y_column"]
                charts.append(
                    {
                        "title": f"Scatter Plot: {y_column} vs {x_column}",
                        "figure": create_scatter_plot(prepared_df, x_column, y_column),
                    }
                )
                self.log(f"Generated scatter plot for '{y_column}' vs '{x_column}'.")
                continue

            if action_type == "boxplot_by_category":
                numeric_column = action["numeric_column"]
                categorical_column = action["categorical_column"]
                charts.append(
                    {
                        "title": f"Boxplot: {numeric_column} by {categorical_column}",
                        "figure": create_boxplot_by_category(
                            prepared_df,
                            numeric_column,
                            categorical_column,
                        ),
                    }
                )
                self.log(
                    f"Generated boxplot for '{numeric_column}' by '{categorical_column}'."
                )
                continue

            if action_type == "time_series":
                datetime_column = action["datetime_column"]
                numeric_column = action["numeric_column"]
                charts.append(
                    {
                        "title": f"Time Series: {numeric_column} over {datetime_column}",
                        "figure": create_time_series_plot(
                            prepared_df,
                            datetime_column,
                            numeric_column,
                        ),
                    }
                )
                self.log(
                    f"Generated time-series plot for '{numeric_column}' over '{datetime_column}'."
                )

        insights = self.generate_insights(
            prepared_df,
            profile,
            missing_df,
            stats_df,
            strong_correlations,
            target_column,
        )
        self.log(f"Generated {len(insights)} textual insights.")

        evaluation = self.evaluate_analysis(
            profile,
            actions,
            insights,
            missing_df,
            chart_count=len(charts),
            target_column=target_column,
        )
        self.log(f"Evaluation completed with score {evaluation['score']} / 100.")

        return {
            "profile": profile,
            "missing_df": missing_df,
            "stats_df": stats_df,
            "actions": actions,
            "charts": charts,
            "insights": insights,
            "evaluation": evaluation,
            "logs": self.logs,
            "strong_correlations": strong_correlations,
            "target_column": target_column,
        }