from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/element2vec-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import load_yaml, markdown_table, resolve_path


STATUS_SCORE = {
    "Failed technically": 0,
    "Invalid due to leakage/confounding": 1,
    "Negative": 2,
    "Weak/ambiguous": 3,
    "Moderate positive": 4,
    "Strong positive": 5,
}


def _load_results(sweep_config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    base_config = load_yaml(sweep_config["base_config"])
    settings = sweep_config["pca_sweep"]
    root = resolve_path(settings["output_root"])
    original = pd.read_csv(resolve_path(base_config["project"]["results_dir"]) / "experiment_registry.csv")
    frames = []
    for dimension in settings["dimensions"]:
        path = root / "results" / f"{int(dimension)}d" / "experiment_registry.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "pca_dimension", int(dimension))
        frames.append(frame)
    sweep = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return original, sweep, root


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _parse_metrics(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {}
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _flatten_scalars(value: Any, prefix: str = "") -> dict[str, float]:
    output: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            output.update(_flatten_scalars(item, name))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            output.update(_flatten_scalars(item, f"{prefix}[{index}]"))
    elif isinstance(value, bool):
        output[prefix] = float(value)
    elif isinstance(value, (int, float)) and np.isfinite(value):
        output[prefix] = float(value)
    return output


def _all_scalar_metrics(
    original: pd.DataFrame,
    sweep: pd.DataFrame,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    excluded = {"pca_requested_dimension", "experiment_elapsed_seconds"}
    original_map: dict[str, dict[str, float]] = {}
    for _, row in original.iterrows():
        metrics = _flatten_scalars(_parse_metrics(row.get("quantitative_metrics")))
        metrics = {key: value for key, value in metrics.items() if key not in excluded}
        primary = pd.to_numeric(pd.Series([row.get("metric_value")]), errors="coerce").iloc[0]
        if np.isfinite(primary):
            metrics[f"primary::{row.get('metric_primary')}"] = float(primary)
        original_map[str(row["experiment_id"])] = metrics

    rows = []
    for _, row in sweep.iterrows():
        experiment_id = str(row["experiment_id"])
        metrics = _flatten_scalars(_parse_metrics(row.get("quantitative_metrics")))
        metrics = {key: value for key, value in metrics.items() if key not in excluded}
        primary = pd.to_numeric(pd.Series([row.get("metric_value")]), errors="coerce").iloc[0]
        if np.isfinite(primary):
            metrics[f"primary::{row.get('metric_primary')}"] = float(primary)
        full_metrics = original_map.get(experiment_id, {})
        for metric_name, value in metrics.items():
            full_value = full_metrics.get(metric_name, np.nan)
            tolerance = max(
                float(settings.get("tolerance_absolute", 0.02)),
                float(settings.get("tolerance_relative", 0.10)) * abs(full_value),
            ) if np.isfinite(full_value) else np.nan
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "experiment_name": row["experiment_name"],
                    "family": row["family"],
                    "representation_level": row["representation_level"],
                    "pca_dimension": int(row["pca_dimension"]),
                    "metric_name": metric_name,
                    "is_primary_metric": metric_name.startswith("primary::"),
                    "metric_value": value,
                    "full_metric_value": full_value,
                    "delta_from_full": value - full_value if np.isfinite(full_value) else np.nan,
                    "absolute_delta_from_full": abs(value - full_value) if np.isfinite(full_value) else np.nan,
                    "tolerance": tolerance,
                    "within_full_tolerance": abs(value - full_value) <= tolerance if np.isfinite(tolerance) else False,
                }
            )
    long = pd.DataFrame(rows)
    metric_rows = []
    for (experiment_id, metric_name), group in long.groupby(["experiment_id", "metric_name"]):
        finite = group.loc[group["metric_value"].notna()].sort_values("pca_dimension")
        comparable = finite.loc[finite["full_metric_value"].notna()]
        metric_range = float(finite["metric_value"].max() - finite["metric_value"].min()) if len(finite) else np.nan
        stable = comparable.loc[comparable["within_full_tolerance"]]
        full_value = float(comparable["full_metric_value"].iloc[0]) if len(comparable) else np.nan
        normalized_range = metric_range / max(abs(full_value), float(settings.get("tolerance_absolute", 0.02))) if np.isfinite(full_value) else np.nan
        metric_rows.append(
            {
                "experiment_id": experiment_id,
                "experiment_name": group["experiment_name"].iloc[0],
                "family": group["family"].iloc[0],
                "representation_level": group["representation_level"].iloc[0],
                "metric_name": metric_name,
                "is_primary_metric": bool(group["is_primary_metric"].iloc[0]),
                "full_metric_value": full_value,
                "metric_range_across_dimensions": metric_range,
                "normalized_metric_range": normalized_range,
                "smallest_dimension_within_full_tolerance": int(stable["pca_dimension"].min()) if len(stable) else np.nan,
                "within_full_tolerance_fraction": float(comparable["within_full_tolerance"].mean()) if len(comparable) else np.nan,
                "varies_across_dimensions": bool(np.isfinite(metric_range) and metric_range > 1e-9),
            }
        )
    metric_analysis = pd.DataFrame(metric_rows)
    informative_names = metric_analysis.loc[
        metric_analysis["varies_across_dimensions"] | metric_analysis["is_primary_metric"],
        ["experiment_id", "metric_name"],
    ]
    informative = long.merge(informative_names.assign(informative=True), on=["experiment_id", "metric_name"], how="left")
    informative = informative.loc[informative["informative"].fillna(False) & informative["full_metric_value"].notna()]
    scalar_summary = (
        informative.groupby("pca_dimension")
        .agg(
            informative_scalar_metrics=("metric_name", "size"),
            all_scalar_within_full_tolerance_fraction=("within_full_tolerance", "mean"),
            all_scalar_median_absolute_delta_from_full=("absolute_delta_from_full", "median"),
        )
        .reset_index()
    )
    return long, metric_analysis, scalar_summary


def _analyze_experiments(
    original: pd.DataFrame,
    sweep: pd.DataFrame,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    original_cols = [
        "experiment_id", "experiment_name", "family", "representation_level",
        "metric_primary", "metric_value", "baseline_value", "effect_size", "status",
        "main_observation", "possible_confounder", "recommended_next_step",
    ]
    original_small = original[original_cols].copy().rename(
        columns={
            "metric_value": "full_metric_value",
            "baseline_value": "full_baseline_value",
            "effect_size": "full_effect_size",
            "status": "full_status",
            "main_observation": "full_observation",
            "possible_confounder": "full_confounder",
            "recommended_next_step": "full_next_step",
        }
    )
    merged = sweep.merge(original_small, on=["experiment_id", "experiment_name", "family", "representation_level", "metric_primary"], how="left")
    for col in ["metric_value", "baseline_value", "effect_size", "full_metric_value", "full_baseline_value", "full_effect_size"]:
        merged[col] = _numeric(merged[col])
    merged["delta_from_full"] = merged["metric_value"] - merged["full_metric_value"]
    merged["absolute_delta_from_full"] = merged["delta_from_full"].abs()
    absolute = float(settings.get("tolerance_absolute", 0.02))
    relative = float(settings.get("tolerance_relative", 0.10))
    merged["tolerance"] = np.maximum(absolute, relative * merged["full_metric_value"].abs())
    merged["within_full_tolerance"] = merged["absolute_delta_from_full"] <= merged["tolerance"]
    merged["status_matches_full"] = merged["status"] == merged["full_status"]
    merged["effect_sign_matches_full"] = (
        np.sign(merged["effect_size"].fillna(0)) == np.sign(merged["full_effect_size"].fillna(0))
    )
    merged["status_score"] = merged["status"].map(STATUS_SCORE)

    invariant = set(settings.get("invariant_experiments", []))
    lower_is_stronger = set(settings.get("lower_is_stronger_experiments", []))
    directionless = set(settings.get("directionless_primary_metrics", []))
    dimensions = sorted(int(value) for value in settings["dimensions"])
    low_ceiling = int(settings.get("low_rank_ceiling", 32))
    analysis_rows = []
    for experiment_id, group in merged.groupby("experiment_id"):
        group = group.sort_values("pca_dimension")
        first = group.iloc[0]
        finite = group.loc[group["metric_value"].notna()]
        stable = finite.loc[finite["within_full_tolerance"]]
        smallest_stable = int(stable["pca_dimension"].min()) if len(stable) else np.nan
        low = finite.loc[finite["pca_dimension"] <= low_ceiling]
        high = finite.loc[finite["pca_dimension"] == max(dimensions)]
        metric_range = float(finite["metric_value"].max() - finite["metric_value"].min()) if len(finite) else np.nan
        metric_direction = "lower is stronger" if experiment_id in lower_is_stronger else "directionless" if experiment_id in directionless else "higher is stronger"
        if len(finite):
            extreme_index = finite["metric_value"].idxmin() if experiment_id in lower_is_stronger else finite["metric_value"].idxmax()
            strongest_dimension = int(finite.loc[extreme_index, "pca_dimension"])
        else:
            strongest_dimension = np.nan
        if experiment_id in invariant:
            classification = "dimension-invariant / unavailable"
        elif not len(finite) or not np.isfinite(first["full_metric_value"]):
            classification = "not numerically comparable"
        elif len(stable) and smallest_stable <= low_ceiling:
            high_value = float(high["metric_value"].iloc[0]) if len(high) else float(first["full_metric_value"])
            low_best = float(low["metric_value"].min()) if experiment_id in lower_is_stronger and len(low) else float(low["metric_value"].max()) if len(low) else np.nan
            tol = float(first["tolerance"])
            strengthened = (
                low_best < high_value - tol
                if experiment_id in lower_is_stronger
                else low_best > high_value + tol
            )
            if experiment_id not in directionless and np.isfinite(low_best) and strengthened:
                classification = "low-rank enhanced"
            else:
                classification = "low-rank preserved"
        elif len(stable):
            classification = "rank-sensitive above 32D"
        else:
            classification = "strongly rank-sensitive"
        analysis_rows.append(
            {
                "experiment_id": experiment_id,
                "experiment_name": first["experiment_name"],
                "family": first["family"],
                "representation_level": first["representation_level"],
                "metric_primary": first["metric_primary"],
                "full_metric_value": first["full_metric_value"],
                "full_effect_size": first["full_effect_size"],
                "full_status": first["full_status"],
                "dimensions_completed": int(group["pca_dimension"].nunique()),
                "smallest_dimension_within_full_tolerance": smallest_stable,
                "metric_direction": metric_direction,
                "strongest_metric_dimension": strongest_dimension,
                "metric_range_across_dimensions": metric_range,
                "status_match_fraction": float(group["status_matches_full"].mean()),
                "effect_sign_match_fraction": float(group["effect_sign_matches_full"].mean()),
                "pca_sensitivity_class": classification,
                "full_observation": first["full_observation"],
                "full_confounder": first["full_confounder"],
                "full_next_step": first["full_next_step"],
            }
        )
    analysis = pd.DataFrame(analysis_rows).sort_values(["family", "experiment_id"])
    return merged, analysis


def _dimension_summary(merged: pd.DataFrame, invariant: set[str]) -> pd.DataFrame:
    relevant = merged.loc[~merged["experiment_id"].isin(invariant)].copy()
    rows = []
    for dimension, group in relevant.groupby("pca_dimension"):
        comparable = group.loc[group["metric_value"].notna() & group["full_metric_value"].notna()]
        rows.append(
            {
                "pca_dimension": int(dimension),
                "experiments": len(group),
                "numerically_comparable": len(comparable),
                "within_full_tolerance_fraction": float(comparable["within_full_tolerance"].mean()),
                "status_match_fraction": float(group["status_matches_full"].mean()),
                "effect_sign_match_fraction": float(comparable["effect_sign_matches_full"].mean()),
                "median_absolute_delta_from_full": float(comparable["absolute_delta_from_full"].median()),
                "strong_positive": int((group["status"] == "Strong positive").sum()),
                "moderate_positive": int((group["status"] == "Moderate positive").sum()),
                "weak_ambiguous": int((group["status"] == "Weak/ambiguous").sum()),
                "negative": int((group["status"] == "Negative").sum()),
                "invalid_or_failed": int(group["status"].isin(["Invalid due to leakage/confounding", "Failed technically"]).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("pca_dimension", ascending=False)


def _family_summary(merged: pd.DataFrame, invariant: set[str]) -> pd.DataFrame:
    relevant = merged.loc[~merged["experiment_id"].isin(invariant)].copy()
    rows = []
    for (family, dimension), group in relevant.groupby(["family", "pca_dimension"]):
        comparable = group.loc[group["metric_value"].notna() & group["full_metric_value"].notna()]
        rows.append(
            {
                "family": family,
                "pca_dimension": int(dimension),
                "experiments": len(group),
                "within_full_tolerance_fraction": float(comparable["within_full_tolerance"].mean()) if len(comparable) else np.nan,
                "status_match_fraction": float(group["status_matches_full"].mean()),
                "median_absolute_delta_from_full": float(comparable["absolute_delta_from_full"].median()) if len(comparable) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _plot_dimension_summary(frame: pd.DataFrame, path: Path, dpi: int) -> None:
    ordered = frame.sort_values("pca_dimension")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(ordered["pca_dimension"], ordered["within_full_tolerance_fraction"], marker="o", label="Metric within full-space tolerance")
    ax.plot(ordered["pca_dimension"], ordered["status_match_fraction"], marker="s", label="Status matches full space")
    ax.plot(ordered["pca_dimension"], ordered["effect_sign_match_fraction"], marker="^", label="Effect sign matches full space")
    if "all_scalar_within_full_tolerance_fraction" in ordered:
        ax.plot(
            ordered["pca_dimension"],
            ordered["all_scalar_within_full_tolerance_fraction"],
            marker="D",
            linestyle="--",
            label="All varying scalar metrics within tolerance",
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(ordered["pca_dimension"])
    ax.set_xticklabels(ordered["pca_dimension"].astype(str))
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("PCA dimension")
    ax.set_ylabel("Fraction of comparable experiments")
    ax.set_title("Preservation of full-space conclusions across nested PCA ranks")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_family_heatmap(frame: pd.DataFrame, path: Path, dpi: int) -> None:
    pivot = frame.pivot(index="family", columns="pca_dimension", values="within_full_tolerance_fraction")
    pivot = pivot.reindex(columns=sorted(pivot.columns, reverse=True))
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), [str(value) for value in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_xlabel("PCA dimension")
    ax.set_ylabel("Experiment family")
    ax.set_title("Fraction of family metrics preserved within full-space tolerance")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white" if value < 0.55 else "black")
    fig.colorbar(image, ax=ax, label="Preserved fraction")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_experiment_curves(merged: pd.DataFrame, figure_dir: Path, dpi: int) -> pd.DataFrame:
    figure_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for experiment_id, group in merged.groupby("experiment_id"):
        group = group.sort_values("pca_dimension")
        if group["metric_value"].notna().sum() < 2:
            continue
        path = figure_dir / f"{experiment_id}_dimension_curve.png"
        fig, ax = plt.subplots(figsize=(7.8, 4.6))
        ax.plot(group["pca_dimension"], group["metric_value"], marker="o", label="PCA-space metric")
        full_value = group["full_metric_value"].iloc[0]
        if np.isfinite(full_value):
            ax.axhline(full_value, color="#b91c1c", linestyle="--", linewidth=1.2, label="Original full-space metric")
        baseline = group["baseline_value"]
        if baseline.notna().any():
            ax.plot(group["pca_dimension"], baseline, marker=".", linestyle=":", label="Recorded baseline")
        ax.set_xscale("log", base=2)
        ax.set_xticks(group["pca_dimension"])
        ax.set_xticklabels(group["pca_dimension"].astype(str))
        ax.set_xlabel("PCA dimension")
        ax.set_ylabel(str(group["metric_primary"].iloc[0]))
        ax.set_title(f"{experiment_id}: dimension sensitivity")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        rows.append({"experiment_id": experiment_id, "figure_path": str(path)})
    return pd.DataFrame(rows)


def build(sweep_config: dict[str, Any]) -> None:
    settings = sweep_config["pca_sweep"]
    original, sweep, root = _load_results(sweep_config)
    reports = root / "reports"
    figures = reports / "figures"
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    expected_dimensions = {int(value) for value in settings["dimensions"]}
    expected_experiments = set(original["experiment_id"].astype(str))
    if sweep.empty:
        raise RuntimeError("No PCA sweep registries were found.")

    merged, analysis = _analyze_experiments(original, sweep, settings)
    scalar_long, scalar_analysis, scalar_dimension_summary = _all_scalar_metrics(original, sweep, settings)
    invariant = set(settings.get("invariant_experiments", []))
    dimension_summary = _dimension_summary(merged, invariant)
    dimension_summary = dimension_summary.merge(scalar_dimension_summary, on="pca_dimension", how="left")
    family_summary = _family_summary(merged, invariant)
    merged.to_csv(reports / "experiment_dimension_metrics.csv", index=False, encoding="utf-8-sig")
    analysis.to_csv(reports / "experiment_dimension_analysis.csv", index=False, encoding="utf-8-sig")
    scalar_long.to_csv(reports / "all_scalar_metrics_long.csv", index=False, encoding="utf-8-sig")
    scalar_analysis.to_csv(reports / "scalar_metric_dimension_analysis.csv", index=False, encoding="utf-8-sig")
    dimension_summary.to_csv(reports / "dimension_summary.csv", index=False, encoding="utf-8-sig")
    family_summary.to_csv(reports / "family_dimension_summary.csv", index=False, encoding="utf-8-sig")
    _plot_dimension_summary(dimension_summary, figures / "dimension_preservation_summary.png", 180)
    _plot_family_heatmap(family_summary, figures / "family_preservation_heatmap.png", 180)
    figure_index = _plot_experiment_curves(merged, figures / "experiments", 160)
    figure_index.to_csv(reports / "figure_index.csv", index=False, encoding="utf-8-sig")

    completed_dimensions = set(merged["pca_dimension"].astype(int))
    counts = merged.groupby("pca_dimension")["experiment_id"].nunique()
    coverage_ok = completed_dimensions == expected_dimensions and all(counts.get(d, 0) == len(expected_experiments) for d in expected_dimensions)
    class_counts = analysis["pca_sensitivity_class"].value_counts().rename_axis("classification").reset_index(name="experiments")
    enhanced = analysis.loc[analysis["pca_sensitivity_class"] == "low-rank enhanced"].sort_values("metric_range_across_dimensions", ascending=False)
    preserved = analysis.loc[analysis["pca_sensitivity_class"] == "low-rank preserved"].sort_values("smallest_dimension_within_full_tolerance")
    sensitive = analysis.loc[analysis["pca_sensitivity_class"].isin(["rank-sensitive above 32D", "strongly rank-sensitive"])].sort_values("metric_range_across_dimensions", ascending=False)
    informative_scalar = scalar_analysis.loc[
        scalar_analysis["varies_across_dimensions"] & scalar_analysis["full_metric_value"].notna()
    ].copy()
    experiment_scalar_summary = (
        informative_scalar.groupby(["experiment_id", "experiment_name", "family"])
        .agg(
            varying_scalar_metrics=("metric_name", "size"),
            low_rank_preserved_metric_fraction=("smallest_dimension_within_full_tolerance", lambda values: float((values <= int(settings.get("low_rank_ceiling", 32))).mean())),
            median_within_full_tolerance_fraction=("within_full_tolerance_fraction", "median"),
            max_normalized_metric_range=("normalized_metric_range", "max"),
        )
        .reset_index()
        .sort_values("max_normalized_metric_range", ascending=False)
    )
    experiment_scalar_summary.to_csv(reports / "experiment_all_metric_summary.csv", index=False, encoding="utf-8-sig")
    changing_secondary = informative_scalar.loc[~informative_scalar["is_primary_metric"]].copy()
    nonsemantic_pattern = (
        r"pvalue|permutation_p|(?:^|[._])p$|variance|sample|class|field|seed|count|available|"
        r"pairs|targets|components|poolings|tokens|prespecified|minerals|groups|"
        r"backgrounds|styles|paths|unavailable|dimensions"
    )
    semantic_changing_secondary = changing_secondary.loc[
        ~changing_secondary["metric_name"].str.contains(nonsemantic_pattern, case=False, regex=True)
    ].sort_values("normalized_metric_range", ascending=False)

    selected_ids = [
        "C1_same_value_context", "C2_element_role_proxies", "C3_contextual_neighbors",
        "D1_rock_continuity", "D2_local_geological_neighbors", "F1_lithology_perturbation",
        "F2_single_element_jacobian", "F3_combined_interventions",
        "J1_beyond_traditional_indicators", "J2_grouped_increment",
        "K1_representation_baselines", "X4_random_missingness_followup",
    ]
    selected = merged.loc[merged["experiment_id"].isin(selected_ids)].copy()
    selected_table = selected.pivot(index=["experiment_id", "metric_primary", "full_metric_value"], columns="pca_dimension", values="metric_value").reset_index()
    selected_table = selected_table.reindex(columns=["experiment_id", "metric_primary", "full_metric_value", 256, 64, 32, 16, 8, 2])
    selected_table.columns = [str(column) if isinstance(column, int) else column for column in selected_table.columns]

    basis_manifest_path = root / "basis" / "basis_manifest.csv"
    basis = pd.read_csv(basis_manifest_path) if basis_manifest_path.exists() else pd.DataFrame()
    basis_highlights = basis.loc[
        basis["basis_name"].isin([
            "sample_trained_encoded_mean_numeric",
            "sample_trained_projected_mean_numeric",
            "sample_trained_cls",
            "token_trained_contextual",
            "token_trained_projected",
        ]),
        ["basis_name", "fitted_components", "variance_2d", "variance_8d", "variance_16d", "variance_32d", "variance_64d", "variance_128d", "variance_256d"],
    ] if len(basis) else pd.DataFrame()
    supported_enhanced = enhanced.loc[enhanced["full_status"] != "Negative"]
    unsupported_enhanced = enhanced.loc[enhanced["full_status"] == "Negative"]
    summary_by_dimension = dimension_summary.set_index("pca_dimension")
    primary_64 = float(summary_by_dimension.loc[64, "within_full_tolerance_fraction"])
    primary_32 = float(summary_by_dimension.loc[32, "within_full_tolerance_fraction"])
    primary_8 = float(summary_by_dimension.loc[8, "within_full_tolerance_fraction"])
    primary_2 = float(summary_by_dimension.loc[2, "within_full_tolerance_fraction"])
    scalar_64 = float(summary_by_dimension.loc[64, "all_scalar_within_full_tolerance_fraction"])
    scalar_32 = float(summary_by_dimension.loc[32, "all_scalar_within_full_tolerance_fraction"])

    def scalar_at(experiment_id: str, metric_name: str, dimension: int) -> float:
        match = scalar_long.loc[
            scalar_long["experiment_id"].eq(experiment_id)
            & scalar_long["metric_name"].eq(metric_name)
            & scalar_long["pca_dimension"].eq(dimension),
            "metric_value",
        ]
        return float(match.iloc[0]) if len(match) else float("nan")

    x4_source_2d = scalar_at("X4_random_missingness_followup", "trained_source_purity_mean", 2)
    x4_rock_2d = scalar_at("X4_random_missingness_followup", "trained_rock_type_purity_mean", 2)
    x4_source_64d = scalar_at("X4_random_missingness_followup", "trained_source_purity_mean", 64)
    x4_rock_64d = scalar_at("X4_random_missingness_followup", "trained_rock_type_purity_mean", 64)
    run_manifest_path = root / "run_manifest.csv"
    runtime = float(pd.read_csv(run_manifest_path)["elapsed_seconds"].sum()) if run_manifest_path.exists() else np.nan

    report = f"""# Element2Vec V2 PCA 低秩子空间复核报告

## 1. 执行摘要

本轮将 Stage 3.6 的 **{len(expected_experiments)} 个实验**分别放入
**256、128、64、32、16、8、5、3、2D** 的嵌套 PCA 子空间中复算，
共形成 **{len(merged)} 条 experiment × dimension 结果**。完整覆盖校验：
**{'通过' if coverage_ok else '未通过'}**。

{markdown_table(class_counts)}

主要结论不是“越低维越好”，而是存在一个明显的层级：

1. **64D 基本保留主体结构。** {primary_64:.1%} 的主指标、{scalar_64:.1%} 的
   全部变化标量仍在原高维结果容差内；32D 分别降至 {primary_32:.1%} 和
   {scalar_32:.1%}。
2. **8D 以下发生普遍坍缩。** 8D 仅 {primary_8:.1%} 的主指标接近高维结果，
   2D 仅 {primary_2:.1%}。因此模型语义并非主要压缩在单一2–8D流形中。
3. **最前面的PC首先保留数据来源，而不是完整地质结构。** X4 中 trained
   source purity 在64D为 {x4_source_64d:.3f}、2D仍为 {x4_source_2d:.3f}；
   rock-type purity 却从 {x4_rock_64d:.3f} 降到 {x4_rock_2d:.3f}。
   这说明2D图中的主导分区很可能优先反映分析来源/缺失面板。
4. **contextual语义主要需要32–64D。** C1/C2/C3 在64D仍接近高维结果，
   但在16D以下持续衰减。元素上下文不是单一低维轴，而是中等秩结构。
5. **部分过程方向确实集中在低秩子空间。** F1/F2在5–3D对共同岩性/单元素
   响应方向的对齐更强，F3在低维更接近加性；这更像“共同响应方向被保留、
   背景特异和非线性残差位于较高秩”，不能简单解释为低维更懂地质过程。

![跨维度结论保留率](figures/dimension_preservation_summary.png)

这里的“低秩增强”只表示原实验的预注册指标在较低维度更强，不自动等于
更真实的地质语义。PCA 使用全审计样本无监督拟合，仍可能优先保留数据来源、
缺失模式或受标签监督塑造的高方差方向。

## 2. 方法

- 对每种表示分别拟合一次最大 256D PCA，然后截取前 k 个主成分；因此九个
  子空间严格嵌套，维度差异不会来自重复拟合的随机旋转。
- PCA 只中心化、不白化。距离、夹角、邻域、probe、轨迹和检索指标均在
  对应低维坐标中重新计算。
- numeric-only、masked-label 和 visible-label sample latent 使用同一 trained
  PCA 基底；随机模型使用自身基底。
- H/X 跨样品—标签空间比较将标签投影到 sample encoded-mean 的同一 PCA
  基底，避免在两个独立旋转后的坐标系中直接计算余弦。
- raw/log/CLR 等传统对照保持原定义；被降维的是 Element2Vec 表示。
- 静态 numeric tag 只有约 34 个可用秩，纯标签约 95 个可用秩；请求维度超过
  数据秩时记录实际有效维度，不虚构额外主成分。

PCA 基底方差记录见 [`basis_manifest.csv`](../basis/basis_manifest.csv)。

{markdown_table(basis_highlights.round(4))}

这组方差谱本身也说明不同表示层的秩不同：projected mean 在2D已解释约84%的
方差，而 contextual token 在16D仅解释约41%、32D约64%；Transformer 写入的
样品上下文明显比输入 MLP token 更分散。

## 3. 总体维度敏感性

{markdown_table(dimension_summary.round(4))}

![各实验家族保留率](figures/family_preservation_heatmap.png)

## 4. 低秩集中或放大候选

这些实验在不高于 {int(settings.get('low_rank_ceiling', 32))}D 时，按预先声明的
指标方向强于256D，并且至少一个低维点接近原始高维指标。它们说明有效信号可能
集中在前若干PC；并不表示一定超过原始表示或传统基线。尤其是内部再次执行
StandardScaler/PCA 的 probe，新增低方差PC可能被等权放大，因此非单调曲线也可能
来自实验管线而非模型语义。

{markdown_table(supported_enhanced[["experiment_id", "family", "metric_primary", "full_metric_value", "metric_direction", "smallest_dimension_within_full_tolerance", "strongest_metric_dimension", "metric_range_across_dimensions", "full_confounder"]], max_rows=25)}

以下指标虽然被低秩放大，但原实验仍为 Negative，不能据此恢复原假设：

{markdown_table(unsupported_enhanced[["experiment_id", "metric_primary", "full_metric_value", "strongest_metric_dimension", "metric_range_across_dimensions", "full_confounder"]], max_rows=15)}

## 5. 代表性实验曲线数据

{markdown_table(selected_table.round(4))}

## 6. 低秩保留结果

{markdown_table(preserved[["experiment_id", "family", "metric_primary", "full_metric_value", "smallest_dimension_within_full_tolerance", "status_match_fraction", "full_confounder"]], max_rows=30)}

## 7. 依赖较高秩或高度不稳定的结果

{markdown_table(sensitive[["experiment_id", "family", "metric_primary", "full_metric_value", "metric_range_across_dimensions", "status_match_fraction", "effect_sign_match_fraction", "full_confounder"]], max_rows=30)}

## 8. 次级定量指标检查

主指标不能覆盖实验内部的全部变化。例如 E7 的主指标是可匹配岩性数量，K3 的
主指标是顺序不变性，但它们还分别记录 centroid distance 和 token-dropout
cosine。本轮额外展开了所有 `metrics.json` 中的数值标量，并单独分析随维度变化
的指标。变化最大的次级指标如下：

{markdown_table(semantic_changing_secondary[["experiment_id", "family", "metric_name", "full_metric_value", "metric_range_across_dimensions", "normalized_metric_range", "smallest_dimension_within_full_tolerance"]], max_rows=30)}

逐指标全集见 [`all_scalar_metrics_long.csv`](all_scalar_metrics_long.csv) 和
[`scalar_metric_dimension_analysis.csv`](scalar_metric_dimension_analysis.csv)。

## 9. 解释边界

1. PCA 最大化方差而不是地质语义；低维结果变强可能是去噪，也可能是放大来源、
   缺失面板或训练标签方向。
2. 本轮是表示空间审计，不是独立泛化检验。checkpoint 仍在全数据标签监督下训练。
3. 一些 probe 实验内部原本还有 12D/24D 降维；代码现按外部 PCA 维数自动截断。
   因此 32–256D 区间可能出现平台，这是模型/实验设计共同导致的能力上限。
4. `B4`、`D3b`、`G3`、`I0`、`K4`、`K5` 属于变量缺失、依赖缺失或技术失败，
   其结论不会因 PCA 维度改变，但仍逐维保留以确保实验全集闭合。
5. 静态 element/tag 空间的样本数远小于 256，不能把高请求维数理解成高有效秩。

## 10. 后续建议

1. 后续定量分析建议把 **64D 作为主工作空间**：它保留率最高且仍有约91%的
   encoded-mean方差；32D适合作为压缩对照，不宜默认替代。
2. 2–8D只用于展示和诊断，不应承担主要地质结论。首先逐PC回归
   `data_source`、缺失token数、publication、SiO2和训练标签。
3. 深入F1/F2/F3时同时报告“低秩共同方向”和被截掉的高秩残差；后者更可能包含
   背景依赖、非线性交互或噪声，不能只保留更漂亮的低维对齐。
4. C1/C2/C3应在32、64D以及完整空间并列复核，并增加publication-held-out和
   真实分析面板控制，因为contextual语义与来源结构都集中在前几十个PC。
5. 优先复核低秩集中且原始混杂风险较低的实验；若某结果只在2–5D出现，应检查
   单一PC与来源变量的关系，不要直接命名为地质过程轴。

## 11. 数据与复现

- [`experiment_dimension_metrics.csv`](experiment_dimension_metrics.csv)：全部逐维原始结果
- [`experiment_dimension_analysis.csv`](experiment_dimension_analysis.csv)：逐实验敏感性分类
- [`all_scalar_metrics_long.csv`](all_scalar_metrics_long.csv)：全部 metrics.json 数值标量长表
- [`scalar_metric_dimension_analysis.csv`](scalar_metric_dimension_analysis.csv)：逐标量指标敏感性
- [`experiment_all_metric_summary.csv`](experiment_all_metric_summary.csv)：逐实验全指标汇总
- [`dimension_summary.csv`](dimension_summary.csv)：逐维总体统计
- [`family_dimension_summary.csv`](family_dimension_summary.csv)：逐家族统计
- [`figure_index.csv`](figure_index.csv)：逐实验维度曲线索引

本轮已登记运行时间合计约 **{runtime:.1f} 秒**。所有实验仍保留各自的配置、样本、
指标、观察、图表与错误记录。
"""
    (reports / "pca_dimension_sweep_report.md").write_text(report, encoding="utf-8")
    coverage = {
        "expected_dimensions": sorted(expected_dimensions, reverse=True),
        "completed_dimensions": sorted(completed_dimensions, reverse=True),
        "expected_experiments_per_dimension": len(expected_experiments),
        "rows_by_dimension": {str(int(k)): int(v) for k, v in counts.items()},
        "total_rows": len(merged),
        "coverage_ok": bool(coverage_ok),
    }
    (reports / "coverage_audit.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[pca-report] rows={len(merged)} coverage_ok={coverage_ok} report={reports / 'pca_dimension_sweep_report.md'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="stage3_6_element2vec_space_exploration/configs/pca_sweep.yaml",
    )
    args = parser.parse_args()
    build(load_yaml(args.config))


if __name__ == "__main__":
    main()
