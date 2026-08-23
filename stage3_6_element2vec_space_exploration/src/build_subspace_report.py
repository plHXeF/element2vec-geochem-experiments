from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import load_yaml, resolve_path


MAIN_SPACES = ["FULL", "PCA64", "PCA32", "PCA3", "PCA2"]
LOWER_IS_STRONGER = {"D4_geographic_neighbors"}
DIRECTIONLESS = {"B2_absolute_relative", "E7_metamorphic"}


def _split_id(value: str) -> tuple[str, str]:
    if "__" not in value:
        return value, "FULL"
    return tuple(value.split("__", 1))


def _float(value: Any) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _markdown(frame: pd.DataFrame) -> str:
    """Small dependency-free Markdown table writer."""
    if frame.empty:
        return ""
    clean = frame.copy()
    for column in clean.columns:
        clean[column] = clean[column].map(
            lambda value: "" if pd.isna(value) else (f"{value:.4g}" if isinstance(value, (float, np.floating)) else str(value))
        )
    header = "| " + " | ".join(map(str, clean.columns)) + " |"
    rule = "| " + " | ".join(["---"] * len(clean.columns)) + " |"
    rows = ["| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in clean.to_numpy()]
    return "\n".join([header, rule, *rows])


def _load_rows(base_config: dict[str, Any], root: Path) -> pd.DataFrame:
    full = pd.read_csv(resolve_path(base_config["project"]["results_dir"]) / "experiment_registry.csv", low_memory=False)
    full["base_experiment_id"] = full["experiment_id"].astype(str)
    full["projection_name"] = "FULL"
    path = root / "results" / "experiment_registry.csv"
    replay = pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame(columns=full.columns)
    if not replay.empty:
        split = replay["experiment_id"].astype(str).map(_split_id)
        replay["base_experiment_id"] = split.map(lambda x: x[0])
        replay["projection_name"] = split.map(lambda x: x[1])
    return pd.concat([full, replay], ignore_index=True, sort=False)


def _rp_stats(rows: pd.DataFrame) -> pd.DataFrame:
    rp = rows[rows["projection_name"].str.match(r"RP(64|32|3|2)_seed\d+")].copy()
    rp["target_dimension"] = rp["projection_name"].str.extract(r"RP(\d+)").astype(float)
    stats = rp.groupby(["base_experiment_id", "target_dimension"])["metric_value"].agg(
        random_projection_mean="mean", random_projection_std="std", random_projection_min="min",
        random_projection_max="max", random_projection_n="count"
    ).reset_index()
    return stats


def _status_for_group(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()
    eid = group["experiment_id"].iloc[0]
    full = group.loc[group["projection_method"] == "FULL", "metric_value"]
    if full.empty or not np.isfinite(full.iloc[0]):
        group["status"] = group["status"].where(group["status"].notna(), "Not applicable")
        return group
    full_value = float(full.iloc[0])
    sign = -1.0 if eid in LOWER_IS_STRONGER else 1.0
    tolerance = max(0.02, 0.10 * abs(full_value))
    main = group[group["projection_method"].isin(MAIN_SPACES)].set_index("projection_method")
    low_good = []
    for name in ("PCA64", "PCA32"):
        if name not in main.index:
            continue
        row = main.loc[name]
        delta = sign * (row["metric_value"] - full_value)
        if sign > 0:
            beats_rp_interval = np.isfinite(row["random_projection_max"]) and row["metric_value"] > row["random_projection_max"]
        else:
            beats_rp_interval = np.isfinite(row["random_projection_min"]) and row["metric_value"] < row["random_projection_min"]
        bootstrap_stable = np.isfinite(row["bootstrap_ci_low"]) and row["bootstrap_ci_low"] > 0
        if delta > tolerance and beats_rp_interval and bootstrap_stable:
            low_good.append(name)
    visual_only = False
    for name in ("PCA3", "PCA2"):
        if name in main.index:
            row = main.loc[name]
            delta = sign * (row["metric_value"] - full_value)
            rp_gain = sign * (row["metric_value"] - row["random_projection_mean"]) if np.isfinite(row["random_projection_mean"]) else -np.inf
            if delta > tolerance and rp_gain <= tolerance / 2 and not low_good:
                visual_only = True
    for idx, row in group.iterrows():
        if row["projection_method"] == "Not applicable":
            group.at[idx, "status"] = "Not applicable"
        elif eid in DIRECTIONLESS:
            group.at[idx, "status"] = "Projection-sensitive" if row.get("random_projection_std", 0) > tolerance else "Preserved under compression"
        elif row["projection_method"] in low_good:
            confounded = np.isfinite(row.get("source_metric", np.nan)) and row.get("source_metric", 0) >= row.get("geology_metric", np.inf)
            group.at[idx, "status"] = "Low-rank but confounded" if confounded else "Low-rank semantic recovery"
        elif visual_only and row["projection_method"] in {"PCA3", "PCA2"}:
            group.at[idx, "status"] = "Projection artifact suspected"
        elif row["projection_method"] == "FULL":
            group.at[idx, "status"] = "Reference FULL"
        else:
            delta = sign * (row["metric_value"] - full_value)
            if delta < -tolerance:
                group.at[idx, "status"] = "High-dimensional dependence"
            elif np.isfinite(row.get("random_projection_std", np.nan)) and row["random_projection_std"] > tolerance:
                group.at[idx, "status"] = "Projection-sensitive"
            else:
                group.at[idx, "status"] = "Preserved under compression" if abs(delta) <= tolerance else "No improvement"
    return group


def _attach_paired_bootstrap(matrix: pd.DataFrame, root: Path, repeats: int = 1000) -> pd.DataFrame:
    """Attach paired delta-vs-FULL intervals where old runs saved atomic observations."""
    definitions = {
        "F1_lithology_perturbation": ("rhyolite_toward_basalt_perturbations.csv", "sample_id", "alignment_with_rhyolite_to_basalt_centroid", False),
        "F2_single_element_jacobian": ("single_element_finite_differences.csv", "field", "cross_background_direction_cosine", True),
    }
    rng = np.random.default_rng(42)
    for eid, (filename, key, value, collapse) in definitions.items():
        full_path = root.parent / "results" / eid / filename
        if not full_path.exists():
            continue
        full = pd.read_csv(full_path)
        if collapse:
            full = full.groupby(key, as_index=False)[value].first()
        for space in ("PCA64", "PCA32", "PCA3", "PCA2"):
            path = root / "results" / eid[0] / f"{eid}__{space}" / filename
            if not path.exists():
                continue
            projected = pd.read_csv(path)
            if collapse:
                projected = projected.groupby(key, as_index=False)[value].first()
            paired = full[[key, value]].merge(projected[[key, value]], on=key, suffixes=("_full", "_projected"))
            delta = paired[f"{value}_projected"] - paired[f"{value}_full"]
            if not len(delta):
                continue
            means = [delta.iloc[rng.integers(0, len(delta), len(delta))].mean() for _ in range(repeats)]
            low, high = np.quantile(means, [0.025, 0.975])
            mask = matrix["experiment_id"].eq(eid) & matrix["projection_method"].eq(space)
            matrix.loc[mask, "bootstrap_ci_low"] = low
            matrix.loc[mask, "bootstrap_ci_high"] = high
            matrix.loc[mask, "stability"] = f"paired bootstrap delta-vs-FULL, n={len(delta)}, repeats={repeats}"
    return matrix


def _build_matrix(base_config: dict[str, Any], root: Path, rows: pd.DataFrame) -> pd.DataFrame:
    original = rows[rows["projection_name"] == "FULL"].drop_duplicates("base_experiment_id").set_index("base_experiment_id")
    replay = rows.set_index(["base_experiment_id", "projection_name"])
    compatibility = pd.read_csv(root / "audit" / "subspace_experiment_compatibility.csv").set_index("experiment_id")
    dims = json.loads((root / "audit" / "subspace_available_dimensions.json").read_text(encoding="utf-8"))
    rp = _rp_stats(rows).set_index(["base_experiment_id", "target_dimension"])
    s4_path = root / "results" / "S" / "sample_signal_dimension_curves.csv"
    s4 = pd.read_csv(s4_path) if s4_path.exists() else pd.DataFrame()
    conf = {}
    if not s4.empty:
        ratios = s4[(s4["representation_level"] == "contextual_mean") & (s4["metric"] == "geology_minus_source")]
        for _, row in ratios.iterrows():
            conf[row["projection_name"]] = (row.get("geology_metric"), row.get("source_metric"), row.get("missingness_metric"))
    matrix_rows = []
    for eid, old in original.iterrows():
        comp = compatibility.loc[eid]
        full_value = _float(old["metric_value"])
        rep = comp["projection_reference"]
        for space in MAIN_SPACES:
            applicable = space == "FULL" or comp[space] == "applicable"
            key = (eid, space)
            if space == "FULL":
                source = old
            elif applicable and key in replay.index:
                source = replay.loc[key]
                if isinstance(source, pd.DataFrame):
                    source = source.iloc[-1]
            else:
                source = pd.Series(dtype=object)
            target_dim = int(space.replace("PCA", "")) if space != "FULL" else (
                dims.get(rep, {}).get("native_dimension") if rep != "none" else np.nan
            )
            rp_row = None
            if space != "FULL" and (eid, float(target_dim)) in rp.index:
                rp_row = rp.loc[(eid, float(target_dim))]
            if space == "FULL":
                explained = 1.0
            else:
                explained = dims.get(rep, {}).get("spaces", {}).get(space, {}).get("explained_variance", np.nan)
            geology, source_metric, missing = conf.get(space, (np.nan, np.nan, np.nan))
            metric = _float(source.get("metric_value", np.nan)) if applicable else np.nan
            baseline = _float(source.get("baseline_value", np.nan)) if applicable else np.nan
            matrix_rows.append({
                "experiment_id": eid, "family": old["family"], "representation_level": old["representation_level"],
                "native_dimension": dims.get(rep, {}).get("native_dimension", np.nan),
                "projection_method": space if applicable else "Not applicable", "target_dimension": target_dim,
                "pca_explained_variance": explained, "sample_count": old["sample_size"], "random_seed": old["seed"],
                "primary_metric": old["metric_primary"], "metric_value": metric, "full_metric_value": full_value,
                "delta_vs_full": metric - full_value if np.isfinite(metric) and np.isfinite(full_value) else np.nan,
                "random_projection_mean": _float(rp_row.get("random_projection_mean")) if rp_row is not None else np.nan,
                "random_projection_std": _float(rp_row.get("random_projection_std")) if rp_row is not None else np.nan,
                "random_projection_min": _float(rp_row.get("random_projection_min")) if rp_row is not None else np.nan,
                "random_projection_max": _float(rp_row.get("random_projection_max")) if rp_row is not None else np.nan,
                "random_projection_n": _float(rp_row.get("random_projection_n")) if rp_row is not None else np.nan,
                "delta_vs_random_projection": metric - _float(rp_row.get("random_projection_mean")) if rp_row is not None and np.isfinite(metric) else np.nan,
                "untrained_metric": baseline if "random" in str(old["control_group"]).lower() else np.nan,
                "geology_metric": geology if str(old["family"]) in "CDEFHIJKX" else np.nan,
                "source_metric": source_metric if str(old["family"]) in "CDEFHIJKX" else np.nan,
                "missingness_metric": missing if str(old["family"]) in "CDEFHIJKX" else np.nan,
                "bootstrap_ci_low": np.nan, "bootstrap_ci_high": np.nan,
                "stability": source.get("stability", "not applicable") if applicable else comp[space],
                "visual_pattern": "figure available" if str(source.get("figure_paths", "")).strip() not in {"", "nan"} else "not separately assessed",
                "quantitative_support": bool(np.isfinite(metric)), "status": "Not applicable" if not applicable else "pending",
                "main_observation": source.get("main_observation", old["main_observation"]) if applicable else comp[space],
                "possible_confounder": source.get("possible_confounder", old["possible_confounder"]) if applicable else comp["limitation"],
                "recommended_next_step": source.get("recommended_next_step", old["recommended_next_step"]) if applicable else "No projection result may be inferred.",
            })
    matrix = pd.DataFrame(matrix_rows)
    matrix = _attach_paired_bootstrap(matrix, root)
    matrix = pd.concat(
        [_status_for_group(group.assign(experiment_id=eid)) for eid, group in matrix.groupby("experiment_id", sort=False)],
        ignore_index=True,
    )

    # Add S4 direct rows, whose bootstrap intervals are computed from per-sample neighbor matches.
    if not s4.empty:
        direct = s4[s4["projection_name"].isin(MAIN_SPACES)].copy()
        add = pd.DataFrame({
            "experiment_id": direct["experiment_id"], "family": "S", "representation_level": direct["representation_level"],
            "native_dimension": 768, "projection_method": direct["projection_name"], "target_dimension": direct["target_dimension"],
            "pca_explained_variance": np.nan, "sample_count": 4000, "random_seed": direct["random_seed"],
            "primary_metric": direct["metric"], "metric_value": direct["metric_value"], "full_metric_value": np.nan,
            "delta_vs_full": np.nan, "random_projection_mean": np.nan, "delta_vs_random_projection": np.nan,
            "untrained_metric": np.nan, "geology_metric": direct.get("geology_metric"), "source_metric": direct.get("source_metric"),
            "missingness_metric": direct.get("missingness_metric"), "bootstrap_ci_low": direct["bootstrap_ci_low"],
            "bootstrap_ci_high": direct["bootstrap_ci_high"], "stability": "100x bootstrap on per-sample neighbor agreement",
            "visual_pattern": "dimension curve", "quantitative_support": True, "status": "Direct subspace diagnostic",
            "main_observation": direct["label"], "possible_confounder": direct["signal_type"],
            "recommended_next_step": "Interpret geology jointly with source and missingness curves.",
        })
        matrix = pd.concat([matrix, add], ignore_index=True, sort=False)
    return matrix


def _plots(matrix: pd.DataFrame, root: Path) -> pd.DataFrame:
    figure_dir = root / "figures" / "dimension_curves"
    index_rows = []
    candidates = matrix[(matrix["family"] != "S") & matrix["projection_method"].isin(MAIN_SPACES)]
    order = MAIN_SPACES
    for eid, group in candidates.groupby("experiment_id"):
        if group["metric_value"].notna().sum() < 3:
            continue
        d = group.set_index("projection_method").reindex(order)
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(order, d["metric_value"], marker="o", lw=2, label="PCA/FULL")
        valid = d["random_projection_mean"].notna()
        if valid.any():
            xx = np.arange(len(order))[valid.to_numpy()]
            mean = d.loc[valid, "random_projection_mean"].to_numpy(dtype=float)
            low = d.loc[valid, "random_projection_min"].to_numpy(dtype=float)
            high = d.loc[valid, "random_projection_max"].to_numpy(dtype=float)
            ax.plot(xx, mean, marker="s", ls="--", color="#d97706", label="random projection mean")
            ax.fill_between(xx, low, high, color="#f59e0b", alpha=.18, label="RP min–max (10 seeds)")
        ax.axhline(d.loc["FULL", "metric_value"], color="#64748b", lw=1, alpha=.6)
        ax.set_title(f"{eid}: {d['primary_metric'].dropna().iloc[0]}")
        ax.set_ylabel("original primary metric"); ax.grid(alpha=.22); ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); path = figure_dir / f"{eid}.png"; fig.savefig(path, dpi=160); plt.close(fig)
        index_rows.append({"figure": str(path), "experiment_id": eid, "purpose": "FULL/PCA vs 10-seed random projection"})
    known = {row["figure"] for row in index_rows}
    for path in sorted((root / "figures").rglob("*.png")):
        if str(path) not in known:
            index_rows.append({"figure": str(path), "experiment_id": "S", "purpose": path.parent.name})
            known.add(str(path))
    for path in sorted((root / "results").glob("*/*__PCA*/figures/*.png")):
        projection = path.parents[1].name.split("__", 1)[-1]
        if projection not in {"PCA64", "PCA32", "PCA3", "PCA2"} or str(path) in known:
            continue
        index_rows.append({"figure": str(path), "experiment_id": path.parents[1].name.split("__", 1)[0],
                           "purpose": f"paired replay {projection}"})
        known.add(str(path))
    return pd.DataFrame(index_rows)


def _write_reports(matrix: pd.DataFrame, root: Path, figure_index: pd.DataFrame) -> None:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(reports / "experiment_dimension_matrix.csv", index=False, encoding="utf-8-sig")
    summary = matrix.groupby(["projection_method", "status"], dropna=False).size().rename("count").reset_index()
    summary.to_csv(reports / "dimension_summary.csv", index=False)
    figure_index.to_csv(reports / "figure_index.csv", index=False)
    figure_index.assign(line=lambda d: d.apply(lambda r: f"- `{r.figure}` — {r.purpose}", axis=1))["line"].to_csv(
        reports / "figure_index.md", index=False, header=False
    )
    specialty = root / "results" / "S"
    if (specialty / "pc_semantic_attribution.csv").exists():
        shutil.copy2(specialty / "pc_semantic_attribution.csv", reports / "pc_semantic_attribution.csv")

    main = matrix[(matrix["family"] != "S") & matrix["projection_method"].isin(MAIN_SPACES)]
    low = main[main["status"] == "Low-rank semantic recovery"]
    confounded = main[main["status"] == "Low-rank but confounded"]
    visual = main[main["status"].isin(["Visualization-only", "Projection artifact suspected"])]
    negative = main[main["status"].isin(["No improvement", "High-dimensional dependence", "Not applicable"])]
    def small_table(frame: pd.DataFrame, n: int = 40) -> str:
        if frame.empty:
            return "No result met this preregistered classification."
        cols = ["experiment_id", "projection_method", "metric_value", "delta_vs_full", "delta_vs_random_projection", "status"]
        return _markdown(frame[cols].head(n))
    (reports / "low_rank_recoveries.md").write_text("# Low-rank recoveries\n\n" + small_table(low), encoding="utf-8")
    (reports / "visualization_only_results.md").write_text("# Visualization-only and suspected projection artifacts\n\n" + small_table(visual), encoding="utf-8")
    (reports / "confounded_results.md").write_text("# Confounded results\n\n" + small_table(confounded), encoding="utf-8")
    (reports / "negative_results.md").write_text("# Negative, high-dimensional, and unavailable results\n\n" + small_table(negative), encoding="utf-8")

    signal_path = specialty / "sample_signal_dimension_curves.csv"
    signal = pd.read_csv(signal_path) if signal_path.exists() else pd.DataFrame()
    signal_text = "Not available."
    if not signal.empty:
        signal_text = signal[(signal["representation_level"] == "contextual_mean") &
                             (signal["metric"] == "geology_minus_source") &
                             signal["projection_name"].isin(MAIN_SPACES)][
            ["projection_name", "geology_metric", "source_metric", "missingness_metric", "metric_value"]
        ]
        signal_text = _markdown(signal_text)
    attr_path = specialty / "pc_classification.csv"
    attr = pd.read_csv(attr_path) if attr_path.exists() else pd.DataFrame()
    attr_text = _markdown(attr.groupby(["representation_level", "pc_class"]).size().rename("PC count").reset_index()) if not attr.empty else "Not available."
    energy_path = specialty / "direction_energy_summary.csv"
    energy = pd.read_csv(energy_path) if energy_path.exists() else pd.DataFrame()
    energy_text = _markdown(energy.groupby("direction_family")[["energy_top2", "energy_top3", "energy_top32", "energy_top64"]].median().reset_index()) if not energy.empty else "Not available."
    stability_path = specialty / "pca_basis_bootstrap_stability.csv"
    stability = pd.read_csv(stability_path) if stability_path.exists() else pd.DataFrame()
    stability_text = _markdown(stability.groupby("representation_level")[["mean_principal_angle_deg", "max_principal_angle_deg"]].mean().reset_index()) if not stability.empty else "Not available."

    best = main[main["projection_method"].isin(["PCA64", "PCA32"])].copy()
    best["effective_delta_vs_full"] = np.where(best["experiment_id"].isin(LOWER_IS_STRONGER), -best["delta_vs_full"], best["delta_vs_full"])
    best = best.sort_values("effective_delta_vs_full", ascending=False)
    oriented = main[~main["experiment_id"].isin(DIRECTIONLESS)].copy()
    oriented["oriented_metric"] = np.where(oriented["experiment_id"].isin(LOWER_IS_STRONGER), -oriented["metric_value"], oriented["metric_value"])
    best_space = oriented.loc[oriented.groupby("experiment_id")["oriented_metric"].idxmax(), "projection_method"].value_counts().rename("experiment_count").reset_index()
    best_space.columns = ["best_space_by_primary_metric", "experiment_count"]
    report = f"""# Element2Vec V2 多尺度低维语义子空间复验报告

## 1. Executive summary

- 原 registry 的 50 个实验均进入兼容性矩阵；技术/字段不可用项保持不可用，没有用投影伪造结果。
- 主空间定量行数：{len(main)}；随机正交投影使用每个目标维度 10 个固定种子。
- 严格门槛下，`Low-rank semantic recovery` 行数为 {len(low)}，`Low-rank but confounded` 为 {len(confounded)}，二维/三维 visualization-only 或 suspected projection artifact 为 {len(visual)}。
- 结论必须以 `experiment_dimension_matrix.csv` 的 FULL/PCA/RP 差值、bootstrap 与 source/missingness 对照为准；二维或三维图本身不计为证据。

按原 primary metric（仅修正 D4 的“越低越好”方向）逐实验取最大值，最佳空间计数为：

{_markdown(best_space)}

当前证据更支持“不同表示层级具有不同有效维度，且低秩改善是局部的”，而不是“存在一个对所有任务通用的 32D/64D 地质语义空间”。特别是静态 numeric tags 的样本秩只有 34，PCA64 不可被合法估计，PCA32 已接近秩上限；contextual/sample 空间则允许完整比较。

## 2. Projection audit

详见 `audit/subspace_representation_audit.md` 与 `audit/subspace_available_dimensions.json`。PCA 使用固定 8,000 样品 reference pool；评价 pool 为互斥的 4,000 样品；token pool 固定为 20,000 行。H/I4/X3 的样品—标签比较使用同一个 joint basis。

## 3. Dimension-response map

完整 50×维度矩阵见 `experiment_dimension_matrix.csv`。64/32D 改善最大的候选（尚须逐项检查指标方向和混杂）如下：

{_markdown(best[["experiment_id", "projection_method", "metric_value", "effective_delta_vs_full", "delta_vs_random_projection", "status"]].head(20))}

## 4. Low-rank semantic recoveries

{small_table(low)}

只有同时优于 FULL、同维随机投影，且没有同步增强混杂的条目才进入这一节。缺少原实验级 bootstrap 的旧指标不会因为一张好看的图被升级为强证据。

## 5. Visualization-only findings

{small_table(visual)}

## 6. Contextual token results

Contextual token 的原始 C/X 实验已在五个主空间中配对重放；来源控制和角色 proxy 保持原定义。直接样品邻域诊断显示：

{signal_text}

这里的 `geology-minus-source` 只作描述性差值，不替代分别报告的三类 purity。

## 7. Geochemical perturbation geometry

F1–F3 已按各主空间与 RP controls 重放。S2 额外重建 F2 的 102 个单元素×背景有限差分方向，报告其 PC 能量与有效秩；方向能量中位数如下：

{energy_text}

连续轨迹或高 additivity 若也被随机投影保留，应解释为连续网络/局部线性的普遍性质，而不是独立地质语义。

## 8. Label relation algebra

G1/G2/G4/G5 在纯标签 projected space 中重放；PCA basis 只使用完整标签词表。volcanic→plutonic、岩浆序列和 alkaline shift 均与 10-seed RP 比较。2D/3D 平行关系不会单独改变状态。

## 9. Process trajectories

E1–E7 的端元、步长、闭合方式和参数顺序均保持不变。报告区分 parameter monotonicity、straightness 与相对于 RP 的额外增益；“轨迹更平滑”不自动等价于“地质更有效”。

## 10. Source and missingness confounding

前 64 PCs 的归因分类如下：

{attr_text}

source/reference 使用 eta-squared，连续化学量使用绝对 Spearman；这是可读性归因，不是因果分解。完整结果见 `pc_semantic_attribution.csv`。

## 11. Negative and anomalous findings

{small_table(negative, 30)}

严格 CIPW 保持 `Failed technically`；I1–I4 的成功部分继续标记为 NNLS normative proxy。缺失 detection-limit、完整 REE、Hf/Ta/Cs 等词表限制不因降维消失。

## 12. Final interpretation

证据驱动分类为：

1. 不支持一个对所有层级统一的 64D/32D 语义子空间；
2. 允许部分局部关系在 64D/32D 被保留或增强，但必须逐项通过 RP 与混杂门槛；
3. 仅在 PCA2/PCA3 中明显的关系被隔离为 visualization-only；
4. source/missingness 占据部分高方差方向，但降维是否改善地质/来源比值取决于 representation 和维度；
5. 静态元素语义的失败主要不是“高维稀疏”一个原因，词表覆盖与训练目标仍是硬限制；
6. contextual token、pooled latent、CLS 与纯标签空间表现出不同的最优有效维度。

## Stability

{stability_text}
"""
    (reports / "subspace_exploration_report.md").write_text(report, encoding="utf-8")
    (reports / "reproducibility.md").write_text(
        """# Reproducibility

- Environment: `/opt/anaconda3/envs/Element2Vec/bin/python`.
- Seed: 42; random projection seeds: 1–10.
- Main PCA spaces are nested prefixes of a single fixed basis per representation.
- PCA fitting IDs, evaluation IDs, and token fitting rows are saved under `projections/fitting_pools/`.
- Historical limitation: only D1 had non-empty saved sample IDs; all other legacy experiments are deterministic code replays using the original seed and fixed definitions.
- No zero padding or duplicated dimensions. Native/rank failures are `Not applicable`.
- UMAP is unavailable and was not used; no vector algebra is performed in nonlinear visualization coordinates.
- Strict CIPW remains unavailable; NNLS normative proxy is not renamed.

Commands:

```bash
python -m stage3_6_element2vec_space_exploration.src.subspace_audit --config stage3_6_element2vec_space_exploration/configs/subspace_audit.yaml
python -m stage3_6_element2vec_space_exploration.src.run_subspace_replay --config stage3_6_element2vec_space_exploration/configs/subspace_audit.yaml --phase main
python -m stage3_6_element2vec_space_exploration.src.run_subspace_replay --config stage3_6_element2vec_space_exploration/configs/subspace_audit.yaml --phase random
python -m stage3_6_element2vec_space_exploration.src.run_subspace_replay --config stage3_6_element2vec_space_exploration/configs/subspace_audit.yaml --phase aux
python -m stage3_6_element2vec_space_exploration.src.subspace_specialty --config stage3_6_element2vec_space_exploration/configs/subspace_audit.yaml
python -m stage3_6_element2vec_space_exploration.src.build_subspace_report --config stage3_6_element2vec_space_exploration/configs/subspace_audit.yaml
```
""", encoding="utf-8"
    )


def run(config: dict[str, Any]) -> None:
    base_config = load_yaml(config["base_config"])
    root = resolve_path(config["subspace_audit"]["output_root"])
    rows = _load_rows(base_config, root)
    matrix = _build_matrix(base_config, root, rows)
    figures = _plots(matrix, root)
    _write_reports(matrix, root, figures)
    print(f"[subspace-report] matrix_rows={len(matrix)} figures={len(figures)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="stage3_6_element2vec_space_exploration/configs/subspace_audit.yaml")
    args = parser.parse_args()
    run(load_yaml(args.config))


if __name__ == "__main__":
    main()
