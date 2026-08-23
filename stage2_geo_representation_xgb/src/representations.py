from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_ELEMENTS = [
    "As",
    "Ni",
    "Cu",
    "Pb",
    "Co",
    "Au",
    "Ag",
    "Bi",
    "Sb",
    "Zn",
    "Se",
    "Te",
    "Tl",
    "Mn",
]


@dataclass
class RepresentationBundle:
    metadata: pd.DataFrame
    values: pd.DataFrame
    selected_elements: list[str]
    eps_by_element: dict[str, float]
    representations: dict[str, pd.DataFrame]
    raw_features: dict[str, pd.DataFrame]
    qc: pd.DataFrame
    feature_definitions: pd.DataFrame


def robust_scale_frame(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.to_numpy(dtype=float)
    med = np.nanmedian(values, axis=0)
    q25 = np.nanpercentile(values, 25, axis=0)
    q75 = np.nanpercentile(values, 75, axis=0)
    iqr = q75 - q25
    iqr[~np.isfinite(iqr) | (iqr == 0)] = 1.0
    imputed = np.where(np.isfinite(values), values, med)
    scaled = (imputed - med) / iqr
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame(scaled, columns=frame.columns, index=frame.index)


def standard_scale_frame(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.to_numpy(dtype=float)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    imputed = np.where(np.isfinite(values), values, mean)
    scaled = (imputed - mean) / std
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame(scaled, columns=frame.columns, index=frame.index)


def compute_eps(values: pd.DataFrame) -> dict[str, float]:
    eps = {}
    for col in values.columns:
        series = pd.to_numeric(values[col], errors="coerce")
        positives = series[np.isfinite(series) & (series > 0)]
        eps[col] = float(positives.min() / 2.0) if len(positives) else 1e-9
        if not math.isfinite(eps[col]) or eps[col] <= 0:
            eps[col] = 1e-9
    return eps


def log10_frame(values: pd.DataFrame, eps_by_element: dict[str, float]) -> pd.DataFrame:
    out = pd.DataFrame(index=values.index)
    for col in values.columns:
        x = pd.to_numeric(values[col], errors="coerce").to_numpy(dtype=float)
        eps = eps_by_element.get(col, 1e-9)
        x = np.where(np.isnan(x), np.nan, np.maximum(x, eps))
        out[col] = np.log10(x)
    return out


def rank_uniform_frame(values: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=values.index)
    for col in values.columns:
        series = pd.to_numeric(values[col], errors="coerce")
        valid = series[np.isfinite(series)]
        if len(valid) == 0:
            out[col] = 0.5
            continue
        ranks = series.rank(method="average", na_option="keep")
        out[col] = ranks / (len(valid) + 1.0)
    out = out.fillna(0.5)
    return standard_scale_frame(out)


def _log_ratio(
    log_values: pd.DataFrame,
    numerator: str,
    denominator: str,
) -> pd.Series:
    return log_values[numerator] - log_values[denominator]


def _safe_sum_log_ratio(
    values: pd.DataFrame,
    group_a: list[str],
    group_b: list[str],
    eps_by_element: dict[str, float],
) -> pd.Series:
    a = values[group_a].sum(axis=1, min_count=1)
    b = values[group_b].sum(axis=1, min_count=1)
    eps_a = sum(eps_by_element.get(el, 1e-9) for el in group_a)
    eps_b = sum(eps_by_element.get(el, 1e-9) for el in group_b)
    a = np.maximum(a.to_numpy(dtype=float), eps_a)
    b = np.maximum(b.to_numpy(dtype=float), eps_b)
    return pd.Series(np.log10(a) - np.log10(b), index=values.index)


def geo_ratio_12(
    values: pd.DataFrame,
    log_values: pd.DataFrame,
    eps_by_element: dict[str, float],
) -> tuple[pd.DataFrame, list[dict]]:
    defs = [
        ("log_Co_Ni", "log10(Co + eps) - log10(Ni + eps)", ["Co"], ["Ni"]),
        ("log_Au_As", "log10(Au + eps) - log10(As + eps)", ["Au"], ["As"]),
        ("log_As_Se", "log10(As + eps) - log10(Se + eps)", ["As"], ["Se"]),
        ("log_Se_Te", "log10(Se + eps) - log10(Te + eps)", ["Se"], ["Te"]),
        ("log_Bi_Te", "log10(Bi + eps) - log10(Te + eps)", ["Bi"], ["Te"]),
        ("log_Sb_Tl", "log10(Sb + eps) - log10(Tl + eps)", ["Sb"], ["Tl"]),
        ("log_Cu_Zn", "log10(Cu + eps) - log10(Zn + eps)", ["Cu"], ["Zn"]),
        ("log_Pb_Zn", "log10(Pb + eps) - log10(Zn + eps)", ["Pb"], ["Zn"]),
        ("log_Ag_Au", "log10(Ag + eps) - log10(Au + eps)", ["Ag"], ["Au"]),
        ("log_Cu_Pb", "log10(Cu + eps) - log10(Pb + eps)", ["Cu"], ["Pb"]),
        ("log_As_Sb", "log10(As + eps) - log10(Sb + eps)", ["As"], ["Sb"]),
    ]
    out = pd.DataFrame(index=values.index)
    rows = []
    for name, formula, group_a, group_b in defs:
        out[name] = _log_ratio(log_values, group_a[0], group_b[0])
        rows.append(
            {
                "representation": "geo_ratio_12",
                "feature": name,
                "formula": formula,
                "positive_group": ",".join(group_a),
                "negative_group": ",".join(group_b),
                "geologic_note": "log element ratio",
            }
        )

    group_a = ["Cu", "Pb", "Zn", "Ag"]
    group_b = ["Au", "As", "Sb", "Tl"]
    out["log_baseMetal_goldPathfinder"] = _safe_sum_log_ratio(values, group_a, group_b, eps_by_element)
    rows.append(
        {
            "representation": "geo_ratio_12",
            "feature": "log_baseMetal_goldPathfinder",
            "formula": "log10(sum(Cu,Pb,Zn,Ag) + eps) - log10(sum(Au,As,Sb,Tl) + eps)",
            "positive_group": ",".join(group_a),
            "negative_group": ",".join(group_b),
            "geologic_note": "base metal suite relative to gold/pathfinder suite",
        }
    )
    return out, rows


def _mean_log(log_values: pd.DataFrame, group: Iterable[str]) -> pd.Series:
    return log_values[list(group)].mean(axis=1, skipna=True)


def geo_balance_12(log_values: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    specs = [
        ("highT_vs_gold", ["Co", "Ni", "Se"], ["Au", "As", "Sb", "Tl"], "high-temperature lattice-compatible suite against Au pathfinders"),
        ("baseMetal_vs_gold", ["Cu", "Pb", "Zn", "Ag"], ["Au", "As", "Sb", "Tl"], "base metals against Au pathfinders"),
        ("semimetal_vs_lattice", ["As", "Sb", "Bi", "Te"], ["Co", "Ni"], "semimetal/telluride suite against Co-Ni lattice proxy"),
        ("lateLowT_vs_highT", ["Sb", "Tl", "Te"], ["Co", "Ni", "Se"], "late low-temperature suite against high-temperature suite"),
        ("AuAs_vs_other_semimetal", ["Au", "As"], ["Sb", "Tl", "Bi", "Te"], "Au-As enrichment against other semimetals"),
        ("telluride_vs_baseMetal", ["Te", "Bi"], ["Cu", "Pb", "Zn", "Ag"], "telluride-related enrichment against base metals"),
        ("CuAsBi_vs_CoNiSe", ["Cu", "As", "Bi"], ["Co", "Ni", "Se"], "Cu-As-Bi overprint against Co-Ni-Se"),
        ("PbZnAg_vs_AuAs", ["Pb", "Zn", "Ag"], ["Au", "As"], "Pb-Zn-Ag suite against Au-As"),
        ("AsSbTl_vs_CoNi", ["As", "Sb", "Tl"], ["Co", "Ni"], "As-Sb-Tl pathfinders against Co-Ni"),
        ("SeTeBi_vs_AsSbTl", ["Se", "Te", "Bi"], ["As", "Sb", "Tl"], "Se-Te-Bi against As-Sb-Tl"),
        ("pyrite_overprint_balance", ["As", "Sb", "Tl"], ["Co", "Ni", "Se"], "overprint-sensitive pathfinders against high-temperature lattice suite"),
        ("Au_enrichment_balance", ["Au"], ["As", "Sb", "Tl", "Te", "Bi"], "Au relative enrichment against associated pathfinders"),
    ]
    out = pd.DataFrame(index=log_values.index)
    rows = []
    for name, plus, minus, note in specs:
        out[name] = _mean_log(log_values, plus) - _mean_log(log_values, minus)
        rows.append(
            {
                "representation": "geo_balance_12",
                "feature": name,
                "formula": f"mean(log10({','.join(plus)})) - mean(log10({','.join(minus)}))",
                "positive_group": ",".join(plus),
                "negative_group": ",".join(minus),
                "geologic_note": note,
            }
        )
    return out, rows


def _share(p: pd.DataFrame, elements: list[str]) -> pd.Series:
    return p[elements].sum(axis=1)


def empirical_tail_prob(values: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=values.index)
    for col in values.columns:
        x = pd.to_numeric(values[col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x)
        probs = np.full(len(x), np.nan, dtype=float)
        if valid.sum() > 0:
            sorted_vals = np.sort(x[valid])
            n = len(sorted_vals)
            idx = np.searchsorted(sorted_vals, x[valid], side="left")
            probs[valid] = (n - idx) / n
        out[col] = probs
    return out


def geo_info_12(
    values: pd.DataFrame,
    log_values: pd.DataFrame,
    eps_by_element: dict[str, float],
) -> tuple[pd.DataFrame, list[dict]]:
    medians = {}
    for col in values.columns:
        finite = pd.to_numeric(values[col], errors="coerce")
        finite = finite[np.isfinite(finite) & (finite > 0)]
        medians[col] = float(finite.median()) if len(finite) else eps_by_element.get(col, 1e-9)
        if not math.isfinite(medians[col]) or medians[col] <= 0:
            medians[col] = eps_by_element.get(col, 1e-9)

    anomaly = pd.DataFrame(index=values.index)
    for col in values.columns:
        baseline = math.log10(medians[col] + eps_by_element.get(col, 1e-9))
        anomaly[col] = np.maximum(0.0, log_values[col] - baseline)
    anomaly = anomaly.fillna(0.0)
    anomaly_sum = anomaly.sum(axis=1)
    p = anomaly.div(anomaly_sum.replace(0, np.nan), axis=0).fillna(0.0)

    p_array = p.to_numpy(dtype=float)
    p_safe = np.where(p_array > 0, p_array, 1.0)
    entropy = -(np.where(p_array > 0, p_array * np.log(p_safe), 0.0)).sum(axis=1)
    d = p.shape[1]
    sorted_p = np.sort(p_array, axis=1)[:, ::-1]
    top3_share = sorted_p[:, : min(3, d)].sum(axis=1)

    tail = empirical_tail_prob(values)
    tail_eps = 1e-12
    rarity_by_el = -np.log(tail + tail_eps)
    rarity_values = rarity_by_el.to_numpy(dtype=float)
    rarity_mean = np.nanmean(rarity_values, axis=1)
    rarity_max = np.nanmax(rarity_values, axis=1)
    top3_rarity = np.full(len(values), np.nan, dtype=float)
    for i, row in enumerate(rarity_values):
        finite = row[np.isfinite(row)]
        if len(finite):
            top = np.sort(finite)[::-1][: min(3, len(finite))]
            top3_rarity[i] = float(np.mean(top))

    out = pd.DataFrame(
        {
            "enrichment_entropy": entropy,
            "normalized_entropy": entropy / math.log(d),
            "effective_elements": np.exp(entropy),
            "top1_share": sorted_p[:, 0],
            "top3_share": top3_share,
            "gold_pathfinder_info_share": _share(p, ["Au", "As", "Sb", "Tl"]),
            "highT_lattice_info_share": _share(p, ["Co", "Ni", "Se"]),
            "base_metal_info_share": _share(p, ["Cu", "Pb", "Zn", "Ag"]),
            "semimetal_telluride_info_share": _share(p, ["As", "Sb", "Bi", "Te", "Se"]),
            "rarity_mean": rarity_mean,
            "rarity_max": rarity_max,
            "rarity_top3_mean": top3_rarity,
        },
        index=values.index,
    )
    rows = [
        ("enrichment_entropy", "-sum(p_i * ln(p_i))", "concentration of positive element anomaly distribution"),
        ("normalized_entropy", "enrichment_entropy / ln(D)", "entropy scaled to 0-1"),
        ("effective_elements", "exp(enrichment_entropy)", "effective number of enriched elements"),
        ("top1_share", "max(p_i)", "dominance of the strongest enriched element"),
        ("top3_share", "sum(top 3 p_i)", "dominance of top three enriched elements"),
        ("gold_pathfinder_info_share", "sum p_i over Au,As,Sb,Tl", "Au pathfinder anomaly share"),
        ("highT_lattice_info_share", "sum p_i over Co,Ni,Se", "high-temperature lattice proxy anomaly share"),
        ("base_metal_info_share", "sum p_i over Cu,Pb,Zn,Ag", "base metal anomaly share"),
        ("semimetal_telluride_info_share", "sum p_i over As,Sb,Bi,Te,Se", "semimetal/telluride anomaly share"),
        ("rarity_mean", "mean(-ln(empirical_tail_prob_i))", "average empirical rarity"),
        ("rarity_max", "max(-ln(empirical_tail_prob_i))", "single-element maximum rarity"),
        ("rarity_top3_mean", "mean(top 3 rarity_i)", "top-three rarity intensity"),
    ]
    defs = [
        {
            "representation": "geo_info_12",
            "feature": name,
            "formula": formula,
            "positive_group": "",
            "negative_group": "",
            "geologic_note": note,
        }
        for name, formula, note in rows
    ]
    return out, defs


def _qc_row(name: str, raw: pd.DataFrame, final: pd.DataFrame) -> dict:
    raw_values = raw.to_numpy(dtype=float)
    final_values = final.to_numpy(dtype=float)
    return {
        "representation": name,
        "n_features": final.shape[1],
        "raw_missing_rate": float(np.isnan(raw_values).mean()),
        "raw_inf_count": int(np.isinf(raw_values).sum()),
        "final_missing_rate": float(np.isnan(final_values).mean()),
        "final_inf_count": int(np.isinf(final_values).sum()),
    }


def build_representations(
    processed_dir: Path,
    selected_elements: list[str] | None = None,
) -> RepresentationBundle:
    metadata = pd.read_csv(processed_dir / "pyrite_metadata.csv")
    values_all = pd.read_csv(processed_dir / "pyrite_element_values.csv")

    selected = selected_elements or DEFAULT_ELEMENTS
    missing = [el for el in selected if el not in values_all.columns]
    if missing:
        raise ValueError(f"Missing required element columns: {missing}")

    sample_ids = values_all["sample_id"].astype(str)
    values = values_all[selected].apply(pd.to_numeric, errors="coerce")
    values.index = sample_ids
    metadata = metadata.copy()
    metadata["sample_id"] = metadata["sample_id"].astype(str)
    metadata = metadata.set_index("sample_id").loc[sample_ids].reset_index()
    values.index = metadata["sample_id"].astype(str)

    eps_by_element = compute_eps(values)
    log_values = log10_frame(values, eps_by_element)

    raw_features: dict[str, pd.DataFrame] = {}
    final_reps: dict[str, pd.DataFrame] = {}
    feature_rows: list[dict] = []

    raw_features["raw_robust_z_14"] = values.copy()
    final_reps["raw_robust_z_14"] = robust_scale_frame(values)
    feature_rows.extend(
        {
            "representation": "raw_robust_z_14",
            "feature": el,
            "formula": f"robust_z(raw {el})",
            "positive_group": el,
            "negative_group": "",
            "geologic_note": "baseline concentration feature",
        }
        for el in selected
    )

    raw_features["log_robust_z_14"] = log_values.copy()
    final_reps["log_robust_z_14"] = robust_scale_frame(log_values)
    feature_rows.extend(
        {
            "representation": "log_robust_z_14",
            "feature": el,
            "formula": f"robust_z(log10({el} + eps))",
            "positive_group": el,
            "negative_group": "",
            "geologic_note": "log concentration baseline",
        }
        for el in selected
    )

    rank_rep = rank_uniform_frame(values)
    raw_features["rank_uniform_z_14"] = rank_rep.copy()
    final_reps["rank_uniform_z_14"] = rank_rep
    feature_rows.extend(
        {
            "representation": "rank_uniform_z_14",
            "feature": el,
            "formula": f"standard_z(percentile_rank({el}))",
            "positive_group": el,
            "negative_group": "",
            "geologic_note": "rank-normalized baseline",
        }
        for el in selected
    )

    ratio_raw, ratio_defs = geo_ratio_12(values, log_values, eps_by_element)
    raw_features["geo_ratio_12"] = ratio_raw
    final_reps["geo_ratio_12"] = robust_scale_frame(ratio_raw)
    feature_rows.extend(ratio_defs)

    balance_raw, balance_defs = geo_balance_12(log_values)
    raw_features["geo_balance_12"] = balance_raw
    final_reps["geo_balance_12"] = robust_scale_frame(balance_raw)
    feature_rows.extend(balance_defs)

    info_raw, info_defs = geo_info_12(values, log_values, eps_by_element)
    raw_features["geo_info_12"] = info_raw
    final_reps["geo_info_12"] = robust_scale_frame(info_raw)
    feature_rows.extend(info_defs)

    qc = pd.DataFrame([_qc_row(name, raw_features[name], final_reps[name]) for name in final_reps])
    feature_definitions = pd.DataFrame(feature_rows)

    return RepresentationBundle(
        metadata=metadata,
        values=values.reset_index(drop=True),
        selected_elements=selected,
        eps_by_element=eps_by_element,
        representations=final_reps,
        raw_features=raw_features,
        qc=qc,
        feature_definitions=feature_definitions,
    )


def save_representations(bundle: RepresentationBundle, out_dir: Path) -> None:
    rep_dir = out_dir / "representations"
    rep_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in bundle.representations.items():
        export = pd.concat([bundle.metadata[["sample_id"]].reset_index(drop=True), frame.reset_index(drop=True)], axis=1)
        export.to_csv(rep_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
        raw_export = pd.concat(
            [bundle.metadata[["sample_id"]].reset_index(drop=True), bundle.raw_features[name].reset_index(drop=True)],
            axis=1,
        )
        raw_export.to_csv(rep_dir / f"{name}_unscaled_features.csv", index=False, encoding="utf-8-sig")

    bundle.qc.to_csv(rep_dir / "representation_qc.csv", index=False, encoding="utf-8-sig")
    bundle.feature_definitions.to_csv(rep_dir / "feature_definitions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {"element": list(bundle.eps_by_element.keys()), "eps": list(bundle.eps_by_element.values())}
    ).to_csv(rep_dir / "eps_by_element.csv", index=False, encoding="utf-8-sig")
