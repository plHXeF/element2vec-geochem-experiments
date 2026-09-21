from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from geochem_workspace import require_run_dir, output_path, asset, REPO_ROOT
if __name__ == "__main__":
    require_run_dir()

import html
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRESENTATION_DIR = output_path("presentation")
ROOT = REPO_ROOT
ASSET_DIR = PRESENTATION_DIR / "assets"
FIG_DIR = ASSET_DIR / "figures"
TABLE_DIR = ASSET_DIR / "tables"
INTERACTIVE_DIR = PRESENTATION_DIR / "interactive"
STAGE2_INTERACTIVE_DIR = INTERACTIVE_DIR / "copied_stage2_umap3d_htmls"
STAGE3_INTERACTIVE_DIR = INTERACTIVE_DIR / "copied_stage3_umap3d_htmls"

FOOTER = "Pyrite geochemistry representation benchmark | Stage 1-3 exploratory report"
CHAPTERS = [
    "背景：表示问题",
    "Stage 1：统计 baseline",
    "Stage 2：人工地质表示",
    "Stage 3：Element2Vec",
    "未来方向：统一过程空间",
]

COLORS = {
    "ink": "#16202a",
    "muted": "#667085",
    "line": "#d8dee8",
    "blue": "#2563eb",
    "teal": "#0891b2",
    "green": "#16a34a",
    "orange": "#f59e0b",
    "red": "#dc2626",
    "purple": "#7c3aed",
}


def ensure_dirs() -> None:
    for path in [ASSET_DIR, FIG_DIR, TABLE_DIR, INTERACTIVE_DIR, STAGE2_INTERACTIVE_DIR, STAGE3_INTERACTIVE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    return path.relative_to(PRESENTATION_DIR).as_posix()


def copy_if_exists(src: Path, dst: Path) -> Path | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def fmt(x: object, digits: int = 3) -> str:
    try:
        if pd.isna(x):
            return "-"
        return f"{float(x):.{digits}f}"
    except Exception:
        return html.escape(str(x))


def simple_table(df: pd.DataFrame, columns: list[str], headers: list[str] | None = None, digits: int = 3) -> str:
    if df.empty:
        return '<div class="placeholder">Table not found.</div>'
    headers = headers or columns
    thead = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    rows = []
    for _, row in df.iterrows():
        cells = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, (int, float, np.floating)):
                cells.append(f"<td>{fmt(value, digits)}</td>")
            else:
                cells.append(f"<td>{html.escape(str(value))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="data-table"><thead><tr>{thead}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def svg_text(x: float, y: float, text: str, size: int = 22, weight: int = 500, fill: str = "#16202a") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{html.escape(text)}</text>'
    )


def write_report_logic_flow_svg() -> None:
    labels = [
        ("地质对象如何编码？", "核心问题"),
        ("表示定义与类比", "representation thinking"),
        ("六维评价体系", "统一 benchmark"),
        ("Stage1 baseline", "raw / log / rank"),
        ("Stage2 地质表示", "ratio / balance / info"),
        ("Stage3 Element2Vec", "element token model"),
        ("统一地质过程空间", "长期目标"),
    ]
    w, h = 1600, 380
    xs = np.linspace(120, 1480, len(labels))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
        f'<rect width="{w}" height="{h}" rx="18" fill="#f8fafc"/>',
    ]
    for i, (x, (title, sub)) in enumerate(zip(xs, labels)):
        color = ["#2563eb", "#0891b2", "#16a34a", "#f59e0b", "#7c3aed", "#dc2626", "#0f766e"][i]
        parts.append(f'<rect x="{x-95}" y="96" width="190" height="92" rx="20" fill="#ffffff" stroke="#d8dee8" stroke-width="2"/>')
        parts.append(f'<circle cx="{x}" cy="86" r="22" fill="{color}" opacity="0.95"/>')
        parts.append(svg_text(x - 74, 134, title, 18, 800))
        parts.append(svg_text(x - 64, 165, sub, 14, 500, "#667085"))
        if i < len(labels) - 1:
            nx = xs[i + 1]
            parts.append(
                f'<path d="M {x + 104} 142 L {nx - 108} 142" stroke="#94a3b8" stroke-width="4" marker-end="url(#arrow)"/>'
            )
    parts.insert(
        2,
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">'
        '<path d="M0,0 L0,6 L9,3 z" fill="#94a3b8"/></marker></defs>',
    )
    parts.append("</svg>")
    write_text(FIG_DIR / "report_logic_flow.svg", "\n".join(parts))


def write_six_dimension_svg() -> None:
    items = [
        ("预测有效性", "对下游任务有没有帮助", "已量化"),
        ("局部邻域一致性", "相近样本是否地质意义相近", "已量化"),
        ("全局结构组织", "是否形成过程域或簇", "已量化"),
        ("原始信息保真", "是否保留元素信息", "待完善"),
        ("地质指标连续性", "proxy 是否连续可解释", "待完善"),
        ("泛化稳定性", "换划分/数据是否稳定", "待完善"),
    ]
    w, h = 1080, 560
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
        '<rect width="1080" height="560" rx="20" fill="#f8fafc"/>',
    ]
    for i, (title, desc, state) in enumerate(items):
        col, row = i % 3, i // 3
        x, y = 60 + col * 335, 85 + row * 210
        color = "#2563eb" if i < 3 else "#94a3b8"
        parts.append(f'<rect x="{x}" y="{y}" width="290" height="150" rx="18" fill="#ffffff" stroke="#d8dee8" stroke-width="2"/>')
        parts.append(f'<circle cx="{x+34}" cy="{y+42}" r="18" fill="{color}"/>')
        parts.append(svg_text(x + 68, y + 48, title, 24, 800))
        parts.append(svg_text(x + 24, y + 90, desc, 18, 500, "#475467"))
        parts.append(
            f'<rect x="{x+24}" y="{y+112}" width="86" height="26" rx="13" fill="{color}" opacity="0.12"/>'
            + svg_text(x + 38, y + 132, state, 15, 700, color)
        )
    parts.append("</svg>")
    write_text(FIG_DIR / "six_dimension_framework.svg", "\n".join(parts))


def write_element2vec_architecture_svg() -> None:
    w, h = 1320, 600
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
        '<rect width="1320" height="600" rx="20" fill="#f8fafc"/>',
        '<defs><marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="4" orient="auto">'
        '<path d="M0,0 L0,8 L11,4 z" fill="#64748b"/></marker></defs>',
    ]

    def box(x: int, y: int, ww: int, hh: int, title: str, sub: str, fill: str = "#ffffff") -> None:
        parts.append(f'<rect x="{x}" y="{y}" width="{ww}" height="{hh}" rx="16" fill="{fill}" stroke="#cbd5e1" stroke-width="2"/>')
        parts.append(svg_text(x + 22, y + 38, title, 22, 800))
        for j, line in enumerate(sub.split("\n")):
            parts.append(svg_text(x + 22, y + 72 + j * 26, line, 17, 500, "#475467"))

    def arrow(x1: int, y1: int, x2: int, y2: int) -> None:
        parts.append(f'<path d="M{x1},{y1} L{x2},{y2}" stroke="#64748b" stroke-width="3" marker-end="url(#arrow)"/>')

    parts.append(svg_text(52, 54, "Element2Vec architecture", 30, 900, "#16202a"))
    box(65, 115, 300, 150, "元素输入", "element embedding 32d\n+ abundance + mask\n每个元素形成 34d 输入", "#eff6ff")
    box(65, 330, 300, 150, "矿床输入", "deposit embedding 32d\n+ 0 + label-mask flag\n形成 34d 输入", "#fff7ed")
    box(455, 205, 250, 155, "共享逻辑：投影", "元素与矿床分别过 MLP\n34d -> 64d token\n进入同一 token 空间", "#ffffff")
    box(790, 178, 230, 205, "Token sequence", "14 个元素 token\n+ 1 个矿床 token\n样品内上下文", "#f0fdf4")
    box(1080, 188, 185, 185, "Transformer", "self-attention\n学习元素关系\n输出 contextual token", "#f5f3ff")
    box(790, 450, 230, 90, "训练任务", "重构被 mask 元素\n预测被 mask 类型")
    box(1080, 450, 185, 90, "主评估", "强制 mask 类型\nmean element latent")
    arrow(365, 190, 455, 248)
    arrow(365, 405, 455, 316)
    arrow(705, 282, 790, 282)
    arrow(1020, 282, 1080, 282)
    arrow(905, 383, 905, 450)
    arrow(1173, 373, 1173, 450)
    parts.append("</svg>")
    write_text(FIG_DIR / "element2vec_architecture.svg", "\n".join(parts))


def save_stage2_ability_profile(unsup: pd.DataFrame, xgb: pd.DataFrame) -> None:
    if unsup.empty or xgb.empty:
        write_placeholder_svg(FIG_DIR / "stage2_ability_profile.svg", "Stage2 metrics not found")
        return
    reps = ["raw_robust_z_14", "log_robust_z_14", "rank_uniform_z_14", "geo_ratio_12", "geo_balance_12", "geo_info_12"]
    strat = xgb[xgb["cv_scheme"].eq("stratified_kfold")].copy()
    merged = unsup.merge(strat[["representation", "macro_f1"]], on="representation", how="left")
    merged = merged[merged["representation"].isin(reps)].set_index("representation").loc[reps].reset_index()
    metrics = pd.DataFrame(
        {
            "Representation": [r.replace("_14", "").replace("_12", "") for r in merged["representation"]],
            "Local neighborhood": merged["input_knn15"].astype(float),
            "Global clustering": merged[["kmeans_nmi", "gmm_nmi"]].astype(float).mean(axis=1),
            "Supervised classification": merged["macro_f1"].astype(float),
        }
    )
    plot_df = metrics.copy()
    for col in ["Local neighborhood", "Global clustering", "Supervised classification"]:
        values = plot_df[col].to_numpy(float)
        span = values.max() - values.min()
        plot_df[col] = 0.12 + 0.88 * (values - values.min()) / span if span else 1.0

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    x = np.arange(len(plot_df))
    width = 0.24
    cols = ["Local neighborhood", "Global clustering", "Supervised classification"]
    colors = ["#2563eb", "#16a34a", "#f59e0b"]
    for i, col in enumerate(cols):
        ax.bar(x + (i - 1) * width, plot_df[col], width=width, label=col, color=colors[i], alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Representation"], rotation=20, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Normalized capability")
    ax.set_title("Stage2 representation capability profile")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stage2_ability_profile.svg", bbox_inches="tight")
    fig.savefig(FIG_DIR / "stage2_ability_profile.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    save_table(metrics, "stage2_ability_profile_table.csv")


def save_stage3_space_comparison(stage3: pd.DataFrame) -> None:
    if stage3.empty:
        write_placeholder_svg(FIG_DIR / "stage3_space_comparison.svg", "Stage3 summary not found")
        return
    wanted = [
        "log_robust_space",
        "sample_input_token_flat_pca12_space",
        "sample_encoded_token_flat_pca12_space",
        "transformer_latent_label_masked_space",
    ]
    df = stage3[stage3["space_name"].isin(wanted)].set_index("space_name").loc[wanted].reset_index()
    labels = ["log robust", "input token PCA12", "encoded token PCA12", "label-masked latent"]
    metrics = ["knn15_neighborhood_purity", "kmeans_nmi", "kmeans_ari", "xgb_macro_f1_mean"]
    names = ["kNN15 purity", "KMeans NMI", "KMeans ARI", "XGB macro-F1"]
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    x = np.arange(len(labels))
    width = 0.19
    colors = ["#2563eb", "#0891b2", "#16a34a", "#f59e0b"]
    for i, (metric, name) in enumerate(zip(metrics, names)):
        ax.bar(x + (i - 1.5) * width, df[metric].astype(float), width=width, label=name, color=colors[i], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Metric value")
    ax.set_title("Stage3 benchmark across representation spaces")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stage3_space_comparison.svg", bbox_inches="tight")
    fig.savefig(FIG_DIR / "stage3_space_comparison.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def radar_plot(
    values: pd.DataFrame,
    label_col: str,
    metric_cols: list[str],
    metric_labels: list[str],
    out_name: str,
    title: str,
    normalize: bool = False,
) -> None:
    if values.empty:
        write_placeholder_svg(FIG_DIR / f"{out_name}.svg", f"{title} metrics not found")
        return
    plot_df = values[[label_col] + metric_cols].copy()
    for col in metric_cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce").fillna(0.0)
        if normalize:
            arr = plot_df[col].to_numpy(float)
            span = arr.max() - arr.min()
            plot_df[col] = 0.10 + 0.90 * (arr - arr.min()) / span if span else 1.0

    n = len(metric_cols)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]
    fig = plt.figure(figsize=(8.2, 7.2))
    ax = plt.subplot(111, polar=True)
    palette = ["#2563eb", "#16a34a", "#f59e0b", "#7c3aed", "#0891b2", "#dc2626"]
    for i, row in plot_df.iterrows():
        vals = [float(row[col]) for col in metric_cols]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2.1, color=palette[i % len(palette)], label=str(row[label_col]))
        ax.fill(angles, vals, color=palette[i % len(palette)], alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.tick_params(axis="x", pad=16)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=9, color="#667085")
    ax.grid(alpha=0.32)
    ax.set_title(title, pad=20, fontsize=18, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=10)
    fig.subplots_adjust(left=0.16, right=0.84, top=0.82, bottom=0.20)
    fig.savefig(FIG_DIR / f"{out_name}.svg", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{out_name}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_stage1_baseline_radar(unsup: pd.DataFrame, xgb: pd.DataFrame) -> None:
    if unsup.empty or xgb.empty:
        write_placeholder_svg(FIG_DIR / "stage1_baseline_radar.svg", "Stage1 metrics not found")
        return
    reps = ["raw_robust_z_14", "log_robust_z_14", "rank_uniform_z_14"]
    strat = xgb[xgb["cv_scheme"].eq("stratified_kfold")]
    df = unsup.merge(strat[["representation", "macro_f1"]], on="representation", how="left")
    df = df[df["representation"].isin(reps)].set_index("representation").loc[reps].reset_index()
    df["label"] = df["representation"].str.replace("_14", "", regex=False)
    radar_plot(
        df,
        "label",
        ["input_knn15", "kmeans_nmi", "macro_f1"],
        ["local\nkNN15", "global\nNMI", "XGB\nmacro-F1"],
        "stage1_baseline_radar",
        "Stage1 baseline ability profile",
        normalize=False,
    )


def save_stage2_ability_radar(unsup: pd.DataFrame, xgb: pd.DataFrame) -> None:
    if unsup.empty or xgb.empty:
        write_placeholder_svg(FIG_DIR / "stage2_ability_radar.svg", "Stage2 metrics not found")
        return
    reps = ["log_robust_z_14", "rank_uniform_z_14", "geo_ratio_12", "geo_balance_12", "geo_info_12"]
    strat = xgb[xgb["cv_scheme"].eq("stratified_kfold")]
    df = unsup.merge(strat[["representation", "macro_f1"]], on="representation", how="left")
    df = df[df["representation"].isin(reps)].set_index("representation").loc[reps].reset_index()
    df["global_cluster"] = df[["kmeans_nmi", "gmm_nmi"]].astype(float).mean(axis=1)
    df["label"] = df["representation"].str.replace("_14", "", regex=False).str.replace("_12", "", regex=False)
    radar_plot(
        df,
        "label",
        ["input_knn15", "global_cluster", "macro_f1"],
        ["local\nneighborhood", "global\nclustering", "supervised\nclassification"],
        "stage2_ability_radar",
        "Stage2 normalized capability profile",
        normalize=True,
    )


def save_stage3_capability_radar(stage3: pd.DataFrame) -> None:
    if stage3.empty:
        write_placeholder_svg(FIG_DIR / "stage3_capability_radar.svg", "Stage3 metrics not found")
        return
    wanted = [
        "log_robust_space",
        "sample_input_token_flat_pca12_space",
        "sample_encoded_token_flat_pca12_space",
        "transformer_latent_label_masked_space",
    ]
    df = stage3[stage3["space_name"].isin(wanted)].set_index("space_name").loc[wanted].reset_index()
    df["label"] = ["log robust", "input token", "encoded token", "label-masked latent"]
    radar_plot(
        df,
        "label",
        ["knn15_neighborhood_purity", "kmeans_nmi", "kmeans_ari", "xgb_macro_f1_mean"],
        ["kNN15\npurity", "KMeans\nNMI", "KMeans\nARI", "XGB\nmacro-F1"],
        "stage3_capability_radar",
        "Stage3 representation spaces",
        normalize=False,
    )


def write_next_step_svgs() -> None:
    specs = [
        ("next_step_framework.svg", "六维评价", ["预测", "邻域", "聚类", "保真", "连续", "稳定"], "#2563eb"),
        ("next_step_geo_prior.svg", "人工地质表示", ["标准化", "CoDA", "元素组", "异常指数"], "#16a34a"),
        ("next_step_embedding.svg", "解释 embedding", ["元素向量", "attention", "共现网络"], "#7c3aed"),
        ("next_step_transfer.svg", "扩大与迁移", ["全岩数据", "预训练", "黄铁矿迁移"], "#f59e0b"),
    ]
    for filename, title, labels, color in specs:
        parts = [
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 360 210">',
            '<rect width="360" height="210" rx="18" fill="#f8fafc" stroke="#d8dee8"/>',
            f'<circle cx="62" cy="62" r="30" fill="{color}" opacity="0.16"/>',
            f'<circle cx="62" cy="62" r="14" fill="{color}"/>',
            svg_text(108, 58, title, 22, 850, "#16202a"),
        ]
        for i, label in enumerate(labels):
            x = 40 + (i % 3) * 100
            y = 120 + (i // 3) * 42
            parts.append(f'<rect x="{x}" y="{y}" width="84" height="26" rx="13" fill="{color}" opacity="0.10"/>')
            parts.append(svg_text(x + 12, y + 19, label, 14, 700, color))
        parts.append("</svg>")
        write_text(FIG_DIR / filename, "\n".join(parts))


def write_placeholder_svg(path: Path, text: str) -> None:
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 460">
<rect width="900" height="460" rx="18" fill="#f8fafc" stroke="#d8dee8"/>
{svg_text(80, 220, text, 28, 700, "#667085")}
</svg>"""
    write_text(path, svg)


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TABLE_DIR / name, index=False, encoding="utf-8-sig")


def copy_inputs() -> dict[str, str | None]:
    sources: dict[str, Path] = {
        "stage2_report": asset("stage2_run") / "stage2_report.md",
        "stage3_report": asset("stage3_run") / "stage3_report.md",
        "stage2_unsupervised": asset("stage2_run") / "unsupervised" / "unsupervised_metrics.csv",
        "stage2_xgb": asset("stage2_run") / "xgb" / "xgb_classification_metrics.csv",
        "stage2_umap_metrics": asset("stage2_run") / "umap3d" / "umap3d_metrics.csv",
        "stage3_summary": asset("stage3_run") / "metrics" / "all_space_benchmark_summary.csv",
    }
    copied: dict[str, str | None] = {}
    for key, src in sources.items():
        dst = TABLE_DIR / src.name
        out = copy_if_exists(src, dst)
        copied[key] = rel(out) if out else None

    stage2_htmls = {
        "stage2_raw": asset("stage2_run") / "umap3d" / "raw_robust_z_14_umap3d_interactive.html",
        "stage2_balance": asset("stage2_run")
        / "umap3d"
        / "geo_balance_12_umap3d_interactive.html",
        "stage2_log": asset("stage2_run") / "umap3d" / "log_robust_z_14_umap3d_interactive.html",
        "stage2_rank": asset("stage2_run") / "umap3d" / "rank_uniform_z_14_umap3d_interactive.html",
        "stage2_ratio": asset("stage2_run") / "umap3d" / "geo_ratio_12_umap3d_interactive.html",
        "stage2_info": asset("stage2_run") / "umap3d" / "geo_info_12_umap3d_interactive.html",
    }
    for key, src in stage2_htmls.items():
        dst = STAGE2_INTERACTIVE_DIR / src.name
        out = copy_if_exists(src, dst)
        copied[key] = rel(out) if out else None

    stage3_htmls = {
        "stage3_baseline": asset("stage3_run")
        / "figures"
        / "previous_baseline_space_3d.html",
        "stage3_encoded": asset("stage3_run")
        / "figures"
        / "sample_encoded_token_flat_pca12_space_3d.html",
        "stage3_input": asset("stage3_run")
        / "figures"
        / "sample_input_token_flat_pca12_space_3d.html",
        "stage3_latent": asset("stage3_run")
        / "figures"
        / "transformer_latent_label_masked_space_3d.html",
        "stage3_log": asset("stage3_run") / "figures" / "log_robust_space_3d.html",
        "stage3_raw476": asset("stage3_run")
        / "raw_embedding_benchmark"
        / "figures"
        / "sample_raw_element_token_flat_476_umap3d.html",
    }
    for key, src in stage3_htmls.items():
        dst = STAGE3_INTERACTIVE_DIR / src.name
        out = copy_if_exists(src, dst)
        copied[key] = rel(out) if out else None

    image_sources = {
        "stage2_balance_png": asset("stage2_run")
        / "umap3d"
        / "geo_balance_12_UMAP1_UMAP2.png",
        "stage3_encoded_png": asset("stage3_run")
        / "figures"
        / "sample_encoded_token_flat_pca12_space_3d.png",
        "stage3_report_capability": asset("stage3_run") / "report_fig" / "capability_heatmap.png",
        "stage3_training": asset("stage3_run") / "report_fig" / "training_loss_and_lr.png",
    }
    for key, src in image_sources.items():
        dst = FIG_DIR / src.name
        out = copy_if_exists(src, dst)
        copied[key] = rel(out) if out else None
    return copied


def fig_img(name: str, alt: str, cls: str = "figure-img") -> str:
    path = FIG_DIR / name
    if path.exists():
        return f'<img class="{cls}" src="{rel(path)}" alt="{html.escape(alt)}">'
    return f'<div class="placeholder">Figure not found: {html.escape(name)}</div>'


def iframe_block(src: str | None, title: str, note: str) -> str:
    if src:
        return f"""
<div class="iframe-wrap">
  <iframe data-src="{html.escape(src)}" title="{html.escape(title)}"></iframe>
</div>
<div class="open-link"><a href="{html.escape(src)}" target="_blank" rel="noreferrer">Open interactive figure</a><span>{html.escape(note)}</span></div>
"""
    return f'<div class="placeholder large">Figure not found: {html.escape(title)}. Please check outputs/umap3d/.</div>'


def callout(text: str) -> str:
    return f'<div class="callout">{text}</div>'


def card(title: str, body: str, tone: str = "") -> str:
    return f'<div class="card {tone}"><h3>{title}</h3><p>{body}</p></div>'


def bullets(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"


def slide(n: int, chapter: str, title: str, body: str, cls: str = "") -> str:
    return f"""
<section class="slide {cls}" id="slide-{n}" data-slide="{n}" data-chapter="{html.escape(chapter)}">
  <div class="slide-inner">
    <div class="chapter-label">{html.escape(chapter)}</div>
    <h2>{title}</h2>
    {body}
  </div>
  <footer>{FOOTER}</footer>
</section>
"""


def build_slides(data: dict[str, object]) -> list[str]:
    stage2_unsup: pd.DataFrame = data["stage2_unsup"]  # type: ignore[assignment]
    stage2_xgb: pd.DataFrame = data["stage2_xgb"]  # type: ignore[assignment]
    stage3: pd.DataFrame = data["stage3"]  # type: ignore[assignment]
    copied: dict[str, str | None] = data["copied"]  # type: ignore[assignment]

    stage1_df = stage2_unsup[stage2_unsup["representation"].isin(["raw_robust_z_14", "log_robust_z_14", "rank_uniform_z_14"])]
    stage1_xgb = stage2_xgb[
        stage2_xgb["cv_scheme"].eq("stratified_kfold")
        & stage2_xgb["representation"].isin(["raw_robust_z_14", "log_robust_z_14", "rank_uniform_z_14"])
    ][["representation", "macro_f1"]]
    stage1 = stage1_df.merge(stage1_xgb, on="representation", how="left")

    stage2_table = stage2_unsup.merge(
        stage2_xgb[stage2_xgb["cv_scheme"].eq("stratified_kfold")][["representation", "macro_f1"]],
        on="representation",
        how="left",
    )
    stage2_focus = stage2_table[
        stage2_table["representation"].isin(["log_robust_z_14", "rank_uniform_z_14", "geo_ratio_12", "geo_balance_12", "geo_info_12"])
    ]

    stage3_focus = stage3[
        stage3["space_name"].isin(
            [
                "log_robust_space",
                "sample_input_token_flat_pca12_space",
                "sample_encoded_token_flat_pca12_space",
                "transformer_latent_label_masked_space",
            ]
        )
    ].copy()

    slides: list[str] = []
    slides.append(
        f"""
<section class="slide title-slide" id="slide-1" data-slide="1" data-chapter="背景">
  <div class="title-block">
    <div class="kicker">Stage 1-3 exploratory report</div>
    <h1>黄铁矿主/微量元素数据的多视图表示 benchmark 与 Element2Vec 初探</h1>
    <p class="subtitle">从人工地质表示到可学习地球化学 embedding</p>
    <p class="question">地质数据进入 AI 模型前，应该如何被表示？</p>
  </div>
  <footer>{FOOTER}</footer>
</section>
"""
    )
    slides.append(
        slide(
            2,
            "背景",
            "主线：从元素浓度矩阵到连续地质过程空间",
            f"""
<div class="full-figure">{fig_img("report_logic_flow.svg", "report logic flow")}</div>
{callout("本阶段以黄铁矿主/微量元素数据为最小实验对象，比较不同表示空间如何组织地质信息，并初步尝试让模型学习元素身份与元素上下文关系。")}
""",
        )
    )
    slides.append(
        slide(
            3,
            "背景",
            "1.1 什么是表示？",
            """
<div class="two-col">
  <div>
    <p class="lead">表示不是简单的数据格式转换。</p>
    <p>它决定模型如何看见尺度、距离、关系和结构。原始地球化学数据包含测量信息，但不一定是最接近地质过程的坐标系。</p>
    <p>一个好的表示应让关键地质结构更容易被学习、比较和解释。</p>
  </div>
  <div class="diagram-card">
    <h3>原始表格</h3>
    <p>Sample × Element concentration</p>
    <div class="arrow-down">↓</div>
    <h3>表示空间</h3>
    <p>异常 / 比值 / 过程轴 / token embedding / 关系网络</p>
  </div>
</div>
""",
        )
    )
    slides.append(
        slide(
            4,
            "背景",
            "1.2 为什么地球化学表示值得研究？",
            f"""
<div class="card-grid three">
  {card("结构更合理", "原始浓度不直接表达背景异常、元素共生、组分关系和过程端元。", "blue")}
  {card("评价更统一", "先建立表示框架，就可以统一比较不同编码方式对模型、聚类和解释的影响。", "green")}
  {card("通向统一空间", "类比 NLP 语义空间和组学状态空间，地球化学也可能形成连续的地质过程空间。", "orange")}
</div>
{callout("表示的目标不是凭空增加信息，而是重组信息结构。")}
""",
        )
    )
    slides.append(
        slide(
            5,
            "背景",
            "从 NLP 和组学看“表示”的价值",
            """
<table class="data-table large-text">
<thead><tr><th>领域</th><th>原始形式</th><th>表示进展</th></tr></thead>
<tbody>
<tr><td>NLP</td><td>字符 / 词序列</td><td>tokenizer → embedding → contextual representation</td></tr>
<tr><td>生物组学</td><td>样本 / 细胞 × 基因表达</td><td>通路得分、模块、共表达网络、细胞状态空间</td></tr>
<tr><td>地球化学</td><td>样本 × 元素浓度</td><td>异常秩、比值、balance、元素共生图、Element2Vec</td></tr>
</tbody>
</table>
{callout("这里不是说地质等同于语言，而是借鉴“从原始观测矩阵中恢复系统内在结构”的思想。")}
""",
        )
    )
    slides.append(
        slide(
            6,
            "背景",
            "2.1 如何评价一种地球化学表示？",
            f"""
<div class="full-figure medium">{fig_img("six_dimension_framework.svg", "six dimension framework")}</div>
{callout("当前阶段主要完成了前 3 个维度的量化，并为后 3 个维度形成后续计划。")}
""",
        )
    )
    slides.append(
        slide(
            7,
            "背景",
            "2.2 已量化的三个能力维度",
            f"""
<div class="card-grid three">
  {card("预测有效性", "XGBoost macro-F1、kNN classifier macro-F1、balanced accuracy。", "blue")}
  {card("局部邻域一致性", "kNN15 / kNN30 neighborhood purity、UMAP trustworthiness。", "green")}
  {card("全局结构组织", "KMeans/GMM ARI、NMI，以及 UMAP-3D 的可视结构。", "orange")}
</div>
<p class="bottom-note">这些指标不是为了追求单一最高分，而是判断表示空间是否更好地组织地质结构。</p>
""",
        )
    )
    slides.append(
        slide(
            8,
            "Stage 1",
            "Stage 1：先建立传统数值表示 baseline",
            """
<div class="two-col">
  <div>
    <ul class="big-list">
      <li>数据：清洗后的公开黄铁矿主/微量元素数据库</li>
      <li>样本：约 7778</li>
      <li>元素：14 个核心元素</li>
      <li>标签：矿床类型 type</li>
      <li>baseline：raw / log / rank</li>
    </ul>
  </div>
  <div class="pipeline-card">
    <div>元素浓度矩阵</div><span>→</span>
    <div>raw_robust_z</div><span>→</span>
    <div>log_robust_z</div><span>→</span>
    <div>rank_uniform_z</div><span>→</span>
    <div>PCA / UMAP / kNN / clustering / XGB</div>
  </div>
</div>
{callout("即使原始信息相同，不同统计表示也会改变样本距离和模型表现。")}
""",
        )
    )
    slides.append(
        slide(
            9,
            "Stage 1",
            "Stage 1 结果：rank/log 是强 baseline",
            f"""
<div class="two-col wide-left">
  <div>{simple_table(stage1, ["representation", "input_knn15", "kmeans_nmi", "macro_f1"], ["representation", "input kNN15", "KMeans NMI", "XGB macro-F1"])}</div>
  <div>
    <p class="lead">rank 表示可以理解为“经验异常秩”。</p>
    <p>它不只看某元素是多少 ppm，而是看它在全数据库背景中有多异常。</p>
    <p>这与勘查地球化学关注背景/异常的思想一致。</p>
  </div>
</div>
{callout("这一阶段建立了强 baseline，也说明表示方式有时比模型结构更影响空间组织。")}
""",
        )
    )
    slides.append(
        slide(
            10,
            "Stage 2",
            "统计表示有效，但还不够地质",
            """
<div class="two-col">
  <div>
    <p class="lead">rank/log 能很好地组织局部邻域和分类边界。</p>
    <p>但它们仍然主要把元素当作独立变量，较少显式表达成矿过程中的相对关系。</p>
  </div>
  <div>
    <h3>矿床地球化学更关心</h3>
    <ul>
      <li>元素比值与元素组合</li>
      <li>路径元素和异常谱型</li>
      <li>高温/低温过程端元</li>
      <li>元素共生关系</li>
    </ul>
  </div>
</div>
{callout("所以 Stage 2 尝试引入人工地质先验表示。")}
""",
        )
    )
    slides.append(
        slide(
            11,
            "Stage 2",
            "3.1 地球化学中的经典表示思想",
            """
<div class="two-col">
  <div class="card">
    <h3>传统操作</h3>
    <ul>
      <li>CI 球粒陨石标准化</li>
      <li>Primitive mantle / UCC / MORB 标准化</li>
      <li>REE 配分曲线与 spider diagram</li>
      <li>元素比值、异常指数、活动性元素对比</li>
    </ul>
  </div>
  <div class="card green">
    <h3>表示意义</h3>
    <p>这些操作不是简单画图，而是在把原始浓度投射到更有地质理论含义的坐标系中。</p>
  </div>
</div>
{callout("传统地球化学本身就包含 representation thinking。")}
""",
        )
    )
    slides.append(
        slide(
            12,
            "Stage 2",
            "3.2 借鉴组学：从单元素到过程组",
            """
<div class="two-col">
  <div>
    <p class="lead">单个基因表达变化不稳定，一组功能相关基因的整体变化更有代表性。</p>
    <p>这启发我们把单元素扩展为元素组、过程组和共现网络。</p>
  </div>
  <table class="data-table">
    <thead><tr><th>组学</th><th>地球化学</th></tr></thead>
    <tbody>
      <tr><td>single gene</td><td>single element</td></tr>
      <tr><td>gene set</td><td>element group</td></tr>
      <tr><td>pathway score</td><td>process-group score</td></tr>
      <tr><td>co-expression network</td><td>element co-occurrence network</td></tr>
      <tr><td>cell state space</td><td>geological process space</td></tr>
    </tbody>
  </table>
</div>
""",
        )
    )
    slides.append(
        slide(
            13,
            "Stage 2",
            "Stage 2：三类人工地质表示",
            f"""
<div class="card-grid three">
  {card("geo_ratio_12", "强调元素之间的相对关系，例如 Co/Ni、Au/As、As/Se、Se/Te、Sb/Tl、Cu/Pb。", "blue")}
  {card("geo_balance_12", "强调地质过程组之间的对立轴，例如 Co-Ni-Se vs Au-As-Sb-Tl。", "green")}
  {card("geo_info_12", "强调异常信息分布：异常由少数元素主导，还是多元素共同富集。", "orange")}
</div>
{callout("三套表示都控制在约 12 维，方便与 14 维 baseline 对比。")}
""",
        )
    )
    slides.append(
        slide(
            14,
            "Stage 2",
            "Stage 2 结果：不是谁完全胜出，而是能力分化",
            f"""
<div class="two-col wide-left">
  <div>{fig_img("stage2_ability_profile.svg", "stage2 ability profile")}</div>
  <div>
    <p class="lead">rank/log：局部邻域和监督判别强。</p>
    <p>geo_ratio：保留部分局部结构。</p>
    <p>geo_balance：全局分群更突出。</p>
    <p>geo_info：单独较弱，更适合作为融合辅助信息。</p>
  </div>
</div>
{callout("人工地质表示没有在 XGB 分类上超过 baseline，但 balance 在 KMeans/GMM 与 UMAP-3D 中表现出更清晰的全局结构。")}
""",
        )
    )
    slides.append(
        slide(
            15,
            "Stage 2",
            "Balance 表示：更像地质过程轴的空间",
            f"""
<div class="interactive-layout">
  <div>{iframe_block(copied.get("stage2_balance"), "Stage2 geo_balance_12 UMAP3D", "geo_balance_12_umap3d_interactive.html")}</div>
  <aside>
    <ol>
      <li>balance 的 XGB 分类不是最高。</li>
      <li>但其 UMAP-3D 肉眼结构更明晰。</li>
      <li>KMeans/GMM 结果也支持其全局分群能力更强。</li>
    </ol>
  </aside>
</div>
""",
            "interactive-slide",
        )
    )
    slides.append(
        slide(
            16,
            "Stage 2",
            "反思：人工表示的价值与局限",
            f"""
<div class="two-col">
  {card("为什么 XGB 较弱？", "ratio/balance 会消除部分绝对丰度信息，而 XGB 很擅长用 Au、Ni、Sb、Se、Te、Bi 等单元素阈值切分已有 type 标签。", "orange")}
  {card("为什么 balance 聚类强？", "balance 把样本沿着少数地质过程组对立轴展开，更像过程空间，而不只是分类器特征。", "green")}
</div>
{callout("人工表示有解释性，但依赖人为预设关系。下一步问题是：能否让模型学习元素身份和上下文关系？")}
""",
        )
    )
    slides.append(
        slide(
            17,
            "Stage 3",
            "从人工表示到 Element2Vec",
            """
<div class="two-col">
  <div>
    <p class="lead">能否让模型学习元素本身的“身份信息”？</p>
    <p>在原始数值矩阵中，计算机只看到列名对应的数值，却不知道 Au、As、Ni 等元素在地质系统中的含义。</p>
    <p>NLP 的 embedding 思路启发我们：把元素视为 token，让模型通过样本上下文学习元素身份、含量状态和元素关系。</p>
  </div>
  <div class="diagram-card">
    <h3>人工表示</h3>
    <p>Au/As, Co/Ni, group balance<br>= 人为定义关系</p>
    <div class="arrow-down">↓</div>
    <h3>Element2Vec</h3>
    <p>element token + value + mask → Transformer<br>= 数据驱动学习关系</p>
  </div>
</div>
""",
        )
    )
    slides.append(
        slide(
            18,
            "Stage 3",
            "4.1 Element2Vec 架构：元素 token + 矿床 token",
            f"""
<div class="full-figure arch">{fig_img("element2vec_architecture.svg", "Element2Vec architecture")}</div>
{callout("元素和矿床类型都被 token 化，并投射到统一 64 维 token 空间；主 benchmark 使用 label-masked latent，避免标签泄漏。")}
""",
        )
    )
    slides.append(
        slide(
            19,
            "Stage 3",
            "4.2 Element2Vec 结果：token/latent 空间结构增强",
            f"""
<div class="two-col wide-left">
  <div>{fig_img("stage3_space_comparison.svg", "stage3 space comparison")}</div>
  <div>
    {simple_table(stage3_focus, ["space_name", "knn15_neighborhood_purity", "kmeans_ari", "kmeans_nmi", "xgb_macro_f1_mean"], ["space", "kNN15", "ARI", "NMI", "XGB macro-F1"])}
  </div>
</div>
{callout("当前 1000 epoch 结果中，label-masked latent 在 kNN/ARI/XGB 上最高，encoded token PCA12 的 NMI 也明显强于 log baseline。")}
""",
        )
    )
    slides.append(
        slide(
            20,
            "Stage 3",
            "Element2Vec encoded token space 的 3D 结构",
            f"""
<div class="interactive-layout">
  <div>{iframe_block(copied.get("stage3_encoded"), "Stage3 sample_encoded_token_flat_pca12 UMAP3D", "sample_encoded_token_flat_pca12_space_3d.html")}</div>
  <aside>
    <ol>
      <li>该空间不是直接原始浓度矩阵。</li>
      <li>它来自元素 token 经 Transformer 编码后的表示。</li>
      <li>相对 log/rank baseline，局部邻域和全局聚类结构均有增强。</li>
    </ol>
    <div class="small-links">
      <a href="{copied.get("stage3_log") or "#"}" target="_blank">log robust</a>
      <a href="{copied.get("stage3_latent") or "#"}" target="_blank">label-masked latent</a>
    </div>
  </aside>
</div>
""",
            "interactive-slide",
        )
    )
    slides.append(
        slide(
            21,
            "Stage 3",
            "4.3 结果边界：这是探索性原型，不是严格泛化证明",
            f"""
<div class="card-grid four">
  {card("数据规模小", "当前约 7778 个样本、14 个核心元素。", "blue")}
  {card("模型规模小", "32d embedding、64d token、2-layer Transformer。", "green")}
  {card("训练目标使用标签", "masked deposit prediction 属于监督/弱监督表示学习，不是纯自监督。", "orange")}
  {card("泛化验证不足", "当前全数据训练，分类指标是 representation sanity check，不是 generalization claim。", "red")}
</div>
{callout("即便如此，token/latent space 的邻域和聚类结构增强，说明该方向值得继续探索。")}
""",
        )
    )
    slides.append(
        slide(
            22,
            "未来方向",
            "阶段总结：三层表示探索",
            """
<div class="layer-stack">
  <div><strong>第一层：统计表示</strong><span>raw / log / rank → 强 baseline，异常秩有效</span></div>
  <div><strong>第二层：人工地质表示</strong><span>ratio / balance / info → ratio 保留相对关系，balance 组织过程域</span></div>
  <div><strong>第三层：可学习 token 表示</strong><span>Element2Vec → 学习元素身份、含量状态和上下文关系</span></div>
</div>
{callout("当前结果支持将研究目标从“寻找单一最优变换”提升为“构建多视图地球化学表示框架”。")}
""",
        )
    )
    slides.append(
        slide(
            23,
            "未来方向",
            "下一步工作",
            f"""
<div class="card-grid four">
  {card("完善六维评价体系", "补充原始信息保真、地质指标连续性、泛化稳定性的量化方法。", "blue")}
  {card("继续探索人工表示", "吸收标准化、CoDA、元素组合、异常指数和网络思想。", "green")}
  {card("解释 Element2Vec", "检查 Au-As-Sb-Tl、Co-Ni-Se、Cu-Pb-Zn-Ag 等结构是否出现在 embedding 中。", "orange")}
  {card("扩大训练并迁移", "用更大全岩地球化学数据训练通用元素表示，再迁移到矿床/矿物微量元素数据。", "purple")}
</div>
""",
        )
    )
    slides.append(
        slide(
            24,
            "未来方向",
            "长期目标：连续统一地质过程空间",
            """
<div class="center-claim">从元素浓度值，到可运算的地质向量。</div>
<table class="data-table large-text">
<thead><tr><th>NLP</th><th>地球化学</th></tr></thead>
<tbody>
<tr><td>token</td><td>元素 / 矿物 / 元素组</td></tr>
<tr><td>tokenizer</td><td>地质对象与过程单元定义</td></tr>
<tr><td>embedding</td><td>元素 / 样本 / 过程向量</td></tr>
<tr><td>semantic space</td><td>统一地质过程空间</td></tr>
<tr><td>word relation</td><td>元素共生 / 矿物共生 / 过程耦合</td></tr>
<tr><td>context</td><td>岩性 / 矿物宿主 / 构造背景 / 成矿阶段</td></tr>
</tbody>
</table>
<p class="bottom-note">理想情况下，元素不再只是浓度数值，而是在地质上下文中具有含义、关系和可运算性的向量单元。</p>
""",
        )
    )
    return slides


def build_slides_v2(data: dict[str, object]) -> list[str]:
    copied: dict[str, str | None] = data["copied"]  # type: ignore[assignment]
    bg, s1, s2, s3, future = CHAPTERS

    def switch_buttons(frame_id: str, items: list[tuple[str, str | None]], active: str) -> str:
        buttons = []
        for label, target in items:
            if not target:
                continue
            active_cls = " active" if label == active else ""
            buttons.append(
                f'<button class="switch-btn{active_cls}" type="button" data-frame="{frame_id}" '
                f'data-target="{html.escape(target)}">{html.escape(label)}</button>'
            )
        return '<div class="figure-switcher">' + "".join(buttons) + "</div>"

    slides: list[str] = []
    slides.append(
        f"""
<section class="slide title-slide" id="slide-1" data-slide="1" data-chapter="{bg}">
  <div class="title-block">
    <div class="kicker">Stage 1-3 exploratory report</div>
    <h1>黄铁矿主/微量元素数据的多视图表示 benchmark 与 Element2Vec 初探</h1>
    <p class="subtitle">从人工地质表示到可学习地球化学 embedding</p>
    <p class="question">地质数据进入 AI 模型前，应该如何被表示？</p>
  </div>
  <footer>{FOOTER}</footer>
</section>
"""
    )
    slides.append(
        slide(
            2,
            bg,
            "主线：从元素浓度矩阵到连续地质过程空间",
            f"""
<div class="full-figure logic-flow">{fig_img("report_logic_flow.svg", "report logic flow")}</div>
<div class="compact-row">
  <span>对象：黄铁矿主/微量元素</span>
  <span>目标：比较表示空间结构</span>
  <span>路径：统计表示 → 地质表示 → 可学习 token</span>
</div>
""",
        )
    )
    slides.append(
        slide(
            3,
            bg,
            "1.1 什么是表示？",
            """
<div class="two-col">
  <div>
    <p class="lead">表示不是简单的数据格式转换。</p>
    <p>它决定模型如何看见尺度、距离、关系和结构。原始地球化学数据包含测量信息，但不一定是最接近地质过程的坐标系。</p>
    <p>一个好的表示应让关键地质结构更容易被学习、比较和解释。</p>
  </div>
  <div class="diagram-card">
    <h3>原始表格</h3>
    <p>Sample × Element concentration</p>
    <div class="arrow-down">→</div>
    <h3>表示空间</h3>
    <p>异常 / 比值 / 过程轴 / token embedding / 关系网络</p>
  </div>
</div>
""",
        )
    )
    slides.append(
        slide(
            4,
            bg,
            "1.2 为什么地球化学表示值得研究？",
            f"""
<div class="card-grid three">
  {card("结构更合理", "原始浓度不直接表达背景异常、元素共生、组分关系和过程端元。", "blue")}
  {card("评价更统一", "先建立表示框架，就可以统一比较不同编码方式对模型、聚类和解释的影响。", "green")}
  {card("通向统一空间", "类比 NLP 语义空间和组学状态空间，地球化学也可能形成连续的地质过程空间。", "orange")}
</div>
""",
        )
    )
    slides.append(
        slide(
            5,
            bg,
            "从 NLP 和组学看“表示”的价值",
            """
<table class="data-table large-text">
<thead><tr><th>领域</th><th>原始形式</th><th>表示进展</th></tr></thead>
<tbody>
<tr><td>NLP</td><td>字符 / 词序列</td><td>tokenizer → embedding → contextual representation</td></tr>
<tr><td>生物组学</td><td>样本 / 细胞 × 基因表达</td><td>通路得分、模块、共表达网络、细胞状态空间</td></tr>
<tr><td>地球化学</td><td>样本 × 元素浓度</td><td>异常秩、比值、balance、元素共生图、Element2Vec</td></tr>
</tbody>
</table>
<p class="inline-note">借鉴的是“从原始观测矩阵中恢复系统内在结构”的方法论，而不是把地质等同于语言。</p>
""",
        )
    )
    slides.append(
        slide(
            6,
            bg,
            "2.1 如何评价一种地球化学表示？",
            f'<div class="full-figure medium">{fig_img("six_dimension_framework.svg", "six dimension framework")}</div>',
        )
    )
    slides.append(
        slide(
            7,
            bg,
            "2.2 已量化的三个能力维度",
            f"""
<div class="card-grid three">
  {card("预测有效性", "XGBoost macro-F1、kNN classifier macro-F1、balanced accuracy。", "blue")}
  {card("局部邻域一致性", "kNN15 / kNN30 neighborhood purity、UMAP trustworthiness。", "green")}
  {card("全局结构组织", "KMeans/GMM ARI、NMI，以及 UMAP-3D 的可视结构。", "orange")}
</div>
""",
        )
    )
    slides.append(
        slide(
            8,
            s1,
            "Stage 1：传统数值表示 baseline",
            """
<div class="two-col">
  <div>
    <ul class="big-list">
      <li>数据：清洗后的公开黄铁矿主/微量元素数据库</li>
      <li>样本：约 7778</li>
      <li>元素：14 个核心元素</li>
      <li>标签：矿床类型 type</li>
      <li>baseline：raw / log / rank</li>
    </ul>
  </div>
  <div class="pipeline-card">
    <div>元素浓度矩阵</div><span>→</span>
    <div>raw_robust_z</div><span>→</span>
    <div>log_robust_z</div><span>→</span>
    <div>rank_uniform_z</div><span>→</span>
    <div>PCA / UMAP / kNN / clustering / XGB</div>
  </div>
</div>
""",
        )
    )
    slides.append(
        slide(
            9,
            s1,
            "Stage 1 结果：rank/log 是强 baseline",
            f"""
<div class="two-col wide-left">
  <div>{fig_img("stage1_baseline_radar.svg", "Stage1 baseline radar")}</div>
  <div>
    <p class="lead">rank 表示可以理解为“经验异常秩”。</p>
    <p>它不只看某元素是多少 ppm，而是看它在全数据库背景中有多异常。</p>
    <p>这与勘查地球化学关注背景/异常的思想一致。</p>
    <div class="metric-cards">
      <span>rank/log：局部邻域强</span>
      <span>raw：空间结构明显较弱</span>
      <span>XGB：三者均为强 baseline</span>
    </div>
  </div>
</div>
""",
        )
    )
    slides.append(
        slide(
            10,
            s2,
            "Stage 2：从统计表示转向地质关系",
            """
<div class="two-col">
  <div>
    <p class="lead">rank/log 能很好地组织局部邻域和分类边界。</p>
    <p>但它们仍然主要把元素当作独立变量，较少显式表达成矿过程中的相对关系。</p>
  </div>
  <div>
    <h3>矿床地球化学更关心</h3>
    <ul>
      <li>元素比值与元素组合</li>
      <li>路径元素和异常谱型</li>
      <li>高温/低温过程端元</li>
      <li>元素共生关系</li>
    </ul>
  </div>
</div>
""",
        )
    )
    slides.append(
        slide(
            11,
            s2,
            "3.1 地球化学中的经典表示思想",
            """
<div class="two-col">
  <div class="card">
    <h3>传统操作</h3>
    <ul>
      <li>CI 球粒陨石标准化</li>
      <li>Primitive mantle / UCC / MORB 标准化</li>
      <li>REE 配分曲线与 spider diagram</li>
      <li>元素比值、异常指数、活动性元素对比</li>
    </ul>
  </div>
  <div class="card green">
    <h3>表示意义</h3>
    <p>这些操作不是简单画图，而是在把原始浓度投射到更有地质理论含义的坐标系中。</p>
  </div>
</div>
""",
        )
    )
    slides.append(
        slide(
            12,
            s2,
            "3.2 借鉴组学：从单元素到过程组",
            """
<div class="two-col">
  <div>
    <p class="lead">单个基因表达变化不稳定，一组功能相关基因的整体变化更有代表性。</p>
    <p>这启发我们把单元素扩展为元素组、过程组和共现网络。</p>
  </div>
  <table class="data-table">
    <thead><tr><th>组学</th><th>地球化学</th></tr></thead>
    <tbody>
      <tr><td>single gene</td><td>single element</td></tr>
      <tr><td>gene set</td><td>element group</td></tr>
      <tr><td>pathway score</td><td>process-group score</td></tr>
      <tr><td>co-expression network</td><td>element co-occurrence network</td></tr>
      <tr><td>cell state space</td><td>geological process space</td></tr>
    </tbody>
  </table>
</div>
""",
        )
    )
    slides.append(
        slide(
            13,
            s2,
            "Stage 2：三类人工地质表示设计",
            """
<div class="card-grid three feature-cards">
  <div class="card blue"><h3>geo_ratio_12</h3><p>强调元素之间的相对关系。</p>
    <ul class="feature-list"><li>Co/Ni, Au/As, As/Se</li><li>Se/Te, Bi/Te, Sb/Tl</li><li>Cu/Zn, Pb/Zn, Ag/Au</li><li>Cu/Pb, As/Sb, base-metal/pathfinder</li></ul>
  </div>
  <div class="card green"><h3>geo_balance_12</h3><p>强调过程组之间的对立轴。</p>
    <ul class="feature-list"><li>Co-Ni-Se vs Au-As-Sb-Tl</li><li>Cu-Pb-Zn-Ag vs Au-As-Sb-Tl</li><li>As-Sb-Bi-Te vs Co-Ni</li><li>Sb-Tl-Te vs Co-Ni-Se</li></ul>
  </div>
  <div class="card orange"><h3>geo_info_12</h3><p>强调异常信息分布。</p>
    <ul class="feature-list"><li>entropy / normalized entropy</li><li>effective elements</li><li>top1 / top3 share</li><li>group info share + rarity</li></ul>
  </div>
</div>
""",
        )
    )
    slides.append(
        slide(
            14,
            s2,
            "Stage 2：人工表示的能力分化",
            f"""
<div class="two-col wide-left">
  <div>{fig_img("stage2_ability_radar.svg", "stage2 ability radar")}</div>
  <div>
    <p class="lead">不同表示对应不同能力侧重。</p>
    <p>rank/log：局部邻域和监督判别强。</p>
    <p>geo_ratio：保留部分局部结构。</p>
    <p>geo_balance：全局分群更突出。</p>
    <p>geo_info：单独较弱，更适合作为融合辅助信息。</p>
  </div>
</div>
""",
        )
    )
    slides.append(
        slide(
            15,
            s2,
            "Stage 2：UMAP-3D 对比",
            f"""
<div class="interactive-layout">
  <div>
    {switch_buttons("stage2Frame", [("balance", copied.get("stage2_balance")), ("ratio", copied.get("stage2_ratio")), ("log", copied.get("stage2_log")), ("rank", copied.get("stage2_rank")), ("info", copied.get("stage2_info"))], "balance")}
    <div class="iframe-wrap"><iframe id="stage2Frame" data-src="{html.escape(copied.get("stage2_balance") or "")}" title="Stage2 UMAP3D"></iframe></div>
  </div>
  <aside>
    <ol>
      <li>balance 的 XGB 分类不是最高。</li>
      <li>但其 UMAP-3D 肉眼结构更明晰。</li>
      <li>KMeans/GMM 结果也支持其全局分群能力更强。</li>
    </ol>
  </aside>
</div>
""",
            "interactive-slide",
        )
    )
    slides.append(
        slide(
            16,
            s2,
            "Stage 2：人工表示的价值与局限",
            f"""
<div class="two-col">
  {card("为什么 XGB 较弱？", "ratio/balance 会消除部分绝对丰度信息，而 XGB 很擅长用 Au、Ni、Sb、Se、Te、Bi 等单元素阈值切分已有 type 标签。", "orange")}
  {card("为什么 balance 聚类强？", "balance 把样本沿着少数地质过程组对立轴展开，更像过程空间，而不只是分类器特征。", "green")}
</div>
<p class="inline-note">人工表示有解释性，但依赖人为预设关系；下一步转向可学习的元素身份与上下文关系。</p>
""",
        )
    )
    slides.append(
        slide(
            17,
            s3,
            "从人工表示到 Element2Vec",
            """
<div class="two-col">
  <div>
    <p class="lead">能否让模型学习元素本身的“身份信息”？</p>
    <p>在原始数值矩阵中，计算机只看到列名对应的数值，却不知道 Au、As、Ni 等元素在地质系统中的含义。</p>
    <p>NLP 的 embedding 思路启发我们：把元素视为 token，让模型通过样本上下文学习元素身份、含量状态和元素关系。</p>
  </div>
  <div class="diagram-card">
    <h3>人工表示</h3>
    <p>Au/As, Co/Ni, group balance<br>= 人为定义关系</p>
    <div class="arrow-down">↓</div>
    <h3>Element2Vec</h3>
    <p>element token + value + mask → Transformer<br>= 数据驱动学习关系</p>
  </div>
</div>
""",
        )
    )
    slides.append(
        slide(
            18,
            s3,
            "4.1 Element2Vec 架构：查表、投影与上下文编码",
            f'<div class="full-figure arch">{fig_img("element2vec_architecture.svg", "Element2Vec architecture")}</div>',
        )
    )
    slides.append(
        slide(
            19,
            s3,
            "4.2 Element2Vec benchmark：空间结构增强",
            f"""
<div class="two-col equal">
  <div>{fig_img("stage3_capability_radar.svg", "stage3 radar")}</div>
  <div>{fig_img("stage3_space_comparison.svg", "stage3 space comparison")}</div>
</div>
""",
        )
    )
    slides.append(
        slide(
            20,
            s3,
            "Element2Vec encoded token space 的 3D 结构",
            f"""
<div class="interactive-layout">
  <div>
    {switch_buttons("stage3Frame", [("encoded", copied.get("stage3_encoded")), ("latent", copied.get("stage3_latent")), ("input", copied.get("stage3_input")), ("log", copied.get("stage3_log")), ("raw476", copied.get("stage3_raw476"))], "encoded")}
    <div class="iframe-wrap"><iframe id="stage3Frame" data-src="{html.escape(copied.get("stage3_encoded") or "")}" title="Stage3 UMAP3D"></iframe></div>
  </div>
  <aside>
    <ol>
      <li>该空间不是直接原始浓度矩阵。</li>
      <li>它来自元素 token 经 Transformer 编码后的表示。</li>
      <li>可与 log、input token、label-masked latent、raw 476D 表示直接切换对比。</li>
    </ol>
  </aside>
</div>
""",
            "interactive-slide",
        )
    )
    slides.append(
        slide(
            21,
            s3,
            "4.3 结果边界：探索性原型，不是泛化证明",
            f"""
<div class="limit-layout">
  <div class="limit-grid">
    {card("数据规模小", "约 7778 个样本、14 个核心元素。", "blue")}
    {card("模型规模小", "32d embedding、64d token、Transformer 原型。", "green")}
    {card("训练目标使用标签", "masked deposit prediction 是弱监督表示学习。", "orange")}
    {card("泛化验证不足", "当前指标是 representation sanity check。", "red")}
  </div>
  <div class="limit-side">
    {fig_img("stage3_capability_radar.svg", "stage3 radar")}
    <p class="small-note">即便如此，token/latent space 的邻域和聚类结构显著增强，说明该方向值得继续探索。</p>
  </div>
</div>
""",
        )
    )
    slides.append(
        slide(
            22,
            future,
            "阶段总结：三层表示探索",
            """
<div class="layer-stack">
  <div><strong>第一层：统计表示</strong><span>raw / log / rank → 强 baseline，异常秩有效</span></div>
  <div><strong>第二层：人工地质表示</strong><span>ratio / balance / info → ratio 保留相对关系，balance 组织过程域</span></div>
  <div><strong>第三层：可学习 token 表示</strong><span>Element2Vec → 学习元素身份、含量状态和上下文关系</span></div>
</div>
""",
        )
    )
    slides.append(
        slide(
            23,
            future,
            "下一步工作",
            f"""
<div class="next-step-grid">
  <div class="next-step-card blue">{fig_img("next_step_framework.svg", "next step framework")}<p>补充原始信息保真、地质指标连续性、泛化稳定性。</p></div>
  <div class="next-step-card green">{fig_img("next_step_geo_prior.svg", "next step geo prior")}<p>吸收标准化、CoDA、元素组合、异常指数和网络思想。</p></div>
  <div class="next-step-card purple">{fig_img("next_step_embedding.svg", "next step embedding")}<p>解释元素 embedding、attention 与元素共现网络。</p></div>
  <div class="next-step-card orange">{fig_img("next_step_transfer.svg", "next step transfer")}<p>用更大全岩数据训练通用元素表示，再迁移到黄铁矿。</p></div>
</div>
""",
        )
    )
    slides.append(
        slide(
            24,
            future,
            "长期目标：连续统一地质过程空间",
            """
<div class="center-claim">从元素浓度值，到可运算的地质向量。</div>
<table class="data-table large-text">
<thead><tr><th>NLP</th><th>地球化学</th></tr></thead>
<tbody>
<tr><td>token</td><td>元素 / 矿物 / 元素组</td></tr>
<tr><td>tokenizer</td><td>地质对象与过程单元定义</td></tr>
<tr><td>embedding</td><td>元素 / 样本 / 过程向量</td></tr>
<tr><td>semantic space</td><td>统一地质过程空间</td></tr>
<tr><td>word relation</td><td>元素共生 / 矿物共生 / 过程耦合</td></tr>
<tr><td>context</td><td>岩性 / 矿物宿主 / 构造背景 / 成矿阶段</td></tr>
</tbody>
</table>
""",
        )
    )
    slides.append(
        slide(
            25,
            future,
            "谢谢，请批评指正",
            """
<div class="thanks-slide">
  <div class="thanks-title">谢谢</div>
  <p>欢迎从地质合理性、表示设计、评价指标和下一阶段实验设计四个角度批评指正。</p>
  <div class="compact-row">
    <span>representation benchmark</span>
    <span>geo-prior features</span>
    <span>Element2Vec prototype</span>
  </div>
</div>
""",
            "thanks",
        )
    )
    return slides


def write_css() -> None:
    css = r"""
:root {
  color-scheme: light;
  --bg: #edf2f7;
  --slide: #ffffff;
  --ink: #16202a;
  --muted: #667085;
  --line: #d8dee8;
  --blue: #2563eb;
  --teal: #0891b2;
  --green: #16a34a;
  --orange: #f59e0b;
  --red: #dc2626;
  --purple: #7c3aed;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: radial-gradient(circle at 18% 12%, #ffffff 0, #edf2f7 34%, #e4ebf5 100%);
  color: var(--ink);
  font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "Segoe UI", Arial, sans-serif;
  overflow: hidden;
}
a { color: var(--blue); text-decoration: none; }
a:hover { text-decoration: underline; }
#deck {
  position: fixed;
  inset: 0;
  display: grid;
  place-items: center;
}
.slide {
  display: none;
  position: relative;
  width: min(100vw, calc(100vh * 16 / 9));
  height: min(100vh, calc(100vw * 9 / 16));
  background: var(--slide);
  box-shadow: 0 24px 80px rgba(15, 23, 42, 0.22);
  overflow: hidden;
}
.slide.active { display: block; }
.slide-inner {
  position: absolute;
  inset: 52px 70px 58px 70px;
  display: flex;
  flex-direction: column;
  gap: 18px;
}
.chapter-label {
  width: max-content;
  padding: 5px 12px;
  border-radius: 999px;
  background: #eff6ff;
  color: var(--blue);
  font-size: 15px;
  font-weight: 800;
  letter-spacing: 0;
}
h1, h2, h3, p { margin: 0; letter-spacing: 0; }
h1 {
  font-size: 54px;
  line-height: 1.15;
  font-weight: 900;
  max-width: 1050px;
}
h2 {
  font-size: 38px;
  line-height: 1.18;
  font-weight: 900;
}
h3 { font-size: 22px; line-height: 1.25; font-weight: 850; }
p, li { font-size: 22px; line-height: 1.5; }
ul, ol { margin: 0; padding-left: 1.25em; }
li + li { margin-top: 8px; }
footer {
  position: absolute;
  left: 70px;
  bottom: 22px;
  color: #98a2b3;
  font-size: 13px;
  font-weight: 600;
}
.title-slide {
  background: linear-gradient(135deg, #ffffff 0%, #f8fbff 60%, #eef6ff 100%);
}
.title-block {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 80px;
  gap: 22px;
}
.kicker {
  color: var(--blue);
  font-size: 20px;
  font-weight: 850;
}
.subtitle {
  font-size: 29px;
  color: var(--muted);
}
.question {
  margin-top: 28px;
  padding: 16px 26px;
  border-left: 5px solid var(--blue);
  background: #f8fafc;
  font-size: 24px;
  color: #344054;
}
.lead {
  font-size: 27px;
  font-weight: 820;
  color: #243447;
  line-height: 1.35;
}
.two-col {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 34px;
  align-items: center;
  flex: 1;
}
.wide-left { grid-template-columns: 1.25fr 0.75fr; }
.card-grid {
  display: grid;
  gap: 22px;
  flex: 1;
  align-items: stretch;
}
.card-grid.three { grid-template-columns: repeat(3, 1fr); }
.card-grid.four { grid-template-columns: repeat(4, 1fr); }
.card, .diagram-card, .pipeline-card {
  border: 1px solid var(--line);
  background: #ffffff;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 8px 28px rgba(16, 24, 40, 0.06);
}
.card p { margin-top: 12px; color: #475467; font-size: 20px; }
.card.blue { border-top: 5px solid var(--blue); }
.card.green { border-top: 5px solid var(--green); }
.card.orange { border-top: 5px solid var(--orange); }
.card.red { border-top: 5px solid var(--red); }
.card.purple { border-top: 5px solid var(--purple); }
.callout {
  padding: 16px 22px;
  background: #f8fafc;
  border-left: 5px solid var(--blue);
  color: #344054;
  font-size: 22px;
  line-height: 1.45;
  font-weight: 650;
}
.bottom-note {
  font-size: 22px;
  color: #475467;
  font-weight: 650;
}
.full-figure {
  display: grid;
  place-items: center;
  flex: 1;
  min-height: 0;
}
.full-figure.logic-flow img { width: 100%; max-height: 430px; }
.full-figure.medium img { max-height: 500px; }
.full-figure.arch img { max-height: 510px; }
.figure-img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}
.diagram-card h3 { color: var(--blue); margin-bottom: 10px; }
.diagram-card p { color: #475467; }
.arrow-down {
  color: var(--muted);
  font-size: 35px;
  text-align: center;
  margin: 18px 0;
}
.pipeline-card {
  display: flex;
  flex-direction: column;
  gap: 10px;
  align-items: stretch;
}
.pipeline-card div {
  padding: 10px 14px;
  background: #f8fafc;
  border: 1px solid var(--line);
  border-radius: 8px;
  font-size: 20px;
  font-weight: 750;
}
.pipeline-card span {
  text-align: center;
  color: var(--muted);
  font-size: 24px;
}
.big-list li { font-size: 24px; }
.data-table {
  width: 100%;
  border-collapse: collapse;
  overflow: hidden;
  border-radius: 10px;
  font-size: 18px;
}
.data-table th {
  background: #eff6ff;
  color: #1d4ed8;
  font-size: 16px;
  text-align: left;
  padding: 11px 12px;
  border-bottom: 1px solid var(--line);
}
.data-table td {
  padding: 10px 12px;
  border-bottom: 1px solid #eef2f7;
  color: #344054;
  font-size: 16px;
}
.data-table.large-text th,
.data-table.large-text td { font-size: 21px; padding: 14px 16px; }
.interactive-layout {
  display: grid;
  grid-template-columns: 1fr 280px;
  gap: 22px;
  flex: 1;
  min-height: 0;
}
.iframe-wrap {
  height: 100%;
  min-height: 450px;
  border: 1px solid var(--line);
  border-radius: 12px;
  overflow: hidden;
  background: #f8fafc;
}
.iframe-wrap iframe {
  width: 100%;
  height: 100%;
  border: 0;
  background: #ffffff;
}
.open-link {
  margin-top: 8px;
  display: flex;
  gap: 16px;
  align-items: center;
  font-size: 15px;
  color: var(--muted);
}
.interactive-layout aside {
  padding: 22px;
  border-left: 1px solid var(--line);
  color: #344054;
}
.interactive-layout aside li { font-size: 19px; }
.figure-switcher {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 10px;
}
.switch-btn {
  border: 1px solid #cbd5e1;
  background: #ffffff;
  color: #344054;
  border-radius: 999px;
  padding: 7px 14px;
  font-size: 14px;
  font-weight: 850;
  cursor: pointer;
}
.switch-btn.active {
  background: #2563eb;
  border-color: #2563eb;
  color: #ffffff;
}
.small-links {
  margin-top: 28px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.feature-cards .card {
  min-height: 420px;
  display: flex;
  flex-direction: column;
}
.feature-list {
  margin-top: 18px;
  padding-left: 1.1em;
}
.feature-list li {
  font-size: 16px;
  line-height: 1.35;
  color: #344054;
}
.metric-cards {
  display: grid;
  gap: 10px;
  margin-top: 24px;
}
.metric-cards span {
  padding: 11px 14px;
  border-radius: 10px;
  background: #f8fafc;
  border: 1px solid #d8dee8;
  font-size: 18px;
  font-weight: 760;
  color: #344054;
}
.compact-row {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  justify-content: center;
  margin-top: 8px;
}
.compact-row span {
  padding: 9px 15px;
  border-radius: 999px;
  background: #f8fafc;
  border: 1px solid #d8dee8;
  color: #344054;
  font-size: 17px;
  font-weight: 760;
}
.inline-note {
  padding: 14px 18px;
  border-radius: 10px;
  background: #f8fafc;
  color: #475467;
  font-size: 20px;
  font-weight: 650;
}
.equal { grid-template-columns: 1fr 1fr; }
.limit-layout {
  display: grid;
  grid-template-columns: 0.9fr 1.1fr;
  gap: 28px;
  flex: 1;
  align-items: center;
}
.limit-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
}
.limit-grid .card {
  min-height: 170px;
  padding: 18px;
}
.limit-grid .card h3 { font-size: 19px; }
.limit-grid .card p { font-size: 16px; line-height: 1.38; }
.limit-side {
  display: grid;
  gap: 10px;
  align-items: center;
}
.limit-side img {
  width: 100%;
  max-height: 430px;
  object-fit: contain;
}
.small-note {
  color: #475467;
  font-size: 17px;
  line-height: 1.45;
  text-align: center;
}
.next-step-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 18px;
  flex: 1;
  align-items: stretch;
}
.next-step-card {
  border: 1px solid #d8dee8;
  border-radius: 12px;
  background: #ffffff;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.next-step-card img {
  width: 100%;
  aspect-ratio: 360 / 210;
  object-fit: contain;
}
.next-step-card p {
  font-size: 17px;
  color: #344054;
  line-height: 1.42;
}
.thanks-slide {
  flex: 1;
  display: grid;
  place-items: center;
  text-align: center;
  align-content: center;
  gap: 24px;
}
.thanks-title {
  font-size: 78px;
  line-height: 1;
  font-weight: 950;
  color: #2563eb;
}
.thanks-slide p {
  max-width: 820px;
  font-size: 28px;
  color: #344054;
}
.layer-stack {
  display: grid;
  gap: 18px;
  flex: 1;
  align-content: center;
}
.layer-stack div {
  border-left: 8px solid var(--blue);
  background: #f8fafc;
  padding: 22px 26px;
  border-radius: 10px;
}
.layer-stack div:nth-child(2) { border-left-color: var(--green); }
.layer-stack div:nth-child(3) { border-left-color: var(--purple); }
.layer-stack strong { display: block; font-size: 27px; margin-bottom: 7px; }
.layer-stack span { display: block; font-size: 22px; color: #475467; }
.center-claim {
  text-align: center;
  font-size: 38px;
  font-weight: 900;
  color: var(--blue);
  padding: 20px;
}
.placeholder {
  display: grid;
  place-items: center;
  min-height: 240px;
  border: 2px dashed #cbd5e1;
  border-radius: 12px;
  color: #667085;
  background: #f8fafc;
  font-size: 20px;
  text-align: center;
  padding: 24px;
}
.placeholder.large { min-height: 520px; }
#hud {
  position: fixed;
  inset: auto 24px 20px auto;
  display: flex;
  gap: 10px;
  align-items: center;
  z-index: 20;
}
.nav-btn {
  width: 38px;
  height: 38px;
  border: 1px solid #cbd5e1;
  background: rgba(255,255,255,0.9);
  color: #243447;
  border-radius: 999px;
  cursor: pointer;
  font-size: 20px;
  font-weight: 900;
}
#pageCounter {
  min-width: 74px;
  text-align: center;
  border: 1px solid #cbd5e1;
  background: rgba(255,255,255,0.9);
  color: #344054;
  border-radius: 999px;
  padding: 9px 12px;
  font-size: 14px;
  font-weight: 850;
}
#chapterBar {
  position: fixed;
  top: 18px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  gap: 8px;
  z-index: 20;
}
.chapter-pill {
  min-width: 126px;
  text-align: center;
  padding: 7px 12px;
  border-radius: 999px;
  background: rgba(255,255,255,0.78);
  border: 1px solid #d8dee8;
  color: #667085;
  font-size: 12px;
  font-weight: 850;
}
.chapter-pill.active {
  background: #2563eb;
  color: #ffffff;
  border-color: #2563eb;
}
.progress-track {
  position: fixed;
  left: 50%;
  bottom: 12px;
  transform: translateX(-50%);
  width: min(820px, 58vw);
  height: 5px;
  background: rgba(203, 213, 225, 0.75);
  border-radius: 999px;
  overflow: hidden;
  z-index: 19;
}
#progressFill {
  height: 100%;
  width: 0;
  background: linear-gradient(90deg, #2563eb, #16a34a);
}
@media (max-width: 900px) {
  .slide-inner { inset: 42px 44px 56px 44px; }
  h1 { font-size: 38px; }
  h2 { font-size: 28px; }
  p, li { font-size: 17px; }
  .two-col, .interactive-layout { grid-template-columns: 1fr; }
  .card-grid.three, .card-grid.four { grid-template-columns: 1fr 1fr; }
  .chapter-pill { min-width: auto; font-size: 11px; }
}
"""
    write_text(ASSET_DIR / "style.css", css)


def write_js() -> None:
    js = r"""
(function () {
  const slides = Array.from(document.querySelectorAll(".slide"));
  const pageCounter = document.getElementById("pageCounter");
  const progressFill = document.getElementById("progressFill");
  const chapterBar = document.getElementById("chapterBar");
  const chapters = JSON.parse(document.getElementById("chapterData").textContent);
  let current = 0;

  chapters.forEach((chapter) => {
    const pill = document.createElement("button");
    pill.className = "chapter-pill";
    pill.textContent = chapter;
    pill.type = "button";
    pill.addEventListener("click", () => {
      const idx = slides.findIndex((slide) => slide.dataset.chapter === chapter);
      if (idx >= 0) show(idx);
    });
    chapterBar.appendChild(pill);
  });

  function parseInitialSlide() {
    const params = new URLSearchParams(window.location.search);
    const fromQuery = parseInt(params.get("slide") || "", 10);
    const fromHash = (window.location.hash.match(/slide-(\d+)/) || [])[1];
    const n = Number.isFinite(fromQuery) ? fromQuery : parseInt(fromHash || "1", 10);
    return Math.min(Math.max((n || 1) - 1, 0), slides.length - 1);
  }

  function loadIframes(slide) {
    slide.querySelectorAll("iframe[data-src]").forEach((frame) => {
      if (!frame.getAttribute("src")) {
        frame.setAttribute("src", frame.dataset.src);
      }
    });
  }

  function show(index) {
    current = Math.min(Math.max(index, 0), slides.length - 1);
    slides.forEach((slide, i) => slide.classList.toggle("active", i === current));
    loadIframes(slides[current]);
    pageCounter.textContent = `${current + 1} / ${slides.length}`;
    progressFill.style.width = `${((current + 1) / slides.length) * 100}%`;
    const chapter = slides[current].dataset.chapter;
    document.querySelectorAll(".chapter-pill").forEach((pill) => {
      pill.classList.toggle("active", pill.textContent === chapter);
    });
    const url = new URL(window.location.href);
    url.searchParams.set("slide", String(current + 1));
    history.replaceState(null, "", `${url.pathname}?${url.searchParams.toString()}#slide-${current + 1}`);
  }

  function next() { show(current + 1); }
  function prev() { show(current - 1); }

  document.getElementById("nextBtn").addEventListener("click", next);
  document.getElementById("prevBtn").addEventListener("click", prev);
  window.addEventListener("keydown", (event) => {
    if (["ArrowRight", "PageDown", " "].includes(event.key)) {
      event.preventDefault();
      next();
    } else if (["ArrowLeft", "PageUp", "Backspace"].includes(event.key)) {
      event.preventDefault();
      prev();
    } else if (event.key === "Home") {
      show(0);
    } else if (event.key === "End") {
      show(slides.length - 1);
    }
  });
  window.addEventListener("hashchange", () => {
    const match = window.location.hash.match(/slide-(\d+)/);
    if (match) show(parseInt(match[1], 10) - 1);
  });
  document.addEventListener("click", (event) => {
    const button = event.target.closest(".switch-btn");
    if (!button) return;
    const frame = document.getElementById(button.dataset.frame);
    if (!frame || !button.dataset.target) return;
    frame.setAttribute("src", button.dataset.target);
    frame.setAttribute("data-src", button.dataset.target);
    button.parentElement.querySelectorAll(".switch-btn").forEach((item) => {
      item.classList.toggle("active", item === button);
    });
  });

  show(parseInitialSlide());
})();
"""
    write_text(ASSET_DIR / "deck.js", js)


def write_index(slides: list[str]) -> None:
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>黄铁矿地球化学 representation benchmark 汇报</title>
  <link rel="stylesheet" href="assets/style.css">
</head>
<body>
  <div id="chapterBar" aria-label="章节进度"></div>
  <main id="deck">
    {''.join(slides)}
  </main>
  <div id="hud">
    <button class="nav-btn" id="prevBtn" type="button" title="上一页">‹</button>
    <div id="pageCounter">1 / {len(slides)}</div>
    <button class="nav-btn" id="nextBtn" type="button" title="下一页">›</button>
  </div>
  <div class="progress-track"><div id="progressFill"></div></div>
  <script id="chapterData" type="application/json">{json.dumps(CHAPTERS, ensure_ascii=False)}</script>
  <script src="assets/deck.js"></script>
</body>
</html>
"""
    write_text(PRESENTATION_DIR / "index.html", html_text)


def write_static_backup(data: dict[str, object]) -> None:
    stage2_unsup: pd.DataFrame = data["stage2_unsup"]  # type: ignore[assignment]
    stage2_xgb: pd.DataFrame = data["stage2_xgb"]  # type: ignore[assignment]
    stage3: pd.DataFrame = data["stage3"]  # type: ignore[assignment]
    copied: dict[str, str | None] = data["copied"]  # type: ignore[assignment]
    stage2_table = stage2_unsup.merge(
        stage2_xgb[stage2_xgb["cv_scheme"].eq("stratified_kfold")][["representation", "macro_f1"]],
        on="representation",
        how="left",
    )
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>静态备份：黄铁矿 representation benchmark 汇报</title>
  <link rel="stylesheet" href="assets/style.css">
  <style>
    body {{ overflow: auto; background: #f3f6fb; }}
    .backup {{ max-width: 1180px; margin: 30px auto; padding: 24px; }}
    .backup-section {{ background:#fff; border:1px solid #d8dee8; border-radius:12px; padding:28px; margin:22px 0; }}
    .backup-section h2 {{ margin-bottom:16px; }}
    .backup img {{ max-width:100%; height:auto; border:1px solid #e5e7eb; border-radius:10px; }}
  </style>
</head>
<body>
  <div class="backup">
    <h1>黄铁矿主/微量元素 representation benchmark 与 Element2Vec 初探</h1>
    <p class="subtitle">静态备份版：当 iframe 或交互图加载失败时使用。</p>
    <div class="backup-section">
      <h2>研究主线</h2>
      {fig_img("report_logic_flow.svg", "report logic flow")}
    </div>
    <div class="backup-section">
      <h2>六维评价框架</h2>
      {fig_img("six_dimension_framework.svg", "six dimension framework")}
    </div>
    <div class="backup-section">
      <h2>Stage2 能力分化</h2>
      {fig_img("stage2_ability_profile.svg", "stage2 ability profile")}
      {simple_table(stage2_table, ["representation", "input_knn15", "kmeans_nmi", "gmm_nmi", "macro_f1"], ["representation", "input kNN15", "KMeans NMI", "GMM NMI", "XGB macro-F1"])}
    </div>
    <div class="backup-section">
      <h2>Stage2 balance UMAP 截图/入口</h2>
      {fig_img("geo_balance_12_UMAP1_UMAP2.png", "stage2 balance png")}
      <p><a href="{copied.get("stage2_balance") or "#"}" target="_blank">打开 Stage2 balance 交互图</a></p>
    </div>
    <div class="backup-section">
      <h2>Element2Vec 架构</h2>
      {fig_img("element2vec_architecture.svg", "element2vec architecture")}
    </div>
    <div class="backup-section">
      <h2>Stage3 关键指标</h2>
      {fig_img("stage3_space_comparison.svg", "stage3 comparison")}
      {simple_table(stage3, ["space_name", "knn15_neighborhood_purity", "kmeans_ari", "kmeans_nmi", "xgb_macro_f1_mean"], ["space", "kNN15", "ARI", "NMI", "XGB macro-F1"])}
    </div>
    <div class="backup-section">
      <h2>Stage3 encoded token UMAP 截图/入口</h2>
      {fig_img("sample_encoded_token_flat_pca12_space_3d.png", "stage3 encoded png")}
      <p><a href="{copied.get("stage3_encoded") or "#"}" target="_blank">打开 Stage3 encoded token 交互图</a></p>
    </div>
    <div class="backup-section">
      <h2>边界与下一步</h2>
      <p>本汇报为 exploratory report。Stage3 是全量训练后的 representation sanity check，不是严格泛化估计；后续需要 GroupKFold、外部数据和跨数据集迁移验证。</p>
    </div>
  </div>
</body>
</html>
"""
    write_text(PRESENTATION_DIR / "static_backup.html", html_text)


def write_readme(copied: dict[str, str | None]) -> None:
    used = [f"- `{k}`: `{v or 'not found'}`" for k, v in copied.items()]
    text = f"""# HTML 汇报页说明

本目录是黄铁矿主/微量元素地球化学 representation benchmark 的本地交互式 HTML 汇报页。

## 如何打开

双击 `index.html`，或用浏览器打开：

```text
presentation_html/index.html
```

如果交互图或 iframe 加载失败，打开静态备份版：

```text
presentation_html/static_backup.html
```

## 如何翻页

- 方向键左/右；
- 空格键进入下一页；
- 页面右下角按钮；
- 支持 `?slide=5` 和 `#slide-5` 定位。

## 交互图

已有 Plotly/UMAP3D HTML 没有 inline 到主页面，而是复制到了：

```text
presentation_html/interactive/
```

主要交互图包括：

- Stage2 balance UMAP3D；
- Stage3 sample encoded token UMAP3D；
- Stage3 log robust / label-masked latent 对照图；
- Stage3 raw 476D token embedding sanity-check 图（若存在）。

## 引用的文件来源

{chr(10).join(used)}

同时生成或复制了汇报图：

- `assets/figures/report_logic_flow.svg`
- `assets/figures/six_dimension_framework.svg`
- `assets/figures/stage2_ability_profile.svg`
- `assets/figures/stage3_space_comparison.svg`
- `assets/figures/element2vec_architecture.svg`

## 注意事项

本汇报为 exploratory report。Stage3 的 Transformer/Element2Vec 结果是全量训练后的 representation sanity check，不是严格泛化估计。

后续需要补充：

- GroupKFold / 按文献或矿区分组验证；
- 外部数据验证；
- 对元素 embedding、attention、元素共现网络的地质解释；
- 在更大全岩地球化学数据上训练通用元素表示并迁移到黄铁矿数据。
"""
    write_text(PRESENTATION_DIR / "README.md", text)


def main() -> None:
    ensure_dirs()
    copied = copy_inputs()

    stage2_unsup_path = asset("stage2_run") / "unsupervised" / "unsupervised_metrics.csv"
    stage2_xgb_path = asset("stage2_run") / "xgb" / "xgb_classification_metrics.csv"
    stage3_summary_path = asset("stage3_run") / "metrics" / "all_space_benchmark_summary.csv"
    stage2_unsup = read_csv_if_exists(stage2_unsup_path)
    stage2_xgb = read_csv_if_exists(stage2_xgb_path)
    stage3 = read_csv_if_exists(stage3_summary_path)

    save_table(stage2_unsup, "stage2_unsupervised_metrics.csv")
    save_table(stage2_xgb, "stage2_xgb_classification_metrics.csv")
    save_table(stage3, "stage3_all_space_benchmark_summary.csv")

    write_report_logic_flow_svg()
    write_six_dimension_svg()
    write_element2vec_architecture_svg()
    save_stage2_ability_profile(stage2_unsup, stage2_xgb)
    save_stage1_baseline_radar(stage2_unsup, stage2_xgb)
    save_stage2_ability_radar(stage2_unsup, stage2_xgb)
    save_stage3_space_comparison(stage3)
    save_stage3_capability_radar(stage3)
    write_next_step_svgs()
    write_css()
    write_js()

    data = {"stage2_unsup": stage2_unsup, "stage2_xgb": stage2_xgb, "stage3": stage3, "copied": copied}
    slides = build_slides_v2(data)
    write_index(slides)
    write_static_backup(data)
    write_readme(copied)

    print("Presentation generated:")
    print("presentation_html/index.html")
    print("presentation_html/static_backup.html")
    print("presentation_html/README.md")


if __name__ == "__main__":
    main()
