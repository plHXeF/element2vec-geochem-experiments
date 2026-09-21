"""
Phase 2a: 训练曲线深度分析

分析 train_log.csv 的:
- Loss 收敛曲线 (total + 5 个分项)
- Learning rate 曲线
- 收敛充分性判断
- Token 分布统计
- 分阶段 loss 变化速率
"""
from __future__ import annotations

import sys as _workspace_sys
from pathlib import Path as _WorkspacePath
_workspace_sys.path.insert(0, str(_WorkspacePath(__file__).resolve().parents[3]))
from geochem_workspace import require_run_dir as _require_run_dir, guard_args as _guard_args, output_path, load_config_file, asset
if __name__ == "__main__":
    _require_run_dir()

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

OUT_DIR = output_path("analysis")
LOG_PATH = asset("v2_run") / "logs" / "train_log.csv"


def load_log() -> pd.DataFrame:
    df = pd.read_csv(LOG_PATH)
    df["epoch"] = df["step"] / df["step"].max() * df["epoch"].max()
    return df


def analyze_convergence(df: pd.DataFrame) -> dict:
    """分析收敛质量"""
    results = {}

    # Overall loss reduction
    initial = df.iloc[0]["total_loss"]
    final = df.iloc[-1]["total_loss"]
    best = df["total_loss"].min()
    best_step = df["total_loss"].idxmin()
    results["initial_loss"] = float(initial)
    results["final_loss"] = float(final)
    results["best_loss"] = float(best)
    results["best_step"] = int(df.loc[best_step, "step"])
    results["reduction_ratio"] = float(initial / final)

    # Stability in final phase
    final_20pct = df[df["step"] > df["step"].max() * 0.8]
    results["final_20pct_mean_loss"] = float(final_20pct["total_loss"].mean())
    results["final_20pct_std_loss"] = float(final_20pct["total_loss"].std())
    results["final_20pct_min"] = float(final_20pct["total_loss"].min())
    results["final_20pct_max"] = float(final_20pct["total_loss"].max())

    # Fluctuation ratio (instability indicator)
    results["cv_in_final_phase"] = float(final_20pct["total_loss"].std() / final_20pct["total_loss"].mean())

    # Per-component analysis
    components = ["numeric_value_loss", "numeric_tag_loss", "rock_group_loss", "rock_origin_loss", "rock_type_loss"]
    for comp in components:
        results[f"{comp}_initial"] = float(df.iloc[0][comp])
        results[f"{comp}_final"] = float(df.iloc[-1][comp])
        results[f"{comp}_best"] = float(df[comp].min())

    # Learning rate
    results["lr_initial"] = float(df.iloc[0]["learning_rate"])
    results["lr_peak"] = float(df["learning_rate"].max())
    results["lr_final"] = float(df.iloc[-1]["learning_rate"])
    results["lr_peak_step"] = int(df.loc[df["learning_rate"].idxmax(), "step"])

    # Token stats
    results["mean_tokens_avg"] = float(df["num_tokens_mean"].mean())
    results["mean_tokens_std"] = float(df["num_tokens_mean"].std())

    return results


def analyze_loss_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """分析损失下降速率变化"""
    window = max(1, len(df) // 20)  # ~5% windows
    rows = []
    for i in range(0, len(df), window):
        chunk = df.iloc[i : i + window]
        rows.append(
            {
                "start_step": int(chunk.iloc[0]["step"]),
                "end_step": int(chunk.iloc[-1]["step"]),
                "avg_total_loss": float(chunk["total_loss"].mean()),
                "lr": float(chunk["learning_rate"].mean()),
                "loss_change_rate": float(
                    (chunk.iloc[-1]["total_loss"] - chunk.iloc[0]["total_loss"])
                    / max(1, chunk.iloc[-1]["step"] - chunk.iloc[0]["step"])
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_loss_curves(df: pd.DataFrame):
    """绘制完整的损失曲线"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    # Panel 1: Total loss
    ax = axes[0, 0]
    ax.plot(df["step"], df["total_loss"], linewidth=0.5, color="#1f77b4", alpha=0.7)
    # Rolling mean
    window = max(1, len(df) // 200)
    ax.plot(df["step"], df["total_loss"].rolling(window=window, center=True).mean(), linewidth=1.5, color="#d62728")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Total Loss")
    ax.grid(True, alpha=0.2)
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

    # Panel 2: Component losses
    ax = axes[0, 1]
    components = {
        "numeric_value_loss": ("numeric_value", "#1f77b4"),
        "numeric_tag_loss": ("numeric_tag", "#ff7f0e"),
        "rock_type_loss": ("rock_type (81)", "#d62728"),
        "rock_origin_loss": ("rock_origin (9)", "#2ca02c"),
        "rock_group_loss": ("rock_group (3)", "#9467bd"),
    }
    for col, (label, color) in components.items():
        window = max(1, len(df) // 200)
        smoothed = df[col].rolling(window=window, center=True).mean()
        ax.plot(df["step"], smoothed, linewidth=1.0, color=color, label=label, alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Component Losses (smoothed)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.2)

    # Panel 3: LR curve
    ax = axes[1, 0]
    ax.plot(df["step"], df["learning_rate"], linewidth=1.0, color="#7b3294")
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Cosine Learning Rate Schedule")
    ax.grid(True, alpha=0.2)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1e"))

    # Panel 4: Loss in linear scale (last 50% of training)
    ax = axes[1, 1]
    half = df[df["step"] >= df["step"].max() * 0.5]
    ax.plot(half["step"], half["total_loss"], linewidth=0.5, color="#1f77b4", alpha=0.5)
    window = max(1, len(half) // 100)
    ax.plot(
        half["step"],
        half["total_loss"].rolling(window=window, center=True).mean(),
        linewidth=1.5,
        color="#d62728",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"Total Loss (last 50%: steps {half['step'].min():.0f}–{half['step'].max():.0f})")
    ax.grid(True, alpha=0.2)

    # Panel 5: Loss velocity (change rate)
    ax = axes[2, 0]
    vel = analyze_loss_velocity(df)
    ax.plot(vel["end_step"], vel["loss_change_rate"], linewidth=1.0, color="#ff7f0e")
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("ΔLoss/step")
    ax.set_title("Loss Change Rate")
    ax.grid(True, alpha=0.2)

    # Panel 6: Token distribution
    ax = axes[2, 1]
    token_bins = np.arange(
        int(df["num_tokens_mean"].min()),
        int(df["num_tokens_mean"].max()) + 2,
    )
    ax.hist(df["num_tokens_mean"], bins=50, color="#17becf", edgecolor="white", alpha=0.8)
    ax.axvline(x=df["num_tokens_mean"].mean(), color="#d62728", linewidth=2, linestyle="--", label=f"mean={df['num_tokens_mean'].mean():.1f}")
    ax.set_xlabel("Tokens per Sample")
    ax.set_ylabel("Frequency")
    ax.set_title("Token Count Distribution per Batch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.suptitle("Stage 3.5 Element2Vec V2 — Training Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "p2a_loss_analysis.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "p2a_loss_analysis.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  图表已保存: p2a_loss_analysis.png/.svg")


def plot_lr_detail(df: pd.DataFrame):
    """LR warmup 阶段的详细图"""
    fig, ax = plt.subplots(figsize=(10, 4))
    first_steps = df[df["step"] <= 500]
    ax.plot(first_steps["step"], first_steps["learning_rate"], linewidth=1.5, color="#7b3294")
    ax.axvline(x=150, color="red", linewidth=1, linestyle="--", alpha=0.5, label="warmup_steps=150")
    ax.axhline(y=df["learning_rate"].max(), color="green", linewidth=0.5, linestyle="--", alpha=0.5, label=f"peak={df['learning_rate'].max():.2e}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate — Warmup Phase Detail")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1e"))
    plt.tight_layout()
    fig.savefig(OUT_DIR / "p2a_lr_warmup.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 65)
    print("Phase 2a: 训练曲线深度分析")
    print("=" * 65)

    df = load_log()
    print(f"  加载 {len(df)} steps 的训练日志")

    # Convergence analysis
    results = analyze_convergence(df)
    print(f"\n--- 收敛摘要 ---")
    print(f"  Initial loss:     {results['initial_loss']:.4f}")
    print(f"  Final loss:       {results['final_loss']:.4f}")
    print(f"  Best loss:        {results['best_loss']:.4f} (step {results['best_step']})")
    print(f"  Reduction ratio:  {results['reduction_ratio']:.1f}x")
    print(f"  Final 20% CV:     {results['cv_in_final_phase']:.3f}")
    print(f"  Final 20% range:  [{results['final_20pct_min']:.4f}, {results['final_20pct_max']:.4f}]")
    print(f"  LR initial/peak/final: {results['lr_initial']:.2e} / {results['lr_peak']:.2e} / {results['lr_final']:.2e}")
    print(f"  平均 tokens/sample: {results['mean_tokens_avg']:.1f} ± {results['mean_tokens_std']:.1f}")

    print(f"\n--- 分项 Loss 变化 ---")
    for comp in ["numeric_value_loss", "numeric_tag_loss", "rock_group_loss", "rock_origin_loss", "rock_type_loss"]:
        init = results[f"{comp}_initial"]
        final_val = results[f"{comp}_final"]
        print(f"  {comp}: {init:.4f} → {final_val:.4f} ({init / max(final_val, 1e-6):.0f}x)")

    # Convergence assessment
    print(f"\n--- 收敛质量判断 ---")
    if results["cv_in_final_phase"] > 0.5:
        print(f"  ⚠️ 最后阶段波动大 (CV={results['cv_in_final_phase']:.2f} > 0.5)")
        print(f"     可能原因: LR 衰减至 0 导致步骤间随机性主导")
    if results["final_loss"] / results["best_loss"] > 2:
        print(f"  ⚠️ 最终 loss ({results['final_loss']:.3f}) 是最优的 {results['final_loss']/results['best_loss']:.1f}x")
        print(f"     最优 step = {results['best_step']} — checkpoint 不是最优步")
    if results["lr_final"] < 1e-8:
        print(f"  ℹ️ LR 已衰减到零 ({results['lr_final']:.2e}) — 训练无法继续改进")
    if results["rock_group_loss_final"] < 0.001 and results["rock_origin_loss_final"] < 0.01:
        print(f"  ✓ rock_group + rock_origin 近乎完美记忆 (< 0.01)")
    if results["rock_type_loss_final"] > 0.1:
        print(f"  ℹ️ rock_type (81类) 仍有交叉熵 ~{results['rock_type_loss_final']:.3f} — 未完全收敛")

    # Generate plots
    print(f"\n--- 生成图表 ---")
    plot_loss_curves(df)
    plot_lr_detail(df)

    # Save results
    pd.Series(results).to_csv(OUT_DIR / "p2a_convergence_metrics.csv", header=["value"])
    print(f"\n  指标已保存: p2a_convergence_metrics.csv")
    print(f"  图表已保存: p2a_lr_warmup.png")


if __name__ == "__main__":
    main()
