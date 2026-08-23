from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import load_yaml, resolve_path, set_reproducible_seed
from .experiment_context import ExperimentContext
from .subspace_projection import ProjectionManager


MAIN_DIMS = (64, 32, 3, 2)


def _dirs(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "audit": root / "audit",
        "projections": root / "projections",
        "pca": root / "projections" / "pca_models",
        "rp": root / "projections" / "random_projection_models",
        "pools": root / "projections" / "fitting_pools",
        "variance": root / "projections" / "explained_variance",
        "results": root / "results",
        "figures": root / "figures",
        "reports": root / "reports",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    for family in list("ABCDEFGHIJKX") + ["S"]:
        (paths["results"] / family).mkdir(parents=True, exist_ok=True)
    for name in ("dimension_curves", "pca2", "pca3", "random_projection_controls", "pc_attribution"):
        (paths["figures"] / name).mkdir(parents=True, exist_ok=True)
    return paths


def _fixed_pools(meta: pd.DataFrame, settings: dict[str, Any], pool_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(settings["seed"]))
    order = rng.permutation(len(meta))
    n_fit = min(int(settings["pca_fit_samples"]), len(meta) - 1)
    fit_idx = np.sort(order[:n_fit])
    remaining = order[n_fit:]
    n_eval = min(int(settings["evaluation_samples"]), len(remaining))
    eval_idx = np.sort(remaining[:n_eval])
    meta.iloc[fit_idx][["sample_id"]].assign(pool="pca_fit").to_csv(pool_dir / "sample_fit_pool.csv", index=False)
    meta.iloc[eval_idx][["sample_id"]].assign(pool="evaluation").to_csv(pool_dir / "sample_evaluation_pool.csv", index=False)
    pd.DataFrame({
        "pool": ["pca_fit", "evaluation"],
        "n": [len(fit_idx), len(eval_idx)],
        "seed": [settings["seed"], settings["seed"]],
        "target_labels_used": [False, False],
    }).to_csv(pool_dir / "pool_manifest.csv", index=False)
    return fit_idx, eval_idx


def representation_references(
    ctx: ExperimentContext, fit_idx: np.ndarray, settings: dict[str, Any], pool_dir: Path
) -> dict[str, np.ndarray]:
    token_meta = ctx.token_meta
    fit_ids = set(ctx.meta.iloc[fit_idx]["sample_id"].tolist())
    token_candidates = np.flatnonzero(token_meta["sample_id"].isin(fit_ids).to_numpy())
    rng = np.random.default_rng(int(settings["seed"]) + 17)
    n_token = min(int(settings["token_fit_samples"]), len(token_candidates))
    token_idx = np.sort(rng.choice(token_candidates, size=n_token, replace=False))
    token_meta.iloc[token_idx][["sample_id", "field"]].assign(pool="token_pca_fit").to_csv(
        pool_dir / "token_fit_pool.csv", index=False
    )
    lat = ctx.latents("numeric_only")
    rand = ctx.latents("random_numeric_only")
    tokens = ctx.token_arrays()
    projected_labels = ctx.project_pure_labels(trained=True)
    projected_labels_random = ctx.project_pure_labels(trained=False)
    refs = {
        "static_trained_numeric": ctx.static_trained["numeric"],
        "static_random_numeric": ctx.static_random["numeric"],
        "static_trained_pure": ctx.static_trained["pure"],
        "static_random_pure": ctx.static_random["pure"],
        "mlp_projected_numeric_token": tokens["projected"][token_idx],
        "contextual_numeric_token": tokens["contextual"][token_idx],
        "sample_projected_mean_numeric": lat["projected_mean_numeric"][fit_idx],
        "sample_contextual_mean_numeric": lat["encoded_mean_numeric"][fit_idx],
        "sample_cls": lat["cls"][fit_idx],
        "random_sample_projected_mean_numeric": rand["projected_mean_numeric"],
        "random_sample_contextual_mean_numeric": rand["encoded_mean_numeric"],
        "random_sample_cls": rand["cls"],
        "pure_label_projected": projected_labels,
        "random_pure_label_projected": projected_labels_random,
        "cross_joint_contextual_label": np.concatenate(
            [lat["encoded_mean_numeric"][fit_idx], projected_labels], axis=0
        ),
    }
    return refs


def _representation_for_experiment(row: pd.Series) -> tuple[str, bool, str]:
    eid = str(row["experiment_id"])
    family = str(row["family"])
    level = str(row["representation_level"]).lower()
    unavailable = eid in {
        "B4_detection_limit_unavailable", "D3b_nonlinear_axis_failure",
        "G3_protolith_metamorphic", "I0_strict_cipw_unavailable",
        "K4_checkpoint_sensitivity_unavailable", "K5_umap_parameter_unavailable",
    }
    if unavailable:
        return "none", False, "invariant technical/coverage limitation"
    if family == "A":
        return "static_trained_numeric", False, "static numeric-tag space"
    if family == "B":
        return "mlp_projected_numeric_token", False, "MLP-projected numeric token"
    if family == "C" or eid in {"X1_context_raw_control", "X2_role_raw_control"}:
        return "contextual_numeric_token", False, "contextual token space"
    if family == "G":
        return "pure_label_projected", False, "pure-label projected space"
    if family == "H" or eid in {"X3_cross_space_relation_followup", "I4_label_residual_norms"}:
        return "cross_joint_contextual_label", True, "sample-label shared coordinate system"
    if "cls" in level:
        return "sample_cls", False, "CLS sample representation"
    return "sample_contextual_mean_numeric", False, "contextual mean sample/pooled latent"


def run(config: dict[str, Any]) -> None:
    base_config = load_yaml(config["base_config"])
    settings = config["subspace_audit"]
    root = resolve_path(settings["output_root"])
    paths = _dirs(root)
    set_reproducible_seed(int(settings["seed"]), int(base_config["runtime"].get("num_threads", 8)))
    ctx = ExperimentContext(base_config)
    fit_idx, eval_idx = _fixed_pools(ctx.meta, settings, paths["pools"])
    refs = representation_references(ctx, fit_idx, settings, paths["pools"])
    manager = ProjectionManager(paths["projections"], seed=int(settings["seed"]), dtype=settings["numerical_dtype"])

    dimension_rows = []
    dimensions_json: dict[str, Any] = {}
    for name, reference in refs.items():
        model = manager.fit_pca(name, reference, max_components=64)
        info = {
            "native_dimension": int(reference.shape[1]),
            "fitting_rows": int(reference.shape[0]),
            "estimable_rank": int(np.asarray(model["estimable_rank"])),
            "fitted_components": int(len(model["components"])),
            "upstream_or_final": "upstream" if name.startswith("static_") else "model_output",
            "spaces": {"FULL": {"applicable": True, "reason": "native output"}},
        }
        for dim in MAIN_DIMS:
            pca_info = manager.pca_info(name, reference, dim)
            info["spaces"][f"PCA{dim}"] = {
                "applicable": bool(pca_info["applicable"]),
                "reason": pca_info["reason"],
                "explained_variance": pca_info["explained_variance"],
            }
            dimension_rows.append(pca_info)
        dimensions_json[name] = info
    manager.write_manifest()
    (paths["audit"] / "subspace_available_dimensions.json").write_text(
        json.dumps(dimensions_json, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
    )
    pd.DataFrame(dimension_rows).to_csv(paths["audit"] / "representation_dimension_table.csv", index=False)

    original_root = resolve_path(base_config["project"]["results_dir"])
    registry = pd.read_csv(original_root / "experiment_registry.csv", low_memory=False)
    compatibility = []
    saved_id_count = 0
    for _, row in registry.iterrows():
        rep, cross, note = _representation_for_experiment(row)
        sample_file = original_root / str(row["experiment_id"]) / "samples.csv"
        saved_rows = len(pd.read_csv(sample_file)) if sample_file.exists() else 0
        saved_id_count += int(saved_rows > 0)
        dim_info = dimensions_json.get(rep)
        compat = {
            "experiment_id": row["experiment_id"],
            "family": row["family"],
            "original_status": row["status"],
            "representation_level": row["representation_level"],
            "projection_reference": rep,
            "cross_space": cross,
            "sample_ids_saved": saved_rows > 0,
            "saved_sample_count": saved_rows,
            "sample_reproduction": "exact saved IDs" if saved_rows else "deterministic code replay from seed 42",
            "FULL": "applicable" if rep != "none" else "invariant_unavailable",
            "limitation": note,
        }
        for dim in MAIN_DIMS:
            if rep == "none":
                compat[f"PCA{dim}"] = "invariant_unavailable"
            else:
                space = dim_info["spaces"][f"PCA{dim}"]
                compat[f"PCA{dim}"] = "applicable" if space["applicable"] else space["reason"]
        compatibility.append(compat)
    comp = pd.DataFrame(compatibility)
    comp.to_csv(paths["audit"] / "subspace_experiment_compatibility.csv", index=False, encoding="utf-8-sig")

    native_lines = []
    for name, info in dimensions_json.items():
        native_lines.append(
            f"| {name} | {info['native_dimension']} | {info['fitting_rows']} | "
            f"{info['estimable_rank']} | {info['upstream_or_final']} |"
        )
    applicable_counts = {space: int((comp[space] == "applicable").sum()) for space in ["FULL", "PCA64", "PCA32", "PCA3", "PCA2"]}
    audit_md = f"""# Element2Vec V2 子空间表示审计

## 已读取的上一轮证据

- 主报告：`stage3_6_element2vec_space_exploration/reports/element2vec_exploration_report.md`。
- 实验 registry：50 行，覆盖 A–K 与 X；所有原实验配置、指标、图和观察目录均存在。
- 共享缓存：12,000 个样品，34,000 个 contextual/projected token，随机模型样品 3,000 个；随机种子为 42。
- 旧 `pca_sweep` 已覆盖 256/128/64/32/16/8/5/3/2D，但属于“把旧实验函数置于投影 context 后重放”的敏感性扫描；本任务不把它单独当作低秩地球化学语义证据，而只作为 paired replay 数据源，并新增 RP、whitening、drop-PC、PC attribution、energy、bootstrap 与混杂控制。
- 历史样品 ID 限制：50 个旧 `samples.csv` 中只有 {saved_id_count} 个非空。其余实验通过固定代码路径、固定 seed=42、固定端元/扰动参数重放；无法声称旧文件保存了逐行样品 ID。

## 原生维度、拟合秩与上游/最终表示

| representation | native D | fitting rows | estimable rank | provenance |
| --- | ---: | ---: | ---: | --- |
{chr(10).join(native_lines)}

`static_*` 的 509D 是 embedding lookup 的上游输出；MLP 后 token 与 Transformer/pooled latent 的最终坐标均为 768D。35 个 numeric tags 虽然原生为 509D，但样本秩至多 34，所以 PCA64 不可估计；PCA32 可估计但已接近最大秩，必须谨慎解释稳定性。本审计不会补零或复制维度。96 个纯标签允许 PCA64，但其小词表稳定性必须由 bootstrap 检查。

## PCA fitting/evaluation pool

- PCA fitting pool：{len(fit_idx)} 个固定样品 ID，不使用目标标签。
- evaluation pool：{len(eval_idx)} 个互斥固定样品 ID。
- token fitting pool：{len(refs['contextual_numeric_token'])} 个来自 fitting 样品的固定 token。
- 纯标签 PCA：在完整可用词表上无监督拟合；跨空间实验使用 `cross_joint_contextual_label` 的单一共享 basis，禁止分别 PCA 后直接比较。
- 旧实验的监督 probe 仍保留原定义以保证 paired replay；新增 S 系列监督/混杂指标仅在固定 evaluation pool 上评价，并把 PCA basis 固定在 reference pool。

## 完整重放兼容性

- FULL/PCA64/PCA32/PCA3/PCA2 可运行实验数：{applicable_counts}。
- 6 个覆盖或技术失败实验保持 invariant unavailable/failed，不能因投影而变成正结果。
- A 家族受 35 个 tag 的拟合秩限制；H、I4、X3 属于跨空间实验，必须使用共享 basis 或训练外 alignment。
- 严格 CIPW 仍未实现；I1–I4 中可运行部分只能称为 NNLS normative proxy。

## 禁止项

- 未使用补零、复制维度或结果驱动的标准化。
- PCA 不使用实验目标标签。
- UMAP 不参与向量差、余弦、类比或状态判定。
"""
    (paths["audit"] / "subspace_representation_audit.md").write_text(audit_md, encoding="utf-8")
    print(f"[subspace-audit] wrote audit to {paths['audit']}")
    print(f"[subspace-audit] representations={len(refs)} experiments={len(comp)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="stage3_6_element2vec_space_exploration/configs/subspace_audit.yaml")
    args = parser.parse_args()
    run(load_yaml(args.config))


if __name__ == "__main__":
    main()
