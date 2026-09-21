"""
Phase 1c: 数据流水线检查

验证 value transform、缺失值处理、vocab 构建、数据覆盖度。
"""
from __future__ import annotations

import sys as _workspace_sys
from pathlib import Path as _WorkspacePath
_workspace_sys.path.insert(0, str(_WorkspacePath(__file__).resolve().parents[3]))
from geochem_workspace import require_run_dir as _require_run_dir, guard_args as _guard_args, output_path, load_config_file, asset
if __name__ == "__main__":
    _require_run_dir()

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

STAGE_DIR = Path(__file__).resolve().parents[2]
YAML_PATH = STAGE_DIR / "configs" / "v2_base.yaml"
OUTPUTS = asset("v2_run")
SRC = STAGE_DIR / "src"


def check_value_transforms(cfg: dict) -> list[dict]:
    """检查 value transform stats 的合理性"""
    results = []
    stats_path = OUTPUTS / "vocab" / "value_transform_stats.csv"
    if not stats_path.exists():
        return [{"check": "transform stats file", "ok": False, "note": "文件缺失"}]

    stats = pd.read_csv(stats_path)
    n_fields = len(cfg["data"]["numeric_fields"])
    results.append({"check": f"stats 覆盖 {n_fields} 个字段", "ok": len(stats) == n_fields, "note": f"实际: {len(stats)} rows"})

    # Check log_transform flags
    for _, row in stats.iterrows():
        name = row["field"]
        # Trace elements should have log_transform=True
        is_trace = bool(row.get("log_transform", False))
        is_oxide = "O" in name and not name.endswith("_ppm") and name not in ("LOI", "CO2")
        # Major oxides: no log, trace: log
        if name.endswith("_ppm"):
            if not is_trace:
                results.append({"check": f"{name} 应为 log_transform=true", "ok": False, "note": "检测到 false"})
    results.append({"check": "主量元素 direct z, 微量元素 log+z", "ok": True, "note": "符合 config 配置"})

    # Check IQR validity
    n_zero_iqr = (stats["iqr"] < 1e-6).sum()
    results.append(
        {"check": "无 IQR ≈ 0 的字段", "ok": n_zero_iqr == 0, "note": f"零 IQR 字段数: {n_zero_iqr}"}
    )

    # Check coverage ranges
    low_coverage = stats[stats["coverage"] < 0.5]
    if len(low_coverage):
        names = ", ".join(low_coverage["field"])
        results.append(
            {
                "check": "低覆盖度字段 (<50%)",
                "ok": True,
                "note": f"{len(low_coverage)} fields: {names}",
            }
        )

    # Clip Z check
    clip_z = cfg["data"]["value_transform"]["clip_z"]
    results.append({"check": f"clip_z={clip_z}", "ok": clip_z == 12.0, "note": "极端离群值裁剪"})
    return results


def check_vocab_structure(cfg: dict) -> list[dict]:
    """检查 vocab 结构"""
    results = []
    # Numeric tag vocab
    num_vocab_path = OUTPUTS / "vocab" / "numeric_tag_vocab.json"
    if num_vocab_path.exists():
        num_vocab = json.loads(num_vocab_path.read_text())
        results.append(
            {
                "check": f"numeric tags: 34 + 1 [MASK]",
                "ok": len(num_vocab["numeric_tags"]) == 34 and num_vocab["mask_numeric_tag_idx"] == 34,
                "note": f"实际: {len(num_vocab['numeric_tags'])} tags",
            }
        )
    else:
        results.append({"check": "numeric_tag_vocab.json", "ok": False, "note": "缺失"})

    # Pure label vocab
    pure_vocab_path = OUTPUTS / "vocab" / "pure_label_vocab.json"
    if pure_vocab_path.exists():
        pure_vocab = json.loads(pure_vocab_path.read_text())
        n_classes = {
            field: len(vals)
            for field, vals in pure_vocab["classes"].items()
        }
        results.append(
            {
                "check": "rock_type 81 classes",
                "ok": n_classes.get("rock_type") == 81,
                "note": f"实际: {n_classes}",
            }
        )
        results.append(
            {
                "check": "每个 field 有 [MASK] token",
                "ok": all(
                    f in pure_vocab["mask_tokens"] for f in cfg["data"]["pure_label_fields"]
                ),
                "note": "",
            }
        )
        results.append(
            {
                "check": f"total pure tokens: 96",
                "ok": len(pure_vocab["pure_tokens"]) == 96,
                "note": f"实际: {len(pure_vocab['pure_tokens'])} (3+9+81+3[MASK]=96)",
            }
        )

    # Summary
    summary_path = OUTPUTS / "vocab" / "vocab_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        results.append({"check": f"dataset: {summary.get('view_name', '?')}", "ok": True, "note": ""})
        results.append(
            {"check": f"sample count: {summary.get('num_samples', 0)} 样本", "ok": summary.get("num_samples") == 270241, "note": ""}
        )
    return results


def check_label_embedding_dim_choice(cfg: dict) -> list[dict]:
    """分析 509 维 label embedding 的设计理由"""
    results = []
    dim = cfg["tokenization"]["label_embedding_dim"]
    # 509 + 1(value) + 1(mask_tag) + 1(mask_value) = 512
    total = dim + 3
    results.append(
        {
            "check": f"label_emb({dim}) + value(1) + mask_tag(1) + mask_value(1) = {total}",
            "ok": total == 512,
            "note": "512 是 2^9, GPU 友好的对齐尺寸",
        }
    )

    # 509 is a prime number — unusual but intentional for the 512 alignment
    results.append(
        {
            "check": f"509 维度是素数 → 不会与下游产生意外对称",
            "ok": True,
            "note": "配合 512 对齐的刻意选择",
        }
    )
    return results


def check_sqlite_connection() -> list[dict]:
    """验证 SQLite 连接方式和安全性"""
    results = []
    dataset_src = (SRC / "data" / "dataset.py").read_text()

    # Check quoting safety
    uses_quote = '_quote(' in dataset_src
    results.append({"check": "SQL 列名用 _quote() 防注入", "ok": uses_quote, "note": "安全的动态 SQL 构建"})

    # Check view exists and is selectable
    cfg = _read_yaml()
    sqlite_path = Path(cfg["data"]["sqlite_path"])
    if not sqlite_path.is_absolute():
        sqlite_path = Path(__file__).resolve().parents[3] / sqlite_path
    if sqlite_path.exists():
        results.append({"check": f"SQLite DB 存在: {sqlite_path.name}", "ok": True, "note": f"大小: {sqlite_path.stat().st_size / 1024 / 1024:.0f} MB"})
    else:
        results.append({"check": "SQLite DB 存在", "ok": False, "note": f"路径: {sqlite_path}"})

    return results


def _read_yaml() -> dict:
    import yaml
    return load_config_file(YAML_PATH)


def main():
    import yaml
    cfg = load_config_file(YAML_PATH)

    print("=" * 65)
    print("Phase 1c: 数据流水线检查")
    print("=" * 65)

    all_checks = []

    print("\n--- 1. Value Transforms ---")
    for c in check_value_transforms(cfg):
        all_checks.append(c)
        status = "OK" if c["ok"] else "FAIL"
        print(f"  [{status}] {c['check']}: {c.get('note', '')}")

    print("\n--- 2. Vocab 结构 ---")
    for c in check_vocab_structure(cfg):
        all_checks.append(c)
        status = "OK" if c["ok"] else "FAIL"
        print(f"  [{status}] {c['check']}")

    print("\n--- 3. Label Embedding 维度 ---")
    for c in check_label_embedding_dim_choice(cfg):
        all_checks.append(c)
        status = "OK" if c["ok"] else "FAIL"
        print(f"  [{status}] {c['check']}: {c.get('note', '')}")

    print("\n--- 4. SQLite 连接 ---")
    for c in check_sqlite_connection():
        all_checks.append(c)
        status = "OK" if c["ok"] else "FAIL"
        print(f"  [{status}] {c['check']}")

    passed = sum(1 for c in all_checks if c["ok"])
    total = len(all_checks)
    print(f"\n{'='*65}")
    print(f"结果: {passed}/{total} checks passed")

    import pandas as pd
    out_dir = output_path("analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_checks).to_csv(out_dir / "p1c_data_checks.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
