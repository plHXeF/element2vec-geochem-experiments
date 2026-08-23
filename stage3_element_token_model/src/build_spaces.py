from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from utils import OUTPUT_DIR, STAGE_DIR, WORKSPACE_DIR, ensure_output_dirs, load_config, save_dataframe


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {"sample_id", "deposit_type"}]


def _pca_space(df: pd.DataFrame, n_components: int, random_seed: int) -> pd.DataFrame:
    cols = _feature_cols(df)
    x = df[cols].to_numpy(dtype=float)
    n = min(n_components, x.shape[1])
    coords = PCA(n_components=n, random_state=random_seed).fit_transform(x)
    out = pd.DataFrame(coords, columns=[f"pc_{i+1}" for i in range(coords.shape[1])])
    out.insert(0, "deposit_type", df["deposit_type"].astype(str).to_numpy())
    out.insert(0, "sample_id", df["sample_id"].astype(str).to_numpy())
    return out


def _maybe_previous_baseline(sample_meta: pd.DataFrame) -> tuple[pd.DataFrame | None, str]:
    candidates = [
        WORKSPACE_DIR / "stage2_geo_representation_xgb" / "outputs" / "representations" / "log_robust_z_14.csv",
        WORKSPACE_DIR / "stage2_geo_representation_xgb" / "outputs" / "representations" / "rank_uniform_z_14.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            if "deposit_type" not in df.columns:
                df = df.merge(sample_meta[["sample_id", "deposit_type"]], on="sample_id", how="left")
                ordered = ["sample_id", "deposit_type"] + [c for c in df.columns if c not in {"sample_id", "deposit_type"}]
                df = df[ordered]
            return df, f"found previous baseline: {path}"
    return None, "previous baseline not found"


def build_spaces(config: dict) -> dict:
    ensure_output_dirs()
    spaces_dir = OUTPUT_DIR / "spaces"
    pca_dim = int(config.get("pca_dim", 12))
    seed = int(config.get("random_seed", 42))

    data = np.load(OUTPUT_DIR / "preprocessed" / "pyrite_token_data.npz", allow_pickle=True)
    sample_meta = pd.DataFrame(
        {
            "sample_id": data["sample_id"].astype(str),
            "deposit_type": data["deposit_type"].astype(str),
        }
    )
    elements = data["elements"].astype(str).tolist()
    log_df = pd.DataFrame(data["element_values"], columns=elements)
    log_df.insert(0, "deposit_type", sample_meta["deposit_type"])
    log_df.insert(0, "sample_id", sample_meta["sample_id"])
    save_dataframe(log_df, spaces_dir / "log_robust_space.csv")
    save_dataframe(_pca_space(log_df, pca_dim, seed), spaces_dir / "log_robust_pca12_space.csv")

    notes = []
    prev, note = _maybe_previous_baseline(sample_meta)
    notes.append(note)
    if prev is not None:
        save_dataframe(prev, spaces_dir / "previous_baseline_space.csv")

    input_flat = pd.read_csv(OUTPUT_DIR / "embeddings" / "sample_input_token_flat.csv")
    encoded_flat = pd.read_csv(OUTPUT_DIR / "embeddings" / "sample_encoded_token_flat.csv")
    save_dataframe(_pca_space(input_flat, pca_dim, seed), spaces_dir / "sample_input_token_flat_pca12_space.csv")
    save_dataframe(_pca_space(encoded_flat, pca_dim, seed), spaces_dir / "sample_encoded_token_flat_pca12_space.csv")

    latent_path = OUTPUT_DIR / "latents" / "transformer_latent_label_masked_mean_element.csv"
    if not latent_path.exists():
        raise FileNotFoundError(f"Missing main transformer latent: {latent_path}")
    latent = pd.read_csv(latent_path)
    save_dataframe(latent, spaces_dir / "transformer_latent_label_masked_space.csv")

    (OUTPUT_DIR / "logs" / "build_spaces_notes.txt").write_text("\n".join(notes), encoding="utf-8")
    print("[build_spaces] wrote benchmark spaces")
    return {"notes": notes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(STAGE_DIR / "configs" / "pyrite_token_transformer.yaml"))
    args = parser.parse_args()
    build_spaces(load_config(args.config))


if __name__ == "__main__":
    main()
