from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from utils import OUTPUT_DIR, save_dataframe


PALETTE = {
    "orogenic": "#1f77b4",
    "sedimentary": "#2ca02c",
    "porphyry": "#ff7f0e",
    "carlin": "#d62728",
    "ni-cu-pge": "#9467bd",
    "vms": "#8c564b",
}


def _plot_static_3d(coords: pd.DataFrame, space_name: str, method: str) -> None:
    fig = plt.figure(figsize=(8.4, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    for label, group in coords.groupby("deposit_type", sort=False):
        ax.scatter(
            group["dim1"],
            group["dim2"],
            group["dim3"],
            s=9,
            alpha=0.72,
            color=PALETTE.get(label, "#777777"),
            label=label,
        )
    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")
    ax.set_zlabel("dim3")
    ax.set_title(f"{space_name}: {method} 3D")
    ax.legend(fontsize=8, loc="best")
    fig.savefig(OUTPUT_DIR / "figures" / f"{space_name}_3d.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "figures" / f"{space_name}_3d.svg", bbox_inches="tight")
    plt.close(fig)


def _write_html(coords: pd.DataFrame, space_name: str, method: str) -> str:
    try:
        import plotly.express as px  # type: ignore

        fig = px.scatter_3d(
            coords,
            x="dim1",
            y="dim2",
            z="dim3",
            color="deposit_type",
            hover_data=["sample_id"],
            title=f"{space_name}: {method} 3D",
            opacity=0.75,
        )
        path = OUTPUT_DIR / "figures" / f"{space_name}_3d.html"
        fig.write_html(path)
        return "plotly package"
    except Exception:
        pass

    traces = []
    for label, group in coords.groupby("deposit_type", sort=False):
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": str(label),
                "x": group["dim1"].round(6).tolist(),
                "y": group["dim2"].round(6).tolist(),
                "z": group["dim3"].round(6).tolist(),
                "text": group["sample_id"].astype(str).tolist(),
                "marker": {"size": 3, "opacity": 0.75, "color": PALETTE.get(label, "#777777")},
            }
        )
    html = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>{space_name} 3D</title></head>
<body>
<div id="plot" style="width:100%;height:92vh;"></div>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script>
const data = {json.dumps(traces)};
const layout = {{title: {json.dumps(space_name + ': ' + method + ' 3D')}, scene: {{xaxis: {{title: 'dim1'}}, yaxis: {{title: 'dim2'}}, zaxis: {{title: 'dim3'}}}}}};
Plotly.newPlot('plot', data, layout);
</script>
</body>
</html>"""
    path = OUTPUT_DIR / "figures" / f"{space_name}_3d.html"
    path.write_text(html, encoding="utf-8")
    return "plotly CDN html fallback"


def make_3d_visualization(space_name: str, x: np.ndarray, sample_id: np.ndarray, labels: np.ndarray, config: dict) -> dict:
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=int(config.get("umap_n_neighbors", 30)),
            min_dist=float(config.get("umap_min_dist", 0.1)),
            random_state=int(config.get("random_seed", 42)),
        )
        coords_arr = reducer.fit_transform(x)
        method = "umap"
        note = "ok"
    except Exception as exc:
        coords_arr = PCA(n_components=min(3, x.shape[1]), random_state=int(config.get("random_seed", 42))).fit_transform(x)
        if coords_arr.shape[1] < 3:
            coords_arr = np.pad(coords_arr, ((0, 0), (0, 3 - coords_arr.shape[1])), mode="constant")
        method = "pca_fallback"
        note = f"UMAP skipped: {type(exc).__name__}: {exc}"

    coords = pd.DataFrame(
        {
            "sample_id": sample_id.astype(str),
            "deposit_type": labels.astype(str),
            "space_name": space_name,
            "method": method,
            "dim1": coords_arr[:, 0],
            "dim2": coords_arr[:, 1],
            "dim3": coords_arr[:, 2],
        }
    )
    save_dataframe(coords, OUTPUT_DIR / "figures" / f"{space_name}_3d_coordinates.csv")
    _plot_static_3d(coords, space_name, method)
    html_note = _write_html(coords, space_name, method)
    return {"visualization_method": method, "visualization_note": note, "html_note": html_note}
