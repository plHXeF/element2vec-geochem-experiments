from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls


OXIDES = ["sio2", "tio2", "al2o3", "fe2o3", "feo", "mgo", "cao", "na2o", "k2o", "p2o5"]
MOLECULAR_WEIGHTS = {
    "SiO2": 60.0843,
    "TiO2": 79.866,
    "Al2O3": 101.961,
    "Fe2O3": 159.688,
    "FeO": 71.844,
    "MgO": 40.304,
    "CaO": 56.077,
    "Na2O": 61.979,
    "K2O": 94.196,
    "P2O5": 141.944,
}


# Mineral formula decompositions into oxide moles. The columns are normalized
# to oxide weight fractions before NNLS. This is deliberately named CIPW-like:
# unlike a strict CIPW allocation sequence it does not enforce mutual exclusion
# of Q/Ne or Hy/Ol and is used only as a broad normative-mineral proxy.
MINERAL_STOICH = {
    "Q": {"SiO2": 1.0},
    "Or": {"K2O": 0.5, "Al2O3": 0.5, "SiO2": 3.0},
    "Ab": {"Na2O": 0.5, "Al2O3": 0.5, "SiO2": 3.0},
    "An": {"CaO": 1.0, "Al2O3": 1.0, "SiO2": 2.0},
    "DiMg": {"CaO": 1.0, "MgO": 1.0, "SiO2": 2.0},
    "DiFe": {"CaO": 1.0, "FeO": 1.0, "SiO2": 2.0},
    "HyMg": {"MgO": 1.0, "SiO2": 1.0},
    "HyFe": {"FeO": 1.0, "SiO2": 1.0},
    "OlMg": {"MgO": 2.0, "SiO2": 1.0},
    "OlFe": {"FeO": 2.0, "SiO2": 1.0},
    "Ne": {"Na2O": 0.5, "Al2O3": 0.5, "SiO2": 1.0},
    "Mt": {"FeO": 1.0, "Fe2O3": 1.0},
    "Il": {"FeO": 1.0, "TiO2": 1.0},
    "Ap": {"CaO": 3.0, "P2O5": 1.0},
}


def mineral_oxide_matrix() -> tuple[np.ndarray, list[str]]:
    names = list(MINERAL_STOICH)
    oxide_names = [name.upper() if name != "p2o5" else "P2O5" for name in OXIDES]
    canonical = {
        "SIO2": "SiO2", "TIO2": "TiO2", "AL2O3": "Al2O3", "FE2O3": "Fe2O3",
        "FEO": "FeO", "MGO": "MgO", "CAO": "CaO", "NA2O": "Na2O", "K2O": "K2O", "P2O5": "P2O5",
    }
    matrix = np.zeros((len(OXIDES), len(names)), dtype=float)
    for j, mineral in enumerate(names):
        contributions = []
        for i, oxide_upper in enumerate(oxide_names):
            oxide = canonical[oxide_upper]
            moles = MINERAL_STOICH[mineral].get(oxide, 0.0)
            weight = moles * MOLECULAR_WEIGHTS[oxide]
            matrix[i, j] = weight
            contributions.append(weight)
        total = sum(contributions)
        if total > 0:
            matrix[:, j] /= total
    return matrix, names


def cipw_like_nnls(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix, names = mineral_oxide_matrix()
    rows, qc = [], []
    for _, row in frame.iterrows():
        oxide = row[OXIDES].to_numpy(dtype=float)
        valid = np.isfinite(oxide)
        if valid.sum() < 8:
            rows.append({name: np.nan for name in names})
            qc.append({"reconstruction_rmse": np.nan, "oxide_sum": np.nansum(oxide), "valid_oxides": int(valid.sum())})
            continue
        coefficients, residual = nnls(matrix[valid], oxide[valid])
        reconstructed = matrix[valid] @ coefficients
        rmse = float(np.sqrt(np.mean((reconstructed - oxide[valid]) ** 2)))
        rows.append(dict(zip(names, coefficients)))
        qc.append({"reconstruction_rmse": rmse, "oxide_sum": float(np.nansum(oxide)), "valid_oxides": int(valid.sum()), "nnls_residual": float(residual)})
    return pd.DataFrame(rows, index=frame.index), pd.DataFrame(qc, index=frame.index)


def ideal_mineral_mixtures() -> pd.DataFrame:
    matrix, names = mineral_oxide_matrix()
    mixtures = {
        "quartz_feldspar_rich": {"Q": 30, "Or": 25, "Ab": 30, "An": 15},
        "plagioclase_pyroxene_rich": {"An": 35, "DiMg": 20, "DiFe": 10, "HyMg": 20, "HyFe": 10, "Il": 5},
        "olivine_rich": {"OlMg": 55, "OlFe": 10, "DiMg": 20, "HyMg": 10, "Il": 5},
        "nepheline_alkaline": {"Ne": 25, "Or": 25, "Ab": 15, "An": 10, "DiMg": 15, "OlMg": 10},
    }
    rows = []
    for label, mixture in mixtures.items():
        coefficients = np.asarray([mixture.get(name, 0.0) for name in names], dtype=float)
        oxide = matrix @ coefficients
        row = {name: value for name, value in zip(OXIDES, oxide)}
        row["ideal_mixture"] = label
        rows.append(row)
    return pd.DataFrame(rows)
