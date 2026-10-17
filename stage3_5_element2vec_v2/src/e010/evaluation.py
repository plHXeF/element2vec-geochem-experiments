"""Shared grouped cohorts and portable embedding/token/latent evaluation artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from contextlib import closing

import numpy as np
import pandas as pd
import torch

from .schema import FIELDS, FAMILIES, GROUPS, LABELS, EXACT, INDEX, hidden_closure
from .prepare import coordinates, save_json, digest_file
from .training import load_model, build, seed_all
from .batching import make_batch, to_device

SHARED34 = "SiO2 TiO2 Al2O3 Fe2O3 FeO MgO CaO Na2O K2O P2O5 MnO LOI CO2 Rb Sr Ba Zr Nb Y V Cr Ni Cu Zn La Ce Nd Th U Pb Sc Co Ga Yb".split()


def hash_int(value):
    return int.from_bytes(hashlib.sha256(str(value).encode()).digest()[:8], "little")


def make_cohort(root, size=12000, token_size=20000):
    root = Path(root)
    with closing(sqlite3.connect(root / "metadata.sqlite")) as c:
        meta = pd.read_sql_query("SELECT * FROM samples ORDER BY row_index", c)
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[max(a,b)] = min(a,b)
    refs = [f"r:{int(r)}" if pd.notna(r) else f"s:{int(s)}" for r, s in zip(meta.canonical_ref_id, meta.sample_id)]
    for ref, dup in zip(refs, meta.dup_group_id):
        if pd.notna(dup): union(ref, "d:"+str(dup))
    components = [find(x) for x in refs]
    split_cache = {x: (0 if (v := hash_int("split42:"+x) % 10000) < 7000 else 1 if v < 8500 else 2) for x in set(components)}
    meta["component"] = components
    meta["split"] = [split_cache[x] for x in components]
    meta["priority"] = [hash_int("cohort42:"+str(x)) for x in meta.sample_id]
    top_sources = set(meta.data_source.value_counts().head(10).index)
    src = meta.data_source.map(lambda x: x if x in top_sources else "other")
    def material(x):
        s = str(x).lower()
        if x is None or s in ("nan", "unknown", "none"): return "unknown"
        if "sediment" in s: return "sediment"
        if "ore" == s or "vein" == s: return "ore_vein"
        if any(t in s for t in ("rock", "outcrop", "core", "drill", "xenolith")): return "rock_core"
        return "other"
    mat = meta.material.map(material)
    panel = pd.cut(meta.observed_count, [0, 1, 4, 9, 19, 39, np.inf], labels=False)
    meta["stratum"] = src.astype(str)+"/"+mat+"/"+panel.astype(str)
    selected = []
    groups = list(meta.groupby(["split", "stratum"], sort=True))
    total = sum(np.sqrt(len(g)) for _, g in groups)
    for _, group in groups:
        take = min(len(group), max(1, int(np.ceil(size*np.sqrt(len(group))/total))))
        selected.append(group.nsmallest(take, "priority"))
    pool = pd.concat(selected).nsmallest(min(size, len(meta)), "priority").sort_values("row_index")
    cols = ["row_index", "sample_id", "component", "split", "stratum"]
    pool[cols].to_csv(root / "cohort.csv", index=False)
    states = np.load(root/"states.npy", mmap_mode="r")
    # Frozen element-specific supplement; selection uses availability and IDs,
    # never concentration magnitude or downstream labels.
    rare = []
    for j in range(len(FIELDS)):
        exact_rows = np.flatnonzero(np.isin(states[:, j], EXACT))
        if not len(exact_rows): continue
        for split_id in range(3):
            eligible = meta.iloc[exact_rows]
            eligible = eligible[eligible.split == split_id].nsmallest(96, "priority")
            for r in eligible[cols].itertuples(index=False, name=None):
                rare.append((*r, j))
    rare_frame = pd.DataFrame(rare, columns=cols+["tag_index"])
    rare_frame.to_csv(root/"rare_field_cohort.csv", index=False)
    candidates = []
    rows = pool.row_index.to_numpy(np.int64)
    per = max(1, token_size//len(FIELDS))
    for j in range(len(FIELDS)):
        available = rows[states[rows, j] != 0]
        available = sorted(available, key=lambda r: hash_int(f"token42:{r}:{j}"))[:per]
        candidates.extend((int(r), j) for r in available)
    pd.DataFrame(candidates, columns=["row_index", "tag_index"]).to_csv(root/"token_cohort.csv", index=False)
    states._mmap.close()
    save_json(root/"cohort_contract.json", {"seed": 42, "rows": len(pool), "token_rows": len(candidates),
              "split_counts": {str(k): int(v) for k, v in pool.split.value_counts().items()},
              "max_population_component": int(pd.Series(components).value_counts().max()),
              "rare_field_rows": len(rare_frame), "rare_per_field_per_split_cap": 96,
              "split_rule": "canonical-reference and duplicate connected components; hash 70/15/15",
              "selection": "sqrt-frequency source/material/panel strata; frozen IDs",
              "boundary": "probe fitting split only; pretraining sees all records and rock labels"})


def extract(checkpoint, data, root, device="cpu", batch_size=32):
    root = Path(root); out = root/"representations"
    out.mkdir(exist_ok=False)
    model, ck = load_model(checkpoint, device)
    if ck["prepared_fingerprint"] != data.fingerprint:
        raise ValueError("Checkpoint and evaluation data differ")
    model.eval()
    pool = pd.read_csv(data.root/"cohort.csv"); indices = pool.row_index.to_numpy(np.int64)
    token_pool = pd.read_csv(data.root/"token_cohort.csv")
    token_lookup = {(int(r), int(t)): i for i, (r, t) in enumerate(token_pool.itertuples(index=False, name=None))}
    latent = {k: np.lib.format.open_memmap(out/(k+".npy"), mode="w+", dtype="float32", shape=(len(indices), model.d_model)) for k in ("mean", "valid_mean", "cls")}
    token_arrays = {}
    keys = ["static", "projected", "rotated_q", "rotated_k"] + [f"contextual_{i}" for i in sorted({1, min(6,len(model.layers)), len(model.layers)})]
    for key in keys:
        width = model.embedding_dim if key == "static" else model.d_model
        token_arrays[key] = np.lib.format.open_memmap(out/("token_"+key+".npy"), mode="w+", dtype="float32", shape=(len(token_pool), width))
    seen = np.zeros(len(token_pool), bool)
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            rows = indices[start:start+batch_size]
            batch = to_device(make_batch(data, rows, model, training=False, chemistry_only=True), device)
            encoded = model.encode(batch, capture=True)
            for name, a in latent.items(): a[start:start+len(rows)] = encoded[name].cpu().numpy()
            tags = batch["target_tag"].cpu().numpy(); mask = batch["mask"].cpu().numpy()
            for b, row in enumerate(rows):
                for j in np.flatnonzero(mask[b]):
                    target = token_lookup.get((int(row), int(tags[b, j])))
                    if target is not None:
                        for name, a in token_arrays.items(): a[target] = encoded[name][b, j].cpu().numpy()
                        seen[target] = True
    if not seen.all(): raise AssertionError("Token cohort incomplete")
    for a in [*latent.values(), *token_arrays.values()]: a.flush()
    np.save(out/"static_numeric.npy", model.numeric_tag_embedding.weight.detach().cpu().numpy()[:len(FIELDS)])
    np.save(out/"static_labels.npy", model.pure_label_embedding.weight.detach().cpu().numpy())
    seed_all(ck["config"]["training"]["seed"])
    random_model = build(ck["config"], data).to(device).eval()
    random_latent = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            b = to_device(make_batch(data, indices[start:start+batch_size], random_model, training=False, chemistry_only=True), device)
            random_latent.append(random_model.encode(b)["mean"].cpu().numpy())
    np.save(out/"random_mean.npy", np.concatenate(random_latent))
    np.save(out/"random_static_numeric.npy", random_model.numeric_tag_embedding.weight.detach().cpu().numpy()[:len(FIELDS)])
    shared_ids = [INDEX[t] for t in SHARED34]
    hide = [j for j in range(len(FIELDS)) if j not in shared_ids]
    # Derived values are excluded from the historical 34-field comparison:
    # absence of a directly reported field must remain absent for every model.
    from .scientific_audit import subset
    shared_data = subset(data, indices)
    shared_data.states[np.isin(shared_data.states, [3, 5])] = 0
    for name, encoder in (("mean_shared34", model), ("random_mean_shared34", random_model)):
        result = []
        with torch.inference_mode():
            for start in range(0, len(indices), batch_size):
                b = make_batch(shared_data, np.arange(start,min(start+batch_size,len(indices))), encoder,
                               training=False, chemistry_only=True, hide=hide)
                result.append(encoder.encode(to_device(b, device))["mean"].cpu().numpy())
        np.save(out/(name+".npy"), np.concatenate(result))
    raw = np.array(data.values[indices]); st = np.array(data.states[indices])
    z = coordinates(raw, st, data.stats); exact = np.isin(st, EXACT)
    np.save(out/"chemistry.npy", np.concatenate((np.where(exact,z,0), exact.astype(float)), axis=1).astype("float32"))
    np.save(out/"missingness.npy", (st != 0).astype("float32"))
    shared_exact = np.isin(st[:, shared_ids], [1, 2])
    np.save(out/"chemistry_shared34.npy", np.concatenate((np.where(shared_exact,z[:,shared_ids],0),shared_exact),axis=1).astype("float32"))
    np.save(out/"missingness_shared34.npy", (shared_data.states[:,shared_ids]!=0).astype("float32"))
    pool.to_csv(out/"cohort.csv", index=False); token_pool.to_csv(out/"token_cohort.csv", index=False)
    save_json(out/"manifest.json", {"checkpoint_sha256": digest_file(checkpoint), "prepared_fingerprint": data.fingerprint,
              "checkpoint_path": str(Path(checkpoint).resolve()),
              "mode": "chemistry_only", "rows": len(indices), "token_rows": len(token_pool),
              "levels": keys, "embedding_dim": model.embedding_dim, "token_dim": model.d_model,
              "shared34": SHARED34, "shared34_derived_excluded": True,
              "untrained_tags": [f for f, active in zip(FIELDS, ck["active_tags"]) if not active],
              "files": {p.name: digest_file(p) for p in out.iterdir() if p.is_file()}})


def grouped_bootstrap(values, groups, seed=42, repeats=500):
    frame = pd.DataFrame({"v": values, "g": groups}).groupby("g").v.agg(["sum", "count"])
    if len(frame) < 2: return [None, None]
    rng = np.random.default_rng(seed); sums = frame["sum"].to_numpy(); counts = frame["count"].to_numpy()
    result = []
    for _ in range(repeats):
        idx = rng.integers(len(frame), size=len(frame)); result.append(sums[idx].sum()/counts[idx].sum())
    return list(map(float, np.quantile(result, [.025, .975])))


def neighbors_without_self(features, k):
    from sklearn.neighbors import NearestNeighbors
    if not 0 < k < len(features): raise ValueError("k must be positive and smaller than the pool")
    candidates = NearestNeighbors(n_neighbors=k+1).fit(features).kneighbors(features, return_distance=False)
    return np.stack([row[row != i][:k] for i, row in enumerate(candidates)])


def evaluate(data, root, device="cpu"):
    """Compact fixed-representation audit. Never label this held-out pretraining."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    root = Path(root); rep = root/"representations"
    manifest = json.loads((rep/"manifest.json").read_text())
    for name, sha in manifest["files"].items():
        if digest_file(rep/name) != sha: raise ValueError("Representation file modified")
    pool = pd.read_csv(rep/"cohort.csv"); rows = pool.row_index.to_numpy(np.int64)
    with closing(sqlite3.connect(data.root/"metadata.sqlite")) as c:
        meta = pd.read_sql_query("SELECT row_index,data_source,material,canonical_ref_id,observed_count FROM samples ORDER BY row_index", c).iloc[rows].reset_index(drop=True)
    split = pool.split.to_numpy(); fit = split == 0; test = split == 2
    if fit.sum() < 3 or test.sum() < 3:
        raise ValueError("Insufficient independent probe groups; cannot substitute a random row split")
    metrics = []; basis_root = root/"evaluation"; basis_root.mkdir(exist_ok=False)
    for name in ("mean", "valid_mean", "cls", "random_mean", "chemistry", "missingness",
                 "mean_shared34", "random_mean_shared34", "chemistry_shared34", "missingness_shared34"):
        x = np.load(rep/(name+".npy"))
        if not np.isfinite(x).all(): raise FloatingPointError(name)
        spaces = {"FULL": x}
        pca = PCA(n_components=min(64, fit.sum()-1, x.shape[1]), svd_solver="full").fit(x[fit])
        np.savez(basis_root/(name+"_pca.npz"), mean=pca.mean_, components=pca.components_, variance=pca.explained_variance_)
        rank = int(np.sum(pca.explained_variance_ > max(1e-12, pca.explained_variance_[0]*1e-8)))
        for d in (2, 3, 32, 64):
            if d <= rank and d <= x.shape[1]:
                spaces[f"PCA{d}"] = pca.transform(x)[:, :d]
                for seed in range(10):
                    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(x.shape[1], d)))
                    spaces[f"RP{d}_{seed}"] = (x-pca.mean_) @ q
        for space, features in spaces.items():
            ytest = features[test]
            k = min(15, len(ytest)-1)
            neighbors = neighbors_without_self(ytest, k)
            for label, y in [(k, np.asarray(data.labels[rows, i])) for i, k in enumerate(LABELS)] + [("source", meta.data_source.fillna("unknown").to_numpy()), ("publication", meta.canonical_ref_id.fillna(-1).to_numpy()), ("material", meta.material.fillna("unknown").to_numpy()), ("panel_count", meta.observed_count.to_numpy())]:
                yt = y[test]; valid = yt != -1 if np.issubdtype(yt.dtype, np.number) and label != "panel_count" else np.ones(len(yt), bool)
                if label == "panel_count": yt = np.digitize(yt, [1, 4, 9, 19, 39])
                neighbor_known = valid[neighbors]
                counts = neighbor_known.sum(1)
                score = np.divide(((yt[neighbors] == yt[:, None]) & neighbor_known).sum(1), counts,
                                  out=np.zeros(len(yt), float), where=counts > 0)
                valid = valid & (counts > 0)
                if valid.any():
                    metrics.append({"representation": name, "space": space, "target": label, "metric": "knn15_purity",
                                    "value": float(score[valid].mean()), "n": int(valid.sum()),
                                    "cluster_ci": grouped_bootstrap(score[valid], pool.component.to_numpy()[test][valid]) if space == "FULL" else None})
            # Probe complexity is explicit; projection comparisons do not silently
            # refit a different hyperparameter budget for each model.
            if space != "FULL": continue
            scaler = StandardScaler().fit(features[fit]); xx = scaler.transform(features)
            for i, label in enumerate(LABELS):
                y = np.asarray(data.labels[rows, i]); train_ok = fit & (y >= 0); test_ok = test & (y >= 0)
                if len(np.unique(y[train_ok])) < 2 or not test_ok.any(): continue
                probe = LogisticRegression(C=1., max_iter=500, solver="lbfgs").fit(xx[train_ok], y[train_ok])
                pred = probe.predict(xx[test_ok])
                metrics.append({"representation": name, "space": space, "target": label, "metric": "linear_balanced_accuracy",
                                "value": float(balanced_accuracy_score(y[test_ok], pred)), "n": int(test_ok.sum()),
                                "probe_parameters": int(probe.coef_.size+probe.intercept_.size)})
    save_json(basis_root/"metrics.json", metrics)
    pd.DataFrame(metrics).to_csv(basis_root/"metrics.csv", index=False)
    save_json(basis_root/"limitations.json", {"pretraining_exposure": "all records and rock labels",
              "not_external_generalization": True, "no_weighted_overall_score": True,
              "PCA_fitted_on_probe_fit_only": True, "random_projections_per_dimension": 10,
              "bootstrap_repeats": 500, "native_rank_limited": True})
    from .scientific_audit import audit
    audit(data, root, manifest["checkpoint_path"], device=device)
    return metrics
