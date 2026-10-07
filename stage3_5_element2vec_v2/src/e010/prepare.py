"""Read-only source audit and immutable, memory-mapped full-coverage preparation."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from .schema import (FIELDS, FAMILIES, RAW_COLUMNS, RAW_TAGS, INDEX, LABELS, EXACT,
                     STATE_NAMES, DERIVED, MISSING, dictionary, convert_frame)


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, data):
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(p)


def source_query():
    raw = [f'm."{c}" AS "{c}"' if not c.endswith("_ppm") else f't."{c}" AS "{c}"' for c in RAW_COLUMNS]
    present = " OR ".join(f'{"t" if c.endswith("_ppm") else "m"}."{c}" IS NOT NULL' for c in RAW_COLUMNS)
    return """SELECT s.sample_id, COALESCE(cr.canonical_ref_id,s.ref_id) AS canonical_ref_id,
    s.material, s.latitude, s.longitude, r.data_source, d.dup_group_id, d.is_kept,
    rg.rock_group, rg.rock_origin, co.rock_type, """ + ",".join(raw) + """
    FROM sample s LEFT JOIN major m ON s.major_id=m.major_id
    LEFT JOIN trace t ON s.trace_id=t.trace_id
    LEFT JOIN canonical_reference_map cr ON s.ref_id=cr.ref_id
    LEFT JOIN reference r ON COALESCE(cr.canonical_ref_id,s.ref_id)=r.ref_id
    LEFT JOIN dedup_decisions d ON d.sample_id=s.sample_id
    LEFT JOIN rockgroup rg ON s.rgroup_id=rg.rgroup_id
    LEFT JOIN computed co ON s.comp_id=co.comp_id WHERE (""" + present + ") ORDER BY s.sample_id"


def weighted_quantile(x, w, qs):
    order = np.argsort(x, kind="stable")
    x, w = np.asarray(x)[order], np.asarray(w, dtype=float)[order]
    positions = (np.cumsum(w) - w / 2) / w.sum()
    return np.interp(qs, positions, x)


def fit_stats(values, states, weights):
    stats = []
    for j in range(len(FIELDS)):
        ok = np.isin(states[:, j], EXACT)
        x, w = values[ok, j].astype(float), weights[ok].astype(float)
        positive = x > 0
        r = float(weighted_quantile(x[positive], w[positive], [.5])[0]) if positive.any() else 1.0
        stats.append({"field": FIELDS[j], "n_exact": int(ok.sum()), "r": r})
    # Shrink sparse positive reference scales in log space within the unit family.
    for family in (0, 1):
        ids = np.flatnonzero(FAMILIES == family)
        robust = [stats[j]["r"] for j in ids if stats[j]["n_exact"] >= 32]
        fallback = float(np.exp(np.median(np.log(robust)))) if robust else 1.0
        for j in ids:
            n = stats[j]["n_exact"]
            if n < 32:
                a = n / (n + 100.0)
                stats[j]["r"] = float(np.exp(a * np.log(stats[j]["r"]) + (1-a)*np.log(fallback)))
    for j, s in enumerate(stats):
        ok = np.isin(states[:, j], EXACT)
        if ok.any():
            y = np.arcsinh(values[ok, j].astype(float) / s["r"])
            q = weighted_quantile(y, weights[ok], [.25, .5, .75])
            s.update(m=float(q[1]), scale=float((q[2]-q[0])/1.349))
        else:
            s.update(m=0.0, scale=0.0)
    for family in (0, 1):
        ids = np.flatnonzero(FAMILIES == family)
        good = [stats[j] for j in ids if stats[j]["n_exact"] >= 32 and stats[j]["scale"] > 1e-8]
        fm = float(np.median([s["m"] for s in good])) if good else 0.0
        fs = float(np.median([s["scale"] for s in good])) if good else 1.0
        for j in ids:
            s = stats[j]; n = s["n_exact"]
            s["calibrated"] = n > 0
            s["shrunk"] = n < 32
            if n < 32:
                a = n / (n + 100.0)
                s["m"] = a*s["m"] + (1-a)*fm
                s["scale"] = a*s["scale"] + (1-a)*fs
            if s["scale"] <= 1e-8:
                s["scale"] = fs
                s["scale_fallback"] = True
    return stats


def coordinates(values, states, stats):
    r = np.array([s["r"] for s in stats]); m = np.array([s["m"] for s in stats]); scale = np.array([s["scale"] for s in stats])
    z = (np.arcsinh(np.nan_to_num(values.astype(float), nan=0., posinf=0., neginf=0.) / r) - m) / scale
    z[~np.isin(states, [1, 2, 3, 4, 5])] = 0.0
    return z.astype(np.float32)


def prepare_database(database, destination, expected_rows=None, chunk_size=8192, censor_rules=()):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    db = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    db.execute("PRAGMA query_only=ON")
    query = source_query()
    n = db.execute("SELECT count(*) FROM (" + query + ")").fetchone()[0]
    if expected_rows is not None and n != expected_rows:
        db.close()
        raise ValueError(f"Coverage changed: expected {expected_rows}, found {n}; revise the data protocol, never force a count")
    f = len(FIELDS)
    arrays = {name: np.lib.format.open_memmap(destination / (name + ".npy"), mode="w+", dtype=dtype, shape=shape)
              for name, dtype, shape in [("values", "float32", (n, f)), ("states", "uint8", (n, f)),
                                        ("sample_ids", "int64", (n,)), ("labels", "int32", (n, 3)),
                                        ("weights", "float32", (n,))]}
    metadata = sqlite3.connect(destination / "metadata.sqlite")
    counts = np.zeros((f, len(STATE_NAMES)), dtype=np.int64)
    sources, materials, duplicates = Counter(), Counter(), Counter()
    label_sets = [set() for _ in LABELS]
    observed_total = 0; offset = 0; last_id = -1; observed_hist = Counter()
    for frame in pd.read_sql_query(query, db, chunksize=chunk_size):
        stop = offset + len(frame)
        ids = frame.sample_id.to_numpy(np.int64)
        if (np.diff(ids) <= 0).any() or ids[0] <= last_id:
            raise ValueError("Source joins duplicated or reordered sample IDs")
        last_id = int(ids[-1])
        v, st = convert_frame(frame, censor_rules)
        arrays["values"][offset:stop], arrays["states"][offset:stop] = v, st
        arrays["sample_ids"][offset:stop] = ids
        obs = frame[RAW_COLUMNS].notna().sum(axis=1).to_numpy()
        observed_total += int(obs.sum()); observed_hist.update(map(int, obs))
        for j in range(f):
            counts[j] += np.bincount(st[:, j], minlength=len(STATE_NAMES))
        meta = frame.drop(columns=RAW_COLUMNS).copy()
        for field, collection in zip(LABELS, label_sets):
            meta[field] = meta[field].map(lambda x: str(x).strip() if pd.notna(x) and str(x).strip() else None)
            collection.update(meta[field].dropna())
        meta.insert(0, "row_index", np.arange(offset, stop))
        meta["observed_count"] = obs
        meta["exact_count"] = np.isin(st, EXACT).sum(axis=1)
        meta.to_sql("samples", metadata, if_exists="append", index=False)
        duplicates.update(meta.dup_group_id.dropna().astype(str))
        sources.update(meta.data_source.fillna("unknown")); materials.update(meta.material.fillna("unknown"))
        offset = stop
        if offset % (chunk_size * 16) == 0:
            print(f"prepare: {offset}/{n} rows", flush=True)
    db.close()
    vocab = {k: sorted(v) for k, v in zip(LABELS, label_sets)}
    maps = {k: {v: i for i, v in enumerate(vocab[k])} for k in LABELS}
    for frame in pd.read_sql_query("SELECT row_index,dup_group_id,rock_group,rock_origin,rock_type FROM samples ORDER BY row_index", metadata, chunksize=chunk_size):
        ids = frame.row_index.to_numpy()
        arrays["weights"][ids] = [1.0/duplicates[str(g)] if pd.notna(g) else 1.0 for g in frame.dup_group_id]
        for j, field in enumerate(LABELS):
            arrays["labels"][ids, j] = [maps[field].get(x, -1) for x in frame[field]]
    metadata.execute("CREATE UNIQUE INDEX sample_index ON samples(sample_id)")
    metadata.commit(); metadata.close()
    for a in arrays.values(): a.flush()
    print("prepare: fitting shared robust coordinates", flush=True)
    stats = fit_stats(arrays["values"], arrays["states"], arrays["weights"])
    coverage = {"rows": n, "original_observations": observed_total, "derived_exact": int(counts[:, DERIVED].sum()),
                "lower_bound_observations": int(counts[:, 5].sum()), "by_field": {tag: dict(zip(STATE_NAMES, map(int, counts[j]))) for j, tag in enumerate(FIELDS)},
                "raw_panel_histogram": dict(sorted(observed_hist.items())), "sources": dict(sources), "materials": dict(materials),
                "duplicate_groups": len(duplicates), "duplicate_group_max": max(duplicates.values(), default=1),
                "training_rows": n, "no_quality_row_filter": True}
    save_json(destination / "dictionary.json", dictionary())
    save_json(destination / "labels.json", vocab)
    save_json(destination / "statistics.json", stats)
    # A bounded analytic mapping is strictly increasing in real arithmetic;
    # float32 can merge nearby concentrations. Report actual support resolution.
    resolution = []
    for j, stat in enumerate(stats):
        x = np.unique(arrays["values"][np.isin(arrays["states"][:, j], EXACT), j])
        if not len(x):
            resolution.append({"field": FIELDS[j], "unique_concentrations": 0}); continue
        z64 = (np.arcsinh(x.astype(float)/stat["r"])-stat["m"])/stat["scale"]
        angle64 = np.arctan(z64/2.)
        angle32 = np.arctan(z64.astype(np.float32)/np.float32(2.))
        resolution.append({"field": FIELDS[j], "unique_concentrations": len(x),
                           "unique_float32_angles": len(np.unique(angle32)),
                           "merged_adjacent_pairs": int((np.diff(angle32)==0).sum()),
                           "max_absolute_angle_error": float(np.max(abs(angle64-angle32))),
                           "angle_min": float(angle32.min()), "angle_max": float(angle32.max()),
                           "nondecreasing": bool(np.all(np.diff(angle32)>=0))})
    save_json(destination / "angle_resolution.json", resolution)
    save_json(destination / "coverage.json", coverage)
    from .evaluation import make_cohort
    make_cohort(destination)
    manifest = {"schema_version": 1, "rows": n, "source_sha256": digest_file(database), "query": query,
                "censor_rules": list(censor_rules), "files": {p.name: digest_file(p) for p in sorted(destination.iterdir()) if p.is_file()}}
    save_json(destination / "manifest.json", manifest)
    return coverage


class PreparedData:
    def __init__(self, root, rehash=False):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["schema_version"] != 1:
            raise ValueError("Unsupported prepared-data version")
        if json.loads((self.root / "dictionary.json").read_text()) != dictionary():
            raise ValueError("Prepared dictionary differs from the implementation")
        if rehash:
            for name, expected in self.manifest["files"].items():
                if digest_file(self.root / name) != expected:
                    raise ValueError(f"Prepared input modified: {name}")
        for name in ("values", "states", "sample_ids", "labels", "weights"):
            setattr(self, name, np.load(self.root / (name + ".npy"), mmap_mode="r"))
        self.stats = json.loads((self.root / "statistics.json").read_text())
        self.label_vocab = json.loads((self.root / "labels.json").read_text())
        self.coverage = json.loads((self.root / "coverage.json").read_text())
        self.fingerprint = digest_file(self.root / "manifest.json")
        if len(self.values) != self.manifest["rows"] or self.values.shape[1] != len(FIELDS):
            raise ValueError("Prepared shape mismatch")

    def __len__(self):
        return len(self.values)
