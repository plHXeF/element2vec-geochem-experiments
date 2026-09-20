"""Versioned, sample-local coordinates; no fitted per-element scale for geo v1."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from .transforms import apply_value_transform, clean_numeric

GEO_VERSION = 'geochemical_v1'


def coordinate_spec(config: dict) -> dict:
    spec = dict(config['data']['value_transform'])
    spec.setdefault('name', 'v2_robust_z')
    return spec


def coordinate_fingerprint(config: dict) -> str:
    payload = {'coordinate': coordinate_spec(config), 'numeric_fields': config['data']['numeric_fields']}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_geo_config(config: dict) -> None:
    spec = coordinate_spec(config)
    if spec['name'] != GEO_VERSION:
        raise ValueError(f'Unsupported geochemical coordinate version: {spec["name"]}')
    fields = {f['name'] for f in config['data']['numeric_fields']}
    major, trace, auxiliary = spec['major'], spec['trace'], spec['auxiliary']
    groups = [set(major['oxide_panel']), set(trace['pm_ppm']), set(auxiliary['fields'])]
    if set.union(*groups) != fields or sum(map(len, groups)) != len(fields):
        raise ValueError('Every numeric field must belong to exactly one coordinate family')
    if len(major['oxide_panel']) != len(groups[0]):
        raise ValueError('Duplicate major oxide in closure panel')
    for f in config['data']['numeric_fields']:
        expected_unit = 'ppm' if f['name'] in groups[1] else 'wt_pct'
        if f['unit'] != expected_unit:
            raise ValueError(f'Coordinate unit mismatch for {f["name"]}')
    if major['coordinate'] != 'sqrt_fraction' or major['fallback'] != 'unclosed_fraction':
        raise ValueError('Geo v1 supports sqrt_fraction with explicit unclosed_fraction fallback')
    if not major['require_complete_panel']:
        raise ValueError('Geo v1 does not impute missing oxides into the closure denominator')
    low, high = major['sum_range_wt_pct']
    if not 0 < float(low) < float(high) or not np.isfinite([low, high]).all():
        raise ValueError('Invalid closure total range')
    for value in [major['family_scale'], trace['family_scale'], auxiliary['family_scale'],
                  major['epsilon_fraction'], trace['epsilon_ppm'], auxiliary['epsilon_fraction'],
                  *trace['pm_ppm'].values()]:
        if not np.isfinite(float(value)) or float(value) <= 0:
            raise ValueError('Scales, epsilon and PM reference values must be finite and positive')
    if not set(auxiliary['signed_fields']).issubset(groups[2]):
        raise ValueError('Signed auxiliary fields must belong to auxiliary family')
    if spec.get('clip_z') is not None:
        raise ValueError('Geo v1 does not use robust-z clipping')
    if not trace.get('source') or not major.get('source'):
        raise ValueError('Reference sources and closure-rule provenance must be recorded')


def transform_numeric_frame(frame: pd.DataFrame, config: dict, vocab) -> tuple[pd.DataFrame, dict]:
    """Return model coordinates in the original numeric column names and QC counts.

    Closure status is diagnostic metadata only; it never becomes a model token.
    All closure decisions are made before masking and reused by training/extraction.
    """
    spec = coordinate_spec(config)
    result = frame.copy()
    if spec['name'] == 'v2_robust_z':
        stats = {r['db_column']: r.to_dict() for _, r in vocab.transform_stats.iterrows()}
        for f in config['data']['numeric_fields']:
            col = f['db_column']
            result[col] = apply_value_transform(frame[col], stats[col], clip_z=spec.get('clip_z'))
        return result, {'coordinate_name': 'v2_robust_z', 'samples': len(frame)}
    validate_geo_config(config)
    columns = {f['name']: f['db_column'] for f in config['data']['numeric_fields']}
    major, trace, auxiliary = spec['major'], spec['trace'], spec['auxiliary']
    panel = [columns[t] for t in major['oxide_panel']]
    raw = frame[panel].apply(clean_numeric)
    complete = raw.notna().all(axis=1)
    nonnegative = raw.ge(0).all(axis=1)
    total = raw.sum(axis=1, min_count=len(panel))
    low, high = major['sum_range_wt_pct']
    eligible = complete & nonnegative & total.between(low, high)
    denominator = total.where(eligible, 100.)
    epsilon = float(major['epsilon_fraction'])
    for col in panel:
        fraction = raw[col].where(raw[col] >= 0) / denominator
        result[col] = float(major['family_scale']) * (np.sqrt(fraction + epsilon) - np.sqrt(epsilon))
    for tag, pm in trace['pm_ppm'].items():
        col = columns[tag]
        value = clean_numeric(frame[col])
        value = value.where(value > 0).clip(lower=float(trace['epsilon_ppm']))
        result[col] = float(trace['family_scale']) * np.log10(value / float(pm))
    for tag in auxiliary['fields']:
        col = columns[tag]
        value = clean_numeric(frame[col])
        if tag not in auxiliary['signed_fields']:
            value = value.where(value >= 0)
        eps = float(auxiliary['epsilon_fraction'])
        result[col] = float(auxiliary['family_scale']) * np.sign(value) * (np.sqrt(abs(value)/100. + eps)-np.sqrt(eps))
    numeric_cols = list(columns.values())
    if np.isinf(result[numeric_cols].to_numpy()).any():
        raise ValueError('Coordinate transform produced infinity')
    qc = {'coordinate_name': GEO_VERSION, 'samples': len(frame), 'closure_eligible': int(eligible.sum()),
          'closure_incomplete_panel': int((~complete).sum()),
          'closure_invalid_oxide': int((complete & ~nonnegative).sum()),
          'closure_total_outside_range': int((complete & nonnegative & ~eligible).sum()),
          'closure_fraction': float(eligible.mean()) if len(frame) else 0.,
          'coordinate_fingerprint': coordinate_fingerprint(config),
          'numeric_tokens': int(result[numeric_cols].notna().sum().sum()),
          'ranges': {t: {'min': float(result[c].min()) if result[c].notna().any() else None,
                         'max': float(result[c].max()) if result[c].notna().any() else None}
                     for t,c in columns.items()}}
    return result, qc
