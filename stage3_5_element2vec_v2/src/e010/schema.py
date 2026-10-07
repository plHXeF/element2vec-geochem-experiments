"""Frozen analyte identities, observation states and stoichiometric dependencies."""
from __future__ import annotations

import numpy as np

ELEMENTS = "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split()
MAJORS = "SiO2 TiO2 Al2O3 Cr2O3 Fe2O3 Fe2O3_tot FeO FeO_tot MgO CaO MnO NiO K2O Na2O SrO P2O5 H2O_plus H2O_minus H2O_tot CO2 SO3 BaO CaCO3 MgCO3 LOI".split()
TRACE_COLUMNS = "f cl br i h c n p s al as ag au b ba be bi ca cd ce co cr cs cu dy er eu fe ga gd ge hf hg ho in ir k la li lu mg mn mo na nd ni nb os pa pb pd pm pr pt rb re rh ru sb sc se si sm sn sr ta tb te th ti tl tm w v u y yb zn zr".split()
FIELDS = ELEMENTS + MAJORS
INDEX = {name: i for i, name in enumerate(FIELDS)}
RAW_COLUMNS = [x.lower() for x in MAJORS] + [x + "_ppm" for x in TRACE_COLUMNS]
RAW_TAGS = MAJORS + [x.capitalize() for x in TRACE_COLUMNS]
LABELS = ["rock_group", "rock_origin", "rock_type"]
MISSING, MEASURED, REPORTED_ZERO, DERIVED, LEFT_CENSORED, LOWER_BOUND, UNKNOWN, INVALID = range(8)
STATE_NAMES = ["missing", "measured", "reported_zero", "derived_equivalent", "left_censored", "lower_bound", "unknown_encoding", "invalid_range"]
EXACT = (MEASURED, REPORTED_ZERO, DERIVED)
SUPERVISED = EXACT + (LEFT_CENSORED, LOWER_BOUND)
ATOMIC_WEIGHTS = {"H": 1.008, "C": 12.011, "O": 15.999, "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06, "K": 39.098, "Ca": 40.078, "Ti": 47.867, "Cr": 51.996, "Mn": 54.938, "Fe": 55.845, "Ni": 58.693, "Sr": 87.62, "Ba": 137.33}
OXIDES = {"Si": ("SiO2", 1, 2), "Ti": ("TiO2", 1, 2), "Al": ("Al2O3", 2, 3), "Cr": ("Cr2O3", 2, 3), "Mg": ("MgO", 1, 1), "Ca": ("CaO", 1, 1), "Mn": ("MnO", 1, 1), "Ni": ("NiO", 1, 1), "K": ("K2O", 2, 1), "Na": ("Na2O", 2, 1), "Sr": ("SrO", 1, 1), "P": ("P2O5", 2, 5), "Ba": ("BaO", 1, 1)}


def oxide_factor(element, atoms, oxygen):
    mass = atoms * ATOMIC_WEIGHTS[element]
    return 10000.0 * mass / (mass + oxygen * ATOMIC_WEIGHTS["O"])


def upper_limit(tag):
    # Total iron expressed as a hypothetical oxide can legitimately exceed
    # 100 wt% in metallic/ore material. Bound the underlying elemental mass.
    if tag == "FeO_tot": return 1e6/oxide_factor("Fe",1,1)
    if tag == "Fe2O3_tot": return 1e6/oxide_factor("Fe",2,3)
    return 100.0 if tag in MAJORS else 1e6


def aliases():
    """Conservative disjoint components prevent both arithmetic and identity leaks."""
    groups = [[e, spec[0]] for e, spec in OXIDES.items() if e not in ("Ca", "Mg")]
    groups += [["Fe", "FeO", "Fe2O3", "FeO_tot", "Fe2O3_tot"],
               ["C", "CO2", "CaCO3", "MgCO3", "Ca", "CaO", "Mg", "MgO"],
               ["H", "H2O_plus", "H2O_minus", "H2O_tot"], ["S", "SO3"]]
    used = {x for g in groups for x in g}
    groups += [[x] for x in FIELDS if x not in used]
    result = np.empty(len(FIELDS), dtype=np.int64)
    for g, tags in enumerate(groups):
        for tag in tags:
            result[INDEX[tag]] = g
    return result


GROUPS = aliases()
FAMILIES = np.array([1] * len(ELEMENTS) + [0] * len(MAJORS), dtype=np.int64)


def dictionary():
    return {"version": 1, "elements": ELEMENTS, "fields": FIELDS, "raw_columns": RAW_COLUMNS,
            "states": STATE_NAMES, "families": FAMILIES.tolist(), "dependency_groups": GROUPS.tolist(),
            "atomic_weights": ATOMIC_WEIGHTS, "atomic_weights_source": "https://ciaaw.org/abridged-atomic-weights.htm",
            "element_source": "https://iupac.org/what-we-do/periodic-table-of-elements/",
            "oxide_rules": {k: list(v) for k, v in OXIDES.items()}, "censor_rules_default": "none; source-specific evidence required",
            "upper_limits": {tag: upper_limit(tag) for tag in FIELDS},
            "partial_species_policy": "H/C/S and single-valence Fe are lower bounds, not elemental totals"}


def convert_frame(frame, censor_rules=()):
    """Keep raw non-null cells, including invalid ones. Never impute missing rows.

    A rule must identify an exact canonical reference, column and evidence URL.
    Merely having data_source='usgs' is deliberately insufficient.
    """
    n = len(frame)
    values = np.full((n, len(FIELDS)), np.nan, dtype=np.float64)
    states = np.zeros(values.shape, dtype=np.uint8)
    for column, tag in zip(RAW_COLUMNS, RAW_TAGS):
        x = frame[column].to_numpy(dtype=float, na_value=np.nan)
        j = INDEX[tag]
        values[:, j] = x
        present = ~np.isnan(x)
        states[present, j] = MEASURED
        states[present & (x == 0), j] = REPORTED_ZERO
        if tag != "LOI":
            states[present & (x < 0), j] = UNKNOWN
        limit = upper_limit(tag)
        states[present & ((x > limit) | ~np.isfinite(x)), j] = INVALID
        if tag == "LOI":
            states[present & (x < -100), j] = INVALID
    for rule in censor_rules:
        if not rule.get("evidence_url") or not rule.get("canonical_ref_ids"):
            raise ValueError("Censor rules require exact canonical references and evidence")
        col = rule["column"]
        j = INDEX[RAW_TAGS[RAW_COLUMNS.index(col)]]
        if FIELDS[j] == "LOI":
            raise ValueError("Negative LOI is not a censor code")
        rows = frame["canonical_ref_id"].isin(rule["canonical_ref_ids"]).to_numpy(copy=True)
        rows &= states[:, j] == UNKNOWN
        if rule["encoding"] != "negative_detection_limit":
            raise ValueError("Unsupported censor encoding")
        rows &= -values[:, j] <= upper_limit(FIELDS[j])
        values[rows, j] *= -1
        states[rows, j] = LEFT_CENSORED

    def candidate(element, cols, factors, state=DERIVED, require_all=True):
        target = INDEX[element]
        js = [INDEX[x] for x in cols]
        good = np.isin(states[:, js], EXACT)
        supported = good.all(1) if require_all else good.any(1)
        use = (states[:, target] == MISSING) & supported
        amount = (np.where(good, values[:, js], 0) * np.array(factors)).sum(1)
        use &= np.isfinite(amount) & (amount >= 0) & (amount <= 1e6)
        values[use, target] = amount[use]
        states[use, target] = state

    for e, (oxide, count, oxygen) in OXIDES.items():
        candidate(e, [oxide], [oxide_factor(e, count, oxygen)])
    f1, f2 = oxide_factor("Fe", 1, 1), oxide_factor("Fe", 2, 3)
    candidate("Fe", ["FeO_tot"], [f1])
    candidate("Fe", ["Fe2O3_tot"], [f2])
    candidate("Fe", ["FeO", "Fe2O3"], [f1, f2])
    candidate("Fe", ["FeO", "Fe2O3"], [f1, f2], LOWER_BOUND, require_all=False)
    candidate("H", ["H2O_tot"], [oxide_factor("H", 2, 1)], LOWER_BOUND)
    candidate("H", ["H2O_plus", "H2O_minus"], [oxide_factor("H", 2, 1)] * 2, LOWER_BOUND, require_all=False)
    candidate("C", ["CO2"], [oxide_factor("C", 1, 2)], LOWER_BOUND)
    candidate("S", ["SO3"], [oxide_factor("S", 1, 3)], LOWER_BOUND)
    for e, carbonate in [("Ca", "CaCO3"), ("Mg", "MgCO3")]:
        f = 1e4 * ATOMIC_WEIGHTS[e] / (ATOMIC_WEIGHTS[e] + ATOMIC_WEIGHTS["C"] + 3 * ATOMIC_WEIGHTS["O"])
        candidate(e, [carbonate], [f], LOWER_BOUND)
    return values, states


def hidden_closure(tags):
    ids = [INDEX[t] if isinstance(t, str) else int(t) for t in tags]
    return np.flatnonzero(np.isin(GROUPS, GROUPS[ids]))
