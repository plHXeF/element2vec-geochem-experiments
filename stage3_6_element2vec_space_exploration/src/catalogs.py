from __future__ import annotations

import numpy as np
import pandas as pd


# Approximate, analysis-only elemental properties. Ionic radii are typical
# Shannon-style six-coordinate values for the listed common valence where a
# conventional value exists. They are not universal constants; that limitation
# is explicitly carried into experiment reports.
_ELEMENTS = {
    "C": (6, 12.011, 14, 2, 2.55, 4, 0.16),
    "Na": (11, 22.990, 1, 3, 0.93, 1, 1.02),
    "Mg": (12, 24.305, 2, 3, 1.31, 2, 0.72),
    "Al": (13, 26.982, 13, 3, 1.61, 3, 0.535),
    "Si": (14, 28.085, 14, 3, 1.90, 4, 0.40),
    "P": (15, 30.974, 15, 3, 2.19, 5, 0.38),
    "K": (19, 39.098, 1, 4, 0.82, 1, 1.38),
    "Ca": (20, 40.078, 2, 4, 1.00, 2, 1.00),
    "Sc": (21, 44.956, 3, 4, 1.36, 3, 0.745),
    "Ti": (22, 47.867, 4, 4, 1.54, 4, 0.605),
    "V": (23, 50.942, 5, 4, 1.63, 3, 0.64),
    "Cr": (24, 51.996, 6, 4, 1.66, 3, 0.615),
    "Mn": (25, 54.938, 7, 4, 1.55, 2, 0.83),
    "Fe2": (26, 55.845, 8, 4, 1.83, 2, 0.78),
    "Fe3": (26, 55.845, 8, 4, 1.83, 3, 0.645),
    "Co": (27, 58.933, 9, 4, 1.88, 2, 0.745),
    "Ni": (28, 58.693, 10, 4, 1.91, 2, 0.69),
    "Cu": (29, 63.546, 11, 4, 1.90, 2, 0.73),
    "Zn": (30, 65.380, 12, 4, 1.65, 2, 0.74),
    "Ga": (31, 69.723, 13, 4, 1.81, 3, 0.62),
    "Rb": (37, 85.468, 1, 5, 0.82, 1, 1.52),
    "Sr": (38, 87.620, 2, 5, 0.95, 2, 1.18),
    "Y": (39, 88.906, 3, 5, 1.22, 3, 0.90),
    "Zr": (40, 91.224, 4, 5, 1.33, 4, 0.72),
    "Nb": (41, 92.906, 5, 5, 1.60, 5, 0.64),
    "Ba": (56, 137.327, 2, 6, 0.89, 2, 1.35),
    "La": (57, 138.905, 3, 6, 1.10, 3, 1.032),
    "Ce": (58, 140.116, 3, 6, 1.12, 3, 1.010),
    "Nd": (60, 144.242, 3, 6, 1.14, 3, 0.983),
    "Yb": (70, 173.045, 3, 6, 1.10, 3, 0.868),
    "Pb": (82, 207.200, 14, 6, 2.33, 2, 1.19),
    "Th": (90, 232.038, 3, 7, 1.30, 4, 0.94),
    "U": (92, 238.029, 3, 7, 1.38, 4, 1.00),
}


TAG_TO_ELEMENT = {
    "SiO2": "Si",
    "TiO2": "Ti",
    "Al2O3": "Al",
    "Fe2O3": "Fe3",
    "FeO": "Fe2",
    "MgO": "Mg",
    "CaO": "Ca",
    "Na2O": "Na",
    "K2O": "K",
    "P2O5": "P",
    "MnO": "Mn",
    "CO2": "C",
    "Rb_ppm": "Rb",
    "Sr_ppm": "Sr",
    "Ba_ppm": "Ba",
    "Zr_ppm": "Zr",
    "Nb_ppm": "Nb",
    "Y_ppm": "Y",
    "V_ppm": "V",
    "Cr_ppm": "Cr",
    "Ni_ppm": "Ni",
    "Cu_ppm": "Cu",
    "Zn_ppm": "Zn",
    "La_ppm": "La",
    "Ce_ppm": "Ce",
    "Nd_ppm": "Nd",
    "Th_ppm": "Th",
    "U_ppm": "U",
    "Pb_ppm": "Pb",
    "Sc_ppm": "Sc",
    "Co_ppm": "Co",
    "Ga_ppm": "Ga",
    "Yb_ppm": "Yb",
}


TAG_GROUPS = {
    "SiO2": ("major_framework", "lithophile", "incompatible"),
    "TiO2": ("major_transition", "lithophile", "compatible"),
    "Al2O3": ("major_framework", "lithophile", "incompatible"),
    "Fe2O3": ("major_transition", "siderophile", "compatible"),
    "FeO": ("major_transition", "siderophile", "compatible"),
    "MgO": ("major_mafic", "lithophile", "compatible"),
    "CaO": ("major_alkaline_earth", "lithophile", "compatible"),
    "Na2O": ("major_alkali", "lithophile", "incompatible"),
    "K2O": ("major_alkali", "lithophile", "incompatible"),
    "P2O5": ("major_accessory", "lithophile", "incompatible"),
    "MnO": ("major_transition", "lithophile", "compatible"),
    "LOI": ("volatile_proxy", "volatile", "not_applicable"),
    "CO2": ("volatile_proxy", "volatile", "not_applicable"),
    "Rb_ppm": ("LILE", "lithophile", "incompatible"),
    "Sr_ppm": ("LILE", "lithophile", "context_dependent"),
    "Ba_ppm": ("LILE", "lithophile", "incompatible"),
    "Zr_ppm": ("HFSE", "lithophile", "incompatible"),
    "Nb_ppm": ("HFSE", "lithophile", "incompatible"),
    "Y_ppm": ("HFSE_REE_like", "lithophile", "incompatible"),
    "V_ppm": ("transition", "lithophile", "compatible"),
    "Cr_ppm": ("transition", "lithophile", "compatible"),
    "Ni_ppm": ("transition", "siderophile", "compatible"),
    "Cu_ppm": ("base_metal", "chalcophile", "context_dependent"),
    "Zn_ppm": ("base_metal", "chalcophile", "context_dependent"),
    "La_ppm": ("LREE", "lithophile", "incompatible"),
    "Ce_ppm": ("LREE", "lithophile", "incompatible"),
    "Nd_ppm": ("LREE", "lithophile", "incompatible"),
    "Th_ppm": ("HFSE_radiogenic", "lithophile", "incompatible"),
    "U_ppm": ("HFSE_radiogenic", "lithophile", "incompatible"),
    "Pb_ppm": ("base_metal", "chalcophile", "incompatible"),
    "Sc_ppm": ("transition", "lithophile", "compatible"),
    "Co_ppm": ("transition", "siderophile", "compatible"),
    "Ga_ppm": ("post_transition", "lithophile", "incompatible"),
    "Yb_ppm": ("HREE", "lithophile", "incompatible"),
}


EXPECTED_ELEMENT_PAIRS = [
    ("Zr_ppm", "Hf_ppm", "Zr–Hf"),
    ("Nb_ppm", "Ta_ppm", "Nb–Ta"),
    ("Rb_ppm", "Cs_ppm", "Rb–Cs"),
    ("Sr_ppm", "Ba_ppm", "Sr–Ba"),
    ("Ni_ppm", "Co_ppm", "Ni–Co"),
    ("Mo_ppm", "W_ppm", "Mo–W"),
    ("La_ppm", "Ce_ppm", "La–Ce"),
    ("Ce_ppm", "Nd_ppm", "Ce–Nd"),
    ("Y_ppm", "Ho_ppm", "Y–Ho"),
    ("Th_ppm", "U_ppm", "Th–U"),
    ("Rb_ppm", "K2O", "Rb–K"),
    ("Sr_ppm", "CaO", "Sr–Ca"),
]


VOLCANIC_PLUTONIC_PAIRS = [
    ("subalkalic basalt", "subalkalic gabbro"),
    ("alkalic basalt", "alkalic gabbro"),
    ("andesite", "diorite"),
    ("basaltic andesite", "gabbroic diorite"),
    ("rhyolite", "granite"),
    ("dacite", "granodiorite"),
    ("trachyte", "syenite"),
    ("phonolite", "nepheline syenite"),
]


EVOLUTION_SEQUENCES = [
    ["subalkalic basalt", "basaltic andesite", "andesite", "dacite", "rhyolite"],
    ["subalkalic gabbro", "gabbroic diorite", "diorite", "granodiorite", "granite"],
]


METAMORPHIC_PAIRS = [
    ("quartz sandstone", "quartzite"),
    ("limestone", "marble"),
    ("shale", "slate"),
    ("shale", "phyllite"),
    ("shale", "schist"),
    ("subalkalic basalt", "metabasalt"),
]


def numeric_tag_catalog(tags: list[str]) -> pd.DataFrame:
    rows = []
    for tag in tags:
        element_key = TAG_TO_ELEMENT.get(tag)
        props = _ELEMENTS.get(element_key) if element_key else None
        primary, affinity, compatibility = TAG_GROUPS.get(tag, ("unclassified", "unclassified", "unclassified"))
        if props:
            atomic_number, atomic_weight, group, period, electronegativity, valence, ionic_radius = props
            ion_potential = valence / ionic_radius if ionic_radius else np.nan
            element = element_key.removesuffix("2").removesuffix("3")
        else:
            atomic_number = atomic_weight = group = period = electronegativity = valence = ionic_radius = ion_potential = np.nan
            element = ""
        rows.append(
            {
                "numeric_tag": tag,
                "element": element,
                "atomic_number": atomic_number,
                "atomic_weight": atomic_weight,
                "periodic_group": group,
                "period": period,
                "electronegativity_pauling": electronegativity,
                "common_valence": valence,
                "typical_ionic_radius_A": ionic_radius,
                "ionic_potential": ion_potential,
                "geochemical_group": primary,
                "goldschmidt_affinity": affinity,
                "compatibility": compatibility,
                "property_caveat": "Typical valence/radius approximation; coordination and oxidation state vary by mineral and process.",
            }
        )
    return pd.DataFrame(rows)
