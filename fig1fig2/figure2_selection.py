"""Single source of truth for the main and supplementary orbit rows."""

from __future__ import annotations


# Each entry is (cache identifier, displayed identifier).  Historical cache
# identifiers P1/P6 are displayed throughout the paper as I2/II5.  The main
# Fig. 2 keeps only regions I and III; the complete appendix atlas restores
# regions II, IV, and V.
REGION_I_ROW = (
    "region_i",
    (("V1", "I1"), ("V2", "I2"), ("V3", "I3"),
     ("V4", "I4"), ("V5", "I5")),
)
REGION_II_ROW = (
    "region_ii",
    (("M2A", "II1"), ("M2L", "II2"), ("M2C", "II3"),
     ("M2R", "II4"), ("M2B", "II5")),
)
REGION_III_ROW = (
    "region_iii",
    (("M3A", "III1"), ("M3L", "III2"), ("L3", "III3"),
     ("M3R", "III4"), ("A9", "III5")),
)
REGION_IV_ROW = (
    "region_iv",
    (("L1", "IV1"), ("L2", "IV2"), ("L3", "IV3"),
     ("L4", "IV4"), ("L5", "IV5")),
)

FIG2_ROWS = (
    REGION_I_ROW,
    REGION_III_ROW,
)

CORE_ATLAS_ROWS = (
    REGION_I_ROW,
    REGION_II_ROW,
    REGION_III_ROW,
    REGION_IV_ROW,
)

REGION_STYLE = {
    "region_i": {"label": "Region I", "color": "#0072B2"},
    "region_ii": {"label": "Region II", "color": "#CC79A7"},
    "region_iii": {"label": "Region III", "color": "#E69F00"},
    "region_iv": {"label": "Region IV", "color": "#009E73"},
    "region_v": {"label": "Region V", "color": "#56B4E9"},
}
RESONANCE_COLOR = "#009E73"


# Coordinates introduced by the strict visual searches.  The remaining
# representatives are selected from the generated landscape and retain their
# historical cache identifiers.
MODE_POINT_COORDINATES = {
    "M2A": (
        -1.14,
        1.36,
        "region-II far-left clean periodic representative from a strict 36-point atlas",
    ),
    "M2L": (
        -0.84,
        1.12,
        "region-II left intermediate from a strict 36-point atlas",
    ),
    "M2C": (
        -0.45,
        0.90,
        "direct region-II representative on the bare resonance line",
    ),
    "M2R": (
        -0.16,
        0.90,
        "region-II clean periodic intermediate from a strict 49-point atlas",
    ),
    "M2B": (
        -0.11,
        1.95,
        "confined ED--TDVP comparison point II5",
    ),
    "M3A": (
        -1.28,
        1.86,
        "region-III far-left representative moved up/left in a strict 72-point atlas",
    ),
    "M3L": (
        -0.88,
        1.46,
        "exact parameter-space reflection of III4 about III3 from a strict 99-point atlas",
    ),
    "M3R": (
        -0.62,
        1.54,
        "region-III clean left-biased intermediate from a strict 30-point atlas",
    ),
}


def flattened_rows() -> tuple[tuple[str, str, str, str], ...]:
    """Return the region-I/III representatives marked in main Fig. 1."""

    rows = []
    for region, point_specs in FIG2_ROWS:
        for column_index, (point_id, display_id) in enumerate(point_specs):
            color = (
                RESONANCE_COLOR
                if column_index == 2 and region in ("region_ii", "region_iii")
                else REGION_STYLE[region]["color"]
            )
            rows.append((region, point_id, display_id, color))
    return tuple(rows)
