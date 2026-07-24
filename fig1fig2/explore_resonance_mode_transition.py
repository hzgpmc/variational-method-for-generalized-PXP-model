#!/usr/bin/env python3
"""Geometry-aware TDVP sampling of low-leakage confinement structures.

This supplementary, manuscript-independent calculation answers three
questions suggested by Fig. 1:

1. How do defect-site trajectories change along the low-leakage valley
   containing P1 and P2?
2. How do they change along the pronounced low-leakage arc through P4?
3. Does a second low-leakage arc through L1 exhibit the same finite-time
   reorganization of defect-site Bloch trajectories?

No numerical dataset is required.  If the leakage background is absent, the
script generates its 301-by-201 map directly from ``tdvpfun.eom`` with
resumable row checkpoints.  The 27 unique physical coordinates are locked
after a
leakage-and-spacing audit, so regenerating the background cannot silently move
the strict diagnostic points.  Every marked
point is reintegrated with exactly the strict TDVP protocol used by the current
Fig. 1/Fig. 2 reproducer: spin 1/2, K=100, T=10, pole bias and initial phase
1e-3, DOP853, rtol=1e-9, atol=1e-11, and max_step=0.02.  The script writes a
compressed trajectory cache, a quantitative JSON summary, and two
publication-standard PDF/PNG supplementary figures.  It never edits the TeX
manuscript.

Run in the QuSpin environment from the repository root:

    conda run -n quspin python fig1fig2/explore_resonance_mode_transition.py

Use ``--force`` to regenerate the numerical cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUTPUT_DIR = HERE / "output"
STYLE_CANDIDATES = (
    HERE / "hzg-paper.mplstyle",
    HERE.parent / "scripts" / "hzg-paper.mplstyle",
)
LEAKAGE_CACHE = DATA_DIR / "fig1_average_leakage.npz"
DEFAULT_CACHE = DATA_DIR / "supp_resonance_dense_sampling.npz"
DEFAULT_SUMMARY = OUTPUT_DIR / "supp_resonance_dense_sampling.json"
DEFAULT_PDF = OUTPUT_DIR / "supp_resonance_mode_transition.pdf"
DEFAULT_PNG = OUTPUT_DIR / "supp_resonance_mode_transition.png"
DEFAULT_PORTRAIT_PDF = OUTPUT_DIR / "supp_low_leakage_portrait_grid.pdf"
DEFAULT_PORTRAIT_PNG = OUTPUT_DIR / "supp_low_leakage_portrait_grid.png"


def resolve_style_file() -> Path:
    """Return the local style override or the repository-wide HZG style."""

    for candidate in STYLE_CANDIDATES:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(path) for path in STYLE_CANDIDATES)
    raise FileNotFoundError(f"missing HZG Matplotlib style; searched: {searched}")


os.environ.setdefault("MPLCONFIGDIR", str(DATA_DIR / "mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import tdvpfun


SCHEMA_VERSION = 6
BACKGROUND_SCHEMA_VERSION = 1

# This block is the single source of truth for all new integrations.
T_MAX = 10.0
SAMPLE_COUNT = 1001
TDVP_PERIOD = 100
POLE_BIAS = 1.0e-3
INITIAL_PHI = 1.0e-3
METHOD = "DOP853"
RTOL = 1.0e-9
ATOL = 1.0e-11
MAX_STEP = 0.02

# Pure-code leakage-background protocol.  The default 301-by-201 grid and
# 250 left-endpoint samples reproduce the historical Fig. 1 landscape.  The
# lower-tolerance scan is used only as a selection/visualization background;
# all 27 marked trajectories are independently reintegrated with the strict
# protocol above.
BACKGROUND_MU_RANGE = (-1.5, 1.5)
BACKGROUND_CHI_RANGE = (0.0, 2.0)
BACKGROUND_MU_COUNT = 301
BACKGROUND_CHI_COUNT = 201
BACKGROUND_DT = 0.04
BACKGROUND_METHOD = "RK45"
BACKGROUND_RTOL = 1.0e-3
BACKGROUND_ATOL = 1.0e-5
BACKGROUND_MAX_STEP = np.inf
GENERATED_BACKGROUND_SOURCE = (
    "generated_by_explore_resonance_mode_transition.py"
)
REPRODUCER_BACKGROUND_SOURCE = "generated_by_reproduce_fig1_fig2.py"
REPRODUCER_BACKGROUND_SCHEMA_VERSION = 2
LEGACY_BACKGROUND_SOURCE = "tdvp_L100_sps2_t30.0.pkl"
GENERATED_BACKGROUND_SOURCES = {
    GENERATED_BACKGROUND_SOURCE,
    REPRODUCER_BACKGROUND_SOURCE,
}

# Point-selection controls.  The non-valley families must lie below 0.05 both
# on the exploratory background and after strict DOP853 reintegration.  The
# slightly wider background threshold is needed to retain the geometric bottom
# of the L1 arc; every selected value remains far below 0.3.  The Euclidean
# spacing is checked globally, not only within each family.
BACKGROUND_LOW_LEAKAGE_TARGET = 0.05
STRICT_LOW_LEAKAGE_TARGET = 0.05
MIN_SELECTION_DISTANCE = 0.05

# Five well-separated anchors follow the narrow P1--P2 valley and include the
# manuscript points P1 and P2 exactly.  This is the one family intentionally
# exempted from the low-leakage threshold.
CANONICAL_VALLEY_POINTS = (
    (-0.020, 0.100),
    (-0.025, 0.150),
    (-0.030, 0.205),
    (-0.035, 0.261),
    (-0.030, 0.340),
)

P4 = (-0.677, 1.417)

# The P4-connected low-leakage structure is a two-sided U-shaped arc.  Points
# are ordered from the negative-delta_res arm through P4 and then along the
# positive-delta_res arm.  Matching chi levels on the two arms make the
# coverage visible without clustering near the arc bottom.
CANONICAL_ARC_POINTS = (
    (-1.310, 1.880),
    (-1.150, 1.740),
    (-0.990, 1.600),
    (-0.780, 1.460),
    P4,
    (-0.590, 1.460),
    (-0.510, 1.600),
    (-0.500, 1.740),
    (-0.510, 1.880),
)

L1 = (-0.430, 0.860)

# A second U-shaped low-leakage structure passes through L1.  The points are
# ordered along its upper-left arm, through the minimum-chi arc bottom B5, and
# then up the nearly vertical right arm.  L1 is stored once here and reused as
# the shared anchor of the resonance-line portrait row.
L1_ARC_POINT_IDS = ("B1", "B2", "B3", "L1", "B5", "B6", "B7", "B8", "B9")
CANONICAL_L1_ARC_POINTS = (
    (-0.920, 1.180),
    (-0.750, 1.060),
    (-0.560, 0.940),
    L1,
    (-0.240, 0.750),
    (-0.140, 0.820),
    (-0.140, 0.920),
    (-0.130, 1.060),
    (-0.120, 1.180),
)

# L1 itself is already stored in the L1 arc.  The remaining four points sample
# the higher-chi low-leakage windows of the bare resonance line without
# duplicating a coordinate or crowding P4.
RESONANCE_LINE_SAMPLES = (
    ("L2", 1.34),
    ("L3", 1.58),
    ("L4", 1.78),
    ("L5", 1.98),
)

GROUP_ORDER = ("valley", "arc", "l1_arc", "resonance_line")
GROUP_STYLE = {
    "valley": {
        "prefix": "V",
        "label": r"P1/P2 valley",
        "color": "#0072B2",
        "marker": "o",
    },
    "arc": {
        "prefix": "A",
        "label": r"P4 two-sided arc",
        "color": "#E69F00",
        "marker": "s",
    },
    "l1_arc": {
        "prefix": "B",
        "label": r"L1 two-sided arc",
        "color": "#CC79A7",
        "marker": "^",
    },
    "resonance_line": {
        "prefix": "L",
        "label": r"$\chi+2\mu=0$",
        "color": "#009E73",
        "marker": "D",
    },
}

# Explicit point IDs prevent a shared physical anchor from being integrated
# twice.  L1 appears in two conceptual rows of Fig. S3 but refers to one cached
# trajectory.
GRID_ROWS = (
    ("valley", ("V1", "V2", "V3", "V4", "V5")),
    ("arc", ("A1", "A4", "A5", "A6", "A9")),
    ("l1_arc", ("B1", "L1", "B5", "B6", "B9")),
    ("resonance_line", ("L1", "L2", "L3", "L4", "L5")),
)


@dataclass(frozen=True)
class SelectedPoint:
    """One deterministic point selected in the original data-driven audit."""

    point_id: str
    group: str
    mu: float
    chi: float
    selection_note: str

    @property
    def delta_res(self) -> float:
        return self.chi + 2.0 * self.mu


def defect_site(length: int) -> int:
    """Return the zero-indexed central defect site used in Figs. 1 and 2."""

    return 2 * ((length + 2) // 4) - 1


def initial_angles(
    length: int,
    bias: float = POLE_BIAS,
    *,
    initial_phi: float = INITIAL_PHI,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the pole-regularized Z2 state with one central defect."""

    theta = np.asarray(
        [bias, np.pi - bias] * (length // 2)
        + [bias] * (length % 2),
        dtype=float,
    )
    theta[defect_site(length)] = bias
    return theta, np.full(length, initial_phi, dtype=float)


def background_times() -> np.ndarray:
    """Return the 250 left endpoints used by the publication background."""

    return np.arange(0.0, T_MAX, BACKGROUND_DT)


def background_grids(
    mu_count: int = BACKGROUND_MU_COUNT,
    chi_count: int = BACKGROUND_CHI_COUNT,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a requested rectangular background grid."""

    if mu_count < 2 or chi_count < 2:
        raise ValueError("background grid counts must both be at least two")
    return (
        np.linspace(*BACKGROUND_MU_RANGE, mu_count),
        np.linspace(*BACKGROUND_CHI_RANGE, chi_count),
    )


def integrate_tdvp_trajectory(
    mu: float,
    chi: float,
    times: np.ndarray,
    *,
    method: str,
    rtol: float,
    atol: float,
    max_step: float,
) -> tuple[np.ndarray, int]:
    """Integrate the K=100 defect trajectory under an explicit protocol."""

    theta0, phi0 = initial_angles(TDVP_PERIOD)
    solution = solve_ivp(
        tdvpfun.eom,
        (float(times[0]), float(times[-1])),
        np.concatenate((theta0, phi0)),
        t_eval=times,
        method=method,
        args=(mu, chi),
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    expected_shape = (2 * TDVP_PERIOD, times.size)
    if not solution.success or solution.y.shape != expected_shape:
        raise RuntimeError(
            f"TDVP integration failed at (mu,chi)=({mu},{chi}): "
            f"{solution.message}"
        )
    trajectory = np.asarray(solution.y, dtype=float)
    if not np.all(np.isfinite(trajectory)):
        raise FloatingPointError(
            f"nonfinite TDVP trajectory at (mu,chi)=({mu},{chi})"
        )
    return trajectory, int(solution.nfev)


def _background_protocol_key(
    mu_grid: np.ndarray,
    chi_grid: np.ndarray,
) -> str:
    payload = {
        "schema_version": BACKGROUND_SCHEMA_VERSION,
        "mu": np.asarray(mu_grid, dtype=float).tolist(),
        "chi": np.asarray(chi_grid, dtype=float).tolist(),
        "times": background_times().tolist(),
        "K": TDVP_PERIOD,
        "pole_bias": POLE_BIAS,
        "initial_phi": INITIAL_PHI,
        "method": BACKGROUND_METHOD,
        "rtol": BACKGROUND_RTOL,
        "atol": BACKGROUND_ATOL,
        "max_step": "infinity",
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _generate_background_row(
    row_index: int,
    mu_value: float,
    chi_grid: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Generate one pickle-safe, resumable row of the leakage map."""

    times = background_times()
    values = np.empty(chi_grid.size, dtype=float)
    for chi_index, chi_value in enumerate(chi_grid):
        trajectory, _ = integrate_tdvp_trajectory(
            float(mu_value),
            float(chi_value),
            times,
            method=BACKGROUND_METHOD,
            rtol=BACKGROUND_RTOL,
            atol=BACKGROUND_ATOL,
            max_step=BACKGROUND_MAX_STEP,
        )
        values[chi_index] = float(
            np.mean(tdvpfun.get_qleak(trajectory))
        )
    return row_index, values


def _valid_background_row(path: Path, expected_size: int) -> bool:
    if not path.exists():
        return False
    try:
        row = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return False
    return bool(
        row.shape == (expected_size,) and np.all(np.isfinite(row))
    )


def generate_background_cache(
    cache_path: Path,
    row_directory: Path,
    *,
    mu_count: int,
    chi_count: int,
    workers: int,
    clear_rows: bool,
) -> None:
    """Generate the leakage background entirely from the TDVP source code.

    A completed mu row is stored atomically as a checkpoint.  Repeating the
    command resumes from all valid rows with the same protocol fingerprint.
    """

    if workers < 1:
        raise ValueError("workers must be a positive integer")
    mu_grid, chi_grid = background_grids(mu_count, chi_count)
    times = background_times()
    protocol_key = _background_protocol_key(mu_grid, chi_grid)
    checkpoint_directory = row_directory / f"background_{protocol_key}"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    row_paths = [
        checkpoint_directory / f"mu_row_{index:04d}.npy"
        for index in range(mu_grid.size)
    ]
    if clear_rows:
        for row_path in row_paths:
            row_path.unlink(missing_ok=True)

    missing = [
        index
        for index, row_path in enumerate(row_paths)
        if not _valid_background_row(row_path, chi_grid.size)
    ]
    if missing:
        completed = mu_grid.size - len(missing)
        print(
            f"Generating {len(missing)}/{mu_grid.size} background rows "
            f"with {workers} worker(s); checkpoints: "
            f"{checkpoint_directory}",
            flush=True,
        )

        def store_row(index: int, row: np.ndarray) -> None:
            nonlocal completed
            temporary_path = row_paths[index].with_suffix(".tmp")
            with temporary_path.open("wb") as handle:
                np.save(handle, row)
            temporary_path.replace(row_paths[index])
            completed += 1
            print(
                f"background rows complete: {completed}/{mu_grid.size}",
                flush=True,
            )

        if workers == 1:
            for index in missing:
                generated_index, row = _generate_background_row(
                    index,
                    float(mu_grid[index]),
                    chi_grid,
                )
                store_row(generated_index, row)
        else:
            try:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(
                            _generate_background_row,
                            index,
                            float(mu_grid[index]),
                            chi_grid,
                        ): index
                        for index in missing
                    }
                    for future in as_completed(futures):
                        generated_index, row = future.result()
                        store_row(generated_index, row)
            except (OSError, PermissionError) as exc:
                # Some managed/macOS environments prohibit the POSIX
                # semaphores used by ProcessPoolExecutor.  Serial generation
                # is slower but preserves exact numerical behavior and all
                # row checkpoints.
                print(
                    f"parallel background generation unavailable ({exc}); "
                    "falling back to one process",
                    flush=True,
                )
                for index in missing:
                    if _valid_background_row(
                        row_paths[index], chi_grid.size
                    ):
                        continue
                    generated_index, row = _generate_background_row(
                        index,
                        float(mu_grid[index]),
                        chi_grid,
                    )
                    store_row(generated_index, row)

    average = np.vstack(
        [np.load(path, allow_pickle=False) for path in row_paths]
    )
    expected_shape = (mu_grid.size, chi_grid.size)
    if average.shape != expected_shape or not np.all(np.isfinite(average)):
        raise AssertionError("assembled leakage background is invalid")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        schema_version=BACKGROUND_SCHEMA_VERSION,
        avg_q_leak=average,
        mu=mu_grid,
        chi=chi_grid,
        t_start=float(times[0]),
        t_stop=float(times[-1]),
        sample_count=times.size,
        J=0.5,
        N=TDVP_PERIOD,
        K=TDVP_PERIOD,
        pole_bias=POLE_BIAS,
        initial_phi=INITIAL_PHI,
        solver=BACKGROUND_METHOD,
        rtol=BACKGROUND_RTOL,
        atol=BACKGROUND_ATOL,
        max_step=BACKGROUND_MAX_STEP,
        protocol_key=protocol_key,
        source_name=GENERATED_BACKGROUND_SOURCE,
    )


def validate_background_cache(
    archive: np.lib.npyio.NpzFile,
    *,
    expected_mu_grid: np.ndarray | None = None,
    expected_chi_grid: np.ndarray | None = None,
) -> tuple[bool, str]:
    """Validate either the historical map or a pure-code generated map."""

    required = {
        "avg_q_leak",
        "mu",
        "chi",
        "t_start",
        "t_stop",
        "sample_count",
        "J",
        "N",
        "K",
        "source_name",
    }
    missing = sorted(required.difference(archive.files))
    if missing:
        return False, f"missing background fields: {', '.join(missing)}"
    try:
        average = np.asarray(archive["avg_q_leak"], dtype=float)
        mu_grid = np.asarray(archive["mu"], dtype=float)
        chi_grid = np.asarray(archive["chi"], dtype=float)
        if average.shape != (mu_grid.size, chi_grid.size):
            return False, "background arrays have inconsistent shapes"
        if mu_grid.size < 2 or chi_grid.size < 2:
            return False, "background grid is too small"
        if not (
            np.all(np.isfinite(average))
            and np.all(np.isfinite(mu_grid))
            and np.all(np.isfinite(chi_grid))
            and np.all(np.diff(mu_grid) > 0.0)
            and np.all(np.diff(chi_grid) > 0.0)
        ):
            return False, "background arrays are nonfinite or unordered"
        if not np.isclose(
            float(np.asarray(archive["t_start"]).item()),
            0.0,
            rtol=0.0,
            atol=1.0e-15,
        ):
            return False, "background t_start differs"
        if not np.isclose(
            float(np.asarray(archive["t_stop"]).item()),
            background_times()[-1],
            rtol=0.0,
            atol=1.0e-14,
        ):
            return False, "background t_stop differs"
        if (
            int(np.asarray(archive["sample_count"]).item())
            != background_times().size
        ):
            return False, "background sample count differs"
        if int(np.asarray(archive["K"]).item()) != TDVP_PERIOD:
            return False, "background K differs"
        if int(np.asarray(archive["N"]).item()) != TDVP_PERIOD:
            return False, "background N differs"
        if not np.isclose(
            float(np.asarray(archive["J"]).item()),
            0.5,
            rtol=0.0,
            atol=1.0e-15,
        ):
            return False, "background J differs"
        if expected_mu_grid is not None and not np.allclose(
            mu_grid,
            expected_mu_grid,
            rtol=0.0,
            atol=1.0e-14,
        ):
            return False, "background mu grid differs from the request"
        if expected_chi_grid is not None and not np.allclose(
            chi_grid,
            expected_chi_grid,
            rtol=0.0,
            atol=1.0e-14,
        ):
            return False, "background chi grid differs from the request"

        source = str(np.asarray(archive["source_name"]).item())
        if source not in {
            LEGACY_BACKGROUND_SOURCE,
            *GENERATED_BACKGROUND_SOURCES,
        }:
            return False, f"unrecognized background source {source!r}"
        if source in GENERATED_BACKGROUND_SOURCES:
            generated_fields = {
                "schema_version",
                "pole_bias",
                "initial_phi",
                "solver",
                "rtol",
                "atol",
                "max_step",
            }
            generated_missing = sorted(
                generated_fields.difference(archive.files)
            )
            if generated_missing:
                return False, (
                    "generated background lacks: "
                    + ", ".join(generated_missing)
                )
            expected_schema = (
                BACKGROUND_SCHEMA_VERSION
                if source == GENERATED_BACKGROUND_SOURCE
                else REPRODUCER_BACKGROUND_SCHEMA_VERSION
            )
            if int(np.asarray(archive["schema_version"]).item()) != expected_schema:
                return False, "generated background schema differs"
            if str(np.asarray(archive["solver"]).item()) != BACKGROUND_METHOD:
                return False, "generated background solver differs"
            for key, expected in (
                ("pole_bias", POLE_BIAS),
                ("initial_phi", INITIAL_PHI),
                ("rtol", BACKGROUND_RTOL),
                ("atol", BACKGROUND_ATOL),
                ("max_step", BACKGROUND_MAX_STEP),
            ):
                if not np.isclose(
                    float(np.asarray(archive[key]).item()),
                    expected,
                    rtol=0.0,
                    atol=1.0e-15,
                ):
                    return False, f"generated background {key} differs"
            # This script's row-checkpoint cache includes a protocol
            # fingerprint.  The main Fig. 1/Fig. 2 reproducer writes the same
            # numerical protocol under schema 2 but predates that optional
            # field; accepting it avoids recomputing the 301-by-201 map.
            if source == GENERATED_BACKGROUND_SOURCE:
                if "protocol_key" not in archive.files:
                    return False, "generated background lacks: protocol_key"
                expected_key = _background_protocol_key(mu_grid, chi_grid)
                if (
                    str(np.asarray(archive["protocol_key"]).item())
                    != expected_key
                ):
                    return False, "generated background protocol key differs"
    except (TypeError, ValueError, OverflowError) as exc:
        return False, str(exc)
    return True, "ok"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bilinear_value(
    mu_grid: np.ndarray,
    chi_grid: np.ndarray,
    values: np.ndarray,
    mu_value: float,
    chi_value: float,
) -> float:
    """Bilinearly interpolate a rectangular ``values[mu, chi]`` grid."""

    if not (
        mu_grid[0] <= mu_value <= mu_grid[-1]
        and chi_grid[0] <= chi_value <= chi_grid[-1]
    ):
        raise ValueError("requested point lies outside the leakage grid")
    i1 = int(np.searchsorted(mu_grid, mu_value, side="right"))
    j1 = int(np.searchsorted(chi_grid, chi_value, side="right"))
    i1 = min(max(i1, 1), len(mu_grid) - 1)
    j1 = min(max(j1, 1), len(chi_grid) - 1)
    i0, j0 = i1 - 1, j1 - 1
    mu_fraction = (mu_value - mu_grid[i0]) / (
        mu_grid[i1] - mu_grid[i0]
    )
    chi_fraction = (chi_value - chi_grid[j0]) / (
        chi_grid[j1] - chi_grid[j0]
    )
    lower = (1.0 - mu_fraction) * values[i0, j0] + (
        mu_fraction * values[i1, j0]
    )
    upper = (1.0 - mu_fraction) * values[i0, j1] + (
        mu_fraction * values[i1, j1]
    )
    return float((1.0 - chi_fraction) * lower + chi_fraction * upper)


def minimum_family_spacing(points: list[SelectedPoint], group: str) -> float:
    """Return the minimum Euclidean (mu, chi) distance within one family."""

    coordinates = np.asarray(
        [(point.mu, point.chi) for point in points if point.group == group],
        dtype=float,
    )
    if len(coordinates) < 2:
        return float("inf")
    differences = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    distances[np.diag_indices_from(distances)] = np.inf
    return float(np.min(distances))


def minimum_global_spacing(points: list[SelectedPoint]) -> float:
    """Return the minimum distance between any two unique selected points."""

    coordinates = np.asarray(
        [(point.mu, point.chi) for point in points],
        dtype=float,
    )
    differences = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    distances[np.diag_indices_from(distances)] = np.inf
    return float(np.min(distances))


def select_points(
    mu_grid: np.ndarray, chi_grid: np.ndarray, leakage: np.ndarray
) -> list[SelectedPoint]:
    """Return the locked, low-leakage, well-separated physical coordinates.

    The full leakage map remains a generated visual/background diagnostic, but
    small solver or grid changes cannot silently move the Appendix-H points.
    This makes the strict 27-trajectory conclusion dataset independent.
    """

    selected: list[SelectedPoint] = []
    if leakage.shape != (len(mu_grid), len(chi_grid)):
        raise ValueError("leakage background and grids are inconsistent")
    if not (
        mu_grid[0] <= BACKGROUND_MU_RANGE[0]
        and mu_grid[-1] >= BACKGROUND_MU_RANGE[1]
        and chi_grid[0] <= BACKGROUND_CHI_RANGE[0]
        and chi_grid[-1] >= BACKGROUND_CHI_RANGE[1]
    ):
        raise ValueError("leakage background does not cover the physical map")

    for number, (mu_value, chi_value) in enumerate(
        CANONICAL_VALLEY_POINTS, start=1
    ):
        selected.append(
            SelectedPoint(
                f"V{number}",
                "valley",
                mu_value,
                chi_value,
                (
                    "well-separated P1/P2-valley anchor; exact P1 and P2 "
                    "are included"
                ),
            )
        )

    for number, (mu_value, chi_value) in enumerate(
        CANONICAL_ARC_POINTS, start=1
    ):
        selected.append(
            SelectedPoint(
                f"A{number}",
                "arc",
                mu_value,
                chi_value,
                (
                    "exact manuscript P4 anchor"
                    if (mu_value, chi_value) == P4
                    else (
                        "well-separated point on the two-sided "
                        "P4-connected low-leakage arc"
                    )
                ),
            )
        )

    for point_id, (mu_value, chi_value) in zip(
        L1_ARC_POINT_IDS, CANONICAL_L1_ARC_POINTS
    ):
        selected.append(
            SelectedPoint(
                point_id,
                "l1_arc",
                mu_value,
                chi_value,
                (
                    "shared L1 resonance-line anchor"
                    if point_id == "L1"
                    else (
                        "well-separated point on the two-sided "
                        "L1-connected low-leakage arc"
                    )
                ),
            )
        )

    # Remaining points lying exactly on the physical resonance line.  L1 is
    # omitted here because it is the shared, singly stored anchor above.
    for point_id, chi_value in RESONANCE_LINE_SAMPLES:
        selected.append(
            SelectedPoint(
                point_id,
                "resonance_line",
                -0.5 * float(chi_value),
                float(chi_value),
                "well-separated low-leakage sample on chi+2mu=0",
            )
        )

    identifiers = [point.point_id for point in selected]
    if len(identifiers) != len(set(identifiers)):
        raise AssertionError("selected point identifiers must be unique")

    for group in GROUP_ORDER:
        spacing = minimum_family_spacing(selected, group)
        if spacing < MIN_SELECTION_DISTANCE - 1.0e-12:
            raise ValueError(
                f"{group} minimum spacing {spacing:.6f} is below "
                f"{MIN_SELECTION_DISTANCE:.6f}"
            )

    global_spacing = minimum_global_spacing(selected)
    if global_spacing < MIN_SELECTION_DISTANCE - 1.0e-12:
        raise ValueError(
            f"global minimum spacing {global_spacing:.6f} is below "
            f"{MIN_SELECTION_DISTANCE:.6f}"
        )

    for point in selected:
        if point.group == "valley":
            continue
        map_leakage = bilinear_value(
            mu_grid,
            chi_grid,
            leakage,
            point.mu,
            point.chi,
        )
        if map_leakage >= BACKGROUND_LOW_LEAKAGE_TARGET:
            raise ValueError(
                f"{point.point_id} background leakage {map_leakage:.6f} "
                f"is not below {BACKGROUND_LOW_LEAKAGE_TARGET:.3f}"
            )
    return selected


def common_times() -> np.ndarray:
    return np.linspace(0.0, T_MAX, SAMPLE_COUNT)


def integrate_point(
    point: SelectedPoint, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """Integrate one strict K=100 trajectory and retain local coordinates."""

    trajectory, function_evaluations = integrate_tdvp_trajectory(
        point.mu,
        point.chi,
        times,
        method=METHOD,
        rtol=RTOL,
        atol=ATOL,
        max_step=MAX_STEP,
    )
    i0 = defect_site(TDVP_PERIOD)
    if i0 != 49 or i0 % 2 != 1:
        raise AssertionError(
            "the defect protocol requires the central odd site i0=49"
        )
    local_sites = i0 + np.asarray([-1, 0, 1], dtype=int)
    theta_local = trajectory[local_sites]
    phi_local = trajectory[TDVP_PERIOD + local_sites]
    qleak = np.asarray(tdvpfun.get_qleak(trajectory), dtype=float)
    if qleak.shape != times.shape or not np.all(np.isfinite(qleak)):
        raise FloatingPointError(f"{point.point_id} leakage is nonfinite")
    minimum_pole_margin = float(
        np.min(np.abs(np.sin(trajectory[:TDVP_PERIOD])))
    )
    return (
        theta_local,
        phi_local,
        qleak,
        function_evaluations,
        minimum_pole_margin,
    )


def trajectory_metrics(
    times: np.ndarray,
    theta_defect: np.ndarray,
    phi_defect: np.ndarray,
    qleak: np.ndarray,
) -> dict[str, float]:
    """Return physical diagnostics of one defect-site Bloch trajectory."""

    x_coordinate = np.sin(theta_defect) * np.cos(phi_defect)
    y_coordinate = np.sin(theta_defect) * np.sin(phi_defect)
    duration = float(times[-1] - times[0])
    if duration <= 0.0:
        raise ValueError("trajectory duration must be positive")
    centroid_x = float(np.trapezoid(x_coordinate, times) / duration)
    centroid_y = float(np.trapezoid(y_coordinate, times) / duration)
    left_fraction = float(
        np.trapezoid((x_coordinate < 0.0).astype(float), times) / duration
    )
    return {
        "mean_leakage": float(np.trapezoid(qleak, times) / duration),
        "centroid_sin_theta_cos_phi": centroid_x,
        "centroid_sin_theta_sin_phi": centroid_y,
        "left_half_plane_fraction": left_fraction,
        "horizontal_span": float(np.ptp(x_coordinate)),
        "vertical_span": float(np.ptp(y_coordinate)),
        "closure_distance_at_T": float(
            np.hypot(
                x_coordinate[-1] - x_coordinate[0],
                y_coordinate[-1] - y_coordinate[0],
            )
        ),
    }


def generate_cache(
    cache_path: Path,
    leakage_path: Path,
) -> None:
    """Select, integrate, diagnose, and cache all audited sample points."""

    with np.load(leakage_path, allow_pickle=False) as background:
        valid, reason = validate_background_cache(background)
        if not valid:
            raise ValueError(f"incompatible leakage background: {reason}")
        mu_grid = np.asarray(background["mu"], dtype=float)
        chi_grid = np.asarray(background["chi"], dtype=float)
        leakage = np.asarray(background["avg_q_leak"], dtype=float)
    points = select_points(mu_grid, chi_grid, leakage)
    times = common_times()

    theta_values: list[np.ndarray] = []
    phi_values: list[np.ndarray] = []
    qleak_values: list[np.ndarray] = []
    map_leakage: list[float] = []
    strict_leakage: list[float] = []
    centroid_x: list[float] = []
    centroid_y: list[float] = []
    left_fraction: list[float] = []
    horizontal_span: list[float] = []
    vertical_span: list[float] = []
    closure_distance: list[float] = []
    function_evaluations: list[int] = []
    pole_margin: list[float] = []

    for point_number, point in enumerate(points, start=1):
        start = time.perf_counter()
        theta_local, phi_local, qleak, nfev, margin = integrate_point(
            point, times
        )
        metrics = trajectory_metrics(
            times, theta_local[1], phi_local[1], qleak
        )
        theta_values.append(theta_local)
        phi_values.append(phi_local)
        qleak_values.append(qleak)
        map_leakage.append(
            bilinear_value(
                mu_grid,
                chi_grid,
                leakage,
                point.mu,
                point.chi,
            )
        )
        strict_leakage.append(metrics["mean_leakage"])
        centroid_x.append(metrics["centroid_sin_theta_cos_phi"])
        centroid_y.append(metrics["centroid_sin_theta_sin_phi"])
        left_fraction.append(metrics["left_half_plane_fraction"])
        horizontal_span.append(metrics["horizontal_span"])
        vertical_span.append(metrics["vertical_span"])
        closure_distance.append(metrics["closure_distance_at_T"])
        function_evaluations.append(nfev)
        pole_margin.append(margin)
        print(
            f"[{point_number:02d}/{len(points)}] {point.point_id}: "
            f"(mu,chi)=({point.mu:+.3f},{point.chi:.3f}), "
            f"delta_res={point.delta_res:+.3f}, "
            f"Gamma_bar={metrics['mean_leakage']:.5f}, "
            f"Xbar={metrics['centroid_sin_theta_cos_phi']:+.4f}, "
            f"{time.perf_counter() - start:.2f} s",
            flush=True,
        )

    for point, strict_value in zip(points, strict_leakage):
        if (
            point.group != "valley"
            and strict_value >= STRICT_LOW_LEAKAGE_TARGET
        ):
            raise ValueError(
                f"{point.point_id} strict mean leakage {strict_value:.6f} "
                f"is not below {STRICT_LOW_LEAKAGE_TARGET:.3f}"
            )

    i0 = defect_site(TDVP_PERIOD)
    local_sites = i0 + np.asarray([-1, 0, 1], dtype=int)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        schema_version=SCHEMA_VERSION,
        background_map_sha256=sha256(leakage_path),
        times=times,
        point_id=np.asarray([point.point_id for point in points]),
        group=np.asarray([point.group for point in points]),
        selection_note=np.asarray([point.selection_note for point in points]),
        mu=np.asarray([point.mu for point in points]),
        chi=np.asarray([point.chi for point in points]),
        delta_res=np.asarray([point.delta_res for point in points]),
        background_map_mean_leakage=np.asarray(map_leakage),
        strict_mean_leakage=np.asarray(strict_leakage),
        theta=np.asarray(theta_values),
        phi=np.asarray(phi_values),
        qleak=np.asarray(qleak_values),
        centroid_sin_theta_cos_phi=np.asarray(centroid_x),
        centroid_sin_theta_sin_phi=np.asarray(centroid_y),
        left_half_plane_fraction=np.asarray(left_fraction),
        horizontal_span=np.asarray(horizontal_span),
        vertical_span=np.asarray(vertical_span),
        closure_distance_at_T=np.asarray(closure_distance),
        function_evaluations=np.asarray(function_evaluations),
        minimum_abs_sin_theta=np.asarray(pole_margin),
        local_sites=local_sites,
        defect_site=i0,
        defect_site_sublattice="odd",
        defect_local_detuning=np.asarray(
            [2.0 * point.mu + point.chi for point in points]
        ),
        K_tdvp=TDVP_PERIOD,
        T=T_MAX,
        sample_count=SAMPLE_COUNT,
        pole_bias=POLE_BIAS,
        initial_phi=INITIAL_PHI,
        tdvp_eom="tdvpfun.eom",
        tdvp_method=METHOD,
        tdvp_rtol=RTOL,
        tdvp_atol=ATOL,
        tdvp_max_step=MAX_STEP,
    )


def _scalar(archive: np.lib.npyio.NpzFile, key: str) -> Any:
    value = np.asarray(archive[key])
    if value.ndim != 0:
        raise ValueError(f"{key} must be a scalar")
    return value.item()


def validate_cache(
    archive: np.lib.npyio.NpzFile,
    leakage_path: Path,
) -> tuple[bool, str]:
    """Reject caches made with different selections or physical protocols."""

    required = {
        "schema_version",
        "background_map_sha256",
        "times",
        "point_id",
        "group",
        "selection_note",
        "mu",
        "chi",
        "delta_res",
        "background_map_mean_leakage",
        "strict_mean_leakage",
        "theta",
        "phi",
        "qleak",
        "centroid_sin_theta_cos_phi",
        "centroid_sin_theta_sin_phi",
        "left_half_plane_fraction",
        "horizontal_span",
        "vertical_span",
        "closure_distance_at_T",
        "function_evaluations",
        "minimum_abs_sin_theta",
        "local_sites",
        "defect_site",
        "defect_site_sublattice",
        "defect_local_detuning",
        "K_tdvp",
        "T",
        "sample_count",
        "pole_bias",
        "initial_phi",
        "tdvp_eom",
        "tdvp_method",
        "tdvp_rtol",
        "tdvp_atol",
        "tdvp_max_step",
    }
    missing = sorted(required.difference(archive.files))
    if missing:
        return False, f"missing fields: {', '.join(missing)}"
    try:
        if int(_scalar(archive, "schema_version")) != SCHEMA_VERSION:
            return False, "schema version differs"
        if str(_scalar(archive, "background_map_sha256")) != sha256(
            leakage_path
        ):
            return False, "background-map hash differs"
        expected_times = common_times()
        if not np.array_equal(np.asarray(archive["times"]), expected_times):
            return False, "time grid differs"
        if int(_scalar(archive, "K_tdvp")) != TDVP_PERIOD:
            return False, "K differs"
        if int(_scalar(archive, "defect_site")) != 49:
            return False, "defect site differs"
        if str(_scalar(archive, "defect_site_sublattice")) != "odd":
            return False, "defect sublattice differs"
        for key, expected in (
            ("T", T_MAX),
            ("pole_bias", POLE_BIAS),
            ("initial_phi", INITIAL_PHI),
            ("tdvp_rtol", RTOL),
            ("tdvp_atol", ATOL),
            ("tdvp_max_step", MAX_STEP),
        ):
            if not np.isclose(
                float(_scalar(archive, key)),
                expected,
                rtol=0.0,
                atol=1.0e-15,
            ):
                return False, f"{key} differs"
        if str(_scalar(archive, "tdvp_eom")) != "tdvpfun.eom":
            return False, "equation differs"
        if str(_scalar(archive, "tdvp_method")) != METHOD:
            return False, "solver differs"

        with np.load(leakage_path, allow_pickle=False) as background:
            points = select_points(
                np.asarray(background["mu"]),
                np.asarray(background["chi"]),
                np.asarray(background["avg_q_leak"]),
            )
        expected_ids = np.asarray([point.point_id for point in points])
        expected_groups = np.asarray([point.group for point in points])
        expected_mu = np.asarray([point.mu for point in points])
        expected_chi = np.asarray([point.chi for point in points])
        if not np.array_equal(archive["point_id"], expected_ids):
            return False, "point identifiers differ"
        if not np.array_equal(archive["group"], expected_groups):
            return False, "point groups differ"
        if not np.allclose(archive["mu"], expected_mu, rtol=0.0, atol=1e-14):
            return False, "mu coordinates differ"
        if not np.allclose(
            archive["chi"], expected_chi, rtol=0.0, atol=1e-14
        ):
            return False, "chi coordinates differ"
        if not np.allclose(
            archive["defect_local_detuning"],
            archive["delta_res"],
            rtol=0.0,
            atol=1e-14,
        ):
            return False, "odd-site local detuning is not delta_res"

        point_count = len(points)
        expected_local_shape = (point_count, 3, SAMPLE_COUNT)
        if np.asarray(archive["theta"]).shape != expected_local_shape:
            return False, "theta shape differs"
        if np.asarray(archive["phi"]).shape != expected_local_shape:
            return False, "phi shape differs"
        if np.asarray(archive["qleak"]).shape != (
            point_count,
            SAMPLE_COUNT,
        ):
            return False, "qleak shape differs"
        for key in required:
            values = np.asarray(archive[key])
            if values.dtype.kind in "fc" and not np.all(np.isfinite(values)):
                return False, f"{key} contains nonfinite values"
    except (TypeError, ValueError, OverflowError) as exc:
        return False, str(exc)
    return True, "ok"


def transition_brackets(
    coordinate: np.ndarray,
    values: np.ndarray,
    target: float,
) -> list[list[float]]:
    """Return every adjacent sampled interval that crosses ``target``.

    No interpolation is reported: finite-time orbit diagnostics can vary
    nonmonotonically between samples, so the data support only brackets.
    """

    order = np.argsort(coordinate)
    x = np.asarray(coordinate[order], dtype=float)
    y = np.asarray(values[order], dtype=float) - target
    candidates: list[list[float]] = []
    for index in range(len(x) - 1):
        if y[index] == 0.0:
            candidates.append([float(x[index]), float(x[index])])
        elif y[index] * y[index + 1] < 0.0:
            candidates.append(
                [float(x[index]), float(x[index + 1])]
            )
    if y[-1] == 0.0:
        candidates.append([float(x[-1]), float(x[-1])])
    return candidates


def signed_arc_length(
    mu_values: np.ndarray,
    chi_values: np.ndarray,
    anchor_index: int,
) -> np.ndarray:
    """Return cumulative parameter-space arc length relative to its bottom."""

    coordinates = np.column_stack((mu_values, chi_values))
    segment_lengths = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if not 0 <= anchor_index < len(cumulative):
        raise IndexError("arc anchor index lies outside the selected path")
    return cumulative - cumulative[anchor_index]


def arc_diagnostic_block(
    archive: np.lib.npyio.NpzFile,
    group: str,
    anchor_id: str,
) -> dict[str, Any]:
    """Build one JSON-ready block for an ordered two-sided arc."""

    groups = archive["group"].astype(str)
    indices = np.flatnonzero(groups == group)
    point_ids = archive["point_id"][indices].astype(str)
    anchor_matches = np.flatnonzero(point_ids == anchor_id)
    if len(anchor_matches) != 1:
        raise ValueError(f"arc anchor {anchor_id} is not unique in {group}")
    mu_values = np.asarray(archive["mu"][indices], dtype=float)
    chi_values = np.asarray(archive["chi"][indices], dtype=float)
    arc_coordinate = signed_arc_length(
        mu_values,
        chi_values,
        int(anchor_matches[0]),
    )
    centroid = np.asarray(
        archive["centroid_sin_theta_cos_phi"][indices], dtype=float
    )
    left_fraction = np.asarray(
        archive["left_half_plane_fraction"][indices], dtype=float
    )
    strict_leakage = np.asarray(
        archive["strict_mean_leakage"][indices], dtype=float
    )
    return {
        "group": group,
        "point_ids": point_ids.tolist(),
        "anchor_point_id": anchor_id,
        "signed_arc_length": arc_coordinate.tolist(),
        "mu": mu_values.tolist(),
        "chi": chi_values.tolist(),
        "delta_res_equals_chi_plus_2mu": np.asarray(
            archive["delta_res"][indices], dtype=float
        ).tolist(),
        "centroid_sin_theta_cos_phi": centroid.tolist(),
        "left_half_plane_fraction": left_fraction.tolist(),
        "strict_mean_leakage": strict_leakage.tolist(),
        "centroid_sign_change_brackets_signed_arc_length": (
            transition_brackets(arc_coordinate, centroid, 0.0)
        ),
        "left_fraction_half_brackets_signed_arc_length": (
            transition_brackets(arc_coordinate, left_fraction, 0.5)
        ),
        "maximum_strict_mean_leakage": float(np.max(strict_leakage)),
    }


def point_rows(archive: np.lib.npyio.NpzFile) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    point_ids = archive["point_id"].astype(str)
    for index, point_id in enumerate(point_ids):
        rows.append(
            {
                "point": point_id,
                "group": str(archive["group"][index]),
                "selection_note": str(archive["selection_note"][index]),
                "mu": float(archive["mu"][index]),
                "chi": float(archive["chi"][index]),
                "delta_res_equals_chi_plus_2mu": float(
                    archive["delta_res"][index]
                ),
                "background_map_mean_leakage": float(
                    archive["background_map_mean_leakage"][index]
                ),
                "strict_mean_leakage": float(
                    archive["strict_mean_leakage"][index]
                ),
                "centroid_sin_theta_cos_phi": float(
                    archive["centroid_sin_theta_cos_phi"][index]
                ),
                "centroid_sin_theta_sin_phi": float(
                    archive["centroid_sin_theta_sin_phi"][index]
                ),
                "left_half_plane_fraction": float(
                    archive["left_half_plane_fraction"][index]
                ),
                "horizontal_span": float(
                    archive["horizontal_span"][index]
                ),
                "vertical_span": float(archive["vertical_span"][index]),
                "closure_distance_at_T": float(
                    archive["closure_distance_at_T"][index]
                ),
                "minimum_abs_sin_theta": float(
                    archive["minimum_abs_sin_theta"][index]
                ),
                "solver_function_evaluations": int(
                    archive["function_evaluations"][index]
                ),
            }
        )
    return rows


def portrait_grid_rows(
    archive: np.lib.npyio.NpzFile,
) -> list[dict[str, Any]]:
    """Return the exact 20 panels displayed in the independent grid."""

    all_rows = {
        row["point"]: row for row in point_rows(archive)
    }
    rows: list[dict[str, Any]] = []
    for display_group, point_ids in GRID_ROWS:
        for point_id in point_ids:
            if point_id not in all_rows:
                raise ValueError(f"portrait point {point_id} is missing")
            source = all_rows[point_id]
            rows.append(
                {
                    "point": point_id,
                    "display_group": display_group,
                    "source_group": source["group"],
                    "mu": source["mu"],
                    "chi": source["chi"],
                    "delta_res_equals_chi_plus_2mu": source[
                        "delta_res_equals_chi_plus_2mu"
                    ],
                    "strict_mean_leakage": source[
                        "strict_mean_leakage"
                    ],
                    "centroid_sin_theta_cos_phi": source[
                        "centroid_sin_theta_cos_phi"
                    ],
                }
            )
    return rows


def build_summary(
    archive: np.lib.npyio.NpzFile,
    cache_path: Path,
    leakage_path: Path,
) -> dict[str, Any]:
    groups = archive["group"].astype(str)
    p4_arc = arc_diagnostic_block(archive, "arc", "A5")
    l1_arc = arc_diagnostic_block(archive, "l1_arc", "B5")

    def reorganizes_near_bottom(block: dict[str, Any]) -> bool:
        bracket_sets = (
            block["centroid_sign_change_brackets_signed_arc_length"],
            block["left_fraction_half_brackets_signed_arc_length"],
        )
        return bool(
            all(bracket_sets)
            and all(
                max(abs(endpoint) for endpoint in bracket) <= 0.15
                for brackets in bracket_sets
                for bracket in brackets
            )
            and block["maximum_strict_mean_leakage"]
            < STRICT_LOW_LEAKAGE_TARGET
        )

    repeated_reorganization = bool(
        reorganizes_near_bottom(p4_arc)
        and reorganizes_near_bottom(l1_arc)
    )

    group_counts = {
        group: int(np.count_nonzero(groups == group)) for group in GROUP_ORDER
    }
    group_minimum_spacing: dict[str, float] = {}
    for group in GROUP_ORDER:
        mask = groups == group
        coordinates = np.column_stack(
            (
                np.asarray(archive["mu"][mask], dtype=float),
                np.asarray(archive["chi"][mask], dtype=float),
            )
        )
        distances = np.linalg.norm(
            coordinates[:, None, :] - coordinates[None, :, :],
            axis=2,
        )
        distances[np.diag_indices_from(distances)] = np.inf
        group_minimum_spacing[group] = float(np.min(distances))
    all_coordinates = np.column_stack(
        (
            np.asarray(archive["mu"], dtype=float),
            np.asarray(archive["chi"], dtype=float),
        )
    )
    all_distances = np.linalg.norm(
        all_coordinates[:, None, :] - all_coordinates[None, :, :],
        axis=2,
    )
    all_distances[np.diag_indices_from(all_distances)] = np.inf
    global_minimum_spacing = float(np.min(all_distances))
    non_valley = groups != "valley"
    strict_non_valley = np.asarray(
        archive["strict_mean_leakage"][non_valley], dtype=float
    )
    background_non_valley = np.asarray(
        archive["background_map_mean_leakage"][non_valley], dtype=float
    )
    with np.load(leakage_path, allow_pickle=False) as background:
        background_source = str(
            np.asarray(background["source_name"]).item()
        )
        background_shape = list(
            np.asarray(background["avg_q_leak"]).shape
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "Geometry-aware, well-separated K=100, T=10 TDVP sampling of "
            "the P1/P2 valley, both arms of two independent low-leakage "
            "arcs through P4 and L1, and the resonance line"
        ),
        "artifacts": {
            "trajectory_cache": str(cache_path.resolve()),
            "leakage_background": str(leakage_path.resolve()),
            "leakage_background_sha256": sha256(leakage_path),
        },
        "protocol": {
            "spin": 0.5,
            "K": TDVP_PERIOD,
            "time_window": [0.0, T_MAX],
            "sample_count": SAMPLE_COUNT,
            "initial_state": (
                "pole-regularized Z2 state with one defect at i0"
            ),
            "pole_bias": POLE_BIAS,
            "initial_phi": INITIAL_PHI,
            "equation": "tdvpfun.eom",
            "solver": METHOD,
            "rtol": RTOL,
            "atol": ATOL,
            "max_step": MAX_STEP,
        },
        "background_protocol": {
            "source": background_source,
            "generated_from_code": (
                background_source in GENERATED_BACKGROUND_SOURCES
            ),
            "shape": background_shape,
            "time_sampling": (
                "250 uniform left endpoints over 0 <= t < 10"
            ),
            "K": TDVP_PERIOD,
            "pole_bias": POLE_BIAS,
            "initial_phi": INITIAL_PHI,
            "equation": "tdvpfun.eom",
            "solver": BACKGROUND_METHOD,
            "rtol": BACKGROUND_RTOL,
            "atol": BACKGROUND_ATOL,
            "max_step": "infinity",
            "role": (
                "visual/selection background only; all marked points use "
                "the strict protocol"
            ),
        },
        "selection": {
            "group_counts": group_counts,
            "total_points": int(len(groups)),
            "background_low_leakage_target_excluding_valley": (
                BACKGROUND_LOW_LEAKAGE_TARGET
            ),
            "strict_low_leakage_target_excluding_valley": (
                STRICT_LOW_LEAKAGE_TARGET
            ),
            "maximum_background_map_mean_leakage_excluding_valley": (
                float(np.max(background_non_valley))
            ),
            "maximum_strict_mean_leakage_excluding_valley": float(
                np.max(strict_non_valley)
            ),
            "minimum_parameter_space_spacing_required": (
                MIN_SELECTION_DISTANCE
            ),
            "minimum_parameter_space_spacing_by_group": (
                group_minimum_spacing
            ),
            "global_minimum_parameter_space_spacing": (
                global_minimum_spacing
            ),
            "P1_P2_valley": {
                "coordinates": [
                    list(point) for point in CANONICAL_VALLEY_POINTS
                ],
                "rule": (
                    "five separated anchors spanning the narrow valley, "
                    "including manuscript points P1 and P2 exactly; this "
                    "family is exempt from the leakage threshold"
                ),
            },
            "P4_arc": {
                "coordinates": [
                    list(point) for point in CANONICAL_ARC_POINTS
                ],
                "anchor": list(P4),
                "rule": (
                    "nine separated low-leakage coordinates ordered from "
                    "the negative-delta arm through P4 to the positive-delta "
                    "arm, with paired chi levels on the two sides"
                ),
            },
            "L1_arc": {
                "point_ids": list(L1_ARC_POINT_IDS),
                "coordinates": [
                    list(point) for point in CANONICAL_L1_ARC_POINTS
                ],
                "resonance_anchor": list(L1),
                "minimum_chi_anchor": list(CANONICAL_L1_ARC_POINTS[4]),
                "rule": (
                    "nine separated low-leakage coordinates ordered along "
                    "the upper-left arm, through the minimum-chi bottom B5, "
                    "and up the nearly vertical right arm; L1 is stored once"
                ),
            },
            "resonance_line": {
                "coordinate": "delta_res = chi + 2 mu",
                "chi_values": [
                    L1[1],
                    *[
                        float(chi_value)
                        for _, chi_value in RESONANCE_LINE_SAMPLES
                    ],
                ],
                "shared_anchor": (
                    "L1 is stored in the L1_arc group and reused logically"
                ),
                "rule": (
                    "five separated low-leakage samples spanning the "
                    "available resonance-line windows, with no duplicated "
                    "physical coordinate"
                ),
            },
        },
        "two_arc_orbit_comparison": {
            "coordinate_definition": (
                "signed cumulative Euclidean arc length in the (mu, chi) "
                "plane, with s_arc=0 at each arc's minimum-chi point"
            ),
            "centroid_definition": (
                "T^-1 integral sin(theta_i0) cos(phi_i0) dt"
            ),
            "left_fraction_definition": (
                "T^-1 integral Theta[-sin(theta_i0) cos(phi_i0)] dt"
            ),
            "P4_arc": p4_arc,
            "L1_arc": l1_arc,
            "supports_repeated_near_bottom_orbit_reorganization": (
                repeated_reorganization
            ),
            "interpretation": (
                "Both independent low-leakage arcs show a finite-time "
                "reorganization between positive and negative Bloch "
                "half-plane trajectories close to their geometric bottoms. "
                "The comparison is not an artifact of selecting one narrow "
                "parameter cut."
                if repeated_reorganization
                else (
                    "The two selected arcs do not both show a well-resolved "
                    "low-leakage orbit reorganization near their bottoms."
                )
            ),
            "scope_limit": (
                "This is a finite-time, single-initial-state TDVP diagnostic; "
                "the diagnostics need not vary monotonically between sparse "
                "samples.  It does not define a unique crossing, establish a "
                "phase transition or long-time chaos, or show that the bare "
                "resonance line alone controls the reorganization."
            ),
        },
        "portrait_grid_representatives": portrait_grid_rows(archive),
        "points": point_rows(archive),
    }


def panel_letter(index: int) -> str:
    """Return (a), ..., (z), (aa), ... for a zero-based panel index."""

    number = index
    letters = ""
    while True:
        number, remainder = divmod(number, 26)
        letters = chr(ord("a") + remainder) + letters
        if number == 0:
            break
        number -= 1
    return f"({letters})"


def plot_parameter_map(
    axis: plt.Axes,
    colorbar_axis: plt.Axes,
    archive: np.lib.npyio.NpzFile,
    mu_grid: np.ndarray,
    chi_grid: np.ndarray,
    leakage: np.ndarray,
) -> None:
    dmu = float(mu_grid[1] - mu_grid[0])
    dchi = float(chi_grid[1] - chi_grid[0])
    image = axis.imshow(
        leakage.T,
        origin="lower",
        extent=(
            mu_grid[0] - dmu / 2.0,
            mu_grid[-1] + dmu / 2.0,
            chi_grid[0] - dchi / 2.0,
            chi_grid[-1] + dchi / 2.0,
        ),
        aspect="auto",
        cmap="inferno",
        interpolation="nearest",
        vmin=0.0,
        vmax=0.10,
        rasterized=True,
    )
    resonance_mu = np.linspace(-1.0, 0.0, 200)
    axis.plot(
        resonance_mu,
        -2.0 * resonance_mu,
        color="white",
        lw=0.8,
        ls=(0, (3, 2)),
        alpha=0.9,
        zorder=2,
    )
    axis.text(
        -0.93,
        1.91,
        r"$\chi+2\mu=0$",
        color="white",
        fontsize=6.5,
        rotation=-33,
        va="top",
        path_effects=[pe.withStroke(linewidth=1.0, foreground="black")],
    )
    groups = archive["group"].astype(str)
    for group in GROUP_ORDER:
        mask = groups == group
        style = GROUP_STYLE[group]
        if group in ("arc", "l1_arc"):
            axis.plot(
                archive["mu"][mask],
                archive["chi"][mask],
                color=style["color"],
                lw=0.75,
                alpha=0.85,
                zorder=3,
            )
        axis.scatter(
            archive["mu"][mask],
            archive["chi"][mask],
            s=17,
            marker=style["marker"],
            facecolors="none",
            edgecolors=style["color"],
            linewidths=0.85,
            label=style["label"],
            zorder=4,
        )
    axis.set(
        xlabel=r"$\mu$",
        ylabel=r"$\chi$",
        xlim=(-1.5, 0.15),
        ylim=(0.0, 2.0),
        xticks=(-1.5, -1.0, -0.5, 0.0),
        yticks=(0.0, 0.5, 1.0, 1.5, 2.0),
    )
    axis.set_box_aspect(1.0)
    axis.text(
        0.025,
        0.97,
        "(a)",
        transform=axis.transAxes,
        va="top",
        color="white",
        fontsize=8.2,
        weight="semibold",
    )
    legend = axis.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.17),
        ncols=2,
        frameon=False,
        handletextpad=0.25,
        columnspacing=0.65,
        borderaxespad=0.0,
        fontsize=6.5,
    )
    for handle in legend.legend_handles:
        handle.set_sizes([20])

    colorbar = axis.figure.colorbar(
        image,
        cax=colorbar_axis,
        orientation="horizontal",
        ticks=(0.0, 0.05, 0.10),
    )
    colorbar.ax.xaxis.set_ticks_position("top")
    colorbar.ax.tick_params(
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False,
        labelsize=7.2,
        length=2.2,
        width=0.6,
        pad=1.0,
    )
    colorbar.outline.set_linewidth(0.6)
    axis.text(
        1.02,
        1.065,
        r"$\bar{\Gamma}$",
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=8.5,
        clip_on=False,
    )


def plot_arc_diagnostics(
    centroid_axis: plt.Axes,
    fraction_axis: plt.Axes,
    leakage_axis: plt.Axes,
    summary: dict[str, Any],
) -> None:
    """Compare the two physical arcs using their unnormalized signed length."""

    comparison = summary["two_arc_orbit_comparison"]
    plotted_arcs = (
        ("P4_arc", "arc", "solid"),
        ("L1_arc", "l1_arc", (0, (3.2, 1.5))),
    )
    for block_name, group, line_style in plotted_arcs:
        block = comparison[block_name]
        coordinate = np.asarray(block["signed_arc_length"], dtype=float)
        style = GROUP_STYLE[group]
        common_style = {
            "color": style["color"],
            "marker": style["marker"],
            "ls": line_style,
            "ms": 3.2,
            "mfc": "white",
            "mew": 0.75,
            "lw": 1.0,
        }
        centroid_axis.plot(
            coordinate,
            block["centroid_sin_theta_cos_phi"],
            label=style["label"],
            **common_style,
        )
        fraction_axis.plot(
            coordinate,
            block["left_half_plane_fraction"],
            **common_style,
        )
        leakage_axis.plot(
            coordinate,
            block["strict_mean_leakage"],
            **common_style,
        )

    centroid_axis.axhline(0.0, color="0.65", lw=0.65)
    centroid_axis.axvline(0.0, color="0.45", lw=0.65, ls=(0, (3, 2)))
    centroid_axis.set(
        xlim=(-0.85, 0.58),
        xticks=(-0.8, -0.4, 0.0, 0.4),
        ylim=(-0.65, 0.58),
        yticks=(-0.5, 0.0, 0.5),
        xlabel=r"$s_{\rm arc}$",
        ylabel=(
            r"$\overline{\sin\theta_{i_0}\cos\phi_{i_0}}$"
        ),
    )
    centroid_axis.text(
        0.025,
        0.97,
        "(b)",
        transform=centroid_axis.transAxes,
        va="top",
        fontsize=8.2,
        weight="semibold",
    )
    fraction_axis.axhline(0.5, color="0.65", lw=0.65)
    fraction_axis.axvline(0.0, color="0.45", lw=0.65, ls=(0, (3, 2)))
    fraction_axis.set(
        xlim=(-0.85, 0.58),
        xticks=(-0.8, -0.4, 0.0, 0.4),
        ylim=(-0.05, 1.05),
        yticks=(0.0, 0.5, 1.0),
        xlabel=r"$s_{\rm arc}$",
        ylabel=(
            r"$\overline{\Theta[-\sin\theta_{i_0}\cos\phi_{i_0}]}$"
        ),
    )
    fraction_axis.text(
        0.025,
        0.97,
        "(c)",
        transform=fraction_axis.transAxes,
        va="top",
        fontsize=8.2,
        weight="semibold",
    )

    leakage_axis.axhline(
        STRICT_LOW_LEAKAGE_TARGET,
        color="0.65",
        lw=0.65,
        ls=(0, (3, 2)),
    )
    leakage_axis.axvline(0.0, color="0.45", lw=0.65, ls=(0, (3, 2)))
    leakage_axis.set(
        xlim=(-0.85, 0.58),
        xticks=(-0.8, -0.4, 0.0, 0.4),
        ylim=(0.01, 0.052),
        yticks=(0.01, 0.03, 0.05),
        xlabel=r"$s_{\rm arc}$",
        ylabel=r"$\bar{\Gamma}$",
    )
    leakage_axis.text(
        0.025,
        0.97,
        "(d)",
        transform=leakage_axis.transAxes,
        va="top",
        fontsize=8.2,
        weight="semibold",
    )


def plot_phase_portrait(
    axis: plt.Axes,
    theta: np.ndarray,
    phi: np.ndarray,
    color: str,
    point_id: str,
    letter: str,
) -> None:
    x_coordinate = np.sin(theta) * np.cos(phi)
    y_coordinate = np.sin(theta) * np.sin(phi)
    circle = np.linspace(0.0, 2.0 * np.pi, 361)
    axis.plot(
        np.cos(circle),
        np.sin(circle),
        color="0.70",
        lw=0.55,
        zorder=0,
    )
    axis.axhline(0.0, color="0.86", lw=0.45, zorder=0)
    axis.axvline(0.0, color="0.86", lw=0.45, zorder=0)
    axis.plot(
        x_coordinate,
        y_coordinate,
        color=color,
        lw=0.72,
        alpha=0.92,
        zorder=2,
    )
    axis.set(
        xlim=(-1.0, 1.0),
        ylim=(-1.0, 1.0),
        xticks=(-1.0, 0.0, 1.0),
        yticks=(-1.0, 0.0, 1.0),
        aspect="equal",
    )
    axis.text(
        0.025,
        0.975,
        letter,
        transform=axis.transAxes,
        va="top",
        fontsize=7.6,
        weight="semibold",
    )
    axis.text(
        0.975,
        0.975,
        point_id,
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7.4,
        weight="semibold",
        color=color,
    )


def plot_figure(
    archive: np.lib.npyio.NpzFile,
    summary: dict[str, Any],
    leakage_path: Path,
    pdf_path: Path,
    png_path: Path,
) -> None:
    """Create one full-width supplementary figure with explicit geometry."""

    plt.style.use(resolve_style_file())
    with np.load(leakage_path) as legacy:
        mu_grid = np.asarray(legacy["mu"])
        chi_grid = np.asarray(legacy["chi"])
        leakage = np.asarray(legacy["avg_q_leak"])

    # ------------------------------------------------------------------
    # Manual geometry controls.
    #
    # figsize: 7.15 in is the standard two-column width.  The compact height
    # keeps this to one supplementary row.  ``width_ratios`` allocates slightly
    # more room to the square parameter map; ``wspace`` is the main knob for
    # formula-label clearance among the three arc-comparison axes.
    # ------------------------------------------------------------------
    figure = plt.figure(figsize=(7.15, 2.45))
    grid = figure.add_gridspec(
        1,
        4,
        left=0.060,
        right=0.993,
        bottom=0.205,
        top=0.805,
        width_ratios=(1.15, 1.0, 1.0, 1.0),
        wspace=0.58,
    )
    map_axis = figure.add_subplot(grid[0])
    centroid_axis = figure.add_subplot(grid[1])
    fraction_axis = figure.add_subplot(grid[2])
    leakage_axis = figure.add_subplot(grid[3])

    # The map colorbar is tied to panel (a), so it follows any later map
    # resizing.  Lower the second coordinate to move the bar closer.
    colorbar_axis = map_axis.inset_axes([0.0, 1.055, 0.90, 0.040])
    plot_parameter_map(
        map_axis,
        colorbar_axis,
        archive,
        mu_grid,
        chi_grid,
        leakage,
    )
    plot_arc_diagnostics(
        centroid_axis,
        fraction_axis,
        leakage_axis,
        summary,
    )

    for axis in [
        map_axis,
        centroid_axis,
        fraction_axis,
        leakage_axis,
    ]:
        axis.tick_params(
            direction="in",
            top=True,
            right=True,
            length=2.6,
            width=0.65,
            pad=1.4,
        )
        for spine in axis.spines.values():
            spine.set_linewidth(0.7)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.025,
    )
    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.025,
    )
    plt.close(figure)


def plot_portrait_grid(
    archive: np.lib.npyio.NpzFile,
    pdf_path: Path,
    png_path: Path,
) -> None:
    """Plot 20 Bloch portraits from four selected physical paths."""

    plt.style.use(resolve_style_file())

    # ------------------------------------------------------------------
    # Manual geometry controls.
    #
    # The width is the APS two-column width.  ``left`` leaves room for the
    # shared physical y label and colored row headings.  ``wspace`` and
    # ``hspace`` control the gaps between independent unit-circle portraits.
    # Increase the height only if the coordinate titles need more clearance.
    # ------------------------------------------------------------------
    figure = plt.figure(figsize=(7.15, 5.35))
    grid = figure.add_gridspec(
        len(GRID_ROWS),
        5,
        left=0.112,
        right=0.990,
        bottom=0.090,
        top=0.955,
        wspace=0.17,
        hspace=0.23,
    )
    point_ids_all = archive["point_id"].astype(str)
    point_index_by_id = {
        point_id: index for index, point_id in enumerate(point_ids_all)
    }
    panel_index = 0
    axes: list[list[plt.Axes]] = []

    for row_index, (group, point_ids) in enumerate(GRID_ROWS):
        row_axes: list[plt.Axes] = []
        missing = [
            point_id
            for point_id in point_ids
            if point_id not in point_index_by_id
        ]
        if missing:
            raise ValueError(
                f"portrait grid is missing points: {', '.join(missing)}"
            )
        selected = [
            point_index_by_id[point_id] for point_id in point_ids
        ]
        for column_index, point_index in enumerate(selected):
            axis = figure.add_subplot(grid[row_index, column_index])
            row_axes.append(axis)
            point_id = str(archive["point_id"][point_index])
            plot_phase_portrait(
                axis,
                np.asarray(archive["theta"][point_index, 1]),
                np.asarray(archive["phi"][point_index, 1]),
                GROUP_STYLE[group]["color"],
                point_id,
                panel_letter(panel_index),
            )
            panel_index += 1
            mu_value = float(archive["mu"][point_index])
            chi_value = float(archive["chi"][point_index])
            axis.set_title(
                rf"$(\mu,\chi)=({mu_value:.3f},{chi_value:.3f})$",
                fontsize=6.2,
                pad=2.0,
            )
            axis.tick_params(
                direction="in",
                top=True,
                right=True,
                length=2.5,
                width=0.62,
                pad=1.2,
                labelsize=7.2,
            )
            for spine in axis.spines.values():
                spine.set_linewidth(0.68)
            if column_index > 0:
                axis.tick_params(labelleft=False)
            if row_index < len(GRID_ROWS) - 1:
                axis.tick_params(labelbottom=False)
        row_axes[0].text(
            -0.37,
            0.5,
            GROUP_STYLE[group]["label"],
            transform=row_axes[0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=7.2,
            color=GROUP_STYLE[group]["color"],
            weight="semibold",
            clip_on=False,
        )
        axes.append(row_axes)

    figure.text(
        0.545,
        0.018,
        r"$\sin\theta_{i_0}\cos\phi_{i_0}$",
        ha="center",
        va="bottom",
        fontsize=10.5,
    )
    figure.text(
        0.012,
        0.520,
        r"$\sin\theta_{i_0}\sin\phi_{i_0}$",
        ha="left",
        va="center",
        rotation=90,
        fontsize=10.5,
    )

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.025,
    )
    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.025,
    )
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--leakage-cache",
        type=Path,
        default=LEAKAGE_CACHE,
        help=(
            "generated leakage-background cache; it is built automatically "
            "when absent"
        ),
    )
    parser.add_argument(
        "--background-row-dir",
        type=Path,
        default=DATA_DIR / "supp_leakage_rows",
        help="resumable row-checkpoint directory for background generation",
    )
    parser.add_argument(
        "--background-workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="worker processes used for independent background mu rows",
    )
    parser.add_argument(
        "--background-mu-count",
        type=int,
        default=BACKGROUND_MU_COUNT,
        help="mu-grid size; the publication background uses 301",
    )
    parser.add_argument(
        "--background-chi-count",
        type=int,
        default=BACKGROUND_CHI_COUNT,
        help="chi-grid size; the publication background uses 201",
    )
    parser.add_argument(
        "--recompute-background",
        action="store_true",
        help="discard compatible row checkpoints and rebuild the background",
    )
    parser.add_argument(
        "--background-only",
        action="store_true",
        help="generate/validate the leakage background and exit",
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument(
        "--portrait-pdf", type=Path, default=DEFAULT_PORTRAIT_PDF
    )
    parser.add_argument(
        "--portrait-png", type=Path, default=DEFAULT_PORTRAIT_PNG
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="reintegrate all points even when a compatible cache exists",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="write NPZ/JSON but skip PDF and PNG generation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path_name in (
        "leakage_cache",
        "background_row_dir",
        "cache",
        "summary",
        "pdf",
        "png",
        "portrait_pdf",
        "portrait_png",
    ):
        setattr(args, path_name, getattr(args, path_name).resolve())
    if args.background_workers < 1:
        raise ValueError("--background-workers must be positive")
    expected_mu_grid, expected_chi_grid = background_grids(
        args.background_mu_count,
        args.background_chi_count,
    )
    background_ok = False
    background_reason = "background cache does not exist"
    if args.leakage_cache.exists() and not args.recompute_background:
        with np.load(args.leakage_cache, allow_pickle=False) as background:
            background_ok, background_reason = validate_background_cache(
                background,
                expected_mu_grid=expected_mu_grid,
                expected_chi_grid=expected_chi_grid,
            )
    if not background_ok:
        if args.leakage_cache.exists() and not args.recompute_background:
            print(
                f"rebuilding incompatible background: {background_reason}",
                flush=True,
            )
        generate_background_cache(
            args.leakage_cache,
            args.background_row_dir,
            mu_count=args.background_mu_count,
            chi_count=args.background_chi_count,
            workers=args.background_workers,
            clear_rows=args.recompute_background,
        )
    with np.load(args.leakage_cache, allow_pickle=False) as background:
        background_ok, background_reason = validate_background_cache(
            background,
            expected_mu_grid=expected_mu_grid,
            expected_chi_grid=expected_chi_grid,
        )
    if not background_ok:
        raise RuntimeError(
            f"generated background failed validation: {background_reason}"
        )
    if args.background_only:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "background": str(args.leakage_cache),
                    "shape": [
                        args.background_mu_count,
                        args.background_chi_count,
                    ],
                    "row_checkpoints": str(args.background_row_dir),
                },
                indent=2,
            )
        )
        return 0

    cache_ok = False
    cache_reason = "cache does not exist"
    if args.cache.exists() and not args.force:
        with np.load(args.cache, allow_pickle=False) as archive:
            cache_ok, cache_reason = validate_cache(
                archive, args.leakage_cache
            )
    if not cache_ok:
        if args.cache.exists() and not args.force:
            print(f"rebuilding incompatible cache: {cache_reason}", flush=True)
        generate_cache(args.cache, args.leakage_cache)

    with np.load(args.cache, allow_pickle=False) as archive:
        valid, reason = validate_cache(archive, args.leakage_cache)
        if not valid:
            raise RuntimeError(f"new cache failed validation: {reason}")
        summary = build_summary(archive, args.cache, args.leakage_cache)
        summary["artifacts"].update(
            {
                "overview_pdf": None if args.data_only else str(args.pdf),
                "overview_png": None if args.data_only else str(args.png),
                "portrait_grid_pdf": (
                    None if args.data_only else str(args.portrait_pdf)
                ),
                "portrait_grid_png": (
                    None if args.data_only else str(args.portrait_png)
                ),
            }
        )
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with args.summary.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
        if not args.data_only:
            plot_figure(
                archive,
                summary,
                args.leakage_cache,
                args.pdf,
                args.png,
            )
            plot_portrait_grid(
                archive,
                args.portrait_pdf,
                args.portrait_png,
            )

    print(
        json.dumps(
            {
                "status": "ok",
                "cache": str(args.cache),
                "summary": str(args.summary),
                "pdf": None if args.data_only else str(args.pdf),
                "png": None if args.data_only else str(args.png),
                "portrait_pdf": (
                    None if args.data_only else str(args.portrait_pdf)
                ),
                "portrait_png": (
                    None if args.data_only else str(args.portrait_png)
                ),
                "supports_repeated_near_bottom_orbit_reorganization": (
                    summary["two_arc_orbit_comparison"][
                        "supports_repeated_near_bottom_orbit_reorganization"
                    ]
                ),
                "P4_arc_centroid_brackets_signed_arc_length": summary[
                    "two_arc_orbit_comparison"
                ]["P4_arc"][
                    "centroid_sign_change_brackets_signed_arc_length"
                ],
                "L1_arc_centroid_brackets_signed_arc_length": summary[
                    "two_arc_orbit_comparison"
                ]["L1_arc"][
                    "centroid_sign_change_brackets_signed_arc_length"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
