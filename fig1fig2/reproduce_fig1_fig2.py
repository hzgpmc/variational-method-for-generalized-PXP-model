#!/usr/bin/env python3
"""Pure-code reproducer for the confinement figures and Bloch-plane portraits.

The original plotting logic is spread over the final cells of
``confinementv3.ipynb``.  This script keeps only the calculations needed for
the two publication figures:

* Fig. 1: the time-averaged TDVP leakage map together with the P1/P6 ED--TDVP
  defect profiles;
* Fig. 2: one-row Bloch-plane portraits at the defect site for P1--P6.

The script also writes the ED and TDVP defect-response widths used in the
manuscript table to ``output/defect_widths.csv`` and JSON.

All generated spin-1/2 trajectories use the specialized TDVP flow in
``tdvpfun``; no notebook is required.

Run from this directory with the supplied conda environment, for example

    conda run -n pxp-variational-repro python reproduce_fig1_fig2.py

No numerical dataset is distributed with this script.  If the leakage cache is
absent, the full 301-by-201 parameter grid is integrated and cached one
``mu`` row at a time, so an interrupted calculation can be resumed.  The
default grid and lower-tolerance RK45 protocol reproduce the landscape used to
select P1--P6.  The six marked points are additionally reintegrated with the
strict DOP853 protocol used for the profile and phase-space calculations.

All newly computed ED and TDVP data use one comparison protocol:
``theta_pole=1e-3``, ``phi_i=1e-3``, TDVP ``K=100``, ED ``L=24``, and DOP853
with ``rtol=1e-9``, ``atol=1e-11``, and ``max_step=0.02``.  The more expensive
defect/reference profiles and phase trajectories are generated once and
cached.  Use ``--recompute-leakage``, ``--recompute-core``, or
``--recompute-phase`` to refresh them.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
DEFAULT_OUTPUT_DIR = HERE / "output"
os.environ.setdefault("MPLCONFIGDIR", str(DATA_DIR / "mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import EDfun
import pxpbasisS
import tdvpfun


POINTS = (
    (-0.025, 0.150, "P1"),
    (-0.035, 0.261, "P2"),
    (-0.858, 1.136, "P3"),
    (-0.677, 1.417, "P4"),
    (-0.105, 1.538, "P5"),
    (-0.120, 1.900, "P6"),
)

# One source of truth for every newly generated panel and table entry.
POLE_BIAS = 1.0e-3
INITIAL_PHI = 1.0e-3
TDVP_PERIOD = 100
ED_LENGTH = 24
TDVP_METHOD = "DOP853"
ED_SOLVER_NAME = "dop853"
SOLVER_RTOL = 1.0e-9
SOLVER_ATOL = 1.0e-11
SOLVER_MAX_STEP = 0.02
CORE_TMAX = 10.0
CORE_SAMPLE_COUNT = 1001
TDVP_REFERENCE_PERIOD = 2
CORE_CACHE_SCHEMA_VERSION = 2
PHASE_CACHE_SCHEMA_VERSION = 2
WIDTH_OUTPUT_SCHEMA_VERSION = 2
LEAKAGE_VALIDATION_SCHEMA_VERSION = 1
LEAKAGE_VALIDATION_MAX_DELTA = 1.0e-3
LEAKAGE_CACHE_SCHEMA_VERSION = 2
LEAKAGE_METHOD = "RK45"
LEAKAGE_RTOL = 1.0e-3
LEAKAGE_ATOL = 1.0e-5
LEAKAGE_MAX_STEP = np.inf
LEAKAGE_DT = 0.04
GENERATED_LEAKAGE_SOURCE = "generated_by_reproduce_fig1_fig2.py"

# Grid and sampling constants reproduce the historical landscape.  A
# compatibility reader for the original cache is retained, but normal use
# generates a schema-versioned cache containing the full numerical protocol.
LEGACY_LEAKAGE_J = 0.5
LEGACY_LEAKAGE_N = 100
LEGACY_LEAKAGE_K = 100
LEGACY_LEAKAGE_T_START = 0.0
LEGACY_LEAKAGE_T_STOP = 9.96
LEGACY_LEAKAGE_SAMPLE_COUNT = 250
LEGACY_LEAKAGE_SHAPE = (301, 201)
LEGACY_LEAKAGE_SOURCE = "tdvp_L100_sps2_t30.0.pkl"
LEGACY_MU_GRID = np.linspace(-1.5, 1.5, LEGACY_LEAKAGE_SHAPE[0])
LEGACY_CHI_GRID = np.linspace(0.0, 2.0, LEGACY_LEAKAGE_SHAPE[1])


def comparison_points() -> np.ndarray:
    """Return the ordered numerical (mu, chi) coordinates for P1--P6."""

    return np.asarray([(mu, chi) for mu, chi, _ in POINTS], dtype=float)


def comparison_labels() -> np.ndarray:
    """Return the ordered labels P1--P6."""

    return np.asarray([label for _, _, label in POINTS])


def core_times() -> np.ndarray:
    """Return the common output-time grid for profiles and table widths."""

    return np.linspace(0.0, CORE_TMAX, CORE_SAMPLE_COUNT)


def defect_site(length: int) -> int:
    """Zero-indexed defect position used by the notebook."""

    return 2 * ((length + 2) // 4) - 1


def initial_angles(
    length: int,
    bias: float = POLE_BIAS,
    *,
    initial_phi: float = INITIAL_PHI,
    with_defect: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Pole-regularized Z2 state, optionally with the central defect.

    ``bias`` and ``initial_phi`` are shared by ED and TDVP.  Keeping both
    explicit prevents a visually small regularization choice from silently
    changing an ED--TDVP comparison.
    """

    theta = np.asarray(
        [bias, np.pi - bias] * (length // 2) + [bias] * (length % 2),
        dtype=float,
    )
    if with_defect:
        theta[defect_site(length)] = bias
    return theta, np.full(length, initial_phi, dtype=float)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )


def leakage_times() -> np.ndarray:
    """Return the 250 uniform left endpoints used for the leakage landscape."""

    return np.arange(LEGACY_LEAKAGE_T_START, CORE_TMAX, LEAKAGE_DT)


def _generate_leakage_row(
    row_index: int,
    mu: float,
    chi_grid: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Integrate one complete ``mu`` row of the leakage landscape.

    A whole row is the unit of multiprocessing and checkpointing.  Keeping the
    worker at module scope makes it pickle-safe on platforms that use the
    ``spawn`` multiprocessing start method.
    """

    times = leakage_times()
    values = np.empty(chi_grid.size, dtype=float)
    for chi_index, chi in enumerate(chi_grid):
        trajectory = integrate_tdvp(
            float(mu),
            float(chi),
            times,
            period=TDVP_PERIOD,
            bias=POLE_BIAS,
            initial_phi=INITIAL_PHI,
            method=LEAKAGE_METHOD,
            rtol=LEAKAGE_RTOL,
            atol=LEAKAGE_ATOL,
            max_step=LEAKAGE_MAX_STEP,
        )
        values[chi_index] = float(np.mean(tdvpfun.get_qleak(trajectory)))
    return row_index, values


def _valid_leakage_row(path: Path, expected_size: int) -> bool:
    """Return whether a row checkpoint is complete and finite."""

    if not path.exists():
        return False
    try:
        row = np.load(path)
    except (OSError, ValueError):
        return False
    return row.shape == (expected_size,) and bool(np.all(np.isfinite(row)))


def generate_leakage_cache(
    cache: Path,
    row_dir: Path,
    *,
    workers: int,
    clear_rows: bool = False,
) -> None:
    """Generate the full leakage map without relying on distributed data.

    Each completed ``mu`` row is written immediately.  Re-running the command
    resumes from valid row files; ``clear_rows=True`` starts the scan again.
    """

    mu_grid = LEGACY_MU_GRID
    chi_grid = LEGACY_CHI_GRID
    times = leakage_times()
    if times.size != LEGACY_LEAKAGE_SAMPLE_COUNT:
        raise AssertionError(
            f"expected {LEGACY_LEAKAGE_SAMPLE_COUNT} leakage samples, "
            f"got {times.size}"
        )

    row_dir.mkdir(parents=True, exist_ok=True)
    row_paths = [
        row_dir / f"mu_row_{row_index:03d}.npy"
        for row_index in range(mu_grid.size)
    ]
    if clear_rows:
        for row_path in row_paths:
            row_path.unlink(missing_ok=True)

    missing = [
        row_index
        for row_index, row_path in enumerate(row_paths)
        if not _valid_leakage_row(row_path, chi_grid.size)
    ]
    if missing:
        print(
            f"Generating {len(missing)} of {mu_grid.size} leakage rows "
            f"with {workers} worker(s).",
            flush=True,
        )
        completed = mu_grid.size - len(missing)

        def store_row(row_index: int, row: np.ndarray) -> None:
            nonlocal completed
            row_path = row_paths[row_index]
            temporary = row_path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.save(handle, row)
            temporary.replace(row_path)
            completed += 1
            print(
                f"Leakage rows complete: {completed}/{mu_grid.size}",
                flush=True,
            )

        if workers == 1:
            for row_index in missing:
                generated_index, row = _generate_leakage_row(
                    row_index,
                    float(mu_grid[row_index]),
                    chi_grid,
                )
                store_row(generated_index, row)
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _generate_leakage_row,
                        row_index,
                        float(mu_grid[row_index]),
                        chi_grid,
                    ): row_index
                    for row_index in missing
                }
                for future in as_completed(futures):
                    row_index, row = future.result()
                    store_row(row_index, row)

    average = np.vstack([np.load(row_path) for row_path in row_paths])
    if average.shape != LEGACY_LEAKAGE_SHAPE:
        raise AssertionError(
            f"generated leakage shape {average.shape} != "
            f"{LEGACY_LEAKAGE_SHAPE}"
        )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        schema_version=LEAKAGE_CACHE_SCHEMA_VERSION,
        avg_q_leak=average,
        mu=mu_grid,
        chi=chi_grid,
        t_start=float(times[0]),
        t_stop=float(times[-1]),
        sample_count=times.size,
        J=LEGACY_LEAKAGE_J,
        N=LEGACY_LEAKAGE_N,
        K=LEGACY_LEAKAGE_K,
        pole_bias=POLE_BIAS,
        initial_phi=INITIAL_PHI,
        solver=LEAKAGE_METHOD,
        rtol=LEAKAGE_RTOL,
        atol=LEAKAGE_ATOL,
        max_step=LEAKAGE_MAX_STEP,
        source_name=GENERATED_LEAKAGE_SOURCE,
    )


def prepare_fig1_cache(legacy_pickle: Path, cache: Path) -> None:
    """Reduce the 693-MB legacy scan to the array actually used in Fig. 1."""

    with legacy_pickle.open("rb") as handle:
        q_leak, _width, mu, chi, times, spin, length, period = pickle.load(handle)
    times = np.asarray(times)
    averaging_mask = (times >= 0.0) & (times < CORE_TMAX)
    sample_count = int(np.count_nonzero(averaging_mask))
    if sample_count != LEGACY_LEAKAGE_SAMPLE_COUNT:
        raise ValueError(
            "Legacy scan does not contain the expected 250 left-endpoint "
            "samples over 0 <= t < 10."
        )
    average = np.mean(q_leak[:, :, averaging_mask], axis=2)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        avg_q_leak=average,
        mu=mu,
        chi=chi,
        t_start=times[averaging_mask][0],
        t_stop=times[averaging_mask][-1],
        sample_count=sample_count,
        J=spin,
        N=length,
        K=period,
        source_name=legacy_pickle.name,
    )


def _cache_scalar(cache: np.lib.npyio.NpzFile, key: str) -> object:
    """Read a scalar cache field and reject accidentally array-valued metadata."""

    value = np.asarray(cache[key])
    if value.ndim != 0:
        raise ValueError(f"cache field {key!r} must be scalar, got {value.shape}")
    return value.item()


def _float_matches(value: object, expected: float) -> bool:
    """Compare protocol scalars without NumPy's permissive default rtol."""

    return bool(np.isclose(float(value), expected, rtol=0.0, atol=1.0e-15))


def validate_leakage_cache(
    leakage: np.lib.npyio.NpzFile,
) -> tuple[bool, str]:
    """Validate a generated cache or the optional historical cache.

    Newly generated caches record the full numerical protocol.  The historical
    compact archive did not contain pole-bias or initial-phi fields, so only
    metadata actually present in that artifact are checked for the legacy
    compatibility path.
    """

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
    missing = sorted(required.difference(leakage.files))
    if missing:
        return False, f"missing fields: {', '.join(missing)}"
    try:
        average = np.asarray(leakage["avg_q_leak"])
        mu = np.asarray(leakage["mu"])
        chi = np.asarray(leakage["chi"])
        if average.shape != LEGACY_LEAKAGE_SHAPE:
            return False, (
                f"avg_q_leak shape {average.shape} != "
                f"{LEGACY_LEAKAGE_SHAPE}"
            )
        if mu.shape != (LEGACY_LEAKAGE_SHAPE[0],):
            return False, f"mu shape {mu.shape} is inconsistent with scan"
        if chi.shape != (LEGACY_LEAKAGE_SHAPE[1],):
            return False, f"chi shape {chi.shape} is inconsistent with scan"
        if average.shape != (mu.size, chi.size):
            return False, "avg_q_leak, mu, and chi shapes are inconsistent"
        if not np.allclose(mu, LEGACY_MU_GRID, rtol=0.0, atol=1.0e-14):
            return False, "mu grid does not match the manuscript scan"
        if not np.allclose(chi, LEGACY_CHI_GRID, rtol=0.0, atol=1.0e-14):
            return False, "chi grid does not match the manuscript scan"
        if not (
            np.all(np.isfinite(average))
            and np.all(np.isfinite(mu))
            and np.all(np.isfinite(chi))
        ):
            return False, "leakage arrays contain NaN or infinity"
        if not (np.all(np.diff(mu) > 0.0) and np.all(np.diff(chi) > 0.0)):
            return False, "mu and chi grids must be strictly increasing"
        if not _float_matches(_cache_scalar(leakage, "J"), LEGACY_LEAKAGE_J):
            return False, f"J must be {LEGACY_LEAKAGE_J}"
        if int(_cache_scalar(leakage, "N")) != LEGACY_LEAKAGE_N:
            return False, f"N must be {LEGACY_LEAKAGE_N}"
        if int(_cache_scalar(leakage, "K")) != LEGACY_LEAKAGE_K:
            return False, f"K must be {LEGACY_LEAKAGE_K}"
        if not _float_matches(
            _cache_scalar(leakage, "t_start"), LEGACY_LEAKAGE_T_START
        ):
            return False, f"t_start must be {LEGACY_LEAKAGE_T_START}"
        if not _float_matches(
            _cache_scalar(leakage, "t_stop"), LEGACY_LEAKAGE_T_STOP
        ):
            return False, f"t_stop must be {LEGACY_LEAKAGE_T_STOP}"
        if (
            int(_cache_scalar(leakage, "sample_count"))
            != LEGACY_LEAKAGE_SAMPLE_COUNT
        ):
            return False, (
                f"sample_count must be {LEGACY_LEAKAGE_SAMPLE_COUNT}"
            )
        source_name = str(_cache_scalar(leakage, "source_name"))
        if source_name not in {
            LEGACY_LEAKAGE_SOURCE,
            GENERATED_LEAKAGE_SOURCE,
        }:
            return False, (
                f"unrecognized source_name {source_name!r}"
            )
        if source_name == GENERATED_LEAKAGE_SOURCE:
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
                generated_fields.difference(leakage.files)
            )
            if generated_missing:
                return False, (
                    "generated cache is missing protocol fields: "
                    + ", ".join(generated_missing)
                )
            if (
                int(_cache_scalar(leakage, "schema_version"))
                != LEAKAGE_CACHE_SCHEMA_VERSION
            ):
                return False, "generated leakage schema version is outdated"
            generated_protocol = (
                ("pole_bias", POLE_BIAS),
                ("initial_phi", INITIAL_PHI),
                ("rtol", LEAKAGE_RTOL),
                ("atol", LEAKAGE_ATOL),
                ("max_step", LEAKAGE_MAX_STEP),
            )
            for key, expected in generated_protocol:
                if not _float_matches(_cache_scalar(leakage, key), expected):
                    return False, f"{key} does not match the leakage protocol"
            if str(_cache_scalar(leakage, "solver")) != LEAKAGE_METHOD:
                return False, "solver does not match the leakage protocol"
    except (TypeError, ValueError, OverflowError) as exc:
        return False, str(exc)
    return True, "ok"


def validate_marked_leakage_points(
    leakage_cache: Path, output_dir: Path
) -> None:
    """Reintegrate the six marked grid points with the strict TDVP protocol.

    The full leakage landscape uses a lower-tolerance exploratory scan.  This
    inexpensive check prevents that numerical tolerance from silently
    changing the broad residual classification used in the manuscript.
    """

    with np.load(leakage_cache) as leakage:
        leakage_ok, leakage_reason = validate_leakage_cache(leakage)
        if not leakage_ok:
            raise ValueError(
                f"Leakage cache {leakage_cache} is incompatible: "
                f"{leakage_reason}"
            )
        mu_grid = np.asarray(leakage["mu"])
        chi_grid = np.asarray(leakage["chi"])
        cached_average = np.asarray(leakage["avg_q_leak"])
        source_name = str(_cache_scalar(leakage, "source_name"))
        if source_name == GENERATED_LEAKAGE_SOURCE:
            landscape_protocol = {
                "source": source_name,
                "time_sampling": (
                    "250 uniform left endpoints over 0 <= t < 10"
                ),
                "solver": str(_cache_scalar(leakage, "solver")),
                "rtol": float(_cache_scalar(leakage, "rtol")),
                "atol": float(_cache_scalar(leakage, "atol")),
            }
        else:
            landscape_protocol = {
                "source": source_name,
                "time_sampling": (
                    "250 uniform left endpoints over 0 <= t < 10"
                ),
                "solver": "RK45",
                "rtol": 1.0e-3,
                "atol": 1.0e-5,
            }

        times = np.arange(0.0, CORE_TMAX, 0.04)
        rows = []
        for marker_mu, marker_chi, label in POINTS:
            mu_index = int(np.argmin(np.abs(mu_grid - marker_mu)))
            chi_index = int(np.argmin(np.abs(chi_grid - marker_chi)))
            grid_mu = float(mu_grid[mu_index])
            grid_chi = float(chi_grid[chi_index])
            trajectory = integrate_tdvp(
                grid_mu,
                grid_chi,
                times,
                period=TDVP_PERIOD,
                bias=POLE_BIAS,
                initial_phi=INITIAL_PHI,
                method=TDVP_METHOD,
                rtol=SOLVER_RTOL,
                atol=SOLVER_ATOL,
                max_step=SOLVER_MAX_STEP,
            )
            strict_average = float(np.mean(tdvpfun.get_qleak(trajectory)))
            cached_value = float(cached_average[mu_index, chi_index])
            rows.append(
                {
                    "point": label,
                    "marker": [float(marker_mu), float(marker_chi)],
                    "nearest_grid_point": [grid_mu, grid_chi],
                    "cached_average_leakage": cached_value,
                    "strict_average_leakage": strict_average,
                    "difference": strict_average - cached_value,
                }
            )

    maximum_delta = max(abs(row["difference"]) for row in rows)
    if maximum_delta > LEAKAGE_VALIDATION_MAX_DELTA:
        raise RuntimeError(
            "The leakage values fail the marked-point convergence "
            f"check: max |delta|={maximum_delta:.6g}."
        )
    payload = {
        "schema_version": LEAKAGE_VALIDATION_SCHEMA_VERSION,
        "description": (
            "Strict reintegration of the nearest leakage-grid points to P1--P6"
        ),
        "landscape_scan": landscape_protocol,
        "strict_protocol": {
            "equation": "tdvpfun.eom",
            "K": TDVP_PERIOD,
            "pole_bias": POLE_BIAS,
            "initial_phi": INITIAL_PHI,
            "solver": TDVP_METHOD,
            "rtol": SOLVER_RTOL,
            "atol": SOLVER_ATOL,
            "max_step": SOLVER_MAX_STEP,
        },
        "maximum_absolute_difference": maximum_delta,
        "acceptance_threshold": LEAKAGE_VALIDATION_MAX_DELTA,
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "leakage_point_convergence.json").open("w") as handle:
        json.dump(payload, handle, indent=2)


def validate_core_cache(
    core: np.lib.npyio.NpzFile,
) -> tuple[bool, str]:
    """Check the full ED/TDVP comparison protocol and all core array shapes."""

    required = {
        "schema_version",
        "times",
        "points",
        "labels",
        "tdvp_profiles",
        "tdvp_reference_profiles",
        "ed_profiles",
        "ed_reference_profiles",
        "N_ed",
        "K_tdvp",
        "pole_bias",
        "initial_phi",
        "response_reference",
        "tdvp_reference_period",
        "tdvp_eom",
        "tdvp_method",
        "tdvp_rtol",
        "tdvp_atol",
        "tdvp_max_step",
        "ed_solver_name",
        "ed_rtol",
        "ed_atol",
        "ed_max_step",
    }
    missing = sorted(required.difference(core.files))
    if missing:
        return False, f"missing fields: {', '.join(missing)}"
    try:
        if (
            int(_cache_scalar(core, "schema_version"))
            != CORE_CACHE_SCHEMA_VERSION
        ):
            return False, (
                f"schema_version must be {CORE_CACHE_SCHEMA_VERSION}"
            )
        if int(_cache_scalar(core, "N_ed")) != ED_LENGTH:
            return False, f"N_ed must be {ED_LENGTH}"
        if int(_cache_scalar(core, "K_tdvp")) != TDVP_PERIOD:
            return False, f"K_tdvp must be {TDVP_PERIOD}"
        if not _float_matches(_cache_scalar(core, "pole_bias"), POLE_BIAS):
            return False, f"pole_bias must be {POLE_BIAS}"
        if not _float_matches(
            _cache_scalar(core, "initial_phi"), INITIAL_PHI
        ):
            return False, f"initial_phi must be {INITIAL_PHI}"
        if (
            int(_cache_scalar(core, "tdvp_reference_period"))
            != TDVP_REFERENCE_PERIOD
        ):
            return False, (
                f"tdvp_reference_period must be {TDVP_REFERENCE_PERIOD}"
            )
        if (
            str(_cache_scalar(core, "response_reference"))
            != "separately_evolved_unperturbed_Z2"
        ):
            return False, "unexpected response_reference"
        if str(_cache_scalar(core, "tdvp_eom")) != "tdvpfun.eom":
            return False, "TDVP equation of motion is not tdvpfun.eom"
        if str(_cache_scalar(core, "tdvp_method")) != TDVP_METHOD:
            return False, f"tdvp_method must be {TDVP_METHOD}"
        if str(_cache_scalar(core, "ed_solver_name")) != ED_SOLVER_NAME:
            return False, f"ed_solver_name must be {ED_SOLVER_NAME}"
        for key, expected in (
            ("tdvp_rtol", SOLVER_RTOL),
            ("tdvp_atol", SOLVER_ATOL),
            ("tdvp_max_step", SOLVER_MAX_STEP),
            ("ed_rtol", SOLVER_RTOL),
            ("ed_atol", SOLVER_ATOL),
            ("ed_max_step", SOLVER_MAX_STEP),
        ):
            if not _float_matches(_cache_scalar(core, key), expected):
                return False, f"{key} must be {expected}"

        expected_times = core_times()
        times = np.asarray(core["times"])
        points = np.asarray(core["points"])
        labels = np.asarray(core["labels"])
        if times.shape != expected_times.shape or not np.allclose(
            times, expected_times, rtol=0.0, atol=1.0e-14
        ):
            return False, "times do not match the common [0,10] grid"
        if points.shape != comparison_points().shape or not np.allclose(
            points, comparison_points(), rtol=0.0, atol=1.0e-15
        ):
            return False, "points do not match P1--P6"
        if not np.array_equal(labels, comparison_labels()):
            return False, "labels do not match P1--P6"

        point_count = len(POINTS)
        time_count = expected_times.size
        expected_shapes = {
            "tdvp_profiles": (point_count, TDVP_PERIOD, time_count),
            "tdvp_reference_profiles": (
                point_count,
                TDVP_PERIOD,
                time_count,
            ),
            "ed_profiles": (point_count, ED_LENGTH, time_count),
            "ed_reference_profiles": (
                point_count,
                ED_LENGTH,
                time_count,
            ),
        }
        for key, expected_shape in expected_shapes.items():
            values = np.asarray(core[key])
            if values.shape != expected_shape:
                return False, f"{key} shape {values.shape} != {expected_shape}"
            if not np.all(np.isfinite(values)):
                return False, f"{key} contains NaN or infinity"
        if not (
            np.all(np.isfinite(times)) and np.all(np.isfinite(points))
        ):
            return False, "times or points contain NaN or infinity"
    except (TypeError, ValueError, OverflowError) as exc:
        return False, str(exc)
    return True, "ok"


def expected_phase_times(tmax: float, dt: float) -> np.ndarray:
    """Return the requested inclusive sampling grid for Fig. 2."""

    return np.arange(0.0, tmax + 0.5 * dt, dt)


def validate_phase_cache(
    phase: np.lib.npyio.NpzFile, *, tmax: float, dt: float
) -> tuple[bool, str]:
    """Check phase-cache points, protocol metadata, shapes, and finiteness."""

    required = {
        "schema_version",
        "times",
        "points",
        "point_indices",
        "labels",
        "local_sites",
        "theta",
        "phi",
        "defect_site",
        "K_tdvp",
        "pole_bias",
        "initial_phi",
        "tdvp_eom",
        "tdvp_method",
        "tdvp_rtol",
        "tdvp_atol",
        "tdvp_max_step",
    }
    missing = sorted(required.difference(phase.files))
    if missing:
        return False, f"missing fields: {', '.join(missing)}"
    try:
        if (
            int(_cache_scalar(phase, "schema_version"))
            != PHASE_CACHE_SCHEMA_VERSION
        ):
            return False, (
                f"schema_version must be {PHASE_CACHE_SCHEMA_VERSION}"
            )
        if int(_cache_scalar(phase, "K_tdvp")) != TDVP_PERIOD:
            return False, f"K_tdvp must be {TDVP_PERIOD}"
        if not _float_matches(_cache_scalar(phase, "pole_bias"), POLE_BIAS):
            return False, f"pole_bias must be {POLE_BIAS}"
        if not _float_matches(
            _cache_scalar(phase, "initial_phi"), INITIAL_PHI
        ):
            return False, f"initial_phi must be {INITIAL_PHI}"
        if str(_cache_scalar(phase, "tdvp_eom")) != "tdvpfun.eom":
            return False, "TDVP equation of motion is not tdvpfun.eom"
        if str(_cache_scalar(phase, "tdvp_method")) != TDVP_METHOD:
            return False, f"tdvp_method must be {TDVP_METHOD}"
        for key, expected in (
            ("tdvp_rtol", SOLVER_RTOL),
            ("tdvp_atol", SOLVER_ATOL),
            ("tdvp_max_step", SOLVER_MAX_STEP),
        ):
            if not _float_matches(_cache_scalar(phase, key), expected):
                return False, f"{key} must be {expected}"

        times = np.asarray(phase["times"])
        expected_times = expected_phase_times(tmax, dt)
        points = np.asarray(phase["points"])
        indices = np.asarray(phase["point_indices"])
        labels = np.asarray(phase["labels"])
        local_sites = np.asarray(phase["local_sites"])
        expected_i0 = defect_site(TDVP_PERIOD)
        expected_sites = expected_i0 + np.asarray([-1, 0, 1])
        if times.shape != expected_times.shape or not np.allclose(
            times, expected_times, rtol=0.0, atol=1.0e-14
        ):
            return False, "times do not match the requested phase grid"
        if points.shape != comparison_points().shape or not np.allclose(
            points, comparison_points(), rtol=0.0, atol=1.0e-15
        ):
            return False, "points do not match P1--P6"
        if not np.array_equal(indices, np.arange(len(POINTS), dtype=int)):
            return False, "point_indices do not match P1--P6"
        if not np.array_equal(labels, comparison_labels()):
            return False, "labels do not match P1--P6"
        if int(_cache_scalar(phase, "defect_site")) != expected_i0:
            return False, f"defect_site must be {expected_i0}"
        if not np.array_equal(local_sites, expected_sites):
            return False, f"local_sites must be {expected_sites.tolist()}"

        expected_shape = (len(POINTS), 3, expected_times.size)
        for key in ("theta", "phi"):
            values = np.asarray(phase[key])
            if values.shape != expected_shape:
                return False, f"{key} shape {values.shape} != {expected_shape}"
            if not np.all(np.isfinite(values)):
                return False, f"{key} contains NaN or infinity"
        if not (
            np.all(np.isfinite(times)) and np.all(np.isfinite(points))
        ):
            return False, "times or points contain NaN or infinity"
    except (TypeError, ValueError, OverflowError) as exc:
        return False, str(exc)
    return True, "ok"


def plot_fig1(leakage_cache: Path, core_cache: Path, output_dir: Path) -> None:
    """Plot the leakage landscape and the four representative profiles."""

    leakage = np.load(leakage_cache)
    core = np.load(core_cache)
    leakage_ok, leakage_reason = validate_leakage_cache(leakage)
    if not leakage_ok:
        raise ValueError(
            f"Leakage cache {leakage_cache} is incompatible: "
            f"{leakage_reason}"
        )
    core_ok, core_reason = validate_core_cache(core)
    if not core_ok:
        raise ValueError(
            f"Core cache {core_cache} is incompatible: {core_reason}"
        )
    average = leakage["avg_q_leak"]
    mu = leakage["mu"]
    chi = leakage["chi"]
    times = core["times"]
    ed_profiles = core["ed_profiles"]
    tdvp_profiles = core["tdvp_profiles"]
    ed_reference_profiles = core["ed_reference_profiles"]
    tdvp_reference_profiles = core["tdvp_reference_profiles"]
    n_ed = int(core["N_ed"])
    k_tdvp = int(core["K_tdvp"])
    i_ed = defect_site(n_ed)
    i_tdvp = defect_site(k_tdvp)

    dmu = mu[1] - mu[0]
    dchi = chi[1] - chi[0]
    extent = [
        mu.min() - dmu / 2,
        mu.max() + dmu / 2,
        chi.min() - dchi / 2,
        chi.max() + dchi / 2,
    ]

    # ------------------------------------------------------------------
    # Figure geometry.  These are the main manual-adjustment parameters.
    # The width is the standard PRB two-column width.  ``bottom`` controls
    # the space below the x ticks, while ``top`` leaves room for panel (a)'s
    # horizontal colorbar and the method headings.  The two small columns after
    # panel (a) keep the
    # shared t label visually separated from the right edge of panel (a).
    # Grid column 5 is the explicit gap between panels (c) and (d); reduce its
    # weight if those two panels should sit closer together.
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(7.15, 2.45))
    grid = fig.add_gridspec(
        1,
        8,
        left=0.065,
        right=0.975,
        bottom=0.18,
        top=0.78,
        width_ratios=(1.50, 0.08, 0.10, 0.80, 0.80, 0.06, 0.80, 0.80),
        wspace=0.045,
    )
    leakage_axis = fig.add_subplot(grid[0, 0])
    profile_axes = [
        fig.add_subplot(grid[0, 3]),
        fig.add_subplot(grid[0, 4]),
        fig.add_subplot(grid[0, 6]),
        fig.add_subplot(grid[0, 7]),
    ]

    # Panel-(a) colorbar coordinates are fractions of panel (a):
    # [left, bottom, width, height].  The bar occupies 90% of the panel
    # width; the remaining space is reserved for the Gamma symbol on its right.
    # Gamma is allowed to extend just beyond the right axis edge so it stays
    # clear of the final numerical tick.
    # The second entry controls the vertical separation.  With a square panel,
    # 1.04 leaves only a narrow gap above the top axis for a tight layout.
    leakage_bar_box = [0.0, 1.04, 0.90, 0.055]
    leakage_colorbar_axis = leakage_axis.inset_axes(
        leakage_bar_box
    )

    # The shared response colorbar is an inset of panel (e), so its position
    # follows that panel automatically when the GridSpec is adjusted.  Bounds
    # are [left, bottom, width, height] in panel-(e) coordinates: the small
    # left offset leaves a readable gap, and height=1 makes both vertical
    # extents identical.  These two values are the main manual spacing/width
    # controls.
    profile_bar_gap = 0.060
    profile_bar_width = 0.055
    profile_colorbar_axis = profile_axes[-1].inset_axes(
        [1.0 + profile_bar_gap, 0.0, profile_bar_width, 1.0]
    )

    leakage_image = leakage_axis.imshow(
        average.T,
        origin="lower",
        # ``set_box_aspect`` below fixes the panel's physical shape, so use
        # ``auto`` here to let the heat map fill that box.
        aspect="auto",
        extent=extent,
        # Use the same palette as panels (b)--(e), as requested.
        cmap="inferno",
        interpolation="nearest",
        vmin=0.0,
        vmax=0.1,
        rasterized=True,
    )
    # Right-align the square axes inside its wider GridSpec cell.  This removes
    # unused space between panels (a) and (b) without changing either panel.
    leakage_axis.set_anchor("E")
    # Matplotlib defines box aspect as axes height / axes width.  A value of
    # 1.0 therefore gives the physical x and y axes exactly the same length.
    # This controls the box shape independently of the plotted data ranges.
    leakage_axis.set_box_aspect(1.0)

    # Larger labels and tick numbers keep panel (a) readable after the full
    # figure is reduced to journal size.  A negative x-label pad moves mu
    # upward, closer to the axis, and makes the whole figure more compact.
    leakage_axis.set_xlabel(r"$\mu$", fontsize=10.5, labelpad=-1.5)
    leakage_axis.set_ylabel(r"$\chi$", fontsize=10.5, labelpad=4.0)
    leakage_axis.set_xticks([-1, 0, 1])
    leakage_axis.set_yticks([0, 0.5, 1, 1.5, 2])
    panel_text = leakage_axis.text(
        0.04,
        0.96,
        "(a)",
        transform=leakage_axis.transAxes,
        color="white",
        fontsize=8.2,
        weight="semibold",
        va="top",
        zorder=4,
    )

    # Offsets are (delta-mu, delta-chi, horizontal alignment).  Keeping them
    # in one dictionary makes manual label adjustment straightforward.
    point_label_offsets = {
        "P1": (0.075, -0.025, "left"),
        "P2": (-0.075, 0.045, "right"),
        "P3": (0.070, 0.000, "left"),
        "P4": (0.070, 0.000, "left"),
        "P5": (0.070, -0.020, "left"),
        "P6": (0.070, 0.022, "left"),
    }
    for mu_value, chi_value, label in POINTS:
        selected = label in {"P1", "P6"}
        leakage_axis.plot(
            mu_value,
            chi_value,
            "o",
            ms=3.4 if selected else 3.4, # use 4.1 if you want to emphasize P1,P6
            mfc="white",
            mec="0.15",
            mew=0.6,
            zorder=3,
        )
        dx, dy, horizontal_alignment = point_label_offsets[label]
        text = leakage_axis.text(
            mu_value + dx,
            chi_value + dy,
            label,
            color="white",
            fontsize=6.8,
            weight="semibold" if selected else "semibold", # you can choose "normal"
            ha=horizontal_alignment,
            va="center",
            zorder=4,
        )
        text.set_path_effects([pe.withStroke(linewidth=0.8, foreground="black")])

    leakage_colorbar = fig.colorbar(
        leakage_image,
        cax=leakage_colorbar_axis,
        ticks=(0.0, 0.05, 0.1),
        orientation="horizontal",
    )
    # Put the numerical scale above the bar.  Gamma is a separate text object
    # on the right, vertically centered with the bar rather than above it.
    leakage_colorbar.ax.xaxis.set_ticks_position("top")
    leakage_colorbar.ax.tick_params(
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False,
        labelsize=7.5,
        length=2.5,
        width=0.6,
        pad=1.0,
    )
    leakage_colorbar.outline.set_linewidth(0.6)
    leakage_axis.text(
        1.05,
        leakage_bar_box[1] + 0.5 * leakage_bar_box[3],
        r"$\bar{\Gamma}$",
        transform=leakage_axis.transAxes,
        ha="right",
        va="center",
        fontsize=9.5,
        clip_on=False,
    )

    dt = times[1] - times[0]
    profile_extent = [
        -i_ed - 0.5,
        n_ed - 1 - i_ed + 0.5,
        times[0] - dt / 2,
        times[-1] + dt / 2,
    ]
    crop_start = i_tdvp - i_ed
    crop_end = crop_start + n_ed
    profile_specs = (
        (0, False, "P1", "ED"),
        (0, True, "P1", "TDVP"),
        (5, False, "P6", "ED"),
        (5, True, "P6", "TDVP"),
    )
    profile_image = None
    for panel_index, (axis, spec) in enumerate(
        zip(profile_axes, profile_specs), start=1
    ):
        point_index, is_tdvp, point_label, method_label = spec
        if is_tdvp:
            probability = normalized_profile(
                tdvp_profiles[point_index],
                tdvp_reference_profiles[point_index],
            )[crop_start:crop_end]
        else:
            probability = normalized_profile(
                ed_profiles[point_index],
                ed_reference_profiles[point_index],
            )
        profile_image = axis.imshow(
            probability.T,
            origin="lower",
            aspect="auto",
            extent=profile_extent,
            cmap="inferno",
            interpolation="nearest",
            vmin=0.0,
            vmax=0.5,
            rasterized=True,
        )
        axis.set_xticks((-8, 0, 8))
        axis.set_yticks((0, 5, 10))
        label_text = axis.text(
            0.02,
            0.93,
            f"({chr(ord('a') + panel_index)})",
            transform=axis.transAxes,
            color="white",
            fontsize=8.2,
            weight="semibold",
            va="top",
        )
        # The confinement point is written inside every response panel in
        # white.  The upper-right corner is dark for all four trajectories.
        axis.text(
            0.95,
            0.93,
            point_label,
            transform=axis.transAxes,
            color="white",
            fontsize=7.8,
            weight="semibold",
            ha="right",
            va="top",
        )
        # A slightly larger pad lifts the ED/TDVP headings away from the axes.
        axis.set_title(method_label, fontsize=7.6, pad=4.0, weight="semibold")
        if panel_index > 1:
            axis.tick_params(labelleft=False)

    # Shared response-axis labels.  The figure-coordinate y value controls the
    # vertical position of i-i0; 0.055 leaves a little more space below the
    # tick labels.  The y label belongs to panel (b) and uses a negative pad so
    # that t stays close to the y axis.
    p1_left = profile_axes[0].get_position().x0
    p6_right = profile_axes[3].get_position().x1
    fig.text(
        0.5 * (p1_left + p6_right),
        0.055,
        r"$i-i_0$",
        ha="center",
        va="bottom",
        fontsize=12,
    )
    # A slightly negative pad pulls t toward the y tick labels while the two
    # spacer columns above preserve a visible gap from panel (a).
    profile_axes[0].set_ylabel(r"$t$", fontsize=12, labelpad=-4.5)

    assert profile_image is not None
    profile_colorbar = fig.colorbar(
        profile_image,
        cax=profile_colorbar_axis,
        ticks=(0.0, 0.25, 0.5),
        orientation="vertical",
    )
    # Put the numerical scale on the right.  The labels remain horizontal and
    # use the requested compact decimal notation.
    profile_colorbar.ax.yaxis.set_ticks_position("right")
    profile_colorbar.set_ticks(
        (0.0, 0.25, 0.5),
        labels=(r"$0$", r"$0.25$", r"$0.5$"),
    )
    profile_colorbar.ax.tick_params(
        right=True,
        left=False,
        labelright=True,
        labelleft=False,
        labelsize=7.5,
        length=2.5,
        width=0.6,
        pad=1.0,
        labelrotation=0,
    )
    profile_colorbar.outline.set_linewidth(0.6)
    # A colorbar title uses the same vertical placement and font size as the
    # ED/TDVP headings, so P_i(t) sits directly above the full-height bar.
    profile_colorbar.ax.set_title(
        r"$\mathscr{P}_i(t)$",
        fontsize=7.6,
        pad=4.0,
        weight="semibold",
    )

    # Final axis styling: outward ticks only on the bottom and left, with
    # light spines.  Panel (a) uses larger tick labels than the compact
    # response panels.  These two label sizes are useful manual tuning knobs.
    leakage_axis.tick_params(
        direction="out",
        top=False,
        right=False,
        length=2.8,
        width=0.65,
        pad=2,
        labelsize=8.5,
    )
    for axis in profile_axes:
        axis.tick_params(
            direction="out",
            top=False,
            right=False,
            length=2.5,
            width=0.6,
            pad=2,
            labelsize=8.5,
        )
    for axis in [leakage_axis, *profile_axes]:
        for spine in axis.spines.values():
            spine.set_linewidth(0.65)
        axis.xaxis.set_ticks_position("bottom")
        axis.yaxis.set_ticks_position("left")
    for axis in profile_axes[1:]:
        axis.tick_params(labelleft=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "fig1_leakage_and_profiles.pdf",bbox_inches='tight')
    fig.savefig(output_dir / "fig1_leakage_and_profiles.png",bbox_inches='tight')
    plt.close(fig)
    write_width_table(core, output_dir)


def integrate_tdvp(
    mu: float,
    chi: float,
    times: np.ndarray,
    *,
    period: int = TDVP_PERIOD,
    bias: float = POLE_BIAS,
    initial_phi: float = INITIAL_PHI,
    method: str = TDVP_METHOD,
    rtol: float = SOLVER_RTOL,
    atol: float = SOLVER_ATOL,
    max_step: float = SOLVER_MAX_STEP,
    with_defect: bool = True,
) -> np.ndarray:
    """Integrate the exact spin-1/2 finite-period TDVP equations."""

    theta0, phi0 = initial_angles(
        period,
        bias,
        initial_phi=initial_phi,
        with_defect=with_defect,
    )
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
    if not solution.success or solution.y.shape[1] != len(times):
        raise RuntimeError(
            f"TDVP integration failed for (mu, chi)=({mu}, {chi}): "
            f"{solution.message}"
        )
    return solution.y


def tdvp_translation_invariant(trajectory: np.ndarray) -> np.ndarray:
    """Return the bond-smoothed occupation profile for a TDVP trajectory."""

    period = trajectory.shape[0] // 2
    theta = trajectory[:period]
    eta = tdvpfun.get_eta(theta)
    magnetization = -1.0 + eta * (1.0 - np.cos(theta))
    occupation = 0.5 * (1.0 + magnetization)
    return occupation + np.roll(occupation, -1, axis=0)


def ed_translation_invariant_pair(
    mu: float,
    chi: float,
    times: np.ndarray,
    *,
    length: int = ED_LENGTH,
    bias: float = POLE_BIAS,
    initial_phi: float = INITIAL_PHI,
    solver_name: str = ED_SOLVER_NAME,
    rtol: float = SOLVER_RTOL,
    atol: float = SOLVER_ATOL,
    max_step: float = SOLVER_MAX_STEP,
) -> tuple[np.ndarray, np.ndarray]:
    """Return defect and unperturbed ED profiles using state generators.

    QuSpin's Schrödinger evolution is explicitly configured to use the same
    DOP853 tolerances and maximum step as the TDVP integration.
    """

    from quspin.operators import hamiltonian
    from quspin.tools.measurements import obs_vs_time

    spin = 0.5
    basis = pxpbasisS.constrained_basis(2, length, None, None)
    delta = np.asarray(
        [2.0 * mu - chi, 2.0 * mu + chi] * (length // 2), dtype=float
    )
    no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)
    static = [
        ["+", [[1.0 / (2.0 * spin), site] for site in range(length)]],
        ["-", [[1.0 / (2.0 * spin), site] for site in range(length)]],
        ["z", [[delta[site] / spin, site] for site in range(length)]],
    ]
    hamiltonian_ed = hamiltonian(static, [], basis=basis, **no_checks)
    z_operators = {
        f"z{site}": hamiltonian(
            [["z", [[1.0 / spin, site]]]],
            [],
            basis=basis,
            dtype=np.float64,
            **no_checks,
        )
        for site in range(length)
    }
    profiles = []
    for with_defect in (True, False):
        theta0, phi0 = initial_angles(
            length,
            bias,
            initial_phi=initial_phi,
            with_defect=with_defect,
        )
        state0 = EDfun.mpsmanifold(theta0, phi0, basis).ravel()
        state_generator = hamiltonian_ed.evolve(
            state0,
            float(times[0]),
            times,
            solver_name=solver_name,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            iterate=True,
        )
        values = obs_vs_time(state_generator, times, z_operators)
        magnetization = np.vstack(
            [np.asarray(values[f"z{site}"]).real for site in range(length)]
        )
        occupation = 0.5 * (1.0 + magnetization)
        profiles.append(occupation + np.roll(occupation, -1, axis=0))
    return profiles[0], profiles[1]


def generate_fig2_core(cache: Path) -> None:
    times = core_times()
    tdvp_period = TDVP_PERIOD
    ed_length = ED_LENGTH
    tdvp_profiles = []
    tdvp_reference_profiles = []
    ed_profiles = []
    ed_reference_profiles = []
    for mu, chi, label in POINTS:
        start = time.perf_counter()
        trajectory = integrate_tdvp(
            mu,
            chi,
            times,
            period=tdvp_period,
            bias=POLE_BIAS,
            initial_phi=INITIAL_PHI,
            method=TDVP_METHOD,
            rtol=SOLVER_RTOL,
            atol=SOLVER_ATOL,
            max_step=SOLVER_MAX_STEP,
        )
        tdvp_profiles.append(tdvp_translation_invariant(trajectory))
        # The unperturbed Z2 state remains exactly in its period-two symmetry
        # sector.  Evolving that minimal cell and tiling its observable profile
        # is numerically more stable than integrating 100 duplicate variables,
        # while representing the same translation-periodic reference.
        reference_trajectory = integrate_tdvp(
            mu,
            chi,
            times,
            period=TDVP_REFERENCE_PERIOD,
            bias=POLE_BIAS,
            initial_phi=INITIAL_PHI,
            method=TDVP_METHOD,
            rtol=SOLVER_RTOL,
            atol=SOLVER_ATOL,
            max_step=SOLVER_MAX_STEP,
            with_defect=False,
        )
        reference_profile = tdvp_translation_invariant(reference_trajectory)
        tdvp_reference_profiles.append(
            np.tile(
                reference_profile,
                (tdvp_period // TDVP_REFERENCE_PERIOD, 1),
            )
        )
        print(
            f"{label}: TDVP defect + reference "
            f"{time.perf_counter() - start:.2f} s",
            flush=True,
        )

        start = time.perf_counter()
        ed_profile, ed_reference = ed_translation_invariant_pair(
            mu,
            chi,
            times,
            length=ed_length,
            bias=POLE_BIAS,
            initial_phi=INITIAL_PHI,
            solver_name=ED_SOLVER_NAME,
            rtol=SOLVER_RTOL,
            atol=SOLVER_ATOL,
            max_step=SOLVER_MAX_STEP,
        )
        ed_profiles.append(ed_profile)
        ed_reference_profiles.append(ed_reference)
        print(
            f"{label}: ED defect + reference "
            f"{time.perf_counter() - start:.2f} s",
            flush=True,
        )

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        schema_version=CORE_CACHE_SCHEMA_VERSION,
        times=times,
        points=comparison_points(),
        labels=comparison_labels(),
        tdvp_profiles=np.asarray(tdvp_profiles),
        tdvp_reference_profiles=np.asarray(tdvp_reference_profiles),
        ed_profiles=np.asarray(ed_profiles),
        ed_reference_profiles=np.asarray(ed_reference_profiles),
        N_ed=ed_length,
        K_tdvp=tdvp_period,
        pole_bias=POLE_BIAS,
        initial_phi=INITIAL_PHI,
        response_reference="separately_evolved_unperturbed_Z2",
        tdvp_reference_period=TDVP_REFERENCE_PERIOD,
        tdvp_eom="tdvpfun.eom",
        tdvp_method=TDVP_METHOD,
        tdvp_rtol=SOLVER_RTOL,
        tdvp_atol=SOLVER_ATOL,
        tdvp_max_step=SOLVER_MAX_STEP,
        ed_solver_name=ED_SOLVER_NAME,
        ed_rtol=SOLVER_RTOL,
        ed_atol=SOLVER_ATOL,
        ed_max_step=SOLVER_MAX_STEP,
    )


def generate_phase_cache(
    cache: Path, *, tmax: float = 10.0, dt: float = 0.01
) -> None:
    if dt <= 0.0 or tmax <= dt:
        raise ValueError("Require dt > 0 and tmax > dt for the phase portrait.")
    times = expected_phase_times(tmax, dt)
    period = TDVP_PERIOD
    i0 = defect_site(period)
    local_sites = i0 + np.asarray([-1, 0, 1], dtype=int)
    local_theta = []
    local_phi = []
    for point_index, (mu, chi, label) in enumerate(POINTS):
        start = time.perf_counter()
        trajectory = integrate_tdvp(
            mu,
            chi,
            times,
            method=TDVP_METHOD,
            rtol=SOLVER_RTOL,
            atol=SOLVER_ATOL,
            max_step=SOLVER_MAX_STEP,
            period=period,
            bias=POLE_BIAS,
            initial_phi=INITIAL_PHI,
        )
        local_theta.append(trajectory[local_sites])
        local_phi.append(trajectory[period + local_sites])
        print(
            f"{label}: phase trajectory {time.perf_counter() - start:.2f} s",
            flush=True,
        )

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        schema_version=PHASE_CACHE_SCHEMA_VERSION,
        times=times,
        points=comparison_points(),
        point_indices=np.arange(len(POINTS), dtype=int),
        labels=comparison_labels(),
        local_sites=local_sites,
        theta=np.asarray(local_theta),
        phi=np.asarray(local_phi),
        defect_site=i0,
        K_tdvp=period,
        pole_bias=POLE_BIAS,
        initial_phi=INITIAL_PHI,
        tdvp_eom="tdvpfun.eom",
        tdvp_method=TDVP_METHOD,
        tdvp_rtol=SOLVER_RTOL,
        tdvp_atol=SOLVER_ATOL,
        tdvp_max_step=SOLVER_MAX_STEP,
    )


def normalized_profile(
    profile: np.ndarray, reference_profile: np.ndarray
) -> np.ndarray:
    """Normalize the defect response relative to the evolved Z2 reference."""

    if profile.shape != reference_profile.shape:
        raise ValueError("Profile and reference profile must have the same shape.")
    difference = np.abs(profile - reference_profile)
    denominator = difference.sum(axis=0, keepdims=True)
    return np.divide(
        difference,
        denominator,
        out=np.zeros_like(difference),
        where=denominator > 0.0,
    )


def mean_profile_width(
    profile: np.ndarray,
    reference_profile: np.ndarray,
    center: int,
    times: np.ndarray,
) -> float:
    """Time-average the RMS response width using trapezoidal quadrature."""

    difference = np.abs(profile - reference_profile)
    denominator = difference.sum(axis=0)
    probability = np.divide(
        difference,
        denominator[None, :],
        out=np.zeros_like(difference),
        where=denominator[None, :] > 0.0,
    )
    distance_squared = (np.arange(profile.shape[0]) - center) ** 2
    width = np.sqrt(np.sum(distance_squared[:, None] * probability, axis=0))
    valid = denominator > 0.0
    if np.count_nonzero(valid) < 2:
        return float("nan")
    valid_times = times[valid]
    duration = valid_times[-1] - valid_times[0]
    if duration <= 0.0:
        return float("nan")
    return float(np.trapezoid(width[valid], valid_times) / duration)


def width_table_rows(core: np.lib.npyio.NpzFile) -> list[dict[str, object]]:
    """Compute the ED and TDVP response widths for P1--P6."""

    times = core["times"]
    n_ed = int(core["N_ed"])
    k_tdvp = int(core["K_tdvp"])
    i_ed = defect_site(n_ed)
    i_tdvp = defect_site(k_tdvp)
    rows: list[dict[str, object]] = []
    for point_index, (mu, chi, label) in enumerate(POINTS):
        omega_ed = mean_profile_width(
            core["ed_profiles"][point_index],
            core["ed_reference_profiles"][point_index],
            i_ed,
            times,
        )
        omega_tdvp = mean_profile_width(
            core["tdvp_profiles"][point_index],
            core["tdvp_reference_profiles"][point_index],
            i_tdvp,
            times,
        )
        rows.append(
            {
                "point": label,
                "mu": float(mu),
                "chi": float(chi),
                "omega_ed": omega_ed,
                "omega_tdvp": omega_tdvp,
            }
        )
    return rows


def write_width_table(
    core: np.lib.npyio.NpzFile, output_dir: Path
) -> None:
    """Write the processed values used by the manuscript width table."""

    rows = width_table_rows(core)
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "point",
        "mu",
        "chi",
        "omega_ed",
        "omega_tdvp",
    ]
    with (output_dir / "defect_widths.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema_version": WIDTH_OUTPUT_SCHEMA_VERSION,
        "time_window": [float(core["times"][0]), float(core["times"][-1])],
        "response_reference": "separately evolved unperturbed Z2 state",
        "physical_width": "time-averaged RMS distance from the defect site",
        "protocol": {
            "points": comparison_points().tolist(),
            "labels": comparison_labels().tolist(),
            "time_sample_count": int(core["times"].size),
            "initial_state": {
                "pole_bias": float(_cache_scalar(core, "pole_bias")),
                "initial_phi": float(_cache_scalar(core, "initial_phi")),
            },
            "tdvp": {
                "equation": str(_cache_scalar(core, "tdvp_eom")),
                "K": int(_cache_scalar(core, "K_tdvp")),
                "solver": str(_cache_scalar(core, "tdvp_method")),
                "rtol": float(_cache_scalar(core, "tdvp_rtol")),
                "atol": float(_cache_scalar(core, "tdvp_atol")),
                "max_step": float(
                    _cache_scalar(core, "tdvp_max_step")
                ),
            },
            "ed": {
                "L": int(_cache_scalar(core, "N_ed")),
                "solver": str(_cache_scalar(core, "ed_solver_name")),
                "rtol": float(_cache_scalar(core, "ed_rtol")),
                "atol": float(_cache_scalar(core, "ed_atol")),
                "max_step": float(_cache_scalar(core, "ed_max_step")),
            },
            "unperturbed_tdvp_reference": {
                "period": int(
                    _cache_scalar(core, "tdvp_reference_period")
                ),
                "construction": (
                    "evolve in the period-2 symmetry sector, then tile "
                    "the observable profile to K"
                ),
            },
        },
        "rows": rows,
    }
    with (output_dir / "defect_widths.json").open("w") as handle:
        json.dump(payload, handle, indent=2)
    tex_lines = [
        "% Generated by reproduce_fig1_fig2.py; do not edit by hand.",
        *[
            (
                f"${row['point']}$ & {row['omega_ed']:.3f} "
                f"& {row['omega_tdvp']:.3f} \\\\"
            )
            for row in rows
        ],
    ]
    (output_dir / "defect_widths_table.tex").write_text(
        "\n".join(tex_lines) + "\n",
        encoding="utf-8",
    )


def plot_bloch_portrait(
    ax: plt.Axes,
    times: np.ndarray,
    theta_local: np.ndarray,
    phi_local: np.ndarray,
    label: str,
    color: str,
    panel_label: str,
) -> dict[str, float | int | list[float]]:
    """Plot one defect-site trajectory in nonsingular Bloch coordinates."""

    theta_defect = theta_local[1]
    phi_defect = phi_local[1]
    plot_stop = min(10.0, float(times[-1]))
    plot_mask = (times >= 0.0) & (times <= plot_stop)
    x_values = np.sin(theta_defect[plot_mask]) * np.cos(phi_defect[plot_mask])
    y_values = np.sin(theta_defect[plot_mask]) * np.sin(phi_defect[plot_mask])
    circle_angle = np.linspace(0.0, 2.0 * np.pi, 361)
    ax.plot(
        np.cos(circle_angle),
        np.sin(circle_angle),
        color="0.70",
        lw=0.55,
        zorder=0,
    )
    ax.axhline(0.0, color="0.86", lw=0.45, zorder=0)
    ax.axvline(0.0, color="0.86", lw=0.45, zorder=0)
    ax.plot(x_values, y_values, color=color, lw=0.72, alpha=0.92, zorder=2)
    ax.set(
        # The Bloch-plane boundary is exactly [-1, 1] on both axes.  Placing
        # the endpoint ticks on the frame avoids the former inner intersections
        # produced by the small +/-1.04 padding.
        xlim=(-1.0, 1.0),
        ylim=(-1.0, 1.0),
        xticks=(-1.0, 0.0, 1.0),
        yticks=(-1.0, 0.0, 1.0),
        aspect="equal",
    )
    # Use the same small inset from the upper corners in all six panels.
    # Decrease this value to move both labels still closer to the frame.
    corner_inset = 0.025
    ax.text(
        corner_inset,
        1.0 - corner_inset,
        panel_label,
        transform=ax.transAxes,
        va="top",
        weight="bold",
    )
    ax.text(
        1.0 - corner_inset,
        1.0 - corner_inset,
        label,
        transform=ax.transAxes,
        va="top",
        ha="right",
        weight="bold",
    )

    return {
        "horizontal_coordinate_span": float(np.ptp(x_values)),
        "vertical_coordinate_span": float(np.ptp(y_values)),
        "plot_time_window": [0.0, plot_stop],
    }


def plot_fig2(
    phase_cache: Path,
    output_dir: Path,
    *,
    tmax: float = CORE_TMAX,
    dt: float = 0.01,
) -> None:
    """Plot P1--P6 in a single row on a common Bloch-plane scale."""

    phase = np.load(phase_cache)
    phase_ok, phase_reason = validate_phase_cache(
        phase, tmax=tmax, dt=dt
    )
    if not phase_ok:
        raise ValueError(
            f"Phase cache {phase_cache} is incompatible: {phase_reason}"
        )
    point_indices = phase["point_indices"]
    expected_indices = np.arange(len(POINTS), dtype=int)
    if not np.array_equal(point_indices, expected_indices):
        raise ValueError(
            "Phase cache does not contain P1--P6 in the expected order; "
            "rebuild it with --recompute-phase."
        )

    fig, axes = plt.subplots(
        1,
        len(POINTS),
        figsize=(7.15, 1.62),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    phase_summary: dict[str, object] = {
        "description": "defect-site Bloch-plane portraits for P1--P6",
        "phase_coordinates": [
            "sin(theta_i0) cos(phi_i0)",
            "sin(theta_i0) sin(phi_i0)",
        ],
        "plot_time_window": [0.0, min(10.0, float(phase["times"][-1]))],
        "zero_indexed_sites": phase["local_sites"].tolist(),
        "protocol": {
            "points": phase["points"].tolist(),
            "K": int(_cache_scalar(phase, "K_tdvp")),
            "pole_bias": float(_cache_scalar(phase, "pole_bias")),
            "initial_phi": float(_cache_scalar(phase, "initial_phi")),
            "equation": str(_cache_scalar(phase, "tdvp_eom")),
            "solver": str(_cache_scalar(phase, "tdvp_method")),
            "rtol": float(_cache_scalar(phase, "tdvp_rtol")),
            "atol": float(_cache_scalar(phase, "tdvp_atol")),
            "max_step": float(_cache_scalar(phase, "tdvp_max_step")),
        },
    }
    for point_index, (axis, (_, _, label)) in enumerate(zip(axes, POINTS)):
        phase_summary[label] = plot_bloch_portrait(
            axis,
            phase["times"],
            phase["theta"][point_index],
            phase["phi"][point_index],
            label,
            "#2166AC",
            f"({chr(ord('a') + point_index)})",
        )
        if point_index > 0:
            axis.tick_params(labelleft=False)

    # Move the shared labels away from the numerical tick labels.  Their
    # figure-coordinate positions are the two manual spacing controls here;
    # tight saving below retains them even though they sit just outside the
    # nominal canvas.
    fig.supxlabel(r"$\sin\theta_{i_0}\cos\phi_{i_0}$", y=-0.015)
    fig.supylabel(r"$\sin\theta_{i_0}\sin\phi_{i_0}$", x=-0.008)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / "fig2_bloch_planes.pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    fig.savefig(
        output_dir / "fig2_bloch_planes.png",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)
    with (output_dir / "fig2_phase_diagnostics.json").open("w") as handle:
        json.dump(phase_summary, handle, indent=2)


def run_self_test() -> None:
    """Run a fast TDVP and constrained-basis smoke test without manuscript data."""

    test_times = np.linspace(0.0, 0.04, 5)
    trajectory = integrate_tdvp(
        -0.12,
        1.90,
        test_times,
        period=6,
    )
    leakage = tdvpfun.get_qleak(trajectory)
    if trajectory.shape != (12, test_times.size):
        raise AssertionError(f"unexpected TDVP shape {trajectory.shape}")
    if not np.all(np.isfinite(leakage)):
        raise AssertionError("TDVP self-test produced nonfinite leakage")

    basis = pxpbasisS.constrained_basis(2, 4, None, None)
    theta0, phi0 = initial_angles(4)
    state = EDfun.mpsmanifold(theta0, phi0, basis).ravel()
    state_norm = float(np.linalg.norm(state))
    if not np.isclose(state_norm, 1.0, rtol=0.0, atol=1.0e-12):
        raise AssertionError(f"ED initial-state norm is {state_norm}")

    print(
        json.dumps(
            {
                "status": "ok",
                "tdvp_shape": list(trajectory.shape),
                "leakage_range": [
                    float(np.min(leakage)),
                    float(np.max(leakage)),
                ],
                "small_basis_size": int(basis.Ns),
                "ed_state_norm": state_norm,
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures", choices=("1", "2", "both"), default="both"
    )
    parser.add_argument(
        "--fig1-cache",
        type=Path,
        default=DATA_DIR / "fig1_average_leakage.npz",
    )
    parser.add_argument(
        "--legacy-fig1-pickle",
        type=Path,
        help=(
            "Optional historical pickle; not required because the scan can "
            "be generated from code."
        ),
    )
    parser.add_argument(
        "--leakage-row-dir",
        type=Path,
        default=DATA_DIR / "leakage_rows",
        help="Directory for resumable leakage-row checkpoints.",
    )
    parser.add_argument("--recompute-leakage", action="store_true")
    parser.add_argument(
        "--core-cache", type=Path, default=DATA_DIR / "fig2_core.npz"
    )
    parser.add_argument(
        "--phase-cache", type=Path, default=DATA_DIR / "fig2_phase_space.npz"
    )
    parser.add_argument("--recompute-core", action="store_true")
    parser.add_argument("--recompute-phase", action="store_true")
    parser.add_argument(
        "--phase-tmax",
        type=float,
        default=10.0,
        help="Phase-trajectory end time; a mismatched cache is rebuilt.",
    )
    parser.add_argument(
        "--phase-dt",
        type=float,
        default=0.01,
        help="Phase-trajectory sampling interval; a mismatched cache is rebuilt.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Worker processes used for the leakage-map rows.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a fast TDVP/QuSpin smoke test and exit.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be a positive integer")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    if args.self_test:
        run_self_test()
        return

    if args.legacy_fig1_pickle is not None:
        prepare_fig1_cache(args.legacy_fig1_pickle, args.fig1_cache)
    if args.figures in ("1", "both"):
        if args.recompute_leakage or not args.fig1_cache.exists():
            generate_leakage_cache(
                args.fig1_cache,
                args.leakage_row_dir,
                workers=args.workers,
                clear_rows=args.recompute_leakage,
            )
        with np.load(args.fig1_cache) as cached_leakage:
            leakage_cache_matches, leakage_reason = validate_leakage_cache(
                cached_leakage
            )
        if not leakage_cache_matches:
            raise ValueError(
                f"Leakage cache {args.fig1_cache} is incompatible: "
                f"{leakage_reason}"
            )

        core_cache_matches = False
        core_reason = "cache does not exist"
        if args.core_cache.exists() and not args.recompute_core:
            with np.load(args.core_cache) as cached_core:
                core_cache_matches, core_reason = validate_core_cache(
                    cached_core
                )
        if not core_cache_matches:
            if args.core_cache.exists() and not args.recompute_core:
                print(
                    f"Rebuilding incompatible core cache: {core_reason}",
                    flush=True,
                )
            generate_fig2_core(args.core_cache)
        validate_marked_leakage_points(args.fig1_cache, args.output_dir)
        plot_fig1(args.fig1_cache, args.core_cache, args.output_dir)

    if args.figures in ("2", "both"):
        phase_cache_matches = False
        phase_reason = "cache does not exist"
        if args.phase_cache.exists() and not args.recompute_phase:
            with np.load(args.phase_cache) as cached_phase:
                phase_cache_matches, phase_reason = validate_phase_cache(
                    cached_phase,
                    tmax=args.phase_tmax,
                    dt=args.phase_dt,
                )
        if not phase_cache_matches:
            if args.phase_cache.exists() and not args.recompute_phase:
                print(
                    f"Rebuilding incompatible phase cache: {phase_reason}",
                    flush=True,
                )
            generate_phase_cache(
                args.phase_cache, tmax=args.phase_tmax, dt=args.phase_dt
            )
        plot_fig2(
            args.phase_cache,
            args.output_dir,
            tmax=args.phase_tmax,
            dt=args.phase_dt,
        )


if __name__ == "__main__":
    main()
