#!/usr/bin/env python3
"""Legacy normalized-response convergence helpers.

This module is retained because the optional Cartesian pole audit imports its
TDVP equations and diagnostic helpers.  Its former normalized-response main
program is no longer part of the Paper-A reproduction workflow.  The active
physical convergence test is ``benchmark_defect_delta_m_convergence.py``.

Run in the requested environment:

    conda run -n quspin python scripts/benchmark_arbitrary_k_convergence.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

# Keep the convergence driver independent of unused legacy QuSpin/joblib
# imports while running it in the manuscript's requested ``quspin`` env.
MATPLOTLIB_CACHE = ROOT / "tmp" / "matplotlib-finite-k-local"
MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE))
GENERAL_CACHE = ROOT / "tmp" / "cache-finite-k-local"
GENERAL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(GENERAL_CACHE))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402


# ---------------------------------------------------------------------------
# One source of truth for the scientific comparison.
# These values exactly match fig1fig2/reproduce_fig1_fig2.py.
# ---------------------------------------------------------------------------
POINTS: dict[str, tuple[float, float]] = {}
DEFAULT_SELECTION_MANIFEST = (
    ROOT / "fig1fig2" / "data" / "selected_points_phi0.json"
)
DEFAULT_LEAKAGE_MAP = (
    ROOT / "fig1fig2" / "data" / "fig1_average_leakage.npz"
)
SAMPLED_PERIODS = (20, 40, 60, 80, 100, 140, 200)
REFERENCE_PERIOD = 500
BACKGROUND_PERIOD = 2
# Fixed-coordinate value used by every active Paper-A panel.  The scan below
# tests sensitivity but does not establish an epsilon->0 limit.
POLE_BIAS = 1.0e-3
INITIAL_PHI = 0.0
REGULARIZATION_PERIOD = 100
ED_LENGTH = 24
ED_SOLVER_NAME = "dop853"
REGULARIZATION_VALUES = tuple(np.logspace(-1.0, -7.0, 13))
STRICT_AUDIT_VALUES = tuple(
    10.0**exponent for exponent in (-2.0, -3.0, -3.5, -4.0, -5.0)
)
TMAX = 10.0
SAMPLE_COUNT = 1001
METHOD = "DOP853"
RTOL = 1.0e-9
ATOL = 1.0e-11
MAX_STEP = 0.02
STRICT_RTOL = 1.0e-11
STRICT_ATOL = 1.0e-13
STRICT_MAX_STEP = 0.01
STANDARD_TIMEOUT_SECONDS = 900.0
STRICT_TIMEOUT_SECONDS = 1200.0
SOLVER_SENSITIVITY_RATIO = 0.10
WINDOW_RADIUS = 8

DEFAULT_OUTPUT_DIR = ROOT / "output" / "finite_k_convergence"
STYLE_FILE = Path(__file__).with_name("hzg-paper.mplstyle")


# ---------------------------------------------------------------------------
# HZG two-column figure controls.
# Increasing left/bottom creates more room for axis labels; wspace/hspace
# control the gaps between the four diagnostics.
# ---------------------------------------------------------------------------
FIGURE_SIZE = (7.10, 5.35)
FIGURE_ADJUST = {
    "left": 0.085,
    "right": 0.985,
    "bottom": 0.105,
    "top": 0.98,
    "wspace": 0.28,
    "hspace": 0.30,
}
X_LIMITS = (20.0, 200.0)
X_TICKS = (20, 60, 100, 140, 200)
REFERENCE_TEXT_POSITION = (0.96, 0.57)
EPSILON_LIMITS = (1.0e-1, 1.0e-7)
TRACE_EPSILONS = (1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5)

CASE_STYLE = {
    "P1": {
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
        "markerfacecolor": "#0072B2",
    },
    "P6": {
        "color": "#D55E00",
        "marker": "s",
        "linestyle": "--",
        "markerfacecolor": "white",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_points(manifest_path: Path, leakage_path: Path) -> None:
    """Load the current data-selected P1/P6 coordinates."""

    global POINTS
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if int(manifest.get("schema_version", -1)) != 1:
        raise ValueError("selection manifest schema differs")
    if manifest.get("map_sha256") != sha256(leakage_path):
        raise ValueError("selection manifest does not match the leakage map")
    protocol = manifest.get("map_protocol", {})
    if not np.isclose(
        float(protocol.get("pole_bias", np.nan)),
        POLE_BIAS,
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise ValueError("selection manifest pole bias differs")
    if not np.isclose(
        float(protocol.get("initial_phi", np.nan)),
        INITIAL_PHI,
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise ValueError("selection manifest initial phase differs")
    main_rows = manifest.get("families", {}).get("main", [])
    rows_by_id = {str(row["point_id"]): row for row in main_rows}
    if not {"P1", "P6"}.issubset(rows_by_id):
        raise ValueError("selection manifest lacks P1 or P6")
    POINTS = {
        label: (
            float(rows_by_id[label]["mu"]),
            float(rows_by_id[label]["chi"]),
        )
        for label in ("P1", "P6")
    }


class IntegrationTimeout(RuntimeError):
    """Raised when a coordinate-pole trajectory exceeds its wall-time cap."""


@contextmanager
def wall_time_limit(seconds: float):
    """Interrupt one POSIX solve after a reproducible wall-time budget."""
    if seconds <= 0.0:
        yield
        return

    def handle_timeout(signum, frame):
        del signum, frame
        raise IntegrationTimeout(
            f"trajectory exceeded the {seconds:g} s wall-time cap"
        )

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def eta_values(theta: np.ndarray) -> np.ndarray:
    """Evaluate the exact spin-1/2 ``eta_i`` recurrence in O(K) work.

    This is the self-contained form of the recurrence used by
    ``fig1fig2/tdvpfun.py``.  Keeping it here avoids importing that legacy
    module and keeps this benchmark focused on the O(K) TDVP calculation.
    """
    theta = np.asarray(theta)
    negative_population = -np.sin(theta / 2.0) ** 2
    period = negative_population.shape[0]

    beta = np.prod(negative_population, axis=0)
    backward_sites = (-np.arange(1, period + 1)) % period
    backward_products = np.cumprod(
        negative_population[backward_sites],
        axis=0,
    )

    eta = np.empty_like(
        negative_population,
        dtype=np.result_type(negative_population, np.float64),
    )
    eta[0] = 1.0 + np.sum(backward_products, axis=0) / (1.0 - beta)
    for site in range(1, period):
        eta[site] = (
            1.0 + negative_population[site - 1] * eta[site - 1]
        )
    return eta


def tdvp_rhs(
    time: float,
    state: np.ndarray,
    mu: float,
    chi: float,
) -> np.ndarray:
    """Exact spin-1/2 application TDVP equation in self-contained form."""
    del time  # The application equation is autonomous.
    period = state.size // 2
    if period % 2:
        raise ValueError("The staggered application protocol requires even K.")

    theta = state[:period]
    phi = state[period:]
    eta = eta_values(theta)

    eta_previous = np.roll(eta, 1)
    eta_next = np.roll(eta, -1)
    theta_previous = np.roll(theta, 1)
    theta_next = np.roll(theta, -1)
    theta_next_next = np.roll(theta, -2)
    phi_previous = np.roll(phi, 1)
    phi_next = np.roll(phi, -1)
    detuning = np.tile(
        np.array([2.0 * mu - chi, 2.0 * mu + chi]),
        period // 2,
    )

    theta_dot = (
        2.0 * np.cos(theta_next / 2.0) * np.sin(phi)
        + eta_previous
        * np.sin(theta_previous)
        * np.sin(theta / 2.0)
        * np.sin(phi_previous)
        / eta
    )
    phi_dot = (
        2.0
        * np.cos(theta_next / 2.0)
        * np.cos(phi)
        / np.tan(theta)
        + 2.0 * detuning
        - np.cos(theta_next_next / 2.0)
        * np.cos(phi_next)
        * np.tan(theta_next / 2.0)
        - eta_previous
        * np.sin(theta_previous)
        * np.cos(phi_previous)
        / (2.0 * eta * np.cos(theta / 2.0))
        - eta
        * np.sin(theta)
        * np.cos(phi)
        * np.sin(theta_next / 2.0)
        * np.tan(theta_next / 2.0)
        / (2.0 * eta_next)
    )
    return np.concatenate((theta_dot, phi_dot))


def defect_site(period: int) -> int:
    """Zero-indexed defect site used in the application code."""
    return 2 * ((period + 2) // 4) - 1


def initial_angles(
    period: int,
    *,
    with_defect: bool,
    regularization: float = POLE_BIAS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the common pole-regularized Z2 initial condition."""
    if regularization <= 0.0:
        raise ValueError("The pole regularization must be positive.")
    theta = np.asarray(
        [regularization, np.pi - regularization] * (period // 2),
        dtype=float,
    )
    if with_defect:
        theta[defect_site(period)] = regularization
    phi = np.full(period, INITIAL_PHI, dtype=float)
    return theta, phi


def integrate_tdvp(
    period: int,
    mu: float,
    chi: float,
    times: np.ndarray,
    *,
    with_defect: bool,
    regularization: float = POLE_BIAS,
    rtol: float = RTOL,
    atol: float = ATOL,
    max_step: float = MAX_STEP,
) -> np.ndarray:
    """Integrate one trajectory with an explicitly recorded solver protocol."""
    theta0, phi0 = initial_angles(
        period,
        with_defect=with_defect,
        regularization=regularization,
    )
    solution = solve_ivp(
        tdvp_rhs,
        (float(times[0]), float(times[-1])),
        np.concatenate((theta0, phi0)),
        t_eval=times,
        method=METHOD,
        args=(mu, chi),
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    expected_shape = (2 * period, len(times))
    if not solution.success or solution.y.shape != expected_shape:
        raise RuntimeError(
            f"TDVP failed for K={period}, epsilon={regularization}, "
            f"(mu,chi)=({mu},{chi}): "
            f"{solution.message}; shape={solution.y.shape}, "
            f"expected={expected_shape}"
        )
    return solution.y


def bond_smoothed_population(trajectory: np.ndarray) -> np.ndarray:
    """Return the same bond-smoothed occupation used in the application."""
    period = trajectory.shape[0] // 2
    theta = trajectory[:period]
    eta = eta_values(theta)
    magnetization = -1.0 + eta * (1.0 - np.cos(theta))
    occupation = 0.5 * (1.0 + magnetization)
    return occupation + np.roll(occupation, -1, axis=0)


def normalized_defect_response(
    defect_profile: np.ndarray,
    background_profile: np.ndarray,
) -> np.ndarray:
    """Normalize |defect-background| independently at every saved time."""
    difference = np.abs(defect_profile - background_profile)
    normalization = np.sum(difference, axis=0, keepdims=True)
    return np.divide(
        difference,
        normalization,
        out=np.zeros_like(difference),
        where=normalization > 0.0,
    )


def centered_window(
    response: np.ndarray, center: int, radius: int
) -> np.ndarray:
    """Extract offsets -radius,...,+radius with periodic indexing."""
    offsets = np.arange(-radius, radius + 1)
    return response[(center + offsets) % response.shape[0]]


def response_window(
    period: int,
    mu: float,
    chi: float,
    times: np.ndarray,
    background_period_two: np.ndarray,
    radius: int,
    *,
    regularization: float = POLE_BIAS,
) -> np.ndarray:
    """Compute the fixed local response window for one finite cell."""
    trajectory = integrate_tdvp(
        period,
        mu,
        chi,
        times,
        with_defect=True,
        regularization=regularization,
    )
    defect_profile = bond_smoothed_population(trajectory)
    background = np.tile(
        background_period_two, (period // BACKGROUND_PERIOD, 1)
    )
    response = normalized_defect_response(defect_profile, background)
    return centered_window(response, defect_site(period), radius)


def leakage_rate(trajectory: np.ndarray) -> np.ndarray:
    """Return the spin-1/2 intensive residual used in the application."""
    period = trajectory.shape[0] // 2
    theta = trajectory[:period]
    eta = eta_values(theta)
    theta_next = np.roll(theta, -1, axis=0)
    eta_next = np.roll(eta, -1, axis=0)
    density = (
        np.sin(theta / 2.0) ** 2
        * np.sin(theta_next / 2.0) ** 2
        * eta
        * (1.0 - eta)
        / eta_next
    )
    return np.sqrt(np.abs(np.mean(density, axis=0)))


def time_average(values: np.ndarray, times: np.ndarray) -> float:
    duration = float(times[-1] - times[0])
    if duration <= 0.0:
        raise ValueError("A positive time interval is required.")
    return float(np.trapezoid(values, times) / duration)


def time_averaged_response_width(
    response: np.ndarray,
    center: int,
    times: np.ndarray,
) -> float:
    """Return the application RMS width averaged over the full time window."""
    distance_squared = (np.arange(response.shape[0]) - center) ** 2
    width = np.sqrt(np.sum(distance_squared[:, None] * response, axis=0))
    return time_average(width, times)


def defect_bloch_coordinates(trajectory: np.ndarray) -> np.ndarray:
    """Return nonsingular physical X,Y coordinates at the defect site."""
    period = trajectory.shape[0] // 2
    site = defect_site(period)
    theta = trajectory[site]
    phi = trajectory[period + site]
    return np.vstack(
        (
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
        )
    )


def regularization_trajectory_data(
    period: int,
    mu: float,
    chi: float,
    times: np.ndarray,
    radius: int,
    regularization: float,
    *,
    rtol: float = RTOL,
    atol: float = ATOL,
    max_step: float = MAX_STEP,
) -> dict[str, np.ndarray | float]:
    """Compute response, leakage, width, and Bloch data at one epsilon."""
    background_trajectory = integrate_tdvp(
        BACKGROUND_PERIOD,
        mu,
        chi,
        times,
        with_defect=False,
        regularization=regularization,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    defect_trajectory = integrate_tdvp(
        period,
        mu,
        chi,
        times,
        with_defect=True,
        regularization=regularization,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    defect_profile = bond_smoothed_population(defect_trajectory)
    background_profile_two = bond_smoothed_population(background_trajectory)
    background_profile = np.tile(
        background_profile_two,
        (period // BACKGROUND_PERIOD, 1),
    )
    response = normalized_defect_response(defect_profile, background_profile)
    gamma = leakage_rate(defect_trajectory)
    return {
        "response_window": centered_window(
            response,
            defect_site(period),
            radius,
        ),
        "mean_gamma": time_average(gamma, times),
        "gamma": gamma,
        "mean_width": time_averaged_response_width(
            response,
            defect_site(period),
            times,
        ),
        "bloch": defect_bloch_coordinates(defect_trajectory),
    }


def regularization_convergence_metrics(
    candidate: dict[str, np.ndarray | float],
    reference: dict[str, np.ndarray | float],
) -> dict[str, float]:
    """Return physical deviations from the finite epsilon reference."""
    response_metrics = convergence_metrics(
        np.asarray(candidate["response_window"]),
        np.asarray(reference["response_window"]),
    )
    bloch_difference = np.asarray(candidate["bloch"]) - np.asarray(
        reference["bloch"]
    )
    return {
        "local_window_rmse_vs_epsilon_ref": response_metrics[
            "local_window_rmse_vs_Kref"
        ],
        "local_window_max_abs_vs_epsilon_ref": response_metrics[
            "local_window_max_abs_vs_Kref"
        ],
        "mean_gamma": float(candidate["mean_gamma"]),
        "abs_mean_gamma_difference_vs_epsilon_ref": abs(
            float(candidate["mean_gamma"]) - float(reference["mean_gamma"])
        ),
        "mean_width": float(candidate["mean_width"]),
        "abs_mean_width_difference_vs_epsilon_ref": abs(
            float(candidate["mean_width"]) - float(reference["mean_width"])
        ),
        "defect_bloch_rmse_vs_epsilon_ref": float(
            np.sqrt(np.mean(np.sum(bloch_difference**2, axis=0)))
        ),
    }


def convergence_metrics(
    candidate: np.ndarray, reference: np.ndarray
) -> dict[str, float]:
    """Return local full-window and central-site convergence measures."""
    difference = candidate - reference
    central = difference[difference.shape[0] // 2]
    return {
        "local_window_rmse_vs_Kref": float(
            np.sqrt(np.mean(difference**2))
        ),
        "local_window_max_abs_vs_Kref": float(
            np.max(np.abs(difference))
        ),
        "central_response_rmse_vs_Kref": float(
            np.sqrt(np.mean(central**2))
        ),
    }


def is_strict_audit_value(epsilon: float) -> bool:
    """Return whether epsilon belongs to the predeclared strict audit."""
    return any(
        np.isclose(epsilon, value, rtol=1.0e-12, atol=0.0)
        for value in STRICT_AUDIT_VALUES
    )


def timed_regularization_data(
    mu: float,
    chi: float,
    times: np.ndarray,
    radius: int,
    epsilon: float,
    *,
    strict: bool,
    timeout_seconds: float,
) -> tuple[dict[str, np.ndarray | float], float]:
    """Run one epsilon protocol and return its data and total wall time."""
    protocol = (
        (STRICT_RTOL, STRICT_ATOL, STRICT_MAX_STEP)
        if strict
        else (RTOL, ATOL, MAX_STEP)
    )
    started = time.perf_counter()
    with wall_time_limit(timeout_seconds):
        data = regularization_trajectory_data(
            REGULARIZATION_PERIOD,
            mu,
            chi,
            times,
            radius,
            epsilon,
            rtol=protocol[0],
            atol=protocol[1],
            max_step=protocol[2],
        )
    return data, time.perf_counter() - started


def neighbor_regularization_metrics(
    candidate: dict[str, np.ndarray | float],
    next_smaller: dict[str, np.ndarray | float],
) -> dict[str, float]:
    """Compare one epsilon with the next smaller half-decade sample."""
    raw = regularization_convergence_metrics(candidate, next_smaller)
    gamma_reference = max(abs(float(next_smaller["mean_gamma"])), 1.0e-15)
    width_reference = max(abs(float(next_smaller["mean_width"])), 1.0e-15)
    return {
        "local_step_rmse": raw["local_window_rmse_vs_epsilon_ref"],
        "local_step_max_abs": raw["local_window_max_abs_vs_epsilon_ref"],
        "abs_mean_gamma_step": raw[
            "abs_mean_gamma_difference_vs_epsilon_ref"
        ],
        "relative_mean_gamma_step": raw[
            "abs_mean_gamma_difference_vs_epsilon_ref"
        ]
        / gamma_reference,
        "abs_mean_width_step": raw[
            "abs_mean_width_difference_vs_epsilon_ref"
        ],
        "relative_mean_width_step": raw[
            "abs_mean_width_difference_vs_epsilon_ref"
        ]
        / width_reference,
        "defect_bloch_step_rmse": raw[
            "defect_bloch_rmse_vs_epsilon_ref"
        ],
    }


def solver_audit_metrics(
    standard: dict[str, np.ndarray | float],
    strict: dict[str, np.ndarray | float],
) -> dict[str, float]:
    """Compare manuscript and tightened solver protocols at fixed epsilon."""
    raw = regularization_convergence_metrics(standard, strict)
    return {
        "solver_local_rmse": raw["local_window_rmse_vs_epsilon_ref"],
        "solver_local_max_abs": raw[
            "local_window_max_abs_vs_epsilon_ref"
        ],
        "solver_abs_mean_gamma": raw[
            "abs_mean_gamma_difference_vs_epsilon_ref"
        ],
        "solver_abs_mean_width": raw[
            "abs_mean_width_difference_vs_epsilon_ref"
        ],
        "solver_defect_bloch_rmse": raw[
            "defect_bloch_rmse_vs_epsilon_ref"
        ],
    }


def generate_data(
    times: np.ndarray,
    radius: int,
    *,
    include_finite_k: bool = True,
    continue_after_timeout: bool = False,
) -> tuple[
    dict[str, np.ndarray],
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
]:
    """Generate the finite-K and finite-epsilon comparisons."""
    arrays: dict[str, np.ndarray] = {
        "times": times,
        "sampled_periods": np.asarray(SAMPLED_PERIODS, dtype=int),
        "reference_period": np.asarray(REFERENCE_PERIOD, dtype=int),
        "window_offsets": np.arange(-radius, radius + 1),
        "regularization_period": np.asarray(
            REGULARIZATION_PERIOD, dtype=int
        ),
        "regularization_values": np.asarray(REGULARIZATION_VALUES),
        "finite_k_available": np.asarray(int(include_finite_k), dtype=int),
    }
    period_rows: list[dict[str, float | int | str]] = []
    regularization_rows: list[dict[str, float | int | str]] = []

    for label, (mu, chi) in POINTS.items():
        if include_finite_k:
            background_trajectory = integrate_tdvp(
                BACKGROUND_PERIOD,
                mu,
                chi,
                times,
                with_defect=False,
            )
            background_profile = bond_smoothed_population(
                background_trajectory
            )
            reference = response_window(
                REFERENCE_PERIOD,
                mu,
                chi,
                times,
                background_profile,
                radius,
            )
            arrays[f"{label}_Kref{REFERENCE_PERIOD}_response_window"] = (
                reference
            )

            errors = []
            for period in SAMPLED_PERIODS:
                candidate = response_window(
                    period,
                    mu,
                    chi,
                    times,
                    background_profile,
                    radius,
                )
                metrics = convergence_metrics(candidate, reference)
                arrays[f"{label}_K{period}_response_window"] = candidate
                errors.append(metrics["local_window_rmse_vs_Kref"])
                period_rows.append(
                    {
                        "point": label,
                        "mu": mu,
                        "chi": chi,
                        "K": period,
                        "K_ref": REFERENCE_PERIOD,
                        "window_radius": radius,
                        **metrics,
                    }
                )
                print(
                    f"{label}, K={period}: "
                    f"epsilon_loc={metrics['local_window_rmse_vs_Kref']:.6e}",
                    flush=True,
                )
            arrays[f"{label}_epsilon_loc"] = np.asarray(errors)
        else:
            print(
                f"{label}: finite-K scan omitted; the phi=0 "
                f"K_ref={REFERENCE_PERIOD} trajectory did not finish",
                flush=True,
            )

        count = len(REGULARIZATION_VALUES)
        standard_data: list[dict[str, np.ndarray | float] | None] = [
            None
        ] * count
        strict_data: list[dict[str, np.ndarray | float] | None] = [
            None
        ] * count
        standard_status = np.full(count, 2, dtype=int)
        strict_status = np.full(count, 2, dtype=int)
        standard_wall_time = np.full(count, np.nan)
        strict_wall_time = np.full(count, np.nan)
        skip_smaller = False

        for index, epsilon in enumerate(REGULARIZATION_VALUES):
            if skip_smaller:
                print(
                    f"{label}, epsilon={epsilon:.1e}: "
                    "skipped after a larger-epsilon timeout",
                    flush=True,
                )
                continue

            try:
                standard_data[index], standard_wall_time[index] = (
                    timed_regularization_data(
                        mu,
                        chi,
                        times,
                        radius,
                        epsilon,
                        strict=False,
                        timeout_seconds=STANDARD_TIMEOUT_SECONDS,
                    )
                )
                standard_status[index] = 0
            except IntegrationTimeout:
                standard_status[index] = 1
                standard_wall_time[index] = STANDARD_TIMEOUT_SECONDS
                skip_smaller = not continue_after_timeout
                print(
                    f"{label}, epsilon={epsilon:.1e}: "
                    f"standard timeout ({STANDARD_TIMEOUT_SECONDS:g} s)",
                    flush=True,
                )
                continue
            except RuntimeError as error:
                standard_status[index] = 3
                skip_smaller = True
                print(
                    f"{label}, epsilon={epsilon:.1e}: "
                    f"standard solve failed ({error})",
                    flush=True,
                )
                continue

            candidate = standard_data[index]
            assert candidate is not None
            print(
                f"{label}, epsilon={epsilon:.1e}: "
                f"standard ok in {standard_wall_time[index]:.2f} s, "
                f"Gamma={float(candidate['mean_gamma']):.6e}, "
                f"omega={float(candidate['mean_width']):.6e}",
                flush=True,
            )

            if not is_strict_audit_value(epsilon):
                continue
            try:
                strict_data[index], strict_wall_time[index] = (
                    timed_regularization_data(
                        mu,
                        chi,
                        times,
                        radius,
                        epsilon,
                        strict=True,
                        timeout_seconds=STRICT_TIMEOUT_SECONDS,
                    )
                )
                strict_status[index] = 0
            except IntegrationTimeout:
                strict_status[index] = 1
                strict_wall_time[index] = STRICT_TIMEOUT_SECONDS
                print(
                    f"{label}, epsilon={epsilon:.1e}: "
                    f"strict timeout ({STRICT_TIMEOUT_SECONDS:g} s)",
                    flush=True,
                )
            except RuntimeError as error:
                strict_status[index] = 3
                print(
                    f"{label}, epsilon={epsilon:.1e}: "
                    f"strict solve failed ({error})",
                    flush=True,
                )

        local_step = np.full(count, np.nan)
        local_step_max = np.full(count, np.nan)
        gamma_step_relative = np.full(count, np.nan)
        width_step_relative = np.full(count, np.nan)
        bloch_step = np.full(count, np.nan)
        solver_local = np.full(count, np.nan)
        mean_gamma = np.full(count, np.nan)
        mean_width = np.full(count, np.nan)

        for index, epsilon in enumerate(REGULARIZATION_VALUES):
            candidate = standard_data[index]
            if candidate is None:
                standard_label = (
                    "timeout"
                    if standard_status[index] == 1
                    else "skipped_after_timeout"
                    if standard_status[index] == 2
                    else "failed"
                )
            else:
                standard_label = "success"
                mean_gamma[index] = float(candidate["mean_gamma"])
                mean_width[index] = float(candidate["mean_width"])

            neighbor = (
                standard_data[index + 1] if index + 1 < count else None
            )
            neighbor_epsilon = (
                REGULARIZATION_VALUES[index + 1]
                if index + 1 < count
                else np.nan
            )
            neighbor_metrics = {
                "local_step_rmse": np.nan,
                "local_step_max_abs": np.nan,
                "abs_mean_gamma_step": np.nan,
                "relative_mean_gamma_step": np.nan,
                "abs_mean_width_step": np.nan,
                "relative_mean_width_step": np.nan,
                "defect_bloch_step_rmse": np.nan,
            }
            if candidate is not None and neighbor is not None:
                neighbor_metrics = neighbor_regularization_metrics(
                    candidate,
                    neighbor,
                )
                local_step[index] = neighbor_metrics["local_step_rmse"]
                local_step_max[index] = neighbor_metrics[
                    "local_step_max_abs"
                ]
                gamma_step_relative[index] = neighbor_metrics[
                    "relative_mean_gamma_step"
                ]
                width_step_relative[index] = neighbor_metrics[
                    "relative_mean_width_step"
                ]
                bloch_step[index] = neighbor_metrics[
                    "defect_bloch_step_rmse"
                ]

            audit_metrics = {
                "solver_local_rmse": np.nan,
                "solver_local_max_abs": np.nan,
                "solver_abs_mean_gamma": np.nan,
                "solver_abs_mean_width": np.nan,
                "solver_defect_bloch_rmse": np.nan,
            }
            strict_candidate = strict_data[index]
            if strict_candidate is not None and candidate is not None:
                audit_metrics = solver_audit_metrics(
                    candidate,
                    strict_candidate,
                )
                solver_local[index] = audit_metrics["solver_local_rmse"]

            strict_label = (
                "success"
                if strict_status[index] == 0
                else "timeout"
                if strict_status[index] == 1
                else "not_scheduled"
                if strict_status[index] == 2
                else "failed"
            )
            regularization_rows.append(
                {
                    "point": label,
                    "mu": mu,
                    "chi": chi,
                    "K": REGULARIZATION_PERIOD,
                    "epsilon": epsilon,
                    "next_smaller_epsilon": neighbor_epsilon,
                    "window_radius": radius,
                    "standard_status": standard_label,
                    "standard_wall_time_seconds": standard_wall_time[index],
                    "strict_status": strict_label,
                    "strict_wall_time_seconds": strict_wall_time[index],
                    "mean_gamma": mean_gamma[index],
                    "mean_width": mean_width[index],
                    **neighbor_metrics,
                    **audit_metrics,
                }
            )

        arrays[f"{label}_regularization_step_loc"] = local_step
        arrays[f"{label}_regularization_step_loc_max"] = local_step_max
        arrays[f"{label}_regularization_step_gamma_relative"] = (
            gamma_step_relative
        )
        arrays[f"{label}_regularization_step_width_relative"] = (
            width_step_relative
        )
        arrays[f"{label}_regularization_step_bloch"] = bloch_step
        arrays[f"{label}_solver_local_rmse"] = solver_local
        arrays[f"{label}_mean_gamma"] = mean_gamma
        arrays[f"{label}_mean_width"] = mean_width
        arrays[f"{label}_standard_status"] = standard_status
        arrays[f"{label}_strict_status"] = strict_status
        arrays[f"{label}_standard_wall_time"] = standard_wall_time
        arrays[f"{label}_strict_wall_time"] = strict_wall_time

    return arrays, period_rows, regularization_rows


def write_csv(
    path: Path, rows: list[dict[str, float | int | str]]
) -> None:
    if not rows:
        path.write_text(
            "point,mu,chi,K,K_ref,window_radius,"
            "local_window_rmse_vs_Kref,local_window_max_abs_vs_Kref,"
            "central_response_rmse_vs_Kref\n",
            encoding="utf-8",
        )
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value):
    """Replace nonfinite scalars by JSON null in nested output payloads."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def plot_convergence(
    output_dir: Path,
    arrays: dict[str, np.ndarray],
) -> tuple[Path, Path]:
    """Create the four-panel finite-K and stability-aware epsilon figure."""
    plt.style.use(STYLE_FILE)
    figure, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
    figure.subplots_adjust(**FIGURE_ADJUST)
    period_axis = axes[0, 0]
    local_axis = axes[0, 1]
    scalar_axis = axes[1, 0]
    runtime_axis = axes[1, 1]

    periods = arrays["sampled_periods"]
    regularizations = arrays["regularization_values"]
    finite_k_available = bool(int(arrays["finite_k_available"]))
    for label in ("P1", "P6"):
        style = CASE_STYLE[label]
        if finite_k_available:
            period_axis.plot(
                periods,
                arrays[f"{label}_epsilon_loc"],
                label=label,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                markersize=4.2,
                markerfacecolor=style["markerfacecolor"],
                markeredgecolor=style["color"],
                markeredgewidth=0.8,
            )
        local_axis.plot(
            regularizations,
            arrays[f"{label}_regularization_step_loc"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markersize=3.8,
            markerfacecolor=style["markerfacecolor"],
            markeredgecolor=style["color"],
            markeredgewidth=0.8,
        )
        local_axis.plot(
            regularizations,
            arrays[f"{label}_solver_local_rmse"],
            color=style["color"],
            marker="x",
            linestyle="none",
            markersize=4.2,
            markeredgewidth=0.9,
        )
        scalar_axis.plot(
            regularizations,
            arrays[f"{label}_regularization_step_gamma_relative"],
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            markersize=3.5,
            markerfacecolor=style["markerfacecolor"],
            markeredgecolor=style["color"],
            markeredgewidth=0.7,
        )
        scalar_axis.plot(
            regularizations,
            arrays[f"{label}_regularization_step_width_relative"],
            color=style["color"],
            marker="^",
            linestyle=":",
            markersize=3.8,
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=0.7,
        )
        runtime_axis.plot(
            regularizations,
            arrays[f"{label}_standard_wall_time"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markersize=3.5,
            markerfacecolor=style["markerfacecolor"],
            markeredgecolor=style["color"],
            markeredgewidth=0.7,
        )
        runtime_axis.plot(
            regularizations,
            arrays[f"{label}_strict_wall_time"],
            color=style["color"],
            marker="x",
            linestyle="none",
            markersize=4.2,
            markeredgewidth=0.9,
        )
        standard_timeout = arrays[f"{label}_standard_status"] == 1
        runtime_axis.plot(
            regularizations[standard_timeout],
            arrays[f"{label}_standard_wall_time"][standard_timeout],
            color=style["color"],
            marker="v",
            linestyle="none",
            markersize=4.6,
            markerfacecolor="white",
            markeredgewidth=0.9,
        )
        strict_timeout = arrays[f"{label}_strict_status"] == 1
        runtime_axis.plot(
            regularizations[strict_timeout],
            arrays[f"{label}_strict_wall_time"][strict_timeout],
            color=style["color"],
            marker="v",
            linestyle="none",
            markersize=4.6,
            markerfacecolor="white",
            markeredgewidth=0.9,
        )

    period_axis.set_xlim(*X_LIMITS)
    period_axis.set_xticks(X_TICKS)
    period_axis.set_xlabel(r"$K$")
    period_axis.set_ylabel(r"$\epsilon_{\rm loc}(K;K_{\rm ref})$")
    if finite_k_available:
        period_axis.set_yscale("log")
        period_axis.margins(y=0.14)
        period_axis.text(
            *REFERENCE_TEXT_POSITION,
            rf"$K_{{\rm ref}}={REFERENCE_PERIOD}$",
            transform=period_axis.transAxes,
            ha="right",
            va="center",
            fontsize=7.6,
            color="0.35",
        )
        period_axis.legend(
            frameon=False,
            loc="upper right",
            handlelength=1.8,
            borderaxespad=0.2,
        )
    else:
        period_axis.set_ylim(0.0, 1.0)
        period_axis.set_yticks([])
        period_axis.text(
            0.5,
            0.47,
            "$K_{\\rm ref}=500$ trajectory not resolved\n"
            "for the $\\phi_i(0)=0$ protocol",
            transform=period_axis.transAxes,
            ha="center",
            va="center",
            fontsize=7.5,
            color="0.30",
        )
        period_axis.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color=CASE_STYLE[label]["color"],
                    marker=CASE_STYLE[label]["marker"],
                    linestyle=CASE_STYLE[label]["linestyle"],
                    markerfacecolor=CASE_STYLE[label]["markerfacecolor"],
                    label=label,
                )
                for label in ("P1", "P6")
            ],
            frameon=False,
            loc="upper right",
            handlelength=1.8,
            borderaxespad=0.2,
        )

    local_axis.set_yscale("log")
    local_axis.set_ylabel(r"$\Delta_{\rm loc}^{\downarrow}$")
    local_axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="0.2",
                marker="o",
                linewidth=1.0,
                markersize=3.5,
                label=r"next-smaller $\epsilon$",
            ),
            Line2D(
                [0],
                [0],
                color="0.2",
                marker="x",
                linewidth=0.0,
                markersize=4.0,
                label="tight-solver audit",
            ),
        ],
        frameon=False,
        loc="lower right",
        handlelength=1.4,
        borderaxespad=0.2,
    )
    scalar_axis.set_yscale("log")
    scalar_axis.set_ylabel("relative adjacent change")
    scalar_axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="0.2",
                marker="o",
                linestyle="-",
                linewidth=1.0,
                markersize=3.5,
                label=r"$\bar{\Gamma}$",
            ),
            Line2D(
                [0],
                [0],
                color="0.2",
                marker="^",
                markerfacecolor="white",
                linestyle=":",
                linewidth=1.0,
                markersize=3.8,
                label=r"$\bar{\omega}$",
            ),
        ],
        frameon=False,
        loc="lower left",
        handlelength=1.4,
        borderaxespad=0.2,
    )

    timeout_values = []
    sensitive_values = []
    for label in ("P1", "P6"):
        for protocol in ("standard", "strict"):
            timeout_mask = arrays[f"{label}_{protocol}_status"] == 1
            timeout_values.extend(regularizations[timeout_mask])
        solver_error = arrays[f"{label}_solver_local_rmse"]
        epsilon_step = arrays[f"{label}_regularization_step_loc"]
        sensitive_mask = (
            np.isfinite(solver_error)
            & np.isfinite(epsilon_step)
            & (solver_error >= SOLVER_SENSITIVITY_RATIO * epsilon_step)
        )
        sensitive_values.extend(regularizations[sensitive_mask])
    instability_values = timeout_values + sensitive_values
    instability_boundary = (
        float(max(instability_values)) if instability_values else None
    )

    runtime_axis.set_yscale("log")
    runtime_axis.set_ylabel("wall time (s)")
    runtime_handles = [
        Line2D(
            [0],
            [0],
            color="0.2",
            marker="o",
            linestyle="-",
            linewidth=1.0,
            markersize=3.5,
            label="standard solver",
        ),
        Line2D(
            [0],
            [0],
            color="0.2",
            marker="x",
            linestyle="none",
            markersize=4.0,
            label="tight solver",
        ),
    ]
    if timeout_values:
        runtime_handles.append(
            Line2D(
                [0],
                [0],
                color="0.2",
                marker="v",
                markerfacecolor="white",
                linestyle="none",
                markersize=4.2,
                label="timeout",
            )
        )
    runtime_axis.legend(
        handles=runtime_handles,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.12, 1.0),
        handlelength=1.4,
        borderaxespad=0.2,
    )

    for axis in (local_axis, scalar_axis, runtime_axis):
        axis.set_xscale("log")
        axis.set_xlim(*EPSILON_LIMITS)
        axis.set_xlabel(r"pole regularization $\epsilon$")
        axis.axvline(
            POLE_BIAS,
            color="0.45",
            linestyle="--",
            linewidth=0.75,
        )
        if instability_boundary is not None:
            axis.axvspan(
                instability_boundary,
                EPSILON_LIMITS[1],
                color="0.7",
                alpha=0.18,
                linewidth=0.0,
            )

    local_axis.text(
        0.54,
        0.94,
        r"displayed protocol: $\epsilon=10^{-3}$",
        transform=local_axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color="0.35",
    )
    if instability_boundary is not None:
        runtime_axis.text(
            0.965,
            0.06,
            "stiff / unresolved",
            transform=runtime_axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.0,
            color="0.38",
        )

    flat_axes = (period_axis, local_axis, scalar_axis, runtime_axis)
    for panel_label, axis in zip(("(a)", "(b)", "(c)", "(d)"), flat_axes):
        axis.text(
            0.03,
            0.94,
            panel_label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            fontweight="semibold",
        )
        axis.tick_params(direction="in", top=True, right=True)
        axis.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.38)
        for spine in axis.spines.values():
            spine.set_linewidth(0.75)

    pdf_path = output_dir / "finite_k_and_regularization_convergence.pdf"
    png_path = output_dir / "finite_k_and_regularization_convergence.png"
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(figure)
    return pdf_path, png_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=DEFAULT_SELECTION_MANIFEST,
        help="data-selected P1/P6 coordinate manifest",
    )
    parser.add_argument(
        "--leakage-map",
        type=Path,
        default=DEFAULT_LEAKAGE_MAP,
        help="generated map whose hash is recorded by the selection manifest",
    )
    parser.add_argument("--samples", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--tmax", type=float, default=TMAX)
    parser.add_argument(
        "--window-radius", type=int, default=WINDOW_RADIUS
    )
    parser.add_argument(
        "--regularization-only",
        action="store_true",
        help=(
            "skip the unbounded K_ref=500 calculation"
        ),
    )
    parser.add_argument(
        "--continue-after-timeout",
        action="store_true",
        help=(
            "attempt smaller epsilon values even after a larger-epsilon "
            "standard solve reaches the 600-s cap"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_points(
        args.selection_manifest.resolve(), args.leakage_map.resolve()
    )
    if args.samples < 5 or args.tmax <= 0.0:
        raise ValueError("positive tmax and at least five samples are required")
    if not 1 <= args.window_radius < min(SAMPLED_PERIODS) // 2:
        raise ValueError("window-radius must fit inside the smallest cell")

    times = np.linspace(0.0, args.tmax, args.samples)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays, period_rows, regularization_rows = generate_data(
        times,
        args.window_radius,
        include_finite_k=not args.regularization_only,
        continue_after_timeout=args.continue_after_timeout,
    )

    npz_path = output_dir / "finite_k_and_regularization_convergence.npz"
    json_path = output_dir / "finite_k_and_regularization_convergence.json"
    period_csv_path = output_dir / "finite_k_local_response_convergence.csv"
    regularization_csv_path = output_dir / "regularization_convergence.csv"
    np.savez_compressed(npz_path, **arrays)
    write_csv(period_csv_path, period_rows)
    write_csv(regularization_csv_path, regularization_rows)
    pdf_path, png_path = plot_convergence(output_dir, arrays)

    payload = {
        "schema_version": 7,
        "configuration": {
            "sampled_periods": list(SAMPLED_PERIODS),
            "finite_reference_period": REFERENCE_PERIOD,
            "x_axis_limits": list(X_LIMITS),
            "points": {
                label: {"mu": mu, "chi": chi}
                for label, (mu, chi) in POINTS.items()
            },
            "selection_provenance": {
                "manifest": str(args.selection_manifest.resolve()),
                "manifest_sha256": sha256(args.selection_manifest.resolve()),
                "leakage_map": str(args.leakage_map.resolve()),
                "leakage_map_sha256": sha256(args.leakage_map.resolve()),
            },
            "time_window": [0.0, args.tmax],
            "samples": args.samples,
            "window_radius": args.window_radius,
            "displayed_pole_bias": POLE_BIAS,
            "initial_phi": INITIAL_PHI,
            "regularization_scan": {
                "K": REGULARIZATION_PERIOD,
                "sampled_epsilon": list(REGULARIZATION_VALUES),
                "theta_and_phi_varied_together": False,
                "theta_regularization_only": True,
                "neighbor_definition": (
                    "next smaller point on a uniform half-decade grid"
                ),
                "strict_audit_epsilon": list(STRICT_AUDIT_VALUES),
                "standard_timeout_seconds": STANDARD_TIMEOUT_SECONDS,
                "strict_timeout_seconds": STRICT_TIMEOUT_SECONDS,
                "solver_sensitive_if_fraction_of_neighbor_step": (
                    SOLVER_SENSITIVITY_RATIO
                ),
                "regularization_only_mode": args.regularization_only,
                "continue_after_timeout": args.continue_after_timeout,
                "finite_K_available": not args.regularization_only,
            },
            "initial_state": (
                "periodic Z2 background with one central "
                "removed-excitation defect"
            ),
            "background_reference_period": BACKGROUND_PERIOD,
            "tdvp_equation": (
                "spin-half equation from fig1fig2/tdvpfun.py::eom; "
                "self-contained O(K) implementation"
            ),
            "solver": METHOD,
            "rtol": RTOL,
            "atol": ATOL,
            "max_step": MAX_STEP,
        },
        "metrics": {
            "finite_K": (
                "RMS difference between normalized defect responses at K "
                "and K_ref over all saved times and |i-i0|<=window_radius"
            ),
            "regularization_primary": (
                "RMS difference between normalized defect responses at each "
                "epsilon and the next smaller half-decade epsilon, at K=100, "
                "over all saved times and |i-i0|<=window_radius"
            ),
            "regularization_secondary": (
                "relative neighboring-epsilon changes of time-averaged "
                "leakage and RMS width, plus RMS physical defect-site "
                "Bloch-coordinate change"
            ),
            "solver_audit": (
                "standard-versus-tight DOP853 differences at fixed epsilon; "
                "timeouts and skipped smaller epsilon are explicit"
            ),
        },
        "finite_K_rows": period_rows,
        "regularization_rows": regularization_rows,
        "suggested_caption": (
            "Sensitivity audit of the phi_i(0)=0 defect protocol at P1 and "
            "P6. Panel (a) compares the local response with the finite "
            "K_ref=500 reference. Panel (b) compares each successful epsilon with the "
            "next smaller half-decade sample and overlays tightened-solver "
            "discrepancies. Panel (c) shows adjacent changes of mean leakage "
            "and width. Panel (d) shows wall times, the 600-s standard cap, "
            "and the 1200-s tightened cap. All epsilon tests use K=100 and "
            "fixed initial phi_i=0; the dashed line marks the displayed "
            "epsilon=10^-3 protocol and unresolved values are not data."
        ),
        "files": {
            "npz": str(npz_path),
            "finite_K_csv": str(period_csv_path),
            "regularization_csv": str(regularization_csv_path),
            "pdf": str(pdf_path),
            "png": str(png_path),
        },
    }
    json_path.write_text(
        json.dumps(json_safe(payload), indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir),
                "finite_reference_period": REFERENCE_PERIOD,
                "sampled_periods": list(SAMPLED_PERIODS),
                "sampled_epsilon": list(REGULARIZATION_VALUES),
                "pdf": str(pdf_path),
                "png": str(png_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
