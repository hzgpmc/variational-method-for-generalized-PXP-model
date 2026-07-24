#!/usr/bin/env python3
"""Finite-K convergence and empirical cost of the application TDVP flow.

This script changes only the periodic TDVP cell length K.  Every physical and
numerical choice is imported from ``fig1fig2/reproduce_fig1_fig2.py``:

* the P1 and P6 points;
* the pole-regularized Z2 state with one central defect;
* theta_pole=phi_i=1e-3;
* DOP853 with rtol=1e-9, atol=1e-11 and max_step=0.02;
* the application window 0 <= t <= 10 by default.

For each K, the defect trajectory is compared with a separately evolved
period-two Z2 reference, exactly as in the manuscript application.  The
largest requested K is a finite reference, not an asserted thermodynamic
limit.  The script records:

* local-window RMSE of the normalized defect response against K_ref;
* time-averaged response width;
* time-averaged and accumulated quantum leakage;
* adaptive-solver wall time and function evaluations;
* a separate median microbenchmark of one vectorized TDVP RHS call.

The runtime power laws are descriptive fits for this implementation and
machine.  They are not algorithmic-complexity theorems.

Examples
--------
Publication data:

    conda run -n quspin python scripts/benchmark_arbitrary_k_convergence.py

Fast pipeline check:

    conda run -n quspin python scripts/benchmark_arbitrary_k_convergence.py \
        --smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURE_CODE = ROOT / "fig1fig2"
if str(FIGURE_CODE) not in sys.path:
    sys.path.insert(0, str(FIGURE_CODE))

# The application module sets a writable Matplotlib cache before importing
# pyplot and supplies the single source of truth for all physical parameters.
import reproduce_fig1_fig2 as APP  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402


STYLE_FILE = Path(__file__).with_name("hzg-paper.mplstyle")
DEFAULT_OUTPUT_DIR = ROOT / "output" / "finite_k_convergence"
DEFAULT_PERIODS = (20, 40, 60, 80, 100, 140, 200)
DEFAULT_RUNTIME_PERIODS = (20, 40, 80, 160, 320, 640, 1280, 2560)
POINTS = {
    label: (float(mu), float(chi))
    for mu, chi, label in APP.POINTS
    if label in {"P1", "P6"}
}

# Plot geometry and typography are kept together for easy manual adjustment.
FIGURE_SIZE = (7.12, 5.05)
FIGURE_MARGINS = {
    "left": 0.085,
    "right": 0.985,
    "bottom": 0.105,
    "top": 0.965,
    "wspace": 0.30,
    "hspace": 0.30,
}
PANEL_TEXT_POSITION = (0.025, 0.955)
PANEL_TEXT_SIZE = 8.2

CASE_STYLE = {
    "P1": {
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
    },
    "P6": {
        "color": "#D55E00",
        "marker": "s",
        "linestyle": "--",
    },
}


def parse_periods(text: str) -> list[int]:
    periods = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not periods:
        raise argparse.ArgumentTypeError("at least one K is required")
    if len(periods) != len(set(periods)):
        raise argparse.ArgumentTypeError("K values must not be duplicated")
    return sorted(periods)


def parse_points(text: str) -> list[str]:
    labels = [item.strip().upper() for item in text.split(",") if item.strip()]
    if not labels or any(label not in POINTS for label in labels):
        raise argparse.ArgumentTypeError("points must be selected from P1,P6")
    if len(labels) != len(set(labels)):
        raise argparse.ArgumentTypeError("points must not be duplicated")
    return labels


def validate_periods(periods: list[int], reference_period: int) -> None:
    if reference_period not in periods:
        raise ValueError("reference-period must be included in periods")
    if reference_period != max(periods):
        raise ValueError("reference-period must be the largest sampled K")
    for period in periods:
        if period < 4 or period % 4:
            raise ValueError(
                "all K must be positive multiples of four so the defect and "
                "Z2 background have the same centered parity"
            )


def integrate_defect(
    period: int,
    mu: float,
    chi: float,
    times: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """Integrate one defect trajectory using the manuscript protocol."""
    theta0, phi0 = APP.initial_angles(
        period,
        APP.POLE_BIAS,
        initial_phi=APP.INITIAL_PHI,
        with_defect=True,
    )
    initial = np.concatenate((theta0, phi0))
    start = time.perf_counter()
    solution = solve_ivp(
        APP.tdvpfun.eom,
        (float(times[0]), float(times[-1])),
        initial,
        t_eval=times,
        method=APP.TDVP_METHOD,
        args=(mu, chi),
        rtol=APP.SOLVER_RTOL,
        atol=APP.SOLVER_ATOL,
        max_step=APP.SOLVER_MAX_STEP,
    )
    wall_seconds = time.perf_counter() - start
    if not solution.success or solution.y.shape != (2 * period, len(times)):
        raise RuntimeError(
            f"TDVP failed at K={period}, (mu,chi)=({mu},{chi}): "
            f"{solution.message}"
        )
    return solution.y, int(solution.nfev), wall_seconds


def integrate_reference(
    mu: float, chi: float, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Evolve the translation-invariant period-two Z2 reference."""
    theta0, phi0 = APP.initial_angles(
        APP.TDVP_REFERENCE_PERIOD,
        APP.POLE_BIAS,
        initial_phi=APP.INITIAL_PHI,
        with_defect=False,
    )
    solution = solve_ivp(
        APP.tdvpfun.eom,
        (float(times[0]), float(times[-1])),
        np.concatenate((theta0, phi0)),
        t_eval=times,
        method=APP.TDVP_METHOD,
        args=(mu, chi),
        rtol=APP.SOLVER_RTOL,
        atol=APP.SOLVER_ATOL,
        max_step=APP.SOLVER_MAX_STEP,
    )
    if not solution.success:
        raise RuntimeError(f"period-two reference failed: {solution.message}")
    return (
        APP.tdvp_translation_invariant(solution.y),
        APP.tdvpfun.get_qleak(solution.y),
    )


def normalized_response(
    profile: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized |defect-reference| and its unnormalized weight."""
    difference = np.abs(profile - reference)
    weight = np.sum(difference, axis=0)
    response = np.divide(
        difference,
        weight[None, :],
        out=np.zeros_like(difference),
        where=weight[None, :] > 0.0,
    )
    return response, weight


def periodic_width(
    response: np.ndarray, center: int
) -> np.ndarray:
    """RMS width using the shortest signed distance on the periodic cell."""
    period = response.shape[0]
    displacement = (
        (np.arange(period) - center + period // 2) % period - period // 2
    )
    return np.sqrt(
        np.sum(displacement[:, None] ** 2 * response, axis=0)
    )


def time_average(values: np.ndarray, times: np.ndarray) -> float:
    duration = float(times[-1] - times[0])
    if duration <= 0.0:
        raise ValueError("time window must have positive duration")
    return float(np.trapezoid(values, times) / duration)


def centered_window(
    values: np.ndarray, center: int, radius: int
) -> np.ndarray:
    offsets = np.arange(-radius, radius + 1)
    return values[(center + offsets) % values.shape[0]]


def benchmark_rhs(
    period: int,
    mu: float,
    chi: float,
    *,
    batches: int = 5,
    target_batch_seconds: float = 0.025,
) -> tuple[float, int]:
    """Median time per RHS call at the common initial defect state."""
    theta0, phi0 = APP.initial_angles(
        period,
        APP.POLE_BIAS,
        initial_phi=APP.INITIAL_PHI,
        with_defect=True,
    )
    state = np.concatenate((theta0, phi0))
    APP.tdvpfun.eom(0.0, state, mu, chi)  # warm up allocations/import paths

    repetitions = 16
    while True:
        start = time.perf_counter()
        for _ in range(repetitions):
            APP.tdvpfun.eom(0.0, state, mu, chi)
        elapsed = time.perf_counter() - start
        if elapsed >= target_batch_seconds or repetitions >= 131072:
            break
        repetitions *= 2

    samples = []
    for _ in range(batches):
        start = time.perf_counter()
        for _ in range(repetitions):
            APP.tdvpfun.eom(0.0, state, mu, chi)
        samples.append((time.perf_counter() - start) / repetitions)
    return float(np.median(samples)), repetitions


def loglog_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Descriptive y=a K^alpha fit and coefficient of determination."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = (x > 0.0) & (y > 0.0) & np.isfinite(y)
    if np.count_nonzero(valid) < 3:
        return {
            "prefactor": float("nan"),
            "exponent": float("nan"),
            "r_squared": float("nan"),
        }
    log_x = np.log(x[valid])
    log_y = np.log(y[valid])
    exponent, intercept = np.polyfit(log_x, log_y, 1)
    prediction = intercept + exponent * log_x
    residual = float(np.sum((log_y - prediction) ** 2))
    total = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r_squared = 1.0 - residual / total if total > 0.0 else float("nan")
    return {
        "prefactor": float(np.exp(intercept)),
        "exponent": float(exponent),
        "r_squared": r_squared,
    }


def generate_data(
    periods: list[int],
    runtime_periods: list[int],
    runtime_fit_min_k: int,
    reference_period: int,
    labels: list[str],
    times: np.ndarray,
    window_radius: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {"times": times}
    rows: list[dict[str, Any]] = []
    intermediate: dict[str, dict[int, dict[str, Any]]] = {}

    # The RHS microbenchmark uses P1 for every K so its physics and initial
    # coordinates are identical apart from the number of repeated bulk sites.
    benchmark_periods = sorted(set(periods).union(runtime_periods))
    benchmark_cost_by_period: dict[int, float] = {}
    benchmark_repetitions_by_period: dict[int, int] = {}
    for period in benchmark_periods:
        seconds, repetitions = benchmark_rhs(
            period, *POINTS["P1"]
        )
        benchmark_cost_by_period[period] = seconds
        benchmark_repetitions_by_period[period] = repetitions
    arrays["runtime_periods"] = np.asarray(runtime_periods, dtype=int)
    arrays["rhs_seconds_per_call"] = np.asarray(
        [benchmark_cost_by_period[period] for period in runtime_periods]
    )
    arrays["rhs_benchmark_repetitions"] = np.asarray(
        [
            benchmark_repetitions_by_period[period]
            for period in runtime_periods
        ],
        dtype=int,
    )

    for label in labels:
        mu, chi = POINTS[label]
        reference_pair, background_gamma = integrate_reference(
            mu, chi, times
        )
        arrays[f"{label}_background_gamma"] = background_gamma
        intermediate[label] = {}
        for period in periods:
            trajectory, nfev, wall_seconds = integrate_defect(
                period, mu, chi, times
            )
            profile = APP.tdvp_translation_invariant(trajectory)
            tiled_reference = np.tile(
                reference_pair,
                (period // APP.TDVP_REFERENCE_PERIOD, 1),
            )
            response, response_weight = normalized_response(
                profile, tiled_reference
            )
            center = APP.defect_site(period)
            width = periodic_width(response, center)
            gamma = APP.tdvpfun.get_qleak(trajectory)
            excess_leakage_weight = period * (
                gamma**2 - background_gamma**2
            )
            response_window = centered_window(
                response, center, window_radius
            )
            central_response = response[center]
            average_gamma = time_average(gamma, times)
            accumulated_gamma = float(np.trapezoid(gamma, times))
            average_excess_leakage_weight = time_average(
                excess_leakage_weight, times
            )
            average_width = time_average(width, times)
            prefix = f"{label}_K{period}"
            arrays[f"{prefix}_response_window"] = response_window
            arrays[f"{prefix}_central_response"] = central_response
            arrays[f"{prefix}_response_width"] = width
            arrays[f"{prefix}_response_weight"] = response_weight
            arrays[f"{prefix}_gamma"] = gamma
            arrays[
                f"{prefix}_excess_leakage_weight"
            ] = excess_leakage_weight
            intermediate[label][period] = {
                "response_window": response_window,
                "central_response": central_response,
                "average_width": average_width,
                "average_gamma": average_gamma,
                "accumulated_gamma": accumulated_gamma,
                "excess_leakage_weight": excess_leakage_weight,
                "average_excess_leakage_weight": (
                    average_excess_leakage_weight
                ),
                "nfev": nfev,
                "wall_seconds": wall_seconds,
            }
            print(
                f"{label}, K={period}: nfev={nfev}, "
                f"wall={wall_seconds:.3f}s, "
                f"Gamma_bar={average_gamma:.6g}, "
                f"width_bar={average_width:.6g}",
                flush=True,
            )

        finite_reference = intermediate[label][reference_period]
        for period in periods:
            item = intermediate[label][period]
            window_difference = (
                item["response_window"]
                - finite_reference["response_window"]
            )
            central_difference = (
                item["central_response"]
                - finite_reference["central_response"]
            )
            row = {
                "point": label,
                "mu": mu,
                "chi": chi,
                "K": period,
                "is_finite_reference": period == reference_period,
                "local_window_rmse_vs_Kref": float(
                    np.sqrt(np.mean(window_difference**2))
                ),
                "local_window_max_abs_vs_Kref": float(
                    np.max(np.abs(window_difference))
                ),
                "central_response_rmse_vs_Kref": float(
                    np.sqrt(np.mean(central_difference**2))
                ),
                "average_width": item["average_width"],
                "average_width_abs_error_vs_Kref": float(
                    abs(
                        item["average_width"]
                        - finite_reference["average_width"]
                    )
                ),
                "average_gamma": item["average_gamma"],
                "average_gamma_abs_error_vs_Kref": float(
                    abs(
                        item["average_gamma"]
                        - finite_reference["average_gamma"]
                    )
                ),
                "accumulated_gamma": item["accumulated_gamma"],
                "accumulated_gamma_abs_error_vs_Kref": float(
                    abs(
                        item["accumulated_gamma"]
                        - finite_reference["accumulated_gamma"]
                    )
                ),
                "average_excess_leakage_weight": (
                    item["average_excess_leakage_weight"]
                ),
                "average_excess_leakage_weight_abs_error_vs_Kref": float(
                    abs(
                        item["average_excess_leakage_weight"]
                        - finite_reference[
                            "average_excess_leakage_weight"
                        ]
                    )
                ),
                "excess_leakage_weight_rmse_vs_Kref": float(
                    np.sqrt(
                        np.mean(
                            (
                                item["excess_leakage_weight"]
                                - finite_reference[
                                    "excess_leakage_weight"
                                ]
                            )
                            ** 2
                        )
                    )
                ),
                "solver_nfev": item["nfev"],
                "solver_wall_seconds": item["wall_seconds"],
                "solver_microseconds_per_nfev": (
                    1.0e6 * item["wall_seconds"] / item["nfev"]
                ),
                "rhs_microseconds_per_call": (
                    1.0e6 * benchmark_cost_by_period[period]
                ),
            }
            rows.append(row)

    periods_array = np.asarray(periods, dtype=float)
    runtime_periods_array = np.asarray(runtime_periods, dtype=float)
    runtime_cost = np.asarray(
        [benchmark_cost_by_period[period] for period in runtime_periods]
    )
    runtime_tail = runtime_periods_array >= runtime_fit_min_k
    rhs_fit_all = loglog_fit(runtime_periods_array, runtime_cost)
    rhs_fit_tail = loglog_fit(
        runtime_periods_array[runtime_tail],
        runtime_cost[runtime_tail],
    )
    median_solver_wall = np.asarray(
        [
            np.median(
                [
                    intermediate[label][period]["wall_seconds"]
                    for label in labels
                ]
            )
            for period in periods
        ]
    )
    median_nfev = np.asarray(
        [
            np.median(
                [
                    intermediate[label][period]["nfev"]
                    for label in labels
                ]
            )
            for period in periods
        ]
    )
    arrays["median_solver_wall_seconds"] = median_solver_wall
    arrays["median_solver_nfev"] = median_nfev
    scaling = {
        "rhs_call_fit_all_sampled_K": rhs_fit_all,
        "rhs_call_fit_K_at_least": {
            "minimum_K": runtime_fit_min_k,
            **rhs_fit_tail,
        },
        "median_adaptive_solver_wall_fit": loglog_fit(
            periods_array, median_solver_wall
        ),
        "median_solver_nfev_fit": loglog_fit(
            periods_array, median_nfev
        ),
    }
    return arrays, rows, scaling


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_latex_table(
    path: Path,
    rows: list[dict[str, Any]],
    reference_period: int,
) -> None:
    """Write a compact standalone appendix table fragment."""
    selected_periods = sorted(
        {
            int(row["K"])
            for row in rows
            if row["K"] in {20, 40, 80, 100, reference_period}
        }
    )
    lines = [
        "% Generated by scripts/benchmark_arbitrary_k_convergence.py",
        "% K_ref is a finite numerical reference, not K=infinity.",
        r"\begin{tabular}{ccrrrrrr}",
        r"\hline\hline",
        (
            r"Point & $K$ & $\epsilon_{\rm loc}$ & $\bar\omega$ & "
            r"$\bar\Gamma$ & $K\overline{\delta\Gamma^2}$ & "
            r"$\Lambda(T)$ & wall time (s) \\"
        ),
        r"\hline",
    ]
    for row in rows:
        if int(row["K"]) not in selected_periods:
            continue
        lines.append(
            (
                f"{row['point']} & {int(row['K'])} & "
                f"{row['local_window_rmse_vs_Kref']:.3e} & "
                f"{row['average_width']:.4f} & "
                f"{row['average_gamma']:.4f} & "
                f"{row['average_excess_leakage_weight']:.4f} & "
                f"{row['accumulated_gamma']:.4f} & "
                f"{row['solver_wall_seconds']:.3f} "
                + r"\\"
            )
        )
    lines.extend(
        [
            r"\hline\hline",
            r"\end{tabular}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_results(
    output_dir: Path,
    periods: list[int],
    reference_period: int,
    labels: list[str],
    rows: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    scaling: dict[str, Any],
) -> tuple[Path, Path]:
    """Create a compact four-panel appendix-quality convergence figure."""
    plt.style.use(STYLE_FILE)
    figure, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
    figure.subplots_adjust(**FIGURE_MARGINS)
    error_axis, leakage_axis, width_axis, runtime_axis = axes.ravel()

    for label in labels:
        style = CASE_STYLE[label]
        selected = [row for row in rows if row["point"] == label]
        k_values = np.asarray([row["K"] for row in selected], dtype=float)
        local_errors = np.asarray(
            [row["local_window_rmse_vs_Kref"] for row in selected]
        )
        nonreference = k_values < reference_period
        error_axis.plot(
            k_values[nonreference],
            local_errors[nonreference],
            label=label,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markersize=4.0,
            markerfacecolor=(
                style["color"] if label == "P1" else "white"
            ),
            markeredgewidth=0.8,
        )
        leakage_axis.plot(
            k_values,
            [
                row["average_excess_leakage_weight"]
                for row in selected
            ],
            label=label,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markersize=4.0,
            markerfacecolor=(
                style["color"] if label == "P1" else "white"
            ),
            markeredgewidth=0.8,
        )
        width_axis.plot(
            k_values,
            [row["average_width"] for row in selected],
            label=label,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markersize=4.0,
            markerfacecolor=(
                style["color"] if label == "P1" else "white"
            ),
            markeredgewidth=0.8,
        )

    error_axis.set_yscale("log")
    error_axis.margins(x=0.05, y=0.14)
    error_axis.set_ylabel(r"$\epsilon_{\rm loc}(K;K_{\rm ref})$")
    error_axis.legend(
        frameon=False,
        loc="best",
        handlelength=1.8,
        borderaxespad=0.2,
    )

    leakage_axis.set_ylabel(
        r"$K\,\overline{(\Gamma_K^2-\Gamma_{\rm bg}^2)}$"
    )
    leakage_axis.margins(x=0.05, y=0.14)
    leakage_axis.axhline(
        0.0,
        color="0.72",
        linewidth=0.65,
        linestyle="-",
        zorder=0,
    )
    leakage_axis.axvline(
        reference_period,
        color="0.72",
        linewidth=0.7,
        linestyle=":",
        zorder=0,
    )

    width_axis.set_xlabel(r"$K$")
    width_axis.set_ylabel(r"$\bar{\omega}$")
    width_axis.margins(x=0.05, y=0.14)
    width_axis.axvline(
        reference_period,
        color="0.72",
        linewidth=0.7,
        linestyle=":",
        zorder=0,
    )

    runtime_periods = arrays["runtime_periods"].astype(float)
    rhs_microseconds = 1.0e6 * arrays["rhs_seconds_per_call"]
    runtime_axis.loglog(
        runtime_periods,
        rhs_microseconds,
        color="#009E73",
        marker="D",
        markersize=3.8,
        markerfacecolor="white",
        markeredgewidth=0.8,
        label="RHS benchmark",
    )
    runtime_fit = scaling["rhs_call_fit_K_at_least"]
    fitted = (
        1.0e6
        * runtime_fit["prefactor"]
        * runtime_periods ** runtime_fit["exponent"]
    )
    runtime_axis.loglog(
        runtime_periods,
        fitted,
        color="#202124",
        linestyle="--",
        linewidth=0.9,
        label=(
            rf"$K\geq {int(runtime_fit['minimum_K'])}$ fit: "
            rf"$K^{{{runtime_fit['exponent']:.2f}}}$"
        ),
    )
    runtime_axis.set_xlabel(r"$K$")
    runtime_axis.set_ylabel(r"RHS time ($\mu$s)")
    runtime_axis.legend(
        frameon=False,
        loc="best",
        handlelength=1.7,
        borderaxespad=0.2,
    )

    for panel_index, axis in enumerate(axes.ravel()):
        axis.text(
            *PANEL_TEXT_POSITION,
            f"({chr(ord('a') + panel_index)})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=PANEL_TEXT_SIZE,
            weight="semibold",
        )
        axis.tick_params(direction="in", top=True, right=True)
        for spine in axis.spines.values():
            spine.set_linewidth(0.75)

    physics_ticks = (20, 60, 100, 140, 200)
    for axis in (error_axis, leakage_axis, width_axis):
        axis.set_xlim(10, 210)
        axis.set_xticks(physics_ticks)

    error_axis.set_xlabel(r"$K$")
    leakage_axis.set_xlabel(r"$K$")
    error_axis.axvline(
        reference_period,
        color="0.72",
        linewidth=0.7,
        linestyle=":",
        zorder=0,
    )
    error_axis.text(
        0.97,
        0.53,
        rf"$K_{{\rm ref}}={reference_period}$",
        transform=error_axis.transAxes,
        ha="right",
        va="center",
        fontsize=7.5,
        color="0.35",
    )
    leakage_axis.text(
        0.88,
        0.52,
        rf"$K_{{\rm ref}}={reference_period}$",
        transform=leakage_axis.transAxes,
        ha="right",
        va="center",
        fontsize=7.5,
        color="0.35",
    )
    width_axis.text(
        0.97,
        0.53,
        rf"$K_{{\rm ref}}={reference_period}$",
        transform=width_axis.transAxes,
        ha="right",
        va="center",
        fontsize=7.5,
        color="0.35",
    )

    pdf_path = output_dir / "finite_k_convergence_and_runtime.pdf"
    png_path = output_dir / "finite_k_convergence_and_runtime.png"
    figure.savefig(
        pdf_path, bbox_inches="tight", pad_inches=0.02
    )
    figure.savefig(
        png_path, dpi=300, bbox_inches="tight", pad_inches=0.02
    )
    plt.close(figure)
    return pdf_path, png_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--periods",
        type=parse_periods,
        default=list(DEFAULT_PERIODS),
    )
    parser.add_argument(
        "--runtime-periods",
        type=parse_periods,
        default=list(DEFAULT_RUNTIME_PERIODS),
        help="K values used only for the isolated RHS timing benchmark",
    )
    parser.add_argument(
        "--runtime-fit-min-k",
        type=int,
        default=320,
        help="lower K cutoff for the overhead-reduced descriptive fit",
    )
    parser.add_argument("--reference-period", type=int, default=200)
    parser.add_argument(
        "--points", type=parse_points, default=["P1", "P6"]
    )
    parser.add_argument("--tmax", type=float, default=APP.CORE_TMAX)
    parser.add_argument("--samples", type=int, default=501)
    parser.add_argument("--window-radius", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    periods = list(args.periods)
    runtime_periods = list(args.runtime_periods)
    runtime_fit_min_k = int(args.runtime_fit_min_k)
    labels = list(args.points)
    reference_period = int(args.reference_period)
    tmax = float(args.tmax)
    samples = int(args.samples)
    window_radius = int(args.window_radius)
    if args.smoke:
        periods = [20, 40]
        runtime_periods = [20, 40, 80]
        runtime_fit_min_k = 20
        reference_period = 40
        labels = ["P1"]
        tmax = min(tmax, 0.5)
        samples = min(samples, 51)
        window_radius = min(window_radius, 6)
    validate_periods(periods, reference_period)
    if any(period < 4 or period % 4 for period in runtime_periods):
        raise ValueError("runtime-periods must be multiples of four")
    if runtime_fit_min_k <= 0:
        raise ValueError("runtime-fit-min-k must be positive")
    if tmax <= 0.0 or samples < 5:
        raise ValueError("positive tmax and at least five samples are required")
    if window_radius < 1 or 2 * window_radius + 1 >= min(periods):
        raise ValueError("window-radius must fit strictly inside every cell")

    times = np.linspace(0.0, tmax, samples)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays, rows, scaling = generate_data(
        periods,
        runtime_periods,
        runtime_fit_min_k,
        reference_period,
        labels,
        times,
        window_radius,
    )
    npz_path = output_dir / "finite_k_convergence_data.npz"
    json_path = output_dir / "finite_k_convergence_summary.json"
    csv_path = output_dir / "finite_k_convergence_table.csv"
    tex_path = output_dir / "finite_k_convergence_table.tex"
    np.savez_compressed(npz_path, **arrays)
    write_csv(csv_path, rows)
    write_latex_table(tex_path, rows, reference_period)

    payload = {
        "schema_version": 1,
        "configuration": {
            "periods": periods,
            "runtime_periods": runtime_periods,
            "runtime_fit_min_k": runtime_fit_min_k,
            "finite_reference_period": reference_period,
            "points": labels,
            "time_window": [0.0, tmax],
            "samples": samples,
            "local_window_radius": window_radius,
            "pole_bias": APP.POLE_BIAS,
            "initial_phi": APP.INITIAL_PHI,
            "initial_state": (
                "periodic Z2 background with one central excited-site defect"
            ),
            "tdvp_equation": "fig1fig2/tdvpfun.py::eom",
            "solver": APP.TDVP_METHOD,
            "rtol": APP.SOLVER_RTOL,
            "atol": APP.SOLVER_ATOL,
            "max_step": APP.SOLVER_MAX_STEP,
            "reference_background_period": APP.TDVP_REFERENCE_PERIOD,
            "leakage": (
                "Gamma(t) from the exact spin-half finite-K expression"
            ),
        },
        "runtime_scaling": scaling,
        "rows": rows,
        "suggested_figure_caption": (
            "Finite-cell convergence and empirical runtime scaling for the "
            "P1 and P6 application protocols. (a) RMS difference of the "
            "normalized defect response in the window |i-i0|<=8 relative to "
            "the largest finite cell Kref=200. (b) Background-subtracted "
            "extensive residual weight K overline[(Gamma_K^2-Gamma_bg^2)], "
            "where Gamma_bg is evaluated on the separately evolved period-two "
            "background. The small negative P6 value means that the defect "
            "slightly reduces the variational residual relative to that "
            "background; neither Gamma_K^2 nor Gamma_bg^2 is negative. "
            "(c) Time-averaged RMS width of the normalized defect response. "
            "(d) Median wall time of one vectorized TDVP right-hand-side "
            "evaluation; the dashed line is a descriptive fit over K>=320. "
            "All physical parameters, initial-state regularization, time "
            "window, and solver tolerances are held fixed as K changes."
        ),
        "interpretation_limits": [
            (
                "K_ref is the largest finite periodic cell in this scan, not "
                "a proof of the K-to-infinity limit."
            ),
            (
                "The leakage is intensive and includes the repeated Z2 "
                "background; the contribution of one defect is diluted as K "
                "increases."
            ),
            (
                "The plotted leakage quantity subtracts the period-two "
                "background at the level of Gamma squared and multiplies by K; "
                "this is the convergent extensive residual weight associated "
                "with the finite defect region."
            ),
            (
                "Its small negative value at P6 means that the defect reduces "
                "the residual relative to the period-two background; it does "
                "not imply a negative leakage or negative residual norm."
            ),
            (
                "Runtime fits are empirical for the current vectorized Python/"
                "NumPy implementation and this machine."
            ),
            (
                "Adaptive-solver wall time also depends on the number of "
                "accepted/rejected steps, so the isolated RHS benchmark is "
                "reported separately."
            ),
        ],
        "files": {
            "npz": str(npz_path),
            "csv": str(csv_path),
            "latex_table": str(tex_path),
        },
    }
    json_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    figure_paths: list[str] = []
    if not args.no_plot:
        pdf_path, png_path = plot_results(
            output_dir,
            periods,
            reference_period,
            labels,
            rows,
            arrays,
            scaling,
        )
        figure_paths = [str(pdf_path), str(png_path)]

    print(
        json.dumps(
            {
                "status": "ok",
                "smoke": bool(args.smoke),
                "output_dir": str(output_dir),
                "finite_reference_period": reference_period,
                "rhs_runtime_fit": scaling[
                    "rhs_call_fit_K_at_least"
                ],
                "figures": figure_paths,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
