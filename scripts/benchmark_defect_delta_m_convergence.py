#!/usr/bin/env python3
"""Benchmark finite-K and pole regularization with defect-site delta M.

The observable is the signed two-site averaged occupation

    M_i(t) = ( <n_i(t)> + <n_{i+1}(t)> ) / 2,
    delta M_i0(t) = M_i0,pert(t) - M_i0,unpert^ED(t).

The unperturbed ED trajectory starts at the strict Z2 product state.  The ED
reference uses the strict central-defect product state and the same
unperturbed trajectory.  Perturbed TDVP trajectories use theta=epsilon or
pi-epsilon, theta=epsilon at the defect, and phi_i=0.  Thus every K and
epsilon is displayed through the same local, unnormalized physical quantity.
Convergence is assessed from plateaus and overlap of delta M itself, not from
pairwise differences between parameter values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG1_DIR = ROOT / "fig1fig2"
if str(FIG1_DIR) not in sys.path:
    sys.path.insert(0, str(FIG1_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

MATPLOTLIB_CACHE = ROOT / "tmp" / "matplotlib-defect-delta-m"
MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE))
GENERAL_CACHE = ROOT / "tmp" / "cache-defect-delta-m"
GENERAL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(GENERAL_CACHE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import benchmark_arbitrary_k_convergence as polar  # noqa: E402


SAMPLED_PERIODS = (
    10,
    12,
    14,
    16,
    18,
    20,
    24,
    28,
    32,
    40,
    60,
    80,
    100,
    140,
    200,
)
REGULARIZATION_PERIOD = 100
REGULARIZATION_VALUES = tuple(np.logspace(-1.0, -7.0, 13))
STRICT_AUDIT_VALUES = (1.0e-3, 1.0e-4, 1.0e-5)
TRACE_EPSILONS = (1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5)
PROBE_TIMES = (2.0, 5.0, 10.0)
ED_LENGTH = 24
ED_SOLVER_NAME = "dop853"
POLE_BIAS = polar.POLE_BIAS
INITIAL_PHI = polar.INITIAL_PHI
TMAX = 10.0
SAMPLE_COUNT = 1001
METHOD = polar.METHOD
RTOL = polar.RTOL
ATOL = polar.ATOL
MAX_STEP = polar.MAX_STEP
STRICT_RTOL = polar.STRICT_RTOL
STRICT_ATOL = polar.STRICT_ATOL
STRICT_MAX_STEP = polar.STRICT_MAX_STEP
STANDARD_TIMEOUT_SECONDS = 900.0
STRICT_TIMEOUT_SECONDS = 1200.0
SCHEMA_VERSION = 1

DEFAULT_OUTPUT_DIR = ROOT / "output" / "defect_delta_m_convergence"
DEFAULT_SELECTION_MANIFEST = polar.DEFAULT_SELECTION_MANIFEST
DEFAULT_LEAKAGE_MAP = polar.DEFAULT_LEAKAGE_MAP
STYLE_FILE = Path(__file__).with_name("hzg-paper.mplstyle")

FIGURE_SIZE = (3.40, 4.70)
FIGURE_ADJUST = {
    "left": 0.195,
    "right": 0.975,
    "bottom": 0.105,
    "top": 0.905,
    "hspace": 0.34,
}

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
DISPLAY_LABEL = {"P1": "I2", "P6": "II5"}
TRACE_STYLE = {
    1.0e-2: ("#56B4E9", "--"),
    1.0e-3: ("#009E73", "-"),
    1.0e-4: ("#E69F00", "-."),
    1.0e-5: ("#CC79A7", ":"),
}
K_TRACE_STYLE = {
    20: ("#56B4E9", "--"),
    40: ("#E69F00", "-."),
    100: ("#009E73", "-"),
    200: ("#CC79A7", ":"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def two_site_average_from_tdvp(trajectory: np.ndarray) -> np.ndarray:
    """Return M_i0=(<n_i0>+<n_i0+1>)/2 for a TDVP trajectory."""
    period = trajectory.shape[0] // 2
    profile_sum = polar.bond_smoothed_population(trajectory)
    return 0.5 * np.asarray(profile_sum[polar.defect_site(period)], dtype=float)


def probe_values(values: np.ndarray, times: np.ndarray) -> dict[float, float]:
    """Return delta M itself at the declared physical probe times."""
    return {
        probe: float(np.interp(probe, times, values))
        for probe in PROBE_TIMES
        if times[0] <= probe <= times[-1]
    }


def _cache_matches(
    data: np.lib.npyio.NpzFile,
    *,
    kind: str,
    label: str,
    mu: float,
    chi: float,
    times: np.ndarray,
    period: int | None = None,
    epsilon: float | None = None,
    strict: bool | None = None,
) -> bool:
    try:
        if int(data["schema_version"]) != SCHEMA_VERSION:
            return False
        if str(data["kind"].item()) != kind:
            return False
        if str(data["label"].item()) != label:
            return False
        if not np.isclose(float(data["mu"]), mu, rtol=0.0, atol=1.0e-14):
            return False
        if not np.isclose(float(data["chi"]), chi, rtol=0.0, atol=1.0e-14):
            return False
        if not np.array_equal(np.asarray(data["times"]), times):
            return False
        if period is not None and int(data["period"]) != period:
            return False
        if epsilon is not None and not np.isclose(
            float(data["epsilon"]), epsilon, rtol=1.0e-13, atol=0.0
        ):
            return False
        if strict is not None and bool(int(data["strict"])) != strict:
            return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def exact_ed_local_response(
    label: str,
    mu: float,
    chi: float,
    times: np.ndarray,
    cache_path: Path,
    *,
    force: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return strict ED background, defect M_i0, and their signed difference."""
    if cache_path.exists() and not force:
        with np.load(cache_path, allow_pickle=False) as cached:
            if _cache_matches(
                cached,
                kind="strict_ed_product_states",
                label=label,
                mu=mu,
                chi=chi,
                times=times,
            ):
                return (
                    np.asarray(cached["m_unperturbed"], dtype=float),
                    np.asarray(cached["m_perturbed"], dtype=float),
                    np.asarray(cached["delta_m_ed"], dtype=float),
                    float(cached["wall_time_seconds"]),
                )

    import reproduce_fig1_fig2 as paper_figures

    started = time.perf_counter()
    profile_perturbed, profile_unperturbed = (
        paper_figures.ed_translation_invariant_pair(
            mu,
            chi,
            times,
            length=ED_LENGTH,
            bias=0.0,
            initial_phi=INITIAL_PHI,
            solver_name=ED_SOLVER_NAME,
            rtol=RTOL,
            atol=ATOL,
            max_step=MAX_STEP,
        )
    )
    i0 = paper_figures.defect_site(ED_LENGTH)
    # The shared plotting helper returns <n_i+n_{i+1}>.  Divide by two to
    # implement the two-site average used by this convergence test.
    m_perturbed = 0.5 * np.asarray(profile_perturbed[i0], dtype=float)
    m_unperturbed = 0.5 * np.asarray(profile_unperturbed[i0], dtype=float)
    delta_m_ed = m_perturbed - m_unperturbed
    wall_time = time.perf_counter() - started
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        schema_version=SCHEMA_VERSION,
        kind="strict_ed_product_states",
        label=label,
        mu=mu,
        chi=chi,
        times=times,
        length=ED_LENGTH,
        m_unperturbed=m_unperturbed,
        m_perturbed=m_perturbed,
        delta_m_ed=delta_m_ed,
        wall_time_seconds=wall_time,
        solver=ED_SOLVER_NAME,
        rtol=RTOL,
        atol=ATOL,
        max_step=MAX_STEP,
        initial_state="strict_theta_0_or_pi_phi_0",
    )
    return m_unperturbed, m_perturbed, delta_m_ed, wall_time


def tdvp_perturbed_local_m(
    label: str,
    mu: float,
    chi: float,
    times: np.ndarray,
    period: int,
    epsilon: float,
    cache_path: Path,
    *,
    strict: bool,
    force: bool,
    timeout_seconds: float,
) -> tuple[np.ndarray, float]:
    """Return regularized perturbed TDVP M_i0 with resumable caching."""
    if cache_path.exists() and not force:
        with np.load(cache_path, allow_pickle=False) as cached:
            if _cache_matches(
                cached,
                kind="regularized_tdvp_defect",
                label=label,
                mu=mu,
                chi=chi,
                times=times,
                period=period,
                epsilon=epsilon,
                strict=strict,
            ):
                return (
                    np.asarray(cached["m_perturbed"], dtype=float),
                    float(cached["wall_time_seconds"]),
                )

    rtol, atol, max_step = (
        (STRICT_RTOL, STRICT_ATOL, STRICT_MAX_STEP)
        if strict
        else (RTOL, ATOL, MAX_STEP)
    )
    started = time.perf_counter()
    with polar.wall_time_limit(timeout_seconds):
        trajectory = polar.integrate_tdvp(
            period,
            mu,
            chi,
            times,
            with_defect=True,
            regularization=epsilon,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
    m_perturbed = two_site_average_from_tdvp(trajectory)
    wall_time = time.perf_counter() - started
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        schema_version=SCHEMA_VERSION,
        kind="regularized_tdvp_defect",
        label=label,
        mu=mu,
        chi=chi,
        times=times,
        period=period,
        epsilon=epsilon,
        strict=int(strict),
        m_perturbed=m_perturbed,
        wall_time_seconds=wall_time,
        solver=METHOD,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        initial_phi=INITIAL_PHI,
    )
    return m_perturbed, wall_time


def epsilon_token(value: float) -> str:
    return f"{value:.12e}".replace("+", "").replace("-", "m").replace(".", "p")


def status_name(status: int) -> str:
    return {0: "success", 1: "timeout", 2: "not_scheduled", 3: "failed"}[int(status)]


def is_selected(value: float, selected: tuple[float, ...]) -> bool:
    return any(np.isclose(value, item, rtol=1.0e-12, atol=0.0) for item in selected)


def generate_data(
    times: np.ndarray,
    output_dir: Path,
    *,
    force: bool,
    continue_after_timeout: bool,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]], list[dict[str, object]]]:
    """Compute exact ED, finite-K, and finite-epsilon defect-site data."""
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=int),
        "times": times,
        "sampled_periods": np.asarray(SAMPLED_PERIODS, dtype=int),
        "regularization_values": np.asarray(REGULARIZATION_VALUES),
        "regularization_period": np.asarray(REGULARIZATION_PERIOD, dtype=int),
        "ed_length": np.asarray(ED_LENGTH, dtype=int),
        "working_epsilon": np.asarray(POLE_BIAS),
    }
    k_rows: list[dict[str, object]] = []
    epsilon_rows: list[dict[str, object]] = []
    cache_dir = output_dir / "cache"

    for label, (mu, chi) in polar.POINTS.items():
        ed_cache = cache_dir / f"{label}_strict_ed_L{ED_LENGTH}.npz"
        m_background, m_ed_perturbed, delta_m_ed, ed_wall = exact_ed_local_response(
            label, mu, chi, times, ed_cache, force=force
        )
        arrays[f"{label}_m_ed_unperturbed"] = m_background
        arrays[f"{label}_m_ed_perturbed"] = m_ed_perturbed
        arrays[f"{label}_delta_m_ed"] = delta_m_ed
        arrays[f"{label}_ed_wall_time"] = np.asarray(ed_wall)
        print(f"{label}: strict ED reference ready in {ed_wall:.2f} s", flush=True)

        k_wall_times = []
        k_delta_series: list[np.ndarray] = []
        for period in SAMPLED_PERIODS:
            cache_path = cache_dir / (
                f"{label}_K{period}_eps{epsilon_token(POLE_BIAS)}_standard.npz"
            )
            m_tdvp, wall_time = tdvp_perturbed_local_m(
                label,
                mu,
                chi,
                times,
                period,
                POLE_BIAS,
                cache_path,
                strict=False,
                force=force,
                timeout_seconds=STANDARD_TIMEOUT_SECONDS,
            )
            delta_m = m_tdvp - m_background
            arrays[f"{label}_K{period}_delta_m"] = delta_m
            k_delta_series.append(delta_m)
            k_wall_times.append(wall_time)
            print(
                f"{label}, K={period}: delta-M(t={times[-1]:g})="
                f"{delta_m[-1]:.6e} ({wall_time:.2f} s)",
                flush=True,
            )
        arrays[f"{label}_K_wall_time"] = np.asarray(k_wall_times)
        for index, period in enumerate(SAMPLED_PERIODS):
            sampled_delta_m = probe_values(k_delta_series[index], times)
            k_rows.append(
                {
                    "point": label,
                    "mu": mu,
                    "chi": chi,
                    "K": period,
                    "epsilon": POLE_BIAS,
                    "ED_L_for_unperturbed_background": ED_LENGTH,
                    **{
                        f"delta_m_t{probe:g}": value
                        for probe, value in sampled_delta_m.items()
                    },
                    "wall_time_seconds": k_wall_times[index],
                }
            )

        count = len(REGULARIZATION_VALUES)
        delta_standard: list[np.ndarray | None] = [None] * count
        delta_strict: list[np.ndarray | None] = [None] * count
        standard_status = np.full(count, 2, dtype=int)
        strict_status = np.full(count, 2, dtype=int)
        standard_wall = np.full(count, np.nan)
        strict_wall = np.full(count, np.nan)
        skip_smaller = False

        for index, epsilon in enumerate(REGULARIZATION_VALUES):
            if skip_smaller:
                continue
            cache_path = cache_dir / (
                f"{label}_K{REGULARIZATION_PERIOD}_eps{epsilon_token(epsilon)}_standard.npz"
            )
            try:
                m_tdvp, standard_wall[index] = tdvp_perturbed_local_m(
                    label,
                    mu,
                    chi,
                    times,
                    REGULARIZATION_PERIOD,
                    epsilon,
                    cache_path,
                    strict=False,
                    force=force,
                    timeout_seconds=STANDARD_TIMEOUT_SECONDS,
                )
                delta_standard[index] = m_tdvp - m_background
                standard_status[index] = 0
            except polar.IntegrationTimeout:
                standard_status[index] = 1
                standard_wall[index] = STANDARD_TIMEOUT_SECONDS
                skip_smaller = not continue_after_timeout
                print(
                    f"{label}, epsilon={epsilon:.3e}: standard timeout",
                    flush=True,
                )
                continue
            except RuntimeError as error:
                standard_status[index] = 3
                skip_smaller = True
                print(
                    f"{label}, epsilon={epsilon:.3e}: standard failed: {error}",
                    flush=True,
                )
                continue

            print(
                f"{label}, epsilon={epsilon:.3e}: delta-M(t={times[-1]:g})="
                f"{delta_standard[index][-1]:.6e} "
                f"({standard_wall[index]:.2f} s)",
                flush=True,
            )

            if not is_selected(epsilon, STRICT_AUDIT_VALUES):
                continue
            strict_cache = cache_dir / (
                f"{label}_K{REGULARIZATION_PERIOD}_eps{epsilon_token(epsilon)}_strict.npz"
            )
            try:
                m_strict, strict_wall[index] = tdvp_perturbed_local_m(
                    label,
                    mu,
                    chi,
                    times,
                    REGULARIZATION_PERIOD,
                    epsilon,
                    strict_cache,
                    strict=True,
                    force=force,
                    timeout_seconds=STRICT_TIMEOUT_SECONDS,
                )
                delta_strict[index] = m_strict - m_background
                strict_status[index] = 0
            except polar.IntegrationTimeout:
                strict_status[index] = 1
                strict_wall[index] = STRICT_TIMEOUT_SECONDS
                print(
                    f"{label}, epsilon={epsilon:.3e}: strict timeout",
                    flush=True,
                )
            except RuntimeError as error:
                strict_status[index] = 3
                print(
                    f"{label}, epsilon={epsilon:.3e}: strict failed: {error}",
                    flush=True,
                )

        delta_matrix = np.full((count, len(times)), np.nan)
        strict_matrix = np.full((count, len(times)), np.nan)

        for index, epsilon in enumerate(REGULARIZATION_VALUES):
            candidate = delta_standard[index]
            if candidate is not None:
                delta_matrix[index] = candidate
            strict_candidate = delta_strict[index]
            if candidate is not None and strict_candidate is not None:
                strict_matrix[index] = strict_candidate
            sampled_delta_m = {
                probe: np.nan for probe in PROBE_TIMES if probe <= times[-1]
            }
            if candidate is not None:
                sampled_delta_m.update(probe_values(candidate, times))
            sampled_strict_delta_m = {
                probe: np.nan for probe in PROBE_TIMES if probe <= times[-1]
            }
            if strict_candidate is not None:
                sampled_strict_delta_m.update(
                    probe_values(strict_candidate, times)
                )
            epsilon_rows.append(
                {
                    "point": label,
                    "mu": mu,
                    "chi": chi,
                    "K": REGULARIZATION_PERIOD,
                    "ED_L": ED_LENGTH,
                    "epsilon": epsilon,
                    "standard_status": status_name(standard_status[index]),
                    "standard_wall_time_seconds": standard_wall[index],
                    "strict_status": status_name(strict_status[index]),
                    "strict_wall_time_seconds": strict_wall[index],
                    **{
                        f"delta_m_t{probe:g}": value
                        for probe, value in sampled_delta_m.items()
                    },
                    **{
                        f"strict_delta_m_t{probe:g}": value
                        for probe, value in sampled_strict_delta_m.items()
                    },
                }
            )

        arrays[f"{label}_epsilon_delta_m"] = delta_matrix
        arrays[f"{label}_strict_delta_m"] = strict_matrix
        arrays[f"{label}_standard_status"] = standard_status
        arrays[f"{label}_strict_status"] = strict_status
        arrays[f"{label}_standard_wall_time"] = standard_wall
        arrays[f"{label}_strict_wall_time"] = strict_wall

    return arrays, k_rows, epsilon_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_convergence(output_dir: Path, arrays: dict[str, np.ndarray]) -> tuple[Path, Path]:
    """Plot delta M itself versus K, epsilon, and time."""
    plt.style.use(STYLE_FILE)
    figure, axes = plt.subplots(2, 1, figsize=FIGURE_SIZE)
    figure.subplots_adjust(**FIGURE_ADJUST)
    k_axis, p1_epsilon_axis = axes

    times = arrays["times"]
    periods = arrays["sampled_periods"]
    epsilons = arrays["regularization_values"]
    probe_markers = {2.0: "o", 5.0: "s", 10.0: "^"}
    probe_linestyles = {2.0: ":", 5.0: "--", 10.0: "-"}
    for label in ("P1", "P6"):
        style = CASE_STYLE[label]
        for probe in PROBE_TIMES:
            if probe > times[-1]:
                continue
            k_values = np.asarray(
                [
                    np.interp(probe, times, arrays[f"{label}_K{int(period)}_delta_m"])
                    for period in periods
                ]
            )
            common = dict(
                color=style["color"],
                marker=probe_markers[probe],
                markersize=3.6,
                markerfacecolor=(
                    style["color"] if label == "P1" else "white"
                ),
                markeredgecolor=style["color"],
                markeredgewidth=0.75,
                linestyle=probe_linestyles[probe],
                linewidth=0.9,
            )
            k_axis.plot(periods, k_values, **common)

    k_axis.set_xscale("log")
    k_axis.set_xlim(9.5, 210)
    k_ticks = (10, 20, 40, 80, 140, 200)
    k_axis.set_xticks(k_ticks)
    k_axis.set_xticklabels(tuple(str(value) for value in k_ticks))
    k_axis.set_xlabel(r"TDVP cell size $K$")
    k_axis.set_ylabel(r"$\delta M_{i_0}(t)$")
    point_handles = [
        Line2D(
            [0],
            [0],
            color=CASE_STYLE[label]["color"],
            label=DISPLAY_LABEL[label],
        )
        for label in ("P1", "P6")
    ]
    time_handles = [
        Line2D(
            [0],
            [0],
            color="0.25",
            marker=probe_markers[probe],
            linestyle=probe_linestyles[probe],
            label=rf"$t={probe:g}$",
        )
        for probe in PROBE_TIMES
        if probe <= times[-1]
    ]
    k_axis.legend(
        handles=point_handles + time_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.54, 1.155),
        ncol=5,
        handlelength=1.15,
        handletextpad=0.28,
        columnspacing=0.48,
        fontsize=5.8,
    )

    matrix = arrays["P1_epsilon_delta_m"]
    for epsilon in TRACE_EPSILONS:
        matches = np.flatnonzero(
            np.isclose(epsilons, epsilon, rtol=1.0e-12, atol=0.0)
        )
        if matches.size != 1:
            continue
        trace = matrix[matches[0]]
        if not np.all(np.isfinite(trace)):
            continue
        color, linestyle = TRACE_STYLE[epsilon]
        p1_epsilon_axis.plot(
            times,
            trace,
            color=color,
            linestyle=linestyle,
            linewidth=1.35 if np.isclose(epsilon, POLE_BIAS) else 1.0,
            label=rf"$\epsilon=10^{{{int(round(np.log10(epsilon)))}}}$",
        )
    p1_epsilon_axis.axhline(0.0, color="0.72", linewidth=0.5)
    p1_epsilon_axis.set_xlim(float(times[0]), float(times[-1]))
    p1_epsilon_axis.set_xlabel(r"time $t$")
    p1_epsilon_axis.set_ylabel(r"$\delta M_{i_0}(t)$")
    p1_epsilon_axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.47, 1.0),
        ncol=2,
        fontsize=7.0,
        handlelength=1.7,
        columnspacing=0.9,
    )

    for panel_label, axis in zip(("(a)", "(b)"), axes):
        axis.text(
            0.025,
            0.95,
            panel_label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.8,
            fontweight="semibold",
        )
        axis.tick_params(direction="in", top=True, right=True)
        axis.grid(axis="y", linestyle=":", linewidth=0.45, alpha=0.33)
        for spine in axis.spines.values():
            spine.set_linewidth(0.75)

    pdf_path = output_dir / "defect_delta_m_convergence.pdf"
    png_path = output_dir / "defect_delta_m_convergence.png"
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    return pdf_path, png_path


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--selection-manifest", type=Path, default=DEFAULT_SELECTION_MANIFEST
    )
    parser.add_argument("--leakage-map", type=Path, default=DEFAULT_LEAKAGE_MAP)
    parser.add_argument("--samples", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--tmax", type=float, default=TMAX)
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute protocol-validated ED and TDVP trajectory caches",
    )
    parser.add_argument(
        "--continue-after-timeout",
        action="store_true",
        help="attempt smaller epsilon after a standard-solver timeout",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    polar.configure_points(
        args.selection_manifest.resolve(), args.leakage_map.resolve()
    )
    if args.samples < 5 or args.tmax <= 0.0:
        raise ValueError("positive tmax and at least five samples are required")

    times = np.linspace(0.0, args.tmax, args.samples)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays, k_rows, epsilon_rows = generate_data(
        times,
        output_dir,
        force=args.force,
        continue_after_timeout=args.continue_after_timeout,
    )

    npz_path = output_dir / "defect_delta_m_convergence.npz"
    json_path = output_dir / "defect_delta_m_convergence.json"
    k_csv_path = output_dir / "finite_k_defect_delta_m_convergence.csv"
    epsilon_csv_path = output_dir / "epsilon_defect_delta_m_convergence.csv"
    np.savez_compressed(npz_path, **arrays)
    write_csv(k_csv_path, k_rows)
    write_csv(epsilon_csv_path, epsilon_rows)
    pdf_path, png_path = plot_convergence(output_dir, arrays)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "definition": {
            "M_i": "(<n_i>+<n_{i+1}>)/2",
            "delta_M_TDVP": (
                "M_i0_perturbed_TDVP(K,epsilon)-M_i0_unperturbed_strict_ED"
            ),
            "delta_M_ED": (
                "M_i0_perturbed_strict_ED-M_i0_unperturbed_strict_ED"
            ),
            "signed": True,
            "spatial_normalization": False,
        },
        "configuration": {
            "points": {
                label: {"mu": mu, "chi": chi}
                for label, (mu, chi) in polar.POINTS.items()
            },
            "selection_manifest": str(args.selection_manifest.resolve()),
            "selection_manifest_sha256": sha256(args.selection_manifest.resolve()),
            "leakage_map": str(args.leakage_map.resolve()),
            "leakage_map_sha256": sha256(args.leakage_map.resolve()),
            "time_window": [0.0, args.tmax],
            "samples": args.samples,
            "ED_L": ED_LENGTH,
            "ED_initial_state": "strict theta=0 or pi, phi=0 product state",
            "sampled_K": list(SAMPLED_PERIODS),
            "working_epsilon": POLE_BIAS,
            "regularization_K": REGULARIZATION_PERIOD,
            "sampled_epsilon": list(REGULARIZATION_VALUES),
            "initial_phi": INITIAL_PHI,
            "solver": METHOD,
            "rtol": RTOL,
            "atol": ATOL,
            "max_step": MAX_STEP,
            "strict_audit_epsilon": list(STRICT_AUDIT_VALUES),
            "standard_timeout_seconds": STANDARD_TIMEOUT_SECONDS,
            "strict_timeout_seconds": STRICT_TIMEOUT_SECONDS,
        },
        "metrics": {
            "convergence_criterion": (
                "delta_M_i0 itself at t=2,5,10 and overlap of its full time "
                "traces; no adjacent-parameter difference is used"
            ),
            "ED_curve": (
                "strict ED defect response is displayed only as a physical "
                "comparison, not as the convergence metric"
            ),
        },
        "finite_K_rows": k_rows,
        "epsilon_rows": epsilon_rows,
        "files": {
            "npz": str(npz_path),
            "finite_K_csv": str(k_csv_path),
            "epsilon_csv": str(epsilon_csv_path),
            "pdf": str(pdf_path),
            "png": str(png_path),
        },
    }
    json_path.write_text(
        json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir),
                "sampled_K": list(SAMPLED_PERIODS),
                "sampled_epsilon": list(REGULARIZATION_VALUES),
                "pdf": str(pdf_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
