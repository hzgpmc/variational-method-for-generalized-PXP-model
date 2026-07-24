#!/usr/bin/env python3
"""Local-response convergence of the arbitrary-K application TDVP.

Only the TDVP cell length is varied.  All physics and solver choices match the
P1/P6 application protocol used by ``fig1fig2/reproduce_fig1_fig2.py``:

* P1=(-0.025, 0.150) and P6=(-0.120, 1.900);
* the same pole-regularized Z2 background with one central defect;
* theta_pole=phi_i=1e-3;
* 0 <= t <= 10;
* DOP853 with rtol=1e-9, atol=1e-11 and max_step=0.02.

The sampled cells are K=20,40,60,80,100,140,200.  A separate K_ref=500
trajectory supplies a finite numerical reference but is not drawn on the
horizontal axis.  For each K, the defect response is defined relative to a
separately evolved period-two Z2 background and normalized over the cell.
The reported convergence metric is

    epsilon_loc(K;K_ref)
      = rms[P_i^(K)(t)-P_i^(K_ref)(t)]

over all saved times and the fixed 17-site window |i-i0|<=8.

This is a finite-reference convergence check for the stated observable and
time window; it is not a proof of a thermodynamic-limit error bound.

Run in the requested environment:

    conda run -n quimb python scripts/benchmark_arbitrary_k_convergence.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

# The quimb environment has the numerical packages needed here but not
# QuSpin/joblib.  Keep the script independent of those unused legacy imports.
MATPLOTLIB_CACHE = ROOT / "tmp" / "matplotlib-finite-k-local"
MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE))
GENERAL_CACHE = ROOT / "tmp" / "cache-finite-k-local"
GENERAL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(GENERAL_CACHE))

import matplotlib.pyplot as plt  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402


# ---------------------------------------------------------------------------
# One source of truth for the scientific comparison.
# These values exactly match fig1fig2/reproduce_fig1_fig2.py.
# ---------------------------------------------------------------------------
POINTS = {
    "P1": (-0.025, 0.150),
    "P6": (-0.120, 1.900),
}
SAMPLED_PERIODS = (20, 40, 60, 80, 100, 140, 200)
REFERENCE_PERIOD = 500
BACKGROUND_PERIOD = 2
POLE_BIAS = 1.0e-3
INITIAL_PHI = 1.0e-3
TMAX = 10.0
SAMPLE_COUNT = 1001
METHOD = "DOP853"
RTOL = 1.0e-9
ATOL = 1.0e-11
MAX_STEP = 0.02
WINDOW_RADIUS = 8

DEFAULT_OUTPUT_DIR = ROOT / "output" / "finite_k_convergence"
STYLE_FILE = Path(__file__).with_name("hzg-paper.mplstyle")


# ---------------------------------------------------------------------------
# HZG single-column figure controls.
# Increasing left/bottom creates more room for axis labels.  The x limits are
# deliberately fixed at [20,200]; K_ref=500 appears only as an annotation.
# ---------------------------------------------------------------------------
FIGURE_SIZE = (3.40, 2.65)
FIGURE_ADJUST = {
    "left": 0.19,
    "right": 0.985,
    "bottom": 0.18,
    "top": 0.975,
}
X_LIMITS = (20.0, 200.0)
X_TICKS = (20, 60, 100, 140, 200)
REFERENCE_TEXT_POSITION = (0.96, 0.57)

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


def eta_values(theta: np.ndarray) -> np.ndarray:
    """Evaluate the exact spin-1/2 ``eta_i`` recurrence in O(K) work.

    This is the self-contained form of the recurrence used by
    ``fig1fig2/tdvpfun.py``.  Keeping it here avoids importing that legacy
    module, whose unrelated QuSpin/joblib dependencies are unavailable in the
    requested ``quimb`` environment.
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
    period: int, *, with_defect: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Return the common pole-regularized Z2 initial condition."""
    theta = np.asarray(
        [POLE_BIAS, np.pi - POLE_BIAS] * (period // 2),
        dtype=float,
    )
    if with_defect:
        theta[defect_site(period)] = POLE_BIAS
    phi = np.full(period, INITIAL_PHI, dtype=float)
    return theta, phi


def integrate_tdvp(
    period: int,
    mu: float,
    chi: float,
    times: np.ndarray,
    *,
    with_defect: bool,
) -> np.ndarray:
    """Integrate one trajectory with the fixed application solver protocol."""
    theta0, phi0 = initial_angles(period, with_defect=with_defect)
    solution = solve_ivp(
        tdvp_rhs,
        (float(times[0]), float(times[-1])),
        np.concatenate((theta0, phi0)),
        t_eval=times,
        method=METHOD,
        args=(mu, chi),
        rtol=RTOL,
        atol=ATOL,
        max_step=MAX_STEP,
    )
    expected_shape = (2 * period, len(times))
    if not solution.success or solution.y.shape != expected_shape:
        raise RuntimeError(
            f"TDVP failed for K={period}, (mu,chi)=({mu},{chi}): "
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
) -> np.ndarray:
    """Compute the fixed local response window for one finite cell."""
    trajectory = integrate_tdvp(
        period, mu, chi, times, with_defect=True
    )
    defect_profile = bond_smoothed_population(trajectory)
    background = np.tile(
        background_period_two, (period // BACKGROUND_PERIOD, 1)
    )
    response = normalized_defect_response(defect_profile, background)
    return centered_window(response, defect_site(period), radius)


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


def generate_data(
    times: np.ndarray, radius: int
) -> tuple[dict[str, np.ndarray], list[dict[str, float | int | str]]]:
    """Generate K_ref and sampled-K windows for P1 and P6."""
    arrays: dict[str, np.ndarray] = {
        "times": times,
        "sampled_periods": np.asarray(SAMPLED_PERIODS, dtype=int),
        "reference_period": np.asarray(REFERENCE_PERIOD, dtype=int),
        "window_offsets": np.arange(-radius, radius + 1),
    }
    rows: list[dict[str, float | int | str]] = []

    for label, (mu, chi) in POINTS.items():
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
        arrays[f"{label}_Kref{REFERENCE_PERIOD}_response_window"] = reference

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
            rows.append(
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

    return arrays, rows


def write_csv(
    path: Path, rows: list[dict[str, float | int | str]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_convergence(
    output_dir: Path,
    arrays: dict[str, np.ndarray],
) -> tuple[Path, Path]:
    """Create the single HZG-style local-response convergence panel."""
    plt.style.use(STYLE_FILE)
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    figure.subplots_adjust(**FIGURE_ADJUST)

    periods = arrays["sampled_periods"]
    for label in ("P1", "P6"):
        style = CASE_STYLE[label]
        axis.plot(
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

    axis.set_yscale("log")
    axis.set_xlim(*X_LIMITS)  # Requested horizontal range ends exactly at 200.
    axis.set_xticks(X_TICKS)
    axis.margins(y=0.14)
    axis.set_xlabel(r"$K$")
    axis.set_ylabel(r"$\epsilon_{\rm loc}(K;K_{\rm ref})$")
    axis.text(
        *REFERENCE_TEXT_POSITION,
        rf"$K_{{\rm ref}}={REFERENCE_PERIOD}$",
        transform=axis.transAxes,
        ha="right",
        va="center",
        fontsize=7.6,
        color="0.35",
    )
    axis.legend(
        frameon=False,
        loc="upper right",
        handlelength=1.8,
        borderaxespad=0.2,
    )
    axis.tick_params(direction="in", top=True, right=True)
    for spine in axis.spines.values():
        spine.set_linewidth(0.75)

    pdf_path = output_dir / "finite_k_local_response_convergence.pdf"
    png_path = output_dir / "finite_k_local_response_convergence.png"
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
    parser.add_argument("--samples", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--tmax", type=float, default=TMAX)
    parser.add_argument(
        "--window-radius", type=int, default=WINDOW_RADIUS
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.samples < 5 or args.tmax <= 0.0:
        raise ValueError("positive tmax and at least five samples are required")
    if not 1 <= args.window_radius < min(SAMPLED_PERIODS) // 2:
        raise ValueError("window-radius must fit inside the smallest cell")

    times = np.linspace(0.0, args.tmax, args.samples)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays, rows = generate_data(times, args.window_radius)

    npz_path = output_dir / "finite_k_local_response_convergence.npz"
    json_path = output_dir / "finite_k_local_response_convergence.json"
    csv_path = output_dir / "finite_k_local_response_convergence.csv"
    np.savez_compressed(npz_path, **arrays)
    write_csv(csv_path, rows)
    pdf_path, png_path = plot_convergence(output_dir, arrays)

    payload = {
        "schema_version": 2,
        "configuration": {
            "sampled_periods": list(SAMPLED_PERIODS),
            "finite_reference_period": REFERENCE_PERIOD,
            "x_axis_limits": list(X_LIMITS),
            "points": {
                label: {"mu": mu, "chi": chi}
                for label, (mu, chi) in POINTS.items()
            },
            "time_window": [0.0, args.tmax],
            "samples": args.samples,
            "window_radius": args.window_radius,
            "pole_bias": POLE_BIAS,
            "initial_phi": INITIAL_PHI,
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
        "metric": (
            "RMS difference between normalized defect responses at K and "
            "K_ref over all saved times and |i-i0|<=window_radius"
        ),
        "rows": rows,
        "suggested_caption": (
            "Finite-cell convergence of the normalized local defect response "
            "at P1 and P6. The RMS difference epsilon_loc is evaluated over "
            "the 17-site window |i-i0|<=8 and 0<=t<=10 relative to the finite "
            "reference K_ref=500. All Hamiltonian parameters, initial-state "
            "regularization, and solver tolerances are held fixed as K changes. "
            "K_ref is a finite numerical reference rather than an asserted "
            "thermodynamic limit."
        ),
        "files": {
            "npz": str(npz_path),
            "csv": str(csv_path),
            "pdf": str(pdf_path),
            "png": str(png_path),
        },
    }
    json_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir),
                "finite_reference_period": REFERENCE_PERIOD,
                "sampled_periods": list(SAMPLED_PERIODS),
                "pdf": str(pdf_path),
                "png": str(png_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
