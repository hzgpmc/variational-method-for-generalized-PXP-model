#!/usr/bin/env python3
"""Calibrate accumulated TDVP leakage against independent ED--TDVP errors.

This script does not modify the manuscript or the existing Fig. 1/Fig. 2
reproducer.  It reuses the matched-parameter ED profiles cached in
``fig1fig2/data/fig2_core.npz`` and adds only the calculations needed for an
appendix-quality leakage validation.

Two complementary checks are performed.

1. Matched observable check (default L=K=24, periodic boundary)
   The cached ED defect and reference profiles are compared with newly
   integrated K=24 TDVP profiles at the same P1--P6 parameters, initialization,
   boundary pattern, solver tolerances, and time grid.  The independent error
   is the site-RMS difference of the defect response

       P_i^defect(t) - P_i^reference(t).

   Because the response contains two evolved states, it is paired with

       Lambda_pair(t) = integral_0^t [Gamma_defect + Gamma_reference] dt.

2. Direct state-space check (default L=K=16, periodic boundary)
   Exact ED states are compared with the finite-ring MPS constructed from the
   TDVP angles.  The error is their projective Fubini--Study distance.  At each
   sampled point the finite-ring Schrödinger residual is also evaluated by a
   directional derivative of the normalized MPS.  This independently checks
   the manuscript normalization ``sqrt(L) Gamma`` and supplies a direct
   accumulated residual against which the state distance can be compared.

The TDVP contractions use the thermodynamic repetition of one K-site cell.
Thus K=L matches the spatial pattern and periodic boundary, but the analytic
Gamma still has exponentially small finite-ring corrections.  Those
corrections are measured rather than assumed away in the direct residual
check.

Run in the QuSpin environment:

    conda run -n quspin python \
        verification/validate_leakage_vs_ed_tdvp_error.py

For a quick end-to-end check:

    conda run -n quspin python \
        verification/validate_leakage_vs_ed_tdvp_error.py \
        --smoke-test --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG_DIR = ROOT / "fig1fig2"
DEFAULT_CORE_CACHE = FIG_DIR / "data" / "fig2_core.npz"
DEFAULT_OUTPUT_DIR = HERE / "output" / "leakage_error_validation"
MPLCONFIG_DIR = DEFAULT_OUTPUT_DIR / "mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

if str(FIG_DIR) not in sys.path:
    sys.path.insert(0, str(FIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import pearsonr, spearmanr

import EDfun
import pxpbasisS
import reproduce_fig1_fig2 as reproduction
import tdvpfun


CACHE_SCHEMA_VERSION = 1

HZG_STYLE = {
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "axes.linewidth": 0.75,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "legend.fontsize": 7.2,
    "lines.linewidth": 1.15,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 180,
    "savefig.dpi": 300,
}

POINT_COLORS = {
    "P1": "#0072B2",
    "P2": "#56B4E9",
    "P3": "#009E73",
    "P4": "#E69F00",
    "P5": "#D55E00",
    "P6": "#CC79A7",
}

POINT_STYLES = {
    "P1": "-",
    "P2": "--",
    "P3": "-.",
    "P4": ":",
    "P5": (0, (5, 1.5)),
    "P6": (0, (3, 1, 1, 1)),
}


def parse_labels(text: str) -> list[str]:
    """Parse and validate a comma-separated subset of P1--P6."""

    labels = [item.strip().upper() for item in text.split(",") if item.strip()]
    allowed = {label for _, _, label in reproduction.POINTS}
    unknown = sorted(set(labels).difference(allowed))
    if not labels or unknown:
        raise argparse.ArgumentTypeError(
            f"labels must be a nonempty subset of {sorted(allowed)}"
        )
    if len(labels) != len(set(labels)):
        raise argparse.ArgumentTypeError("point labels must not repeat")
    return labels


def cumulative_trapezoid(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral with zero at the first time."""

    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    if values.shape[-1] != times.size:
        raise ValueError("integration axis does not match time grid")
    increments = 0.5 * (values[..., 1:] + values[..., :-1]) * np.diff(times)
    zeros = np.zeros(values.shape[:-1] + (1,), dtype=float)
    return np.concatenate((zeros, np.cumsum(increments, axis=-1)), axis=-1)


def core_scalar(core: np.lib.npyio.NpzFile, name: str) -> Any:
    """Read one scalar from the existing core cache."""

    value = np.asarray(core[name])
    if value.shape != ():
        raise ValueError(f"core field {name!r} is not scalar")
    return value.item()


def selected_core_data(
    core_cache: Path,
    labels: list[str],
    tmax: float,
    stride: int,
) -> dict[str, np.ndarray | dict[str, Any]]:
    """Load, validate, subset, and stride the existing ED profile cache."""

    with np.load(core_cache, allow_pickle=False) as core:
        valid, reason = reproduction.validate_core_cache(core)
        if not valid:
            raise ValueError(f"incompatible core cache: {reason}")
        all_labels = [str(value) for value in core["labels"]]
        indices = np.asarray([all_labels.index(label) for label in labels])
        full_times = np.asarray(core["times"], dtype=float)
        time_indices = np.flatnonzero(full_times <= tmax + 1.0e-13)[::stride]
        if time_indices.size == 0 or time_indices[-1] != np.flatnonzero(
            full_times <= tmax + 1.0e-13
        )[-1]:
            time_indices = np.append(
                time_indices,
                np.flatnonzero(full_times <= tmax + 1.0e-13)[-1],
            )
        metadata = {
            "schema_version": int(core_scalar(core, "schema_version")),
            "N_ed": int(core_scalar(core, "N_ed")),
            "K_tdvp_original": int(core_scalar(core, "K_tdvp")),
            "pole_bias": float(core_scalar(core, "pole_bias")),
            "initial_phi": float(core_scalar(core, "initial_phi")),
            "tdvp_method": str(core_scalar(core, "tdvp_method")),
            "tdvp_rtol": float(core_scalar(core, "tdvp_rtol")),
            "tdvp_atol": float(core_scalar(core, "tdvp_atol")),
            "tdvp_max_step": float(core_scalar(core, "tdvp_max_step")),
            "ed_solver_name": str(core_scalar(core, "ed_solver_name")),
            "ed_rtol": float(core_scalar(core, "ed_rtol")),
            "ed_atol": float(core_scalar(core, "ed_atol")),
            "ed_max_step": float(core_scalar(core, "ed_max_step")),
            "response_reference": str(
                core_scalar(core, "response_reference")
            ),
        }
        return {
            "times": full_times[time_indices],
            "points": np.asarray(core["points"])[indices],
            "labels": np.asarray(core["labels"])[indices],
            "ed_profiles": np.asarray(core["ed_profiles"])[
                indices
            ][:, :, time_indices],
            "ed_reference_profiles": np.asarray(
                core["ed_reference_profiles"]
            )[indices][:, :, time_indices],
            "metadata": metadata,
        }


def tdvp_occupation(trajectory: np.ndarray) -> np.ndarray:
    """Return the site occupation of a spin-1/2 TDVP trajectory."""

    period = trajectory.shape[0] // 2
    theta = trajectory[:period]
    eta = tdvpfun.get_eta(theta)
    return 0.5 * eta * (1.0 - np.cos(theta))


def tdvp_pair_profile(trajectory: np.ndarray) -> np.ndarray:
    """Return n_i+n_{i+1}, the profile used in the existing application."""

    occupation = tdvp_occupation(trajectory)
    return occupation + np.roll(occupation, -1, axis=0)


def matched_observable_validation(
    selected: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Compute K=L TDVP trajectories against cached L-site ED profiles."""

    times = np.asarray(selected["times"], dtype=float)
    points = np.asarray(selected["points"], dtype=float)
    labels = np.asarray(selected["labels"])
    metadata = dict(selected["metadata"])
    length = int(metadata["N_ed"])
    bias = float(metadata["pole_bias"])
    initial_phi = float(metadata["initial_phi"])

    tdvp_profiles = []
    tdvp_reference_profiles = []
    gamma_defect = []
    gamma_reference = []
    solver_stats: dict[str, Any] = {}

    for (mu, chi), label_value in zip(points, labels, strict=True):
        label = str(label_value)
        trajectory = reproduction.integrate_tdvp(
            float(mu),
            float(chi),
            times,
            period=length,
            bias=bias,
            initial_phi=initial_phi,
            method=str(metadata["tdvp_method"]),
            rtol=float(metadata["tdvp_rtol"]),
            atol=float(metadata["tdvp_atol"]),
            max_step=float(metadata["tdvp_max_step"]),
            with_defect=True,
        )
        reference = reproduction.integrate_tdvp(
            float(mu),
            float(chi),
            times,
            period=2,
            bias=bias,
            initial_phi=initial_phi,
            method=str(metadata["tdvp_method"]),
            rtol=float(metadata["tdvp_rtol"]),
            atol=float(metadata["tdvp_atol"]),
            max_step=float(metadata["tdvp_max_step"]),
            with_defect=False,
        )
        tdvp_profiles.append(tdvp_pair_profile(trajectory))
        tiled_reference = np.tile(tdvp_pair_profile(reference), (length // 2, 1))
        tdvp_reference_profiles.append(tiled_reference)
        gamma_defect.append(tdvpfun.get_qleak(trajectory))
        gamma_reference.append(tdvpfun.get_qleak(reference))
        solver_stats[label] = {
            "mu": float(mu),
            "chi": float(chi),
            "matched_period": length,
            "reference_period": 2,
        }

    tdvp_profiles_array = np.asarray(tdvp_profiles)
    tdvp_reference_array = np.asarray(tdvp_reference_profiles)
    gamma_defect_array = np.asarray(gamma_defect)
    gamma_reference_array = np.asarray(gamma_reference)
    ed_profiles = np.asarray(selected["ed_profiles"])
    ed_reference = np.asarray(selected["ed_reference_profiles"])

    defect_difference = tdvp_profiles_array - ed_profiles
    response_tdvp = tdvp_profiles_array - tdvp_reference_array
    response_ed = ed_profiles - ed_reference
    response_difference = response_tdvp - response_ed
    defect_rmse = np.sqrt(np.mean(defect_difference**2, axis=1))
    response_rmse = np.sqrt(np.mean(response_difference**2, axis=1))
    paired_gamma = gamma_defect_array + gamma_reference_array

    arrays = {
        "observable_times": times,
        "matched_tdvp_profiles": tdvp_profiles_array,
        "matched_tdvp_reference_profiles": tdvp_reference_array,
        "gamma_defect": gamma_defect_array,
        "gamma_reference": gamma_reference_array,
        "accumulated_gamma_defect": cumulative_trapezoid(
            gamma_defect_array, times
        ),
        "accumulated_gamma_pair": cumulative_trapezoid(
            paired_gamma, times
        ),
        "defect_profile_rmse": defect_rmse,
        "response_profile_rmse": response_rmse,
    }
    diagnostics = {
        "matched_length_and_period": length,
        "initial_defect_rmse_max": float(np.max(defect_rmse[:, 0])),
        "initial_response_rmse_max": float(np.max(response_rmse[:, 0])),
        "solver_protocol": solver_stats,
    }
    return arrays, diagnostics


def constrained_occupancies(
    basis: Any, length: int
) -> np.ndarray:
    """Return basis occupations with shape (Ns, L) in physical site order."""

    states = np.asarray(basis.states, dtype=np.uint64)
    powers = 2 ** np.arange(length - 1, -1, -1, dtype=np.uint64)
    return ((states[:, None] // powers[None, :]) % 2).astype(bool)


def vectorized_mps_state(
    theta: np.ndarray,
    phi: np.ndarray,
    occupations: np.ndarray,
) -> np.ndarray:
    """Construct the normalized finite-ring D=2 MPS without Python basis loops.

    For an allowed spin-1/2 configuration, every excited tensor contributes
    ``sin(theta_i/2) exp(-i phi_i)``.  A down-site cosine contributes unless
    that site immediately follows an excitation.  This scalar form is exactly
    equivalent to tracing the 2-by-2 matrices in ``EDfun.mpsmanifold``.
    """

    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    excited = np.asarray(occupations, dtype=bool)
    previous_excited = np.roll(excited, 1, axis=1)
    up_weight = np.sin(theta / 2.0) * np.exp(-1.0j * phi)
    down_weight = np.cos(theta / 2.0)
    amplitude = np.prod(
        np.where(excited, up_weight[None, :], 1.0), axis=1
    )
    cosine_sites = (~excited) & (~previous_excited)
    amplitude *= np.prod(
        np.where(cosine_sites, down_weight[None, :], 1.0), axis=1
    )
    norm = np.linalg.norm(amplitude)
    if norm == 0.0 or not np.isfinite(norm):
        raise FloatingPointError("finite-ring MPS has invalid norm")
    return amplitude / norm


def build_ed_hamiltonian(
    length: int,
    mu: float,
    chi: float,
    basis: Any,
) -> Any:
    """Build the same normalized spin-1/2 Hamiltonian as the figure reproducer."""

    from quspin.operators import hamiltonian

    spin = 0.5
    delta = np.asarray(
        [2.0 * mu - chi, 2.0 * mu + chi] * (length // 2),
        dtype=float,
    )
    static = [
        ["+", [[1.0 / (2.0 * spin), site] for site in range(length)]],
        ["-", [[1.0 / (2.0 * spin), site] for site in range(length)]],
        ["z", [[delta[site] / spin, site] for site in range(length)]],
    ]
    no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)
    return hamiltonian(static, [], basis=basis, **no_checks)


def finite_ring_residual_rate(
    time: float,
    state_parameters: np.ndarray,
    parameter_velocity: np.ndarray,
    hamiltonian_ed: Any,
    occupations: np.ndarray,
    direction_step: float,
) -> float:
    """Evaluate the projective finite-ring Schrödinger residual norm."""

    period = state_parameters.size // 2
    theta = state_parameters[:period]
    phi = state_parameters[period:]
    state = vectorized_mps_state(theta, phi, occupations)

    velocity_scale = max(float(np.max(np.abs(parameter_velocity))), 1.0)
    step = direction_step / velocity_scale
    plus_parameters = state_parameters + step * parameter_velocity
    minus_parameters = state_parameters - step * parameter_velocity
    plus = vectorized_mps_state(
        plus_parameters[:period], plus_parameters[period:], occupations
    )
    minus = vectorized_mps_state(
        minus_parameters[:period], minus_parameters[period:], occupations
    )
    state_derivative = (plus - minus) / (2.0 * step)
    state_derivative -= state * np.vdot(state, state_derivative)

    h_state = hamiltonian_ed.dot(state)
    energy = float(np.vdot(state, h_state).real)
    centered_h_state = h_state - energy * state
    residual = state_derivative + 1.0j * centered_h_state
    rate = float(np.linalg.norm(residual))
    if not math.isfinite(rate):
        raise FloatingPointError(f"nonfinite residual at t={time}")
    return rate


def direct_state_validation(
    points: np.ndarray,
    labels: np.ndarray,
    metadata: dict[str, Any],
    length: int,
    times: np.ndarray,
    direction_step: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Compute exact state distance, local error, and finite-ring residual."""

    basis = pxpbasisS.constrained_basis(2, length, None, None)
    occupations = constrained_occupancies(basis, length)
    theta0, phi0 = reproduction.initial_angles(
        length,
        float(metadata["pole_bias"]),
        initial_phi=float(metadata["initial_phi"]),
        with_defect=True,
    )
    initial_state = vectorized_mps_state(theta0, phi0, occupations)

    fs_distances = []
    local_rmse_values = []
    analytic_total_rates = []
    direct_residual_rates = []
    solver_diagnostics: dict[str, Any] = {}

    for (mu, chi), label_value in zip(points, labels, strict=True):
        label = str(label_value)
        trajectory = reproduction.integrate_tdvp(
            float(mu),
            float(chi),
            times,
            period=length,
            bias=float(metadata["pole_bias"]),
            initial_phi=float(metadata["initial_phi"]),
            method=str(metadata["tdvp_method"]),
            rtol=float(metadata["tdvp_rtol"]),
            atol=float(metadata["tdvp_atol"]),
            max_step=float(metadata["tdvp_max_step"]),
            with_defect=True,
        )
        hamiltonian_ed = build_ed_hamiltonian(
            length, float(mu), float(chi), basis
        )
        exact_states = hamiltonian_ed.evolve(
            initial_state,
            float(times[0]),
            times,
            solver_name=str(metadata["ed_solver_name"]),
            rtol=float(metadata["ed_rtol"]),
            atol=float(metadata["ed_atol"]),
            max_step=float(metadata["ed_max_step"]),
        )
        exact_states = np.asarray(exact_states)
        if exact_states.shape != (basis.Ns, times.size):
            raise RuntimeError(
                f"unexpected ED state array shape {exact_states.shape}"
            )

        gamma = tdvpfun.get_qleak(trajectory)
        analytic_total_rates.append(math.sqrt(length) * gamma)
        tdvp_occupation_values = tdvp_occupation(trajectory)
        exact_probabilities = np.abs(exact_states) ** 2
        exact_occupation_values = (
            occupations.astype(float).T @ exact_probabilities
        )
        local_rmse_values.append(
            np.sqrt(
                np.mean(
                    (tdvp_occupation_values - exact_occupation_values) ** 2,
                    axis=0,
                )
            )
        )

        point_fs = np.empty(times.size, dtype=float)
        point_residual = np.empty(times.size, dtype=float)
        for time_index, time_value in enumerate(times):
            parameters = trajectory[:, time_index]
            theta = parameters[:length]
            phi = parameters[length:]
            approximate_state = vectorized_mps_state(
                theta, phi, occupations
            )
            overlap = abs(
                np.vdot(exact_states[:, time_index], approximate_state)
            )
            point_fs[time_index] = math.acos(
                float(np.clip(overlap, 0.0, 1.0))
            )
            velocity = tdvpfun.eom(
                float(time_value),
                parameters,
                float(mu),
                float(chi),
            )
            point_residual[time_index] = finite_ring_residual_rate(
                float(time_value),
                parameters,
                velocity,
                hamiltonian_ed,
                occupations,
                direction_step,
            )
        fs_distances.append(point_fs)
        direct_residual_rates.append(point_residual)
        solver_diagnostics[label] = {
            "mu": float(mu),
            "chi": float(chi),
            "basis_dimension": int(basis.Ns),
        }

    analytic_rate_array = np.asarray(analytic_total_rates)
    direct_rate_array = np.asarray(direct_residual_rates)
    analytic_accumulated = cumulative_trapezoid(
        analytic_rate_array, times
    )
    direct_accumulated = cumulative_trapezoid(direct_rate_array, times)
    fs_array = np.asarray(fs_distances)
    local_rmse_array = np.asarray(local_rmse_values)
    rate_relative_error = np.abs(
        analytic_rate_array - direct_rate_array
    ) / np.maximum(direct_rate_array, 1.0e-12)

    arrays = {
        "state_times": times,
        "fs_distance": fs_array,
        "state_local_rmse": local_rmse_array,
        "analytic_total_residual_rate": analytic_rate_array,
        "direct_finite_ring_residual_rate": direct_rate_array,
        "analytic_accumulated_total_leakage": analytic_accumulated,
        "direct_accumulated_residual": direct_accumulated,
    }
    diagnostics = {
        "state_length_and_period": length,
        "constrained_basis_dimension": int(basis.Ns),
        "directional_derivative_step": direction_step,
        "initial_fs_distance_max": float(np.max(fs_array[:, 0])),
        "median_total_rate_relative_error": float(
            np.median(rate_relative_error)
        ),
        "maximum_total_rate_relative_error": float(
            np.max(rate_relative_error)
        ),
        "maximum_direct_bound_excess": float(
            np.max(fs_array - direct_accumulated)
        ),
        "maximum_analytic_bound_excess": float(
            np.max(fs_array - analytic_accumulated)
        ),
        "solver_protocol": solver_diagnostics,
    }
    return arrays, diagnostics


def correlation_diagnostics(
    arrays: dict[str, np.ndarray],
    labels: np.ndarray,
) -> dict[str, Any]:
    """Quantify ranking and linear association without claiming causation."""

    times = arrays["observable_times"]
    leakage = arrays["accumulated_gamma_pair"]
    error = arrays["response_profile_rmse"]
    diagnostics: dict[str, Any] = {}

    for target in (2.0, 5.0, 10.0):
        if target > times[-1] + 1.0e-12:
            continue
        index = int(np.argmin(np.abs(times - target)))
        spearman = spearmanr(leakage[:, index], error[:, index])
        pearson = pearsonr(leakage[:, index], error[:, index])
        diagnostics[f"t={times[index]:g}"] = {
            "spearman_r": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
        }

    final_spearman = spearmanr(leakage[:, -1], error[:, -1])
    diagnostics["final_values"] = {
        "labels": [str(label) for label in labels],
        "paired_accumulated_leakage": leakage[:, -1].tolist(),
        "response_rmse": error[:, -1].tolist(),
        "spearman_r": float(final_spearman.statistic),
        "spearman_p": float(final_spearman.pvalue),
    }
    return diagnostics


def configuration(args: argparse.Namespace, selected: dict[str, Any]) -> dict[str, Any]:
    """Parameters that define numerical cache compatibility."""

    metadata = dict(selected["metadata"])
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "core_cache": str(args.core_cache.resolve()),
        "labels": args.point_labels,
        "tmax": args.tmax,
        "observable_stride": args.observable_stride,
        "matched_length": int(metadata["N_ed"]),
        "state_length": args.state_length,
        "state_samples": args.state_samples,
        "direction_step": args.direction_step,
        "pole_bias": metadata["pole_bias"],
        "initial_phi": metadata["initial_phi"],
        "tdvp_method": metadata["tdvp_method"],
        "tdvp_rtol": metadata["tdvp_rtol"],
        "tdvp_atol": metadata["tdvp_atol"],
        "tdvp_max_step": metadata["tdvp_max_step"],
        "ed_solver_name": metadata["ed_solver_name"],
        "ed_rtol": metadata["ed_rtol"],
        "ed_atol": metadata["ed_atol"],
        "ed_max_step": metadata["ed_max_step"],
    }


def configuration_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.025,
        0.965,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        fontweight="semibold",
    )


def plot_validation(
    arrays: dict[str, np.ndarray],
    labels: np.ndarray,
    metadata: dict[str, Any],
    output_dir: Path,
    cache_key: str,
) -> tuple[Path, Path]:
    """Create the four-panel appendix validation figure."""

    observable_times = arrays["observable_times"]
    state_times = arrays["state_times"]
    accumulated_pair = arrays["accumulated_gamma_pair"]
    response_rmse = arrays["response_profile_rmse"]
    fs_distance = arrays["fs_distance"]
    analytic_accumulated = arrays[
        "analytic_accumulated_total_leakage"
    ]

    # HZG geometry controls: margins move the complete block, while wspace and
    # hspace change only inter-panel gaps.  The canvas is standard two-column
    # APS width.
    geometry = {
        "figsize": (7.10, 4.75),
        "left": 0.092,
        "right": 0.985,
        "bottom": 0.105,
        "top": 0.975,
        "wspace": 0.31,
        "hspace": 0.32,
    }

    with plt.rc_context(HZG_STYLE):
        figure = plt.figure(figsize=geometry["figsize"])
        grid = figure.add_gridspec(
            2,
            2,
            left=geometry["left"],
            right=geometry["right"],
            bottom=geometry["bottom"],
            top=geometry["top"],
            wspace=geometry["wspace"],
            hspace=geometry["hspace"],
        )
        axes = [
            figure.add_subplot(grid[0, 0]),
            figure.add_subplot(grid[0, 1]),
            figure.add_subplot(grid[1, 0]),
            figure.add_subplot(grid[1, 1]),
        ]
        leakage_axis, error_axis, calibration_axis, state_axis = axes

        for point_index, label_value in enumerate(labels):
            label = str(label_value)
            color = POINT_COLORS[label]
            linestyle = POINT_STYLES[label]
            leakage_axis.plot(
                observable_times,
                accumulated_pair[point_index],
                color=color,
                linestyle=linestyle,
                label=label,
            )
            error_axis.plot(
                observable_times,
                response_rmse[point_index],
                color=color,
                linestyle=linestyle,
                label=label,
            )
            calibration_axis.plot(
                accumulated_pair[point_index],
                response_rmse[point_index],
                color=color,
                linestyle=linestyle,
                linewidth=1.0,
            )
            sample_indices = np.linspace(
                0, observable_times.size - 1, 12, dtype=int
            )
            calibration_axis.scatter(
                accumulated_pair[point_index, sample_indices],
                response_rmse[point_index, sample_indices],
                s=8,
                color=color,
                edgecolors="none",
                zorder=3,
            )

            positive = (
                (analytic_accumulated[point_index] > 0.0)
                & (fs_distance[point_index] > 0.0)
            )
            state_axis.plot(
                analytic_accumulated[point_index, positive],
                fs_distance[point_index, positive],
                color=color,
                linestyle=linestyle,
                linewidth=1.0,
            )
            marker_indices = np.flatnonzero(positive)[:: max(1, np.count_nonzero(positive) // 10)]
            state_axis.scatter(
                analytic_accumulated[point_index, marker_indices],
                fs_distance[point_index, marker_indices],
                s=9,
                color=color,
                edgecolors="none",
                zorder=3,
            )

        leakage_axis.set_xlabel(r"$t$")
        leakage_axis.set_ylabel(
            r"$\Lambda_{\rm pair}(t)$"
        )
        error_axis.set_xlabel(r"$t$")
        error_axis.set_ylabel(r"$\epsilon_{\rm resp}(t)$")
        error_axis.legend(
            frameon=False,
            ncol=3,
            loc="upper left",
            bbox_to_anchor=(0.10, 1.00),
            handlelength=1.9,
            handletextpad=0.35,
            columnspacing=0.8,
            borderaxespad=0.15,
        )
        calibration_axis.set_xlabel(r"$\Lambda_{\rm pair}(t)$")
        calibration_axis.set_ylabel(r"$\epsilon_{\rm resp}(t)$")

        positive_x = analytic_accumulated[analytic_accumulated > 0.0]
        positive_y = fs_distance[fs_distance > 0.0]
        lower = max(
            min(float(np.min(positive_x)), float(np.min(positive_y))) * 0.7,
            1.0e-6,
        )
        upper = max(
            float(np.max(analytic_accumulated)),
            float(np.max(fs_distance)),
        ) * 1.15
        diagonal = np.geomspace(lower, upper, 200)
        state_axis.plot(
            diagonal,
            diagonal,
            color="0.45",
            linewidth=0.7,
            linestyle="--",
            zorder=0,
        )
        state_axis.set_xscale("log")
        state_axis.set_yscale("log")
        state_axis.set_xlim(lower, upper)
        state_axis.set_ylim(lower, upper)
        state_axis.set_xlabel(r"$\sqrt{L}\int_0^t\Gamma(t')\,dt'$")
        state_axis.set_ylabel(r"$d_{\rm FS}(t)$")
        state_axis.text(
            0.975,
            0.055,
            rf"$L=K={metadata['direct_state']['state_length_and_period']}$",
            transform=state_axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
        )

        for axis, panel in zip(
            axes, ("(a)", "(b)", "(c)", "(d)"), strict=True
        ):
            add_panel_label(axis, panel)
            axis.tick_params(width=0.65, length=2.7, pad=1.7)
            for spine in axis.spines.values():
                spine.set_linewidth(0.7)

        figure.canvas.draw()
        pdf_path = output_dir / (
            f"leakage_error_validation_{cache_key}.pdf"
        )
        png_path = output_dir / (
            f"leakage_error_validation_{cache_key}.png"
        )
        # Preserve the declared 7.10-inch APS two-column canvas.  A tight
        # bounding box would silently shrink the exported width and make the
        # final typography inconsistent with figures sized at inclusion.
        figure.savefig(pdf_path)
        figure.savefig(png_path, dpi=300)
        plt.close(figure)
    return pdf_path, png_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--core-cache",
        type=Path,
        default=DEFAULT_CORE_CACHE,
        help="existing matched-parameter Fig. 1 ED/TDVP profile cache",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="new validation cache/data/figure directory",
    )
    parser.add_argument(
        "--point-labels",
        type=parse_labels,
        default=parse_labels("P1,P2,P3,P4,P5,P6"),
        help="subset of marked confinement points",
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=10.0,
        help="validation end time, not exceeding the core cache",
    )
    parser.add_argument(
        "--observable-stride",
        type=int,
        default=1,
        help="stride applied to the cached L=24 observable time grid",
    )
    parser.add_argument(
        "--state-length",
        type=int,
        default=16,
        help="even periodic length K=L for direct state-space validation",
    )
    parser.add_argument(
        "--state-samples",
        type=int,
        default=101,
        help="uniform time samples for direct ED state validation",
    )
    parser.add_argument(
        "--direction-step",
        type=float,
        default=1.0e-6,
        help="maximum coordinate displacement in residual directional derivative",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute an existing compatible validation cache",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="compute/cache data without exporting PDF/PNG",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="use P1/P6, t<=1, L=10, and sparse grids",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.smoke_test:
        args.point_labels = ["P1", "P6"]
        args.tmax = 1.0
        args.observable_stride = 10
        args.state_length = 10
        args.state_samples = 21
    if not args.core_cache.exists():
        raise FileNotFoundError(args.core_cache)
    if args.tmax <= 0.0 or args.tmax > reproduction.CORE_TMAX:
        raise ValueError(
            f"--tmax must lie in (0, {reproduction.CORE_TMAX}]"
        )
    if args.observable_stride < 1:
        raise ValueError("--observable-stride must be positive")
    if args.state_length < 4 or args.state_length % 2:
        raise ValueError("--state-length must be even and at least 4")
    if args.state_length > 24:
        raise ValueError(
            "--state-length>24 is intentionally disabled for this ED validation"
        )
    if args.state_samples < 5:
        raise ValueError("--state-samples must be at least 5")
    if args.direction_step <= 0.0:
        raise ValueError("--direction-step must be positive")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)

    selected = selected_core_data(
        args.core_cache,
        args.point_labels,
        args.tmax,
        args.observable_stride,
    )
    config = configuration(args, selected)
    cache_key = configuration_hash(config)
    data_path = output_dir / f"leakage_error_validation_{cache_key}.npz"
    metadata_path = output_dir / (
        f"leakage_error_validation_{cache_key}.json"
    )

    if data_path.exists() and metadata_path.exists() and not args.force:
        with np.load(data_path, allow_pickle=False) as archive:
            arrays = {
                name: np.asarray(archive[name]) for name in archive.files
            }
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        status = "loaded"
    else:
        observable_arrays, observable_diagnostics = (
            matched_observable_validation(selected)
        )
        state_times = np.linspace(0.0, args.tmax, args.state_samples)
        state_arrays, state_diagnostics = direct_state_validation(
            np.asarray(selected["points"]),
            np.asarray(selected["labels"]),
            dict(selected["metadata"]),
            args.state_length,
            state_times,
            args.direction_step,
        )
        arrays = {
            "points": np.asarray(selected["points"]),
            "labels": np.asarray(selected["labels"]),
            **observable_arrays,
            **state_arrays,
        }
        metadata = {
            "cache_key": cache_key,
            "configuration": config,
            "matched_observable": observable_diagnostics,
            "direct_state": state_diagnostics,
            "correlations": correlation_diagnostics(
                arrays, np.asarray(selected["labels"])
            ),
            "scientific_scope": [
                (
                    "The observable calibration is empirical: an intensive "
                    "leakage is not a pointwise upper bound on one local RMSE."
                ),
                (
                    "The direct finite-ring residual supplies the relevant "
                    "state-space bound and independently checks sqrt(L) Gamma."
                ),
                (
                    "A positive correlation across six selected points is "
                    "supportive but is not a universal monotonic calibration."
                ),
            ],
        }
        np.savez_compressed(data_path, **arrays)
        write_json(metadata_path, metadata)
        status = "generated"

    figures: list[str] = []
    if not args.no_plot:
        pdf_path, png_path = plot_validation(
            arrays,
            np.asarray(arrays["labels"]),
            metadata,
            output_dir,
            cache_key,
        )
        figures = [str(pdf_path), str(png_path)]
        metadata["figure"] = {
            "pdf": str(pdf_path),
            "png": str(png_path),
        }
        write_json(metadata_path, metadata)

    print(
        json.dumps(
            {
                "status": "ok",
                "cache_status": status,
                "data": str(data_path),
                "metadata": str(metadata_path),
                "figures": figures,
                "matched_observable": metadata["matched_observable"],
                "direct_state": metadata["direct_state"],
                "correlations": metadata["correlations"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
