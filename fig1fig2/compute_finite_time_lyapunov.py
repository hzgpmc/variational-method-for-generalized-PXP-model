#!/usr/bin/env python3
"""Finite-time tangent amplification for the Region-I/III TDVP orbits.

This script computes the largest singular-value amplification of the complete
``2K``-dimensional spin-1/2 TDVP flow, rather than evolving one arbitrarily
chosen perturbation.  Coordinate tangent vectors ``(dtheta,dphi)`` are
measured with the all-site Bloch-sphere norm

    ||delta z||_B^2 = sum_i [dtheta_i^2 + sin(theta_i)^2 dphi_i^2] / K.

The common factor ``1/K`` cancels from every amplification ratio.  The metric
is regular at the Bloch poles even though the angular TDVP equations are not.
The tangent equation is evaluated by a batched complex-step Jacobian-vector
product and integrated directly in the orthonormal Bloch frame, which avoids
putting the large coordinate components ``dphi ~ 1/sin(theta)`` into the ODE
error norm.  Periodic QR factorizations control the tangent-map dynamic range;
the product of the resulting triangular factors retains the exact finite-time
tangent map.  With the default ``--subspace-size 200`` this is the
full tangent space for ``K=100``, so the plotted value is

    lambda_max^FT(t) = log(s_max[M_B(t)]) / t,

where ``M_B`` maps initial to final Bloch-orthonormal tangent components.

The same full map is then projected onto
``W_r={i0-r,...,i0+r}``, for ``r=0,1,2,3``.  Two distinct diagnostics are
saved: ``P_Wr M_B P_Wr^T`` (local-to-local) and ``P_D M_B P_Wr^T``
(perturbations in the window measured at the defect ``D={i0}``).  Projection
is applied only after full propagation, so sites outside the window continue
to evolve and mediate feedback normally.

The manuscript stability figure uses only the two-component defect-site
response to a perturbation initially applied at that same site (the r=0
subblock).  Its matrix elements are derivatives of the final defect
Bloch-frame perturbation with respect to its initial value, and the plotted
rate is lambda_max(t) = log(s_max[M_defect(t)]) / t.  The complete tangent
map and the radius-resolved subblocks are retained as numerical cross-checks.

The numerical protocol matches the active Paper-A trajectory figures:
``K=100``, pole bias ``epsilon=1e-3``, ``phi_i=0``, DOP853, ``rtol=1e-9``,
``atol=1e-11``, and ``max_step=0.02``.  The five Region-I and five Region-III
coordinates are loaded from the auditable selection manifest and the shared
figure-selection module.

Run from the repository root in the QuSpin environment:

    conda run -n quspin python fig1fig2/compute_finite_time_lyapunov.py

For a short implementation benchmark without overwriting the default cache:

    conda run -n quspin python fig1fig2/compute_finite_time_lyapunov.py \
        --points I2 --t-max 0.2 --output-dt 0.1 \
        --cache fig1fig2/data/ftle_benchmark.npz \
        --output-prefix fig1fig2/output/ftle_benchmark --force
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUTPUT_DIR = HERE / "output"
DEFAULT_MANIFEST = DATA_DIR / "selected_points_phi0.json"
DEFAULT_CACHE = DATA_DIR / "region_i_iii_ftle.npz"
DEFAULT_OUTPUT_PREFIX = OUTPUT_DIR / "region_i_iii_ftle"
DEFAULT_PROJECTED_PREFIX = OUTPUT_DIR / "region_i_iii_projected_ftle"
DEFAULT_STABILITY_PREFIX = OUTPUT_DIR / "region_i_iii_local_stability"
STYLE_CANDIDATES = (
    HERE / "hzg-paper.mplstyle",
    HERE.parent / "scripts" / "hzg-paper.mplstyle",
)

os.environ.setdefault("MPLCONFIGDIR", str(DATA_DIR / "mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import svd, svdvals

import tdvpfun
from figure2_selection import FIG2_ROWS, MODE_POINT_COORDINATES


SCHEMA_VERSION = 2
K = 100
POLE_BIAS = 1.0e-3
INITIAL_PHI = 0.0
METHOD = "DOP853"
RTOL = 1.0e-9
ATOL = 1.0e-11
MAX_STEP = 0.02
T_MAX = 10.0
OUTPUT_DT = 0.1
COMPLEX_STEP = 1.0e-20
POLE_ABORT = 1.0e-10
PROJECTED_RADII = (0, 1, 2, 3)


@dataclass(frozen=True)
class Point:
    source_id: str
    display_id: str
    region: str
    mu: float
    chi: float


def resolve_style_file() -> Path:
    for candidate in STYLE_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "No HZG Matplotlib style found at: "
        + ", ".join(str(path) for path in STYLE_CANDIDATES)
    )


def defect_site(length: int) -> int:
    return 2 * ((length + 2) // 4) - 1


def initial_angles(length: int = K) -> tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(
        [POLE_BIAS, np.pi - POLE_BIAS] * (length // 2)
        + [POLE_BIAS] * (length % 2),
        dtype=float,
    )
    theta[defect_site(length)] = POLE_BIAS
    phi = np.full(length, INITIAL_PHI, dtype=float)
    return theta, phi


def load_points(manifest_path: Path) -> tuple[Point, ...]:
    """Load the ten shared Region-I/III coordinates without trajectory data."""

    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if int(manifest.get("schema_version", -1)) != 1:
        raise ValueError("selected-point manifest schema differs")
    protocol = manifest.get("map_protocol", {})
    expected_protocol = {
        "pole_bias": POLE_BIAS,
        "initial_phi": INITIAL_PHI,
        "K": K,
    }
    for key, expected in expected_protocol.items():
        actual = protocol.get(key)
        if isinstance(expected, float):
            valid = np.isclose(
                float(actual), expected, rtol=0.0, atol=1.0e-15
            )
        else:
            valid = int(actual) == expected
        if not valid:
            raise ValueError(
                f"selection protocol {key}={actual!r}, expected {expected!r}"
            )

    coordinate_lookup: dict[str, tuple[float, float]] = {}
    for rows in manifest.get("families", {}).values():
        for row in rows:
            coordinate_lookup[str(row["point_id"])] = (
                float(row["mu"]),
                float(row["chi"]),
            )
    for point_id, (mu, chi, _note) in MODE_POINT_COORDINATES.items():
        coordinate_lookup[point_id] = (float(mu), float(chi))

    points: list[Point] = []
    for region_key, point_specs in FIG2_ROWS:
        if region_key not in {"region_i", "region_iii"}:
            continue
        region = "I" if region_key == "region_i" else "III"
        for source_id, display_id in point_specs:
            if source_id not in coordinate_lookup:
                raise KeyError(f"coordinate for {source_id} is unavailable")
            mu, chi = coordinate_lookup[source_id]
            points.append(Point(source_id, display_id, region, mu, chi))
    if [point.display_id for point in points] != [
        "I1", "I2", "I3", "I4", "I5",
        "III1", "III2", "III3", "III4", "III5",
    ]:
        raise AssertionError("Region-I/III point ordering unexpectedly changed")
    return tuple(points)


def eom_columns(
    y: np.ndarray, mu: float, chi: float
) -> np.ndarray:
    """Vectorized copy of ``tdvpfun.eom`` for state columns.

    ``y`` has shape ``(2K, n_columns)``.  Keeping the algebra term-for-term
    identical to the active scalar implementation permits one complex-step
    evaluation to return many Jacobian-vector products.
    """

    values = np.asarray(y)
    squeeze = values.ndim == 1
    if squeeze:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] % 2:
        raise ValueError(f"invalid TDVP column array shape {values.shape}")
    length = values.shape[0] // 2
    theta = values[:length]
    phi = values[length:]
    eta = tdvpfun.get_eta(theta)

    plus_one = np.arange(1, length + 1) % length
    plus_two = np.arange(2, length + 2) % length
    minus_one = np.arange(-1, length - 1) % length
    eta_minus_one = eta[minus_one]
    eta_plus_one = eta[plus_one]
    theta_plus_one = theta[plus_one]
    theta_plus_two = theta[plus_two]
    theta_minus_one = theta[minus_one]
    phi_plus_one = phi[plus_one]
    phi_minus_one = phi[minus_one]

    delta = np.asarray(
        [2.0 * mu - chi, 2.0 * mu + chi] * (length // 2)
        + [2.0 * mu - chi] * (length % 2),
        dtype=float,
    )[:, None]

    dtheta = (
        2.0 * np.cos(theta_plus_one / 2.0) * np.sin(phi)
        + eta_minus_one
        * np.sin(theta_minus_one)
        * np.sin(theta / 2.0)
        * np.sin(phi_minus_one)
        / eta
    )
    dphi = (
        2.0
        * np.cos(theta_plus_one / 2.0)
        * np.cos(phi)
        / np.tan(theta)
        + 2.0 * delta
        - np.cos(theta_plus_two / 2.0)
        * np.cos(phi_plus_one)
        * np.tan(theta_plus_one / 2.0)
        - eta_minus_one
        * np.sin(theta_minus_one)
        * np.cos(phi_minus_one)
        / (2.0 * eta * np.cos(theta / 2.0))
        - eta
        * np.sin(theta)
        * np.cos(phi)
        * np.sin(theta_plus_one / 2.0)
        * np.tan(theta_plus_one / 2.0)
        / (2.0 * eta_plus_one)
    )
    result = np.concatenate((dtheta, dphi), axis=0)
    return result[:, 0] if squeeze else result


def physical_from_coordinate_tangents(
    state: np.ndarray, tangents: np.ndarray
) -> np.ndarray:
    """Map angular tangent components to an orthonormal Bloch-frame basis."""

    length = state.size // 2
    physical = np.array(tangents, copy=True)
    physical[length:] *= np.sin(state[:length])[:, None]
    return physical


def coordinate_from_physical_tangents(
    state: np.ndarray, physical: np.ndarray
) -> tuple[np.ndarray, float]:
    """Map Bloch-frame components to angular coordinates and report pole margin."""

    length = state.size // 2
    sin_theta = np.sin(state[:length])
    margin = float(np.min(np.abs(sin_theta)))
    if margin <= POLE_ABORT:
        raise FloatingPointError(
            f"angular trajectory approached a pole too closely: {margin:.3e}"
        )
    tangents = np.array(physical, copy=True)
    tangents[length:] /= sin_theta[:, None]
    return tangents, margin


def tangent_rhs(
    _time: float,
    augmented: np.ndarray,
    *,
    mu: float,
    chi: float,
    dimension: int,
    subspace_size: int,
    complex_step: float,
) -> np.ndarray:
    state = augmented[:dimension]
    physical_tangents = augmented[dimension:].reshape(
        dimension, subspace_size
    )
    length = dimension // 2
    sin_theta = np.sin(state[:length])
    margin = float(np.min(np.abs(sin_theta)))
    if margin <= POLE_ABORT:
        raise FloatingPointError(
            f"angular trajectory approached a pole too closely: {margin:.3e}"
        )
    coordinate_tangents = np.array(physical_tangents, copy=True)
    coordinate_tangents[length:] /= sin_theta[:, None]
    state_rhs = np.asarray(eom_columns(state, mu, chi), dtype=float)
    shifted = state[:, None].astype(complex) + (
        1j * complex_step * coordinate_tangents
    )
    coordinate_derivative = (
        np.imag(eom_columns(shifted, mu, chi)) / complex_step
    )

    # If q=(dtheta,sin(theta)dphi)=D(y) delta y, then
    # qdot = D J delta y + Ddot delta y.  This form is algebraically
    # equivalent to the angular tangent equation but remains naturally scaled
    # in the physical Bloch norm near a regularized pole.
    physical_derivative = np.empty_like(coordinate_derivative)
    physical_derivative[:length] = coordinate_derivative[:length]
    physical_derivative[length:] = (
        sin_theta[:, None] * coordinate_derivative[length:]
        + np.cos(state[:length])[:, None]
        * state_rhs[:length, None]
        * coordinate_tangents[length:]
    )
    return np.concatenate((state_rhs, physical_derivative.ravel()))


def _positive_diagonal_qr(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q_basis, r_factor = np.linalg.qr(matrix, mode="reduced")
    signs = np.sign(np.diag(r_factor))
    signs[signs == 0.0] = 1.0
    q_basis *= signs[None, :]
    r_factor *= signs[:, None]
    return q_basis, r_factor


def output_times(t_max: float, output_dt: float) -> np.ndarray:
    count = int(round(t_max / output_dt))
    if count < 1 or not np.isclose(
        count * output_dt, t_max, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("t_max must be a positive integer multiple of output_dt")
    return np.linspace(0.0, t_max, count + 1)


def physical_indices_for_sites(
    sites: np.ndarray, length: int
) -> np.ndarray:
    """Return ``(dtheta,sin(theta)dphi)`` indices for selected sites."""

    site_indices = np.asarray(sites, dtype=int)
    if np.any(site_indices < 0) or np.any(site_indices >= length):
        raise IndexError(f"site window leaves the K={length} unit cell")
    return np.concatenate((site_indices, length + site_indices))


def compute_point_ftle(
    point: Point,
    times: np.ndarray,
    *,
    subspace_size: int,
    method: str,
    rtol: float,
    atol: float,
    max_step: float,
    complex_step: float,
    random_seed: int,
    progress: bool = True,
) -> dict[str, np.ndarray | float | int | str]:
    """Propagate and factorize the tangent map for one parameter point."""

    theta0, phi0 = initial_angles(K)
    state = np.concatenate((theta0, phi0))
    dimension = state.size
    if not 1 <= subspace_size <= dimension:
        raise ValueError(
            f"subspace_size={subspace_size} outside [1,{dimension}]"
        )

    if subspace_size == dimension:
        physical_basis = np.eye(dimension)
    else:
        rng = np.random.default_rng(random_seed)
        physical_basis, _ = np.linalg.qr(
            rng.standard_normal((dimension, subspace_size)), mode="reduced"
        )
    initial_physical_basis = np.array(physical_basis, copy=True)
    _unused, minimum_pole_margin = coordinate_from_physical_tangents(
        state, physical_basis
    )
    physical_tangents = physical_basis

    reduced_map = np.eye(subspace_size)
    logarithmic_scale = 0.0
    log_gain = np.zeros(times.size, dtype=float)
    ftle = np.full(times.size, np.nan, dtype=float)
    projected_local_log_gain = np.zeros(
        (len(PROJECTED_RADII), times.size), dtype=float
    )
    projected_local_ftle = np.full_like(
        projected_local_log_gain, np.nan
    )
    projected_defect_log_gain = np.zeros_like(projected_local_log_gain)
    projected_defect_ftle = np.full_like(
        projected_local_log_gain, np.nan
    )
    pole_margin = np.empty(times.size, dtype=float)
    pole_margin[0] = minimum_pole_margin
    nfev = np.zeros(times.size - 1, dtype=int)
    wall_seconds = np.zeros(times.size - 1, dtype=float)

    for interval, (left, right) in enumerate(zip(times[:-1], times[1:])):
        augmented0 = np.concatenate((state, physical_tangents.ravel()))
        started = time.perf_counter()
        solution = solve_ivp(
            lambda t, y: tangent_rhs(
                t,
                y,
                mu=point.mu,
                chi=point.chi,
                dimension=dimension,
                subspace_size=subspace_size,
                complex_step=complex_step,
            ),
            (float(left), float(right)),
            augmented0,
            t_eval=[float(right)],
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        wall_seconds[interval] = time.perf_counter() - started
        nfev[interval] = int(solution.nfev)
        if not solution.success or solution.y.shape != (augmented0.size, 1):
            raise RuntimeError(
                f"{point.display_id} tangent integration failed on "
                f"[{left},{right}]: {solution.message}"
            )
        endpoint = np.asarray(solution.y[:, -1], dtype=float)
        if not np.all(np.isfinite(endpoint)):
            raise FloatingPointError(
                f"{point.display_id} produced nonfinite tangent data"
            )
        state = endpoint[:dimension]
        physical_tangents = endpoint[dimension:].reshape(
            dimension, subspace_size
        )
        physical_basis, r_factor = _positive_diagonal_qr(physical_tangents)
        _unused, current_margin = coordinate_from_physical_tangents(
            state, physical_basis
        )
        physical_tangents = physical_basis
        minimum_pole_margin = min(minimum_pole_margin, current_margin)
        pole_margin[interval + 1] = current_margin

        trial_map = r_factor @ reduced_map
        segment_scale = float(svdvals(trial_map, check_finite=False)[0])
        if not np.isfinite(segment_scale) or segment_scale <= 0.0:
            raise FloatingPointError(
                f"invalid singular-value scale {segment_scale} at t={right}"
            )
        reduced_map = trial_map / segment_scale
        logarithmic_scale += np.log(segment_scale)
        log_gain[interval + 1] = logarithmic_scale
        ftle[interval + 1] = logarithmic_scale / float(right)

        # The columns of ``physical_basis @ reduced_map`` are the complete
        # tangent map from the initial to current Bloch-orthonormal frames,
        # apart from the common exp(logarithmic_scale) factor.  Projection is
        # therefore performed only after the full dynamics has propagated;
        # degrees of freedom outside W_r are never frozen.
        if subspace_size == dimension:
            scaled_full_map = physical_basis @ reduced_map
            center = defect_site(K)
            defect_indices = physical_indices_for_sites(
                np.asarray([center]), K
            )
            for radius_index, radius in enumerate(PROJECTED_RADII):
                sites = np.arange(center - radius, center + radius + 1)
                window_indices = physical_indices_for_sites(sites, K)
                local_block = scaled_full_map[
                    np.ix_(window_indices, window_indices)
                ]
                defect_block = scaled_full_map[
                    np.ix_(defect_indices, window_indices)
                ]
                local_scale = float(
                    svdvals(local_block, check_finite=False)[0]
                )
                defect_scale = float(
                    svdvals(defect_block, check_finite=False)[0]
                )
                if local_scale <= 0.0 or defect_scale <= 0.0:
                    raise FloatingPointError(
                        "projected tangent map became exactly rank zero"
                    )
                local_log = logarithmic_scale + np.log(local_scale)
                defect_log = logarithmic_scale + np.log(defect_scale)
                projected_local_log_gain[
                    radius_index, interval + 1
                ] = local_log
                projected_defect_log_gain[
                    radius_index, interval + 1
                ] = defect_log
                projected_local_ftle[
                    radius_index, interval + 1
                ] = local_log / float(right)
                projected_defect_ftle[
                    radius_index, interval + 1
                ] = defect_log / float(right)

        if progress:
            print(
                f"{point.display_id:>4s} t={right:5.2f}/{times[-1]:.2f} "
                f"lambda_FT={ftle[interval + 1]: .6f} "
                f"logG={logarithmic_scale: .5f} "
                f"min|sin(theta)|={minimum_pole_margin:.2e}",
                flush=True,
            )

    left_vectors, reduced_singular_values, right_vectors_h = svd(
        reduced_map, full_matrices=False, check_finite=False
    )
    leading_initial_direction = (
        initial_physical_basis @ right_vectors_h[0]
    )
    leading_final_direction = physical_basis @ left_vectors[:, 0]
    if not np.isclose(
        reduced_singular_values[0], 1.0, rtol=2.0e-11, atol=2.0e-11
    ):
        raise AssertionError(
            "final scaled tangent map lost its unit leading singular value"
        )

    projected_local_leading_initial_direction = np.zeros(
        (len(PROJECTED_RADII), dimension), dtype=float
    )
    projected_defect_leading_initial_direction = np.zeros_like(
        projected_local_leading_initial_direction
    )
    if subspace_size == dimension:
        final_scaled_map = physical_basis @ reduced_map
        center = defect_site(K)
        defect_indices = physical_indices_for_sites(
            np.asarray([center]), K
        )
        for radius_index, radius in enumerate(PROJECTED_RADII):
            sites = np.arange(center - radius, center + radius + 1)
            window_indices = physical_indices_for_sites(sites, K)
            local_block = final_scaled_map[
                np.ix_(window_indices, window_indices)
            ]
            defect_block = final_scaled_map[
                np.ix_(defect_indices, window_indices)
            ]
            _u_local, _s_local, vh_local = svd(
                local_block, full_matrices=False, check_finite=False
            )
            _u_defect, _s_defect, vh_defect = svd(
                defect_block, full_matrices=False, check_finite=False
            )
            projected_local_leading_initial_direction[
                radius_index, window_indices
            ] = vh_local[0]
            projected_defect_leading_initial_direction[
                radius_index, window_indices
            ] = vh_defect[0]

    return {
        "source_id": point.source_id,
        "display_id": point.display_id,
        "region": point.region,
        "mu": point.mu,
        "chi": point.chi,
        "times": times,
        "ftle": ftle,
        "log_gain": log_gain,
        "projected_local_ftle": projected_local_ftle,
        "projected_local_log_gain": projected_local_log_gain,
        "projected_defect_ftle": projected_defect_ftle,
        "projected_defect_log_gain": projected_defect_log_gain,
        "pole_margin": pole_margin,
        "minimum_pole_margin": minimum_pole_margin,
        "nfev": nfev,
        "wall_seconds": wall_seconds,
        "final_state": state,
        "leading_initial_direction": leading_initial_direction,
        "leading_final_direction": leading_final_direction,
        "projected_local_leading_initial_direction": (
            projected_local_leading_initial_direction
        ),
        "projected_defect_leading_initial_direction": (
            projected_defect_leading_initial_direction
        ),
    }


def validate_vectorized_eom(points: Iterable[Point]) -> None:
    """Check the batched algebra against the active scalar TDVP function."""

    theta, phi = initial_angles(K)
    state = np.concatenate((theta, phi))
    for point in points:
        scalar = tdvpfun.eom(0.0, state, point.mu, point.chi)
        batched = eom_columns(state[:, None], point.mu, point.chi)[:, 0]
        if not np.allclose(scalar, batched, rtol=2.0e-14, atol=2.0e-14):
            maximum_error = float(np.max(np.abs(scalar - batched)))
            raise AssertionError(
                f"vectorized EOM mismatch at {point.display_id}: "
                f"max abs error={maximum_error:.3e}"
            )


def write_cache(
    cache_path: Path,
    results: list[dict[str, object]],
    *,
    times: np.ndarray,
    subspace_size: int,
    rtol: float,
    atol: float,
    max_step: float,
    complex_step: float,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        schema_version=np.asarray(SCHEMA_VERSION),
        times=times,
        source_id=np.asarray([row["source_id"] for row in results]),
        display_id=np.asarray([row["display_id"] for row in results]),
        region=np.asarray([row["region"] for row in results]),
        mu=np.asarray([row["mu"] for row in results], dtype=float),
        chi=np.asarray([row["chi"] for row in results], dtype=float),
        ftle=np.asarray([row["ftle"] for row in results], dtype=float),
        log_gain=np.asarray(
            [row["log_gain"] for row in results], dtype=float
        ),
        projected_radii=np.asarray(PROJECTED_RADII, dtype=int),
        projected_window_site_count=np.asarray(
            [2 * radius + 1 for radius in PROJECTED_RADII], dtype=int
        ),
        projected_local_ftle=np.asarray(
            [row["projected_local_ftle"] for row in results],
            dtype=float,
        ),
        projected_local_log_gain=np.asarray(
            [row["projected_local_log_gain"] for row in results],
            dtype=float,
        ),
        projected_defect_ftle=np.asarray(
            [row["projected_defect_ftle"] for row in results],
            dtype=float,
        ),
        projected_defect_log_gain=np.asarray(
            [row["projected_defect_log_gain"] for row in results],
            dtype=float,
        ),
        pole_margin=np.asarray(
            [row["pole_margin"] for row in results], dtype=float
        ),
        minimum_pole_margin=np.asarray(
            [row["minimum_pole_margin"] for row in results], dtype=float
        ),
        pole_margin_sampling=np.asarray("QR/output endpoints only"),
        nfev=np.asarray([row["nfev"] for row in results], dtype=int),
        wall_seconds=np.asarray(
            [row["wall_seconds"] for row in results], dtype=float
        ),
        final_state=np.asarray(
            [row["final_state"] for row in results], dtype=float
        ),
        leading_initial_direction=np.asarray(
            [row["leading_initial_direction"] for row in results],
            dtype=float,
        ),
        leading_final_direction=np.asarray(
            [row["leading_final_direction"] for row in results],
            dtype=float,
        ),
        projected_local_leading_initial_direction=np.asarray(
            [
                row["projected_local_leading_initial_direction"]
                for row in results
            ],
            dtype=float,
        ),
        projected_defect_leading_initial_direction=np.asarray(
            [
                row["projected_defect_leading_initial_direction"]
                for row in results
            ],
            dtype=float,
        ),
        K=np.asarray(K),
        pole_bias=np.asarray(POLE_BIAS),
        initial_phi=np.asarray(INITIAL_PHI),
        method=np.asarray(METHOD),
        rtol=np.asarray(rtol),
        atol=np.asarray(atol),
        max_step=np.asarray(max_step),
        complex_step=np.asarray(complex_step),
        subspace_size=np.asarray(subspace_size),
        tangent_dimension=np.asarray(2 * K),
        full_tangent_space=np.asarray(subspace_size == 2 * K),
        projected_maps_available=np.asarray(subspace_size == 2 * K),
        metric=np.asarray(
            "sum_i[dtheta_i^2+sin(theta_i)^2*dphi_i^2]/K"
        ),
        tangent_jvp=np.asarray("batched complex step"),
        projected_local_definition=np.asarray(
            "P_Wr M_B(t) P_Wr^T, W_r={i0-r,...,i0+r}"
        ),
        projected_defect_definition=np.asarray(
            "P_D M_B(t) P_Wr^T, D={i0}"
        ),
    )


def load_cache(cache_path: Path) -> dict[str, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as archive:
        if int(np.asarray(archive["schema_version"]).item()) != SCHEMA_VERSION:
            raise ValueError("FTLE cache schema differs")
        return {key: np.asarray(archive[key]) for key in archive.files}


def write_tables(prefix: Path, archive: dict[str, np.ndarray]) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "display_id", "source_id", "region", "mu", "chi", "t",
                "lambda_max_FT", "log_gain", "gain",
            ]
        )
        for index, display_id in enumerate(archive["display_id"]):
            for time_value, ftle_value, log_value in zip(
                archive["times"], archive["ftle"][index],
                archive["log_gain"][index],
            ):
                writer.writerow(
                    [
                        str(display_id), str(archive["source_id"][index]),
                        str(archive["region"][index]),
                        f"{float(archive['mu'][index]):.12g}",
                        f"{float(archive['chi'][index]):.12g}",
                        f"{float(time_value):.12g}",
                        "" if not np.isfinite(ftle_value)
                        else f"{float(ftle_value):.12g}",
                        f"{float(log_value):.12g}",
                        f"{float(np.exp(np.clip(log_value, -700, 700))):.12g}",
                    ]
                )

    summary_rows = []
    for index, display_id in enumerate(archive["display_id"]):
        final_log_gain = float(archive["log_gain"][index, -1])
        summary_rows.append(
            {
                "display_id": str(display_id),
                "source_id": str(archive["source_id"][index]),
                "region": str(archive["region"][index]),
                "mu": float(archive["mu"][index]),
                "chi": float(archive["chi"][index]),
                "lambda_max_FT_at_T": float(archive["ftle"][index, -1]),
                "log_gain_at_T": final_log_gain,
                "gain_at_T": float(np.exp(np.clip(final_log_gain, -700, 700))),
                "minimum_output_abs_sin_theta": float(
                    archive["minimum_pole_margin"][index]
                ),
                "function_evaluations": int(
                    np.sum(archive["nfev"][index])
                ),
                "wall_seconds": float(
                    np.sum(archive["wall_seconds"][index])
                ),
            }
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "definition": (
            "lambda_max^FT(t)=log(s_max(M_B(t)))/t, where M_B is the "
            "full TDVP tangent map represented in all-site Bloch-orthonormal "
            "frames"
        ),
        "interpretation_warning": (
            "A positive finite-time value over T=10 does not by itself "
            "establish a positive asymptotic Lyapunov exponent or chaos."
        ),
        "protocol": {
            "K": int(np.asarray(archive["K"]).item()),
            "pole_bias": float(np.asarray(archive["pole_bias"]).item()),
            "initial_phi": float(np.asarray(archive["initial_phi"]).item()),
            "method": str(np.asarray(archive["method"]).item()),
            "rtol": float(np.asarray(archive["rtol"]).item()),
            "atol": float(np.asarray(archive["atol"]).item()),
            "max_step": float(np.asarray(archive["max_step"]).item()),
            "complex_step": float(
                np.asarray(archive["complex_step"]).item()
            ),
            "subspace_size": int(
                np.asarray(archive["subspace_size"]).item()
            ),
            "tangent_dimension": int(
                np.asarray(archive["tangent_dimension"]).item()
            ),
            "full_tangent_space": bool(
                np.asarray(archive["full_tangent_space"]).item()
            ),
            "metric": str(np.asarray(archive["metric"]).item()),
        },
        "points": summary_rows,
    }
    with prefix.with_suffix(".json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")


def write_projected_tables(
    prefix: Path, archive: dict[str, np.ndarray]
) -> None:
    """Write radius-resolved projected amplification in long-form tables."""

    if not bool(np.asarray(archive["projected_maps_available"]).item()):
        raise ValueError("projected maps require the complete tangent space")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    radii = np.asarray(archive["projected_radii"], dtype=int)
    times = np.asarray(archive["times"], dtype=float)
    local_ftle = np.asarray(archive["projected_local_ftle"], dtype=float)
    local_log = np.asarray(
        archive["projected_local_log_gain"], dtype=float
    )
    defect_ftle = np.asarray(
        archive["projected_defect_ftle"], dtype=float
    )
    defect_log = np.asarray(
        archive["projected_defect_log_gain"], dtype=float
    )

    with prefix.with_suffix(".csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "display_id", "source_id", "region", "mu", "chi",
                "radius", "window_site_count", "t",
                "lambda_Wr_from_Wr_FT", "log_gain_Wr_from_Wr",
                "gain_Wr_from_Wr", "lambda_D_from_Wr_FT",
                "log_gain_D_from_Wr", "gain_D_from_Wr",
            ]
        )
        for point_index, display_id in enumerate(archive["display_id"]):
            for radius_index, radius in enumerate(radii):
                for time_index, time_value in enumerate(times):
                    local_value = local_ftle[
                        point_index, radius_index, time_index
                    ]
                    defect_value = defect_ftle[
                        point_index, radius_index, time_index
                    ]
                    local_log_value = local_log[
                        point_index, radius_index, time_index
                    ]
                    defect_log_value = defect_log[
                        point_index, radius_index, time_index
                    ]
                    writer.writerow(
                        [
                            str(display_id),
                            str(archive["source_id"][point_index]),
                            str(archive["region"][point_index]),
                            f"{float(archive['mu'][point_index]):.12g}",
                            f"{float(archive['chi'][point_index]):.12g}",
                            int(radius),
                            int(2 * radius + 1),
                            f"{time_value:.12g}",
                            "" if not np.isfinite(local_value)
                            else f"{local_value:.12g}",
                            f"{local_log_value:.12g}",
                            f"{np.exp(np.clip(local_log_value, -700, 700)):.12g}",
                            "" if not np.isfinite(defect_value)
                            else f"{defect_value:.12g}",
                            f"{defect_log_value:.12g}",
                            f"{np.exp(np.clip(defect_log_value, -700, 700)):.12g}",
                        ]
                    )

    point_rows = []
    for point_index, display_id in enumerate(archive["display_id"]):
        radius_rows = []
        for radius_index, radius in enumerate(radii):
            local_final_log = float(local_log[point_index, radius_index, -1])
            defect_final_log = float(
                defect_log[point_index, radius_index, -1]
            )
            radius_rows.append(
                {
                    "radius": int(radius),
                    "window_site_count": int(2 * radius + 1),
                    "lambda_Wr_from_Wr_FT_at_T": float(
                        local_ftle[point_index, radius_index, -1]
                    ),
                    "gain_Wr_from_Wr_at_T": float(
                        np.exp(np.clip(local_final_log, -700, 700))
                    ),
                    "lambda_D_from_Wr_FT_at_T": float(
                        defect_ftle[point_index, radius_index, -1]
                    ),
                    "gain_D_from_Wr_at_T": float(
                        np.exp(np.clip(defect_final_log, -700, 700))
                    ),
                }
            )
        point_rows.append(
            {
                "display_id": str(display_id),
                "source_id": str(archive["source_id"][point_index]),
                "region": str(archive["region"][point_index]),
                "mu": float(archive["mu"][point_index]),
                "chi": float(archive["chi"][point_index]),
                "radii": radius_rows,
            }
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "definitions": {
            "W_r": "{i0-r,...,i0+r}",
            "local_to_local": str(
                np.asarray(archive["projected_local_definition"]).item()
            ),
            "defect_output": str(
                np.asarray(archive["projected_defect_definition"]).item()
            ),
            "rate": "lambda^FT(t)=log(s_max(projected map))/t",
            "metric": str(np.asarray(archive["metric"]).item()),
        },
        "interpretation_warning": (
            "Projected rates are finite-window local tangent-amplification "
            "diagnostics, not asymptotic coordinate-independent Lyapunov "
            "exponents. A negative value can mean that tangent weight left "
            "the observed window."
        ),
        "protocol": {
            "K": int(np.asarray(archive["K"]).item()),
            "pole_bias": float(np.asarray(archive["pole_bias"]).item()),
            "initial_phi": float(np.asarray(archive["initial_phi"]).item()),
            "T": float(times[-1]),
            "output_dt": float(times[1] - times[0]),
            "method": str(np.asarray(archive["method"]).item()),
            "rtol": float(np.asarray(archive["rtol"]).item()),
            "atol": float(np.asarray(archive["atol"]).item()),
            "max_step": float(np.asarray(archive["max_step"]).item()),
        },
        "points": point_rows,
    }
    with prefix.with_suffix(".json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")


def plot_results(prefix: Path, archive: dict[str, np.ndarray]) -> None:
    plt.style.use(resolve_style_file())

    # Geometry and typography controls kept together for manual adjustment.
    figure_width = 7.12
    figure_height = 2.72
    left_margin = 0.085
    right_margin = 0.985
    bottom_margin = 0.19
    top_margin = 0.93
    panel_gap = 0.17
    curve_width = 1.15

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(figure_width, figure_height),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(
        left=left_margin,
        right=right_margin,
        bottom=bottom_margin,
        top=top_margin,
        wspace=panel_gap,
    )

    region_indices = {
        "I": np.flatnonzero(archive["region"] == "I"),
        "III": np.flatnonzero(archive["region"] == "III"),
    }
    palettes = {
        "I": plt.get_cmap("Blues")(np.linspace(0.45, 0.92, 5)),
        "III": plt.get_cmap("YlOrBr")(np.linspace(0.38, 0.88, 5)),
    }
    times = archive["times"]
    for panel, (axis, region) in enumerate(zip(axes, ("I", "III"))):
        indices = region_indices[region]
        colors = palettes[region]
        if len(indices) != len(colors):
            colors = plt.get_cmap(
                "Blues" if region == "I" else "YlOrBr"
            )(np.linspace(0.45, 0.9, max(len(indices), 1)))
        for color, index in zip(colors, indices):
            axis.plot(
                times[1:],
                archive["ftle"][index, 1:],
                color=color,
                linewidth=curve_width,
                label=str(archive["display_id"][index]),
            )
        axis.axhline(0.0, color="0.72", linewidth=0.65, linestyle="--")
        axis.set_xlim(float(times[1]), float(times[-1]))
        axis.tick_params(direction="in", top=True, right=True)
        axis.text(
            0.025,
            0.96,
            f"({chr(ord('a') + panel)})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="semibold",
            fontsize=8.2,
        )
        axis.text(
            0.975,
            0.96,
            f"Region {region}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.2,
        )
        if len(indices):
            axis.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 0.875),
                ncol=min(5, len(indices)),
                frameon=False,
                handlelength=1.35,
                handletextpad=0.35,
                columnspacing=0.75,
                borderaxespad=0.0,
                fontsize=7.3,
            )

    fig.supxlabel(r"$t$", y=0.035, fontsize=10)
    fig.supylabel(
        r"$\lambda_{\max}^{\mathrm{FT}}(t)$",
        x=0.018,
        fontsize=10,
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(
        prefix.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def plot_local_stability(
    prefix: Path, archive: dict[str, np.ndarray]
) -> tuple[Path, Path]:
    """Plot the defect-site r=0 tangent rate used in the manuscript."""

    if not bool(np.asarray(archive["projected_maps_available"]).item()):
        raise ValueError("local defect response requires the complete tangent map")
    radii = np.asarray(archive["projected_radii"], dtype=int)
    zero_radius = np.flatnonzero(radii == 0)
    if zero_radius.size != 1:
        raise ValueError("cache must contain exactly one r=0 response")
    values = np.asarray(
        archive["projected_defect_ftle"][:, int(zero_radius[0]), :],
        dtype=float,
    )

    plt.style.use(resolve_style_file())

    # Single-column PRB geometry.  The panels are stacked because each must
    # carry five curves and the two regions require different vertical scales.
    figure_width = 3.39
    figure_height = 3.85
    left_margin = 0.19
    right_margin = 0.985
    bottom_margin = 0.115
    top_margin = 0.965
    vertical_gap = 0.20
    curve_width = 1.10
    # The five colors encode the ordered points within each region.  A single
    # solid line style avoids a redundant second visual channel.
    colors = ("#0072B2", "#56B4E9", "#009E73", "#E69F00", "#CC79A7")

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(figure_width, figure_height),
        sharex=True,
    )
    fig.subplots_adjust(
        left=left_margin,
        right=right_margin,
        bottom=bottom_margin,
        top=top_margin,
        hspace=vertical_gap,
    )

    times = np.asarray(archive["times"], dtype=float)
    for panel, (axis, region) in enumerate(zip(axes, ("I", "III"))):
        indices = np.flatnonzero(archive["region"] == region)
        if indices.size != 5:
            raise ValueError(f"expected five Region-{region} trajectories")
        for color, index in zip(colors, indices):
            axis.plot(
                times[1:],
                values[index, 1:],
                color=color,
                linestyle="-",
                linewidth=curve_width,
                label=str(archive["display_id"][index]),
            )
        axis.axhline(0.0, color="0.70", linewidth=0.6, linestyle="--")
        axis.set_xlim(0.0, float(times[-1]))
        axis.set_xticks([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        axis.tick_params(
            direction="in",
            top=True,
            right=True,
            labelsize=7.7,
            length=2.6,
            width=0.65,
        )
        axis.text(
            0.025,
            0.95,
            f"({chr(ord('a') + panel)})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="semibold",
            fontsize=8.2,
        )
        axis.text(
            0.975,
            0.95,
            f"Region {region}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.0,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.25},
            zorder=5,
        )
        legend_position = (
            {"loc": "upper center", "bbox_to_anchor": (0.64, 0.82)}
            if region == "I"
            else {"loc": "lower right", "bbox_to_anchor": (0.98, 0.055)}
        )
        axis.legend(
            **legend_position,
            ncol=5,
            frameon=False,
            handlelength=1.25,
            handletextpad=0.25,
            columnspacing=0.45,
            borderaxespad=0.0,
            fontsize=6.7,
        )

    axes[-1].set_xlabel(r"$t$", fontsize=9.0, labelpad=2.0)
    fig.supylabel(r"$\lambda_{\max}(t)$", x=0.035, fontsize=9.5)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = prefix.with_suffix(".pdf")
    png_path = prefix.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return pdf_path, png_path


def _radius_styles(radii: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    colors = np.asarray(["#202124", "#0072B2", "#D55E00", "#009E73"])
    line_styles = ("-", "--", "-.", ":")
    if len(radii) > len(colors):
        raise ValueError("add radius styles before plotting more windows")
    return colors[: len(radii)], line_styles[: len(radii)]


def plot_projected_grid(
    prefix: Path,
    archive: dict[str, np.ndarray],
    *,
    quantity: str,
) -> tuple[Path, Path]:
    """Plot four radii for every selected point as a two-by-five grid."""

    if quantity == "local":
        values = archive["projected_local_ftle"]
        suffix = "local_to_local"
        ylabel = r"$\lambda_{W_r\leftarrow W_r}^{\mathrm{FT}}(t)$"
    elif quantity == "defect":
        values = archive["projected_defect_ftle"]
        suffix = "defect_output"
        ylabel = r"$\lambda_{D\leftarrow W_r}^{\mathrm{FT}}(t)$"
    else:
        raise ValueError(f"unknown projected quantity {quantity!r}")

    plt.style.use(resolve_style_file())
    radii = np.asarray(archive["projected_radii"], dtype=int)
    colors, line_styles = _radius_styles(radii)
    times = np.asarray(archive["times"], dtype=float)

    # Geometry controls: two rows reproduce the Region-I/III portrait order;
    # small horizontal gaps keep all ten time traces readable at 7.12 inches.
    figure_width = 7.12
    figure_height = 3.82
    left_margin = 0.078
    right_margin = 0.992
    bottom_margin = 0.13
    top_margin = 0.875
    horizontal_gap = 0.10
    vertical_gap = 0.13
    curve_width = 1.0

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(figure_width, figure_height),
        sharex=True,
        sharey="row",
    )
    fig.subplots_adjust(
        left=left_margin,
        right=right_margin,
        bottom=bottom_margin,
        top=top_margin,
        wspace=horizontal_gap,
        hspace=vertical_gap,
    )

    for point_index, axis in enumerate(axes.ravel()):
        for radius_index, (radius, color, line_style) in enumerate(
            zip(radii, colors, line_styles)
        ):
            axis.plot(
                times[1:],
                values[point_index, radius_index, 1:],
                color=color,
                linestyle=line_style,
                linewidth=curve_width,
                label=rf"$r={radius}$",
            )
        axis.axhline(0.0, color="0.72", linewidth=0.55, linestyle="--")
        axis.set_xlim(float(times[1]), float(times[-1]))
        axis.set_xticks([2.0, 6.0, 10.0])
        axis.tick_params(
            direction="in",
            top=True,
            right=True,
            labelsize=7.1,
            length=2.5,
            width=0.65,
        )
        axis.text(
            0.965,
            0.94,
            str(archive["display_id"][point_index]),
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7.7,
        )
        if point_index in (0, 5):
            region = "I" if point_index == 0 else "III"
            axis.text(
                0.035,
                0.94,
                f"Region {region}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=7.4,
            )

    legend_handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.995),
        ncol=len(radii),
        frameon=False,
        handlelength=2.1,
        handletextpad=0.45,
        columnspacing=1.25,
        fontsize=7.7,
    )
    fig.supxlabel(r"$t$", y=0.025, fontsize=9.5)
    fig.supylabel(ylabel, x=0.012, fontsize=9.5)

    pdf_path = prefix.with_name(f"{prefix.name}_{suffix}.pdf")
    png_path = prefix.with_name(f"{prefix.name}_{suffix}.png")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return pdf_path, png_path


def plot_projected_endpoint_summary(
    prefix: Path, archive: dict[str, np.ndarray]
) -> tuple[Path, Path]:
    """Show the radius dependence of both projected rates at the final time."""

    plt.style.use(resolve_style_file())
    radii = np.asarray(archive["projected_radii"], dtype=int)
    region_indices = {
        "I": np.flatnonzero(archive["region"] == "I"),
        "III": np.flatnonzero(archive["region"] == "III"),
    }
    palettes = {
        "I": plt.get_cmap("Blues")(np.linspace(0.45, 0.92, 5)),
        "III": plt.get_cmap("YlOrBr")(np.linspace(0.38, 0.88, 5)),
    }
    quantities = (
        (
            "projected_local_ftle",
            r"$\lambda_{W_r\leftarrow W_r}^{\mathrm{FT}}(T)$",
        ),
        (
            "projected_defect_ftle",
            r"$\lambda_{D\leftarrow W_r}^{\mathrm{FT}}(T)$",
        ),
    )

    # Geometry controls for a compact convergence-oriented four-panel view.
    figure_width = 7.12
    figure_height = 4.35
    left_margin = 0.10
    right_margin = 0.985
    bottom_margin = 0.12
    top_margin = 0.94
    horizontal_gap = 0.16
    vertical_gap = 0.16

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(figure_width, figure_height),
        sharex=True,
    )
    fig.subplots_adjust(
        left=left_margin,
        right=right_margin,
        bottom=bottom_margin,
        top=top_margin,
        wspace=horizontal_gap,
        hspace=vertical_gap,
    )
    for row, (key, ylabel) in enumerate(quantities):
        values = archive[key]
        for column, region in enumerate(("I", "III")):
            axis = axes[row, column]
            indices = region_indices[region]
            for color, point_index in zip(palettes[region], indices):
                axis.plot(
                    radii,
                    values[point_index, :, -1],
                    color=color,
                    linewidth=1.1,
                    marker="o",
                    markersize=3.2,
                    label=str(archive["display_id"][point_index]),
                )
            axis.axhline(
                0.0, color="0.72", linewidth=0.6, linestyle="--"
            )
            axis.set_xticks(radii)
            axis.tick_params(direction="in", top=True, right=True)
            axis.text(
                0.025,
                0.95,
                f"({chr(ord('a') + 2 * row + column)})",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontweight="semibold",
                fontsize=8.2,
            )
            if row == 0:
                axis.text(
                    0.54,
                    0.95,
                    f"Region {region}",
                    transform=axis.transAxes,
                    ha="center",
                    va="top",
                    fontsize=8.2,
                )
            if column == 0:
                axis.set_ylabel(ylabel)
    handles_i, labels_i = axes[0, 0].get_legend_handles_labels()
    handles_iii, labels_iii = axes[0, 1].get_legend_handles_labels()
    fig.legend(
        handles_i,
        labels_i,
        loc="upper center",
        bbox_to_anchor=(0.29, 0.995),
        ncol=5,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.3,
        columnspacing=0.65,
        fontsize=7.2,
    )
    fig.legend(
        handles_iii,
        labels_iii,
        loc="upper center",
        bbox_to_anchor=(0.76, 0.995),
        ncol=5,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.3,
        columnspacing=0.65,
        fontsize=7.2,
    )
    fig.supxlabel(r"window radius $r$", y=0.025, fontsize=9.5)

    pdf_path = prefix.with_name(f"{prefix.name}_endpoint_radius.pdf")
    png_path = prefix.with_name(f"{prefix.name}_endpoint_radius.png")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return pdf_path, png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX
    )
    parser.add_argument(
        "--projected-prefix", type=Path, default=DEFAULT_PROJECTED_PREFIX
    )
    parser.add_argument(
        "--stability-prefix", type=Path, default=DEFAULT_STABILITY_PREFIX
    )
    parser.add_argument("--t-max", type=float, default=T_MAX)
    parser.add_argument("--output-dt", type=float, default=OUTPUT_DT)
    parser.add_argument("--subspace-size", type=int, default=2 * K)
    parser.add_argument("--rtol", type=float, default=RTOL)
    parser.add_argument("--atol", type=float, default=ATOL)
    parser.add_argument("--max-step", type=float, default=MAX_STEP)
    parser.add_argument("--complex-step", type=float, default=COMPLEX_STEP)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument(
        "--points",
        nargs="*",
        default=None,
        help="display IDs to compute, e.g. I2 III3; default is all ten",
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite the numerical cache"
    )
    parser.add_argument(
        "--plot-only", action="store_true", help="reuse an existing cache"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="suppress interval progress"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_points = load_points(args.manifest)
    if args.points:
        requested = set(args.points)
        points = tuple(
            point for point in all_points if point.display_id in requested
        )
        missing = requested - {point.display_id for point in points}
        if missing:
            raise ValueError(f"unknown point IDs: {sorted(missing)}")
    else:
        points = all_points

    if args.plot_only:
        if not args.cache.is_file():
            raise FileNotFoundError(args.cache)
    else:
        if args.cache.exists() and not args.force:
            raise FileExistsError(
                f"cache exists: {args.cache}; use --plot-only or --force"
            )
        validate_vectorized_eom(points)
        times = output_times(args.t_max, args.output_dt)
        results = []
        for point_index, point in enumerate(points):
            print(
                f"Starting {point.display_id}: "
                f"(mu,chi)=({point.mu:.6g},{point.chi:.6g}), "
                f"tangent subspace={args.subspace_size}/{2*K}",
                flush=True,
            )
            results.append(
                compute_point_ftle(
                    point,
                    times,
                    subspace_size=args.subspace_size,
                    method=METHOD,
                    rtol=args.rtol,
                    atol=args.atol,
                    max_step=args.max_step,
                    complex_step=args.complex_step,
                    random_seed=args.seed + point_index,
                    progress=not args.quiet,
                )
            )
        write_cache(
            args.cache,
            results,
            times=times,
            subspace_size=args.subspace_size,
            rtol=args.rtol,
            atol=args.atol,
            max_step=args.max_step,
            complex_step=args.complex_step,
        )

    archive = load_cache(args.cache)
    write_tables(args.output_prefix, archive)
    plot_results(args.output_prefix, archive)
    stability_paths = plot_local_stability(args.stability_prefix, archive)
    projected_paths: list[Path] = []
    if bool(np.asarray(archive["projected_maps_available"]).item()):
        write_projected_tables(args.projected_prefix, archive)
        if archive["display_id"].size == 10:
            projected_paths.extend(
                plot_projected_grid(
                    args.projected_prefix, archive, quantity="local"
                )
            )
            projected_paths.extend(
                plot_projected_grid(
                    args.projected_prefix, archive, quantity="defect"
                )
            )
            projected_paths.extend(
                plot_projected_endpoint_summary(
                    args.projected_prefix, archive
                )
            )
    if not args.plot_only:
        print(f"Wrote {args.cache}")
    else:
        print(f"Read {args.cache}")
    print(f"Wrote {args.output_prefix.with_suffix('.csv')}")
    print(f"Wrote {args.output_prefix.with_suffix('.json')}")
    print(f"Wrote {args.output_prefix.with_suffix('.pdf')}")
    print(f"Wrote {args.output_prefix.with_suffix('.png')}")
    for path in stability_paths:
        print(f"Wrote {path}")
    if projected_paths:
        print(f"Wrote {args.projected_prefix.with_suffix('.csv')}")
        print(f"Wrote {args.projected_prefix.with_suffix('.json')}")
        for path in projected_paths:
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
