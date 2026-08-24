#!/usr/bin/env python3
"""Select and plot five well-separated low-leakage points in region V.

This supplementary figure loads the code-generated leakage landscape,
greedily selects the five lowest map-leakage points in a declared positive-mu
window subject to a minimum Euclidean separation, and then reintegrates every
selected trajectory with the strict Paper-A K=100 protocol.

Run from the repository root with the QuSpin environment:

    conda run -n quspin python fig1fig2/explore_region_v_trajectories.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import explore_resonance_mode_transition as common


HERE = Path(__file__).resolve().parent
LEAKAGE_CACHE = HERE / "data" / "fig1_average_leakage.npz"
DATA_CACHE = HERE / "data" / "region_v_classical_trajectories.npz"
OUTPUT_PDF = HERE / "output" / "region_v_classical_trajectories.pdf"
OUTPUT_PNG = HERE / "output" / "region_v_classical_trajectories.png"
OUTPUT_JSON = HERE / "output" / "region_v_classical_trajectories.json"

SCHEMA_VERSION = 1
REGION_V_WINDOW = {
    "mu_min": 0.20,
    "mu_max": 1.20,
    "chi_min": 1.25,
    "chi_max": 1.95,
}
POINT_COUNT = 5
MINIMUM_DISTANCE = 0.25
MAXIMUM_MAP_LEAKAGE = 0.10
TRAJECTORY_COLOR = "#56B4E9"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_points(
    mu_grid: np.ndarray,
    chi_grid: np.ndarray,
    leakage: np.ndarray,
) -> list[common.SelectedPoint]:
    """Greedily minimize map leakage subject to the declared separation."""

    candidates: list[tuple[float, float, float]] = []
    for mu_index, mu_value in enumerate(mu_grid):
        if not (
            REGION_V_WINDOW["mu_min"]
            <= mu_value
            <= REGION_V_WINDOW["mu_max"]
        ):
            continue
        for chi_index, chi_value in enumerate(chi_grid):
            if not (
                REGION_V_WINDOW["chi_min"]
                <= chi_value
                <= REGION_V_WINDOW["chi_max"]
            ):
                continue
            map_value = float(leakage[mu_index, chi_index])
            if map_value < MAXIMUM_MAP_LEAKAGE:
                candidates.append(
                    (map_value, float(mu_value), float(chi_value))
                )
    candidates.sort(key=lambda row: (row[0], row[1], row[2]))

    chosen: list[tuple[float, float, float]] = []
    for candidate in candidates:
        _, mu_value, chi_value = candidate
        if all(
            np.hypot(mu_value - old_mu, chi_value - old_chi)
            >= MINIMUM_DISTANCE - 1.0e-12
            for _, old_mu, old_chi in chosen
        ):
            chosen.append(candidate)
        if len(chosen) == POINT_COUNT:
            break
    if len(chosen) != POINT_COUNT:
        raise RuntimeError(
            f"selected only {len(chosen)} region-V points, expected "
            f"{POINT_COUNT}"
        )

    # Display left-to-right in mu; the selection rank is retained in the
    # note and output metadata.
    ranked = {
        (mu_value, chi_value): rank
        for rank, (_, mu_value, chi_value) in enumerate(chosen, start=1)
    }
    ordered = sorted(chosen, key=lambda row: (row[1], row[2]))
    return [
        common.SelectedPoint(
            f"V{display_index}",
            "region_v",
            mu_value,
            chi_value,
            (
                f"greedy leakage rank {ranked[(mu_value, chi_value)]} "
                f"under pairwise distance >= {MINIMUM_DISTANCE}"
            ),
        )
        for display_index, (_, mu_value, chi_value) in enumerate(
            ordered, start=1
        )
    ]


def recurrence_metrics(
    times: np.ndarray,
    theta_defect: np.ndarray,
    phi_defect: np.ndarray,
) -> dict[str, float]:
    """Return a finite-time return diagnostic in the displayed plane."""

    x_value = np.sin(theta_defect) * np.cos(phi_defect)
    y_value = np.sin(theta_defect) * np.sin(phi_defect)
    distance = np.hypot(x_value - x_value[0], y_value - y_value[0])
    mask = times >= 1.0
    return_index = np.flatnonzero(mask)[np.argmin(distance[mask])]
    return {
        "minimum_return_distance_after_t1": float(distance[return_index]),
        "minimum_return_time_after_t1": float(times[return_index]),
    }


def main() -> int:
    with np.load(LEAKAGE_CACHE, allow_pickle=False) as background:
        valid, reason = common.validate_background_cache(background)
        if not valid:
            raise ValueError(f"incompatible leakage background: {reason}")
        mu_grid = np.asarray(background["mu"], dtype=float)
        chi_grid = np.asarray(background["chi"], dtype=float)
        leakage = np.asarray(background["avg_q_leak"], dtype=float)
        points = select_points(mu_grid, chi_grid, leakage)

    times = common.common_times()
    theta_values: list[np.ndarray] = []
    phi_values: list[np.ndarray] = []
    qleak_values: list[np.ndarray] = []
    rows: list[dict[str, object]] = []

    for point in points:
        theta, phi, qleak, nfev, pole_margin = common.integrate_point(
            point, times
        )
        metrics = common.trajectory_metrics(times, theta[1], phi[1], qleak)
        metrics.update(recurrence_metrics(times, theta[1], phi[1]))
        with np.load(LEAKAGE_CACHE, allow_pickle=False) as background:
            map_leakage = common.bilinear_value(
                np.asarray(background["mu"]),
                np.asarray(background["chi"]),
                np.asarray(background["avg_q_leak"]),
                point.mu,
                point.chi,
            )
        theta_values.append(theta)
        phi_values.append(phi)
        qleak_values.append(qleak)
        rows.append(
            {
                "point": point.point_id,
                "mu": point.mu,
                "chi": point.chi,
                "defect_detuning": 2.0 * point.mu + point.chi,
                "selection_note": point.selection_note,
                "background_map_mean_leakage": map_leakage,
                "strict_mean_leakage": metrics.pop("mean_leakage"),
                **metrics,
                "function_evaluations": nfev,
                "minimum_abs_sin_theta_all_sites": pole_margin,
            }
        )

    DATA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA_CACHE,
        schema_version=SCHEMA_VERSION,
        background_map_sha256=sha256(LEAKAGE_CACHE),
        times=times,
        point_id=np.asarray([point.point_id for point in points]),
        mu=np.asarray([point.mu for point in points]),
        chi=np.asarray([point.chi for point in points]),
        theta=np.asarray(theta_values),
        phi=np.asarray(phi_values),
        qleak=np.asarray(qleak_values),
        K=common.TDVP_PERIOD,
        pole_bias=common.POLE_BIAS,
        initial_phi=common.INITIAL_PHI,
        method=common.METHOD,
        rtol=common.RTOL,
        atol=common.ATOL,
        max_step=common.MAX_STEP,
    )

    import matplotlib.pyplot as plt

    plt.style.use(common.resolve_style_file())
    figure = plt.figure(figsize=(7.15, 1.90))
    grid = figure.add_gridspec(
        1,
        POINT_COUNT,
        left=0.085,
        right=0.995,
        bottom=0.205,
        top=0.855,
        wspace=0.16,
    )
    for index, (point, theta, phi) in enumerate(
        zip(points, theta_values, phi_values)
    ):
        axis = figure.add_subplot(grid[0, index])
        common.plot_phase_portrait(
            axis,
            theta[1],
            phi[1],
            TRAJECTORY_COLOR,
            point.point_id,
            common.panel_letter(index),
        )
        axis.set_title(
            rf"$(\mu,\chi)=({point.mu:.2f},{point.chi:.2f})$",
            fontsize=6.7,
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
        if index > 0:
            axis.tick_params(labelleft=False)
    figure.text(
        0.54,
        0.035,
        r"$\sin\theta_{i_0}\cos\phi_{i_0}$",
        ha="center",
        va="bottom",
        fontsize=10.5,
    )
    figure.text(
        0.014,
        0.53,
        r"$\sin\theta_{i_0}\sin\phi_{i_0}$",
        ha="left",
        va="center",
        rotation=90,
        fontsize=10.5,
    )
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.025)
    figure.savefig(
        OUTPUT_PNG,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.025,
    )
    plt.close(figure)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "description": "Supplementary region-V classical trajectories",
        "selection": {
            "window": REGION_V_WINDOW,
            "point_count": POINT_COUNT,
            "minimum_pairwise_distance": MINIMUM_DISTANCE,
            "maximum_map_leakage": MAXIMUM_MAP_LEAKAGE,
            "rule": (
                "greedily minimize generated-map leakage subject to the "
                "minimum Euclidean separation, then display in increasing mu"
            ),
        },
        "protocol": {
            "spin": 0.5,
            "K": common.TDVP_PERIOD,
            "time_window": [0.0, common.T_MAX],
            "sample_count": common.SAMPLE_COUNT,
            "pole_bias": common.POLE_BIAS,
            "initial_phi": common.INITIAL_PHI,
            "solver": common.METHOD,
            "rtol": common.RTOL,
            "atol": common.ATOL,
            "max_step": common.MAX_STEP,
        },
        "artifacts": {
            "pdf": str(OUTPUT_PDF.resolve()),
            "png": str(OUTPUT_PNG.resolve()),
            "npz": str(DATA_CACHE.resolve()),
        },
        "points": rows,
    }
    with OUTPUT_JSON.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
