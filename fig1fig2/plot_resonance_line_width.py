#!/usr/bin/env python3
"""Plot defect-orbit horizontal width along the bare resonance line.

Every point is reintegrated with the strict Paper-A K=100 protocol.  The
uniform scan uses 2*mu+chi=0 and chi=0,0.05,...,2.0.  Points whose strict
time-averaged leakage exceeds 0.1 are shown in gray, as requested in the
supplementary diagnostic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import explore_resonance_mode_transition as common


HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "resonance_line_horizontal_width.npz"
OUTPUT_PDF = HERE / "output" / "resonance_line_horizontal_width.pdf"
OUTPUT_PNG = HERE / "output" / "resonance_line_horizontal_width.png"
OUTPUT_JSON = HERE / "output" / "resonance_line_horizontal_width.json"

SCHEMA_VERSION = 1
CHI_VALUES = np.linspace(0.0, 2.0, 41)
LEAKAGE_THRESHOLD = 0.10


def cache_valid(archive: np.lib.npyio.NpzFile) -> bool:
    required = {
        "schema_version",
        "chi",
        "mu",
        "horizontal_span",
        "mean_leakage",
        "K",
        "pole_bias",
        "initial_phi",
        "method",
        "rtol",
        "atol",
        "max_step",
        "T",
    }
    if not required.issubset(archive.files):
        return False
    return bool(
        int(np.asarray(archive["schema_version"]).item()) == SCHEMA_VERSION
        and np.array_equal(np.asarray(archive["chi"]), CHI_VALUES)
        and int(np.asarray(archive["K"]).item()) == common.TDVP_PERIOD
        and float(np.asarray(archive["pole_bias"]).item()) == common.POLE_BIAS
        and float(np.asarray(archive["initial_phi"]).item()) == common.INITIAL_PHI
        and str(np.asarray(archive["method"]).item()) == common.METHOD
        and float(np.asarray(archive["rtol"]).item()) == common.RTOL
        and float(np.asarray(archive["atol"]).item()) == common.ATOL
        and float(np.asarray(archive["max_step"]).item()) == common.MAX_STEP
        and float(np.asarray(archive["T"]).item()) == common.T_MAX
    )


def compute_scan() -> dict[str, np.ndarray]:
    times = common.common_times()
    rows: list[dict[str, float]] = []
    for index, chi_value in enumerate(CHI_VALUES, start=1):
        mu_value = -0.5 * float(chi_value)
        point = common.SelectedPoint(
            f"R{index:02d}",
            "resonance_line_width",
            mu_value,
            float(chi_value),
            "uniform sample on 2*mu+chi=0",
        )
        theta, phi, qleak, nfev, pole_margin = common.integrate_point(
            point, times
        )
        metrics = common.trajectory_metrics(times, theta[1], phi[1], qleak)
        rows.append(
            {
                "chi": float(chi_value),
                "mu": mu_value,
                "horizontal_span": metrics["horizontal_span"],
                "mean_leakage": metrics["mean_leakage"],
                "centroid_x": metrics["centroid_sin_theta_cos_phi"],
                "function_evaluations": float(nfev),
                "minimum_abs_sin_theta": float(pole_margin),
            }
        )
        print(
            f"[{index:02d}/{len(CHI_VALUES)}] chi={chi_value:.2f}, "
            f"DeltaX={metrics['horizontal_span']:.6f}, "
            f"Gamma_bar={metrics['mean_leakage']:.6f}",
            flush=True,
        )

    data = {
        key: np.asarray([row[key] for row in rows], dtype=float)
        for key in rows[0]
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE,
        schema_version=SCHEMA_VERSION,
        **data,
        K=common.TDVP_PERIOD,
        pole_bias=common.POLE_BIAS,
        initial_phi=common.INITIAL_PHI,
        method=common.METHOD,
        rtol=common.RTOL,
        atol=common.ATOL,
        max_step=common.MAX_STEP,
        T=common.T_MAX,
        sample_count=common.SAMPLE_COUNT,
    )
    return data


def load_or_compute(force: bool) -> dict[str, np.ndarray]:
    if CACHE.exists() and not force:
        with np.load(CACHE, allow_pickle=False) as archive:
            if cache_valid(archive):
                return {
                    key: np.asarray(archive[key], dtype=float)
                    for key in (
                        "chi",
                        "mu",
                        "horizontal_span",
                        "mean_leakage",
                        "centroid_x",
                        "function_evaluations",
                        "minimum_abs_sin_theta",
                    )
                }
    return compute_scan()


def make_plot(data: dict[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    plt.style.use(common.resolve_style_file())

    # Single-column APS geometry for the accepted large-chi trend only.
    figure = plt.figure(figsize=(3.40, 2.30))
    axis = figure.add_axes((0.18, 0.20, 0.79, 0.75))

    chi = data["chi"]
    width = data["horizontal_span"]
    accepted = data["mean_leakage"] <= LEAKAGE_THRESHOLD
    displayed = (chi > 0.5) & accepted

    axis.plot(
        chi[displayed], width[displayed],
        color="#4D4D4D", linewidth=0.80, zorder=1
    )
    axis.scatter(
        chi[displayed],
        width[displayed],
        s=17,
        marker="o",
        facecolor="#009E73",
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    axis.set_xlim(0.65, 2.02)
    axis.set_ylim(0.10, 0.95)
    axis.set_xticks((0.75, 1.0, 1.25, 1.5, 1.75, 2.0))
    axis.set_yticks((0.2, 0.4, 0.6, 0.8))
    axis.set_xlabel(r"$\chi$ on $2\mu+\chi=0$", labelpad=2.0)
    axis.set_ylabel(r"$\Delta X_{i_0}$", labelpad=2.5)
    axis.tick_params(
        direction="out",
        top=False,
        right=False,
        length=2.6,
        width=0.65,
        pad=1.8,
    )

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.025)
    figure.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)

    rows = [
        {key: float(data[key][index]) for key in data}
        for index in range(len(data["chi"]))
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "Uniform strict K=100 scan of horizontal defect-orbit width "
            "along the bare resonance line"
        ),
        "scan": {
            "chi_min": float(CHI_VALUES[0]),
            "chi_max": float(CHI_VALUES[-1]),
            "point_count": len(CHI_VALUES),
            "spacing": float(CHI_VALUES[1] - CHI_VALUES[0]),
            "constraint": "2*mu+chi=0",
            "leakage_threshold": LEAKAGE_THRESHOLD,
        },
        "protocol": {
            "K": common.TDVP_PERIOD,
            "epsilon": common.POLE_BIAS,
            "initial_phi": common.INITIAL_PHI,
            "time_window": [0.0, common.T_MAX],
            "solver": common.METHOD,
            "rtol": common.RTOL,
            "atol": common.ATOL,
            "max_step": common.MAX_STEP,
        },
        "points": rows,
    }
    with OUTPUT_JSON.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_or_compute(args.force)
    make_plot(data)
    print(
        json.dumps(
            {
                "status": "ok",
                "pdf": str(OUTPUT_PDF.resolve()),
                "png": str(OUTPUT_PNG.resolve()),
                "data": str(CACHE.resolve()),
                "gray_point_count": int(
                    np.count_nonzero(data["mean_leakage"] > LEAKAGE_THRESHOLD)
                ),
                "displayed_point_count": int(
                    np.count_nonzero(
                        (data["chi"] > 0.5)
                        & (data["mean_leakage"] <= LEAKAGE_THRESHOLD)
                    )
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
