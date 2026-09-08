# Arbitrary-period TDVP for generalized PXP models

This repository contains the Python source code used to generate every
numerical figure in the accompanying manuscript.  It intentionally contains
no numerical dataset, checkpoint, processed table, or rendered figure: all
such products are generated locally by the scripts.

## Environment

Create and activate the manuscript environment with

```bash
conda env create -f environment.yml
conda activate quspin
```

The calculations use NumPy/SciPy/QuSpin only.  Mathematica is not used for
numerical evolution.

## One-command reproduction

From the repository root, generate all eight figures used by the manuscript:

```bash
PAPER_A_PYTHON=python PAPER_A_WORKERS=8 \
  scripts/reproduce_paper_a_figures.sh --from-scratch
```

The command generates the leakage landscape, reselects representative points,
integrates the ED and TDVP dynamics, and creates

- `fig1fig2/output/fig1_leakage_map.pdf`;
- `fig1fig2/output/fig2_orbit_mode_transition.pdf`;
- `fig1fig2/output/region_i_iii_projected_ftle_defect_output.pdf`;
- `output/defect_delta_m_convergence/defect_delta_m_convergence.pdf`;
- `fig1fig2/output/resonance_line_horizontal_width.pdf`;
- `fig1fig2/output/ed_tdvp_defect_profiles.pdf`;
- `fig1fig2/output/region_i_iii_projected_ftle_endpoint_radius.pdf`;
- `fig1fig2/output/regions_i_v_trajectory_atlas.pdf`.

The full `301 x 201` leakage scan is the expensive stage.  It is checkpointed
one mass-parameter row at a time and can be resumed by rerunning the command
without `--from-scratch`.  Propagation of the complete `200 x 200` tangent map
for all ten representative trajectories is also computationally intensive;
its compressed checkpoint is likewise reused unless `--from-scratch` is
specified.

## Numerical protocol

All active TDVP panels use a period-`K=100` cell, pole regularization
`epsilon=1e-3`, initial azimuth `phi_i=0`, and time window `0 <= t <= 10`.
Strict representative trajectories use DOP853 with `rtol=1e-9`,
`atol=1e-11`, and maximum step `0.02`.  ED comparisons use a periodic
`L=24` constrained ring.  The broad leakage map uses its separately recorded
exploratory integration tolerance and is used only for selection and
visualization.

Generated caches carry schema and protocol metadata.  Plotting and analysis
drivers reject incompatible caches rather than silently mixing initial-state
or solver conventions.

## Source layout

- `fig1fig2/tdvpfun.py`: spin-1/2 arbitrary-period TDVP equations,
  observables, transfer weights, and intensive leakage.
- `fig1fig2/EDfun.py` and `fig1fig2/pxpbasisS.py`: constrained exact
  diagonalization and initial-state construction.
- `fig1fig2/explore_resonance_mode_transition.py`: leakage-map generation,
  strict orbit integrations, main phase portraits, and the complete I-V atlas.
- `fig1fig2/select_low_leakage_points.py` and
  `fig1fig2/figure2_selection.py`: deterministic point-selection protocol and
  the displayed orbit coordinates.
- `fig1fig2/reproduce_fig1_fig2.py`: main leakage map and ED-TDVP response
  panels.
- `fig1fig2/plot_resonance_line_width.py`: resonance-line width scan.
- `fig1fig2/explore_region_v_trajectories.py`: region-V selection and strict
  trajectories used in the appendix atlas.
- `fig1fig2/compute_finite_time_lyapunov.py`: full Bloch-frame tangent-map
  propagation, radius-resolved local projections, nonlinear validation data,
  and the two finite-time tangent-amplification figures.
- `scripts/benchmark_defect_delta_m_convergence.py`: cell-size and pole-
  regularization convergence of the defect-site response.
- `scripts/reproduce_paper_a_figures.sh`: complete orchestration entry point.

## Generated files

The directories `fig1fig2/data/`, `fig1fig2/output/`, `output/`, and `tmp/`
are created at runtime and ignored by Git.  They can be deleted and recreated
from the source code.
