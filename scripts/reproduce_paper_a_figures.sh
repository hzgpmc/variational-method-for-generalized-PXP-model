#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
worker_count="${PAPER_A_WORKERS:-8}"

if [[ -n "${PAPER_A_PYTHON:-}" ]]; then
  python_command=("${PAPER_A_PYTHON}")
elif command -v conda >/dev/null 2>&1; then
  python_command=(conda run --no-capture-output -n quspin python)
else
  python_command=(python)
fi

from_scratch=0
if [[ "${1:-}" == "--from-scratch" ]]; then
  from_scratch=1
elif [[ -n "${1:-}" ]]; then
  echo "usage: $0 [--from-scratch]" >&2
  exit 2
fi

cd "${project_dir}"

background_command=(
  "${python_command[@]}"
  fig1fig2/explore_resonance_mode_transition.py
  --background-only
  --background-workers "${worker_count}"
)
supplement_command=(
  "${python_command[@]}"
  fig1fig2/explore_resonance_mode_transition.py
)
main_command=(
  "${python_command[@]}"
  fig1fig2/reproduce_fig1_fig2.py
  --figures both
)
convergence_command=(
  "${python_command[@]}"
  scripts/benchmark_defect_delta_m_convergence.py
)
resonance_width_command=(
  "${python_command[@]}"
  fig1fig2/plot_resonance_line_width.py
)
region_v_command=(
  "${python_command[@]}"
  fig1fig2/explore_region_v_trajectories.py
)
if [[ "${from_scratch}" -eq 1 ]]; then
  background_command+=(--recompute-background)
  supplement_command+=(--force)
  main_command+=(--recompute-core --recompute-phase)
  convergence_command+=(--force)
  resonance_width_command+=(--force)
fi

"${background_command[@]}"

"${python_command[@]}" fig1fig2/select_low_leakage_points.py

"${region_v_command[@]}"

"${supplement_command[@]}"

"${main_command[@]}"

"${resonance_width_command[@]}"

"${convergence_command[@]}"

echo "Paper-A figures and numerical manifests regenerated successfully."
