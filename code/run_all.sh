#!/usr/bin/env bash
# runs everything in order, logs go to data/results/
# the two form factor steps take ~20 min, the rest a few minutes
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ../data/results ../figures

run() {
    echo "== $1"
    python3 "$1" | tee "../data/results/${1%.py}.log"
}

run rhophi_params.py       # m_q, beta for rho and phi
run charmonium_params.py   # m_c, beta, gamma for J/psi and psi(2S)
run fit_dipole_F2.py       # dipole fit -> dipole_fit.json
run dipole_checks.py
run norms_and_fV.py        # normalisation + decay constants
run ff_rho_phi.py          # -> ff_rho_phi.npz
run ff_charmonium.py       # -> ff_charmonium.npz
run psi2s_ratio.py
run numbers_in_text.py
run plot_main.py
run plot_charm.py
run plot_F2fit.py
