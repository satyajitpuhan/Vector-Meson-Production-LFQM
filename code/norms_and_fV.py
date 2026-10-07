#!/usr/bin/env python3
# normalisation checks and decay constants, S-1 vs S-2
import numpy as np

from lfvm.wavefn import LABEL, MESONS, ORDER, SPINS, build, grid_for, phi_rad
from lfvm.overlap import decay_constants

# |f_V| from Gamma(V -> e+ e-) = 4 pi alpha^2 f_V^2/(3 M_V) with the PDG 2024 widths,
# rho 7.04(6) keV and phi 1.27(4) keV
F_EXP = {'rho': (221.2, 1.0), 'phi': (228.6, 3.6)}


def norm_radial(key, grid):
    p = MESONS[key]
    X, K = grid.x[:, None], grid.k[None, :]
    Phi, _ = phi_rad(X, K, p['m'], p['beta'])
    w = (grid.wx[:, None] * (2.0 * np.pi * grid.k * grid.wk)[None, :]
         / (16.0 * np.pi ** 3))
    return float(np.sum(w * Phi ** 2))


def main():
    print('radial normalisation   int dx d^2k/(16 pi^3) |Phi|^2')
    for key in ORDER:
        g = grid_for(key)
        print(f'  {key:4s}  {norm_radial(key, g):.10f}')

    print('\nspin normalisation     n_L, n_T   (4 pi = '
          f'{4.0 * np.pi:.6f} for S-1, exactly)')
    for key in ORDER:
        g = grid_for(key)
        for kind in SPINS:
            wf = build(g, key, kind)
            print(f'  {key:4s} {LABEL[kind]:4s}  '
                  f'n_L = {wf["nL"]:.6f}   n_T = {wf["nT"]:.6f}')

    print('\ndecay constants |f_par| [MeV]')
    print(f'  {"":4s}  {LABEL["S1"]:>8s}  {LABEL["S2"]:>8s}  '
          f'{"Gamma_ee":>10s}')
    for key in ORDER:
        g = grid_for(key)
        vals = [1000.0 * abs(decay_constants(g, key, k)[0]) for k in SPINS]
        exp, err = F_EXP[key]
        print(f'  {key:4s}  {vals[0]:8.1f}  {vals[1]:8.1f}  '
              f'{exp:7.1f}({err:.1f})   '
              f'[{100 * (vals[0] - exp) / exp:+5.1f}%, '
              f'{100 * (vals[1] - exp) / exp:+5.1f}%]')


if __name__ == '__main__':
    main()
