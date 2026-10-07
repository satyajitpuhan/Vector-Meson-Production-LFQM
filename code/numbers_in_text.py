#!/usr/bin/env python3
# misc numbers quoted in the paper
import numpy as np

from plotstyle import at_points, series_kinematics, xm_of, XM_FIT
from lfvm.overlap import OMEGA
from lfvm.wavefn import SPINS
from lfvm import expdata as expt
import psi2s_ratio as cr


def ratio_to_data(s, key):
    W, eps = series_kinematics(s)
    o = np.argsort(s.x)
    out = {}
    for k in SPINS:
        th = at_points(key, k, s.x[o], W[o], eps[o])
        ok = xm_of(key, s.x[o], W[o]) < XM_FIT
        out[k] = (s.x[o][ok], s.y[o][ok] / th[ok])
    return out


def main():
    print('sigma(Q^2): data/theory, range and mean in the lowest and highest third of Q^2')
    for key in ('rho', 'phi'):
        for s in expt.sigma_Q2(key):
            rr = ratio_to_data(s, key)
            line = f'  {key:3s} {s.label:10s} N = {len(s):3d}'
            for k in SPINS:
                q, r = rr[k]
                n = len(r) // 3
                line += (f'   {k}: {r.min():.2f}-{r.max():.2f}'
                         + (f' (low {r[:n].mean():.2f}, high {r[-n:].mean():.2f})' if n else ''))
            print(line)

    print('\nE665 sigma_L and sigma_T: data/theory, range and value at the lowest Q^2')
    for s in expt.sigma_LT('rho'):
        if s.label != 'E665 1997':
            continue
        W = np.resize(np.atleast_1d(s.W), len(s))
        for k in SPINS:
            r = s.y / at_points('rho', k, s.x, W, OMEGA, s.kind)
            print(f'  {s.kind}  {k}: {r.min():.2f}-{r.max():.2f}, at Q2 = {s.x.min():.3g}: '
                  f'{r[np.argmin(s.x)]:.2f}')

    print('\nJ/psi photoproduction at W = 90 GeV (H1 2005: 73.1 +- 6.5 nb)')
    for k in SPINS:
        print(f'  {k}: {cr.xs("jpsi", k, 0.0, 90.0)["sigma_tot"]:.1f} nb')

    print('\npsi(2S)/J/psi at W = 90 GeV')
    for Q2 in (0.0, 10.0, 30.0):
        print(f'  Q2 = {Q2:4.1f}:  S-1 {cr.ratio("S1", Q2, 90.0):.3f}   S-2 {cr.ratio("S2", Q2, 90.0):.3f}')


if __name__ == '__main__':
    main()
