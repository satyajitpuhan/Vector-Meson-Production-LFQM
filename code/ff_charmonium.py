#!/usr/bin/env python3
# J/psi and psi(2S) form factors, chi2 vs lattice
import os

import numpy as np
from ff_rho_phi import Meson
from lfvm.wavefn import HBARC

HERE = os.path.dirname(os.path.abspath(__file__))

PRES = ('BH', 'GK', 'no+0', 'no+-', 'LSQ')


def lattice():
    rows = [l.strip().split(',') for l in open(os.path.join(HERE, '..', 'data', 'lattice_jpsi_ff.csv'))
            if not l.startswith('#') and not l.startswith('source')]
    return [(r[0], r[1], float(r[2]), float(r[3]), float(r[4])) for r in rows]


def main():
    Q2 = np.unique(np.concatenate([[1e-4, 2e-3, 4e-3, 6e-3, 8e-3],
                                   np.geomspace(0.01, 100.0, 50),
                                   np.linspace(0.1, 8.0, 80)]))
    out = {'Q2': Q2}
    lat = lattice()
    for key in ('jpsi', 'psi2s'):
        for kind in ('S1', 'S2'):
            M = Meson(key, kind)
            I0 = M.amplitudes(1e-4)[3].real
            G = {n: [] for n in PRES + ('I', 'Delta')}
            for q2 in Q2:
                r = M.form_factors(np.sqrt(q2), I0)
                for n in G:
                    G[n].append(r[n])
            for n in G:
                out[f'{key}_{kind}_{n}'] = np.array(G[n])
            # static properties from a quadratic fit to the lowest Q^2 points
            sel = Q2 <= 0.0101
            line = f'{key} {kind}:'
            for n in ('GK', 'BH'):
                g = out[f'{key}_{kind}_{n}']
                c = np.polyfit(Q2[sel], g[sel, 0], 2)
                r2 = -6 * c[1] * HBARC ** 2
                line += (f'  [{n}] r_C={np.sqrt(r2):.4f} fm  G_M(0)={g[0,1]:.4f}'
                         f'  G_Q(0)={g[0,2]:.4f}  Q={g[0,2]/M.MV**2*HBARC**2:.5f} fm^2')
                out[f'{key}_{kind}_{n}_static'] = np.array([np.sqrt(r2), g[0, 1], g[0, 2]])
            print(line)
            if key == 'jpsi':
                # chi^2 of every prescription against the lattice points
                c2 = {n: {'Dudek06': [0., 0], 'Delaney23': [0., 0]} for n in PRES}
                for src, ff, q, g, e in lat:
                    if q < 1e-3:
                        continue                # G_C(0) = 1 by construction
                    r = M.form_factors(np.sqrt(q), I0)
                    i = ('GC', 'GM', 'GQ').index(ff)
                    for n in PRES:
                        c2[n][src][0] += ((r[n][i] - g) / e) ** 2
                        c2[n][src][1] += 1
                for n in PRES:
                    a, b = c2[n]['Dudek06'], c2[n]['Delaney23']
                    print(f'   lattice chi2/N  {n:5s}: Dudek06 {a[0]/a[1]:8.2f} ({a[1]})'
                          f'  Delaney23 {b[0]/b[1]:8.2f} ({b[1]})'
                          f'  all {(a[0]+b[0])/(a[1]+b[1]):8.2f}')
                    out[f'jpsi_{kind}_chi2_{n}'] = np.array([a[0]/a[1], b[0]/b[1], (a[0]+b[0])/(a[1]+b[1])])
    os.makedirs(os.path.join(HERE, '..', 'data', 'results'), exist_ok=True)
    np.savez(os.path.join(HERE, '..', 'data', 'results', 'ff_charmonium.npz'), **out)


if __name__ == '__main__':
    main()
