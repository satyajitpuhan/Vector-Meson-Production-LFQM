#!/usr/bin/env python3
# compare our dipole fit with the published ones
import numpy as np
from scipy.optimize import least_squares
from scipy.special import k0e, k1e

import fit_dipole_F2 as sf


def hera(masses, q2max=sf.Q2_MAX):
    Q2, x, y, sr, e = sf.load_hera(q2max).T
    return sf.DataSet('HERA', Q2, x, sr, e, masses, 'sr', y)


def published_fits():
    print('1. published IIM fits on the 524 HERA points')
    cases = [('fit A  m_ud = 0.046, m_s = 0.357, m_c = 1.27', dict(u=.046, d=.046, s=.357, c=1.27),
              (26.3, .741, .219, 1.81e-5), '535'),
             ('fit B  m_ud = 0.046, m_s = 0.14,  m_c = 1.27', dict(u=.046, d=.046, s=.14, c=1.27),
              (24.9, .722, .222, 1.80e-5), '529'),
             ('fit C  m_ud = m_s = 0.14,        m_c = 1.27', dict(u=.14, d=.14, s=.14, c=1.27),
              (29.9, .724, .206, 6.33e-6), '554'),
             ('Rezaeian-Schmidt, m_l = 0.001,   m_c = 1.27', dict(u=1e-3, d=1e-3, s=1e-3, c=1.27),
              (21.85, .762, .232, 6.226e-5), '562')]
    for name, m, p, pub in cases:
        h = hera(m)
        print(f'   {name}:  chi2 = {h.chi2(sf.params(*p)):7.1f}   (published {pub})')
        if name.startswith('fit C'):
            # fit C is quoted to three digits; minimise inside that rounding
            lo = [29.85, 0.7235, 0.2055, np.log(6.325e-6)]
            hi = [29.95, 0.7245, 0.2065, np.log(6.335e-6)]
            r = least_squares(lambda v: (h.theory(sf.params(v[0], v[1], v[2], np.exp(v[3]))) - h.val) / h.err,
                              [29.9, .724, .206, np.log(6.33e-6)], bounds=(lo, hi))
            print(f'      lowest chi2 within the rounding of the quoted parameters: {np.sum(r.fun ** 2):.1f}')


def iim_matching():
    print('\n2. IIM amplitude at r Q_s = 2: value and slope from both sides')
    for gs in (0.63, 0.7404):
        for lam, x in ((0.16, 1e-4), (0.22, 1e-6)):
            x0, h = 1e-5, 1e-6
            Qs = (x0 / x) ** (lam / 2)
            r = np.array([2 / Qs - h / Qs, 2 / Qs, 2 / Qs + h / Qs])
            N = sf.n_dipole(np.array([x]), r, gs, lam, x0, 0.7, 9.9)[0]
            print(f'   gamma_s = {gs}, x = {x:g}:  N = {N[0]:.6f} | {N[2]:.6f}   '
                  f'slope = {(N[1] - N[0]) / h:.5f} | {(N[2] - N[1]) / h:.5f}')


def photon_overlaps():
    print('\n3. photon overlaps: paper convention against Kowalski-Motyka-Watt')
    Nc, a, ef, m, Q2, z, r = 3.0, sf.ALPHA_EM, 2 / 3, 0.266, 5.0, 0.3, 1.7
    e2 = 4 * np.pi * a
    eps = np.sqrt(z * (1 - z) * Q2 + m * m)
    K0, K1 = k0e(eps * r) * np.exp(-eps * r), k1e(eps * r) * np.exp(-eps * r)
    # Eq. (13) of the paper summed over helicities
    L_paper = 2 * (Nc / (4 * np.pi)) * e2 * ef ** 2 * (2 * z * (1 - z)) ** 2 * Q2 * K0 ** 2 / (2 * np.pi) ** 2
    T_paper = (Nc / (2 * np.pi)) * e2 * ef ** 2 * ((z ** 2 + (1 - z) ** 2) * eps ** 2 * K1 ** 2
                                                   + m * m * K0 ** 2) / (2 * np.pi) ** 2
    # KMW Eqs. (15), (16), divided by 4 pi for the measure
    L_kmw = 8 * Nc / np.pi * a * ef ** 2 * Q2 * z ** 2 * (1 - z) ** 2 * K0 ** 2 / (4 * np.pi)
    T_kmw = 2 * Nc / np.pi * a * ef ** 2 * ((z ** 2 + (1 - z) ** 2) * eps ** 2 * K1 ** 2
                                            + m * m * K0 ** 2) / (4 * np.pi)
    print(f'   longitudinal ratio {L_paper / L_kmw:.12f}   transverse ratio {T_paper / T_kmw:.12f}')


def gbw_comparison():
    print('\n4. GBW model, HERA points with Q^2 <= 10 GeV^2 (Golec-Biernat and Sapeta, Table 1)')

    def theory(h, masses, s0, lam, x0):
        sT = np.zeros_like(h.Q2); sL = np.zeros_like(h.Q2)
        for f, m in masses.items():
            xf = h.x * (1 + 4 * m * m / h.Q2)
            Qs2 = (x0 / xf) ** lam
            S = s0 * sf.MB_TO_GEV2 * (1 - np.exp(-np.outer(Qs2, h.r ** 2) / 4))
            S[xf > 0.1] = 0.0
            wT, wL = h.w[f]
            sT += np.sum(wT * S, axis=1); sL += np.sum(wL * S, axis=1)
        pref = h.Q2 / (4 * np.pi ** 2 * sf.ALPHA_EM)
        return pref * (sT + sL) - h.y ** 2 / (1 + (1 - h.y) ** 2) * pref * sL

    cases = [('light quarks, m_l = 0.14    (their fit 0: 23.58, 0.270, 2.24e-4, 1.83)',
              dict(u=.14, d=.14, s=.14)),
             ('m_l = 0.14, m_c = 1.4      (their fit 1: 27.32, 0.248, 0.42e-4, 1.60)',
              dict(u=.14, d=.14, s=.14, c=1.4)),
             ('m_l = 0.28, m_c = 1.4      (they quote chi2/Ndof = 2.78)',
              dict(u=.28, d=.28, s=.28, c=1.4)),
             ('LFQM masses 0.266, 0.497, 1.606', sf.LFQM_MASSES)]
    for name, m in cases:
        h = hera(m, q2max=10.0)
        r = least_squares(lambda v: (theory(h, m, v[0], v[1], np.exp(v[2])) - h.val) / h.err,
                          [27.3, 0.25, np.log(0.4e-4)], x_scale='jac', xtol=1e-12, ftol=1e-12)
        n = len(h.val) - 3
        print(f'   {name}\n      ours: sigma0 = {r.x[0]:.2f} mb, lambda = {r.x[1]:.3f}, '
              f'x0 = {np.exp(r.x[2]):.3e}, chi2/dof = {np.sum(r.fun ** 2):.0f}/{n} = {np.sum(r.fun ** 2) / n:.2f}')


def charm_only_shift():
    print('\n5. x shifted for charm only instead of every flavour (our masses, all fitted data)')
    ds = sf.datasets()
    p0 = sf.params(58.9, 0.740, 0.161, 1.16e-7)
    q, e, c2, n = sf.fit(ds, sf.FITTED, p0)
    print(f'   every flavour: chi2/dof = {c2:.1f}/{n} = {c2 / n:.3f}   sigma0 = {q["s0"]:.2f}, '
          f'gamma_s = {q["gs"]:.4f}, lambda = {q["lam"]:.4f}, x0 = {q["x0"]:.3e}')
    for k in sf.FITTED:
        for f in ds[k].xm:
            if f != 'c':
                ds[k].xm[f] = ds[k].x.copy()
    q, e, c2, n = sf.fit(ds, sf.FITTED, p0)
    print(f'   charm only:    chi2/dof = {c2:.1f}/{n} = {c2 / n:.3f}   sigma0 = {q["s0"]:.2f}, '
          f'gamma_s = {q["gs"]:.4f}, lambda = {q["lam"]:.4f}, x0 = {q["x0"]:.3e}')


if __name__ == '__main__':
    published_fits()
    iim_matching()
    photon_overlaps()
    gbw_comparison()
    charm_only_shift()
