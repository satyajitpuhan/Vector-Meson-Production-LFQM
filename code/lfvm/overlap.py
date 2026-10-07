# overlaps, amplitudes, cross sections, decay constants
from __future__ import annotations

import numpy as np
from scipy.special import k0, k1

from .wavefn import ALPHA_EM, GEV2_TO_NB, HBARC, MESONS, NC
from .cgc_dipole import sigma_hat, slope_BD

E_CHARGE = np.sqrt(4.0 * np.pi * ALPHA_EM)

# The real-part correction tan(pi alpha/2) is meaningful only for alpha < 1.  alpha stays
# below 0.7 at every data point used here (the largest value, 0.66, is at the NMC point with
# x_m = 0.15) and grows quickly only for x_m > 0.17.  Above ALPHA_MAX the cross section is
# returned as NaN, so curves stop there instead of rising steeply.
ALPHA_MAX = 0.80

# photon polarisation parameter eps of the HERA measurements, sigma = sigma_T + eps sigma_L;
# the fixed-target data carry their own eps
OMEGA = 0.98


def im_amplitude(grid, wf, Q2, xm):
    p = wf['p']
    m, ef = p['m'], p['ef']
    X = grid.x[:, None]
    xb = X * (1.0 - X)
    R = grid.r[None, :]
    eps = np.sqrt(xb * Q2 + m * m)
    K0e, K1e = k0(eps * R), k1(eps * R)
    sh = sigma_hat(xm, grid.r)[None, :]
    w = grid.wx[:, None] * (2.0 * np.pi * grid.r * grid.wr)[None, :]

    preL = (2.0 * np.sqrt(NC / (4.0 * np.pi)) * E_CHARGE * ef
            * 2.0 * np.sqrt(Q2) / (2.0 * np.pi))
    AL = float(np.sum(w * (preL * xb * K0e * wf['HL']) * sh))

    preT = np.sqrt(NC / (2.0 * np.pi)) * E_CHARGE * ef / (2.0 * np.pi)
    intT = preT * (eps * K1e * (X * wf['H1a'] + (1.0 - X) * wf['H1b'])
                   + m * K0e * wf['H0c'])
    AT = float(np.sum(w * intT * sh))
    return AL, AT


def cross_sections(grid, wf, Q2, W, dlog=0.05, eps_pol=OMEGA):
    p = wf['p']
    MV = p['M']
    xm = (Q2 + MV * MV) / (W * W)
    AL, AT = im_amplitude(grid, wf, Q2, xm)
    ALp, ATp = im_amplitude(grid, wf, Q2, xm * np.exp(-dlog))
    ALm, ATm = im_amplitude(grid, wf, Q2, xm * np.exp(dlog))
    BD = slope_BD(Q2, MV)
    out = {}
    for tag, A, Ap, Am in (('L', AL, ALp, ALm), ('T', AT, ATp, ATm)):
        if not (abs(A) > 0 and abs(Ap) > 0 and abs(Am) > 0):
            out['sigma_' + tag] = 0.0      # Q^2 = 0: the longitudinal vanishes
            continue
        # real-part correction from the effective power of 1/x_m
        alpha = (np.log(abs(Ap)) - np.log(abs(Am))) / (2.0 * dlog)
        if not np.isfinite(alpha) or alpha >= ALPHA_MAX:
            out['sigma_' + tag] = np.nan       # see ALPHA_MAX
            continue
        beta = np.tan(np.pi * alpha / 2.0)
        out['sigma_' + tag] = (A * A * (1.0 + beta * beta)
                               / (16.0 * np.pi * BD) * GEV2_TO_NB)
    out['sigma_tot'] = out['sigma_T'] + eps_pol * out['sigma_L']
    out['R'] = (out['sigma_L'] / out['sigma_T']
                if out['sigma_T'] else np.nan)
    out['BD'] = BD
    out['xm'] = xm
    return out


def dsigma_dt(grid, wf, Q2, W, t, eps_pol=OMEGA):
    cs = cross_sections(grid, wf, Q2, W, eps_pol=eps_pol)
    BD = cs['BD']
    return cs['sigma_tot'] * BD * np.exp(-BD * np.asarray(t))


def decay_constants(grid, key, kind):
    from .wavefn import radial, spin_harmonics
    p = MESONS[key]
    m, beta, MV = p['m'], p['beta'], p['M']
    X, K = grid.x[:, None], grid.k[None, :]
    Phi, kin = radial(p, X, K)
    w = (grid.wx[:, None] * (2.0 * np.pi * grid.k * grid.wk)[None, :]
         / (16.0 * np.pi ** 3))
    if kind == 'S1':
        R0, D = kin['R0'], kin['D']
        OV = np.sqrt(2.0) * R0 * (2.0 * m + 4.0 * K * K / D)
        OT = np.sqrt(2.0) * R0 * (2.0 * m + 2.0 * K * K / D)
    else:
        # in terms of the harmonics OV = 4 L_updn and OT = 2 sqrt2 T_upup; S-2 is not
        # unitary, so each polarisation is divided by the square root of its norm, as in build()
        c = spin_harmonics('S2', kin, X, K, m, MV)
        PL = np.sum(w * Phi ** 2 * 2.0 * c['L_updn'] ** 2)
        PT = np.sum(w * Phi ** 2 * (c['T_updn'] ** 2 + c['T_dnup'] ** 2
                                    + c['T_upup'] ** 2))
        OV = 4.0 * c['L_updn'] / np.sqrt(PL)
        OT = 2.0 * np.sqrt(2.0) * c['T_upup'] / np.sqrt(PT)
    return (float(np.sqrt(NC) * np.sum(w * Phi * OV)),
            float(np.sqrt(NC) * np.sum(w * Phi * OT)))


def overlaps(grid, wf, Q2):
    p = wf['p']
    m, ef = p['m'], p['ef']
    X = grid.x[:, None]
    xb = X * (1.0 - X)
    R = grid.r[None, :]
    eps = np.sqrt(xb * Q2 + m * m)
    K0e, K1e = k0(eps * R), k1(eps * R)

    preL = (2.0 * np.sqrt(NC / (4.0 * np.pi)) * E_CHARGE * ef
            * 2.0 * np.sqrt(Q2) / (2.0 * np.pi))
    oL = np.sum(grid.wx[:, None] * preL * xb * K0e * wf['HL'], axis=0)
    preT = np.sqrt(NC / (2.0 * np.pi)) * E_CHARGE * ef / (2.0 * np.pi)
    oT = np.sum(grid.wx[:, None] * preT
                * (eps * K1e * (X * wf['H1a'] + (1.0 - X) * wf['H1b'])
                   + m * K0e * wf['H0c']), axis=0)
    r = grid.r * HBARC
    return r, (r / 2.0) * oT, (r / 2.0) * oL


def densities(grid, wf):
    return 2.0 * wf['HL'] ** 2, wf['H1a'] ** 2 + wf['H1b'] ** 2 + wf['H0c'] ** 2
