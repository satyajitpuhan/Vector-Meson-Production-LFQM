# LF wave functions (S-1 and S-2) and meson parameters
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import j0, j1, jn

HBARC = 0.1973269804
ALPHA_EM = 1.0 / 137.035999
NC = 3.0
GEV2_TO_NB = 3.893793e5

# meson parameters; ef is the effective quark charge of the meson,
# (u ubar - d dbar)/sqrt2 for the rho, s sbar for the phi and c cbar for charmonium
MESONS = {
    # rho and phi: Choi-Ji potential, m_q from the physical mass (rhophi_params.py)
    'rho': dict(name=r'\rho', tex=r'\rho^{0}', plain='rho',
                m=0.2660, beta=0.3226, M=0.77526, ef=1.0 / np.sqrt(2.0)),
    'phi': dict(name=r'\phi', tex=r'\phi', plain='phi',
                m=0.4966, beta=0.3718, M=1.019461, ef=1.0 / 3.0),
    # J/psi and psi(2S): screened potential, m_c from the J/psi mass; the psi(2S) has the
    # same m_c and beta, the 2S parameter gamma and its mass in the same Hamiltonian
    # (charmonium_params.py)
    'jpsi': dict(name=r'J/\psi', tex=r'J/\psi', plain='J/psi',
                 m=1.6061, beta=0.6602, M=3.0969, ef=2.0 / 3.0, n=1),
    'psi2s': dict(name=r'\psi(2S)', tex=r'\psi(2S)', plain='psi(2S)',
                  m=1.6061, beta=0.6602, M=3.6483, ef=2.0 / 3.0, n=2,
                  gamma=0.7317653),
}
CHARM = ['jpsi', 'psi2s']
ORDER = ['rho', 'phi']

# the two spin wave functions: internal key -> label used in figures and tables
SPINS = ('S1', 'S2')
LABEL = {'S1': 'S-1', 'S2': 'S-2'}
STYLE = {'S1': '-', 'S2': '--'}
LONG = {'S1': 'Melosh--Wigner rotation',
        'S2': 'spin-improved ansatz'}


class Grid:

    def __init__(self, nx=160, nk=320, nr=220, beta=0.32,
                 kmax_over_beta=24.0, rmax=28.0):
        gx, wx = leggauss(nx)
        self.x, self.wx = 0.5 * (gx + 1.0), 0.5 * wx
        kmax = kmax_over_beta * beta
        gk, wk = leggauss(nk)
        self.k, self.wk = 0.5 * kmax * (gk + 1.0), 0.5 * kmax * wk
        gr, wr = leggauss(nr)
        u = 0.5 * (gr + 1.0)
        self.r = rmax * u ** 2                    # quadratic: dense at small r
        self.wr = rmax * 2.0 * u * (0.5 * wr)


def kinematics(X, K, m):
    xb = X * (1.0 - X)
    M0 = np.sqrt((K * K + m * m) / xb)
    return dict(xb=xb, M0=M0, D=M0 + 2.0 * m, R0=1.0 / np.sqrt(m * m + K * K),
                M1=X * M0 + m, M2=(1.0 - X) * M0 + m, jac=M0 / (4.0 * xb))


def phi_rad(X, K, m, beta, n=1, gamma=0.5):
    kin = kinematics(X, K, m)
    kz = (2.0 * X - 1.0) * kin['M0'] / 2.0
    kv2 = K * K + kz * kz
    if n == 1:
        Phi = (4.0 * np.pi ** 0.75 / beta ** 1.5) * np.exp(-kv2 / (2.0 * beta ** 2))
    elif n == 2:
        g = gamma
        N2 = np.sqrt(96.0 * np.sqrt(2.0) * g ** 3.5 / (20.0 * g * g - 4.0 * g + 5.0))
        Phi = ((4.0 * np.pi ** 0.75 / beta ** 1.5) * N2
               * (3.0 * beta ** 2 - (1.0 + 2.0 * g) * kv2) / (3.0 * beta ** 2)
               * np.exp(-g * kv2 / beta ** 2))
    else:
        raise ValueError(n)
    return np.sqrt(kin['jac']) * Phi, kin


def radial(p, X, K):
    return phi_rad(X, K, p['m'], p['beta'], p.get('n', 1), p.get('gamma', 0.5))


def spin_harmonics(kind, kin, X, K, m, MV):
    if kind == 'S1':
        R0, D = kin['R0'], kin['D']
        return dict(L_updn=R0 * (m + 2.0 * K * K / D) / np.sqrt(2.0),
                    T_updn=R0 * K * kin['M1'] / D,
                    T_dnup=-R0 * K * kin['M2'] / D,
                    T_upup=R0 * (m + K * K / D))
    if kind == 'S2':
        xb = kin['xb']
        return dict(L_updn=0.5 * (1.0 + (m * m + K * K) / (xb * MV * MV)),
                    T_updn=X * K / (2.0 * xb),
                    T_dnup=-(1.0 - X) * K / (2.0 * xb),
                    T_upup=m / (2.0 * xb))
    raise ValueError(f'unknown spin wave function {kind!r}')


def hankel(grid, f, order):
    arg = np.outer(grid.k, grid.r)
    J = {0: j0, 1: j1}.get(order, lambda a: jn(order, a))(arg)
    return np.einsum('xk,k,kr->xr', f, grid.k * grid.wk, J)


def build(grid, key, kind):
    p = MESONS[key]
    m, beta, MV = p['m'], p['beta'], p['M']
    X, K = grid.x[:, None], grid.k[None, :]
    Phi, kin = radial(p, X, K)
    c = spin_harmonics(kind, kin, X, K, m, MV)

    wk = (grid.wx[:, None] * (2.0 * np.pi * grid.k * grid.wk)[None, :]
          / (2.0 * np.pi) ** 2)
    if kind == 'S1':
        R0, D = kin['R0'], kin['D']
        extraL = 2.0 * (R0 * K * (kin['M2'] - kin['M1']) / D) ** 2 / 2.0
        extraT = (R0 * K * K / D) ** 2
    else:
        extraL = extraT = 0.0
    nL = np.sum(wk * Phi ** 2 * (2.0 * c['L_updn'] ** 2 + extraL))
    nT = np.sum(wk * Phi ** 2 * (c['T_updn'] ** 2 + c['T_dnup'] ** 2
                                 + c['T_upup'] ** 2 + extraT))
    f = 1.0 / (2.0 * np.pi)
    return dict(p=p, key=key, kind=kind, kin=kin, Phi=Phi, c=c, nL=nL, nT=nT,
                HL=f * hankel(grid, Phi * c['L_updn'], 0) / np.sqrt(nL),
                H1a=f * hankel(grid, Phi * c['T_updn'], 1) / np.sqrt(nT),
                H1b=f * hankel(grid, -Phi * c['T_dnup'], 1) / np.sqrt(nT),
                H0c=f * hankel(grid, Phi * c['T_upup'], 0) / np.sqrt(nT))


def grid_for(key, **kw):
    p = MESONS[key]
    rmax = kw.pop('rmax', 28.0 * 0.32 / p['beta'])
    return Grid(beta=p['beta'], rmax=rmax, **kw)
