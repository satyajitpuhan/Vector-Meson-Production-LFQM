#!/usr/bin/env python3
# m_q and beta for rho and phi from the Choi-Ji HO Hamiltonian
import numpy as np
from scipy.special import k1e
from scipy.optimize import brentq, minimize_scalar

A, B, KAPPA = -0.144, 0.010329, 0.607          # GeV, GeV^3, dimensionless (Choi and Ji)
SS_V, SS_P = 0.25, -0.75


def H0(m, beta):
    z = m * m / beta ** 2
    return 2 * beta / np.sqrt(np.pi) * z * k1e(z / 2)   # e^{z/2} K1(z/2) = k1e(z/2)


def V0(beta):
    return A + 1.5 * B / beta ** 2 - 8 * KAPPA * beta / (3 * np.sqrt(np.pi))


def Vhyp(m, beta, ss):
    return 32 * KAPPA * beta ** 3 * ss / (9 * np.sqrt(np.pi) * m * m)


def beta_var(m):
    f = lambda b: (H0(m, b * 1.0001) + V0(b * 1.0001) - H0(m, b * 0.9999) - V0(b * 0.9999))
    return brentq(f, 0.1, 1.5, xtol=1e-12)


def mass(m, ss=SS_V):
    b = beta_var(m)
    return H0(m, b) + V0(b) + Vhyp(m, b, ss), b


def m_for(target, ss=SS_V):
    m_min = minimize_scalar(lambda m: mass(m, ss)[0], bounds=(0.1, 0.6), method='bounded').x
    return brentq(lambda m: mass(m, ss)[0] - target, m_min, 1.5, xtol=1e-12)


if __name__ == '__main__':
    for name, target in (('rho', 0.77526), ('phi', 1.019461)):     # PDG masses in GeV
        m = m_for(target)
        M, b = mass(m)
        print(f'{name}: m_q = {m:.4f} GeV, beta = {b:.4f} GeV, M_V = {1000 * M:.2f} MeV '
              f'(pseudoscalar partner {1000 * mass(m, SS_P)[0]:.1f} MeV)')
