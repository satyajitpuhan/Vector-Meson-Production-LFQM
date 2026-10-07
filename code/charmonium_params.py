#!/usr/bin/env python3
# m_c, beta and gamma for J/psi and psi(2S), screened potential
import numpy as np
from scipy import integrate
from scipy.optimize import brentq
from scipy.special import k1e, erfcx, spherical_jn

A, B, C, AS = -0.411, 0.18, 0.027, 0.402      # potential parameters
SS_V, SS_P = 0.25, -0.75                      # <S_q . S_qbar> for vector, pseudoscalar
A2, B2 = 1.88684, 1.54943                     # 2S coefficients of Ridwan et al.
M_JPSI = 3.096900                             # PDG, GeV


# ---- 1S state: analytic expectation values for the Gaussian
def H0(m, b):
    z = m * m / b ** 2
    return 2 * b / np.sqrt(np.pi) * z * k1e(z / 2)


def V0(b):
    u = C / (2 * b)
    e_cr = (1 + 2 * u * u) * erfcx(u) - 2 * u / np.sqrt(np.pi)       # <exp(-c r)>
    return A + B / C * (1 - e_cr) - 8 * AS * b / (3 * np.sqrt(np.pi))


def Vhyp(m, b, ss):
    return 32 * AS * b ** 3 * ss / (9 * np.sqrt(np.pi) * m * m)


def beta_var(m):
    h = 1e-5
    f = lambda b: (H0(m, b + h) + V0(b + h) - H0(m, b - h) - V0(b - h)) / (2 * h)
    return brentq(f, 0.2, 2.5, xtol=1e-12)


def mass(m, ss=SS_V):
    b = beta_var(m)
    return H0(m, b) + V0(b) + Vhyp(m, b, ss), b


# ---- 2S state: numerical expectation values
def gamma_2S():
    return (3 * B2 / A2 - 1) / 2


def mass_2S(m, beta, gamma, ss=SS_V):
    shape = lambda k: (3 * beta ** 2 - (1 + 2 * gamma) * k ** 2) * np.exp(-gamma * k ** 2 / beta ** 2)
    norm = np.sqrt(integrate.quad(lambda k: 4 * np.pi * k ** 2 * shape(k) ** 2, 0, np.inf)[0])
    phi = lambda k: shape(k) / norm
    h0 = integrate.quad(lambda k: 4 * np.pi * k ** 2 * phi(k) ** 2 * 2 * np.sqrt(m * m + k * k),
                        0, np.inf, limit=400)[0]
    # position-space wave function from the spherical Bessel transform
    k = np.linspace(1e-6, 30 * beta, 8000)
    pk = phi(k)
    psi = lambda r: np.sqrt(2 / np.pi) * np.trapz(k ** 2 * spherical_jn(0, k * r) * pk, k)
    r = np.concatenate([[1e-8], np.logspace(-3, np.log10(80.0), 3000)])
    w = 4 * np.pi * r ** 2 * np.array([psi(x) for x in r]) ** 2
    vconf = np.trapz(w * (A + B * (1 - np.exp(-C * r)) / C), r)
    vcoul = np.trapz(w * (-4 * AS / (3 * r)), r)
    vhyp = ss * 32 * np.pi * AS / (9 * m * m) * psi(0.0) ** 2
    return h0 + vconf + vcoul + vhyp


if __name__ == '__main__':
    m = brentq(lambda m: mass(m)[0] - M_JPSI, 1.0, 2.5, xtol=1e-12)
    M, b = mass(m)
    g = gamma_2S()
    print(f'J/psi  : m_c = {m:.4f} GeV, beta = {b:.4f} GeV, M = {M:.4f} GeV '
          f'(eta_c: {mass(m, SS_P)[0]:.4f} GeV, PDG 2.9841)')
    print(f'psi(2S): gamma = {g:.7f}, M = {mass_2S(m, b, g):.4f} GeV (PDG 3.6861)')
