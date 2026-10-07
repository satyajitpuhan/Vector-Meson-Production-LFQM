# CGC dipole cross section
from __future__ import annotations

import numpy as np

MB_TO_GEV2 = 2.56819                  # 1 mb in GeV^-2

SIGMA0 = 58.919*MB_TO_GEV2            # GeV^-2
GAMMA_S = 0.7404
LAMBDA = 0.1609
X0 = 1.1649e-7
N0 = 0.7
KAPPA0 = 9.9

_A = -N0**2*GAMMA_S**2/((1.0 - N0)**2*np.log(1.0 - N0))
_B = 0.5*(1.0 - N0)**(-(1.0 - N0)/(N0*GAMMA_S))


def saturation_scale(xm):
    return (X0/xm)**(LAMBDA/2.0)


def n_dipole(xm, r):
    Qs = saturation_scale(xm)
    rq = np.asarray(r)*Qs
    out = np.empty_like(rq, dtype=float)
    small = rq <= 2.0
    rqs = np.where(small, np.maximum(rq, 1e-300), 1.0)
    expo = 2.0*(GAMMA_S + np.log(2.0/rqs)
                / (KAPPA0*LAMBDA*np.log(1.0/xm)))
    out[small] = (N0*(rqs[small]/2.0)**expo[small])
    big = ~small
    rqb = np.where(big, rq, 1.0)
    out[big] = 1.0 - np.exp(-_A*np.log(_B*rqb[big])**2)
    return out


def sigma_hat(xm, r):
    return SIGMA0*n_dipole(xm, r)


def slope_BD(Q2, MV, N=0.55):
    return N*(14.0*(1.0/(Q2 + MV*MV))**0.2 + 1.0)
