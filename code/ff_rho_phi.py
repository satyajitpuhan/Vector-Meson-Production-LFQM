#!/usr/bin/env python3
# rho and phi e.m. form factors (G_C, G_M, G_Q)
import os

import numpy as np
from lfvm.wavefn import MESONS, radial, HBARC

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'results')

# ---------------------------------------------------------------- Dirac algebra
I2, Z2 = np.eye(2), np.zeros((2, 2))
SIG = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]),
       np.array([[1, 0], [0, -1]], complex)]
G0 = np.block([[I2, Z2], [Z2, -I2]]).astype(complex)
GI = [np.block([[Z2, s], [-s, Z2]]) for s in SIG]
AP = [G0 @ GI[0], G0 @ GI[1]]
CHI = {+1: np.array([1, 0, 1, 0], complex) / np.sqrt(2),
       -1: np.array([0, 1, 0, -1], complex) / np.sqrt(2)}
HEL = (+1, -1)


def spinor(pp, px, py, m, lam, anti=False):
    op = (pp[..., None, None] * np.eye(4) + (-m if anti else m) * G0
          + px[..., None, None] * AP[0] + py[..., None, None] * AP[1])
    chi = CHI[-lam] if anti else CHI[lam]
    return (op @ chi) / np.sqrt(pp)[..., None]


def bar(u):
    return np.conj(u) @ G0


def slash(e0, e1, e2, e3):
    return (e0[..., None, None] * G0 - e1[..., None, None] * GI[0]
            - e2[..., None, None] * GI[1] - e3[..., None, None] * GI[2])


def check_spinors():
    rng = np.random.default_rng(3)
    m = 0.25
    px, py, pp = rng.normal(size=5), rng.normal(size=5), rng.uniform(0.3, 2, 5)
    pm = (px**2 + py**2 + m * m) / pp
    ps = slash((pp + pm) / 2, px, py, (pp - pm) / 2)
    err = 0.0
    for lam in HEL:
        u, v = spinor(pp, px, py, m, lam), spinor(pp, px, py, m, lam, anti=True)
        err = max(err, np.abs(np.einsum('nij,nj->ni', ps, u) - m * u).max(),
                  np.abs(np.einsum('nij,nj->ni', ps, v) + m * v).max(),
                  np.abs(np.einsum('ni,ni->n', bar(u), u) - 2 * m).max(),
                  np.abs(np.einsum('ni,ni->n', bar(v), v) + 2 * m).max())
    return err


# -------------------------------------------------------- spin wave functions
def spin_S1(x, kx, ky, m):
    k2 = kx**2 + ky**2
    M0 = np.sqrt((k2 + m * m) / (x * (1 - x)))
    p1p, p2p = x, 1 - x
    k1 = dict(p=p1p, x=kx, y=ky, m=(k2 + m * m) / p1p)
    k2v = dict(p=p2p, x=-kx, y=-ky, m=(k2 + m * m) / p2p)
    dvec = [(k1['p'] + k1['m'] - k2v['p'] - k2v['m']) / 2, k1['x'] - k2v['x'],
            k1['y'] - k2v['y'], (k1['p'] - k1['m'] - k2v['p'] + k2v['m']) / 2]
    out = np.zeros(x.shape + (3, 2, 2), complex)
    for il, L in enumerate((+1, 0, -1)):
        if L == 0:      # eps(0) = (eps+ = P+/M0, eps- = -M0/P+, eps_perp = 0)
            ep, em = 1 / M0, -M0
            e = [(ep + em) / 2, 0 * x, 0 * x, (ep - em) / 2]
        else:           # eps(+-) = (0, 0, -+(1, +-i)/sqrt2)
            e = [0 * x, -L / np.sqrt(2) + 0 * x, -1j / np.sqrt(2) + 0 * x, 0 * x]
        edot = e[0] * dvec[0] - e[1] * dvec[1] - e[2] * dvec[2] - e[3] * dvec[3]
        Gam = -slash(*e) + (edot / (M0 + 2 * m))[..., None, None] * np.eye(4)
        for ih, h in enumerate(HEL):
            u = spinor(x + 0 * kx, kx, ky, m, h)
            for jh, hb in enumerate(HEL):
                v = spinor(1 - x + 0 * kx, -kx, -ky, m, hb, anti=True)
                out[..., il, ih, jh] = (np.einsum('...i,...ij,...j->...', bar(u), Gam, v)
                                        / (np.sqrt(2) * M0))
    return out


def spin_S2(x, kx, ky, m, MV):
    xb = x * (1 - x)
    kR, kL = kx + 1j * ky, kx - 1j * ky
    out = np.zeros(x.shape + (3, 2, 2), complex)
    L0 = 0.5 * (1 + (m * m + kx**2 + ky**2) / (xb * MV * MV))
    out[..., 1, 0, 1] = L0
    out[..., 1, 1, 0] = L0
    out[..., 0, 0, 0] = m / (2 * xb)
    out[..., 0, 0, 1] = x * kR / (2 * xb)
    out[..., 0, 1, 0] = -(1 - x) * kR / (2 * xb)
    out[..., 2, 1, 1] = m / (2 * xb)
    out[..., 2, 0, 1] = (1 - x) * kL / (2 * xb)
    out[..., 2, 1, 0] = -x * kL / (2 * xb)
    return out


# ------------------------------------------------------ covariant decomposition
MET = np.diag([1., -1., -1., -1.])


def lf_pol(pp, px, M):
    out = []
    for L in (+1, 0, -1):
        if L == 0:
            ep, eperp, em = pp / M, np.array([px, 0.0]) / M, (px * px - M * M) / (M * pp)
        else:
            eperp = -L * np.array([1, L * 1j]) / np.sqrt(2)
            ep, em = 0.0, 2 * eperp[0] * px / pp
        out.append(np.array([(ep + em) / 2, eperp[0], eperp[1], (ep - em) / 2], complex))
    return out


def covariant_map(Q, M):
    p = np.array([M, 0, 0, 0.])
    pmi = (Q * Q + M * M) / M
    pf = np.array([(M + pmi) / 2, Q, 0, (M - pmi) / 2])
    q, P = pf - p, p + pf
    d = lambda a, b: a @ MET @ b
    plus = lambda v: v[0] + v[3]
    ei, ef = lf_pol(M, 0.0, M), lf_pol(M, Q, M)
    A = np.zeros((4, 3), complex)
    for r, (a, b) in enumerate([(0, 0), (0, 1), (0, 2), (1, 1)]):
        e, epc = ei[b], np.conj(ef[a])
        A[r, 0] = -d(epc, e) * plus(P)
        A[r, 1] = -(plus(e) * d(epc, q) - plus(epc) * d(e, q))
        A[r, 2] = d(e, q) * d(epc, q) * plus(P) / (2 * M * M)
        A[r] /= plus(P)
    return A


# --------------------------------------------------------------- calculation
class Meson:
    def __init__(self, key, kind, nx=90, nk=90, nph=36):
        p = MESONS[key]
        self.p = p
        self.m, self.beta, self.MV, self.kind = p['m'], p['beta'], p['M'], kind
        gx, wx = np.polynomial.legendre.leggauss(nx)
        gk, wk = np.polynomial.legendre.leggauss(nk)
        x = 0.5 * (gx + 1); wx = 0.5 * wx
        t = 0.5 * (gk + 1); kmax = 12 * self.beta
        k = kmax * t**2; wkk = kmax * 2 * t * 0.5 * wk
        ph = (np.arange(nph) + 0.5) * 2 * np.pi / nph
        X, K, PH = np.meshgrid(x, k, ph, indexing='ij')
        self.w = (wx[:, None, None] * (wkk * k)[None, :, None]
                  * np.full(nph, 2 * np.pi / nph)[None, None, :]) / (16 * np.pi**3)
        self.x, self.kx, self.ky = X, K * np.cos(PH), K * np.sin(PH)
        self.psi = self.wf(self.x, self.kx, self.ky)
        # per-helicity normalisation (S-2 is not pointwise unitary)
        self.norm = np.einsum('ijklab,ijklab,ijk->l', np.conj(self.psi), self.psi, self.w).real
        self.psi /= np.sqrt(self.norm)[None, None, None, :, None, None]

    def wf(self, x, kx, ky):
        K = np.sqrt(kx**2 + ky**2)
        Phi, _ = radial(self.p, x, K)
        S = spin_S1(x, kx, ky, self.m) if self.kind == 'S1' else spin_S2(x, kx, ky, self.m, self.MV)
        return Phi[..., None, None, None] * S

    def amplitudes(self, Q):
        psif = self.wf(self.x, self.kx + (1 - self.x) * Q, self.ky)
        psif /= np.sqrt(self.norm)[None, None, None, :, None, None]
        I = np.einsum('ijklab,ijkmab,ijk->lm', np.conj(psif), self.psi, self.w)
        return np.array([I[0, 0], I[0, 1], I[0, 2], I[1, 1]])

    def form_factors(self, Q, I0):
        b = self.amplitudes(Q) / I0
        A = covariant_map(Q, self.MV)
        eta = Q * Q / (4 * self.MV**2)
        res = {}
        # amplitudes ordered (++, +0, +-, 00); each prescription drops one of them,
        # 'LSQ' uses all four in a least-squares sense
        for name, rows in (('BH', [1, 2, 3]), ('GK', [0, 1, 2]), ('no+0', [0, 2, 3]),
                           ('no+-', [0, 1, 3]), ('LSQ', [0, 1, 2, 3])):
            G1, G2, G3 = np.linalg.lstsq(A[rows], b[rows], rcond=None)[0].real
            GQ = G1 - G2 + (1 + eta) * G3
            res[name] = (G1 + 2 / 3 * eta * GQ, G2, GQ)
        # light-front angular condition, zero for any covariant current:
        # Delta = (1+2 eta) I_{++} - sqrt(8 eta) I_{+0} + I_{+-} - I_{00}
        res['I'] = b.real
        res['Delta'] = ((1 + 2 * eta) * b[0] - np.sqrt(8 * eta) * b[1] + b[2] - b[3]).real
        return res


def main():
    print('spinor check (Dirac eq., normalisation):', f'{check_spinors():.1e}')
    x = np.array([0.3, 0.6]); kx = np.array([0.4, 0.2]); ky = np.array([0.1, -0.3])
    S = spin_S1(x, kx, ky, 0.25)
    print('S-1 unitarity sum|S|^2 per helicity:', np.round(np.sum(np.abs(S)**2, axis=(-1, -2)), 10))
    # the S-1 components must agree with the harmonics used in lfvm/wavefn.py
    from lfvm.wavefn import kinematics, spin_harmonics
    X, K = np.array([[0.3]]), np.array([[0.5]])
    c = spin_harmonics('S1', kinematics(X, K, 0.25), X, K, 0.25, 0.775)
    Sx = spin_S1(np.array([0.3]), np.array([0.5]), np.array([0.0]), 0.25)[0]
    print('S-1 |L_updn|, |T_upup|, |T_updn|, |T_dnup|  here :',
          np.round([abs(Sx[1, 0, 1]), abs(Sx[0, 0, 0]), abs(Sx[0, 0, 1]), abs(Sx[0, 1, 0])], 6))
    print('                                         wavefn.py:',
          np.round([c['L_updn'][0, 0], c['T_upup'][0, 0], c['T_updn'][0, 0], abs(c['T_dnup'][0, 0])], 6))

    Q2 = np.concatenate([[1e-4], np.geomspace(0.01, 100.0, 50)])
    out = {'Q2': Q2}
    for key in ('rho', 'phi'):
        for kind in ('S1', 'S2'):
            M = Meson(key, kind)
            I0 = M.amplitudes(1e-4)[3].real
            G = {n: [] for n in ('BH', 'GK', 'no+0', 'no+-', 'LSQ', 'I', 'Delta')}
            for q2 in Q2:
                r = M.form_factors(np.sqrt(q2), I0)
                for n in G:
                    G[n].append(r[n])
            for n in G:
                out[f'{key}_{kind}_{n}'] = np.array(G[n])
            g = out[f'{key}_{kind}_GK']        # prescription used in the paper
            MV = M.MV
            # charge radius from a quadratic fit of G_C on 0 < Q^2 <= 0.01 GeV^2
            q = np.array([1e-4, 0.002, 0.004, 0.006, 0.008, 0.01])
            gc = [M.form_factors(np.sqrt(v), I0)['GK'][0] for v in q]
            r2 = -6 * np.polyfit(q, gc, 2)[1] * HBARC**2
            print(f'{key} {kind} [GK]: G_C(0)={g[0,0]:.3f}  mu=G_M(0)={g[0,1]:.3f}  '
                  f'G_Q(0)={g[0,2]:.3f} -> Q={g[0,2]/MV**2*HBARC**2:.4f} fm^2   '
                  f'r_C={np.sqrt(r2):.3f} fm')
    os.makedirs(RESULTS, exist_ok=True)
    np.savez(os.path.join(RESULTS, 'ff_rho_phi.npz'), **out)


if __name__ == '__main__':
    main()
