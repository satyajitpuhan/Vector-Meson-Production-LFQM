#!/usr/bin/env python3
# fit the CGC dipole to the inclusive sigma_r / F2 data
import csv
import glob
import json
import os

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import least_squares
from scipy.special import k0e, k1e

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
RESULTS = os.path.join(HERE, '..', 'data', 'results')

ALPHA_EM = 1.0 / 137.035999
NC = 3.0
MB_TO_GEV2 = 2.56819
CHARGES = {'u': 2 / 3, 'd': -1 / 3, 's': -1 / 3, 'c': 2 / 3}

# LFQM quark masses of the paper (Table I): rho, phi and J/psi values
LFQM_MASSES = dict(u=0.2660, d=0.2660, s=0.4966, c=1.6061)

# kinematic range of the fit
X_MAX, Q2_MIN, Q2_MAX = 0.01, 0.045, 45.0

# data sets used in the fit
FITTED = ('HERA', 'E665', 'NMC', 'charm')


# ---------------------------------------------------------------- data
def load_hera(q2max=Q2_MAX):
    rows = []
    for f in sorted(glob.glob(os.path.join(DATA, 'hera2015', 'hera_t*.csv'))):
        lines = open(f).read().splitlines()
        sqs = float(next(l for l in lines if l.startswith('#: SQRT(S)')).split(',')[1])
        body = [r for r in csv.reader(l for l in lines if l and not l.startswith('#'))][1:]
        for r in body:
            if not (r and r[0][:1].isdigit()):            # header text wrapped over lines
                continue
            Q2, x, sr = float(r[0]), float(r[1]), float(r[2])
            if not (x <= X_MAX and Q2_MIN <= Q2 <= q2max):
                continue
            # stat, uncorrelated, correlated and the seven procedural sources, in per cent
            pct = [abs(float(r[3 + 2 * i].rstrip('%'))) for i in range(10)]
            rows.append((Q2, x, Q2 / (sqs * sqs * x), sr, sr * np.sqrt(np.sum(np.square(pct))) / 100))
    return np.array(rows)


def _hepdata_blocks(path):
    out, qual, hdr, rows = [], {}, None, []
    for line in open(path).read().splitlines():
        if line.startswith('#:'):
            if hdr is not None:
                out.append((qual, hdr, rows)); qual, hdr, rows = {}, None, []
            k, _, v = line[2:].partition(',')
            qual[k.strip()] = v.strip(',').strip()
        elif not line.strip():
            continue
        elif hdr is None and not line[:1].isdigit() and not line.startswith('-'):
            hdr = next(csv.reader([line]))
        elif hdr is not None:
            r = next(csv.reader([line]))
            if r[0][:1].isdigit() or r[0].startswith('-'):
                rows.append(r)
    if hdr is not None:
        out.append((qual, hdr, rows))
    return out


def load_fixed_target(prefix):
    pts = []
    for f in sorted(glob.glob(os.path.join(DATA, 'fixed_target_F2', f'{prefix}_t*.csv'))):
        txt = open(f).read()
        if 'P -->' not in txt.split('keyword reactions:')[1].split('\n')[0]:
            continue                                      # deuteron table
        x = float(txt.split('#: X,')[1].split('\n')[0])
        if x > X_MAX:
            continue
        for qual, hdr, rows in _hepdata_blocks(f):
            iq = hdr.index('Q**2 [GEV**2]'); i2 = hdr.index('F2')
            for r in rows:
                Q2 = float(r[iq])
                if Q2_MIN <= Q2 <= Q2_MAX:
                    e = np.hypot(float(r[i2 + 1]), float(r[i2 + 3]))
                    pts.append((Q2, x, float(r[i2]), e))
    return np.array(pts)


def load_charm():
    return np.loadtxt(os.path.join(DATA, 'hera2018_charm_sigma_r.csv'), delimiter=',', comments='#')


# ---------------------------------------------------------------- model
def overlaps(Q2, masses, nr=260, nz=200):
    lnr = np.linspace(np.log(1e-4), np.log(800.0), nr)
    r = np.exp(lnr)
    wr = np.full(nr, lnr[1] - lnr[0]); wr[[0, -1]] *= 0.5
    g, wg = leggauss(nz)
    a, b = np.log(1e-10), np.log(0.5)
    t = 0.5 * (b - a) * (g + 1) + a
    z = np.exp(t); wz = 0.5 * (b - a) * wg * z * 2.0        # factor 2 for z > 1/2
    if len(Q2) > 40:                                        # in chunks, to limit memory
        parts = [overlaps(Q2[i:i + 40], masses, nr, nz)[1] for i in range(0, len(Q2), 40)]
        return r, {f: tuple(np.concatenate([p[f][k] for p in parts]) for k in (0, 1)) for f in masses}
    out = {}
    for f, m in masses.items():
        ef2 = CHARGES[f] ** 2
        eps = np.sqrt(z[None, :, None] * (1 - z[None, :, None]) * Q2[:, None, None] + m * m)
        er = eps * r[None, None, :]
        K0 = k0e(er) * np.exp(-er); K1 = k1e(er) * np.exp(-er)
        zz = z[None, :, None]
        T = 2 * NC / np.pi * ALPHA_EM * ef2 * ((zz ** 2 + (1 - zz) ** 2) * eps ** 2 * K1 ** 2 + m * m * K0 ** 2)
        L = 8 * NC / np.pi * ALPHA_EM * ef2 * Q2[:, None, None] * (zz * (1 - zz)) ** 2 * K0 ** 2
        fac = wz[None, :, None] / (4 * np.pi)
        jac = 2 * np.pi * r ** 2 * wr                       # d^2 r = 2 pi r^2 d ln r
        out[f] = (np.sum(T * fac, axis=1) * jac, np.sum(L * fac, axis=1) * jac)
    return r, out


def n_dipole(xm, r, gs, lam, x0, N0, kap):
    A = -N0 ** 2 * gs ** 2 / ((1 - N0) ** 2 * np.log(1 - N0))
    B = 0.5 * (1 - N0) ** (-(1 - N0) / (N0 * gs))
    Qs = (x0 / xm) ** (lam / 2)
    rq = np.maximum(r[None, :] * Qs[:, None], 1e-300)
    L1x = np.log(1 / xm)[:, None]
    small = N0 * (rq / 2) ** (2 * (gs + np.log(2 / rq) / (kap * lam * L1x)))
    big = 1 - np.exp(-A * np.log(B * np.maximum(rq, 2.0)) ** 2)
    return np.where(rq <= 2, small, big)


def sigma_hat(xm, r, p):
    ok = xm < 1
    N = np.zeros((len(xm), len(r)))
    if ok.any():
        N[ok] = n_dipole(xm[ok], r, p['gs'], p['lam'], p['x0'], p['N0'], p['kap'])
    return p['s0'] * MB_TO_GEV2 * N


class DataSet:

    def __init__(self, name, Q2, x, val, err, masses, kind='sr', y=None, flavours=None):
        self.name, self.Q2, self.x, self.val, self.err, self.kind, self.y = name, Q2, x, val, err, kind, y
        use = {f: m for f, m in masses.items() if flavours is None or f in flavours}
        uq, inv = np.unique(Q2, return_inverse=True)
        self.r, w = overlaps(uq, use)
        self.w = {f: (a[inv], b[inv]) for f, (a, b) in w.items()}
        self.xm = {f: x * (1 + 4 * m * m / Q2) for f, m in use.items()}

    def theory(self, p):
        sT = 0; sL = 0
        for f, (wT, wL) in self.w.items():
            S = sigma_hat(self.xm[f], self.r, p)
            sT = sT + np.sum(wT * S, axis=1); sL = sL + np.sum(wL * S, axis=1)
        pref = self.Q2 / (4 * np.pi ** 2 * ALPHA_EM)
        F2, FL = pref * (sT + sL), pref * sL
        if self.kind == 'F2':
            return F2
        if self.kind == 'FL':
            return FL
        return F2 - self.y ** 2 / (1 + (1 - self.y) ** 2) * FL

    def chi2(self, p):
        return float(np.sum(((self.theory(p) - self.val) / self.err) ** 2))


def datasets(masses=LFQM_MASSES):
    Q2, x, y, sr, e = load_hera().T
    ds = {'HERA': DataSet('HERA 2015 sigma_r', Q2, x, sr, e, masses, 'sr', y)}
    c = load_charm()
    ds['charm'] = DataSet('HERA 2018 charm sigma_r', c[:, 0], c[:, 1], c[:, 2], c[:, 2] * c[:, 3] / 100,
                          masses, 'sr', c[:, 0] / (318.0 ** 2 * c[:, 1]), flavours='c')
    for key, prefix in (('E665', 'e665'), ('NMC', 'nmc')):
        d = load_fixed_target(prefix)
        ds[key] = DataSet(f'{key} F2', d[:, 0], d[:, 1], d[:, 2], d[:, 3], masses, 'F2')
    return ds


def params(s0, gs, lam, x0, N0=0.7, kap=9.9):
    return dict(s0=s0, gs=gs, lam=lam, x0=x0, N0=N0, kap=kap)


def fit(ds, use, p0, free=('s0', 'gs', 'lam', 'x0')):
    base = dict(p0)
    lo = {'s0': 1, 'gs': 0.2, 'lam': 0.02, 'x0': np.log(1e-30), 'N0': 0.05, 'kap': 0.5}
    hi = {'s0': 5000, 'gs': 1.5, 'lam': 1.0, 'x0': np.log(1.0), 'N0': 0.999, 'kap': 500}

    def unpack(v):
        q = dict(base)
        for n, val in zip(free, v):
            q[n] = np.exp(val) if n == 'x0' else val
        return q

    def residuals(v):
        q = unpack(v)
        return np.concatenate([(ds[k].theory(q) - ds[k].val) / ds[k].err for k in use])

    v0 = [np.log(base[n]) if n == 'x0' else base[n] for n in free]
    out = least_squares(residuals, v0, bounds=([lo[n] for n in free], [hi[n] for n in free]),
                        x_scale='jac', xtol=1e-12, ftol=1e-12)
    q = unpack(out.x)
    cov = np.linalg.pinv(out.jac.T @ out.jac)
    err = dict(zip(free, np.sqrt(np.diag(cov))))
    if 'x0' in err:
        err['x0'] *= q['x0']
    npts = sum(len(ds[k].val) for k in use)
    return q, err, float(np.sum(out.fun ** 2)), npts - len(free)


# ---------------------------------------------------------------- the adopted fit
def main():
    ds = datasets()

    def residuals(v):
        p = params(v[0], v[1], v[2], np.exp(v[3]))
        return np.concatenate([(ds[k].theory(p) - ds[k].val) / ds[k].err for k in FITTED])

    # the same minimum is reached from very different starting values
    best = None
    for start in ([58.1, 0.744, 0.165, 1.57e-7], [40, 0.70, 0.20, 1e-6], [80, 0.80, 0.14, 1e-8],
                  [26.3, 0.741, 0.219, 1.81e-5]):
        v0 = [start[0], start[1], start[2], np.log(start[3])]
        r = least_squares(residuals, v0, bounds=([1, 0.2, 0.02, np.log(1e-30)], [5000, 1.5, 1.0, 0.0]),
                          x_scale='jac', xtol=1e-12, ftol=1e-12)
        c2 = float(np.sum(r.fun ** 2))
        print(f'start {start} -> chi2 = {c2:.2f}')
        if best is None or c2 < best[1]:
            best = (r, c2)
    r, c2 = best
    v = r.x
    npts = sum(len(ds[k].val) for k in FITTED)
    ndf = npts - 4
    scale = np.sqrt(c2 / ndf)
    cov = np.linalg.inv(r.jac.T @ r.jac)
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    p = params(v[0], v[1], v[2], np.exp(v[3]))

    print(f'\nfit to {", ".join(FITTED)} with m_ud = {LFQM_MASSES["u"]}, m_s = {LFQM_MASSES["s"]}, '
          f'm_c = {LFQM_MASSES["c"]} GeV')
    print(f'  chi2 = {c2:.1f}   N = {npts}   chi2/N = {c2 / npts:.3f}   chi2/dof = {c2 / ndf:.3f}')
    print('  uncertainties: Delta chi2 = 1, and in brackets multiplied by sqrt(chi2/dof)')
    print(f'  sigma0  = {v[0]:.3f} +- {sd[0]:.3f} ({scale * sd[0]:.3f}) mb')
    print(f'  gamma_s = {v[1]:.4f} +- {sd[1]:.4f} ({scale * sd[1]:.4f})')
    print(f'  lambda  = {v[2]:.4f} +- {sd[2]:.4f} ({scale * sd[2]:.4f})')
    print(f'  x0      = {p["x0"]:.4e}   ln x0 = {v[3]:.3f} +- {sd[3]:.3f} ({scale * sd[3]:.3f}); '
          f'scaled range {np.exp(v[3] - scale * sd[3]):.3e} .. {np.exp(v[3] + scale * sd[3]):.3e}')
    print('  correlation matrix (sigma0, gamma_s, lambda, ln x0):')
    print(np.array2string(corr, precision=3))
    print(f'  saturation scale Q_s^2(x = 1e-4) = {(p["x0"] / 1e-4) ** p["lam"]:.3f} GeV^2')
    per = {}
    print('\n  chi2/N per data set:')
    for k in FITTED:
        ck = ds[k].chi2(p)
        per[k] = (ck, len(ds[k].val))
        print(f'   {k:6s} N = {len(ds[k].val):3d}   chi2/N = {ck / len(ds[k].val):.2f}')
    h = ds['HERA']
    pull = (h.theory(p) - h.val) / h.err
    for lab, s in (('Q2 < 1', h.Q2 < 1), ('Q2 >= 1', h.Q2 >= 1)):
        print(f'   HERA {lab:8s} N = {s.sum():3d}   chi2/N = {np.mean(pull[s] ** 2):.2f}')

    os.makedirs(RESULTS, exist_ok=True)
    json.dump(dict(p=p, chi2=c2, npts=npts, ndf=ndf, sd=sd.tolist(), scale=scale, corr=corr.tolist(),
                   per=per), open(os.path.join(RESULTS, 'dipole_fit.json'), 'w'), indent=1, default=float)

    # the diffractive calculation reads its dipole parameters from lfvm/cgc_dipole.py
    from lfvm import cgc_dipole as dipole
    used = (dipole.SIGMA0 / dipole.MB_TO_GEV2, dipole.GAMMA_S, dipole.LAMBDA, dipole.X0)
    fitted = (p['s0'], p['gs'], p['lam'], p['x0'])
    if any(abs(a / b - 1) > 5e-4 for a, b in zip(used, fitted)):
        print('\nWARNING: lfvm/cgc_dipole.py does not hold these parameters; update it and rerun.')


if __name__ == '__main__':
    main()
