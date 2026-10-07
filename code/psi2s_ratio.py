#!/usr/bin/env python3
# psi(2S)/J/psi cross section ratio vs HERA data
import csv
import os

import numpy as np

from lfvm.wavefn import build, grid_for, SPINS
from lfvm.overlap import cross_sections

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
W_REF = 90.0                       # energy of the Q^2 dependence in the paper
QBIN = (5.0, 80.0)                 # Q^2 range of the ZEUS 2016 W and |t| bins

_WF = {}


def wf(key, kind):
    if (key, kind) not in _WF:
        g = grid_for(key)
        _WF[key, kind] = (g, build(g, key, kind))
    return _WF[key, kind]


def xs(key, kind, Q2, W):
    g, w = wf(key, kind)
    out = cross_sections(g, w, float(Q2), float(W))
    if not np.isfinite(out['sigma_L']):
        from lfvm.overlap import im_amplitude
        from lfvm.wavefn import GEV2_TO_NB
        MV = w['p']['M']
        xm = (Q2 + MV * MV) / (W * W)
        AL, AT = im_amplitude(g, w, float(Q2), xm)
        _, ATp = im_amplitude(g, w, float(Q2), xm * np.exp(-0.05))
        _, ATm = im_amplitude(g, w, float(Q2), xm * np.exp(0.05))
        alpha = (np.log(abs(ATp)) - np.log(abs(ATm))) / 0.1
        beta = np.tan(np.pi * alpha / 2.0)
        out['sigma_L'] = AL * AL * (1 + beta * beta) / (16 * np.pi * out['BD']) * GEV2_TO_NB
        out['sigma_tot'] = out['sigma_T'] + 0.98 * out['sigma_L']
        out['R'] = out['sigma_L'] / out['sigma_T']
    return out


def ratio(kind, Q2, W):
    return xs('psi2s', kind, Q2, W)['sigma_tot'] / xs('jpsi', kind, Q2, W)['sigma_tot']


def ratio_binQ2(kind, lo, hi, W, n=9):
    q = np.geomspace(lo, hi, n)
    lq = np.log(q)
    s2 = np.array([xs('psi2s', kind, x, W)['sigma_tot'] for x in q])
    s1 = np.array([xs('jpsi', kind, x, W)['sigma_tot'] for x in q])
    return np.trapz(s2, lq) / np.trapz(s1, lq)


def ratio_tbin(kind, tlo, thi, Q2, W):
    a, b = xs('jpsi', kind, Q2, W), xs('psi2s', kind, Q2, W)
    f = lambda c: c['sigma_tot'] * (np.exp(-c['BD'] * tlo) - np.exp(-c['BD'] * thi))
    return f(b) / f(a)


def ratio_tbin_Qavg(kind, tlo, thi, lo=QBIN[0], hi=QBIN[1], W=W_REF, n=9):
    q = np.geomspace(lo, hi, n)
    lq = np.log(q)
    num, den = [], []
    for x in q:
        a, b = xs('jpsi', kind, x, W), xs('psi2s', kind, x, W)
        num.append(b['sigma_tot'] * (np.exp(-b['BD'] * tlo) - np.exp(-b['BD'] * thi)))
        den.append(a['sigma_tot'] * (np.exp(-a['BD'] * tlo) - np.exp(-a['BD'] * thi)))
    return np.trapz(num, lq) / np.trapz(den, lq)


# ------------------------------------------------------------------- data
def _rows(name):
    with open(os.path.join(DATA, name)) as fh:
        return list(csv.DictReader(l for l in fh if not l.startswith('#')))


def _sym(up, dn):
    return 0.5 * (float(up) + float(dn))


def datasets():
    out = []
    for r in _rows('h1_1998_psi2s_over_jpsi.csv'):
        e = np.sqrt(float(r['stat'])**2 + float(r['syst'])**2 + float(r['br'])**2)
        out.append(dict(label='H1 1998', var='Q2', x=0.0, W=float(r['W']), y=float(r['ratio']),
                        err=e, stat=float(r['stat']), Wlo=float(r['Wlo']), Whi=float(r['Whi'])))
    for r in _rows('h1_2002_psi2s_over_jpsi_all.csv'):
        e = np.sqrt(float(r['stat'])**2 + float(r['syst'])**2 + float(r['br'])**2)
        out.append(dict(label='H1 2002', var='Q2', x=0.0, W=W_REF, y=float(r['ratio']),
                        err=e, stat=float(r['stat'])))
    for r in _rows('zeus_2022_psi2s_over_jpsi_all.csv'):
        e = np.hypot(float(r['stat']), _sym(r['syst_up'], r['syst_dn']))
        out.append(dict(label='ZEUS 2022', var='Q2', x=0.0, W=float(r['W_mean']), y=float(r['ratio']),
                        err=e, stat=float(r['stat'])))
    for r in _rows('h1_1999_psi2s_over_jpsi_Q2.csv'):
        out.append(dict(label='H1 1999', var='Q2', x=float(r['Q2']), W=float(r['W']),
                        lo=float(r['Q2lo']), hi=float(r['Q2hi']), y=float(r['ratio']),
                        err=float(r['tot']), stat=float(r['stat'])))
    for r in _rows('zeus_2016_psi2s_over_jpsi.csv'):
        lo, hi = float(r['lo']), float(r['hi'])
        e = np.hypot(float(r['stat']), _sym(r['syst_up'], r['syst_dn']))
        x = np.sqrt(lo * hi) if r['var'] == 'Q2' else (0.5 * (lo + hi))
        out.append(dict(label='ZEUS 2016', var=r['var'], x=x, lo=lo, hi=hi, W=W_REF,
                        y=float(r['ratio']), err=e, stat=float(r['stat']), dis=True))
    for r in _rows('h1_2002_psi2s_over_jpsi_W.csv'):
        e = np.hypot(float(r['stat']), float(r['syst']))
        out.append(dict(label='H1 2002', var='W', x=float(r['W']), y=float(r['ratio']),
                        err=e, stat=float(r['stat'])))
    for r in _rows('zeus_2022_psi2s_over_jpsi_W.csv'):
        e = np.hypot(float(r['stat']), _sym(r['syst_up'], r['syst_dn']))
        out.append(dict(label='ZEUS 2022', var='W', x=float(r['W']), y=float(r['ratio']),
                        err=e, stat=float(r['stat'])))
    for r in _rows('zeus_2022_psi2s_over_jpsi_t.csv'):
        e = np.hypot(float(r['stat']), _sym(r['syst_up'], r['syst_dn']))
        out.append(dict(label='ZEUS 2022', var='t', x=float(r['t_mean']), lo=float(r['tlo']),
                        hi=float(r['thi']), y=float(r['ratio']), err=e, stat=float(r['stat'])))
    # the H1 1998 photoproduction value is also shown against W, at its mean W
    out.append(dict(out[0], var='W', x=out[0]['W'], lo=out[0]['Wlo'], hi=out[0]['Whi']))
    return out


def model_at(d, kind):
    if d['var'] == 'Q2':
        if d['x'] == 0.0:
            return ratio(kind, 0.0, d['W'])
        return ratio_binQ2(kind, d['lo'], d['hi'], d['W'])
    if d['var'] == 'W':
        if d.get('dis'):
            return ratio_binQ2(kind, *QBIN, W=0.5 * (d['lo'] + d['hi']))
        return ratio(kind, 0.0, d['x'])
    if d['var'] == 't':
        if d.get('dis'):
            return ratio_tbin_Qavg(kind, d['lo'], d['hi'])
        return ratio_tbin(kind, d['lo'], d['hi'], 0.0, W_REF)
    raise ValueError(d)


def chi2():
    res = {}
    for kind in SPINS:
        per = {}
        for d in datasets():
            m = model_at(d, kind)
            c = ((d['y'] - m) / d['err']) ** 2
            for key in ((d['var'], 'all'), (d['var'], d['label'])):
                per.setdefault(key, [0.0, 0])
                per[key][0] += c
                per[key][1] += 1
        res[kind] = per
    return res


def main():
    for kind in SPINS:
        print(kind, 'R(Q2=0, W=90) = %.3f' % ratio(kind, 0.0, 90.0),
              ' R(Q2=10, W=90) = %.3f' % ratio(kind, 10.0, 90.0),
              ' <R>_{5<Q2<80} = %.3f' % ratio_binQ2(kind, *QBIN, W=90.0))
    res = chi2()
    for kind in SPINS:
        print(kind)
        for k, (c, m) in sorted(res[kind].items()):
            print(f'     {k[0]:3s} {k[1]:10s}  N={m:2d}  chi2/N={c / m:6.2f}')


if __name__ == '__main__':
    main()
