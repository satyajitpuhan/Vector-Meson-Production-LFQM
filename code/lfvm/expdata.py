# reads the HERA / fixed target vector meson data
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'data')

# eps of the HERA measurements
EPS_HERA = 0.98


@dataclass
class Series:
    label: str
    x: np.ndarray
    y: np.ndarray
    err: np.ndarray
    Q2: float | None = None
    W: float | None = None
    eps: float | np.ndarray | None = EPS_HERA
    kind: str = 'T+eL'          # 'T+eL', 'T+L', 'ratio', 'R', 'dsdt'
    note: str = ''
    extra: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.x)


# ------------------------------------------------------------------ readers
def _path(name):
    p = os.path.join(DATA, name)
    return p if os.path.exists(p) else None


def load(name):
    p = _path(name)
    if p is None:
        return None
    rows, hdr = [], None
    for line in open(p):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if hdr is None:
            hdr = line.split(',')
            continue
        rows.append([float(v) for v in line.split(',')])
    a = np.array(rows, dtype=float)
    return {h: a[:, i] for i, h in enumerate(hdr)}


def hep_blocks(name):
    p = _path(name)
    if p is None:
        return []
    blocks, meta, hdr, rows = [], {}, None, []

    def flush():
        if hdr is not None and rows:
            blocks.append((dict(meta), list(hdr), [list(r) for r in rows]))

    for line in open(p):
        line = line.rstrip('\n')
        if line.startswith('#:'):
            if hdr is not None and rows:
                flush()
                hdr, rows, meta = None, [], {}
            if ',' in line:
                k, _, v = line[2:].partition(',')
                meta[k.strip()] = v.strip()
            continue
        if not line.strip():
            continue
        cells = next(csv.reader([line]))
        if hdr is None:
            hdr = cells
        else:
            rows.append(cells)
    flush()
    return blocks


def _f(seq):
    out = []
    for v in seq:
        try:
            out.append(float(str(v).replace('%', '')))
        except ValueError:
            out.append(np.nan)
    return np.asarray(out)


def _col(hdr, *prefixes):
    for i, h in enumerate(hdr):
        for p in prefixes:
            if h.upper().startswith(p.upper()):
                return i
    return None


def _meta_number(meta, key):
    import re
    for k, v in meta.items():
        if k.startswith(key):
            m = re.search(r'([0-9]+\.?[0-9]*)', v)
            if m:
                return float(m.group(1))
    return None


def _hep_points(name, ycol_prefix='SIG'):
    blocks = hep_blocks(name)
    if not blocks:
        return None
    meta, hdr, rows = blocks[0]
    return _block_points(meta, hdr, rows, ycol_prefix) + (meta,)


def _block_points(meta, hdr, rows, ycol_prefix='SIG'):
    cols = {h: [r[i] if i < len(r) else '' for r in rows]
            for i, h in enumerate(hdr)}
    keys = list(cols)
    iy = _col(hdr, ycol_prefix)
    x = _f(cols[keys[0]])
    y = _f(cols[keys[iy]])
    lo = np.zeros_like(y)
    hi = np.zeros_like(y)
    for i, k in enumerate(keys):
        if i <= iy:
            continue
        if k.startswith('stat') or k.startswith('sys'):
            v = _f(cols[k])
            if '%' in ''.join(cols[k]):
                v = v / 100.0 * y
            v = np.nan_to_num(v)
            if k.endswith('+'):
                hi += v ** 2
            else:
                lo += v ** 2
        elif k.upper().startswith(ycol_prefix.upper()):
            break
    err = 0.5 * (np.sqrt(lo) + np.sqrt(hi))
    # empty HEPData cells ('-') are not data points
    ok = np.isfinite(y)
    return x[ok], y[ok], err[ok]


# HEPData values that differ from the published paper, corrected to the paper:
# (file, Q2 of the block, x, HEPData value, paper value).  ZEUS 2007 [0708.1478] Table 21,
# Q2 = 3.7 GeV^2, W = 36 GeV: HEPData has 3240.8 nb, the paper 240.8 nb.
HEP_ERRATA = [('zeus_rho_2007_table21.csv', 3.7, 36.0, 3240.8, 240.8)]


def _apply_errata(name, q2, x, y):
    for f, q, xv, bad, good in HEP_ERRATA:
        if f == name and q2 is not None and abs(q2 - q) < 1e-6:
            m = (np.abs(x - xv) < 1e-6) & (np.abs(y - bad) < 1e-6)
            y = np.where(m, good, y)
    return y


def quad(*errs):
    return np.sqrt(sum(np.asarray(e, dtype=float) ** 2 for e in errs))


# ==========================================================================
#  sigma(Q^2)
# ==========================================================================
# (label, file, W, eps, kind) of every sigma(Q^2) measurement; 'hep:' marks a HEPData file
_SIGMA_Q2 = {
    'rho': [
        ('H1 2010', 'hep:h1_2010_table1.csv', 75.0, EPS_HERA, 'T+eL'),
        ('ZEUS 2007', 'hep:zeus_rho_2007_table20.csv', 90.0, EPS_HERA, 'T+eL'),
        ('H1 2000', 'own:h1_2000_rho_sigma_Q2.csv', 75.0, EPS_HERA, 'T+eL'),
        ('E665 1997', 'own:e665_1997_rho_sigma_Q2.csv', 18.1, None, 'T+eL'),
        ('NMC 1994', 'own:nmc_1994_rho_sigma_Q2.csv', None, None, 'T+eL'),
    ],
    'phi': [
        ('H1 2010', 'hep:h1_2010_table3.csv', 75.0, EPS_HERA, 'T+eL'),
        ('ZEUS 2005', 'hep:zeus_phi_2005_table6.csv', 75.0, EPS_HERA, 'T+eL'),
    ],
}


def sigma_Q2(key):
    out = []
    for label, spec, W, eps, kind in _SIGMA_Q2[key]:
        src, name = spec.split(':', 1)
        if src == 'hep':
            r = _hep_points(name)
            if r is None:
                continue
            x, y, e, _ = r
            out.append(Series(label, x, y, e, W=W, eps=eps, kind=kind))
        else:
            d = load(name)
            if d is None:
                continue
            err = quad(d['stat'], d.get('syst', 0.0 * d['stat']))
            out.append(Series(label, d['Q2'], d['sigma_nb'], err,
                              W=W if W is not None else None,
                              eps=d['eps'] if 'eps' in d else eps,
                              kind=kind,
                              extra={'W': d['W']} if 'W' in d else {}))
    return out


# ==========================================================================
#  sigma(W) in Q^2 bins
# ==========================================================================
_H1 = 'h1_2010_table{}.csv'
_ZR = 'zeus_rho_2007_table{}.csv'
_ZP = 'zeus_phi_2005_table{}.csv'
_SIGMA_W_HEP = {
    'rho': [('H1 2010', _H1.format(i)) for i in (9, 10, 11, 12, 13)]
           + [('ZEUS 2007', _ZR.format(i)) for i in (21, 22)],
    'phi': [('H1 2010', _H1.format(i)) for i in (17, 18, 19)]
           + [('ZEUS 2005', _ZP.format(i)) for i in (1, 2, 3, 4)],
}


def sigma_W(key):
    out = []
    for label, name in _SIGMA_W_HEP.get(key, []):
        # every Q^2 block of the file (the ZEUS 2007 tables have three each)
        for meta, hdr, rows in hep_blocks(name):
            x, y, e = _block_points(meta, hdr, rows)
            q2 = _meta_number(meta, 'Q**2')
            y = _apply_errata(name, q2, x, y)
            out.append(Series(label, x, y, e, Q2=q2, eps=EPS_HERA))
    if key != 'rho':
        return out
    for name, label, eps, kind in (
            ('h1_2000_rho_sigma_W.csv', 'H1 2000', EPS_HERA, 'T+eL'),
            ('zeus_1999_rho_sigma_W.csv', 'ZEUS 1999', EPS_HERA, 'T+eL'),
            ('h1_1996_rho_sigma_W.csv', 'H1 1996', EPS_HERA, 'T+eL'),
            ('e665_1997_rho_sigma_W.csv', 'E665 1997', None, 'T+L')):
        d = load(name)
        if d is None:
            continue
        err = quad(d['stat'], d['syst']) if 'syst' in d else d['stat']
        for q2 in np.unique(d['Q2']):
            m = d['Q2'] == q2
            out.append(Series(label, d['W'][m], d['sigma_nb'][m], err[m],
                              Q2=float(q2), eps=eps, kind=kind))
    return out


# ==========================================================================
#  dsigma/d|t|
# ==========================================================================
def dsdt(key):
    out = []
    name = {'rho': 'h1_2010_dsdt_21.csv', 'phi': 'h1_2010_dsdt_23.csv'}[key]
    for meta, hdr, rows in hep_blocks(name):
        ix = _col(hdr, 'ABS(T)')
        iy = _col(hdr, 'D(SIG)/DT')
        if ix is None or iy is None:
            continue
        x = _f([r[ix] for r in rows])
        y = _f([r[iy] for r in rows])
        err = np.zeros_like(y)
        for j in range(iy + 1, len(hdr)):
            if hdr[j].startswith('stat') or hdr[j].startswith('sys'):
                err = np.hypot(err, np.nan_to_num(_f([r[j] for r in rows])))
        out.append(Series('H1 2010', x, y, err / np.sqrt(2.0),
                          Q2=_meta_number(meta, 'Q**2'), W=75.0,
                          eps=EPS_HERA, kind='dsdt'))
    if key == 'rho':
        d = load('zeus_2007_rho_dsdt.csv')
        if d is not None:
            err = quad(d['stat'], d['syst'])
            for q2 in np.unique(d['Q2']):
                m = d['Q2'] == q2
                out.append(Series('ZEUS 2007', d['t'][m],
                                  d['dsdt_nb_per_GeV2'][m], err[m],
                                  Q2=float(q2), W=90.0, eps=EPS_HERA,
                                  kind='dsdt'))
    return out


# ==========================================================================
#  R = sigma_L / sigma_T
# ==========================================================================
_R_FILES = {
    'rho': [('H1 2010', 'h1_2010_rho_R_Q2.csv', 75.0),
            ('ZEUS 2007', 'zeus_2007_rho_R_Q2.csv', None),
            ('H1 2000', 'h1_2000_rho_R.csv', 75.0),
            ('ZEUS 1999', 'zeus_1999_rho_R_Q2.csv', None),
            ('H1 1996', 'h1_1996_rho_R_Q2.csv', 90.0),
            ('E665 1997', 'e665_1997_rho_R_Q2.csv', 18.1),
            ('HERMES 2009', 'hermes_2009_rho_R_Q2.csv', 4.8)],
    'phi': [('H1 2010', 'h1_2010_phi_R_Q2.csv', 75.0),
            ('H1 2000', 'h1_2000_phi_R.csv', 75.0)],
}


def R_Q2(key):
    out = []
    for label, name, W in _R_FILES.get(key, []):
        if name.startswith('h1_2000'):
            r = _hep_points(name, ycol_prefix='SIG')
            if r is None:
                continue
            x, y, e, _ = r
            out.append(Series(label, x, y, e, W=W, eps=None, kind='R'))
            continue
        d = load(name)
        if d is None:
            continue
        if 'err' in d:
            err = d['err']
        elif 'err_up' in d:
            err = 0.5 * (d['err_up'] + d['err_dn'])
        else:
            err = quad(d['stat'], d['syst'])
        out.append(Series(label, d['Q2'], d['R'], err,
                          W=float(np.mean(d['W'])) if 'W' in d else W,
                          eps=None, kind='R',
                          extra={'W': d['W']} if 'W' in d else {}))
    return out


def R_W(key):
    if key != 'rho':
        return []
    d = load('h1_2010_rho_R_W.csv')
    if d is None:
        return []
    err = quad(d['stat'], d['syst'])
    return [Series('H1 2010', d['W'][d['Q2'] == q], d['R'][d['Q2'] == q],
                   err[d['Q2'] == q], Q2=float(q), eps=None, kind='R')
            for q in np.unique(d['Q2'])]


# ==========================================================================
#  sigma_L and sigma_T separately
# ==========================================================================
def sigma_LT(key):
    out = []
    name = {'rho': 'h1_2010_table7.csv', 'phi': 'h1_2010_table8.csv'}[key]
    blocks = hep_blocks(name)
    for (meta, hdr, rows), tag in zip(blocks, ('T', 'L')):
        x, y, e = _block_points(meta, hdr, rows)
        out.append(Series('H1 2010', x, y, e, W=75.0, eps=EPS_HERA,
                          kind='sigma_' + tag, extra={'pol': tag}))
    if key == 'rho':
        d = load('e665_1997_rho_sigma_LT.csv')
        if d is not None:
            note = ('separated using the R(Q^2) parametrisation of the same '
                    'paper, so sigma_T and sigma_L are not independent')
            out.append(Series('E665 1997', d['Q2'], d['sigma_T_nb'],
                              d['err_T'], W=18.1, eps=None, kind='sigma_T',
                              note=note, extra={'pol': 'T'}))
            out.append(Series('E665 1997', d['Q2'], d['sigma_L_nb'],
                              d['err_L'], W=18.1, eps=None, kind='sigma_L',
                              note=note, extra={'pol': 'L'}))
    return out


# ==========================================================================
#  the phi / rho ratio
# ==========================================================================
def phi_over_rho():
    out = []
    r = _hep_points('h1_2010_table5.csv')
    if r is not None:
        x, y, e, _ = r
        out.append(Series('H1 2010', x, y, e, W=75.0, eps=None, kind='ratio'))
    rp = _hep_points('zeus_rho_2007_table20.csv')
    ph = _hep_points('zeus_phi_2005_table6.csv')
    if rp is not None and ph is not None:
        xr, yr, er, _ = rp
        xp, yp, ep, _ = ph
        xs, ys, es = [], [], []
        for q, v, d in zip(xp, yp, ep):
            j = int(np.argmin(np.abs(xr - q)))
            if abs(xr[j] - q) / q > 0.25:
                continue
            ratio = v / yr[j]
            xs.append(q)
            ys.append(ratio)
            es.append(ratio * np.hypot(d / v, er[j] / yr[j]))
        out.append(Series('ZEUS 2005/2007', np.array(xs), np.array(ys),
                          np.array(es), eps=None, kind='ratio',
                          note='ratio formed here from two separate ZEUS '
                               'measurements at different mean W; the '
                               'normalisations do not cancel'))
    return out


# ==========================================================================
#  inventory
# ==========================================================================
def inventory():
    rows = []
    for key in ('rho', 'phi'):
        for s in sigma_Q2(key):
            rows.append((key, 'sigma(Q2)', s.label, len(s)))
        for s in sigma_W(key):
            rows.append((key, f'sigma(W) at Q2={s.Q2}', s.label, len(s)))
        for s in dsdt(key):
            rows.append((key, f'dsigma/dt at Q2={s.Q2}', s.label, len(s)))
        for s in R_Q2(key):
            rows.append((key, 'R(Q2)', s.label, len(s)))
        for s in R_W(key):
            rows.append((key, f'R(W) at Q2={s.Q2}', s.label, len(s)))
        for s in sigma_LT(key):
            rows.append((key, s.kind + '(Q2)', s.label, len(s)))
    for s in phi_over_rho():
        rows.append(('phi/rho', 'ratio(Q2)', s.label, len(s)))
    return rows
