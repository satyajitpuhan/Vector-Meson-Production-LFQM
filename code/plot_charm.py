#!/usr/bin/env python3
# charmonium plots + angular condition for all four mesons
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

from plotstyle import (HERE, FULL, MS, LW, XQ2, XW, SCOLOR, SSTYLE, LEGEND,
                      style, stamp, grid, save)
from lfvm.wavefn import LABEL, SPINS
import psi2s_ratio as cr

RESULTS = os.path.join(HERE, '..', 'data', 'results')
PRES = 'GK'                                 # form-factor prescription used throughout
YRPSI = r'$\sigma_{\psi(2S)}/\sigma_{J/\psi}$'

# experiment -> (marker, colour) for the ratio data
EXP = {'H1 1998': ('D', '#DDAA33'), 'H1 1999': ('s', '#AA4499'),
       'H1 2002': ('o', '#117733'), 'ZEUS 2016': ('^', '#EE7733'),
       'ZEUS 2022': ('v', '#44AA99')}
NAME = {'H1 1998': r'H1 1998 ($\gamma p$)', 'H1 1999': 'H1 1999 (DIS)',
        'H1 2002': r'H1 2002 ($\gamma p$)', 'ZEUS 2016': 'ZEUS 2016 (DIS)',
        'ZEUS 2022': r'ZEUS 2022 ($\gamma p$)'}

# in the angular-condition figure the panel is the spin structure and the line is the meson
MSTYLE = {'rho': ('#222222', '-'), 'phi': ('#117733', '--'),
          'jpsi': ('#EE7733', '-.'), 'psi2s': ('#AA4499', ':')}
MTEX = {'rho': r'$\rho$', 'phi': r'$\phi$', 'jpsi': r'$J/\psi$', 'psi2s': r'$\psi(2S)$'}

# lattice J/psi form factors: source -> (marker, colour)
LAT = {'Dudek06': ('s', '#117733'), 'Delaney23': ('D', '#AA4499')}


def dpt(ax, d, dx=0.0):
    mk, c = EXP[d['label']]
    ax.errorbar([d['x'] + dx], [d['y']], yerr=[d['err']], ls='none', marker=mk, ms=MS,
                mfc=c, mec=c, mew=1.0, color=c, elinewidth=0.9, capsize=1.5, zorder=6)


def proxy(label):
    mk, c = EXP[label]
    return Line2D([], [], ls='none', marker=mk, ms=MS + 0.4, mfc=c, mec=c, color=c, label=NAME[label])


def sproxy():
    return [Line2D([], [], color=SCOLOR[k], ls=SSTYLE[k], lw=LW, label=LEGEND[k]) for k in SPINS]


def p9_angular_condition():
    dl = np.load(os.path.join(RESULTS, 'ff_rho_phi.npz'))
    dc = np.load(os.path.join(RESULTS, 'ff_charmonium.npz'))
    fig = plt.figure(figsize=(FULL, 2.6))
    axes = fig.subplots(1, 2, sharey=True, gridspec_kw=dict(wspace=0.0))
    for ax, kind in zip(axes, SPINS):
        for key in ('rho', 'phi', 'jpsi', 'psi2s'):
            d = dl if key in ('rho', 'phi') else dc
            Q2 = d['Q2']; o = np.argsort(Q2); sel = Q2[o] > 0.005
            c, ls = MSTYLE[key]
            ax.plot(Q2[o][sel], d[f'{key}_{kind}_Delta'][o][sel], color=c, ls=ls, lw=LW, zorder=3)
        ax.axhline(0, color='0.6', lw=0.6, zorder=1)
        ax.set_xscale('log'); ax.set_xlim(1e-2, 1e2)
        tk = [1e-2, 1e-1, 1, 10] if kind == 'S1' else [1e-1, 1, 10, 100]
        ax.set_xticks(tk)
        ax.set_xticklabels([rf'$10^{{{int(np.log10(v))}}}$' for v in tk])
        stamp(ax, LABEL[kind], loc='upper left')
        ax.set_xlabel(XQ2)
        grid(ax)
    axes[0].set_ylabel(r'$\Delta(Q^{2})$')
    h = [Line2D([], [], color=MSTYLE[k][0], ls=MSTYLE[k][1], lw=LW, label=MTEX[k])
         for k in ('rho', 'phi', 'jpsi', 'psi2s')]
    axes[0].legend(handles=h, loc='upper right', ncol=2, columnspacing=1.2,
                   handlelength=2.6, fontsize=8.5)
    fig.subplots_adjust(left=0.085, right=0.975, bottom=0.18, top=0.98)
    save(fig, 'angular_cond')


def p10_psi_ratio():
    ds = cr.datasets()
    fig = plt.figure(figsize=(FULL * 2.0 / 3.0, 2.9))
    axes = fig.subplots(1, 2)
    # left: Q^2 dependence at W = 90 GeV (photoproduction at Q^2 = 0 and the DIS points)
    ax = axes[0]
    Q2 = np.concatenate([np.linspace(0.0, 1.0, 11), np.geomspace(1.1, 60.0, 40)])
    for kind in SPINS:
        ax.plot(Q2, [cr.ratio(kind, q, cr.W_REF) for q in Q2], ls=SSTYLE[kind],
                color=SCOLOR[kind], lw=LW, zorder=3)
    shift = {'H1 1998': -0.3, 'H1 2002': 0.0, 'ZEUS 2022': 0.3}   # separate the Q^2 = 0 points
    labs = []
    for d in ds:
        if d['var'] != 'Q2':
            continue
        dpt(ax, d, dx=shift.get(d['label'], 0.0) if d['x'] == 0.0 else 0.0)
        if d['label'] not in labs:
            labs.append(d['label'])
    ax.set_xscale('symlog', linthresh=1.0, linscale=0.6)
    ax.set_xlim(-0.6, 60.0); ax.set_ylim(0.0, 0.95)
    ax.set_xticks([0, 1, 10])
    ax.set_xticklabels(['0', '1', '10'])
    ax.xaxis.set_minor_locator(matplotlib.ticker.SymmetricalLogLocator(
        linthresh=1.0, base=10, subs=np.arange(2, 10)))
    stamp(ax, rf'$W={cr.W_REF:g}$ GeV', loc='upper left')
    ax.legend(handles=[proxy(l) for l in labs], loc='upper left',
              bbox_to_anchor=(0.0, 0.89), fontsize=6.6)
    ax.set_xlabel(XQ2)
    # right: W dependence in photoproduction
    ax = axes[1]
    W = np.linspace(25.0, 185.0, 25)
    for kind in SPINS:
        ax.plot(W, [cr.ratio(kind, 0.0, w) for w in W], ls=SSTYLE[kind],
                color=SCOLOR[kind], lw=LW, zorder=3)
    labs = []
    for d in ds:
        if d['var'] == 'W' and not d.get('dis'):
            dpt(ax, d)
            if d['label'] not in labs:
                labs.append(d['label'])
    ax.set_xlim(25, 185); ax.set_ylim(0.0, 0.30)
    stamp(ax, r'$Q^2=0$', loc='upper left')
    leg = ax.legend(handles=[proxy(l) for l in labs], loc='upper right', fontsize=6.6)
    ax.add_artist(leg)
    ax.legend(handles=sproxy(), loc='lower right', fontsize=6.6)
    ax.set_xlabel(XW)
    axes[0].set_ylabel(YRPSI)
    for a in axes:
        grid(a)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.16, top=0.98, wspace=0.28)
    save(fig, 'psi2S_over_Jpsi')


def lattice():
    rows = [l.strip().split(',') for l in open(os.path.join(HERE, '..', 'data', 'lattice_jpsi_ff.csv'))
            if not l.startswith('#') and not l.startswith('source')]
    return [(r[0], r[1], float(r[2]), float(r[3]), float(r[4])) for r in rows]


def ff_figure(key, name, with_lattice):
    d = np.load(os.path.join(RESULTS, 'ff_charmonium.npz'))
    Q2 = d['Q2']
    o = np.argsort(Q2)
    Q2 = Q2[o]
    fig = plt.figure(figsize=(FULL, 2.7))
    axes = fig.subplots(1, 3)
    names = ('GC', 'GM', 'GQ')
    ylab = [r'$G_C(Q^2)$', r'$G_M(Q^2)$', r'$G_Q(Q^2)$']
    lat = lattice() if with_lattice else []
    for i, ax in enumerate(axes):
        logx = i < 2
        sel = Q2 > (0.005 if logx else -1)
        for kind in SPINS:
            ax.plot(Q2[sel], d[f'{key}_{kind}_{PRES}'][o][sel, i], color=SCOLOR[kind],
                    ls=SSTYLE[kind], lw=LW, zorder=3)
        for src, (mk, c) in LAT.items():
            pts = [(q, g, e) for s_, ff, q, g, e in lat
                   if s_ == src and ff == names[i] and q > (0.005 if logx else -1)
                   and (i == 0 or q > 1e-3)]
            if pts:
                q, g, e = map(np.array, zip(*pts))
                ax.errorbar(q, g, yerr=e, ls='none', marker=mk, ms=4.0, mfc='none', mec=c,
                            color=c, elinewidth=0.9, capsize=1.5, mew=1.1, zorder=6)
        if logx:
            ax.set_xscale('log')
            if i == 0:
                ax.set_xlim(1e-2, 10.0); ax.set_xticks([1e-2, 1e-1, 1, 10])
                ax.set_xticklabels([r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$'])
            else:
                ax.set_xlim(1e-2, 1e2); ax.set_xticks([1e-2, 1e-1, 1, 10, 100])
                ax.set_xticklabels([r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$', r'$10^{2}$'])
        else:
            ax.set_xlim(0, 6.0)
        ax.axhline(0, color='0.6', lw=0.6, zorder=1)
        ax.set_ylabel(ylab[i]); ax.set_xlabel(XQ2)
        grid(ax)
    if key == 'jpsi':
        axes[0].set_ylim(-0.05, 1.03); axes[1].set_ylim(-0.05, 2.5); axes[2].set_ylim(-0.72, 0.03)
    else:
        axes[0].set_ylim(-0.1, 1.03); axes[1].set_ylim(-0.1, 2.3)
        gq = np.concatenate([d[f'{key}_{k}_{PRES}'][o][Q2 <= 6.0, 2] for k in SPINS])
        pad = 0.08 * (gq.max() - gq.min())
        axes[2].set_ylim(gq.min() - pad, gq.max() + pad)
    stamp(axes[1], MTEX[key], loc='upper right')
    h = [Line2D([], [], color=SCOLOR[k], ls=SSTYLE[k], lw=LW, label=LABEL[k]) for k in SPINS]
    axes[0].legend(handles=h, loc='lower left', fontsize=7.2)
    if with_lattice:
        hl = tuple(Line2D([], [], ls='none', marker=mk, mfc='none', mec=c, color=c, ms=4.6, mew=1.1)
                   for mk, c in LAT.values())
        axes[2].legend([hl], ['LQCD'], handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
                       loc='lower right', fontsize=8.0)
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.17, top=0.98, wspace=0.40)
    save(fig, name)


def p11_jpsi_form_factors():
    ff_figure('jpsi', 'Jpsi_FFs_lattice', True)


def a1_psi2s_form_factors():
    ff_figure('psi2s', 'psi2S_FFs', False)


if __name__ == '__main__':
    style()
    for f in (p9_angular_condition, p10_psi_ratio, p11_jpsi_form_factors, a1_psi2s_form_factors):
        f()
