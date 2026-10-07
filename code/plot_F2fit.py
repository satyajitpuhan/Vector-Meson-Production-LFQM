#!/usr/bin/env python3
# plots of the dipole fit to the structure function data
import json
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plotstyle import (HERE, FULL, FS_LAB, FS_LEG, MS, LW, XQ2, DCOLOR, MARKER,
                      style, stamp, grid, trim_edge_ticks, save)
import fit_dipole_F2 as sf

FIT = json.load(open(os.path.join(HERE, '..', 'data', 'results', 'dipole_fit.json')))['p']
M = sf.LFQM_MASSES
INK = '#1a1a1a'
XX = r'$x_{Bj}$'
# HERA beam energies: colour and marker of data and curve
ECOL = {318: '#117733', 300: '#EE7733', 251: '#44AA99', 225: '#882255'}
EMK = {318: 'o', 300: 's', 251: '^', 225: 'D'}


def pts(ax, x, y, e, c='#117733', mk='o', filled=True, ms=MS - 0.6):
    ax.errorbar(x, y, yerr=e, ls='none', marker=mk, ms=ms, mfc=c if filled else 'none', mec=c,
                color=c, elinewidth=0.9, capsize=1.3, mew=1.0, zorder=6)


def a2_hera():
    h = sf.datasets(M)['HERA']
    Q2, x, sr, e = h.Q2, h.x, h.val, h.err
    sqs = np.rint(np.sqrt(Q2 / (x * h.y))).astype(int)
    pull = (h.theory(FIT) - sr) / e
    bins = np.unique(Q2)
    ncol, nrow = 4, int(np.ceil(len(bins) / 4))
    fig = plt.figure(figsize=(FULL, 1.0 * nrow + 0.6))
    axes = fig.subplots(nrow, ncol, sharex=True, sharey='row',
                        gridspec_kw=dict(wspace=0.0, hspace=0.0)).ravel()
    rows = {}
    for i, (ax, q) in enumerate(zip(axes, bins)):
        s = Q2 == q
        ys = rows.setdefault(i // ncol, [])
        ys += list(sr[s] - e[s]) + list(sr[s] + e[s])
        for sq in np.unique(sqs[s]):
            ss = s & (sqs == sq)
            xs = np.geomspace(x[ss].min() / 1.25, min(x[ss].max() * 1.25, 0.01), 30)
            th = sf.DataSet('', np.full_like(xs, q), xs, xs, xs, M, 'sr', q / (sq ** 2 * xs)).theory(FIT)
            ys += list(th)
            ax.plot(xs, th, color=ECOL[sq], lw=1.5, zorder=4)
            pts(ax, x[ss], sr[ss], e[ss], ECOL[sq], EMK[sq], sq == 318, ms=3.2)
        stamp(ax, rf'$Q^2={q:g}$', loc='upper right', fs=7.4)
        low = q < 0.18                      # in the lowest bins the data sit at the left edge
        ax.text(0.96 if low else 0.04, 0.05, rf'$\chi^2/N={np.mean(pull[s] ** 2):.1f}$',
                transform=ax.transAxes, ha='right' if low else 'left', va='bottom',
                fontsize=6.6, color='#444444')
        grid(ax)
    for r, ys in rows.items():
        lo, hi = min(ys), max(ys)
        axes[r * ncol].set_ylim(lo - 0.06 * (hi - lo), hi + 0.38 * (hi - lo))
        axes[r * ncol].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
    axes[0].set_xscale('log'); axes[0].set_xlim(1e-7, 5e-2)
    axes[0].set_xticks([1e-6, 1e-4, 1e-2])
    axes[0].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    for ax in axes:
        ax.tick_params(labelsize=7.6)
    for ax in axes[len(bins):]:
        ax.set_visible(False)
    trim_edge_ticks(list(axes[:len(bins)]), ncol, len(bins))
    fig.supxlabel(XX, fontsize=FS_LAB, y=0.005)
    fig.supylabel(r'$\sigma_r$', fontsize=FS_LAB, x=0.0)
    hdl = [Line2D([], [], ls='-', lw=1.5, marker=EMK[s], ms=MS, mec=ECOL[s], color=ECOL[s],
                  mfc=ECOL[s] if s == 318 else 'none', label=rf'$\sqrt{{s}}={s}$ GeV')
           for s in (318, 300, 251, 225)]
    fig.legend(handles=hdl, loc='upper center', ncol=4, fontsize=FS_LEG, bbox_to_anchor=(0.5, 1.0),
               columnspacing=1.6, handletextpad=0.4, handlelength=2.6,
               title='HERA data and CGC fit (this work)', title_fontsize=FS_LEG)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.045, top=1 - 0.5 / (1.0 * nrow + 0.6))
    save(fig, 'sigma_r_HERA_fit')


def a4_fixed_target():
    e665, nmc = sf.load_fixed_target('e665'), sf.load_fixed_target('nmc')
    sets = ([('E665', x, e665) for x in np.unique(e665[:, 1])]
            + [('NMC', x, nmc) for x in np.unique(nmc[:, 1])])
    fig, axes = plt.subplots(2, 5, figsize=(FULL, 3.3), sharey=True,
                             gridspec_kw=dict(wspace=0.08, hspace=0.32))
    axes = axes.ravel()
    col = {'E665': DCOLOR['E665 1997'], 'NMC': DCOLOR['NMC 1994']}
    mk = {'E665': MARKER['E665 1997'], 'NMC': MARKER['NMC 1994']}
    for ax, (lab, x, d) in zip(axes, sets):
        s = d[:, 1] == x
        Q = np.geomspace(d[s, 0].min() / 1.25, d[s, 0].max() * 1.25, 30)
        ax.plot(Q, sf.DataSet('', Q, np.full_like(Q, x), Q, Q, M, 'F2').theory(FIT), color=col[lab],
                lw=1.5, zorder=4)
        pts(ax, d[s, 0], d[s, 2], d[s, 3], col[lab], mk[lab])
        ax.set_xscale('log')
        ticks = [t for t in (0.2, 0.5, 1, 2, 5) if Q.min() < t < Q.max()]
        ax.set_xticks(ticks); ax.set_xticklabels([f'{t:g}' for t in ticks])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        stamp(ax, f'{lab}\n$x={x:g}$', loc='upper left', fs=7.6)
    axes[0].set_ylim(0, 0.56)
    for ax in axes:
        grid(ax); ax.tick_params(labelsize=8)
    fig.supxlabel(XQ2, fontsize=FS_LAB, y=0.0)
    fig.supylabel(r'$F_2^{p}$', fontsize=FS_LAB, x=0.0)
    h = [Line2D([], [], ls='-', lw=1.5, marker=mk[k], ms=MS, color=col[k], mec=col[k],
                label=f'{k} {"1996" if k == "E665" else "1997"} and CGC fit') for k in ('E665', 'NMC')]
    fig.legend(handles=h, loc='upper center', ncol=3, fontsize=FS_LEG, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.14, top=0.91)
    save(fig, 'F2_fixed_target')


def a6_charm():
    c = sf.load_charm()
    bins = np.unique(c[:, 0])
    fig, axes = plt.subplots(2, 3, figsize=(FULL, 3.5), sharex=True,
                             gridspec_kw=dict(wspace=0.26, hspace=0.08))
    axes = axes.ravel()
    for ax, q in zip(axes, bins):
        s = c[:, 0] == q
        xs = np.geomspace(c[s, 1].min() / 1.3, min(c[s, 1].max() * 1.3, 0.01), 30)
        th = sf.DataSet('', np.full_like(xs, q), xs, xs, xs, M, 'sr', q / (318.0 ** 2 * xs),
                        flavours='c').theory(FIT)
        ax.plot(xs, th, color=INK, lw=LW, zorder=4)
        pts(ax, c[s, 1], c[s, 2], c[s, 2] * c[s, 3] / 100, '#117733', 'o')
        ax.set_xscale('log'); ax.set_ylim(0, 1.25 * c[s, 2].max() + 0.02)
        stamp(ax, rf'$Q^2={q:g}$ GeV$^2$', loc='upper right', fs=7.8)
        ax.tick_params(labelsize=7.8)
    for ax in axes:
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        grid(ax)
    fig.supxlabel(XX, fontsize=FS_LAB, y=0.0)
    fig.supylabel(r'$\sigma_r^{c\bar c}$', fontsize=FS_LAB, x=0.0)
    axes[0].legend(handles=[Line2D([], [], color=INK, lw=LW, label='CGC fit (this work)'),
                            Line2D([], [], ls='none', marker='o', color='#117733', ms=MS,
                                   label='H1+ZEUS 2018')], loc='lower left', fontsize=FS_LEG - 0.4)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.13, top=0.985)
    save(fig, 'F2_charm')


if __name__ == '__main__':
    style()
    a2_hera()
    a4_fixed_target()
    a6_charm()
