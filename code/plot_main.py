#!/usr/bin/env python3
# main plots: cross sections, R, t and W dependence, form factors
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

from plotstyle import (HERE, FULL, COL, MS, LW, FS_STAMP, XQ2, XW, XT, YSIG, YDSDT, YR, GRID, GREY,
                      SCOLOR, SSTYLE, W_HERA, COLLIDER, LOW_ENERGY, XM_FIT, DRAWN,
                      style, gw, curve, at_points, xm_of, series_kinematics, count, draw,
                      dproxy, sproxy, line, stamp, mes, wtag, block, finish, save)
from lfvm.wavefn import LABEL, ORDER, SPINS
from lfvm.overlap import cross_sections
from lfvm import expdata as expt

RESULTS = os.path.join(HERE, '..', 'data', 'results')


def p1_sigma_Q2():
    fig = plt.figure(figsize=(FULL, 2.75))
    axes = block(fig, 4, 4)
    Q2 = np.logspace(np.log10(0.8), np.log10(120.0), 80)
    for ax, key in zip(axes[:2], ORDER):
        series = [s for s in expt.sigma_Q2(key) if s.label in COLLIDER]
        # the rho data sets overlap; H1 2010 (the most precise) is drawn last
        for s in sorted(series, key=lambda s: s.label == 'H1 2010'):
            draw(ax, s.x, s.y, s.err, s.label, ms=3.4 if key == 'rho' else MS)
            count('P1', s, 'sigma', key)
        for kind in SPINS:
            line(ax, Q2, curve(key, kind, Q2), kind, key, W_HERA)
        stamp(ax, mes(key) + ', ' + wtag(W_HERA))
        h = [dproxy(s.label) for s in series] + (sproxy() if key == 'rho' else [])
        ax.legend(handles=h, loc='lower left')
    for ax, lab in zip(axes[2:], ['E665 1997', 'NMC 1994']):
        s = [x for x in expt.sigma_Q2('rho') if x.label == lab][0]
        draw(ax, s.x, s.y, s.err, s.label)
        count('P1', s, 'sigma', 'rho')
        W, eps = series_kinematics(s)
        if np.ptp(W) < 1e-9 and np.ptp(eps) < 1e-9:
            q = np.logspace(np.log10(max(0.9 * s.x.min(), 0.15)), np.log10(1.15 * s.x.max()), 45)
            for kind in SPINS:
                line(ax, q, curve('rho', kind, q, W=W[0], eps=eps[0]), kind, 'rho', W[0])
        else:
            # NMC: W and eps change from point to point, so the calculation is joined point by point
            o = np.argsort(s.x)
            for kind in SPINS:
                y = at_points('rho', kind, s.x[o], W[o], eps[o])
                fit = xm_of('rho', s.x[o], W[o]) < XM_FIT
                ax.plot(np.where(fit, s.x[o], np.nan), np.where(fit, y, np.nan),
                        ls=SSTYLE[kind], color=SCOLOR[kind], lw=LW, marker='.', ms=4, zorder=3)
        stamp(ax, mes('rho') + ', ' + wtag(W))
        ax.legend(handles=[dproxy(lab)], loc='lower left')
    for ax in axes:
        ax.set_xscale('log'); ax.set_yscale('log')
    axes[0].set_xlim(0.12, 130); axes[0].set_ylim(3e-3, 3e4)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.165, top=0.985)
    finish(fig, axes, 4, XQ2, YSIG)
    save(fig, 'sigma_vs_Q2')


def p2_sigma_LT():
    fig = plt.figure(figsize=(FULL, 3.9))
    axes = block(fig, 6, 3)
    cols = [('rho', 'H1 2010'), ('rho', 'E665 1997'), ('phi', 'H1 2010')]
    for r, pol in enumerate(('L', 'T')):
        for c, (key, lab) in enumerate(cols):
            ax = axes[3 * r + c]
            s = [x for x in expt.sigma_LT(key) if x.kind == 'sigma_' + pol and x.label == lab][0]
            draw(ax, s.x, s.y, s.err, s.label)
            count('P2', s, key=key)
            W = s.W
            q = np.logspace(np.log10(max(0.8 * s.x.min(), 0.12)), np.log10(1.3 * s.x.max()), 50)
            for kind in SPINS:
                line(ax, q, curve(key, kind, q, W=W, field='sigma_' + pol), kind, key, W)
            stamp(ax, mes(key) + rf', $\sigma_{{{pol}}}$, ' + wtag(W))
            h = [dproxy(lab)] + (sproxy() if (r, c) == (1, 0) else [])
            ax.legend(handles=h, loc='lower left')
    for ax in axes:
        ax.set_xscale('log'); ax.set_yscale('log')
    axes[0].set_xlim(0.12, 60); axes[0].set_ylim(5e-3, 4e4)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.11, top=0.99)
    finish(fig, axes, 3, XQ2, r'$\sigma_{L},\ \sigma_{T}$  [nb]')
    save(fig, 'sigmaL_sigmaT_E665')


def p3_R():
    fig = plt.figure(figsize=(FULL, 2.85))
    left, right = fig.subfigures(1, 2, width_ratios=[1.0, 1.0], wspace=0.0)
    # left: R against Q^2 for the rho and the phi
    axes = block(left, 2, 2)
    Q2 = np.logspace(np.log10(0.3), np.log10(60.0), 70)
    for ax, key in zip(axes, ORDER):
        series = expt.R_Q2(key)
        for s in series:
            draw(ax, s.x, s.y, s.err, s.label)
            count('P3', s, 'R(Q2)', key)
        for kind in SPINS:
            line(ax, Q2, curve(key, kind, Q2, field='R'), kind, key, W_HERA)
        ax.set_xscale('log'); ax.set_yscale('log')
        stamp(ax, mes(key) + ', ' + wtag(W_HERA), loc='upper left')
        main = [dproxy(s.label) for s in series if s.label not in LOW_ENERGY]
        side = [dproxy(s.label) for s in series if s.label in LOW_ENERGY]
        if key == 'phi':
            main += sproxy()
        lg = ax.legend(handles=main, loc='upper left', bbox_to_anchor=(0.0, 0.9), fontsize=7.0)
        if side:
            ax.add_artist(lg)
            ax.legend(handles=side, loc='lower right', fontsize=7.0)
    axes[0].set_xlim(0.3, 60); axes[0].set_ylim(0.12, 60)
    left.subplots_adjust(left=0.17, right=1.0, bottom=0.17, top=0.985)
    finish(left, axes, 2, XQ2, YR)
    # right: R against W for the rho in three Q^2 bins
    series = sorted(expt.R_W('rho'), key=lambda s: s.Q2)
    bx = block(right, len(series), 3)
    W = np.linspace(30.0, 165.0, 30)
    for ax, s in zip(bx, series):
        draw(ax, s.x, s.y, s.err, s.label)
        count('P3', s, 'R(W)', 'rho')
        for kind in SPINS:
            g, wf = gw('rho', kind)
            ax.plot(W, [cross_sections(g, wf, s.Q2, float(w))['R'] for w in W],
                    ls=SSTYLE[kind], color=SCOLOR[kind], lw=LW, zorder=3)
        stamp(ax, mes('rho') + rf', $Q^{{2}}={s.Q2:g}$ GeV$^{{2}}$', loc='upper left', fs=6.6)
    bx[0].set_xlim(30, 165); bx[0].set_ylim(0, 12.5)
    bx[0].legend(handles=[dproxy('H1 2010')], loc='upper right', bbox_to_anchor=(1.0, 0.88))
    right.subplots_adjust(left=0.1, right=0.99, bottom=0.17, top=0.985)
    finish(right, bx, 3, XW, None)
    save(fig, 'R_LT_ratio')


def p4_dsdt():
    groups = [('rho', 'H1 2010'), ('rho', 'ZEUS 2007'), ('phi', 'H1 2010')]
    fig = plt.figure(figsize=(FULL, 4.3))
    axes = block(fig, 3, 3)
    t = np.linspace(0.0, 1.0, 60)
    for ax, (key, lab) in zip(axes, groups):
        series = sorted([s for s in expt.dsdt(key) if s.label == lab], key=lambda s: s.Q2)
        W = series[0].W
        for i, s in enumerate(series):
            f = 10.0 ** (-2 * i)                # successive Q^2 bins scaled by 10^-2
            draw(ax, s.x, s.y, s.err, lab, scale=f)
            count('P4', s, 'dsdt', key)
            for kind in SPINS:
                g, wf = gw(key, kind)
                cs = cross_sections(g, wf, s.Q2, W)
                BD = cs['BD']
                ax.plot(t, cs['sigma_tot'] * BD * np.exp(-BD * t) * f,
                        ls=SSTYLE[kind], color=SCOLOR[kind], lw=LW, zorder=3)
            tag = rf'$Q^{{2}}={s.Q2:g}$ GeV$^{{2}}$'
            if i:
                tag += rf' ($\times10^{{{-2 * i}}}$)'
            ax.text(0.5, cs['sigma_tot'] * BD * np.exp(-BD * 0.5) * f * 9.0, tag,
                    fontsize=7.8, va='bottom', ha='center', zorder=9,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.9))
        ax.set_yscale('log')
        stamp(ax, mes(key) + ', ' + wtag(W))
        h = [dproxy(lab)] + (sproxy() if ax is axes[-1] else [])
        ax.legend(handles=h, loc='lower left')
    axes[0].set_xlim(0.0, 0.99); axes[0].set_ylim(1e-14, 3e6)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.1, top=0.99)
    finish(fig, axes, 3, XT, YDSDT)
    save(fig, 'dsigma_dt')


def p5_sigma_W():
    panels = []
    for key in ORDER:
        by = {}
        for s in expt.sigma_W(key):
            if s.Q2 is not None:
                by.setdefault(s.label, []).append(s)
        for lab in sorted(by, key=lambda l: (l.split()[0], l)):
            panels.append((key, lab, sorted(by[lab], key=lambda s: s.Q2)))
    fig = plt.figure(figsize=(FULL, 4.3))
    axes = block(fig, len(panels), 4)
    W = np.logspace(np.log10(8.0), np.log10(210.0), 40)
    for ax, (key, lab, group) in zip(axes, panels):
        eps = group[0].eps if group[0].eps is not None else 1.0
        xlab = 1.42 * max(s.x.max() for s in group)     # Q^2 labels to the right of all curves
        for s in group:
            draw(ax, s.x, s.y, s.err, lab, ms=3.8)
            count('P5', s, 'sigma(W)', key)
            m = (W > 0.75 * s.x.min()) & (W < 1.35 * s.x.max())
            for kind in SPINS:
                g, wf = gw(key, kind)
                y = np.array([cross_sections(g, wf, s.Q2, float(w), eps_pol=eps)['sigma_tot']
                              for w in W[m]])
                ax.plot(W[m], y, ls=SSTYLE[kind], color=SCOLOR[kind], lw=1.3, zorder=3)
            ax.text(xlab, y[-1], rf'${s.Q2:g}$', fontsize=7.0,
                    va='center', ha='left', color='#444444', zorder=8)
        # the E665 labels fill the top of their panel, so its title goes to the lower right
        if lab.startswith('E665'):
            ax.text(0.96, 0.05, mes(key) + ', ' + lab, transform=ax.transAxes, ha='right',
                    va='bottom', fontsize=FS_STAMP, zorder=8)
        else:
            stamp(ax, mes(key) + ', ' + lab, loc='upper left')
    for ax in axes:
        ax.set_xscale('log'); ax.set_yscale('log')
    axes[0].set_xlim(8.0, 600.0); axes[0].set_ylim(0.5, 2e4)
    axes[1].legend(handles=sproxy(), loc='lower right')
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.1, top=0.99)
    finish(fig, axes, 4, XW, YSIG)
    save(fig, 'sigma_vs_W')


def p6_phi_rho():
    fig, ax = plt.subplots(figsize=(COL, 2.7))
    Q2 = np.logspace(np.log10(1.6), np.log10(60.0), 50)
    series = expt.phi_over_rho()
    for s in series:
        draw(ax, s.x, s.y, s.err, s.label, ms=4.6)
        count('P6', s, 'ratio', 'phi/rho')
    for kind in SPINS:
        ax.plot(Q2, curve('phi', kind, Q2) / curve('rho', kind, Q2),
                ls=SSTYLE[kind], color=SCOLOR[kind], lw=LW, zorder=3)
    ax.axhline(2.0 / 9.0, color=GREY, lw=1.0, ls=':', zorder=2)
    ref = Line2D([], [], color=GREY, lw=1.0, ls=':', label=r'$e_s^2/e_\rho^2=2/9$')
    ax.set_xscale('log')
    ax.set_xlim(1.6, 60); ax.set_ylim(0.0, 0.30)
    ax.set_xlabel(XQ2); ax.set_ylabel(r'$\sigma_{\phi}/\sigma_{\rho}$')
    ax.grid(True, which='major', color=GRID, lw=0.5, alpha=0.8); ax.set_axisbelow(True)
    stamp(ax, wtag(W_HERA), loc='upper left')
    ax.legend(handles=[dproxy(s.label) for s in series] + sproxy() + [ref],
              loc='lower right', ncol=2, columnspacing=0.8)
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.17, top=0.98)
    save(fig, 'phi_to_rho')


def p7_form_factors():
    d = np.load(os.path.join(RESULTS, 'ff_rho_phi.npz'))
    Q2 = d['Q2']
    lat = []
    for fn, mk, c in [('lattice_qcdsf2008_rho.csv', 's', '#117733'),
                      ('lattice_hadspec2015_rho.csv', 'D', '#AA4499')]:
        rows = [l.strip().split(',') for l in open(os.path.join(HERE, '..', 'data', fn)) if not l.startswith('#')]
        lat.append((rows, mk, c))
    fig = plt.figure(figsize=(FULL, 2.7))
    axes = fig.subplots(1, 3)
    names = ['GC', 'GM', 'GQ']
    ylab = [r'$G_C(Q^2)$', r'$G_M(Q^2)$', r'$G_Q(Q^2)$']
    for i, ax in enumerate(axes):
        logx = i < 2                        # G_C, G_M on a log axis, G_Q on a linear one
        sel = Q2 > (0.005 if logx else -1)
        for kind in SPINS:
            ax.plot(Q2[sel], d[f'rho_{kind}_GK'][sel, i], color=SCOLOR[kind], ls=SSTYLE[kind],
                    lw=LW, zorder=3)
        for rows, mk, c in lat:
            r = [x for x in rows if x[0] == names[i] and (float(x[1]) > 0.005 or not logx)]
            q = np.array([float(x[1]) for x in r]); g = np.array([float(x[2]) for x in r])
            e = np.array([float(x[3]) for x in r])
            ax.errorbar(q, g, yerr=e, ls='none', marker=mk, ms=4.0, mfc='none', mec=c, color=c,
                        elinewidth=0.9, capsize=1.5, mew=1.1, zorder=6)
        if logx:
            ax.set_xscale('log')
            if i == 0:
                ax.set_xlim(1e-2, 5.0); ax.set_xticks([1e-2, 1e-1, 1, 5])
                ax.set_xticklabels([r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', '5'])
            else:
                ax.set_xlim(1e-2, 1e2)
                ax.set_xticks([1e-2, 1e-1, 1, 10, 100])
                ax.set_xticklabels([r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$', r'$10^{2}$'])
        else:
            ax.set_xlim(0, 3.0); ax.axhline(0, color='0.6', lw=0.6, zorder=1)
        ax.set_ylabel(ylab[i]); ax.set_xlabel(XQ2)
        ax.grid(True, which='major', color=GRID, lw=0.5, alpha=0.8); ax.set_axisbelow(True)
    axes[0].set_ylim(-0.1, 1.02); axes[1].set_ylim(-0.05, 2.5); axes[2].set_ylim(-0.95, 0.12)
    axes[0].axhline(0, color='0.6', lw=0.6, zorder=1); axes[1].axhline(0, color='0.6', lw=0.6, zorder=1)
    h = [Line2D([], [], color=SCOLOR[k], ls=SSTYLE[k], lw=LW, label=LABEL[k]) for k in SPINS]
    axes[0].legend(handles=h, loc='lower left', fontsize=7.2)
    hl = tuple(Line2D([], [], ls='none', marker=mk, mfc='none', mec=c, color=c, ms=4.6, mew=1.1)
               for rows, mk, c in lat)
    axes[2].legend([hl], ['LQCD'], handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
                   loc='lower right', fontsize=8.0)
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.17, top=0.98, wspace=0.34)
    save(fig, 'rho_FFs')


def p8_phi_form_factors():
    d = np.load(os.path.join(RESULTS, 'ff_rho_phi.npz'))
    Q2 = d['Q2']
    sel = Q2 > 0.005
    fig, ax = plt.subplots(figsize=(COL, 2.8))
    for kind in SPINS:
        G = d[f'phi_{kind}_GK']
        for i in range(3):
            ax.plot(Q2[sel], G[sel, i], color=SCOLOR[kind], ls=SSTYLE[kind], lw=LW, zorder=3)
    for txt, y in ((r'$G_M$', 2.28), (r'$G_C$', 1.12), (r'$G_Q$', -0.56)):
        ax.text(0.0125, y, txt, fontsize=9, va='center', ha='left', zorder=8)
    ax.axhline(0, color='0.6', lw=0.6, zorder=1)
    ax.set_xscale('log'); ax.set_xlim(1e-2, 1e2)
    ax.set_xticks([1e-2, 1e-1, 1, 10, 100])
    ax.set_xticklabels([r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$', r'$10^{2}$'])
    ax.set_ylim(-0.85, 2.45)
    ax.set_xlabel(XQ2); ax.set_ylabel(r'$G_C,\ G_M,\ G_Q$')
    ax.grid(True, which='major', color=GRID, lw=0.5, alpha=0.8); ax.set_axisbelow(True)
    h = [Line2D([], [], color=SCOLOR[k], ls=SSTYLE[k], lw=LW, label=LABEL[k]) for k in SPINS]
    ax.legend(handles=h, loc='upper right', fontsize=7.4)
    fig.subplots_adjust(left=0.2, right=0.955, bottom=0.16, top=0.98)
    save(fig, 'phi_FFs')


def main():
    style()
    for f in (p1_sigma_Q2, p2_sigma_LT, p3_R, p4_dsdt, p5_sigma_W, p6_phi_rho,
              p7_form_factors, p8_phi_form_factors):
        f()
    inv = expt.inventory()
    n_inv = sum(r[3] for r in inv)
    n_drawn = sum(DRAWN.values())
    print(f'\ndata inventory: {len(inv)} data sets, {n_inv} points')
    print(f'drawn:          {len(DRAWN)} data sets, {n_drawn} points')
    if (len(inv), n_inv) != (len(DRAWN), n_drawn):
        sys.exit('some data points are not drawn')


if __name__ == '__main__':
    main()
