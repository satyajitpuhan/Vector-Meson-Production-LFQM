# common plot settings
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from lfvm.wavefn import HBARC, LABEL, MESONS, ORDER, SPINS, build, grid_for
from lfvm.overlap import OMEGA, cross_sections

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'figures')

# ---------------------------------------------------------------- sizes and fonts
FULL, COL = 7.2, 3.45                      # printed widths in inches
FS_LAB, FS_TICK, FS_LEG, FS_STAMP = 10.0, 8.5, 7.6, 8.4
MS, LW = 4.2, 1.6

# ---------------------------------------------------------------- axis labels
XQ2 = r'$Q^{2}$  [GeV$^{2}$]'
XW = r'$W$  [GeV]'
XT = r'$|t|$  [GeV$^{2}$]'
YSIG = r'$\sigma(\gamma^{*}p \to Vp)$  [nb]'
YDSDT = r'$\mathrm{d}\sigma/\mathrm{d}|t|$  [nb/GeV$^{2}$]'
YR = r'$R = \sigma_{L}/\sigma_{T}$'

# ---------------------------------------------------------------- colours
INK, GRID, GREY = '#1a1a1a', '#dcdbd8', '#8c8c8c'
SCOLOR = {'S1': '#1f5fae', 'S2': '#d62728'}
SSTYLE = {'S1': '-', 'S2': '--'}
LEGEND = {k: f'{LABEL[k]} (This work)' for k in SPINS}

MARKER = {'H1 2010': 'o', 'H1 2000': 's', 'H1 1996': 'D',
          'ZEUS 2007': '^', 'ZEUS 2005': 'v', 'ZEUS 1999': '<',
          'E665 1997': 'P', 'NMC 1994': 'X', 'HERMES 2009': '*',
          'ZEUS 2005/2007': 'h'}
DCOLOR = {'H1 2010': '#117733', 'H1 2000': '#AA4499', 'H1 1996': '#DDAA33',
          'ZEUS 2007': '#EE7733', 'ZEUS 2005': '#44AA99',
          'ZEUS 1999': '#882255', 'E665 1997': '#999933',
          'NMC 1994': '#555555', 'HERMES 2009': '#AA7744',
          'ZEUS 2005/2007': '#44AA99'}

# drawn with open symbols: HERMES lies outside the x range of the dipole fit, and the
# ZEUS phi/rho ratio is formed here from two separate publications
OPEN = {'HERMES 2009', 'ZEUS 2005/2007'}

# reference energy for panels that overlay several HERA data sets
W_HERA = 75.0

# the collider data sets (eps ~ 0.98, W = 75-90 GeV), which can share one curve
COLLIDER = {'H1 2010', 'H1 2000', 'H1 1996', 'ZEUS 2007', 'ZEUS 2005', 'ZEUS 1999'}

# low-energy R data, given their own legend in the R panels
LOW_ENERGY = ['H1 1996', 'E665 1997', 'HERMES 2009']

# curves are cut where x_m exceeds this value (the CGC form diverges as x_m -> 1)
XM_FIT = 0.5


def style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
        'mathtext.fontset': 'dejavuserif', 'mathtext.default': 'it',
        'font.size': 10, 'axes.labelsize': 12.5, 'legend.fontsize': 8.5,
        'legend.frameon': False, 'legend.handlelength': 2.0,
        'legend.labelspacing': 0.28, 'legend.borderpad': 0.12,
        'legend.borderaxespad': 0.35, 'legend.handletextpad': 0.5,
        'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
        'axes.edgecolor': INK, 'axes.linewidth': 1.0,
        'text.color': INK, 'axes.labelcolor': INK,
        'xtick.color': INK, 'ytick.color': INK,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.major.size': 5.0, 'ytick.major.size': 5.0,
        'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
        'xtick.minor.size': 2.8, 'ytick.minor.size': 2.8,
        'xtick.minor.visible': True, 'ytick.minor.visible': True,
        'lines.linewidth': 1.8, 'figure.dpi': 170,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.04,
    })
    plt.rcParams.update({
        'font.size': 9, 'axes.labelsize': FS_LAB, 'legend.fontsize': FS_LEG,
        'xtick.labelsize': FS_TICK, 'ytick.labelsize': FS_TICK,
        'lines.linewidth': LW, 'axes.linewidth': 0.9,
        'xtick.major.size': 4.0, 'ytick.major.size': 4.0,
        'legend.handlelength': 2.2, 'legend.labelspacing': 0.22,
        'savefig.bbox': 'standard',
        'xtick.minor.visible': False, 'ytick.minor.visible': False,
        'xtick.minor.size': 0, 'ytick.minor.size': 0,
        'xtick.minor.width': 0, 'ytick.minor.width': 0,
        'font.weight': 'bold', 'axes.labelweight': 'bold',
        'mathtext.default': 'bf',
    })


# ---------------------------------------------------------------- the model
_CACHE = {}


def gw(key, kind):
    if key not in _CACHE:
        _CACHE[key] = (grid_for(key), {})
    g, wfs = _CACHE[key]
    if kind not in wfs:
        wfs[kind] = build(g, key, kind)
    return g, wfs[kind]


def curve(key, kind, Q2, W=W_HERA, eps=OMEGA, field='sigma_tot'):
    g, wf = gw(key, kind)
    return np.array([cross_sections(g, wf, float(q), W, eps_pol=eps)[field]
                     for q in np.atleast_1d(Q2)])


def at_points(key, kind, Q2, W, eps, field='sigma_tot'):
    g, wf = gw(key, kind)
    Q2, W, eps = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (Q2, W, eps))
    n = max(len(Q2), len(W), len(eps))
    Q2, W, eps = (np.resize(v, n) for v in (Q2, W, eps))
    return np.array([cross_sections(g, wf, float(q), float(w), eps_pol=float(e))[field]
                     for q, w, e in zip(Q2, W, eps)])


def xm_of(key, Q2, W):
    return (np.asarray(Q2, dtype=float) + MESONS[key]['M'] ** 2) / (W * W)


def series_kinematics(s):
    n = len(s)
    W = s.extra.get('W', s.W if s.W is not None else W_HERA)
    eps = s.eps if s.eps is not None else OMEGA
    return np.resize(np.atleast_1d(W), n), np.resize(np.atleast_1d(eps), n)


# ---------------------------------------------------------------- drawing
DRAWN = {}


def count(fig, s, what='', key=''):
    DRAWN[(fig, key, s.label, what or getattr(s, 'kind', ''), getattr(s, 'Q2', None))] = len(s)


def draw(ax, x, y, err, label, ms=MS, scale=1.0):
    c = DCOLOR.get(label, 'k')
    ax.errorbar(np.asarray(x, float), np.asarray(y, float) * scale,
                yerr=np.asarray(err, float) * scale, ls='none',
                marker=MARKER.get(label, 'o'), ms=ms,
                mfc=c if label not in OPEN else 'none', mec=c, color=c,
                elinewidth=0.9, capsize=1.5, mew=1.0, zorder=6 + (label == 'H1 2010'))


def data_proxy(label, filled=None):
    if filled is None:
        filled = label not in OPEN
    c = DCOLOR.get(label, INK)
    return Line2D([], [], color=c, ls='none', marker=MARKER.get(label, 'o'),
                  ms=5.0, mfc=c if filled else 'none', mec=c, label=label)


def dproxy(label):
    h = data_proxy(label)
    h.set_markersize(MS + 0.4)
    return h


def sproxy():
    return [Line2D([], [], color=SCOLOR[k], ls=SSTYLE[k], lw=LW, label=LEGEND[k]) for k in SPINS]


def line(ax, x, y, kind, key=None, W=None, lw=LW):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(y)
    if key is not None and W is not None:
        ok &= xm_of(key, x, W) < XM_FIT
    ax.plot(np.where(ok, x, np.nan), np.where(ok, y, np.nan),
            ls=SSTYLE[kind], color=SCOLOR[kind], lw=lw, zorder=3)


def stamp(ax, text, loc='upper right', fs=FS_STAMP):
    xy = {'upper left': (0.04, 0.955, 'left', 'top'),
          'upper right': (0.96, 0.955, 'right', 'top')}[loc]
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha=xy[2], va=xy[3],
            fontsize=fs, zorder=8)


def mes(key):
    return rf'${MESONS[key]["tex"]}$'


def wtag(W):
    W = np.atleast_1d(np.asarray(W, dtype=float))
    if W.size > 1 and np.ptp(W) > 0.5:
        return rf'$W = {W.min():.0f}$' + '–' + rf'${W.max():.0f}$ GeV'
    return rf'$W = {float(W[0]):.0f}$ GeV'


def block(fig, n, ncol, sharex=True, sharey=True):
    nrow = int(np.ceil(n / ncol))
    axes = fig.subplots(nrow, ncol, sharex=sharex, sharey=sharey,
                        gridspec_kw=dict(wspace=0.0, hspace=0.0))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return axes[:n]


def trim_edge_ticks(axes, ncol, n):
    nrow = int(np.ceil(n / ncol))
    for ax in axes:
        ax.figure.canvas.draw()
    for i, ax in enumerate(axes):
        row, col = divmod(i, ncol)
        for axis, lo_hi, drop_lo, drop_hi in (
                (ax.xaxis, ax.get_xlim(), col > 0, col < ncol - 1),
                (ax.yaxis, ax.get_ylim(), row < nrow - 1, row > 0)):
            lo, hi = sorted(lo_hi)
            span = hi - lo
            # only ticks inside the view count (log locators return decades outside it)
            vis = [t for t in axis.get_major_ticks()
                   if lo + 1e-9 * span <= t.get_loc() <= hi - 1e-9 * span and t.label1.get_text()]
            if not vis:
                continue
            if drop_lo:
                vis[0].label1.set_visible(False)
            if drop_hi and len(vis) > 1:
                vis[-1].label1.set_visible(False)


def finish(fig, axes, ncol, xl, yl, ylx=0.0, xly=0.0):
    for ax in axes:
        ax.grid(True, which='major', color=GRID, lw=0.5, alpha=0.8)
        ax.set_axisbelow(True)
    trim_edge_ticks(axes, ncol, len(axes))
    fig.supxlabel(xl, fontsize=FS_LAB, y=xly)
    if yl:
        fig.supylabel(yl, fontsize=FS_LAB, x=ylx)


def grid(ax):
    ax.grid(True, which='major', color=GRID, lw=0.5, alpha=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + '.pdf'))
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=150)
    plt.close(fig)
    print(f'  {name}')
