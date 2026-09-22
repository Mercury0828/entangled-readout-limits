"""Main-text figures 1-5, vector PDF at final size (88 mm single column, 180 mm double).

    python c121_figures.py            # reads data/selfcheck/figure_data.json only -> figs/fig{1..5}.pdf/.png

Palette: magenta / turquoise / grey / black (no red-green pairing). One font family (DejaVu Sans,
upright and bold only; no mathtext, so no shrunken super/subscripts). The legends are in the paper.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FD = json.load(open(os.path.join(HERE, 'data', 'selfcheck', 'figure_data.json')))
FIGS = os.path.join(HERE, 'figs')
MM = 1 / 25.4
MAG, TUR, GRY, BLK = '#C2185B', '#00897B', '#8C8C8C', '#000000'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.labelsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
                     'axes.linewidth': 0.6, 'lines.linewidth': 1.2, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'axes.unicode_minus': True, 'axes.formatter.use_mathtext': False})
ETA_STAR = np.sqrt(2 / 3)
SUP = str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻')


def logticks(ax, fmt=None):
    f = fmt or (lambda v, _: '10' + str(int(round(np.log10(v)))).translate(SUP))
    ax.xaxis.set_major_formatter(mt.FuncFormatter(f))
    ax.xaxis.set_minor_formatter(mt.NullFormatter())


def panel(ax, letter):
    ax.set_title(letter, loc='left', fontsize=8, fontweight='bold', pad=3)


def new(w_mm, h_mm, ncols=1, **kw):
    return plt.subplots(1, ncols, figsize=(w_mm * MM, h_mm * MM), layout='constrained', **kw)


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, name + '.pdf'))
    fig.savefig(os.path.join(FIGS, name + '.png'), dpi=300)
    plt.close(fig)


def fig1():
    """Overview figure: 180 x 108 mm, 2 x 2."""
    from matplotlib.patches import Ellipse, Rectangle
    th = FD['theory']
    W, H = 180.0, 108.0
    fig = plt.figure(figsize=(W * MM, H * MM))
    boxes = {'a': (3, 63, 84, 42), 'b': (93, 63, 84, 42), 'c': (3, 3, 84, 54), 'd': (93, 3, 84, 54)}
    titles = {'a': 'Requested observables', 'b': 'Measurement designs', 'c': 'Optimal cost', 'd': 'Optimal mixture'}
    axes = {}
    for k, (x0, y0, w, h) in boxes.items():
        ax = fig.add_axes([x0 / W, y0 / H, w / W, h / H])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
        ax.text(0.0, 0.99, k, fontsize=8, fontweight='bold', va='top', ha='left', transform=ax.transAxes)
        ax.text(0.065, 0.99, titles[k], fontsize=8, fontweight='bold', va='top', ha='left', transform=ax.transAxes)
        axes[k] = (ax, w, h)

    def node(ax, w, h, x, y, label=None, edge=BLK, lw=1.0):
        d = 2.8
        ax.add_patch(Ellipse((x, y), d / w, d / h, facecolor='white', edgecolor=edge, lw=lw, zorder=3))
        if label:
            ax.text(x, y, label, ha='center', va='center', fontsize=7, zorder=4)

    def edge(ax, p, q, col=BLK, lw=1.0, w=None, h=None):
        # stop at the circle boundaries (radius 1.4 mm), measured in physical units
        (px, py), (qx, qy) = p, q
        dx, dy = (qx - px) * w, (qy - py) * h
        L = (dx * dx + dy * dy) ** 0.5
        f = 1.4 / L
        ax.plot([px + (qx - px) * f, qx - (qx - px) * f], [py + (qy - py) * f, qy - (qy - py) * f],
                color=col, lw=lw, zorder=1, solid_capstyle='butt')

    # ---- a: requested-observable graphs
    ax, w, h = axes['a']
    ax.text(0.50, 0.88, 'Edges = requested two-qubit Pauli expectations', ha='center', va='center')
    ax.text(0.24, 0.755, 'Degree ≥3', ha='center', va='center')
    c0 = (0.24, 0.45)
    nbrs = [(0.24, 0.66), (0.149, 0.345), (0.331, 0.345)]
    for p in nbrs:
        edge(ax, c0, p, MAG, 1.2, w, h)
    node(ax, w, h, *c0)
    for p in nbrs:
        node(ax, w, h, *p)
    ax.text(0.725, 0.755, 'Chain', ha='center', va='center')
    xs = [0.51, 0.596, 0.682, 0.768, 0.854, 0.94]
    for i in range(5):
        edge(ax, (xs[i], 0.45), (xs[i + 1], 0.45), TUR, 1.2, w, h)
    for x in xs:
        node(ax, w, h, x, 0.45)
    ax.text(0.24, 0.255, '27 correlations, budget ≤ 3', ha='center', va='center')
    ax.text(0.24, 0.165, 'Product optimal', ha='center', va='center')
    ax.text(0.24, 0.055, 'C = 9 for all η', ha='center', va='center')
    ax.text(0.725, 0.165, 'Mixed design can win', ha='center', va='center')
    ax.text(0.725, 0.055, 'η > η*', ha='center', va='center')

    # ---- b: measurement designs on the chain
    ax, w, h = axes['b']
    X = [0.25, 0.38, 0.51, 0.64, 0.77, 0.90]
    ax.text(0.50, 0.88, 'P: independently choose X, Y or Z', ha='center', va='center')
    for i, x in enumerate(X):
        ax.text(x, 0.80, str(i + 1), ha='center', va='center')
    rows = [(0.72, 'Product', 'w', []), (0.46, 'Cover A', '(1−w)/2', [(0, 1), (2, 3), (4, 5)]),
            (0.20, 'Cover B', '(1−w)/2', [(1, 2), (3, 4)])]
    for y, lab, prob, pairs in rows:
        for i in range(5):
            edge(ax, (X[i], y), (X[i + 1], y), GRY, 1.0, w, h)
        paired = {i for p in pairs for i in p}
        for i, x in enumerate(X):
            if i in paired:
                node(ax, w, h, x, y)
            else:
                node(ax, w, h, x, y, 'P', edge=MAG, lw=1.2)
        ax.text(0.015, y + 0.015, lab, ha='left', va='center')
        ax.text(0.015, y - 0.075, prob, ha='left', va='center')
        for i, j in pairs:
            xm = (X[i] + X[j]) / 2
            top = y + 2.8 / h
            gy = y + 0.143
            gh = 3.5 / h
            ax.plot([X[i], X[i], X[j], X[j]], [y + 1.4 / h, top, top, y + 1.4 / h], color=TUR, lw=1.2, zorder=2)
            ax.plot([xm, xm], [top, gy - gh / 2], color=TUR, lw=1.2, zorder=2)
            gw = 4.0 / w
            ax.add_patch(Rectangle((xm - gw / 2, gy - gh / 2), gw, gh, facecolor='white', edgecolor=TUR,
                                   lw=1.2, zorder=3))
            ax.text(xm, gy, 'G', ha='center', va='center', zorder=4)
            dw = 5.8 / w
            dx = xm + 0.083
            ax.add_patch(Rectangle((dx - dw / 2, gy - gh / 2), dw, gh, facecolor='white', edgecolor=GRY,
                                   lw=1.0, ls='--', zorder=3))
            ax.text(dx, gy, 'Dη', ha='center', va='center', zorder=4)
            ax.annotate('', xy=(dx - dw / 2, gy), xytext=(xm + gw / 2, gy),
                        arrowprops=dict(arrowstyle='->', lw=0.8, color=BLK, shrinkA=0, shrinkB=0), zorder=4)

    # ---- c, d: theory curves in a common frame
    eta = np.array(th['eta']); Cc = np.array(th['C_chain']); wv = np.array(th['w'])
    lo = eta <= ETA_STAR
    for k in ('c', 'd'):
        ax, w, h = axes[k]
        x0, y0, bw, bh = boxes[k]
        pa = fig.add_axes([(x0 + 0.15 * bw) / W, (y0 + 0.20 * bh) / H, 0.805 * bw / W, 0.62 * bh / H])
        pa.set_xlim(0.5, 1.0)
        pa.patch.set_visible(False)
        pa.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        pa.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f'{v:.1f}'))
        pa.axvline(ETA_STAR, color=GRY, lw=1.0, ls=(0, (2, 2)))
        ax.text(0.55, 0.065, 'Gate quality η', ha='center', va='center')
        ax.text(0.660, 0.895, 'η* = √(2/3)', ha='center', va='center')
        axes[k] = (ax, w, h, pa)
    ax, w, h, pa = axes['c']
    pa.set_ylim(6.7, 9.6); pa.set_yticks([7, 8, 9])
    ax.text(0.035, 0.51, 'Optimal constant C', rotation=90, ha='center', va='center')
    pa.plot(eta, np.full_like(eta, 9.0), color=MAG, lw=1.6, ls=(0, (5, 2.5)))
    pa.plot(eta[lo], np.full(lo.sum(), 9.0), color=BLK, lw=1.6)
    pa.plot(eta[~lo], Cc[~lo], color=TUR, lw=1.6)
    pa.plot([1.0], [9.0], 's', ms=6, color=MAG, clip_on=False)
    pa.plot([1.0], [7.0], 'o', ms=6, color=TUR, clip_on=False)
    ax.text(0.36, 0.755, 'Both: C = 9', ha='center', va='center')
    ax.text(0.81, 0.755, 'Degree ≥3', ha='center', va='center')
    ax.text(0.73, 0.435, 'Chain', ha='center', va='center')
    ax.text(0.73, 0.375, '3+4/η²', ha='center', va='center')
    ax, w, h, pa = axes['d']
    pa.set_ylim(0, 1.1); pa.set_yticks([0, 1 / 3, 2 / 3, 1]); pa.set_yticklabels(['0', '1/3', '2/3', '1'])
    ax.text(0.035, 0.51, 'Product weight w', rotation=90, ha='center', va='center')
    pa.plot(eta[lo], wv[lo], color=MAG, lw=1.6, ls=(0, (5, 2.5)))
    pa.plot(eta[~lo], wv[~lo], color=TUR, lw=1.6)
    pa.plot([ETA_STAR], [1.0], 's', ms=6, color=MAG)
    pa.plot([ETA_STAR], [1 / 3], 'o', ms=6, mfc='white', mec=TUR, mew=1.2)
    pa.plot([1.0], [3 / 7], 'o', ms=6, color=TUR, clip_on=False)
    ax.text(0.35, 0.815, 'Product only', ha='center', va='center')
    ax.text(0.81, 0.61, 'Mixed', ha='center', va='center')
    ax.text(0.81, 0.55, '3η²/(4+3η²)', ha='center', va='center')
    ax.text(0.92, 0.355, '3/7', ha='center', va='center')
    for k in ('c', 'd'):
        pa = axes[k][3]
        pa.spines['top'].set_visible(False); pa.spines['right'].set_visible(False)
        pa.spines['left'].set_linewidth(1.0); pa.spines['bottom'].set_linewidth(1.0)
    fig.savefig(os.path.join(FIGS, 'fig1.pdf'))
    fig.savefig(os.path.join(FIGS, 'fig1.png'), dpi=300)
    plt.close(fig)


def fig2():
    th = FD['theory']
    fig, axs = new(180, 60, 2)
    ax = axs[0]
    br = np.array(th['bracket'])
    ax.fill_between(100 * br[:, 0], br[:, 1], br[:, 2], color=TUR, alpha=0.3, lw=0)
    ax.plot(100 * br[:, 0], br[:, 1], color=TUR, lw=1.0, label='lower bound')
    ax.plot(100 * br[:, 0], br[:, 2], color=MAG, lw=1.0, ls='--', label='upper bound')
    e = np.linspace(0.5, 0.998, 400)
    x = 75 * (1 - e)
    ax.plot(x, np.where(e * e <= 2 / 3, 9.0, 3 + 4 / (e * e)), color=BLK, lw=0.8, ls=':',
            label='pair depolarising (exact)')
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(mt.FixedLocator([0.2, 0.5, 1, 2, 5, 10, 20]))
    logticks(ax, lambda v, _: f'{v:g}')
    ax.set_xlim(0.15, 30)
    ax.set_xlabel('average gate infidelity (%)'); ax.set_ylabel('minimax constant C')
    ax.legend(loc='upper left'); panel(ax, 'a')

    ax = axs[1]
    ax.plot(th['ww'], th['Cw'], color=TUR)
    ax.axvline(3 / 7, color=GRY, lw=0.6, ls=':')
    ax.text(3 / 7 - 0.03, 14.2, 'slope −16.3', ha='right')
    ax.text(3 / 7 + 0.03, 14.2, 'slope +2.7', ha='left')
    ax.set_xlabel('product weight w'); ax.set_ylabel('worst-coordinate constant (η = 1)')
    ax.set_xticks([0.2, 3 / 7, 0.6, 0.8]); ax.set_xticklabels(['0.2', '3/7', '0.6', '0.8'])
    ax.set_ylim(6.5, 15); panel(ax, 'b')
    save(fig, 'fig2')


def fig3():
    f3 = FD['fig3']
    fig, axs = new(180, 64, 2)
    ax = axs[0]
    rng = np.random.default_rng(3)
    d = np.array(f3['c110_edge_points_dev']); s = np.array(f3['c110_edge_points_sim'])
    ax.scatter(d[:, 0] - 0.12 + rng.uniform(-0.05, 0.05, len(d)), d[:, 1] - d[:, 0], s=3, color=MAG,
               alpha=0.35, lw=0, label='device')
    ax.scatter(s[:, 0] + 0.12 + rng.uniform(-0.05, 0.05, len(s)), s[:, 1] - s[:, 0], s=6, color=GRY,
               alpha=0.8, lw=0, marker="s", label="simulator")
    m = f3['c110']['prod_edge']
    for x0 in (-1, 1):
        xx = np.array([x0 - 0.2, x0 + 0.2])
        ax.plot(xx, [(m['a_dev'] - 1) * x0 + m['c_dev']] * 2, color=BLK, lw=1.0,
                label='fit, device' if x0 == -1 else None)
    ax.axhline(0, color=GRY, lw=0.5)
    ax.set_xticks([-1, 1]); ax.set_xlim(-1.6, 1.6)
    ax.set_xlabel('realised sign of the circuit'); ax.set_ylabel('observed parity − realised sign')
    ax.legend(loc='upper right', markerscale=2); panel(ax, 'a')

    ax = axs[1]
    names = [('prod_single', 'prod_single_z', 'product, Z'), ('prod_edge', 'prod_edge_zz', 'product, ZZ'),
             (None, 'prod_edge_yy', 'product, YY'), ('bell_w1', 'bell_w1_zz', 'Bell, ZZ'),
             (None, 'bell_w2_yy', 'Bell, YY')]
    for k, (a110, a112, lab) in enumerate(names):
        if a110 in f3['c110']:
            v = f3['c110'][a110]
            ax.errorbar(v['a_dev'], k + 0.13, xerr=1.96 * v['se_dev'], fmt='o', ms=3, color=MAG,
                        label='held-out run' if k == 0 else None)
        v = f3['c112'][a112]
        ax.errorbar(v['a'], k - 0.13, xerr=1.96 * v['se'], fmt='s', ms=3, color=TUR,
                    label='target-resolved run' if k == 0 else None)
    ax.axvline(1.0, color=GRY, lw=0.8, ls='--', label='simulator')
    ax.set_yticks(range(5)); ax.set_yticklabels([n[2] for n in names]); ax.set_ylim(4.6, -0.6)
    ax.set_xlim(0.935, 1.008)
    ax.set_xlabel('response slope a')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, borderaxespad=0.2); panel(ax, 'b')
    save(fig, 'fig3')


LABEL = {'prod_single_z': 'product, Z on q2', 'prod_edge_zz': 'product, ZZ on (2,3)',
         'prod_edge_yy': 'product, YY on (2,3)', 'bell_w1_zz': 'Bell, ZZ on (2,3)', 'bell_w2_yy': 'Bell, YY on (2,3)'}


def fig4():
    f4, tau = FD['fig4'], FD['tau']
    fig, axs = new(180, 66, 2)
    ax = axs[0]
    modes = list(LABEL)
    for model, col, off, mk in (('target', TUR, -0.14, 's'), ('pooled', MAG, 0.14, 'o')):
        r = f4[model]
        for k, md in enumerate(modes):
            v, lo, hi = r['per_mode'][md], r['per_mode_lo'][md], r['per_mode_hi'][md]
            ax.errorbar(1e3 * v, k + off, xerr=[[1e3 * (v - lo)], [1e3 * (hi - v)]], fmt=mk, ms=3, color=col,
                        label=('target-resolved' if model == 'target' else 'pooled') if k == 0 else None)
    for t in (-tau, tau):
        ax.plot([1e3 * t] * 2, [len(modes) - 0.4, -0.55], color=BLK, lw=0.7, ls=':')
        ax.text(1e3 * t, -0.62, '−τ' if t < 0 else '+τ', ha='center', va='bottom')
    ax.plot([0, 0], [len(modes) - 0.4, -0.55], color=GRY, lw=0.5)
    ax.xaxis.set_major_locator(mt.MultipleLocator(5))
    ax.set_yticks(range(len(modes))); ax.set_yticklabels([LABEL[m] for m in modes])
    ax.set_ylim(len(modes) - 0.4, -1.0)
    ax.set_xlabel('residual bias β (×10⁻³)')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, borderaxespad=0.2); panel(ax, 'a')

    ax = axs[1]
    bins = np.linspace(0, 14, 57)
    top = 0
    for model, col, kind in (('target', TUR, 'stepfilled'), ('pooled', MAG, 'step')):
        r = f4[model]
        h, *_ = ax.hist(1e3 * np.array(r['Bboot']), bins=bins, color=col, histtype=kind,
                        alpha=0.5 if kind == 'stepfilled' else 1.0, lw=1.2,
                        label='target-resolved' if model == 'target' else 'pooled')
        top = max(top, h.max())
        ax.axvline(1e3 * r['UB95'], color=col, lw=1.0, ls='--')
    ax.set_ylim(0, top * 1.25)
    for model, col, ha, dx in (('target', TUR, 'right', -0.15), ('pooled', MAG, 'left', 0.15)):
        ax.text(1e3 * f4[model]['UB95'] + dx, top * 1.17, f"{1e3 * f4[model]['UB95']:.2f}", color=col, ha=ha)
    ax.axvline(1e3 * tau, color=BLK, lw=1.0, ls=':')
    ax.text(1e3 * tau + 0.2, top * 1.04, 'τ', ha='left')
    ax.set_xlabel('worst-coordinate bias B (×10⁻³)'); ax.set_ylabel('bootstrap draws')
    ax.set_xlim(0, 14)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, borderaxespad=0.2); panel(ax, 'b')
    save(fig, 'fig4')


def fig5():
    rc = FD['fig5']
    N = np.array(rc['N'])
    fig, ax = new(88, 62)
    for model, col, ls in (('target', TUR, '-'), ('pooled', MAG, '--')):
        r = rc[model]
        ax.fill_between(N, r['ratio_lo'], r['ratio_hi'], color=col, alpha=0.35, lw=0)
        ax.plot(N, r['ratio'], color=col, ls=ls, label='target-resolved' if model == 'target' else 'pooled')
    ax.axhline(1, color=BLK, lw=0.7, ls=':')
    ax.text(N[-1], 0.985, 'mixture preferred', ha='right', va='top')
    ax.set_xscale('log'); logticks(ax); ax.set_xlabel('estimation copies N (calibration excluded)')
    ax.set_ylabel('risk ratio, mixture / product'); ax.legend(loc='upper left')
    save(fig, 'fig5')


CAPTIONS = """# Figure captions (draft; each opens with what kind of quantity is plotted)

**Fig. 1 | Derived: the minimax constant is set by the graph of requested observables.** a, Schematic of two
requested-observable graphs: a vertex of degree 3 and a chain. b, Derived minimax constant C(η) from Theorems 1
and 2: 9 at every η whenever a vertex has degree ≥ 3; on the chain 9 for η² ≤ 2/3 and 3 + 4/η² above. Dotted
line: η* = √(2/3) (average gate infidelity 13.76%). c, Derived optimal product weight w = 3η²/(4 + 3η²) of the
chain design; the remainder is split equally between two alternating Bell-dimer coverings.

**Fig. 2 | Derived: robustness of the chain result.** a, Derived two-sided bounds on the chain constant for
sparse Pauli–Lindblad noise (six single-site and three two-site generators) against average gate infidelity. The
bounds coincide for pair-depolarising noise. b, Derived worst-coordinate constant of the chain design against the
product weight at η = 1, max{3/w, 18/(3 − w)}. The optimum w = 3/7 is a kink with one-sided slopes −16.3 and +2.7.

**Fig. 3 | Measured and fitted: implemented response on ibm_cleveland.** a, Measured parity minus realised sign
for each circuit of the product ZZ mode in the held-out run (705 circuits, 60 shots each; magenta, device; grey,
noiseless simulator of the identical circuits). Points are jittered horizontally and the two sources offset for
visibility; bars, least-squares fit y = ax + c to the device data, shown as the residual (a − 1)x + c. b, Fitted
response slope a per measured mode with 95% bootstrap intervals over circuits (1.96 × s.e., 400 resamples;
intervals smaller than the markers where not visible): held-out run (magenta, 705–2,069 circuits per mode; the
Bell YY mode was not measured in that run) and target-resolved run (turquoise, 1,800 circuits per mode). Dashed:
the simulator returns a = 1 in every mode.

**Fig. 4 | Fitted: prospective transfer test.** a, Residual bias β of each target coordinate on evaluation
circuits, for predictions from the target-resolved (turquoise) and pooled (magenta) calibration models fitted to
the same calibration circuits, with 95% bootstrap intervals (1,000 joint resamples of calibration and evaluation
circuits); dotted, the declared tolerance ±τ = ±0.0085. b, Bootstrap distribution of the worst-coordinate bias B;
dashed, one-sided 95% upper bounds (labelled, ×10⁻³); dotted, τ.

**Fig. 5 | Derived (model-based projection): which estimator is preferred.** Ratio of the worst-coordinate risk
envelope of the entangled mixture to that of product measurement against total copies N, under the fitted
response model, for target-resolved (turquoise) and pooled (magenta) calibration, with pointwise 95% bootstrap
bands (1,000 resamples). Below 1, the mixture is preferred. Kinks mark a change of the coordinate that dominates
the worst-case envelope. The pooled band crosses 1 from N ≈ 8 × 10⁴; simultaneous one-sided 95% bounds on the
worst-case ratio over the range: 0.945 (target-resolved), 1.039 (pooled). Range as declared, 10²–10⁵; no
extrapolation.
"""

if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig4(); fig5()
    print('figures written to', FIGS)
