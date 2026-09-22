"""Supplementary Figures 1-4, from frozen data and closed forms only.

    python c124_si_figures.py        # -> figs/si_fig{1..4}.pdf/.png ; data/selfcheck/si_figure_data.json

S1  engineered-noise response (parsed from data/selfcheck/c106_frozen.txt, which c123 regenerates from raw counts)
S2  per-coordinate risk ratio, both calibration policies (sealed c114 data, exact model-internal MSE, c130)
S3  worst-coordinate constant C(w) at several gate qualities (closed form)
S4  worst-coordinate versus average-coordinate constants within the product/Bell family (closed form)
"""
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
from scipy.optimize import minimize_scalar

import c114_sealed as S
import c115_risk_curves as RC

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, 'figs')
MM = 1 / 25.4
MAG, TUR, GRY, BLK = '#C2185B', '#00897B', '#8C8C8C', '#000000'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.linewidth': 0.8, 'lines.linewidth': 1.2,
                     'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False, 'axes.formatter.use_mathtext': False})
SUP = str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻')
OUT = {}


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name + '.pdf'))
    fig.savefig(os.path.join(FIGS, name + '.png'), dpi=300)
    plt.close(fig)


def panel(ax, k):
    ax.set_title(k, loc='left', fontsize=8, fontweight='bold', pad=3)


def s1():
    txt = open(os.path.join(HERE, 'data', 'selfcheck', 'c106_frozen.txt'), encoding='utf-8').read()
    blockA = txt.split('(A,B)')[1].split('mean ratio')[0]
    blockD = txt.split('(D) TESTING')[1].split('|z| max')[0]
    rows = [tuple(map(float, re.findall(r'[-+]?\d+\.\d+', l)[:3])) for l in blockA.splitlines()
            if re.match(r'\s+0\.\d{4} \|', l)]
    rowsD = [tuple(map(float, re.findall(r'[-+]?\d+\.\d+', l)[:3])) for l in blockD.splitlines()
             if re.match(r'\s+0\.\d{4} \|', l)]
    eta = np.array([r[0] for r in rows]); sl = np.array([r[1] for r in rows]); se = np.array([r[2] for r in rows])
    w1 = np.array([r[1] for r in rowsD]); w2 = np.array([r[2] for r in rowsD])
    assert len(eta) == 9 and len(w1) == 9
    k = float((sl * eta / se ** 2).sum() / (eta * eta / se ** 2).sum())
    assert abs(k - 0.99925) < 5e-5, k
    OUT['s1'] = dict(eta=eta.tolist(), slope=sl.tolist(), se=se.tolist(), w1=w1.tolist(), w2=w2.tolist(), k=k)
    fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 62 * MM), layout='constrained')
    ax = axs[0]
    ax.errorbar(eta, sl, yerr=1.96 * se, fmt='o', ms=3.5, color=TUR, label='fitted Bell response')
    xx = np.array([0.68, 1.0])
    ax.plot(xx, k * xx, color=BLK, lw=1.0, label='proportional fit')
    ax.plot(xx, xx, color=GRY, lw=0.8, ls=':', label='response equal to injected η')
    ax.set_xlabel('injected gate quality η'); ax.set_ylabel('Bell response slope')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2); panel(ax, 'a')
    ax = axs[1]
    ax.plot(eta, w1 / eta, 'o', ms=3.5, color=TUR, label='one-qubit readout')
    ax.plot(eta, w2 / eta, 's', ms=3.5, mfc='white', mec=MAG, label='parity readout')
    ax.axhline(1.0, color=GRY, lw=0.8, ls=':')
    ax.set_xlabel('injected gate quality η'); ax.set_ylabel('response slope / injected η')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2); panel(ax, 'b')
    save(fig, 'si_fig1')


def s2():
    circuits, meta = S.build()
    raw = json.load(open(S.RAW)); pred = json.load(open(S.PRED))
    assert raw['meta_sha256'] == S.meta_sha(meta) and pred['raw_sha256'] == S.sha(S.RAW)
    ys = [S.observe(m, c) for m, c in zip(meta, raw['counts'])]
    ev = {}
    for m, y in zip(meta, ys):
        if m['role'] == 'eval':
            ev.setdefault(m['mode'], []).append((m['x'], y))
    ev = {k: np.array(v, float) for k, v in ev.items()}
    evfit = {k: S.fit(v) for k, v in ev.items()}
    N = RC.NGRID
    fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 62 * MM), layout='constrained', sharey=True)
    OUT['s2'] = {'N': N.tolist()}
    for ax, model, lab in ((axs[0], 'target', 'a'), (axs[1], 'pooled', 'b')):
        pr = {m: tuple(pred['point'][f'{model}|{m}']) for m in RC.NU}
        OUT['s2'][model] = {}
        styles = {'single z': (MAG, '--'), 'edge zz': (TUR, '-'), 'edge yy': (BLK, ':')}
        import c130_exact_mse_projection as X
        P_, M_ = X.lines_exact(evfit, pr)
        for k, (name, (pm, bm)) in enumerate(RC.COORDS.items()):
            rp = P_[k, 0] / N + P_[k, 1]
            rm = M_[k, 0] / N + M_[k, 1]
            OUT['s2'][model][name] = (rm / rp).tolist()
            col, ls = styles[name]
            ax.plot(N, rm / rp, color=col, ls=ls, label=name.replace('single z', 'single site Z')
                    .replace('edge zz', 'edge ZZ').replace('edge yy', 'edge YY'))
        ax.axhline(1, color=GRY, lw=0.8, ls=':')
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: '10' + str(int(round(np.log10(v)))).translate(SUP)))
        ax.xaxis.set_minor_formatter(mt.NullFormatter())
        ax.set_xlabel('estimation copies N')
        panel(ax, lab)
    axs[0].set_ylabel('risk ratio per coordinate, mixture / product')
    fig.legend(*axs[0].get_legend_handles_labels(), loc='outside upper center', ncol=3)
    save(fig, 'si_fig2')


def Cw(w, r):
    return np.maximum(3 / w, 1 / (w / 9 + (1 - w) * r / 6))


def s3():
    fig, ax = plt.subplots(figsize=(88 * MM, 62 * MM), layout='constrained')
    w = np.linspace(0.15, 1, 800)
    OUT['s3'] = {'w': w.tolist()}
    for eta, col, ls in ((1.0, TUR, '-'), (0.95, BLK, '--'), (0.9, MAG, '-.'), (0.85, GRY, ':')):
        c = np.where(Cw(w, eta * eta) <= 14, Cw(w, eta * eta), np.nan)
        OUT['s3'][str(eta)] = c.tolist()
        ax.plot(w, c, color=col, ls=ls, label=f'η = {eta:g}')
        ws = 3 * eta ** 2 / (4 + 3 * eta ** 2)
        ax.plot([ws], [3 / ws], 'o', ms=3, color=col)
    ax.set_ylim(6.5, 14); ax.set_xlabel('product weight w'); ax.set_ylabel('worst-coordinate constant')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2)
    save(fig, 'si_fig3')


def avg_C(w, r):
    return (3 * 3 / w + 9 / (w / 9 + (1 - w) * r / 6)) / 12


def s4():
    etas = np.linspace(0.75, 1.0, 251)
    worst, avg, wavg = [], [], []
    for e in etas:
        r = e * e
        worst.append(9.0 if r <= 2 / 3 else 3 + 4 / r)
        res = minimize_scalar(lambda w: avg_C(w, r), bounds=(1e-6, 1), method='bounded')
        a = min(res.fun, avg_C(1.0, r))
        avg.append(a)
    worst, avg = np.array(worst), np.array(avg)
    gain_w, gain_a = 1 - worst / 9, 1 - avg / 7.5
    thr = etas[np.argmax(gain_a > 1e-6)]
    OUT['s4'] = dict(eta=etas.tolist(), gain_worst=gain_w.tolist(), gain_avg=gain_a.tolist(),
                     gain_worst_eta1=float(gain_w[-1]), gain_avg_eta1=float(gain_a[-1]), avg_threshold=float(thr))
    assert abs(gain_a[-1] - 0.0838) < 1e-3 and abs(thr - 0.8607) < 2e-3
    fig, ax = plt.subplots(figsize=(88 * MM, 62 * MM), layout='constrained')
    ax.plot(etas, 100 * gain_w, color=TUR, label='worst coordinate (minimax)')
    ax.plot(etas, 100 * gain_a, color=MAG, ls='--', label='average coordinate (this family)')
    ax.axvline(np.sqrt(2 / 3), color=GRY, lw=0.8, ls=':')
    ax.set_xlabel('gate quality η'); ax.set_ylabel('saving against product (%)')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=1)
    save(fig, 'si_fig4')


def s5():
    """fitted response slope of every calibration combination of the prospective test (calibration circuits only)"""
    circuits, meta = S.build()
    raw = json.load(open(S.RAW))
    assert raw['meta_sha256'] == S.meta_sha(meta)
    groups = {}
    for m, c in zip(meta, raw['counts']):
        if m['role'] != 'cal':
            continue
        groups.setdefault(S.key_str(m['fam'], m['key']), []).append((m['x'], S.observe(m, c)))
    fits, cls = {}, {}
    for k, v in groups.items():
        fam, key = json.loads(k)
        key = tuple(tuple(z) if isinstance(z, list) else z for z in key)
        fits[k] = S.fit(np.array(v, float))[0]
        cls[k] = S.pool_class(fam, key)
    targets = {S.key_str(f, k) for _, f, k in S.TARGETS}
    order = ['S', 'E', 'Bw1', 'Bw2']
    names = {'S': 'product, one site', 'E': 'product, edge', 'Bw1': 'Bell, one-qubit', 'Bw2': 'Bell, parity'}
    fig, ax = plt.subplots(figsize=(180 * MM, 66 * MM), layout='constrained')
    x0, OUT['s5'] = 0, {}
    for c in order:
        ks = sorted(k for k in fits if cls[k] == c)
        vals = np.array([fits[k] for k in ks])
        xs = np.arange(x0, x0 + len(ks))
        ax.plot(xs, vals, 'o', ms=2.5, color=GRY, label='combination (80 circuits)' if c == 'S' else None)
        ax.plot([xs[0] - 0.4, xs[-1] + 0.4], [vals.mean()] * 2, color=BLK, lw=1.0,
                label='class mean (pooled model)' if c == 'S' else None)
        for k, xx, vv in zip(ks, xs, vals):
            if k in targets:
                ax.plot([xx], [vv], 's', ms=4, color=TUR, label='target (1,200 circuits)' if c == 'S' else None)
        ax.text(xs.mean(), 0.905, names[c], ha='center', va='bottom')
        OUT['s5'][c] = dict(n=len(ks), mean=float(vals.mean()), sd=float(vals.std(ddof=1)),
                            min=float(vals.min()), max=float(vals.max()))
        x0 += len(ks) + 4
    ax.set_xticks([]); ax.set_ylim(0.90, 1.02); ax.set_ylabel('fitted response slope a')
    ax.set_xlabel('calibration combination, grouped by pooled class')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3)
    save(fig, 'si_fig5')


if __name__ == '__main__':
    s1(); s2(); s3(); s4(); s5()
    print(OUT['s5'])
    json.dump(OUT, open(os.path.join(HERE, 'data', 'selfcheck', 'si_figure_data.json'), 'w'))
    print('k =', round(OUT['s1']['k'], 5), '| avg gain at eta=1:', round(OUT['s4']['gain_avg_eta1'], 4),
          '| avg threshold:', round(OUT['s4']['avg_threshold'], 4))
