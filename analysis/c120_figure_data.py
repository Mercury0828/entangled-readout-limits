"""Every number a figure plots, computed from FROZEN data only (data/raw, data/sealed) and theory.

    python c120_figure_data.py        # -> data/selfcheck/figure_data.json; asserts headline numbers

No vendor API is touched. The noiseless simulator for Fig. 3 is run here on the identical circuits.
"""
import json
import os

import numpy as np

import c109_realised_reference as C
import c110_paired_damping as P
import c112_targeted_calibration as T
import c114_sealed as S
import c115_risk_curves as RC

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, 'data', 'raw')
OUT = os.path.join(HERE, 'data', 'selfcheck', 'figure_data.json')
G = 'lab'


def frozen(jid):
    return json.load(open(os.path.join(RAW, f'{jid}.json')))['counts']


def theory():
    eta = np.linspace(0.5, 1, 501)
    r = eta ** 2
    C_chain = np.where(r <= 2 / 3, 9.0, 3 + 4 / r)
    w = np.where(r <= 2 / 3, 1.0, 3 * r / (4 + 3 * r))
    ww = np.linspace(0.2, 0.8, 601)
    Cw = np.maximum(3 / ww, 18 / (3 - ww))
    # two-sided bracket, sparse Pauli-Lindblad noise (rate seed 4)
    KEYS2 = [a + b for a in 'ixyz' for b in 'ixyz']
    anti = lambda p, q: sum(1 for c1, c2 in zip(p, q) if c1 != 'i' and c2 != 'i' and c1 != c2) % 2
    GEN = [a + 'i' for a in 'xyz'] + ['i' + a for a in 'xyz'] + ['xx', 'yy', 'zz']
    L = lambda x: 9.0 if x <= 2 / 3 else 3 + 4 / x
    br = []
    for scale in np.geomspace(0.002, 0.2, 40):
        rng = np.random.default_rng(4)
        rates = {k: scale * (0.5 + rng.random()) for k in GEN}
        f = {P_: float(np.exp(-2 * sum(v for k, v in rates.items() if anti(P_, k)))) for P_ in KEYS2}
        p = {Q: float(sum(((-1) ** anti(P_, Q)) * f[P_] for P_ in KEYS2)) / 16 for Q in KEYS2}
        chi = float(1 - 2 * np.sort([p[k] for k in KEYS2])[:8].sum())
        rB = max((f[a + 'i'] ** 2 + f['i' + b] ** 2 + f[a + b] ** 2) / 3 for a in 'xyz' for b in 'xyz')
        inf = 1 - (4 * sum(f.values()) / 16 + 1) / 5
        br.append((inf, L(chi ** 2), min(9.0, L(rB))))
    return dict(eta=eta.tolist(), C_chain=C_chain.tolist(), w=w.tolist(), ww=ww.tolist(), Cw=Cw.tolist(),
                bracket=br)


def fig3():
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    circuits, meta = C.build_with_realised()
    cnt = frozen('daoa3j8pqrnc7399miv0')
    assert len(cnt) == len(circuits), 'frozen c108 counts do not match the rebuilt schedule'
    dev = P.per_circuit_pairs(lambda i: cnt[i], meta)
    sb = AerSimulator(seed_simulator=110)
    rs = sb.run(transpile(circuits, sb, optimization_level=0), shots=P.M.SHOTS).result()
    sim = P.per_circuit_pairs(lambda i: rs.get_counts(i), meta)
    modes = {}
    for mode in ('prod_single', 'prod_edge', 'bell_w1', 'bell_w2'):
        if len(dev[mode]) < 20:
            continue
        a_d, c_d = P.fit(dev[mode]); a_s, c_s = P.fit(sim[mode])
        sa_d, sc_d = P.boot(dev[mode]); sa_s, _ = P.boot(sim[mode])
        modes[mode] = dict(n=len(dev[mode]), a_dev=a_d, se_dev=sa_d, c_dev=c_d, se_c=sc_d,
                           a_sim=a_s, se_sim=sa_s, z=(a_d - a_s) / np.hypot(sa_d, sa_s))
    pts = [(x, y) for x, y, _ in dev['prod_edge']]
    spts = [(x, y) for x, y, _ in sim['prod_edge']]
    # c112: target-resolved modes from the frozen job
    c2, m2 = T.build()
    cnt2 = frozen('daobh65r85ps73ff233g')
    assert len(cnt2) == len(c2)
    rows = {name: [] for name, *_ in T.MODES}
    for idx, m in enumerate(m2):
        rows[m[0]].append((m[4], T.observe(m, cnt2[idx])))
    rng = np.random.default_rng(1121)
    tr = {}
    for name, arr in rows.items():
        arr = np.array(arr, float)
        a, c = T.fit(arr[:, 0], arr[:, 1])
        bs = [T.fit(*arr[rng.integers(0, len(arr), len(arr))].T)[0] for _ in range(400)]
        tr[name] = dict(n=len(arr), a=a, se=float(np.std(bs, ddof=1)), c=c)
    return dict(c110=modes, c110_edge_points_dev=pts, c110_edge_points_sim=spts, c112=tr)


def fig4():
    circuits, meta = S.build()
    raw = json.load(open(S.RAW)); pred = json.load(open(S.PRED))
    assert raw['meta_sha256'] == S.meta_sha(meta) and pred['raw_sha256'] == S.sha(S.RAW)
    ys = [S.observe(m, c) for m, c in zip(meta, raw['counts'])]
    ev = {}
    for m, y in zip(meta, ys):
        if m['role'] == 'eval':
            ev.setdefault(m['mode'], []).append((m['x'], y))
    ev = {k: np.array(v, float) for k, v in ev.items()}
    beta = lambda a_e, c_e, a_h, c_h: (a_e / a_h - 1) * S.THETA_REF + (c_e - c_h) / a_h
    rng = np.random.default_rng(2)                  # the pre-registered evaluation seed
    out = {}
    for model in ('target', 'pooled'):
        pt = {md: pred['point'][f'{model}|{md}'] for md in ev}
        eps = {md: S.fit(ev[md]) for md in ev}
        per = {md: beta(*eps[md], *pt[md]) for md in ev}
        Bs, perb = [], {md: [] for md in ev}
        for b in pred['boot']:
            e_b = {md: S.fit(v[rng.integers(0, len(v), len(v))]) for md, v in ev.items()}
            vals = {md: beta(*e_b[md], *b[f'{model}|{md}']) for md in ev}
            for md in ev:
                perb[md].append(vals[md])
            Bs.append(max(abs(v) for v in vals.values()))
        out[model] = dict(per_mode=per, per_mode_lo=({md: float(np.percentile(perb[md], 2.5)) for md in ev}),
                          per_mode_hi=({md: float(np.percentile(perb[md], 97.5)) for md in ev}),
                          B=max(abs(v) for v in per.values()), UB95=float(np.percentile(Bs, 95)), Bboot=Bs)
    assert abs(out['target']['UB95'] - 0.00550) < 5e-6 and abs(out['pooled']['UB95'] - 0.00997) < 5e-6, \
        (out['target']['UB95'], out['pooled']['UB95'])
    return out


if __name__ == '__main__':
    rc = json.load(open(os.path.join(HERE, 'data', 'selfcheck', 'c130_exact_mse.json')))   # exact model-internal MSE (c130)
    fd = dict(theory=theory(), fig3=fig3(), fig4=fig4(), fig5=rc, tau=S.TAU)
    json.dump(fd, open(OUT, 'w'), default=float)
    f3 = fd['fig3']
    for k, v in f3['c110'].items():
        print(f"  c110 {k:<12s} n={v['n']:5d}  a_dev={v['a_dev']:.5f}+-{v['se_dev']:.5f}  a_sim={v['a_sim']:.5f}  z={v['z']:+.1f}")
    for k, v in f3['c112'].items():
        print(f"  c112 {k:<14s} n={v['n']:5d}  a={v['a']:.5f}+-{v['se']:.5f}")
    for m in ('target', 'pooled'):
        print(f"  c114 {m:<7s} B={fd['fig4'][m]['B']:.5f}  UB95={fd['fig4'][m]['UB95']:.5f}")
    print('figure data frozen:', OUT)
