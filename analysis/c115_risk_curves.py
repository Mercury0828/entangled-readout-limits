"""Fixed design coefficients and probe definitions for the Fig. 5 projection; earlier second-moment version.

    python c115_risk_curves.py      # -> data/sealed/c115_risk_curves.json

Defines the coordinates, the per-copy design second moments NU = 1/p, the fixed mixing weight and the evaluation
grid used by c130_exact_mse_projection.py, which computes the reported projection with the exact mean squared error
of the full estimator. The envelopes computed here use the design second moment divided by a_hat^2 as the
variance, an upper-bound proxy that the paper does not report.
"""
import json
import numpy as np

import c114_sealed as S

W_MIX = 0.4172
THETA = S.THETA_REF
NGRID = np.logspace(2, 5, 31)
NU = {'prod_single_z': 3.0, 'prod_edge_zz': 9.0, 'prod_edge_yy': 9.0,
      'bell_w1_zz': 6.0, 'bell_w2_yy': 6.0}
COORDS = {'single z':  ('prod_single_z', None),
          'edge zz':   ('prod_edge_zz', 'bell_w1_zz'),
          'edge yy':   ('prod_edge_yy', 'bell_w2_yy')}


def arm(ev, pr, mode):
    a_e, c_e = ev[mode]; a_h, c_h = pr[mode]
    beta = (a_e / a_h - 1) * THETA + (c_e - c_h) / a_h
    nu = NU[mode] / a_h ** 2                     # correction rescales the per-shot variance
    return nu, beta


def envelopes(ev, pr):
    """R_product(N), R_mixture(N) and which coordinate dominates, on NGRID"""
    Rp, Rm, dp, dm = [], [], [], []
    for N in NGRID:
        rp, rm = {}, {}
        for name, (pm, bm) in COORDS.items():
            nu_p, b_p = arm(ev, pr, pm)
            rp[name] = nu_p / N + b_p ** 2                      # product only, w = 1
            nu_pw = nu_p / W_MIX                                 # product share of the mixture
            if bm is None:
                rm[name] = nu_pw / N + b_p ** 2                  # Bell arm carries no information
            else:
                nu_b, b_b = arm(ev, pr, bm)
                nu_bw = nu_b / (1 - W_MIX)
                ip, ib = 1 / nu_pw, 1 / nu_bw                    # inverse-variance combination
                rm[name] = 1 / (ip + ib) / N + ((ip * b_p + ib * b_b) / (ip + ib)) ** 2
        kp = max(rp, key=rp.get); km = max(rm, key=rm.get)
        Rp.append(rp[kp]); Rm.append(rm[km]); dp.append(kp); dm.append(km)
    return np.array(Rp), np.array(Rm), dp, dm


def run(model, pred, evfit, evboot):
    pr = {m: tuple(pred['point'][f'{model}|{m}']) for m in NU}
    Rp, Rm, dp, dm = envelopes(evfit, pr)
    bp, bm = [], []
    for b, eb in zip(pred['boot'], evboot):
        prb = {m: tuple(b[f'{model}|{m}']) for m in NU}
        x, y, _, _ = envelopes(eb, prb)
        bp.append(x); bm.append(y)
    bp, bm = np.array(bp), np.array(bm)
    ratio = bm / bp                                              # < 1 means mixture preferred
    # SIMULTANEOUS over the declared range: the pointwise bands certify each N on its own;
    # "resolved throughout" needs the bound on the WORST N of each bootstrap draw.
    sim_hi = float(np.percentile(ratio.max(axis=1), 95))
    return dict(Rp=Rp, Rm=Rm, dp=dp, dm=dm,
                ratio=Rm / Rp, ratio_lo=np.percentile(ratio, 2.5, axis=0),
                ratio_hi=np.percentile(ratio, 97.5, axis=0), sim_hi=sim_hi)


if __name__ == '__main__':
    circuits, meta = S.build()
    raw = json.load(open(S.RAW)); pred = json.load(open(S.PRED))
    assert raw['meta_sha256'] == S.meta_sha(meta), 'schedule mismatch'
    assert pred['raw_sha256'] == S.sha(S.RAW), 'raw data changed'
    ys = [S.observe(m, c) for m, c in zip(meta, raw['counts'])]
    ev = {}
    for m, y in zip(meta, ys):
        if m['role'] == 'eval':
            ev.setdefault(m['mode'], []).append((m['x'], y))
    ev = {k: np.array(v, float) for k, v in ev.items()}
    evfit = {k: S.fit(v) for k, v in ev.items()}
    rng = np.random.default_rng(1150)
    evboot = [{k: S.fit(v[rng.integers(0, len(v), len(v))]) for k, v in ev.items()}
              for _ in range(len(pred['boot']))]

    out = {}
    for model in ('target', 'pooled'):
        r = run(model, pred, evfit, evboot)
        out[model] = r
        print('=' * 100)
        print(f'CONDITIONAL RISK PROJECTION -- {model.upper()} calibration, sealed c114 data, '
              f'w_mix = {W_MIX}')
        print('   MODEL-BASED PROJECTION, not a measured MSE. Reference-amplitude convention (theta_ref = +0.5).')
        print('=' * 100)
        print('    N = TOTAL COPIES: every copy is measured on every qubit and informs every coordinate.')
        print(f"    {'N':>8s} | {'R_product':>11s} {'R_mixture':>11s} | {'ratio':>7s} "
              f"{'95% band':>17s} | {'dominant (prod / mix)':>24s} | preferred")
        for i in range(0, len(NGRID), 5):
            N = NGRID[i]
            lo, hi = r['ratio_lo'][i], r['ratio_hi'][i]
            pref = ('MIXTURE (resolved)' if hi < 1 else
                    'product (resolved)' if lo > 1 else 'not resolved')
            print(f"    {N:8.0f} | {r['Rp'][i]:11.3e} {r['Rm'][i]:11.3e} | {r['ratio'][i]:7.4f} "
                  f"[{lo:.4f}, {hi:.4f}] | {r['dp'][i]:>11s} / {r['dm'][i]:<10s} | {pref}")
        resolved_all = bool(np.all(r['ratio_hi'] < 1))
        print(f"    pointwise: mixture resolved at every N in [1e2, 1e5]: {resolved_all}")
        print(f"    SIMULTANEOUS one-sided 95% bound on max over N of the ratio: {r['sim_hi']:.4f}  -> "
              f"{'RESOLVED throughout, simultaneously' if r['sim_hi'] < 1 else 'NOT resolved simultaneously'}")
        print(f"    advantage at N = 1e2: {100*(1-r['ratio'][0]):.1f}%   at N = 1e5: "
              f"{100*(1-r['ratio'][-1]):.1f}%")
    json.dump({m: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
               for m, r in out.items()} | {'N': NGRID.tolist()},
              open(S.DATA + '/c115_risk_curves.json', 'w'))
    print('=' * 100)
