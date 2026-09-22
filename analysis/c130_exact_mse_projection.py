"""Fig. 5: exact model-internal mean squared error of the full estimators (Methods; Supplementary Note 14).

    python c130_exact_mse_projection.py      # -> data/selfcheck/c130_exact_mse.json

Under the fitted response model (mean outcome a_e * theta + c_e on the evaluation circuits, calibrated response
(a_hat, c_hat)), each arm's per-copy estimator is X = 1[covered]/p * (s - c_hat)/a_hat with s = +-1,
E s = mu = a_e theta + c_e, and coverage probability p (product: 1/3 single site, 1/9 edge; Bell arm: 1/6). Exactly:
    E X = (mu - c_hat)/a_hat,   E X^2 = (1 - 2 c_hat mu + c_hat^2)/(p a_hat^2),   bias = E X - theta.
The mixture uses w N product copies and (1 - w) N Bell copies and combines the arm means with fixed design
coefficients gamma = i_p/(i_p + i_b), i = arm share * p * a_hat^2 (c115). Everything is evaluated at the reference
amplitude theta_ref = +0.5 of each target's response model. Each coordinate's MSE is A/N + B^2, and the
continuous-range maximum of the ratio is computed exactly as in c128.
"""
import json
import os

import numpy as np

import c114_sealed as S
import c115_risk_curves as RC
from c128_endpoint_checks import load, max_ratio

HERE = os.path.dirname(os.path.abspath(__file__))
TH = S.THETA_REF
P_COVER = {'prod_single_z': 1 / 3, 'prod_edge_zz': 1 / 9, 'prod_edge_yy': 1 / 9, 'bell_w1_zz': 1 / 6,
           'bell_w2_yy': 1 / 6}


def arm_exact(ev, pr, mode):
    a_e, c_e = ev[mode]; a_h, c_h = pr[mode]
    mu = a_e * TH + c_e
    m1 = (mu - c_h) / a_h
    m2 = (1 - 2 * c_h * mu + c_h ** 2) / (P_COVER[mode] * a_h ** 2)
    return m2 - m1 ** 2, m1 - TH                          # per-copy variance, bias


def lines_exact(ev, pr):
    P, M = [], []
    for name, (pm, bm) in RC.COORDS.items():
        v_p, b_p = arm_exact(ev, pr, pm)
        P.append((v_p, b_p ** 2))                        # product only: all N copies in the product arm
        if bm is None:
            M.append((v_p / RC.W_MIX, b_p ** 2))
        else:
            v_b, b_b = arm_exact(ev, pr, bm)
            nu_p, _ = RC.arm(ev, pr, pm); nu_b, _ = RC.arm(ev, pr, bm)
            ip, ib = RC.W_MIX / nu_p, (1 - RC.W_MIX) / nu_b          # the fixed combination rule of c115
            g = ip / (ip + ib)
            M.append((g ** 2 * v_p / RC.W_MIX + (1 - g) ** 2 * v_b / (1 - RC.W_MIX),
                      (g * b_p + (1 - g) * b_b) ** 2))
    return np.array(P), np.array(M)


def envelope_ratio(P, M, N):
    t = 1 / np.asarray(N)
    return (M[:, :1] * t + M[:, 1:]).max(axis=0) / (P[:, :1] * t + P[:, 1:]).max(axis=0)


if __name__ == '__main__':
    pred, ev = load()
    evfit = {k: S.fit(v) for k, v in ev.items()}
    rng = np.random.default_rng(1150)
    evboot = [{k: S.fit(v[rng.integers(0, len(v), len(v))]) for k, v in ev.items()} for _ in range(len(pred['boot']))]
    N = RC.NGRID
    out = {'N': N.tolist()}
    for model in ('target', 'pooled'):
        pr = {m: tuple(pred['point'][f'{model}|{m}']) for m in RC.NU}
        P, M = lines_exact(evfit, pr)
        ratio = envelope_ratio(P, M, N)
        boots, sims = [], []
        for b, eb in zip(pred['boot'], evboot):
            prb = {m: tuple(b[f'{model}|{m}']) for m in RC.NU}
            Pb, Mb = lines_exact(eb, prb)
            boots.append(envelope_ratio(Pb, Mb, N))
            sims.append(max_ratio(Pb, Mb))
        boots = np.array(boots)
        out[model] = dict(ratio=ratio.tolist(), ratio_lo=np.percentile(boots, 2.5, axis=0).tolist(),
                          ratio_hi=np.percentile(boots, 97.5, axis=0).tolist(),
                          sim_hi=float(np.percentile(sims, 95)),
                          adv_1e2=float(1 - ratio[0]), adv_1e5=float(1 - ratio[-1]),
                          first_band_cross=(float(N[np.argmax(np.percentile(boots, 97.5, axis=0) > 1)])
                                            if (np.percentile(boots, 97.5, axis=0) > 1).any() else None))
        print(model, {k: out[model][k] for k in ('sim_hi', 'adv_1e2', 'adv_1e5', 'first_band_cross')})
    # sensitivity to the mixing weight, same exact MSE (Supplementary Table on w)
    w0 = RC.W_MIX
    out['w_mix'] = {}
    for w in (0.30, 0.35, 0.4172, 0.45, 0.50):
        RC.W_MIX = w
        out['w_mix'][w] = {}
        for model in ('target', 'pooled'):
            sims = []
            for b, eb in zip(pred['boot'], evboot):
                prb = {m: tuple(b[f'{model}|{m}']) for m in RC.NU}
                sims.append(max_ratio(*lines_exact(eb, prb)))
            out['w_mix'][w][model] = float(np.percentile(sims, 95))
        print('w', w, out['w_mix'][w])
    RC.W_MIX = w0
    json.dump(out, open(os.path.join(HERE, 'data', 'selfcheck', 'c130_exact_mse.json'), 'w'), indent=1)
