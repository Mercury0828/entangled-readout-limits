"""Two post-hoc checks of the prospective test (Supplementary Note 19); the pre-registered result is unchanged.

    python c128_endpoint_checks.py      # -> data/selfcheck/c128_endpoint_checks.json

(1) Worst bias over the amplitude interval |theta| <= 0.5: sup |beta(theta)| = |d_c| + 0.5 |d_a| with
    d_a = a_eval/a_hat - 1, d_c = (c_eval - c_hat)/a_hat; endpoint = max over the five targets; one-sided 95% upper bound
    from the same joint bootstrap draws as c114 (evaluation seed 2).
(2) The simultaneous bound over the continuous range 1e2 <= N <= 1e5. With t = 1/N every coordinate risk is a line
    nu t + b^2, each envelope is a maximum of lines, and the ratio of two such envelopes on a segment with fixed active
    lines is linear-fractional, hence monotone, so its maximum is attained at an endpoint or at a breakpoint of either
    envelope; these are evaluated exactly (max_ratio).
"""
import json
import os

import numpy as np

import c114_sealed as S
import c115_risk_curves as RC

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    circuits, meta = S.build()
    raw = json.load(open(S.RAW)); pred = json.load(open(S.PRED))
    assert raw['meta_sha256'] == S.meta_sha(meta) and pred['raw_sha256'] == S.sha(S.RAW)
    ys = [S.observe(m, c) for m, c in zip(meta, raw['counts'])]
    ev = {}
    for m, y in zip(meta, ys):
        if m['role'] == 'eval':
            ev.setdefault(m['mode'], []).append((m['x'], y))
    return pred, {k: np.array(v, float) for k, v in ev.items()}


def sup_bias(a_e, c_e, a_h, c_h, amp=0.5):
    return abs((c_e - c_h) / a_h) + amp * abs(a_e / a_h - 1)


def interval_endpoint(pred, ev):
    rng = np.random.default_rng(2)                  # same draws as the pre-registered evaluation
    out = {}
    for model in ('target', 'pooled'):
        pt = {md: pred['point'][f'{model}|{md}'] for md in ev}
        eps = {md: S.fit(ev[md]) for md in ev}
        B = max(sup_bias(*eps[md], *pt[md]) for md in ev)
        Bs = []
        for b in pred['boot']:
            e_b = {md: S.fit(v[rng.integers(0, len(v), len(v))]) for md, v in ev.items()}
            Bs.append(max(sup_bias(*e_b[md], *b[f'{model}|{md}']) for md in ev))
        out[model] = dict(B=B, UB95=float(np.percentile(Bs, 95)), certified=bool(np.percentile(Bs, 95) < S.TAU))
    return out


def lines(ev, pr):
    """product and mixture risk lines (slope nu, intercept b^2) for each measured coordinate"""
    P, M = [], []
    for name, (pm, bm) in RC.COORDS.items():
        nu_p, b_p = RC.arm(ev, pr, pm)
        P.append((nu_p, b_p ** 2))
        nu_pw = nu_p / RC.W_MIX
        if bm is None:
            M.append((nu_pw, b_p ** 2))
        else:
            nu_b, b_b = RC.arm(ev, pr, bm)
            nu_bw = nu_b / (1 - RC.W_MIX)
            ip, ib = 1 / nu_pw, 1 / nu_bw
            M.append((1 / (ip + ib), ((ip * b_p + ib * b_b) / (ip + ib)) ** 2))
    return np.array(P), np.array(M)


def max_ratio(P, M, t_lo=1e-5, t_hi=1e-2):
    cand = [t_lo, t_hi]
    for L in (P, M):
        for i in range(len(L)):
            for j in range(i + 1, len(L)):
                ds = L[i, 0] - L[j, 0]
                if ds != 0:
                    t = (L[j, 1] - L[i, 1]) / ds
                    if t_lo < t < t_hi:
                        cand.append(t)
    cand = np.array(cand)
    rp = (P[:, :1] * cand + P[:, 1:]).max(axis=0)
    rm = (M[:, :1] * cand + M[:, 1:]).max(axis=0)
    return float((rm / rp).max())


def continuous_bound(pred, ev):
    evfit = {k: S.fit(v) for k, v in ev.items()}
    rng = np.random.default_rng(1150)                 # same evaluation draws as c115
    evboot = [{k: S.fit(v[rng.integers(0, len(v), len(v))]) for k, v in ev.items()} for _ in range(len(pred['boot']))]
    out = {}
    for model in ('target', 'pooled'):
        grid = []
        cont = []
        for b, eb in zip(pred['boot'], evboot):
            prb = {m: tuple(b[f'{model}|{m}']) for m in RC.NU}
            P, M = lines(eb, prb)
            cont.append(max_ratio(P, M))
            t = 1 / RC.NGRID
            grid.append(float(((M[:, :1] * t + M[:, 1:]).max(axis=0) / (P[:, :1] * t + P[:, 1:]).max(axis=0)).max()))
        out[model] = dict(grid_sim_hi=float(np.percentile(grid, 95)), continuous_sim_hi=float(np.percentile(cont, 95)))
    return out


if __name__ == '__main__':
    pred, ev = load()
    res = dict(interval_endpoint=interval_endpoint(pred, ev), continuous=continuous_bound(pred, ev), tau=S.TAU)
    json.dump(res, open(os.path.join(HERE, 'data', 'selfcheck', 'c128_endpoint_checks.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))
