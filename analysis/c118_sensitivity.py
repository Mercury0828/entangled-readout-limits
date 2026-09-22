"""Post-hoc sensitivity analysis of the prospective test (c114); secondary to the pre-registered result.

    python c118_sensitivity.py      # -> data/selfcheck/c118_sensitivity.json

The pre-registered verdict is the one computed by `c114_sealed.py evaluate` with the committed predictions, seed 2,
tau = 0.0085 and theta_ref = 0.5. Varied here, after unsealing: the evaluation bootstrap seed, the prediction
bootstrap seed (predictions refitted from the same calibration circuits), theta_ref and the tolerance.
"""
import json
import os
import sys

import numpy as np

import c114_sealed as S
import c115_risk_curves as RC

OUT = os.path.join(S.HERE, 'data', 'selfcheck')


def load():
    circuits, meta = S.build()
    raw = json.load(open(S.RAW)); pred = json.load(open(S.PRED))
    assert raw['meta_sha256'] == S.meta_sha(meta), 'schedule mismatch'
    assert pred['raw_sha256'] == S.sha(S.RAW), 'raw data changed'
    ys = [S.observe(m, c) for m, c in zip(meta, raw['counts'])]
    return meta, ys, pred


def verdict(meta, ys, pred, seed):
    r = S.evaluate(meta, ys, pred, np.random.default_rng(seed))
    return {m: dict(B=r[m]['B'], UB95=r[m]['UB95'], adequate=bool(r[m]['adequate'])) for m in r}


if __name__ == '__main__':
    meta, ys, pred = load()
    out = {}

    base = verdict(meta, ys, pred, 2)
    out['preregistered'] = base
    assert base['target']['adequate'] and not base['pooled']['adequate'], 'pre-registered verdict not reproduced'

    # (1) evaluation bootstrap seed
    out['eval_seed'] = {s: verdict(meta, ys, pred, s) for s in (3, 4, 5, 6, 7)}

    # (2) prediction bootstrap seed -- refit from the SAME calibration circuits
    out['pred_seed'] = {}
    for s in (11, 12, 13):
        p2 = S.predict(meta, ys, np.random.default_rng(s))
        out['pred_seed'][s] = verdict(meta, ys, p2, 2)

    # (3) theta_ref
    out['theta_ref'] = {}
    t0 = S.THETA_REF
    for th in (0.25, 0.75, 1.0):
        S.THETA_REF = th
        out['theta_ref'][th] = verdict(meta, ys, pred, 2)
    S.THETA_REF = t0

    # (4) risk curves under w_mix
    out['w_mix'] = {}
    ev = {}
    for m, y in zip(meta, ys):
        if m['role'] == 'eval':
            ev.setdefault(m['mode'], []).append((m['x'], y))
    ev = {k: np.array(v, float) for k, v in ev.items()}
    evfit = {k: S.fit(v) for k, v in ev.items()}
    rng = np.random.default_rng(1150)
    evboot = [{k: S.fit(v[rng.integers(0, len(v), len(v))]) for k, v in ev.items()}
              for _ in range(len(pred['boot']))]
    w0 = RC.W_MIX
    for w in (0.35, 0.3, 0.4172, 0.45, 0.5):
        RC.W_MIX = w
        out['w_mix'][w] = {m: dict(sim_hi=RC.run(m, pred, evfit, evboot)['sim_hi']) for m in ('target', 'pooled')}
    RC.W_MIX = w0

    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, 'c118_sensitivity.json'), 'w'), indent=1, default=float)

    print('=' * 96)
    print('POST-HOC SENSITIVITY (secondary; the pre-registered verdict is unchanged by construction)')
    print('=' * 96)

    def row(label, v):
        t, p = v['target'], v['pooled']
        print(f"  {label:<28s} target UB95 {t['UB95']:.5f} ({'adequate' if t['UB95'] < S.TAU else 'NOT'})"
              f"   pooled UB95 {p['UB95']:.5f} ({'adequate' if p['UB95'] < S.TAU else 'not demonstrated'})")

    row('pre-registered (seed 2)', base)
    for s, v in out['eval_seed'].items():
        row(f'eval bootstrap seed {s}', v)
    for s, v in out['pred_seed'].items():
        row(f'prediction bootstrap seed {s}', v)
    for th, v in out['theta_ref'].items():
        row(f'theta_ref = {th}', v)
    all_rows = [base] + list(out['eval_seed'].values()) + list(out['pred_seed'].values())
    tmax = max(v['target']['UB95'] for v in all_rows)
    pmin = min(v['pooled']['UB95'] for v in all_rows)
    print(f'  tau window over which the verdict pair holds (seed variations, theta_ref = 0.5): '
          f'({tmax:.5f}, {pmin:.5f});  declared tau = {S.TAU}')
    for w, v in out['w_mix'].items():
        print(f"  w_mix = {w:<7}  simultaneous bound: target {v['target']['sim_hi']:.4f}   "
              f"pooled {v['pooled']['sim_hi']:.4f}")
    print('=' * 96)
    stable = all(v['target']['adequate'] and not v['pooled']['adequate'] for v in all_rows)
    print(f'verdict pair stable across seed variations: {stable}')
    sys.exit(0 if stable else 1)
