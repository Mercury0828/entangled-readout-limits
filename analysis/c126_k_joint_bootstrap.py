"""Statistical error of the engineered-noise constant k with the SHARED product reference resampled jointly.

    python c126_k_joint_bootstrap.py        # -> data/selfcheck/c126_k_joint.json

c106 treats the nine slopes as independent. They share the same 2,400 product-reference circuits, so each
bootstrap draw here resamples the product reference once and every Bell block independently, and recomputes k
with the published inverse-variance weights. Frozen counts only (data/raw).
"""
import json
import os

import numpy as np

import c106_recalibration as C
import c101_threshold_measurement as T

HERE = os.path.dirname(os.path.abspath(__file__))
NB = 1000

if __name__ == '__main__':
    frozen = json.load(open(os.path.join(HERE, 'data', 'raw', C.JOB_C101 + '.json')))['counts']
    _c, meta = T.build()
    assert len(frozen) == len(_c)
    prod, bell, mode = C.collect(lambda i: frozen[i], meta)
    etas = np.array(C.ETAS)
    x_all, _ = C.agg(prod)
    sl, se = [], []
    rng = np.random.default_rng(1261)
    for e in C.ETAS:
        y, _ = C.agg(bell[e])
        sl.append(C.fit_slope(x_all, y)[0])
    sl = np.array(sl)
    # published weights: per-eta independent bootstrap se (recomputed here the same way)
    ind = []
    for e in C.ETAS:
        d = []
        for _ in range(300):
            yb, _ = C.agg(bell[e], rng.integers(0, len(bell[e]), len(bell[e])))
            xb, _ = C.agg(prod, rng.integers(0, len(prod), len(prod)))
            d.append(C.fit_slope(xb, yb)[0])
        ind.append(np.std(d, ddof=1))
    w = 1 / np.array(ind) ** 2
    k = float((w * etas * sl).sum() / (w * etas * etas).sum())
    ks = []
    for _ in range(NB):
        xb, _ = C.agg(prod, rng.integers(0, len(prod), len(prod)))
        sb = [C.fit_slope(xb, C.agg(bell[e], rng.integers(0, len(bell[e]), len(bell[e])))[0])[0] for e in C.ETAS]
        ks.append(float((w * etas * np.array(sb)).sum() / (w * etas * etas).sum()))
    out = dict(k=k, se_independent=float(np.sqrt(1 / (w * etas * etas).sum())), se_joint=float(np.std(ks, ddof=1)))
    json.dump(out, open(os.path.join(HERE, 'data', 'selfcheck', 'c126_k_joint.json'), 'w'), indent=1)
    print(out)
