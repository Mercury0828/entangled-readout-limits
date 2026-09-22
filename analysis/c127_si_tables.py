"""LaTeX rows for two Supplementary Tables, generated from code and frozen data (no transcription).

    python c127_si_tables.py > paper/si_tables_generated.tex

1. Bell decoder readout map: which frame reads each two-site Pauli, and through which readout (c99 decoder table).
2. Prospective test: predicted (a_hat, c_hat) of both calibration models and the evaluation fit (a, c) per target.
"""
import json

import numpy as np

import c99_run_device as R
import c114_sealed as S

AX = 'XYZ'
TAG = {'i': 'one-qubit, first qubit', 'j': 'one-qubit, second qubit', 'ij': 'parity'}
NAME = {'prod_single_z': 'product, $Z$ on qubit 2', 'prod_edge_zz': 'product, $ZZ$ on (2,3)',
        'prod_edge_yy': 'product, $YY$ on (2,3)', 'bell_w1_zz': 'Bell, $ZZ$ (one-qubit)',
        'bell_w2_yy': 'Bell, $YY$ (parity)'}

if __name__ == '__main__':
    tbl = R.bell_observables()
    print('% --- decoder map (from c99_run_device.bell_observables)')
    for k in range(3):
        for tag, a, b, sgn in tbl[k]:
            print(f"{k} & ${'+' if sgn > 0 else '-'}{AX[a]}{AX[b]}$ & {TAG[tag]}\\\\")
    circuits, meta = S.build()
    raw = json.load(open(S.RAW)); pred = json.load(open(S.PRED))
    assert raw['meta_sha256'] == S.meta_sha(meta) and pred['raw_sha256'] == S.sha(S.RAW)
    ys = [S.observe(m, c) for m, c in zip(meta, raw['counts'])]
    ev = {}
    for m, y in zip(meta, ys):
        if m['role'] == 'eval':
            ev.setdefault(m['mode'], []).append((m['x'], y))
    print('% --- calibration vs evaluation (sealed c114)')
    for mode, _, _ in S.TARGETS:
        a, c = S.fit(np.array(ev[mode], float))
        at, ct = pred['point'][f'target|{mode}']
        ap, cp = pred['point'][f'pooled|{mode}']
        print(f"{NAME[mode]} & {at:.5f} & {ct:+.5f} & {ap:.5f} & {cp:+.5f} & {a:.5f} & {c:+.5f}\\\\")
