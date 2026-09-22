"""Per-circuit paired regression of the held-out run (Fig. 3a and the held-out slopes of Fig. 3b).

For each circuit that covers a target, the observed mean parity y_c is regressed on the circuit's own realised
sign x_c, y_c = a x_c + c. The slope a is the response of the measured mode and the intercept c its offset
(I(0) = a^2 / (1 - c^2)). Each circuit is compared with itself and the sign is known exactly, so the fit does not
depend on which subset of circuits covers the target and the response cannot average away. The identical circuits
run on a noiseless simulator give the null.

Modes: product single site, product edge, Bell one-qubit readout (w1) and Bell parity readout (w2).
Uncertainty from a bootstrap over circuits.
"""
import json
import numpy as np
from qiskit import transpile

import c108_measurement_choice as M
import c109_realised_reference as C
import c99_run_device as R

JOB = C.JOB
INSTANCE = C.INSTANCE
NBOOT = 400
rng = np.random.default_rng(1101)


def per_circuit_pairs(get_counts, meta):
    """Return {mode: [(x_c, y_c, shots_c), ...]} over evaluation circuits covering the target."""
    tbl = R.bell_observables()
    out = {'prod_single': [], 'prod_edge': [], 'bell_w1': [], 'bell_w2': []}
    for idx, m in enumerate(meta):
        if m[0] != 'ev':
            continue
        _, w, tgt, th, arm, info, x = m
        counts = get_counts(idx)
        tot = sum(counts.values())
        if arm == 'prod':
            if tgt[0] == 'S':
                q, a = tgt[1], tgt[2]
                if info[q] != a:
                    continue
                y = sum(R.bits_of(k, M.n)[q] * ct for k, ct in counts.items()) / tot
                out['prod_single'].append((x, y, tot))
            else:
                i, j = M.EDGES[tgt[1]]
                if not (info[i] == tgt[2] and info[j] == tgt[3]):
                    continue
                y = sum(R.bits_of(k, M.n)[i] * R.bits_of(k, M.n)[j] * ct
                        for k, ct in counts.items()) / tot
                out['prod_edge'].append((x, y, tot))
        elif tgt[0] == 'E':
            ci, ks = info
            i, j = M.EDGES[tgt[1]]
            for (ii, jj), kk in zip(M.COVERINGS[ci], ks):
                if {ii, jj} != {i, j}:
                    continue
                for (tag, aa, bb, sgn) in tbl[kk]:
                    if (aa, bb) != (tgt[2], tgt[3]):
                        continue
                    y = 0.0
                    for k, ct in counts.items():
                        sg = R.bits_of(k, M.n)
                        par = (sg[ii] if tag == 'i' else
                               (sg[jj] if tag == 'j' else sg[ii] * sg[jj]))
                        y += sgn * par * ct
                    y /= tot
                    out['bell_w2' if tag == 'ij' else 'bell_w1'].append((x, y, tot))
    return out


def fit(pairs):
    x = np.array([p[0] for p in pairs], float)
    y = np.array([p[1] for p in pairs], float)
    A = np.vstack([x, np.ones_like(x)]).T
    (a, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(a), float(c)


def boot(pairs):
    n = len(pairs)
    arr = np.array(pairs, float)
    aa, cc = [], []
    for _ in range(NBOOT):
        s = arr[rng.integers(0, n, n)]
        A = np.vstack([s[:, 0], np.ones(n)]).T
        (a, c), *_ = np.linalg.lstsq(A, s[:, 1], rcond=None)
        aa.append(a); cc.append(c)
    return float(np.std(aa, ddof=1)), float(np.std(cc, ddof=1))


if __name__ == '__main__':
    circuits, meta = C.build_with_realised()
    from qiskit_ibm_runtime import QiskitRuntimeService
    tok = json.load(open(M.KEYPATH, encoding='utf-8'))['apikey']
    svc = QiskitRuntimeService(channel='ibm_quantum_platform', token=tok, instance=INSTANCE)
    res = svc.job(JOB).result()
    dev = per_circuit_pairs(lambda i: res[i].data.c.get_counts(), meta)

    from qiskit_aer import AerSimulator
    sb = AerSimulator()
    rs = sb.run(transpile(circuits, sb, optimization_level=0), shots=M.SHOTS).result()
    sim = per_circuit_pairs(lambda i: rs.get_counts(i), meta)

    print('=' * 100)
    print('PAIRED DAMPING: per-circuit observed parity regressed on its own realised sign')
    print('=' * 100)
    print(f"    {'mode':<12s} {'n circ':>7s} | {'a (sim)':>9s} {'a (dev)':>9s} {'se':>7s} "
          f"{'a_dev-a_sim':>12s} {'z':>6s} | {'c (dev)':>9s} {'se':>7s}")
    print('    ' + '-' * 90)
    results = {}
    for mode in ('prod_single', 'prod_edge', 'bell_w1', 'bell_w2'):
        if len(dev[mode]) < 20:
            print(f'    {mode:<12s} too few circuits ({len(dev[mode])})'); continue
        a_s, c_s = fit(sim[mode]); a_d, c_d = fit(dev[mode])
        sa_s, _ = boot(sim[mode]); sa_d, sc_d = boot(dev[mode])
        se = (sa_s ** 2 + sa_d ** 2) ** 0.5
        z = (a_d - a_s) / se
        results[mode] = (a_d, a_s, sa_d, c_d, sc_d)
        print(f"    {mode:<12s} {len(dev[mode]):7d} | {a_s:9.5f} {a_d:9.5f} {sa_d:7.5f} "
              f"{a_d-a_s:+12.5f} {z:+6.2f} | {c_d:+9.5f} {sc_d:7.5f}")
    print('    ' + '-' * 90)
    print()
    print('    Expectations from the device calibration (NOT fitted here):')
    print('      readout error ~0.6% per qubit  ->  a ~ 1 - 2p ~ 0.988 per qubit read')
    print('      so product single-site a ~ 0.988; product edge ~ 0.976 (two qubits read);')
    print('      Bell modes additionally carry one CZ (~0.998) and, per the earlier run, w2 < w1.')
    if 'bell_w1' in results and 'bell_w2' in results:
        d = results['bell_w1'][0] - results['bell_w2'][0]
        s = (results['bell_w1'][2] ** 2 + results['bell_w2'][2] ** 2) ** 0.5
        print()
        print(f'    Mode asymmetry on held-out data: a(w1) - a(w2) = {d:+.5f} +- {s:.5f}  '
              f'(z = {d/s:+.2f})')
        print(f'      The earlier injected-noise run found ~ +0.029, systematically positive.')
    print('=' * 100)
