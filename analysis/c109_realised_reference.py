"""Analysis of the held-out run (job daoa3j8pqrnc7399miv0) against the realised probe value of each circuit.

The probe of each circuit is drawn from a seeded classical mixture, so its realised value differs from the nominal
one by finite sampling. This script reproduces the circuit schedule of c108 draw for draw, records the realised
signs, and scores estimates against them. It also runs the same analysis on a noiseless simulation of the identical
circuits, which is the null of the device comparison.
"""
import json
import sys
import numpy as np
from qiskit import QuantumCircuit, transpile

import c108_measurement_choice as M
import c99_run_device as R

JOB = 'daoa3j8pqrnc7399miv0'
INSTANCE = 'DQC methods'


def build_with_realised():
    """c108.build(), reproduced draw for draw, additionally returning the realised target value
    (sign for a single-site target, parity for an edge target) of every evaluation circuit."""
    rng = np.random.default_rng(20260920)
    circuits, meta = [], []
    n = M.n
    for rep in range(240):
        axes = [int(x) for x in rng.integers(0, 3, n)]
        sgns = [1 if rng.random() < 0.5 else -1 for _ in range(n)]
        for arm in ('prod', 'bell'):
            qc = QuantumCircuit(n, n)
            for q in range(n):
                M.prep_eigenstate(qc, q, axes[q], sgns[q])
            qc.barrier()
            if arm == 'prod':
                bases = tuple(int(x) for x in rng.integers(0, 3, n))
                M.product_meas(qc, bases)
                meta.append(('cal', arm, (axes, sgns), bases, None))
            else:
                ci = int(rng.integers(0, 2))
                ks = [int(x) for x in rng.integers(0, 3, len(M.COVERINGS[ci]))]
                M.bell_meas(qc, M.COVERINGS[ci], ks)
                meta.append(('cal', arm, (axes, sgns), (ci, tuple(ks)), None))
            circuits.append(qc)
    targets = [('S', 2, 2), ('E', 2, 2, 2)]
    for w in M.LIBRARY:
        for tgt in targets:
            for th in M.THETAS:
                for u in range(M.NUNIT):
                    qc = QuantumCircuit(n, n)
                    signs = M.probe_circuit_prefix(qc, rng, tgt, th)     # <-- now CAPTURED
                    if tgt[0] == 'S':
                        realised = signs[tgt[1]][1]
                    else:
                        i, j = M.EDGES[tgt[1]]
                        realised = signs[i][1] * signs[j][1]
                    qc.barrier()
                    if rng.random() < w:
                        bases = tuple(int(x) for x in rng.integers(0, 3, n))
                        M.product_meas(qc, bases)
                        meta.append(('ev', w, tgt, th, 'prod', bases, realised))
                    else:
                        ci = int(rng.integers(0, 2))
                        ks = [int(x) for x in rng.integers(0, 3, len(M.COVERINGS[ci]))]
                        M.bell_meas(qc, M.COVERINGS[ci], ks)
                        meta.append(('ev', w, tgt, th, 'bell', (ci, tuple(ks)), realised))
                    circuits.append(qc)
    return circuits, meta


def per_cell(get_counts, meta):
    """Per (w, target, theta): inverse-variance-combined estimate, realised theta, and the true
    residual bias with a circuit-level standard error."""
    tbl = R.bell_observables()
    acc = {}
    real = {}
    for idx, m in enumerate(meta):
        if m[0] != 'ev':
            continue
        _, w, tgt, th, arm, info, rv = m
        key = (w, tgt, th)
        real.setdefault(key, []).append(rv)
        s1, s2, nn = acc.get((key, arm), (0.0, 0.0, 0))
        for k, ct in get_counts(idx).items():
            sg = R.bits_of(k, M.n)
            v = 0.0
            if arm == 'prod':
                if tgt[0] == 'S':
                    if info[tgt[1]] == tgt[2]:
                        v = 3.0 * sg[tgt[1]]
                else:
                    i, j = M.EDGES[tgt[1]]
                    if info[i] == tgt[2] and info[j] == tgt[3]:
                        v = 9.0 * sg[i] * sg[j]
            elif tgt[0] == 'E':
                ci, ks = info
                i, j = M.EDGES[tgt[1]]
                for (ii, jj), kk in zip(M.COVERINGS[ci], ks):
                    if {ii, jj} != {i, j}:
                        continue
                    for (tag, aa, bb, sgn) in tbl[kk]:
                        if (aa, bb) != (tgt[2], tgt[3]):
                            continue
                        v = 6.0 * sgn * (sg[ii] if tag == 'i' else
                                         (sg[jj] if tag == 'j' else sg[ii] * sg[jj]))
            s1 += v * ct; s2 += v * v * ct; nn += ct
        acc[(key, arm)] = (s1, s2, nn)
    out = {}
    for key in real:
        inv = 0.0; num = 0.0; ntot = 0
        for arm in ('prod', 'bell'):
            if (key, arm) not in acc:
                continue
            s1, s2, nn = acc[(key, arm)]
            if nn == 0:
                continue
            ntot += nn
            mu = s1 / nn; var = s2 / nn - mu ** 2
            if var <= 1e-12 or (key[1][0] == 'S' and arm == 'bell'):
                continue
            inv += nn / var; num += nn * mu / var
        if inv <= 0:
            continue
        th_hat = num / inv
        th_real = float(np.mean(real[key]))
        C = ntot / inv
        se = (C / ntot) ** 0.5                   # shot-level: the realised reference removes the
        out[key] = dict(th_hat=th_hat, th_real=th_real, th_nom=key[2],   # probe-draw variance
                        resid=th_hat - th_real, se=se, C=C, ntot=ntot)
    return out


if __name__ == '__main__':
    circuits, meta = build_with_realised()
    ref_c, ref_m = M.build()
    # compare ONLY the original fields: we append one field (the realised value, or None for
    # calibration entries), so a fixed-width slice would compare tuples of different length.
    # Verified separately: 27480/27480 entries match on their original fields.
    same = len(ref_m) == len(meta) and all(a == b[:len(a)] for a, b in zip(ref_m, meta))
    print(f'schedule reproduces c108 draw-for-draw: {same}  ({len(meta)} circuits)')
    if not same:
        print('*** schedule mismatch -- refusing to analyse ***'); sys.exit(1)

    from qiskit_ibm_runtime import QiskitRuntimeService
    tok = json.load(open(M.KEYPATH, encoding='utf-8'))['apikey']
    svc = QiskitRuntimeService(channel='ibm_quantum_platform', token=tok, instance=INSTANCE)
    res = svc.job(JOB).result()
    dev = per_cell(lambda i: res[i].data.c.get_counts(), meta)

    from qiskit_aer import AerSimulator
    sim_b = AerSimulator()
    rs = sim_b.run(transpile(circuits, sim_b, optimization_level=0), shots=M.SHOTS).result()
    sim = per_cell(lambda i: rs.get_counts(i), meta)

    print()
    print('=' * 100)
    print('RESIDUAL BIAS against the REALISED probe value -- device vs noiseless simulator')
    print('=' * 100)
    print(f"    {'w':>7s} {'tgt':>4s} {'nom':>5s} {'real':>7s} | {'sim resid':>10s} {'dev resid':>10s} "
          f"{'se':>7s} | {'z sim':>6s} {'z dev':>6s} {'z(dev-sim)':>10s}")
    print('    ' + '-' * 92)
    zs, zd, zdiff = [], [], []
    by_tgt = {'S': [], 'E': []}
    for key in sorted(dev, key=lambda k: (-k[0], k[1][0], -k[2])):
        if key not in sim:
            continue
        d, s_ = dev[key], sim[key]
        z_s = s_['resid'] / s_['se']; z_d = d['resid'] / d['se']
        z_x = (d['resid'] - s_['resid']) / (d['se'] ** 2 + s_['se'] ** 2) ** 0.5
        zs.append(z_s); zd.append(z_d); zdiff.append(z_x)
        by_tgt[key[1][0]].append((d['resid'], s_['resid'], d['se']))
        print(f"    {key[0]:7.4f} {key[1][0]+'2':>4s} {key[2]:+5.2f} {d['th_real']:+7.4f} | "
              f"{s_['resid']:+10.4f} {d['resid']:+10.4f} {d['se']:7.4f} | "
              f"{z_s:+6.2f} {z_d:+6.2f} {z_x:+10.2f}")
    print('    ' + '-' * 92)
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
    print(f"    RMS z of simulator residuals : {rms(zs):.2f}   (expect ~1 if the estimator is unbiased)")
    print(f"    RMS z of device residuals    : {rms(zd):.2f}")
    print(f"    RMS z of device - simulator  : {rms(zdiff):.2f}   (expect ~1 if device and sim are")
    print(f"                                          independent draws of the same physics)")
    print()
    for t, lab in (('S', 'single-site (product arm only)'), ('E', 'edge (both arms)')):
        v = by_tgt[t]
        md = np.mean([x[0] for x in v]); ms = np.mean([x[1] for x in v])
        se = np.sqrt(np.mean([x[2] ** 2 for x in v]) / len(v))
        print(f"    {lab:<32s}: mean device resid {md:+.4f}, sim {ms:+.4f}, "
              f"difference {md-ms:+.4f} +- {se*np.sqrt(2):.4f}")
    print('=' * 100)
