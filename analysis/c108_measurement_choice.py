"""Held-out measurement-choice experiment on ibm_cleveland (job daoa3j8pqrnc7399miv0).

    python c108_measurement_choice.py sim      # simulator validation (no device time)
    python c108_measurement_choice.py run      # ibm_cleveland

Product measurement and product/Bell mixtures at w in {1.00, 0.60, 0.4172, 0.25, 0.15} on probe states, with no
injected noise. The paper uses this run for the paired regression of c110 (Fig. 3).

PROBE STATES.  rho = 2^-n (I + theta P), realised as a CLASSICAL mixture of product Pauli
eigenstates: for a single-site target P = sigma_a(q), qubit q is prepared in the +1 eigenstate
of sigma_a with probability (1+theta)/2 and the -1 eigenstate otherwise, while every other qubit
is prepared in a uniformly random Pauli eigenstate (which averages to I/2).  For an edge target
sigma_a(i) sigma_b(j), the two signs are drawn with biased PARITY.

The probes are product states prepared with SINGLE-QUBIT gates only, so the preparation carries no two-qubit gate
error, and the classical mixing probability is exact by construction.
"""
import json
import sys
import numpy as np
from qiskit import QuantumCircuit, transpile

import c99_run_device as R

KEYPATH = R.KEYPATH
BACKEND = R.BACKEND
RUN_INSTANCE = 'DQC methods'
n = 6
QUBITS = R.CHAIN_QUBITS[:n]

LIBRARY = [1.00, 0.60, 0.4172, 0.25, 0.15]      # FROZEN
N_REF = 10000                                    # declared reference shot count for MSE
THETAS = [0.5, -0.5, 0.2]                        # FROZEN
NUNIT = 900                                      # acquisition units per (allocation, probe)
SHOTS = 60
ETA_DEV = 0.99832
A_W2_RATIO = 0.96270 / 0.98389                   # one-qubit vs parity readout asymmetry

AX = {0: 'x', 1: 'y', 2: 'z'}
EDGES = [(q, q + 1) for q in range(n - 1)]
SINGLE = [(q, a) for q in range(n) for a in range(3)]
EDGEC = [(e, a, b) for e in range(len(EDGES)) for a in range(3) for b in range(3)]
si = {k: t for t, k in enumerate(SINGLE)}
ei = {k: t for t, k in enumerate(EDGEC)}
NC = len(SINGLE) + len(EDGEC)
off = len(SINGLE)


def prep_eigenstate(qc, q, axis, sign):
    """prepare the +-1 eigenstate of sigma_axis on qubit q (single-qubit gates only)"""
    if sign < 0:
        qc.x(q)
    if axis == 0:        # x
        qc.h(q)
    elif axis == 1:      # y
        qc.h(q); qc.s(q)


def probe_circuit_prefix(qc, rng, target, theta):
    """classical-mixture probe. target is ('S', q, a) or ('E', e, a, b)."""
    signs = {}
    if target[0] == 'S':
        _, q, a = target
        s = +1 if rng.random() < (1 + theta) / 2 else -1
        prep_eigenstate(qc, q, a, s)
        signs[q] = (a, s)
    else:
        _, e, a, b = target
        i, j = EDGES[e]
        par = +1 if rng.random() < (1 + theta) / 2 else -1
        s_i = +1 if rng.random() < 0.5 else -1
        s_j = par * s_i
        prep_eigenstate(qc, i, a, s_i); signs[i] = (a, s_i)
        prep_eigenstate(qc, j, b, s_j); signs[j] = (b, s_j)
    for q in range(n):
        if q not in signs:
            a = int(rng.integers(0, 3)); s = 1 if rng.random() < 0.5 else -1
            prep_eigenstate(qc, q, a, s)
            signs[q] = (a, s)
    return signs


def product_meas(qc, bases):
    for q in range(n):
        R.add_basis_rotation(qc, q, bases[q])
    qc.measure(range(n), range(n))


def bell_meas(qc, covering, ks):
    for (i, j), k in zip(covering, ks):
        R.add_cyc(qc, j, k)
        qc.cx(i, j)
        qc.h(i)
    qc.measure(range(n), range(n))


COVERINGS = [[(i, i + 1) for i in range(0, n - 1, 2)],
             [(i, i + 1) for i in range(1, n - 1, 2)]]


def build():
    """Calibration block, then the evaluation block. Returns circuits + metadata."""
    rng = np.random.default_rng(20260920)
    circuits, meta = [], []

    # ---- calibration: product eigenstates, both arms, ideal expectations known exactly
    for rep in range(240):
        axes = [int(x) for x in rng.integers(0, 3, n)]
        sgns = [1 if rng.random() < 0.5 else -1 for _ in range(n)]
        for arm in ('prod', 'bell'):
            qc = QuantumCircuit(n, n)
            for q in range(n):
                prep_eigenstate(qc, q, axes[q], sgns[q])
            qc.barrier()
            if arm == 'prod':
                bases = tuple(int(x) for x in rng.integers(0, 3, n))
                product_meas(qc, bases)
                meta.append(('cal', arm, (axes, sgns), bases))
            else:
                ci = int(rng.integers(0, 2))
                ks = [int(x) for x in rng.integers(0, 3, len(COVERINGS[ci]))]
                bell_meas(qc, COVERINGS[ci], ks)
                meta.append(('cal', arm, (axes, sgns), (ci, tuple(ks))))
            circuits.append(qc)

    # ---- evaluation: for each allocation, each probe, NUNIT acquisition units
    targets = [('S', 2, 2), ('E', 2, 2, 2)]      # one single-site and one edge target
    for w in LIBRARY:
        for tgt in targets:
            for th in THETAS:
                for u in range(NUNIT):
                    qc = QuantumCircuit(n, n)
                    probe_circuit_prefix(qc, rng, tgt, th)
                    qc.barrier()
                    use_prod = rng.random() < w
                    if use_prod:
                        bases = tuple(int(x) for x in rng.integers(0, 3, n))
                        product_meas(qc, bases)
                        meta.append(('ev', w, tgt, th, 'prod', bases))
                    else:
                        ci = int(rng.integers(0, 2))
                        ks = [int(x) for x in rng.integers(0, 3, len(COVERINGS[ci]))]
                        bell_meas(qc, COVERINGS[ci], ks)
                        meta.append(('ev', w, tgt, th, 'bell', (ci, tuple(ks))))
                    circuits.append(qc)
    return circuits, meta


def tgt_index(tgt):
    return si[(tgt[1], tgt[2])] if tgt[0] == 'S' else off + ei[(tgt[1], tgt[2], tgt[3])]


def analyse(get_counts, meta):
    """Per-ARM per-shot estimator moments, including the zeros WITHIN each arm.

    Two rules:
      (1) the zeros are not optional -- unbiasedness of `3*s` / `9*s*s` / `6*sgn*par` depends on
          averaging the multiplier against the coverage probability WITHIN that arm;
      (2) the two arms are then combined by INVERSE-VARIANCE WEIGHTING, never by pooling their
          per-shot values. Pooling leaves the estimate scaled by the arm probability w,
          which is exactly the -0.4267 bias we measured at w = 0.15 (0.15*(1/3)*3*0.5 = 0.075
          against a true 0.5).

    With per-arm per-shot variances v_p, v_b and arm fractions w, 1-w, the combined estimator
    over N shots has variance 1/(N(w/v_p + (1-w)/v_b)), so the per-shot constant is
        C(w) = 1 / (w/v_p + (1-w)/v_b)
    which is the theory's own formula, and reduces to 3/w for a single-site coordinate because
    the Bell arm supplies nothing (v_b = infinity).
    """
    tbl = R.bell_observables()
    acc = {}
    for idx, m in enumerate(meta):
        if m[0] != 'ev':
            continue
        _, w, tgt, th, arm, info = m
        key0 = (w, tgt, th, arm)
        s1, s2, nn = acc.get(key0, (0.0, 0.0, 0))
        for key, ct in get_counts(idx).items():
            sg = R.bits_of(key, n)
            v = 0.0
            if arm == 'prod':
                bases = info
                if tgt[0] == 'S':
                    q, a = tgt[1], tgt[2]
                    if bases[q] == a:
                        v = 3.0 * sg[q]
                else:
                    e, a, b = tgt[1], tgt[2], tgt[3]
                    i, j = EDGES[e]
                    if bases[i] == a and bases[j] == b:
                        v = 9.0 * sg[i] * sg[j]
            elif tgt[0] == 'E':
                ci, ks = info
                e, a, b = tgt[1], tgt[2], tgt[3]
                i, j = EDGES[e]
                for (ii, jj), k in zip(COVERINGS[ci], ks):
                    if {ii, jj} != {i, j}:
                        continue
                    for (tag, aa, bb, sgn) in tbl[k]:
                        if (aa, bb) != (a, b):
                            continue
                        v = 6.0 * sgn * (sg[ii] if tag == 'i' else
                                         (sg[jj] if tag == 'j' else sg[ii] * sg[jj]))
            s1 += v * ct; s2 += v * v * ct; nn += ct
        acc[key0] = (s1, s2, nn)
    return acc


def report(acc):
    print('=' * 100)
    print('HELD-OUT MEASUREMENT CHOICE -- bias, variance and MSE reported SEPARATELY')
    print('=' * 100)
    keys = sorted({(w, t, th) for (w, t, th, _a) in acc}, key=lambda k: (-k[0], str(k[1]), -k[2]))
    print(f"    {'w':>7s} {'target':>7s} {'theta':>6s} | {'bias':>9s} {'se':>7s} {'z':>6s} "
          f"{'C=var/shot':>11s} {'MSE@Nref':>10s}")
    print('    ' + '-' * 74)
    worst, worstse, worst_mse, zmax, VP = {}, {}, {}, [0.0], {}
    for (w, tgt, th) in keys:
        inv = 0.0; est_num = 0.0; ntot = 0; parts = []
        for arm in ('prod', 'bell'):
            if (w, tgt, th, arm) not in acc:
                continue
            s1, s2, nn = acc[(w, tgt, th, arm)]
            if nn == 0:
                continue
            ntot += nn
            mean = s1 / nn
            var = s2 / nn - mean ** 2
            if var <= 1e-12 or (tgt[0] == 'S' and arm == 'bell'):
                continue                       # the Bell arm carries no single-site information
            frac = nn  # weight by actual shots taken in that arm
            inv += frac / var
            est_num += frac * mean / var
            VP[(w, tgt, th, arm)] = (var, nn)
            # uncertainty on C is dominated by the uncertainty on each arm's variance, and the
            # independent units are CIRCUITS (nn/SHOTS), not shots.
            ncirc = max(nn / SHOTS, 1.0)
            parts.append((frac / var, (2.0 / ncirc) ** 0.5))
        if inv <= 0:
            continue
        theta_hat = est_num / inv
        C = ntot / inv                          # per-shot variance constant (theory's C)
        # C = ntot / sum_a (n_a/v_a); d(C)/C = sum_a (share_a * relse(v_a)) in quadrature
        se_C = C * sum((wt / inv * rs) ** 2 for wt, rs in parts) ** 0.5 if inv > 0 else 0.0
        bias = theta_hat - th
        # 🔴 bias^2 is NOT scaled by the shot count. MSE of the mean over N shots is
        #    MSE(N) = C/N + bias^2 ,
        # so we quote it at a DECLARED reference N. And the independent acquisition units are
        # the CIRCUITS (the probe preparation is redrawn per circuit), not the shots -- so the bias uncertainty uses NUNIT, not ntot.
        # each row is ONE theta, so the independent preparations are NUNIT, not NUNIT*len(THETAS)
        se_bias = (C / max(NUNIT, 1)) ** 0.5
        mse = C / N_REF + bias ** 2
        lbl = f"{tgt[0]}{tgt[1]}"
        z = bias / se_bias if se_bias > 0 else float('nan')
        print(f"    {w:7.4f} {lbl:>7s} {th:+6.2f} | {bias:+9.4f} {se_bias:7.4f} {z:+6.2f} "
              f"{C:11.4f} {mse:10.5f}")
        if C > worst.get(w, 0.0):
            worst[w] = C
            worstse[w] = se_C
        worst_mse[w] = max(worst_mse.get(w, 0.0), mse)
        zmax[0] = max(zmax[0], abs(z) if z == z else 0.0)
    print('    ' + '-' * 70)
    print()
    pred = {1.00: 9.000, 0.60: 7.676, 0.4172: 7.192, 0.25: 12.000, 0.15: 20.000}
    base = worst.get(1.00, float('nan'))
    print(f"    {'w':>7s} | {'worst C':>9s} {'+-':>7s} {'vs product':>11s} | {'pred C':>8s} "
          f"{'predicted':>10s} | {'z vs pred':>9s}")
    print('    ' + '-' * 74)
    for w in sorted(worst, reverse=True):
        z = (worst[w] - pred[w]) / worstse[w] if worstse[w] > 0 else float('nan')
        print(f"    {w:7.4f} | {worst[w]:9.4f} {worstse[w]:7.4f} "
              f"{100*(worst[w]-base)/base:+10.1f}% | "
              f"{pred[w]:8.3f} {100*(pred[w]-pred[1.00])/pred[1.00]:+9.1f}% | {z:+9.2f}")
    print(f"    max |z| on any bias = {zmax[0]:.2f}  "
          f"{'(consistent with zero)' if zmax[0] < 3 else '*** REAL BIAS PRESENT ***'}")
    print('    ' + '-' * 58)
    print()
    print('    per-arm per-shot variances (theory: v_p = 3 single / 9 edge, v_b = 6 edge):')
    seen = set()
    for (w, tgt, th, arm), (var, nn) in sorted(VP.items()):
        k = (tgt[0], arm)
        if k in seen:
            continue
        seen.add(k)
        print(f'      {tgt[0]}-target {arm:>4s} arm: v = {var:7.4f}  (n = {nn})')
    order = sorted(worst, key=lambda w: worst[w])
    pred_order = sorted(pred, key=lambda w: pred[w])
    print(f"    measured ordering (best first): {order}")
    print(f"    predicted ordering            : {pred_order}")
    ok = order == pred_order
    print(f"    ORDERING REPRODUCED: {ok}")
    if base == min(worst.values()):
        print('    *** FALSIFIER TRIPPED: pure product beat every mixture ***')
    return ok


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'sim'
    circuits, meta = build()
    nev = sum(1 for m in meta if m[0] == 'ev')
    print(f'{len(circuits)} circuits ({len(meta)-nev} calibration, {nev} evaluation) '
          f'x {SHOTS} shots = {len(circuits)*SHOTS:,} shots')
    if mode == 'sim':
        from qiskit_aer import AerSimulator
        sim = AerSimulator()
        r = sim.run(transpile(circuits, sim, optimization_level=0), shots=SHOTS).result()
        gc = lambda i: r.get_counts(i)
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        tok = json.load(open(KEYPATH, encoding='utf-8'))['apikey']
        svc = QiskitRuntimeService(channel='ibm_quantum_platform', token=tok,
                                   instance=RUN_INSTANCE)
        u = svc.usage()
        print(f'{RUN_INSTANCE}: {u["usage_remaining_seconds"]} s remaining')
        if u['usage_limit_reached']:
            print('no budget -- refusing to submit'); sys.exit(1)
        bk = svc.backend(BACKEND)
        tqs = transpile(circuits, backend=bk, initial_layout=QUBITS,
                        optimization_level=1, seed_transpiler=7)
        job = SamplerV2(mode=bk).run(tqs, shots=SHOTS)
        print(f'job {job.job_id()}')
        res = job.result()
        gc = lambda i: res[i].data.c.get_counts()
    print('done.\n')
    report(analyse(gc, meta))
