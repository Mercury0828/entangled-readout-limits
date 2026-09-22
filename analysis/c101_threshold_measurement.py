"""Engineered-noise sweep on ibm_cleveland (job dam6td02fm4c73f2gl1g): circuit schedule and first analysis.

Builds the product arm (no entangling gate, shared by every injected gate quality) and the Bell-dimer arm with an
injected pair-depolarising channel of strength eta_inj in 0.70-0.98, with a balanced, shuffled schedule of bases
and settings. The fitted Bell response slope against eta_inj gives the constant k of Supplementary Note 6. The
re-analysis actually reported is c106 (run on the frozen counts by c123).
"""
import json
import sys
import numpy as np
from qiskit import transpile

import c99_run_device as R

KEYPATH = R.KEYPATH
BACKEND = R.BACKEND
n = 6
QUBITS = R.CHAIN_QUBITS[:n]
ETAS = [0.98, 0.95, 0.92, 0.89, 0.86, 0.83, 0.78, 0.74, 0.70]
NCIRC_BELL, SHOTS = 1200, 160          # per eta  -> 192k shots
NCIRC_PROD, SHOTS_P = 2400, 160        # shared   -> 384k shots
SEED_STATE = 11

COVERINGS = [[(i, i + 1) for i in range(0, n - 1, 2)],
             [(i, i + 1) for i in range(1, n - 1, 2)]]
EDGE = [(q, a, b) for q in range(n - 1) for a in range(3) for b in range(3)]
ei = {k: t for t, k in enumerate(EDGE)}
NE = len(EDGE)


def balanced(rng, length, m):
    """a deterministic balanced cycle through 0..m-1, shuffled within blocks"""
    reps = int(np.ceil(length / m))
    seq = np.tile(np.arange(m), reps)[:length].copy()
    rng.shuffle(seq)
    return seq


def build():
    rng = np.random.default_rng(20260917)
    circuits, meta = [], []

    # ---- shared product dataset (eta-independent: no entangling gate in the measurement)
    bases_cols = [balanced(rng, NCIRC_PROD, 3) for _ in range(n)]
    for t in range(NCIRC_PROD):
        bases = tuple(int(bases_cols[q][t]) for q in range(n))
        circuits.append(R.product_instance(n, bases, SEED_STATE))
        meta.append(('prod', None, bases))

    # ---- one Bell dataset per eta
    for eta in ETAS:
        cov_seq = balanced(rng, NCIRC_BELL, 2)
        set_cols = [balanced(rng, NCIRC_BELL, 3) for _ in range(3)]
        for t in range(NCIRC_BELL):
            ci = int(cov_seq[t]); cov = COVERINGS[ci]
            ks = [int(set_cols[d][t]) for d in range(len(cov))]
            inj = [((int(rng.integers(0, 4)), int(rng.integers(0, 4))))
                   if rng.random() < (1 - eta) else None for _ in cov]
            circuits.append(R.bell_instance(n, cov, ks, inj, SEED_STATE))
            meta.append(('bell', eta, (ci, tuple(ks))))
    return circuits, meta


def analyse(get_counts, meta):
    tbl = R.bell_observables()
    prod = {'s': np.zeros(NE), 'nn': np.zeros(NE)}
    bell = {e: {'s': np.zeros(NE), 'nn': np.zeros(NE)} for e in ETAS}
    for idx, (arm, eta, info) in enumerate(meta):
        for key, ct in get_counts(idx).items():
            sg = R.bits_of(key, n)
            if arm == 'prod':
                for q in range(n - 1):
                    t = ei[(q, info[q], info[q + 1])]
                    prod['s'][t] += sg[q] * sg[q + 1] * ct
                    prod['nn'][t] += ct
            else:
                ci, ks = info
                A = bell[eta]
                for (i, j), k in zip(COVERINGS[ci], ks):
                    for (tag, a, b, sgn) in tbl[k]:
                        par = sg[i] if tag == 'i' else (sg[j] if tag == 'j' else sg[i] * sg[j])
                        t = ei[(i, a, b)]
                        A['s'][t] += sgn * par * ct
                        A['nn'][t] += ct
    return prod, bell


def report(prod, bell):
    okp = prod['nn'] > 500
    x_all = np.where(okp, prod['s'] / np.maximum(prod['nn'], 1), 0.0)
    print('=' * 100)
    print('THRESHOLD MEASUREMENT -- eta* from the measured damping, not from C_meas')
    print('=' * 100)
    print(f"    {'eta_inj':>8s} | {'eta_emp':>9s} {'+- err':>8s} | {'r_emp':>8s} {'r_nom':>8s} | "
          f"{'coords':>7s}")
    print('    ' + '-' * 62)
    rows = []
    for eta in ETAS:
        B = bell[eta]
        ok = okp & (B['nn'] > 500)
        x = x_all[ok]
        y = B['s'][ok] / B['nn'][ok]
        slope = float(x @ y / (x @ x))
        resid = y - slope * x
        err = float(np.sqrt((resid @ resid) / max(len(x) - 1, 1) / (x @ x)))
        rows.append((eta, slope, err))
        print(f"    {eta:8.4f} | {slope:9.5f} {err:8.5f} | {slope**2:8.5f} {eta**2:8.5f} | "
              f"{int(ok.sum()):7d}")
    print('    ' + '-' * 62)

    inj = np.array([r[0] for r in rows])
    es = np.array([r[1] for r in rows])
    er = np.array([r[2] for r in rows])
    w = 1.0 / er ** 2
    k = float((w * inj * es).sum() / (w * inj * inj).sum())
    kerr = float(np.sqrt(1.0 / (w * inj * inj).sum()))
    chi2 = float((((es - k * inj) / er) ** 2).sum())
    eta_star = np.sqrt(2 / 3) / k
    eta_star_err = np.sqrt(2 / 3) * kerr / k ** 2

    print(f'    damping fit   eta_emp = k * eta_inj :  k = {k:.5f} +- {kerr:.5f}   '
          f'chi2 = {chi2:.2f} / {len(es)-1} dof')
    print()
    print('    ' + '=' * 62)
    print(f'    MEASURED THRESHOLD   eta* = sqrt(2/3)/k = {eta_star:.5f} +- {eta_star_err:.5f}')
    print(f'    PREDICTED            eta* = sqrt(2/3)   = {np.sqrt(2/3):.5f}')
    dev = (eta_star - np.sqrt(2 / 3)) / eta_star_err
    print(f'    deviation = {dev:+.2f} sigma   '
          f'-> {"CONSISTENT" if abs(dev) < 3 else "*** TENSION ***"}')
    print('    ' + '=' * 62)
    print()
    print('    This is a device-derived threshold with an error bar, comparable against a')
    print('    parameter-free prediction. Unlike C_meas it is NOT reproduced by our')
    print('    own design constants: it is fixed by the ratio of the two arms measured')
    print('    information, in which preparation and readout error cancel.')
    print('=' * 100)


if __name__ == '__main__':
    simulate = (len(sys.argv) > 1 and sys.argv[1] == 'sim')
    circuits, meta = build()
    tot = NCIRC_PROD * SHOTS_P + len(ETAS) * NCIRC_BELL * SHOTS
    print(f'{len(circuits)} circuits, {tot:,} shots '
          f'({"SIMULATOR" if simulate else "ibm_cleveland"})')
    if simulate:
        from qiskit_aer import AerSimulator
        sim = AerSimulator()
        r = sim.run(transpile(circuits, sim, optimization_level=0), shots=SHOTS).result()
        gc = lambda i: r.get_counts(i)
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        tok = json.load(open(KEYPATH, encoding='utf-8'))['apikey']
        svc = QiskitRuntimeService(channel='ibm_quantum_platform', token=tok, instance='auto')
        bk = svc.backend(BACKEND)
        tqs = transpile(circuits, backend=bk, initial_layout=QUBITS,
                        optimization_level=1, seed_transpiler=7)
        job = SamplerV2(mode=bk).run(tqs, shots=SHOTS)
        print(f'job id {job.job_id()}')
        res = job.result()
        gc = lambda i: res[i].data.c.get_counts()
    print('done.\n')
    prod, bell = analyse(gc, meta)
    report(prod, bell)
