"""Circuit construction and device runner shared by the ibm_cleveland runs.

    python c99_run_device.py smoke     # ~20k shots, end-to-end check, ~1 minute of device time

State preparation, Pauli bases, the Bell-dimer decoder and its readout table (bell_observables), and the
engineered-noise injection: D_eta = eta*id + (1-eta)*(uniform Pauli twirl), so per CIRCUIT INSTANCE we
draw: with probability (1-eta) apply a uniformly random two-qubit Pauli after the entangling
gate, otherwise nothing.  Averaged over instances this is exactly D_eta. A circuit's shots share its gates, so
randomisation must happen ACROSS instances -- hence many circuits with few shots each.

The API token is read at run time from the file KEYPATH (a JSON object with an "apikey" field) and never printed
or stored.
"""
import json
import sys
import numpy as np
from qiskit import QuantumCircuit, transpile

KEYPATH = 'apikey.json'
BACKEND = 'ibm_cleveland'
CHAIN_QUBITS = [50, 51, 58, 71, 70, 69, 68, 67, 66, 65, 77, 85]

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
SIG = [X, Y, Z]


# ------------------------------------------------------------------------------------------
# target state: a fixed, reproducible, entangled shallow circuit
# ------------------------------------------------------------------------------------------
def state_prep(qc, n, seed=11):
    r = np.random.default_rng(seed)
    for q in range(n):
        qc.ry(float(r.uniform(0.3, 1.2)), q)
    for q in range(0, n - 1, 2):
        qc.cz(q, q + 1)
    for q in range(1, n - 1, 2):
        qc.cz(q, q + 1)
    for q in range(n):
        qc.rx(float(r.uniform(0.2, 0.9)), q)


def add_basis_rotation(qc, q, a):
    if a == 0:
        qc.h(q)
    elif a == 1:
        qc.sdg(q); qc.h(q)


def add_cyc(qc, q, k):
    for _ in range(k % 3):
        qc.h(q); qc.s(q)


def add_pauli(qc, q, p):
    if p == 1:
        qc.x(q)
    elif p == 2:
        qc.y(q)
    elif p == 3:
        qc.z(q)


def product_instance(n, bases, seed_state):
    qc = QuantumCircuit(n, n)
    state_prep(qc, n, seed_state)
    qc.barrier()
    for q in range(n):
        add_basis_rotation(qc, q, bases[q])
    qc.measure(range(n), range(n))
    return qc


def bell_instance(n, covering, ks, inj, seed_state):
    qc = QuantumCircuit(n, n)
    state_prep(qc, n, seed_state)
    qc.barrier()
    for (i, j), k, pp in zip(covering, ks, inj):
        add_cyc(qc, j, k)
        qc.cx(i, j)
        if pp is not None:
            add_pauli(qc, i, pp[0]); add_pauli(qc, j, pp[1])
        qc.h(i)
    meas = sorted({q for d in covering for q in d})
    qc.measure(meas, meas)
    return qc


# ------------------------------------------------------------------------------------------
# reference values, by exact simulation of the (noiseless) prep circuit
# ------------------------------------------------------------------------------------------
def reference_values(n, seed_state):
    from qiskit.quantum_info import Statevector
    qc = QuantumCircuit(n)
    state_prep(qc, n, seed_state)
    psi = Statevector(qc).data
    rho = np.outer(psi, psi.conj())

    def emb(ops):
        M = np.array([[1.0 + 0j]])
        for q in range(n - 1, -1, -1):          # qiskit little-endian
            M = np.kron(M, ops.get(q, I2))
        return M

    single = {(q, a): float(np.real(np.trace(rho @ emb({q: SIG[a]}))))
              for q in range(n) for a in range(3)}
    edge = {(q, a, b): float(np.real(np.trace(rho @ emb({q: SIG[a], q + 1: SIG[b]}))))
            for q in range(n - 1) for a in range(3) for b in range(3)}
    return single, edge


def bits_of(key, n):
    """qiskit returns bitstrings with classical bit 0 rightmost"""
    s = key.replace(' ', '')
    return [1 - 2 * int(s[n - 1 - q]) for q in range(n)]


BELL_TABLE = {}


def bell_observables():
    """For each setting k, which edge Pauli (and sign) does each outcome parity measure?
    Determined numerically from the circuit, never by hand (see c98)."""
    from qiskit.quantum_info import Operator
    if BELL_TABLE:
        return BELL_TABLE
    for k in range(3):
        qc = QuantumCircuit(2)
        add_cyc(qc, 1, k)
        qc.cx(0, 1)
        qc.h(0)
        V = Operator(qc).data
        rows = []
        for (zop, tag) in ((np.kron(I2, Z), 'i'), (np.kron(Z, I2), 'j'), (np.kron(Z, Z), 'ij')):
            O = V.conj().T @ zop @ V
            for a in range(3):
                for b in range(3):
                    cand = np.kron(SIG[b], SIG[a])
                    for sgn in (1.0, -1.0):
                        if np.abs(O - sgn * cand).max() < 1e-9:
                            rows.append((tag, a, b, sgn))
        BELL_TABLE[k] = rows
    return BELL_TABLE


def run_pilot(simulate=False):
    """Sweep eta and measure the constant.

    The constant IS a variance, and a variance is directly estimable from single-shot estimator
    values, so NO repetition structure is needed.  That is why this costs ~1M shots and not the
    43M that repeated MSE estimation would need.

    simulate=True runs the identical pipeline on AerSimulator, so the analysis path can be
    validated without consuming device time."""
    n = 6
    ETAS = [0.95, 0.90, 0.8165, 0.75, 0.70]
    NCIRC, SHOTS = 500, 100          # per (eta, arm) -> 50k shots
    qubits = CHAIN_QUBITS[:n]
    seed_state = 11
    rng = np.random.default_rng(777)
    tbl = bell_observables()

    COVERINGS = [[(i, i + 1) for i in range(0, n - 1, 2)],
                 [(i, i + 1) for i in range(1, n - 1, 2)]]
    SINGLE = [(q, a) for q in range(n) for a in range(3)]
    EDGE = [(q, a, b) for q in range(n - 1) for a in range(3) for b in range(3)]
    si = {k: t for t, k in enumerate(SINGLE)}
    ei = {k: t for t, k in enumerate(EDGE)}

    circuits, meta = [], []
    for eta in ETAS:
        for _ in range(NCIRC):
            bases = tuple(int(x) for x in rng.integers(0, 3, n))
            circuits.append(product_instance(n, bases, seed_state))
            meta.append((eta, 'prod', bases))
        for _ in range(NCIRC):
            ci = int(rng.integers(0, 2)); cov = COVERINGS[ci]
            ks = [int(x) for x in rng.integers(0, 3, len(cov))]
            inj = [((int(rng.integers(0, 4)), int(rng.integers(0, 4))))
                   if rng.random() < (1 - eta) else None for _ in cov]
            circuits.append(bell_instance(n, cov, ks, inj, seed_state))
            meta.append((eta, 'bell', (ci, tuple(ks))))

    print(f'PILOT{"-SIM" if simulate else ""}: {len(circuits)} circuits x {SHOTS} shots '
          f'= {len(circuits)*SHOTS:,} shots over {len(ETAS)} eta points')

    if simulate:
        from qiskit_aer import AerSimulator
        sim = AerSimulator()
        tqs = transpile(circuits, backend=sim, optimization_level=0)
        result = sim.run(tqs, shots=SHOTS).result()
        get_counts = lambda idx: result.get_counts(idx)
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        tok = json.load(open(KEYPATH, encoding='utf-8'))['apikey']
        svc = QiskitRuntimeService(channel='ibm_quantum_platform', token=tok, instance='auto')
        bk = svc.backend(BACKEND)
        tqs = transpile(circuits, backend=bk, initial_layout=qubits,
                        optimization_level=1, seed_transpiler=7)
        job = SamplerV2(mode=bk).run(tqs, shots=SHOTS)
        print(f'job id {job.job_id()} submitted; waiting...')
        res = job.result()
        get_counts = lambda idx: res[idx].data.c.get_counts()
    print('done.')

    NC = len(SINGLE) + len(EDGE)
    stats = {(e, arm): {'s': np.zeros(NC), 's2': np.zeros(NC), 'nn': 0}
             for e in ETAS for arm in ('prod', 'bell')}
    for idx, m in enumerate(meta):
        eta, arm, info = m
        st = stats[(eta, arm)]
        for key, ct in get_counts(idx).items():
            sg = bits_of(key, n)
            v = np.zeros(NC)
            if arm == 'prod':
                bases = info
                for q in range(n):
                    v[si[(q, bases[q])]] = 3.0 * sg[q]
                for q in range(n - 1):
                    v[len(SINGLE) + ei[(q, bases[q], bases[q + 1])]] = 9.0 * sg[q] * sg[q + 1]
            else:
                ci, ks = info
                for (i, j), k in zip(COVERINGS[ci], ks):
                    for (tag, a, b, sgn) in tbl[k]:
                        par = sg[i] if tag == 'i' else (sg[j] if tag == 'j' else sg[i] * sg[j])
                        v[len(SINGLE) + ei[(i, a, b)]] = 3.0 * 2.0 * sgn * par / eta
            st['s'] += v * ct
            st['s2'] += v * v * ct
            st['nn'] += ct

    print('=' * 100)
    print('PILOT RESULT -- measured constant vs theory')
    print('=' * 100)
    print(f"    {'eta':>8s} | {'C_meas(p=inf)':>14s} {'theory':>9s} | {'w*_meas':>9s} "
          f"{'w*_theory':>10s} | {'C_meas(p=1)':>12s}")
    print('    ' + '-' * 76)
    off = len(SINGLE)
    for eta in ETAS:
        r2 = eta ** 2
        P, B = stats[(eta, 'prod')], stats[(eta, 'bell')]
        # The second moment is a DESIGN parameter, not a measured quantity:
        #   product arm  X^2 = 9  * 1[basis matches]   -> E[X^2] = 3 (single), 9 (edge)
        #   Bell arm     X^2 = 36/eta^2 * 1[covered]   -> E[X^2] = 6/r
        # X^2 is deterministic given the setting draw, so measuring it only injects the
        # sampling noise of the setting schedule -- which, taken through a MAX over 54
        # coordinates, inflated C by ~30% in the first pilot.  We use the exact values and
        # measure only the means.
        E2p = np.concatenate([np.full(off, 3.0), np.full(NC - off, 9.0)])
        E2b = np.concatenate([np.zeros(off), np.full(NC - off, 6.0 / r2)])
        vp = E2p - (P['s'] / P['nn']) ** 2
        vb = E2b - (B['s'] / B['nn']) ** 2

        def C_of(w, p):
            inv = w / np.maximum(vp, 1e-12)
            inv[off:] = inv[off:] + (1 - w) / np.maximum(vb[off:], 1e-12)
            var = 1.0 / inv
            if not np.isfinite(p):
                return float(var.max())
            return float((np.ones(NC) / NC @ var ** p) ** (1.0 / p))

        ws = np.linspace(0.02, 1.0, 400)
        cinf = [C_of(w, np.inf) for w in ws]
        c1 = [C_of(w, 1.0) for w in ws]
        th_C = 9.0 if r2 <= 2 / 3 else 3 + 4 / r2
        th_w = 1.0 if r2 <= 2 / 3 else 3 * r2 / (4 + 3 * r2)
        print(f"    {eta:8.4f} | {min(cinf):14.4f} {th_C:9.4f} | "
              f"{ws[int(np.argmin(cinf))]:9.4f} {th_w:10.4f} | {min(c1):12.4f}")
    print('    ' + '-' * 76)
    print('    NOTE: measured C includes device readout and prep error on top of the injected')
    print('    channel, so on hardware it should sit slightly ABOVE theory; the SHAPE and the')
    print('    crossover location are the predictions, not the absolute value.')
    print('=' * 100)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'smoke'
    if mode == 'smoke':
        n, ETA, NCIRC, SHOTS = 4, 0.85, 480, 48
    elif mode in ('pilot', 'pilot-sim'):
        run_pilot(simulate=(mode == 'pilot-sim')); sys.exit(0)
    else:
        print('The full run is 43.2M shots (~144 min of device time). Confirm the budget')
        print('and then edit this guard. Refusing to submit it unprompted.')
        sys.exit(1)

    qubits = CHAIN_QUBITS[:n]
    seed_state = 11
    single_ref, edge_ref = reference_values(n, seed_state)
    rng = np.random.default_rng(4242)

    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    tok = json.load(open(KEYPATH, encoding='utf-8'))['apikey']
    svc = QiskitRuntimeService(channel='ibm_quantum_platform', token=tok, instance='auto')
    bk = svc.backend(BACKEND)
    print(f'backend {bk.name}, using physical qubits {qubits}')

    COVERINGS = [[(i, i + 1) for i in range(0, n - 1, 2)],
                 [(i, i + 1) for i in range(1, n - 1, 2)]]

    circuits, meta = [], []
    for _ in range(NCIRC // 2):
        bases = tuple(int(x) for x in rng.integers(0, 3, n))
        circuits.append(product_instance(n, bases, seed_state))
        meta.append(('prod', bases, None, None))
    for _ in range(NCIRC // 2):
        ci = int(rng.integers(0, 2))
        cov = COVERINGS[ci]
        if not cov:
            continue
        ks = [int(x) for x in rng.integers(0, 3, len(cov))]
        inj = []
        for _d in cov:
            if rng.random() < (1 - ETA):
                inj.append((int(rng.integers(0, 4)), int(rng.integers(0, 4))))
            else:
                inj.append(None)
        circuits.append(bell_instance(n, cov, ks, inj, seed_state))
        meta.append(('bell', ci, tuple(ks), tuple(inj)))

    print(f'submitting {len(circuits)} circuits x {SHOTS} shots '
          f'= {len(circuits)*SHOTS:,} shots (eta_inj = {ETA})')
    tqs = transpile(circuits, backend=bk, initial_layout=qubits,
                    optimization_level=1, seed_transpiler=7)
    n2 = [sum(1 for i in c.data if len(i.qubits) == 2) for c in tqs]
    print(f'transpiled: depth {min(c.depth() for c in tqs)}-{max(c.depth() for c in tqs)}, '
          f'2q gates {min(n2)}-{max(n2)}')

    sampler = SamplerV2(mode=bk)
    job = sampler.run(tqs, shots=SHOTS)
    print(f'job id {job.job_id()} submitted; waiting...')
    res = job.result()
    print('done.')

    # ---- estimate single-site Paulis from the product arm only (smoke-level check)
    acc = {k: [0.0, 0] for k in single_ref}
    for r, m in zip(res, meta):
        if m[0] != 'prod':
            continue
        bases = m[1]
        counts = r.data.c.get_counts()
        for key, ct in counts.items():
            sg = bits_of(key, n)
            for q in range(n):
                k = (q, bases[q])
                acc[k][0] += sg[q] * ct
                acc[k][1] += ct
    print()
    print('=' * 92)
    print('SMOKE CHECK -- single-site Pauli estimates from the product arm vs ideal simulation')
    print('=' * 92)
    print(f"    {'coord':>10s} {'measured':>10s} {'ideal':>10s} {'diff':>9s} {'shots':>7s}")
    devs = []
    for (q, a), (s, ns) in sorted(acc.items()):
        if ns == 0:
            continue
        est = s / ns
        ref = single_ref[(q, a)]
        devs.append(abs(est - ref))
        print(f"    q{q} {'xyz'[a]}      {est:10.4f} {ref:10.4f} {est-ref:+9.4f} {ns:7d}")
    print('-' * 92)
    print(f'    mean |deviation| = {np.mean(devs):.4f}   max = {np.max(devs):.4f}')
    print('    (deviations of a few percent are the DEVICE, not the pipeline: readout error is')
    print('     ~0.6% per qubit and the prep circuit carries two CZ layers.)')
    print('=' * 92)
