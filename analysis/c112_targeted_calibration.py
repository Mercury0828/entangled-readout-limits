"""Target-resolved, interleaved calibration run (job daobh65r85ps73ff233g; Fig. 3b, turquoise).

    python c112_targeted_calibration.py sim     # must return a = 1.00000 in every mode
    python c112_targeted_calibration.py run     # ibm_cleveland

Every circuit is directed at the target coordinate that the evaluation uses, so the response is resolved per
target; spectator qubits keep random eigenstates and bases. All modes are shuffled into one interleaved sequence
and each circuit records its position, so drift can be measured. The run includes the YY target on edge (2,3),
which the Bell decoder reads through the two-qubit parity (w2); the script asserts from the decoder table which
readout mode each target uses. Each circuit prepares a balanced realised sign x = +-1 and the analysis regresses the
observed parity on it, y = a x + c.
"""
import json
import sys
import numpy as np
from qiskit import QuantumCircuit, transpile

import c108_measurement_choice as M
import c99_run_device as R

RUN_INSTANCE = 'DQC methods'
n = M.n
QUBITS = M.QUBITS
NPER = 1800            # circuits per mode (1500 target + 20% for drift resolution)
SHOTS = 60
NBLOCK = 10
EDGE = (2, 3)
COV_A = M.COVERINGS[0]          # [(0,1),(2,3),(4,5)] -- contains the target edge
assert EDGE in COV_A

# (name, kind, axes, arm)
MODES = [('prod_single_z', 'S', (2,), 'prod'),
         ('prod_edge_zz', 'E', (2, 2), 'prod'),
         ('prod_edge_yy', 'E', (1, 1), 'prod'),
         ('bell_w1_zz', 'E', (2, 2), 'bell'),
         ('bell_w2_yy', 'E', (1, 1), 'bell')]


def bell_row(axes):
    """(k, tag, sign) with which the decoder reads sigma_a x sigma_b on the target dimer."""
    tbl = R.bell_observables()
    for k in range(3):
        for (tag, aa, bb, sgn) in tbl[k]:
            if (aa, bb) == axes:
                return k, tag, sgn
    raise RuntimeError(f'decoder never reads {axes}')


ZZ = bell_row((2, 2)); YY = bell_row((1, 1))
assert ZZ[1] in ('i', 'j'), f'zz must be read through a w1 parity, got tag {ZZ[1]}'
assert YY[1] == 'ij', f'yy must be read through the w2 parity, got tag {YY[1]}'


def build():
    rng = np.random.default_rng(20260921)
    items = []
    for name, kind, axes, arm in MODES:
        for _ in range(NPER):
            x = 1 if rng.random() < 0.5 else -1
            qc = QuantumCircuit(n, n)
            fixed = {}
            if kind == 'S':
                q = 2
                M.prep_eigenstate(qc, q, axes[0], x); fixed[q] = axes[0]
            else:
                i, j = EDGE
                si = 1 if rng.random() < 0.5 else -1
                M.prep_eigenstate(qc, i, axes[0], si); fixed[i] = axes[0]
                M.prep_eigenstate(qc, j, axes[1], x * si); fixed[j] = axes[1]
            for q in range(n):
                if q not in fixed:
                    M.prep_eigenstate(qc, q, int(rng.integers(0, 3)),
                                      1 if rng.random() < 0.5 else -1)
            qc.barrier()
            if arm == 'prod':
                bases = [int(rng.integers(0, 3)) for _ in range(n)]
                for q, a in fixed.items():
                    bases[q] = a                         # DIRECTED on the target
                M.product_meas(qc, tuple(bases))
                info = tuple(bases)
            else:
                k_t = ZZ[0] if axes == (2, 2) else YY[0]
                ks = [int(rng.integers(0, 3)) for _ in COV_A]
                ks[COV_A.index(EDGE)] = k_t              # DIRECTED on the target dimer
                M.bell_meas(qc, COV_A, ks)
                info = tuple(ks)
            items.append((qc, (name, kind, axes, arm, x, info)))
    order = rng.permutation(len(items))                   # INTERLEAVE every mode together
    circuits = [items[i][0] for i in order]
    meta = [items[i][1] + (pos,) for pos, i in enumerate(order)]
    return circuits, meta


def observe(meta_row, counts):
    name, kind, axes, arm, x, info, pos = meta_row
    tot = sum(counts.values()); y = 0.0
    for key, ct in counts.items():
        sg = R.bits_of(key, n)
        if arm == 'prod':
            v = sg[2] if kind == 'S' else sg[EDGE[0]] * sg[EDGE[1]]
        else:
            k, tag, sgn = ZZ if axes == (2, 2) else YY
            i, j = EDGE
            v = sgn * (sg[i] if tag == 'i' else (sg[j] if tag == 'j' else sg[i] * sg[j]))
        y += v * ct
    return y / tot


def fit(xs, ys):
    A = np.vstack([xs, np.ones_like(xs)]).T
    (a, c), *_ = np.linalg.lstsq(A, ys, rcond=None)
    return float(a), float(c)


def analyse(get_counts, meta, rng=np.random.default_rng(1121)):
    rows = {name: [] for name, *_ in MODES}
    for idx, m in enumerate(meta):
        rows[m[0]].append((m[4], observe(m, get_counts(idx)), m[6]))
    N = len(meta)
    out = {}
    print('=' * 100)
    print('TARGET-RESOLVED, INTERLEAVED CALIBRATION')
    print('=' * 100)
    print(f"    decoder check: zz read via tag '{ZZ[1]}' (w1) at k={ZZ[0]}; "
          f"yy via tag '{YY[1]}' (w2) at k={YY[0]}")
    print(f"    {'mode':<16s} {'n':>5s} | {'a':>9s} {'se':>8s} {'c':>9s} {'se':>8s} | "
          f"{'drift slope/block':>18s} {'z':>6s}")
    print('    ' + '-' * 90)
    for name, *_ in MODES:
        arr = np.array(rows[name], float)
        a, c = fit(arr[:, 0], arr[:, 1])
        bs_a, bs_c = [], []
        for _ in range(400):
            s = arr[rng.integers(0, len(arr), len(arr))]
            aa, cc = fit(s[:, 0], s[:, 1]); bs_a.append(aa); bs_c.append(cc)
        sa, sc = float(np.std(bs_a, ddof=1)), float(np.std(bs_c, ddof=1))
        blk = np.minimum((arr[:, 2] * NBLOCK / N).astype(int), NBLOCK - 1)
        ab = []
        for b in range(NBLOCK):
            s = arr[blk == b]
            if len(s) > 10:
                ab.append((b, fit(s[:, 0], s[:, 1])[0]))
        bb = np.array(ab)
        slope = float(np.polyfit(bb[:, 0], bb[:, 1], 1)[0])
        # se of the drift slope, by bootstrap over circuits
        sl = []
        for _ in range(200):
            idx = rng.integers(0, len(arr), len(arr))
            s = arr[idx]; bk = blk[idx]; pts = []
            for b in range(NBLOCK):
                t = s[bk == b]
                if len(t) > 10:
                    pts.append((b, fit(t[:, 0], t[:, 1])[0]))
            p = np.array(pts); sl.append(np.polyfit(p[:, 0], p[:, 1], 1)[0])
        ssl = float(np.std(sl, ddof=1))
        out[name] = dict(a=a, sa=sa, c=c, sc=sc, drift=slope, sdrift=ssl, blocks=bb)
        print(f"    {name:<16s} {len(arr):5d} | {a:9.5f} {sa:8.5f} {c:+9.5f} {sc:8.5f} | "
              f"{slope:+18.6f} {slope/ssl:+6.2f}")
    print('    ' + '-' * 90)
    if 'bell_w1_zz' in out and 'bell_w2_yy' in out:
        d = out['bell_w1_zz']['a'] - out['bell_w2_yy']['a']
        s = (out['bell_w1_zz']['sa'] ** 2 + out['bell_w2_yy']['sa'] ** 2) ** 0.5
        print(f"\n    Mode asymmetry on held-out, target-resolved data: a(w1,zz) - a(w2,yy) = {d:+.5f} "
              f"+- {s:.5f}  (z = {d/s:+.2f})")
    return out


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'sim'
    circuits, meta = build()
    print(f'{len(circuits)} circuits x {SHOTS} shots = {len(circuits)*SHOTS:,} shots, '
          f'{len(MODES)} modes interleaved')
    if mode == 'sim':
        from qiskit_aer import AerSimulator
        sim = AerSimulator()
        r = sim.run(transpile(circuits, sim, optimization_level=0), shots=SHOTS).result()
        gc = lambda i: r.get_counts(i)
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        tok = json.load(open(M.KEYPATH, encoding='utf-8'))['apikey']
        svc = QiskitRuntimeService(channel='ibm_quantum_platform', token=tok,
                                   instance=RUN_INSTANCE)
        u = svc.usage()
        print(f"{RUN_INSTANCE}: {u['usage_remaining_seconds']} s remaining")
        if u['usage_limit_reached']:
            print('no budget -- refusing to submit'); sys.exit(1)
        bk = svc.backend(M.BACKEND)
        tqs = transpile(circuits, backend=bk, initial_layout=QUBITS,
                        optimization_level=1, seed_transpiler=7)
        job = SamplerV2(mode=bk).run(tqs, shots=SHOTS)
        print(f'job {job.job_id()}')
        res = job.result()
        gc = lambda i: res[i].data.c.get_counts()
    print('done.\n')
    analyse(gc, meta)
