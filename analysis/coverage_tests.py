"""Close the two limits production_tests.py stated.

    python coverage_tests.py        # exit 0 only if both parts pass

LIMIT 2 -- COVERAGE.  A single run within 3 sigma does not show that the production bootstrap
intervals are right. Here the PRODUCTION error estimator c110.boot (c112 uses the same
circuit-bootstrap algorithm inline) is run on R independent synthetic replications, and we count
how often the truth lies inside a +-1.96 SE interval. Correct errors give ~95%.

The generator is exact and fast, and keeps the real sampling hierarchy: circuits containing 60
shots each; a realised sign x per circuit; for two-bit parities a random individual sign s_i per
circuit (s_j = x s_i), whose readout offsets add circuit-level variance exactly as in the device
data; and an optional decoder damping d for the Bell modes. Truths:
    single bit            a = 1 - e0 - e1
    two-bit parity        a = a_i a_j
    Bell, one decoded bit d a_j
    Bell, decoded parity  d a_i a_j
A shot-level SE is computed alongside, as the naive error, and must UNDER-cover.

LIMIT 1 -- GATE NOISE.  The readout-only channel never exercised the decoder's own damping. Here
Aer runs the production c112 circuits with a KNOWN two-qubit depolarising channel on every CX,
    rho -> (1 - lam) rho + lam I/4 ,
which damps every non-identity two-qubit Pauli by (1 - lam); the decoder's trailing H is a
single-qubit gate and does not change that factor. Then the readout channel. Truth: product modes
unchanged, Bell w1 = (1 - lam) a3, Bell w2 = (1 - lam) a2 a3. c112.analyse must recover them.
"""
import sys
import numpy as np

import c110_paired_damping as C10
import c112_targeted_calibration as C12
import production_tests as P

LOG = []


def check(name, ok, detail):
    LOG.append((name, bool(ok), detail))


# =============================================================================================
# part 1 -- coverage of the production bootstrap SE
# =============================================================================================
def read_bits(rng, ideal_pm, e0, e1):
    """ideal_pm: +-1 ideal outcomes (+1 == bit 0). Apply the asymmetric readout channel."""
    bit0 = ideal_pm > 0
    flip = np.where(bit0, rng.random(ideal_pm.shape) < e0, rng.random(ideal_pm.shape) < e1)
    return np.where(flip, -ideal_pm, ideal_pm)


def gen_mode(rng, kind, n_circ, shots, e, d=1.0):
    """kind in {'single', 'parity'}; e = [(e0,e1)] per bit read; d = decoder damping."""
    x = np.where(rng.random(n_circ) < 0.5, 1.0, -1.0)
    if kind == 'single':
        # decoded (or directly measured) ideal value is x with prob (1+d)/2 per shot
        ideal = np.where(rng.random((n_circ, shots)) < (1 + d) / 2, 1.0, -1.0) * x[:, None]
        y = read_bits(rng, ideal, *e[0]).mean(axis=1)
    else:
        si = np.where(rng.random(n_circ) < 0.5, 1.0, -1.0)
        keep = np.where(rng.random((n_circ, shots)) < (1 + d) / 2, 1.0, -1.0)
        bi = np.repeat(si[:, None], shots, axis=1)
        bj = (x * si)[:, None] * keep                     # damping acts on the parity
        y = (read_bits(rng, bi, *e[0]) * read_bits(rng, bj, *e[1])).mean(axis=1)
    return [(float(xx), float(yy), shots) for xx, yy in zip(x, y)]


def se_shots(pairs, shots):
    """shot-level SE: pretends n_circ*shots independent outcomes -- the naive error"""
    arr = np.array(pairs, float)
    x, y = arr[:, 0], arr[:, 1]
    A = np.vstack([x, np.ones_like(x)]).T
    (a, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    resid_shot_var = 1.0 - (a * x + c) ** 2                # per-shot binary variance
    return float(np.sqrt(resid_shot_var.mean() / (len(x) * shots) / np.mean(x * x)))


def coverage_part():
    rng = np.random.default_rng(2026)
    e2, e3 = (0.004, 0.010), (0.009, 0.022)               # q2, q3 readout, from production_tests
    a2, a3 = 1 - sum(e2), 1 - sum(e3)
    d = 0.9685                                             # the measured decoder loss, as a truth
    modes = [('prod_single_z', 'single', [e2], 1.0, a2),
             ('prod_edge_zz', 'parity', [e2, e3], 1.0, a2 * a3),
             ('bell_w1_zz', 'single', [e3], d, d * a3),
             ('bell_w2_yy', 'parity', [e2, e3], d, d * a2 * a3)]
    R, n_circ, shots = 300, 1800, 60
    print('=' * 100)
    print(f'COVERAGE of the production bootstrap SE (c110.boot): {R} replications per mode')
    print('=' * 100)
    print(f"    {'mode':<16s} {'truth':>8s} | {'cover(boot)':>12s} {'cover(shot)':>12s} | "
          f"{'mean se boot':>12s} {'sd of a_hat':>12s} {'ratio':>6s}")
    print('    ' + '-' * 90)
    for name, kind, e, dd, truth in modes:
        hits_b = hits_s = 0
        ahat, seb = [], []
        for _ in range(R):
            pr = gen_mode(rng, kind, n_circ, shots, e, dd)
            a, _c = C10.fit(pr)
            sb, _sc = C10.boot(pr)                        # <-- PRODUCTION function
            ss = se_shots(pr, shots)
            hits_b += abs(a - truth) < 1.96 * sb
            hits_s += abs(a - truth) < 1.96 * ss
            ahat.append(a); seb.append(sb)
        cb, cs = hits_b / R, hits_s / R
        sd = float(np.std(ahat, ddof=1)); ms = float(np.mean(seb))
        # binomial band for R = 300 at p = 0.95: +-2.5 sd = +-0.031
        check(f'coverage {name}', 0.92 <= cb <= 0.98,
              f'bootstrap covers {cb:.3f} (shot-level {cs:.3f}); se/sd = {ms/sd:.3f}')
        print(f"    {name:<16s} {truth:8.5f} | {cb:12.3f} {cs:12.3f} | {ms:12.5f} {sd:12.5f} "
              f"{ms/sd:6.3f}")
    print('    ' + '-' * 90)
    print('    ratio = mean reported SE / actual spread of the estimate; 1 is right')


# =============================================================================================
# part 2 -- gate noise through Aer, on the production c112 circuits
# =============================================================================================
def gate_noise_part():
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    lam = 0.03
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(lam, 2), ['cx'])
    circ, meta = C12.build()
    print()
    print('=' * 100)
    print(f'GATE NOISE: Aer, depolarising lam = {lam} on every CX, + readout channel, c112.analyse')
    print('=' * 100)
    sim = AerSimulator(noise_model=nm)
    tq = transpile(circ, sim, basis_gates=['cx', 'h', 's', 'sdg', 'x', 'y', 'z'],
                   optimization_level=0)
    r = sim.run(tq, shots=P.SHOTS, seed_simulator=31).result()
    ideal = [r.get_counts(i) for i in range(len(circ))]
    counts = P.readout(ideal, P.E0, P.E1, seed=33)
    out = C12.analyse(lambda i: counts[i], meta)
    a, _ = P.truth(P.E0, P.E1)
    exp = {'prod_single_z': a[2], 'prod_edge_zz': a[2] * a[3], 'prod_edge_yy': a[2] * a[3],
           'bell_w1_zz': (1 - lam) * a[3], 'bell_w2_yy': (1 - lam) * a[2] * a[3]}
    for mode, t in exp.items():
        o = out[mode]; z = (o['a'] - t) / o['sa']
        check(f'gate-noise {mode}', abs(z) < 3,
              f"fit {o['a']:.5f} +- {o['sa']:.5f}  truth {t:.5f}  z {z:+.2f}")
    # the decoder factor, recovered the way the paper reports it (same-qubit isolation)
    dec = out['bell_w1_zz']['a'] / (out['prod_edge_zz']['a'] / out['prod_single_z']['a'])
    check('gate-noise decoder factor (1 - lam), same-qubit isolation', abs(dec - (1 - lam)) < 0.006,
          f'recovered {dec:.5f} vs truth {1-lam:.5f}')


if __name__ == '__main__':
    coverage_part()
    gate_noise_part()
    print()
    print('=' * 100)
    print('SUMMARY')
    print('=' * 100)
    for name, ok, detail in LOG:
        print(f"    {'PASS' if ok else '*** FAIL ***':<12s} {name}")
        print(f"                 {detail}")
    npass = sum(ok for _, ok, _ in LOG)
    print(f'    {npass} / {len(LOG)} checks passed')
    print('=' * 100)
    sys.exit(0 if npass == len(LOG) else 1)
