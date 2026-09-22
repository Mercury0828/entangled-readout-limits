"""Negative controls on the PRODUCTION analysis code, with known ground truth.

    python production_tests.py          # slow (Aer); exit 0 only if every check passes

regression_tests.py checks the principles on reference implementations. This file runs the ACTUAL estimators
(c110.per_circuit_pairs / fit / boot, and c112.analyse) on the ACTUAL production circuits and metadata, fed with
counts whose physics is known exactly.

GROUND TRUTH.  Ideal counts from Aer, then a known CLASSICAL READOUT CHANNEL applied per qubit in
numpy: a 0 reads as 1 with probability e0[q], a 1 reads as 0 with probability e1[q]. For a Z
expectation that gives exactly
        a_q = 1 - e0[q] - e1[q]        c_q = e1[q] - e0[q]
and for a two-bit parity with independent errors the paired-regression slope is a_i * a_j. The
channel is deliberately UNEQUAL across qubits and ASYMMETRIC (so offsets are nonzero), a negative control with known
unequal slopes, nonzero offsets, drift and calibration mismatch.

Readout does not depend on the Pauli axis measured, so in this model the axis effect is exactly
zero: prod_edge_zz and prod_edge_yy must agree. That is a built-in control for an axis confound.

CHECKS.
  P1  every mode's slope recovered within 3 bootstrap SE of the known truth
  P2  single-bit offsets recovered within 3 SE
  P3  STEERING: a second, different truth must give a different answer that tracks it
  P4  drift injected across the run must be REPORTED by c112's per-block fit; none injected -> none
  P5  MUTANT: feeding realised labels decoupled from the circuits must destroy the recovery -- so the check has
      power over the production input contract
"""
import sys
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

import c108_measurement_choice as M
import c109_realised_reference as C9
import c110_paired_damping as C10
import c112_targeted_calibration as C12
import c99_run_device as R

n = M.n
SHOTS = 60
LOG = []


def check(name, ok, detail):
    LOG.append((name, bool(ok), detail))
    return bool(ok)


def ideal(circuits):
    sim = AerSimulator()
    r = sim.run(transpile(circuits, sim, optimization_level=0), shots=SHOTS,
                seed_simulator=11).result()
    return [r.get_counts(i) for i in range(len(circuits))]


def readout(counts_list, e0, e1, seed, drift=0.0):
    """Apply the known readout channel. drift scales the flip rates linearly with circuit index
    from 1 to (1 + drift), modelling a device whose readout degrades during the job."""
    rng = np.random.default_rng(seed)
    e0 = np.asarray(e0); e1 = np.asarray(e1)
    N = len(counts_list)
    out = []
    for idx, cnt in enumerate(counts_list):
        scale = 1.0 + drift * idx / max(N - 1, 1)
        keys = list(cnt)
        bits = np.array([[int(k.replace(' ', '')[n - 1 - q]) for q in range(n)] for k in keys])
        shots = np.repeat(bits, [cnt[k] for k in keys], axis=0)
        pflip = np.where(shots == 0, e0 * scale, e1 * scale)
        shots = shots ^ (rng.random(shots.shape) < pflip)
        new = {}
        for row in shots:
            s = ''.join(str(row[n - 1 - p]) for p in range(n))
            new[s] = new.get(s, 0) + 1
        out.append(new)
    return out


def truth(e0, e1):
    a = [1 - e0[q] - e1[q] for q in range(n)]
    c = [e1[q] - e0[q] for q in range(n)]
    return a, c


E0 = [0.010, 0.012, 0.004, 0.009, 0.006, 0.011]      # unequal across qubits
E1 = [0.020, 0.018, 0.010, 0.022, 0.015, 0.019]      # asymmetric -> nonzero offsets
E0b = [2.2 * x for x in E0]                          # a DIFFERENT truth, for the steering test
E1b = [2.2 * x for x in E1]


def c110_block(counts, meta):
    pairs = C10.per_circuit_pairs(lambda i: counts[i], meta)
    res = {}
    for mode, pr in pairs.items():
        if len(pr) < 20:
            continue
        a, c = C10.fit(pr)
        sa, sc = C10.boot(pr)
        res[mode] = (a, sa, c, sc)
    return res


def expected_c110(e0, e1):
    a, c = truth(e0, e1)
    return {'prod_single': (a[2], c[2]),              # sigma_z on q2
            'prod_edge': (a[2] * a[3], None),         # zz parity on (2,3)
            'bell_w1': (a[3], c[3])}                  # zz read on q3 after the decoder


if __name__ == '__main__':
    print('building production circuits (c109.build_with_realised, identical to the device run)...')
    circ9, meta9 = C9.build_with_realised()
    print(f'  {len(circ9)} circuits; running ideal Aer...')
    ideal9 = ideal(circ9)

    print('=' * 100)
    print('P1 / P2  c110 on production metadata, known unequal slopes and offsets')
    print('=' * 100)
    res_a = c110_block(readout(ideal9, E0, E1, seed=1), meta9)
    exp_a = expected_c110(E0, E1)
    for mode, (ta, tc) in exp_a.items():
        if mode not in res_a:
            check(f'P1 {mode} present', False, 'mode missing'); continue
        a, sa, c, sc = res_a[mode]
        za = (a - ta) / sa
        check(f'P1 slope {mode}', abs(za) < 3, f'fit {a:.5f} +- {sa:.5f}  truth {ta:.5f}  z {za:+.2f}')
        if tc is not None:
            zc = (c - tc) / sc
            check(f'P2 offset {mode}', abs(zc) < 3,
                  f'fit {c:+.5f} +- {sc:.5f}  truth {tc:+.5f}  z {zc:+.2f}')

    print('=' * 100)
    print('P3  STEERING: a different truth must give a different, tracking answer')
    print('=' * 100)
    res_b = c110_block(readout(ideal9, E0b, E1b, seed=2), meta9)
    exp_b = expected_c110(E0b, E1b)
    for mode in exp_a:
        if mode not in res_a or mode not in res_b:
            continue
        a1, s1 = res_a[mode][0], res_a[mode][1]
        a2, s2 = res_b[mode][0], res_b[mode][1]
        moved = (a1 - a2) / np.hypot(s1, s2)
        tracks = abs((a2 - exp_b[mode][0]) / s2) < 3
        check(f'P3 {mode} moves', abs(moved) > 5,
              f'truth A {exp_a[mode][0]:.5f} -> fit {a1:.5f};  truth B {exp_b[mode][0]:.5f} '
              f'-> fit {a2:.5f};  separation {moved:+.1f} se')
        check(f'P3 {mode} tracks B', tracks, f'B fit {a2:.5f} vs truth {exp_b[mode][0]:.5f}')

    print('=' * 100)
    print('P5  MUTANT: realised labels decoupled from their circuits (an input error)')
    print('=' * 100)
    rng = np.random.default_rng(5)
    xs = np.array([m[6] if m[0] == 'ev' else 0 for m in meta9])
    shuf = xs.copy(); ev = np.where([m[0] == 'ev' for m in meta9])[0]
    shuf[ev] = rng.permutation(xs[ev])
    meta_bad = [m[:6] + (int(shuf[i]),) if m[0] == 'ev' else m for i, m in enumerate(meta9)]
    res_bad = c110_block(readout(ideal9, E0, E1, seed=1), meta_bad)
    for mode, (ta, _) in exp_a.items():
        if mode in res_bad:
            a, sa, _, _ = res_bad[mode]
            caught = abs((a - ta) / sa) > 10
            check(f'P5 {mode} mutant caught', caught,
                  f'decoupled-label fit {a:+.5f} vs truth {ta:.5f} -- the recovery must fail')

    print('=' * 100)
    print('P4  c112 targeted calibration: drift must be reported when present, absent when not')
    print('=' * 100)
    circ12, meta12 = C12.build()
    print(f'  {len(circ12)} c112 circuits; running ideal Aer...')
    ideal12 = ideal(circ12)
    a_t, c_t = truth(E0, E1)
    exp12 = {'prod_single_z': a_t[2], 'prod_edge_zz': a_t[2] * a_t[3],
             'prod_edge_yy': a_t[2] * a_t[3], 'bell_w1_zz': a_t[3],
             'bell_w2_yy': a_t[2] * a_t[3]}
    for label, drift in (('no drift', 0.0), ('drift +150%', 1.5)):
        counts = readout(ideal12, E0, E1, seed=12, drift=drift)
        out = C12.analyse(lambda i: counts[i], meta12)
        for mode, ta in exp12.items():
            o = out[mode]
            zdrift = o['drift'] / o['sdrift']
            if drift == 0.0:
                z = (o['a'] - ta) / o['sa']
                check(f'P4 {label} slope {mode}', abs(z) < 3,
                      f'fit {o["a"]:.5f} +- {o["sa"]:.5f}  truth {ta:.5f}  z {z:+.2f}')
                check(f'P4 {label} drift {mode}', abs(zdrift) < 3.5,
                      f'drift/block {o["drift"]:+.6f}  z {zdrift:+.2f} (must be ~0)')
            else:
                check(f'P4 {label} drift {mode}', zdrift < -3,
                      f'drift/block {o["drift"]:+.6f}  z {zdrift:+.2f} (degrading readout -> '
                      f'must be significantly NEGATIVE)')
        if drift == 0.0:
            d = out['prod_edge_zz']['a'] - out['prod_edge_yy']['a']
            s = np.hypot(out['prod_edge_zz']['sa'], out['prod_edge_yy']['sa'])
            check('P4 axis control (zz == yy, axis effect is 0 by construction)', abs(d / s) < 3,
                  f'zz - yy = {d:+.5f} +- {s:.5f}')

    print()
    print('=' * 100)
    print('PRODUCTION NEGATIVE CONTROLS -- summary')
    print('=' * 100)
    for name, ok, detail in LOG:
        print(f"    {'PASS' if ok else '*** FAIL ***':<12s} {name}")
        print(f"                 {detail}")
    npass = sum(1 for _, ok, _ in LOG if ok)
    print('    ' + '-' * 84)
    print(f'    {npass} / {len(LOG)} checks passed')
    print('=' * 100)
    sys.exit(0 if npass == len(LOG) else 1)
