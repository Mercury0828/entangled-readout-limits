"""Random-circuit checks of the three budget lemmas of Supplementary Note 3 and 4, with asserts.

    python c125_lemma_checks.py      # exit 0 only if no violation is found

Three qubits (L, v, R). Each circuit is a random sequence of Haar single-qubit unitaries and Haar two-qubit
unitaries on random pairs, each two-qubit unitary followed by the noise channel, then a Haar-random single-qubit
basis measurement on every qubit. Fisher information at rho_0 = I/8 is computed exactly from the effects
(Heisenberg picture), for the 3 single-site and 18 edge coordinates at v.

  touched   : A + B <= 3 r              (pair depolarising)
  untouched : A <= 1 and B <= 2
  Pauli     : A + B <= 3 chi^2           (random two-qubit Pauli channels)
"""
import itertools
import json
import os
import sys

import numpy as np

rng = np.random.default_rng(125)
PM = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.diag([1.0 + 0j, -1])]
I2 = np.eye(2, dtype=complex)
L, V, R = 0, 1, 2


def kron(*ms):
    out = np.array([[1.0 + 0j]])
    for m in ms:
        out = np.kron(out, m)
    return out


def on(site_ops):
    return kron(*[site_ops.get(q, I2) for q in range(3)])


SINGLES = [on({V: P}) for P in PM]
EDGES = [on({L: P, V: Q}) for P in PM for Q in PM] + [on({V: P, R: Q}) for P in PM for Q in PM]


def haar(d):
    z = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / abs(np.diag(r)))


def embed2(U, i, j):
    """two-qubit unitary U on qubits (i, j) of three"""
    perm = [i, j] + [q for q in range(3) if q not in (i, j)]
    full = np.kron(U, I2)
    P = np.zeros((8, 8))
    for b in range(8):
        bits = [(b >> (2 - k)) & 1 for k in range(3)]
        src = [bits[perm.index(q)] for q in range(3)]
        P[int(''.join(map(str, bits)), 2) if False else b, int(''.join(map(str, src)), 2)] = 1
    return P.T @ full @ P


def pauli2(a, b, i, j):
    ops = {i: [I2] + PM, j: [I2] + PM}
    return on({i: ops[i][a], j: ops[j][b]})


def adj_pauli_channel(E, probs, i, j):
    """Heisenberg adjoint of sum_Q p_Q Q . Q on pair (i, j)"""
    out = np.zeros_like(E)
    for (a, b), p in probs.items():
        Q = pauli2(a, b, i, j)
        out += p * Q @ E @ Q
    return out


def depol_probs(eta):
    pr = {(a, b): (1 - eta) / 16 for a in range(4) for b in range(4)}
    pr[(0, 0)] += eta
    return pr


def fisher(effects):
    A = B = 0.0
    for E in effects:
        t = np.trace(E).real
        if t < 1e-14:
            continue
        py = t / 8
        w = E / t
        A += py * sum(np.trace(w @ P).real ** 2 for P in SINGLES)
        B += py * sum(np.trace(w @ P).real ** 2 for P in EDGES)
    return A, B


def random_circuit(depth, allow_v):
    pairs = [(L, V), (V, R), (L, R)] if allow_v else [(L, R)]
    ops = []
    for _ in range(depth):
        if rng.random() < 0.4:
            q = int(rng.integers(0, 3))
            ops.append(('u1', q, haar(2)))
        else:
            i, j = pairs[int(rng.integers(0, len(pairs)))]
            ops.append(('u2', (i, j), haar(4)))
    return ops


def effects_of(ops, probs):
    bases = [haar(2) for _ in range(3)]
    effects = []
    for bits in itertools.product((0, 1), repeat=3):
        proj = {q: np.outer(bases[q][:, b], bases[q][:, b].conj()) for q, b in enumerate(bits)}
        E = on(proj)
        for kind, where, U in reversed(ops):
            if kind == 'u1':
                Uf = on({where: U})
                E = Uf.conj().T @ E @ Uf
            else:
                i, j = where
                E = adj_pauli_channel(E, probs, i, j)
                Uf = embed2(U, i, j)
                E = Uf.conj().T @ E @ Uf
        effects.append(E)
    return effects


def random_pauli_channel():
    x = rng.dirichlet(np.full(16, 0.3)) * rng.uniform(0.02, 0.4)
    p = {(a, b): x[4 * a + b] for a in range(4) for b in range(4)}
    p[(0, 0)] += 1 - sum(x)
    return p


def chi_of(p):
    s = np.sort(np.array(list(p.values())))
    return 1 - 2 * s[:8].sum()


if __name__ == '__main__':
    log, fails = {}, []
    # validity: the untouched product bound and the touched bound are both reachable
    worst_touch, worst_untouch_B, worst_untouch_A, worst_pauli = -9, -9, -9, -9
    for trial in range(150):
        eta = rng.uniform(0.3, 1.0)
        ops = random_circuit(int(rng.integers(1, 6)), True)
        if not any(k == 'u2' and V in w for k, w, _ in ops):
            ops.append(('u2', (L, V), haar(4)))
        A, B = fisher(effects_of(ops, depol_probs(eta)))
        worst_touch = max(worst_touch, A + B - 3 * eta ** 2)
    for trial in range(150):
        ops = random_circuit(int(rng.integers(0, 5)), False)
        A, B = fisher(effects_of(ops, depol_probs(rng.uniform(0.3, 1.0))))
        worst_untouch_A = max(worst_untouch_A, A - 1)
        worst_untouch_B = max(worst_untouch_B, B - 2)
    for trial in range(150):
        p = random_pauli_channel()
        ops = random_circuit(int(rng.integers(1, 5)), True)
        if not any(k == 'u2' and V in w for k, w, _ in ops):
            ops.append(('u2', (V, R), haar(4)))
        A, B = fisher(effects_of(ops, p))
        worst_pauli = max(worst_pauli, A + B - 3 * chi_of(p) ** 2)
    # attainment checks: product measurement of an untouched v gives B = 2 in the Pauli bases; a noiseless Bell
    # decoder on (L, v) followed by Z readout gives A + B = 3
    H = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
    CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], complex)
    Udec = np.kron(H, I2) @ CX
    effects = []
    for bits in itertools.product((0, 1), repeat=3):
        proj = {q: np.diag([1 - b, b]).astype(complex) for q, b in enumerate(bits)}
        E = on(proj)
        Uf = embed2(Udec, L, V)
        effects.append(Uf.conj().T @ E @ Uf)
    Ab, Bb = fisher(effects)
    log = dict(max_touched_excess=worst_touch, max_untouched_A_excess=worst_untouch_A,
               max_untouched_B_excess=worst_untouch_B, max_pauli_excess=worst_pauli, bell_A_plus_B=Ab + Bb)
    ok = (worst_touch < 1e-9 and worst_untouch_A < 1e-9 and worst_untouch_B < 1e-9 and worst_pauli < 1e-9
          and abs(Ab + Bb - 3) < 1e-9)
    json.dump(log, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'selfcheck',
                                     'c125_lemma_checks.json'), 'w'), indent=1)
    for k, v in log.items():
        print(f'  {k:28s} {v:+.3e}')
    print('all lemma checks hold' if ok else 'VIOLATION')
    sys.exit(0 if ok else 1)
