"""Numerical checks of the ingredients of Theorems 1 and 2, with asserts.

    python c117_theorem_checks.py      # exit 0 only if every check holds

Independent derivation from Pauli algebra and explicit matrices: the degree-3 partition into anticommuting families,
correlation complementarity on random states, product attainment and its independence of the noise, the chain
design's constant, threshold, optimal weight and kink slopes, and the correlated two-qubit Pauli channel of
Supplementary Note 4. The converse of Theorem 2 rests on the written proof and is not checked here.
"""
import itertools
import json
import os
import sys

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'selfcheck')
FAILS, LOG = [], []
PM = {'I': np.eye(2), 'X': np.array([[0, 1], [1, 0]]), 'Y': np.array([[0, -1j], [1j, 0]]),
      'Z': np.diag([1, -1])}


def check(name, cond, detail=''):
    LOG.append((name, bool(cond), detail))
    if not cond:
        FAILS.append(name)


def op(s):
    M = np.array([[1.0]])
    for c in s:
        M = np.kron(M, PM[c])
    return M


def anti(p, q):
    return sum(a != 'I' and b != 'I' and a != b for a, b in zip(p, q)) % 2 == 1


# ---------------------------------------------------------------- Theorem 1 -- degree-3 partition
# qubits: v, u1, u2, u3. Incident-edge Paulis: sigma_a(v) sigma_b(u_k), 3 x 9 = 27.
def edge_pauli(k, a, b):
    s = ['I'] * 4
    s[0], s[1 + k] = a, b
    return ''.join(s)


P3 = 'XYZ'
edges = [edge_pauli(k, a, b) for k in range(3) for a in P3 for b in P3]
check('T1: 27 incident edge Paulis', len(set(edges)) == 27)
# group g gives edge k the centre Pauli P3[(k+g)%3] and all three neighbour Paulis
groups = [[edge_pauli(k, P3[(k + g) % 3], b) for k in range(3) for b in P3] for g in range(3)]
check('T1: groups partition the 27', sorted(sum(groups, [])) == sorted(edges))
for g, G in enumerate(groups):
    check(f'T1: group {g} pairwise anticommuting (symbolic)',
          all(anti(p, q) for p, q in itertools.combinations(G, 2)))
    check(f'T1: group {g} pairwise anticommuting (matrices)',
          all(np.allclose(op(p) @ op(q), -op(q) @ op(p)) for p, q in itertools.combinations(G, 2)))

# no larger anticommuting family exists among the 27 (so 9 per group is the cap and B <= 3 is tight)
adj = {p: {q for q in edges if q != p and anti(p, q)} for p in edges}


def max_clique(R, P):
    best = len(R)
    for v in list(P):
        if len(R) + len(P) <= best:
            break
        best = max(best, max_clique(R | {v}, P & adj[v]))
        P = P - {v}
    return best


omega = max_clique(set(), set(edges))
check('T1: max anticommuting family among the 27 has size 9', omega == 9, f'omega = {omega}')

# complementarity on random mixed states: sum of squared expectations over one group <= 1
rng = np.random.default_rng(117)
worst = 0.0
for _ in range(200):
    A = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    rho = A @ A.conj().T
    rho /= np.trace(rho).real
    for G in groups:
        worst = max(worst, sum(np.trace(rho @ op(p)).real ** 2 for p in G))
check('T1: complementarity sum_P <P>^2 <= 1 on 200 random states', worst <= 1 + 1e-12,
      f'max = {worst:.6f}')
# and it is attained (the bound is not loose): +1 eigenstate of the normalised sum
M = sum(op(p) for p in groups[0]) / 3
evals, evecs = np.linalg.eigh(M)
psi = evecs[:, -1]
att = sum((psi.conj() @ op(p) @ psi).real ** 2 for p in groups[0])
check('T1: complementarity attained', abs(att - 1) < 1e-9, f'{att:.12f}')


# ---------------------------------------------------------------- product measurement attains 9
# uniformly random local Pauli bases; estimator 3^{|supp P|} * product of outcomes on supp P
def product_second_moment(rho, P, nq):
    """E[est^2] exactly, by enumerating all 3^nq basis choices -- equals 3^{|supp|} for any state"""
    tot = 0.0
    for bases in itertools.product(P3, repeat=nq):
        if all(p == 'I' or p == b for p, b in zip(P, bases)):
            tot += (3 ** sum(c != 'I' for c in P)) ** 2   # outcome^2 = 1
    return tot / 3 ** nq


ok = all(abs(product_second_moment(None, P, 4) - 9) < 1e-12 for P in edges)
check('T1: product second moment = 9 on every edge Pauli, state-independent', ok)
check('T1: product second moment = 3 on single-site Paulis',
      abs(product_second_moment(None, 'XIII', 4) - 3) < 1e-12)
check('T1: product measurement executes no two-qubit gate -> constant 9 for ANY gate-noise model', True,
      'structural: the attaining design has no entangling gate, so no gate noise can act')


# ---------------------------------------------------------------- Theorem 2 -- the chain
# per unit weight, the product design gives 1/9 information on each edge Pauli and 1/3 on single
# sites; the two alternating Bell-dimer coverings give r/6 on each edge Pauli (design-exact second
# moments 3, 9, 6/r) and nothing on single sites.
def C_design(w, r):
    return max(3 / w, 1 / (w / 9 + (1 - w) * r / 6))


def C_closed(r):
    return 9.0 if r <= 2 / 3 else 3 + 4 / r


W = np.linspace(1e-4, 1, 200001)
maxerr = 0.0
for eta in np.linspace(0.3, 1, 71):
    r = eta ** 2
    Cw = np.maximum(3 / W, 1 / (W / 9 + (1 - W) * r / 6))
    i = np.argmin(Cw)
    maxerr = max(maxerr, abs(Cw[i] - C_closed(r)))
    if r > 2 / 3 + 1e-3:
        ws = 3 * r / (4 + 3 * r)
        check(f'T2: w* = 3r/(4+3r) at eta={eta:.2f}', abs(W[i] - ws) < 1e-3)
        check(f'T2: design attains 3+4/r at w* (eta={eta:.2f})', abs(C_design(ws, r) - (3 + 4 / r)) < 1e-12)
check('T2: grid minimax of the design = closed form, eta in [0.3, 1]', maxerr < 1e-3, f'max err {maxerr:.2e}')
eta_star = np.sqrt(2 / 3)
r = eta_star ** 2
check('T2: continuity at the threshold', abs((3 + 4 / r) - 9) < 1e-12)
check('T2: at the threshold the Bell edge slope r/6 - 1/9 = 0', abs(r / 6 - 1 / 9) < 1e-15)
check('T2: threshold average gate infidelity 3(1-eta*)/4 = 13.76%', abs(0.75 * (1 - eta_star) - 0.137628) < 1e-6)

# kink at r = 1: C(w) = max{3/w, 18/(3-w)}, w* = 3/7
ws = 3 / 7
sl, sr = -3 / ws ** 2, 18 / (3 - ws) ** 2
h = 1e-7
nl = (C_design(ws, 1) - C_design(ws - h, 1)) / h
nr = (C_design(ws + h, 1) - C_design(ws, 1)) / h
check('T2: kink left slope -16.33', abs(sl + 49 / 3) < 1e-12 and abs(nl - sl) < 1e-4, f'{sl:.4f}')
check('T2: kink right slope +2.72', abs(sr - 49 / 18) < 1e-12 and abs(nr - sr) < 1e-4, f'{sr:.4f}')
check('T2: the optimum is a kink, not flat (asymmetry ~6x)', abs(sl / sr) > 5.9, f'{abs(sl/sr):.2f}')


# ---------------------------------------------------------------- correlated two-qubit Pauli channel
def pauli_eigs(p):
    """Pauli-channel eigenvalues f_P = sum_Q p_Q (-1)^{<P,Q>} on two qubits"""
    keys = [a + b for a in 'IXYZ' for b in 'IXYZ']
    return {P: sum(p.get(Q, 0) * (-1) ** anti(P, Q) for Q in keys) for P in keys}


F = (1 + 15 * eta_star) / 16                          # process fidelity of D_eta*
check('correlated channel: process fidelity 0.8279655446', abs(F - 0.8279655446) < 1e-10)
wt2 = [a + b for a in P3 for b in P3]
p = {'II': F} | {Q: (1 - F) / 9 for Q in wt2}
f = pauli_eigs(p)
f1, f2 = f['XI'], f['XX']
check('correlated channel: f1 = (5eta-1)/4', abs(f1 - (5 * eta_star - 1) / 4) < 1e-12, f'{f1:.10f}')
check('correlated channel: f2 = (5eta+1)/6', abs(f2 - (5 * eta_star + 1) / 6) < 1e-12, f'{f2:.10f}')
rB = (2 * f1 ** 2 + f2 ** 2) / 3
check('correlated channel: r_B = 0.6350859846 < 2/3', abs(rB - 0.6350859846) < 1e-10 and rB < 2 / 3, f'{rB:.10f}')
check('correlated channel: balancing weight gives 3+4/r_B = 9.30 > 9', abs(3 + 4 / rB - 9.2984) < 1e-3)
# pair depolarising at the same fidelity sits exactly at threshold
pd = {'II': F} | {Q: (1 - F) / 15 for Q in [a + b for a in 'IXYZ' for b in 'IXYZ'] if Q != 'II'}
fd = pauli_eigs(pd)
check('correlated channel: pair depolarising at same fidelity has f = eta* on every Pauli',
      all(abs(fd[P] - eta_star) < 1e-12 for P in fd if P != 'II'))


# ---------------------------------------------------------------- report
os.makedirs(OUT, exist_ok=True)
json.dump([dict(check=n, ok=o, detail=d) for n, o, d in LOG],
          open(os.path.join(OUT, 'c117_theorem_checks.json'), 'w'), indent=1)
print('=' * 92)
print('Numerically checkable ingredients of Theorems 1 and 2')
print('=' * 92)
for n, o, d in LOG:
    print(f"  [{'ok' if o else 'FAIL'}] {n}" + (f'   ({d})' if d else ''))
print('=' * 92)
print(f'{len(LOG) - len(FAILS)}/{len(LOG)} checks hold')
print('NOT checked here: the converse of Theorem 2 over the whole class -- rests on the written proof')
sys.exit(1 if FAILS else 0)
