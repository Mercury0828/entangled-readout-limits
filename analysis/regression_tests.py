"""Executable regression tests: each test pairs a correct analysis implementation with a mutant.

    python regression_tests.py            # exit 0 only if EVERY test passes
    python -m pytest regression_tests.py  # the same tests, pytest-discoverable

THE CONTRACT.  Every test below is a PAIR:
    (a) the CORRECT implementation must satisfy the check, and
    (b) a MUTANT containing a specified analysis fault must FAIL the check.
A test whose mutant passes has no teeth, and is reported as a failure of the suite.

Ground truth comes from a synthetic generator with KNOWN per-mode response (slope a, offset c),
known per-circuit realised signs, per-circuit coverage and a real sampling hierarchy (circuits
containing shots) -- so each check can compare against the truth rather than against the theory.
"""
import inspect
import shutil
import subprocess
import sys

import numpy as np

RESULTS = []


def pair(name, ref, check, correct, mutant, detail=''):
    """Run the correct implementation and its mutant through the same check."""
    ok_c = bool(check(correct))
    ok_m = bool(check(mutant))
    passed = ok_c and not ok_m
    RESULTS.append((name, ref, ok_c, ok_m, passed, detail))
    assert ok_c, f'{name}: CORRECT implementation fails its own check'
    assert not ok_m, f'{name}: MUTANT passes -- the test has no teeth'


# =============================================================================================
# synthetic ground truth
# =============================================================================================
def gen_arm(rng, n_circ, shots, cover_p, a, c, x_circ):
    """Per-circuit coverage and per-shot outcomes for one measurement arm.

    Returns (covered[n_circ], outcomes[n_circ, shots]) with outcomes = 0 where uncovered.
    Pr(+1 | covered, x) = (1 + a*x + c)/2 -- the affine response the calibration must recover."""
    covered = rng.random(n_circ) < cover_p
    p = np.clip(0.5 * (1 + a * x_circ + c), 0, 1)
    out = np.where(rng.random((n_circ, shots)) < p[:, None], 1.0, -1.0)
    out[~covered] = 0.0
    return covered, out


def realised(rng, n_circ, theta):
    """classical-mixture probe: each circuit's realised sign, drawn from the nominal theta"""
    return np.where(rng.random(n_circ) < (1 + theta) / 2, 1.0, -1.0)


# =============================================================================================
# 1. dropping the zeros  (c108: conditioning on coverage inverted the ordering)
# =============================================================================================
def est_keep_zeros(covered, out, mult):
    return float((mult * out).mean())                    # zeros from uncovered circuits kept


def est_drop_zeros(covered, out, mult):
    return float((mult * out[covered]).mean())            # conditional mean: the mutant


def test_01_zeros():
    rng = np.random.default_rng(1)
    theta, a = 0.5, 0.97
    x = realised(rng, 40000, theta)
    cov, out = gen_arm(rng, 40000, 20, 1 / 3, a, 0.0, x)
    truth = a * x.mean()                                  # what an unbiased estimator targets
    check = lambda f: abs(f(cov, out, 3.0) - truth) < 0.01
    pair('01 keep the zeros', 'c108', check, est_keep_zeros, est_drop_zeros)


# =============================================================================================
# 2. pooling the arms instead of inverse-variance weighting  
# =============================================================================================
def combine_ivw(parts):
    """parts: list of (estimate, variance) from independent arms"""
    w = np.array([1 / v for _, v in parts]); e = np.array([m for m, _ in parts])
    return float((w * e).sum() / w.sum())


def combine_pool(parts, fracs):
    """per-shot pooling: arm estimates weighted by the arm probability (the mutant)"""
    return float(sum(f * m for (m, _), f in zip(parts, fracs)))


def test_02_ivw():
    # single-site target: the Bell arm carries NO information about it, so its unbiased
    # estimate is simply absent. Pooling still averages the zeros in, scaling theta by w.
    w, theta = 0.15, 0.5
    prod = (theta, 3.0 / 1000)                            # unbiased product-arm estimate
    parts_correct = [prod]                                # Bell arm contributes nothing
    check = lambda f: abs(f() - theta) < 0.02
    pair('02 inverse-variance, not pooling', 'weighting',
         check,
         lambda: combine_ivw(parts_correct),
         lambda: combine_pool([prod, (0.0, 1e9)], [w, 1 - w]))


# =============================================================================================
# 3. MSE = C/N + bias^2, not C + bias^2 * N  (c108)
# =============================================================================================
def mse_correct(C, b, N):
    return C / N + b * b


def mse_mutant(C, b, N):
    return C + b * b * N


def test_03_mse():
    rng = np.random.default_rng(3)
    C, b = 7.0, 0.05
    emp = {}
    for N in (50, 5000):
        est = rng.normal(b, np.sqrt(C), size=(3000, N)).mean(axis=1)   # R replications
        emp[N] = float(np.mean(est ** 2))                               # truth is 0
    check = lambda f: all(abs(f(C, b, N) - emp[N]) / emp[N] < 0.08 for N in emp)
    pair('03 MSE = C/N + b^2', 'c108', check, mse_correct, mse_mutant)


# =============================================================================================
# 4. the sampling hierarchy: SE must match the replication spread  
# =============================================================================================
def se_cluster(vals):
    """vals[n_circ, shots]: circuits are the resampling unit"""
    per = vals.mean(axis=1)
    return float(per.std(ddof=1) / np.sqrt(len(per)))


def se_shots(vals):
    """every shot treated as independent -- the recurring error"""
    v = vals.ravel()
    return float(v.std(ddof=1) / np.sqrt(v.size))


def test_04_hierarchy():
    rng = np.random.default_rng(4)
    theta, n_circ, shots, R = 0.5, 200, 60, 600
    ests, ses = [], {'c': [], 's': []}
    for _ in range(R):
        x = realised(rng, n_circ, theta)                  # circuit-level randomness
        _, out = gen_arm(rng, n_circ, shots, 1.0, 1.0, 0.0, x)
        ests.append(out.mean()); ses['c'].append(se_cluster(out)); ses['s'].append(se_shots(out))
    ests = np.array(ests)

    def coverage(key):
        return float(np.mean(np.abs(ests - theta) < 1.96 * np.array(ses[key])))

    check = lambda key: 0.92 < coverage(key) < 0.98
    pair('04 SE from the actual sampling hierarchy', 'sampling', check, 'c', 's',
         detail=f"coverage cluster {coverage('c'):.3f}, shot-level {coverage('s'):.3f}")


# =============================================================================================
# 5. score against the REALISED theta, not the nominal  
# =============================================================================================
def test_05_realised():
    rng = np.random.default_rng(5)
    theta, n_circ = 0.2, 900
    x = realised(rng, n_circ, theta)                      # e.g. realises ~0.17, not 0.2
    cov, out = gen_arm(rng, n_circ, 60, 1.0, 1.0, 0.0, x) # NOISELESS: no true bias exists
    est = float(out.mean())
    se = se_cluster(out - x[:, None] * (out != 0))
    ref_real, ref_nom = float(x.mean()), theta
    # on a noiseless device the residual must be consistent with zero
    check = lambda ref: abs(est - ref) < 3 * max(se, 1e-4)
    pair('05 realised reference, not nominal', 'reference', check, ref_real, ref_nom,
         detail=f'realised {ref_real:+.4f} vs nominal {ref_nom:+.4f}')


# =============================================================================================
# 6. fit the SLOPE on the realised sign; do not average residuals over mixed-sign theta  (c109)
# =============================================================================================
def damping_by_slope(x_all, y_all):
    A = np.vstack([x_all, np.ones_like(x_all)]).T
    return float(np.linalg.lstsq(A, y_all, rcond=None)[0][0])


def damping_by_mean_residual(thetas, ys):
    """infer a from the mean residual averaged over thetas -- cancels for mixed signs"""
    t = np.array(thetas); y = np.array(ys)
    return float(1 + np.mean(y - t) / np.mean(t))


def test_06_slope():
    rng = np.random.default_rng(6)
    a = 0.955
    thetas = [0.5, -0.5, 0.2]
    xs, ys, cell_t, cell_y = [], [], [], []
    for th in thetas:
        x = realised(rng, 3000, th)
        _, out = gen_arm(rng, 3000, 60, 1.0, a, 0.0, x)
        y = out.mean(axis=1)
        xs.append(x); ys.append(y); cell_t.append(x.mean()); cell_y.append(y.mean())
    xa, ya = np.concatenate(xs), np.concatenate(ys)
    check = lambda f: abs(f() - a) < 0.005
    pair('06 slope on realised sign, not mean residual', 'c109', check,
         lambda: damping_by_slope(xa, ya),
         lambda: damping_by_mean_residual(cell_t, cell_y))


# =============================================================================================
# 7. interleave calibration with evaluation, or drift biases the prediction  
# =============================================================================================
def test_07_interleave():
    rng = np.random.default_rng(7)
    n = 12000
    t = np.linspace(0, 1, n)
    a_t = 0.97 - 0.03 * t                                 # linear drift over the job
    x = realised(rng, n, 0.0)
    p = 0.5 * (1 + a_t * x)
    y = np.where(rng.random((n, 60)) < p[:, None], 1.0, -1.0).mean(axis=1)
    is_cal = np.zeros(n, bool)
    first = is_cal.copy(); first[: n // 4] = True         # all calibration FIRST (the mutant)
    inter = is_cal.copy(); inter[rng.permutation(n)[: n // 4]] = True   # interleaved
    eval_truth = float(a_t[~inter].mean())

    def predict(mask):
        return damping_by_slope(x[mask], y[mask])

    check = lambda mask: abs(predict(mask) - float(a_t[~mask].mean())) < 0.004
    pair('07 interleaved calibration', 'drift', check, inter, first,
         detail=f'eval truth {eval_truth:.4f}')


# =============================================================================================
# 8. an isolation ratio must hold everything else fixed, including which qubit  
# =============================================================================================
def decoder_correct(a_single_q2, a_edge_q2q3, a_bell_q3):
    a_q3 = a_edge_q2q3 / a_single_q2
    return a_bell_q3 / a_q3


def decoder_mutant(a_single_q2, a_edge_q2q3, a_bell_q3):
    return a_bell_q3 / a_single_q2                        # divides by the WRONG qubit


def test_08_isolation():
    a_q2, a_q3, a_dec = 0.9922, 0.9840, 0.9685           # truth
    check = lambda f: abs(f(a_q2, a_q2 * a_q3, a_q3 * a_dec) - a_dec) < 1e-6
    pair('08 isolation ratio on the same qubit', 'isolation', check, decoder_correct, decoder_mutant)


# =============================================================================================
# 9. a contrast varying two factors needs a control varying one  
# =============================================================================================
def test_09_confound():
    axis_eff, mode_eff = 0.0027, 0.0040                  # truth
    a_prod_zz, a_prod_yy = 0.976, 0.976 - axis_eff
    a_bell_zz = 0.953
    a_bell_yy = a_bell_zz - axis_eff - mode_eff
    correct = lambda: (a_bell_zz - a_bell_yy) - (a_prod_zz - a_prod_yy)
    mutant = lambda: (a_bell_zz - a_bell_yy)              # raw contrast
    check = lambda f: abs(f() - mode_eff) < 1e-6
    pair('09 subtract the one-factor control', 'control', check, correct, mutant)


# =============================================================================================
# 10. the offset term belongs in the corrected bias  
# =============================================================================================
def beta_correct(a_e, c_e, a_c, c_c, th):
    return (a_e / a_c - 1) * th + (c_e - c_c) / a_c


def beta_mutant(a_e, c_e, a_c, c_c, th):
    return (a_e / a_c - 1) * th


def test_10_offset():
    """The offset term is 0.00094 here. The FIRST version of this test sampled 2e5 outcomes, whose
    standard error (~0.0023) exceeded both its tolerance and the effect itself, so it failed the
    CORRECT formula by chance. The pair contract caught that: an underpowered test cannot reliably
    give 'correct passes, mutant fails'. Rule: the check's noise must be well below the difference
    between the correct and the mutant prediction -- here ~6 sigma."""
    rng = np.random.default_rng(10)
    a_e, c_e, a_c, c_c, th = 0.9555, 0.0034, 0.9530, 0.0025, 0.5
    N = 50_000_000                                        # binomial aggregation keeps this cheap
    n_plus = int(round(N * (1 + th) / 2)); n_minus = N - n_plus
    xbar = (n_plus - n_minus) / N
    corrected = 0.0
    for x, nn in ((+1, n_plus), (-1, n_minus)):
        k = rng.binomial(nn, 0.5 * (1 + a_e * x + c_e))
        obs_mean = 2 * k / nn - 1
        corrected += nn / N * (obs_mean - c_c) / a_c
    emp_bias = corrected - xbar
    se = 1 / np.sqrt(N) / a_c
    gap = abs(beta_correct(a_e, c_e, a_c, c_c, xbar) - beta_mutant(a_e, c_e, a_c, c_c, xbar))
    assert gap > 5 * se, f'test 10 underpowered: gap {gap:.2e} vs se {se:.2e}'
    check = lambda f: abs(f(a_e, c_e, a_c, c_c, xbar) - emp_bias) < 2.5 * se
    pair('10 offset term in the bias', 'offset', check, beta_correct, beta_mutant,
         detail=f'empirical bias {emp_bias:+.5f} (se {se:.1e}); offset term = {gap:.5f} = '
                f'{gap/se:.1f} se')


# =============================================================================================
# 11. the minimax optimum is a KINK: mis-allocation costs first order, asymmetrically  
# =============================================================================================
def C_path(w, r=1.0):
    return max(3.0 / w, 1.0 / (w / 9 + (1 - w) * r / 6))


def cost_direct(w0, dw):
    return C_path(w0 + dw) - C_path(w0)


def cost_quadratic(w0, dw):
    """assumes a smooth minimum: symmetric second-order cost from the HIGH side"""
    h = abs(dw)
    return (C_path(w0 + h) - C_path(w0)) * (dw / h) ** 2


def test_11_kink():
    w0 = 3 / 7
    truth_low = cost_direct(w0, -0.0105)
    check = lambda f: abs(f(w0, -0.0105) - truth_low) / truth_low < 0.2
    pair('11 kink, not a smooth minimum', 'kink', check, cost_direct, cost_quadratic,
         detail=f'true low-side cost {truth_low:.4f}, high-side {cost_direct(w0, 0.0105):.4f}')


# =============================================================================================
# 12. a pipe must not mask the exit code  
# =============================================================================================
def test_12_pipefail():
    bash = shutil.which('bash')
    if bash is None:
        RESULTS.append(('12 exit code survives the pipe', 'pipe', None, None, None, 'SKIP: no bash'))
        return
    cmd = f'"{sys.executable}" -c "import sys; sys.exit(1)" | tail -n 1'
    correct = lambda: subprocess.run([bash, '-c', 'set -o pipefail; ' + cmd]).returncode
    mutant = lambda: subprocess.run([bash, '-c', cmd]).returncode
    check = lambda f: f() != 0                             # a failing script must look failed
    pair('12 exit code survives the pipe', 'pipe', check, correct, mutant)


# =============================================================================================
# 13. Fisher information carries the 1/D  (c96)
# =============================================================================================
def test_13_fisher_norm():
    Z = np.diag([1.0, -1.0])
    E = [np.diag([1.0, 0.0]), np.diag([0.0, 1.0])]        # Z-basis POVM on one qubit

    def F(normalise):
        D = 2
        f = sum(np.trace(e @ Z) ** 2 / np.trace(e) for e in E)
        return f / D if normalise else f

    # measuring the matching basis gives F = 1; averaged over 3 bases that is the 1/3 of theory
    check = lambda norm: abs(F(norm) - 1.0) < 1e-12
    pair('13 Fisher information 1/D', 'c96', check, True, False)


# =============================================================================================
# 14. basis rotations must actually rotate into the named eigenbasis  (c96)
# =============================================================================================
def test_14_rotation():
    Y = np.array([[0, -1j], [1j, 0]])
    correct = np.array([[1, 1], [1j, -1j]]) / np.sqrt(2)
    mutant = np.array([[1, -1j], [1, 1j]]) / np.sqrt(2)  # the mutant

    def is_y_eigenbasis(U):
        return all(np.allclose(Y @ U[:, k], ev * U[:, k]) for k, ev in ((0, 1), (1, -1)))

    pair('14 Y rotation columns are Y eigenvectors', 'c96', is_y_eigenbasis, correct, mutant)


# =============================================================================================
# 15. injected-noise labels must never reach the estimator  
# =============================================================================================
def test_15_labels():
    import c101_threshold_measurement as T

    def leaks(fn):
        src = inspect.getsource(fn)
        return ('inj' in inspect.signature(fn).parameters) or ("'inj'" in src) or (' inj' in src)

    def mutant_analyse(get_counts, meta, inj):             # an estimator that sees the labels
        return inj

    check = lambda fn: not leaks(fn)
    pair('15 injection labels never reach the estimator', 'labels', check, T.analyse, mutant_analyse)


# =============================================================================================
TESTS = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]

if __name__ == '__main__':
    failures = 0
    for t in TESTS:
        try:
            t()
        except AssertionError as e:
            failures += 1
            if not any(r[0].split()[0] == t.__name__.split('_')[1] for r in RESULTS):
                RESULTS.append((t.__name__, '?', None, None, False, str(e)))
    print('=' * 100)
    print('REGRESSION TESTS -- each: correct must pass AND the reintroduced error must be caught')
    print('=' * 100)
    print(f"    {'test':<48s} {'from':<6s} {'correct':>8s} {'mutant':>8s} | verdict")
    print('    ' + '-' * 84)
    for name, ref, oc, om, passed, detail in RESULTS:
        if passed is None:
            v = 'SKIP'
        elif passed:
            v = 'PASS'
        elif oc is False:
            v = '*** CORRECT CODE FAILS ***'
        else:
            v = '*** NO TEETH: mutant passes ***'
        fmt = lambda b: '-' if b is None else ('ok' if b else 'FAIL')
        print(f"    {name:<48s} {ref:<6s} {fmt(oc):>8s} {fmt(om):>8s} | {v}")
        if detail:
            print(f"      {detail}")
    npass = sum(1 for r in RESULTS if r[4])
    nskip = sum(1 for r in RESULTS if r[4] is None)
    print('    ' + '-' * 84)
    print(f"    {npass} passed, {nskip} skipped, {len(RESULTS)-npass-nskip} failed "
          f"of {len(RESULTS)}")
    print('=' * 100)
    sys.exit(0 if npass + nskip == len(RESULTS) and failures == 0 else 1)
