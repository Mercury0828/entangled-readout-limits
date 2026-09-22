"""Re-analysis of the engineered-noise sweep as a calibration study (Supplementary Note 6).

  (A) k, the ratio of the fitted Bell response to the injected gate quality, is the primary quantity.
  (B) Uncertainty from a bootstrap over circuits, the independent acquisition units.
  (C) The same coordinates recur at every injected gate quality; their common offset is modelled.
  (D) The one-qubit and parity readout modes of the Bell decoder are fitted separately, which tests whether
      readout error cancels between the arms.
  (E) Fits with an intercept, with axis dependence and with acquisition-time dependence are compared; with
      Pr(Y) = (1 +- (a*theta + c))/2 the Fisher information is a^2/(1-c^2).
  (F) Calibration and evaluation are separated: k is fitted on one subset of gate qualities and predicts the other.

Run through c123_rerun_c106_frozen.py on the frozen counts in data/raw.
"""
import json
import numpy as np

import c99_run_device as R
import c101_threshold_measurement as T

KEYPATH = R.KEYPATH
JOB_C101 = 'dam6td02fm4c73f2gl1g'          # 13,200 pubs -- identified by pub count, not order
INSTANCE = 'QTM161'
NBOOT = 300
rng = np.random.default_rng(1061)

n = T.n
ETAS = T.ETAS
COV = T.COVERINGS
ei = T.ei
NE = T.NE


def collect(get_counts, meta):
    """Per-CIRCUIT sums, so the bootstrap can resample acquisition units.

    prod_s[c][coord], prod_n[c][coord] for product circuit c;
    bell_s[eta][c][coord], bell_n[...] likewise, plus the measured-mode tag of each entry.
    """
    tbl = R.bell_observables()
    prod = []
    bell = {e: [] for e in ETAS}
    mode = {e: [] for e in ETAS}          # per circuit: dict coord -> 'w1' or 'w2'
    for idx, (arm, eta, info) in enumerate(meta):
        s = np.zeros(NE); nn = np.zeros(NE)
        md = {}
        for key, ct in get_counts(idx).items():
            sg = R.bits_of(key, n)
            if arm == 'prod':
                for q in range(n - 1):
                    t = ei[(q, info[q], info[q + 1])]
                    s[t] += sg[q] * sg[q + 1] * ct
                    nn[t] += ct
            else:
                ci, ks = info
                for (i, j), k in zip(COV[ci], ks):
                    for (tag, a, b, sgn) in tbl[k]:
                        par = sg[i] if tag == 'i' else (sg[j] if tag == 'j' else sg[i] * sg[j])
                        t = ei[(i, a, b)]
                        s[t] += sgn * par * ct
                        nn[t] += ct
                        md[t] = 'w2' if tag == 'ij' else 'w1'
        if arm == 'prod':
            prod.append((s, nn))
        else:
            bell[eta].append((s, nn))
            mode[eta].append(md)
    return prod, bell, mode


def agg(units, sel=None):
    S = np.zeros(NE); N = np.zeros(NE)
    idxs = range(len(units)) if sel is None else sel
    for u in idxs:
        s, nn = units[u]
        S += s; N += nn
    return np.where(N > 0, S / np.maximum(N, 1), np.nan), N


def fit_slope(x, y, intercept=False):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if intercept:
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return float(coef[0]), float(coef[1])
    return float(x @ y / (x @ x)), 0.0


if __name__ == '__main__':
    from qiskit_ibm_runtime import QiskitRuntimeService
    tok = json.load(open(KEYPATH, encoding='utf-8'))['apikey']
    svc = QiskitRuntimeService(channel='ibm_quantum_platform', token=tok, instance=INSTANCE)
    job = svc.job(JOB_C101)
    print(f'job {JOB_C101}: {job.status()}')
    res = job.result()
    _c, meta = T.build()
    prod, bell, mode = collect(lambda i: res[i].data.c.get_counts(), meta)
    print(f'{len(prod)} product circuits, '
          f'{ {e: len(bell[e]) for e in ETAS} } Bell circuits\n')

    x_all, _ = agg(prod)

    # ---------------- (A)+(B) k with CIRCUIT-level bootstrap ----------------
    print('=' * 100)
    print('(A,B) k as the primary quantity, with the bootstrap over CIRCUITS not coordinates')
    print('=' * 100)
    print(f"    {'eta':>7s} | {'slope':>9s} {'boot se':>9s} {'old se':>9s} {'ratio':>7s}")
    print('    ' + '-' * 52)
    slopes, boot_se, old_se = {}, {}, {}
    boot_draws = {}
    for eta in ETAS:
        y_all, _ = agg(bell[eta])
        s0, _ = fit_slope(x_all, y_all)
        slopes[eta] = s0
        ok = np.isfinite(x_all) & np.isfinite(y_all)
        resid = y_all[ok] - s0 * x_all[ok]
        old_se[eta] = float(np.sqrt((resid @ resid) / max(ok.sum() - 1, 1)
                                    / (x_all[ok] @ x_all[ok])))
        draws = []
        nb, npd = len(bell[eta]), len(prod)
        for _ in range(NBOOT):
            yb, _ = agg(bell[eta], rng.integers(0, nb, nb))
            xb, _ = agg(prod, rng.integers(0, npd, npd))
            sb, _ = fit_slope(xb, yb)
            draws.append(sb)
        draws = np.array(draws)
        boot_draws[eta] = draws
        boot_se[eta] = float(draws.std(ddof=1))
        print(f"    {eta:7.4f} | {s0:9.5f} {boot_se[eta]:9.5f} {old_se[eta]:9.5f} "
              f"{boot_se[eta]/old_se[eta]:7.2f}")
    print('    ' + '-' * 52)
    print(f"    mean ratio boot/old = "
          f"{np.mean([boot_se[e]/old_se[e] for e in ETAS]):.2f}  "
          f"(>1 means our published error was OPTIMISTIC, <1 that it was conservative)")

    # ---------------- (C) common-mode covariance across eta ----------------
    print()
    print('=' * 100)
    print('(C) common-mode component: the SAME 45 coordinates recur at every eta')
    print('=' * 100)
    inj = np.array(ETAS)
    sl = np.array([slopes[e] for e in ETAS])
    ratios = sl / inj
    # a common multiplicative offset moves all ratios together; estimate its share
    within = np.mean([boot_se[e] / inj[i] for i, e in enumerate(ETAS)])
    between = float(ratios.std(ddof=1))
    print(f"    per-eta ratio k_i = slope/eta_inj : mean {ratios.mean():.5f}  sd {between:.5f}")
    print(f"    mean within-eta bootstrap se on the ratio          : {within:.5f}")
    print(f"    -> between-eta scatter is {'SMALLER' if between < within else 'LARGER'} than "
          f"within-eta noise")
    print(f"    A common-mode offset shared by all eta is NOT constrained by the between-eta")
    print(f"    scatter at all. Naive pooling gives se = {between/np.sqrt(len(ETAS)):.5f};")
    print(f"    an unmodelled common-mode term of even 0.5% would dominate that entirely.")

    # ---------------- (D) the SPAM-cancellation hypothesis ----------------
    print()
    print('=' * 100)
    print('(D) TESTING (not assuming) SPAM cancellation: weight-1 vs weight-2 measured modes')
    print('=' * 100)
    print('    If readout error cancelled between the arms, the slope would be the SAME for the')
    print('    Bell correlations read out as single-qubit Z (w1) and as a two-qubit parity (w2).')
    print(f"    {'eta':>7s} | {'k(w1)':>9s} {'k(w2)':>9s} {'diff':>9s} {'boot se':>9s} {'z':>6s}")
    print('    ' + '-' * 60)
    zs = []
    for eta in ETAS:
        md = mode[eta]
        w1 = set(); w2 = set()
        for d in md:
            for t, m in d.items():
                (w1 if m == 'w1' else w2).add(t)
        w2 -= w1           # a coordinate seen in both modes is ambiguous; drop from the contrast
        m1 = np.array(sorted(w1)); m2 = np.array(sorted(w2))
        if len(m1) < 5 or len(m2) < 5:
            print(f'    {eta:7.4f} | too few coordinates per mode ({len(m1)}/{len(m2)})')
            continue
        y_all, _ = agg(bell[eta])
        k1, _ = fit_slope(x_all[m1], y_all[m1])
        k2, _ = fit_slope(x_all[m2], y_all[m2])
        dd = []
        nb, npd = len(bell[eta]), len(prod)
        for _ in range(120):
            yb, _ = agg(bell[eta], rng.integers(0, nb, nb))
            xb, _ = agg(prod, rng.integers(0, npd, npd))
            a1, _ = fit_slope(xb[m1], yb[m1]); a2, _ = fit_slope(xb[m2], yb[m2])
            dd.append(a1 - a2)
        se = float(np.std(dd, ddof=1))
        z = (k1 - k2) / se if se > 0 else np.nan
        zs.append(z)
        print(f"    {eta:7.4f} | {k1:9.5f} {k2:9.5f} {k1-k2:+9.5f} {se:9.5f} {z:+6.2f}")
    print('    ' + '-' * 60)
    if zs:
        zs = np.array(zs)
        print(f"    |z| max = {np.abs(zs).max():.2f}, mean z = {zs.mean():+.2f}  -> "
              f"{'NO evidence against cancellation' if np.abs(zs).max() < 3 else '*** MODE-DEPENDENT SLOPE: cancellation FAILS ***'}")

    # ---------------- (E) intercept, and acquisition-time dependence ----------------
    print()
    print('=' * 100)
    print('(E) intercept and drift')
    print('=' * 100)
    print(f"    {'eta':>7s} | {'k (no int.)':>11s} {'k (w/ int.)':>11s} {'intercept':>10s} "
          f"{'int. se':>9s} | {'k 1st half':>10s} {'k 2nd half':>10s}")
    print('    ' + '-' * 82)
    for eta in ETAS:
        y_all, _ = agg(bell[eta])
        k0, _ = fit_slope(x_all, y_all)
        k1, c1 = fit_slope(x_all, y_all, intercept=True)
        cc = []
        nb, npd = len(bell[eta]), len(prod)
        for _ in range(120):
            yb, _ = agg(bell[eta], rng.integers(0, nb, nb))
            xb, _ = agg(prod, rng.integers(0, npd, npd))
            _, c = fit_slope(xb, yb, intercept=True)
            cc.append(c)
        cse = float(np.std(cc, ddof=1))
        half = len(bell[eta]) // 2
        ya, _ = agg(bell[eta], range(half))
        yb2, _ = agg(bell[eta], range(half, len(bell[eta])))
        ka, _ = fit_slope(x_all, ya)
        kb, _ = fit_slope(x_all, yb2)
        print(f"    {eta:7.4f} | {k0:11.5f} {k1:11.5f} {c1:+10.5f} {cse:9.5f} | "
              f"{ka:10.5f} {kb:10.5f}")

    # ---------------- (F) calibration / evaluation split ----------------
    print()
    print('=' * 100)
    print('(F) held-out check: fit k on half the eta points, predict the other half')
    print('=' * 100)
    cal = [0, 2, 4, 6, 8]
    ev = [1, 3, 5, 7]
    kc = float(np.mean([slopes[ETAS[i]] / ETAS[i] for i in cal]))
    print(f"    calibration eta: {[ETAS[i] for i in cal]}  ->  k_cal = {kc:.5f}")
    print(f"    {'eta':>7s} | {'predicted':>10s} {'observed':>10s} {'resid':>9s} {'boot se':>9s}")
    print('    ' + '-' * 52)
    for i in ev:
        e = ETAS[i]
        pred = kc * e
        obs = slopes[e]
        print(f"    {e:7.4f} | {pred:10.5f} {obs:10.5f} {obs-pred:+9.5f} {boot_se[e]:9.5f}")

    # ---------------- derived threshold, clearly labelled ----------------
    print()
    print('=' * 100)
    print('DERIVED quantity (not a measurement): calibration-implied crossover')
    print('=' * 100)
    w = np.array([1.0 / boot_se[e] ** 2 for e in ETAS])
    k = float((w * inj * sl).sum() / (w * inj * inj).sum())
    kse = float(np.sqrt(1.0 / (w * inj * inj).sum()))
    print(f'    PRIMARY MEASURED   k    = {k:.5f} +- {kse:.5f}   (circuit-level bootstrap)')
    print(f'    DERIVED            eta* = sqrt(2/3)/k = {np.sqrt(2/3)/k:.5f} '
          f'+- {np.sqrt(2/3)*kse/k**2:.5f}')
    print(f'    label this "calibration-implied crossover under the assumed measurement/noise')
    print(f'    model", NOT an independent determination of the minimax threshold.')
    print('=' * 100)


# =============================================================================================
# SYSTEMATIC BUDGET -- the point of the whole re-analysis.
#
# The statistical error is NOT the dominant one. From the runs above:
#   mode dependence   mean |k(w1) - k(w2)| = 0.0291   -> half-spread systematic +-0.0146
#   acquisition drift mean |1st - 2nd half| = 0.0188  -> half-spread systematic +-0.0094
#   intercept         mean +0.00267, POSITIVE at all nine eta
#
#   k    = 0.99925 +- 0.00267 (stat) +- 0.01732 (syst)  =  +-0.01753 total
#   eta* = 0.81711 +- 0.00218 (stat) +- 0.01417 (syst)  =  +-0.01433 total
#
# Against the predicted 0.816497 that is 0.04 sigma -- still consistent -- but the published
# +-0.00397 UNDERSTATED the uncertainty by a factor 3.6. This is a ~1.8% determination of k,
# not a 0.5% one.
# =============================================================================================
