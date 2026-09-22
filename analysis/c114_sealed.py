"""The prospective transfer test: staged acquisition, prediction and evaluation.

    python c114_sealed.py sim        # full three-stage dry run on a synthetic device of known truth
    python c114_sealed.py acquire    # ibm_cleveland: submit, save RAW counts, print NO analysis
    python c114_sealed.py predict    # CALIBRATION circuits only -> predictions.json (commit it)
    python c114_sealed.py evaluate   # refuses unless predictions.json is committed and unmodified

WHY THREE STAGES.  The predictions are computed from the calibration circuits alone and committed to git BEFORE any
evaluation circuit is read, and the evaluate stage enforces that in code.

WHAT WAS FIXED IN ADVANCE:
  * one job, every circuit interleaved in one shuffled order, seed 20260922
  * CALIBRATION: a directed grid over every qubit/axis (product single), every edge/axis pair
    (product edge) and every Bell dimer/setting/readout (Bell) -- ONE dataset feeding BOTH models
  * EVALUATION: a separate, disjoint set of directed circuits at the five evaluation targets
  * estimator: paired regression y_c = a x_c + c on the realised sign (as in c110/c112)
  * pooled model: unweighted mean of per-combination (a, c) over all combinations of the mode
  * target-resolved model: (a, c) fitted on the target combination only
  * residual bias: beta = (a_eval/a_hat - 1) theta_ref + (c_eval - c_hat)/a_hat, theta_ref = 0.5
  * endpoint: B = max over the five targets of |beta|
  * uncertainty: circuit bootstrap, 1000 resamples, calibration and evaluation resampled
    independently; one-sided 95% upper bound = 95th percentile of the bootstrap B
  * tolerance tau = 0.0085: bias^2 <= 10% of the MSE of the best mixture (C = 7.19) at N = 1e4
  * a model is ADEQUATE iff UB95(B) < tau
  * exclusions: none. A job returning fewer circuits or shots than submitted aborts the analysis.
"""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile

import c108_measurement_choice as M
import c99_run_device as R

SEED = 20260922
SHOTS = 60
N_TARGET = 1200            # calibration circuits per TARGET combination
N_OTHER = 80               # calibration circuits per non-target combination
N_EVAL = 1200              # evaluation circuits per target
THETA_REF = 0.5
TAU = 0.0085
NBOOT = 1000
RUN_INSTANCE = 'DQC methods'
n = M.n
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data', 'sealed')
RAW = os.path.join(DATA, 'c114_raw.json')
PRED = os.path.join(DATA, 'c114_predictions.json')
COV = {'A': [(0, 1), (2, 3), (4, 5)], 'B': [(1, 2), (3, 4)]}
EDGES = [(q, q + 1) for q in range(n - 1)]
TBL = R.bell_observables()

# the five evaluation targets: (mode, family, combination key)
TARGETS = [('prod_single_z', 'S', (2, 2)),
           ('prod_edge_zz', 'E', ((2, 3), (2, 2))),
           ('prod_edge_yy', 'E', ((2, 3), (1, 1))),
           ('bell_w1_zz', 'B', ('A', (2, 3), 0, 'j')),
           ('bell_w2_yy', 'B', ('A', (2, 3), 0, 'ij'))]
POOL = {'prod_single_z': 'S', 'prod_edge_zz': 'E', 'prod_edge_yy': 'E',
        'bell_w1_zz': 'Bw1', 'bell_w2_yy': 'Bw2'}


def all_combos():
    S = [(q, a) for q in range(n) for a in range(3)]
    E = [(e, (a, b)) for e in EDGES for a in range(3) for b in range(3)]
    B = []
    for cv, dims in COV.items():
        for d in dims:
            for k in range(3):
                for (tag, aa, bb, sgn) in TBL[k]:
                    B.append((cv, d, k, tag))
    return S, E, B


def bell_axes(k, tag):
    for (t, aa, bb, sgn) in TBL[k]:
        if t == tag:
            return aa, bb, sgn
    raise KeyError((k, tag))


def make(rng, fam, key):
    """one directed circuit: prepare a realised sign x on the combination, randomise the rest"""
    x = 1 if rng.random() < 0.5 else -1
    qc = QuantumCircuit(n, n)
    fixed = {}
    if fam == 'S':
        q, a = key
        M.prep_eigenstate(qc, q, a, x); fixed[q] = a
    elif fam == 'E':
        (i, j), (a, b) = key
        si = 1 if rng.random() < 0.5 else -1
        M.prep_eigenstate(qc, i, a, si); fixed[i] = a
        M.prep_eigenstate(qc, j, b, x * si); fixed[j] = b
    else:
        cv, (i, j), k, tag = key
        aa, bb, _ = bell_axes(k, tag)
        si = 1 if rng.random() < 0.5 else -1
        M.prep_eigenstate(qc, i, aa, si); fixed[i] = aa
        M.prep_eigenstate(qc, j, bb, x * si); fixed[j] = bb
    for q in range(n):
        if q not in fixed:
            M.prep_eigenstate(qc, q, int(rng.integers(0, 3)), 1 if rng.random() < 0.5 else -1)
    qc.barrier()
    if fam in ('S', 'E'):
        bases = [int(rng.integers(0, 3)) for _ in range(n)]
        for q, a in fixed.items():
            bases[q] = a
        M.product_meas(qc, tuple(bases))
    else:
        cv, d, k, tag = key
        dims = COV[cv]
        ks = [int(rng.integers(0, 3)) for _ in dims]
        ks[dims.index(d)] = k
        M.bell_meas(qc, dims, ks)
    return qc, x


def build():
    rng = np.random.default_rng(SEED)
    S, E, B = all_combos()
    tkeys = {(f, k) for _, f, k in TARGETS}
    items = []
    for fam, combos in (('S', S), ('E', E), ('B', B)):
        for key in combos:
            reps = N_TARGET if (fam, key) in tkeys else N_OTHER
            for _ in range(reps):
                qc, x = make(rng, fam, key)
                items.append((qc, dict(role='cal', fam=fam, key=key, x=x)))
    for mode, fam, key in TARGETS:
        for _ in range(N_EVAL):
            qc, x = make(rng, fam, key)
            items.append((qc, dict(role='eval', fam=fam, key=key, x=x, mode=mode)))
    order = rng.permutation(len(items))                  # ONE interleaved sequence
    circuits = [items[i][0] for i in order]
    meta = [dict(items[i][1], pos=int(p)) for p, i in enumerate(order)]
    return circuits, meta


def observe(m, counts):
    tot = sum(counts.values()); y = 0.0
    fam, key = m['fam'], m['key']
    for s, ct in counts.items():
        sg = R.bits_of(s, n)
        if fam == 'S':
            v = sg[key[0]]
        elif fam == 'E':
            (i, j), _ = key; v = sg[i] * sg[j]
        else:
            cv, (i, j), k, tag = key
            _, _, sgn = bell_axes(k, tag)
            v = sgn * (sg[i] if tag == 'i' else (sg[j] if tag == 'j' else sg[i] * sg[j]))
        y += v * ct
    return y / tot


def fit(xy):
    A = np.vstack([xy[:, 0], np.ones(len(xy))]).T
    (a, c), *_ = np.linalg.lstsq(A, xy[:, 1], rcond=None)
    return float(a), float(c)


def key_str(fam, key):
    return json.dumps([fam, key])


def pool_class(fam, key):
    if fam == 'S':
        return 'S'
    if fam == 'E':
        return 'E'
    return 'Bw2' if key[3] == 'ij' else 'Bw1'


# ------------------------------------------------------------------------------------------
# stage 2: predictions from CALIBRATION circuits only
# ------------------------------------------------------------------------------------------
def predict(meta, ys, rng):
    groups = {}
    for m, y in zip(meta, ys):
        if m['role'] != 'cal':
            continue                                      # evaluation circuits are NOT read
        groups.setdefault(key_str(m['fam'], m['key']), []).append((m['x'], y))
    groups = {k: np.array(v, float) for k, v in groups.items()}
    classes = {}
    for k in groups:
        fam, key = json.loads(k)
        key = tuple(tuple(z) if isinstance(z, list) else z for z in key) if isinstance(key, list) else key
        classes.setdefault(pool_class(fam, key), []).append(k)

    def models(gs):
        fits = {k: fit(v) for k, v in gs.items()}
        out = {}
        for mode, fam, key in TARGETS:
            ks = key_str(fam, key)
            out[('target', mode)] = fits[ks]
            cl = classes[POOL[mode]]
            out[('pooled', mode)] = (float(np.mean([fits[c][0] for c in cl])),
                                     float(np.mean([fits[c][1] for c in cl])))
        return out

    point = models(groups)
    boots = []
    for _ in range(NBOOT):
        gs = {k: v[rng.integers(0, len(v), len(v))] for k, v in groups.items()}
        boots.append(models(gs))
    ser = lambda d: {f'{m}|{md}': list(v) for (m, md), v in d.items()}
    return dict(point=ser(point), boot=[ser(b) for b in boots],
                n_cal=int(sum(len(v) for v in groups.values())))


# ------------------------------------------------------------------------------------------
# stage 3: evaluation against the COMMITTED predictions
# ------------------------------------------------------------------------------------------
def evaluate(meta, ys, pred, rng):
    ev = {}
    for m, y in zip(meta, ys):
        if m['role'] == 'eval':
            ev.setdefault(m['mode'], []).append((m['x'], y))
    ev = {k: np.array(v, float) for k, v in ev.items()}

    def beta(a_e, c_e, a_h, c_h):
        return (a_e / a_h - 1) * THETA_REF + (c_e - c_h) / a_h

    res = {}
    for model in ('target', 'pooled'):
        pt = {mode: pred['point'][f'{model}|{mode}'] for mode, _, _ in TARGETS}
        eps = {mode: fit(ev[mode]) for mode in ev}
        B_point = max(abs(beta(*eps[md], *pt[md])) for md in ev)
        per = {md: beta(*eps[md], *pt[md]) for md in ev}
        Bs = []
        for b in pred['boot']:
            e_b = {md: fit(v[rng.integers(0, len(v), len(v))]) for md, v in ev.items()}
            Bs.append(max(abs(beta(*e_b[md], *b[f'{model}|{md}'])) for md in ev))
        ub = float(np.percentile(Bs, 95))
        res[model] = dict(B=B_point, UB95=ub, adequate=ub < TAU, per_mode=per, eval=eps, pred=pt)
    return res


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def meta_sha(meta):
    """predict/evaluate REBUILD the metadata from the seed; this hash, sealed with the raw counts,
    proves the rebuild is the schedule that actually ran. A mismatch means counts would be
    attributed to the wrong circuits, silently -- so every later stage refuses on it."""
    return hashlib.sha256(json.dumps(meta, sort_keys=True, default=str).encode()).hexdigest()


def committed_clean(path):
    rel = os.path.relpath(path, os.path.dirname(HERE))
    top = os.path.dirname(HERE)
    tracked = subprocess.run(['git', 'ls-files', '--error-unmatch', rel], cwd=top,
                             capture_output=True).returncode == 0
    dirty = subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', rel], cwd=top).returncode != 0
    return tracked and not dirty


def report(res):
    print('=' * 100)
    print(f'SEALED VERDICT  (tau = {TAU}, theta_ref = {THETA_REF}, endpoint = max over 5 targets)')
    print('=' * 100)
    for model in ('target', 'pooled'):
        r = res[model]
        print(f"  {model.upper():<8s} B = {r['B']:.5f}   UB95 = {r['UB95']:.5f}   "
              f"-> {'ADEQUATE' if r['adequate'] else 'NOT ADEQUATE'}")
        for md, b in r['per_mode'].items():
            a_e, c_e = r['eval'][md]; a_h, c_h = r['pred'][md]
            print(f"      {md:<14s} beta {b:+.5f}   a_eval {a_e:.5f} vs a_pred {a_h:.5f}   "
                  f"c_eval {c_e:+.5f} vs c_pred {c_h:+.5f}")
    print('=' * 100)


def synthetic_counts(circuits, meta):
    """dry run: ideal Aer + known gate noise + a readout channel HETEROGENEOUS across qubits, so
    that pooling over qubits is genuinely wrong and target-resolution genuinely right."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    import production_tests as P
    nm = NoiseModel(); nm.add_all_qubit_quantum_error(depolarizing_error(0.03, 2), ['cx'])
    sim = AerSimulator(noise_model=nm)
    tq = transpile(circuits, sim, basis_gates=['cx', 'h', 's', 'sdg', 'x', 'y', 'z'],
                   optimization_level=0)
    r = sim.run(tq, shots=SHOTS, seed_simulator=41).result()
    ideal = [r.get_counts(i) for i in range(len(circuits))]
    e0 = [0.030, 0.012, 0.004, 0.009, 0.025, 0.011]       # q2, q3 good; others much worse
    e1 = [0.045, 0.018, 0.010, 0.022, 0.040, 0.019]
    return P.readout(ideal, e0, e1, seed=43)


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'sim'
    os.makedirs(DATA, exist_ok=True)
    circuits, meta = build()
    ncal = sum(m['role'] == 'cal' for m in meta)
    print(f'{len(circuits)} circuits ({ncal} calibration, {len(meta)-ncal} evaluation) x {SHOTS} '
          f'shots = {len(circuits)*SHOTS:,}')

    if stage == 'sim':
        counts = synthetic_counts(circuits, meta)
        ys = [observe(m, c) for m, c in zip(meta, counts)]
        pred = predict(meta, ys, np.random.default_rng(1))
        pred = json.loads(json.dumps(pred))
        report(evaluate(meta, ys, pred, np.random.default_rng(2)))
        print('dry run only: this synthetic device has heterogeneous readout, so POOLED must be')
        print('NOT ADEQUATE and TARGET must be ADEQUATE, or the design lacks power.')

    elif stage == 'acquire':
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        tok = json.load(open(M.KEYPATH, encoding='utf-8'))['apikey']
        svc = QiskitRuntimeService(channel='ibm_quantum_platform', token=tok, instance=RUN_INSTANCE)
        u = svc.usage()
        print(f"{RUN_INSTANCE}: {u['usage_remaining_seconds']} s remaining")
        if u['usage_limit_reached']:
            print('no budget -- refusing'); sys.exit(1)
        bk = svc.backend(M.BACKEND)
        tqs = transpile(circuits, backend=bk, initial_layout=M.QUBITS, optimization_level=1,
                        seed_transpiler=7)
        job = SamplerV2(mode=bk).run(tqs, shots=SHOTS)
        print(f'job {job.job_id()}')
        res = job.result()
        counts = [res[i].data.c.get_counts() for i in range(len(circuits))]
        if len(counts) != len(circuits) or any(sum(c.values()) != SHOTS for c in counts):
            print('*** job incomplete -- analysis aborted per pre-registration ***'); sys.exit(1)
        json.dump(dict(job=job.job_id(), meta_sha256=meta_sha(meta), counts=counts), open(RAW, 'w'))
        print(f'raw counts sealed in {RAW}  sha256 {sha(RAW)}')
        print('NO analysis performed. Next: commit the raw file, then run "predict".')

    elif stage == 'predict':
        raw = json.load(open(RAW))
        if raw['meta_sha256'] != meta_sha(meta):
            print('*** rebuilt schedule differs from the one that ran -- refusing ***'); sys.exit(1)
        ys = [observe(m, c) if m['role'] == 'cal' else None for m, c in zip(meta, raw['counts'])]
        pred = predict(meta, ys, np.random.default_rng(1))
        pred['raw_sha256'] = sha(RAW); pred['job'] = raw['job']
        json.dump(pred, open(PRED, 'w'))
        print(f'predictions from {pred["n_cal"]} calibration circuits written to {PRED}')
        for k, v in pred['point'].items():
            print(f'    {k:<28s} a = {v[0]:.5f}   c = {v[1]:+.5f}')
        print('Evaluation circuits were NOT read. Commit this file, then run "evaluate".')

    elif stage == 'evaluate':
        if not committed_clean(PRED):
            print('*** predictions file is not committed and clean -- refusing to evaluate ***')
            sys.exit(1)
        rt = subprocess.run([sys.executable, os.path.join(HERE, 'regression_tests.py')],
                            capture_output=True)
        if rt.returncode != 0:
            print('*** regression_tests.py does not pass -- refusing to evaluate ***'); sys.exit(1)
        raw = json.load(open(RAW)); pred = json.load(open(PRED))
        if raw['meta_sha256'] != meta_sha(meta):
            print('*** rebuilt schedule differs from the one that ran -- refusing ***'); sys.exit(1)
        if pred['raw_sha256'] != sha(RAW):
            print('*** raw data changed since the predictions were made -- refusing ***'); sys.exit(1)
        ys = [observe(m, c) for m, c in zip(meta, raw['counts'])]
        report(evaluate(meta, ys, pred, np.random.default_rng(2)))
