"""Coverage of the one-sided 95% bootstrap upper bound on the ENDPOINT B = max_j |beta_j| (target-resolved model).

    python c129_endpoint_coverage.py      # -> data/selfcheck/c129_endpoint_coverage.json

Parametric replication of the sealed design with known truth: five targets, 1,200 calibration and 1,200 evaluation
circuits each, 60 shots per circuit, balanced realised signs x = +-1, outcome probability (1 + a x + c)/2. The
calibration response of each target is its fitted target-resolved value; the evaluation response differs so that the
true endpoint B_true takes a chosen value (spread over one mode, as in the data). The pipeline is the production one:
least-squares slope and offset per mode, beta at theta_ref = 0.5, B = max over targets, and the 95th percentile of B
over bootstrap draws that resample calibration and evaluation circuits independently. Coverage = P(UB95 >= B_true).
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(1290)
N_CIRC, SHOTS, NB, REPS, TH = 1200, 60, 400, 300, 0.5
CAL = [(0.98855, 0.00806), (0.97997, 0.00052), (0.97756, -0.00052), (0.95990, 0.00291), (0.95036, 0.00160)]


def simulate(a, c, reps):
    x = np.where(rng.random((reps, N_CIRC)) < 0.5, 1.0, -1.0)
    p = (1 + a * x + c) / 2
    y = 2 * rng.binomial(SHOTS, p) / SHOTS - 1
    return x, y


def fit(x, y):
    xm, ym = x.mean(-1, keepdims=True), y.mean(-1, keepdims=True)
    a = ((x - xm) * (y - ym)).sum(-1) / ((x - xm) ** 2).sum(-1)
    return a, (ym[..., 0] - a * xm[..., 0])


def beta(ae, ce, ah, ch):
    return (ae / ah - 1) * TH + (ce - ch) / ah


def run(b_true_mode, b_true):
    """evaluation response of mode b_true_mode shifted so that its beta equals b_true; others equal calibration"""
    B_pts, UBs = np.zeros(REPS), np.zeros(REPS)
    betas_b = np.zeros((REPS, NB, len(CAL)))
    for j, (a, c) in enumerate(CAL):
        ae, ce = a, c
        if j == b_true_mode:
            ae = a * (1 + b_true / TH)          # pure multiplicative shift: beta = b_true exactly
        xc, yc = simulate(a, c, REPS)
        xe, ye = simulate(ae, ce, REPS)
        ic = rng.integers(0, N_CIRC, (REPS, NB, N_CIRC))
        ie = rng.integers(0, N_CIRC, (REPS, NB, N_CIRC))
        ahb, chb = fit(np.take_along_axis(xc[:, None, :], ic, -1), np.take_along_axis(yc[:, None, :], ic, -1))
        aeb, ceb = fit(np.take_along_axis(xe[:, None, :], ie, -1), np.take_along_axis(ye[:, None, :], ie, -1))
        betas_b[:, :, j] = beta(aeb, ceb, ahb, chb)
    UBs = np.percentile(np.abs(betas_b).max(-1), 95, axis=1)
    true_B = max(abs(b_true), 0.0)
    return dict(b_true=b_true, coverage=float(np.mean(UBs >= true_B)), mean_UB=float(UBs.mean()))


if __name__ == '__main__':
    out = [run(3, b) for b in (0.003, 0.0055, 0.0085)]
    json.dump(out, open(os.path.join(HERE, 'data', 'selfcheck', 'c129_endpoint_coverage.json'), 'w'), indent=1)
    for r in out:
        print(r)
