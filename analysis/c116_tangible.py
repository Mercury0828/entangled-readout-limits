"""The main results expressed in numbers of copies (Supplementary Note 5). Output in data/selfcheck/.

    python c116_tangible.py        # exit 0 only if every assertion holds

Convention: for the pair-depolarising channel D_eta(rho) = eta rho + (1 - eta) I/4 Tr rho the average gate fidelity
is F_avg = (1 + 3 eta)/4, so the average gate infidelity is 3(1 - eta)/4, and a vendor two-qubit gate error (an
average gate infidelity) corresponds to eta = 1 - (4/3) * error.
"""
import json
import os
import sys

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'selfcheck')
EPS = 0.01                      # target precision: RMSE 0.01 on EVERY requested Pauli expectation
VENDOR_2Q_ERR = 0.00168         # ibm_cleveland median CZ error (vendor, RB) -- c95 survey
FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
    return cond


def C_chain(eta):
    r = eta * eta
    return 9.0 if r <= 2 / 3 else 3 + 4 / r


def infid(eta):
    return 3 * (1 - eta) / 4


def eta_from_infid(e):
    return 1 - 4 * e / 3


out = {}

# ---- the threshold, in the units a device specification uses
eta_star = np.sqrt(2 / 3)
out['threshold'] = dict(eta=eta_star, avg_gate_infidelity=infid(eta_star),
                        entanglement_infidelity=(1 - eta_star) * 15 / 16,
                        replacement_probability=1 - eta_star)
check(abs(infid(eta_star) - 0.137628) < 1e-6, 'threshold avg infidelity != 13.7628%')

# ---- the device, with the consistent conversion and with the one we used
eta_dev = eta_from_infid(VENDOR_2Q_ERR)
eta_dev_used = 1 - VENDOR_2Q_ERR
out['device'] = dict(vendor_2q_error=VENDOR_2Q_ERR, eta_consistent=eta_dev,
                     eta_used_in_hardware_work=eta_dev_used,
                     margin_below_threshold=infid(eta_star) / VENDOR_2Q_ERR)
check(eta_dev < eta_dev_used, 'consistent eta must be below the one used')

# ---- Theorem 1: copies needed on any graph with a degree-3 vertex (e.g. any 2D lattice)
N_prod = 9 / EPS ** 2
out['theorem1'] = dict(eps=EPS, copies_product=N_prod,
                       statement='no single-copy measurement, entangled or not, needs fewer')
check(abs(N_prod - 90000) < 1e-6, 'Theorem 1 copies at eps=0.01 != 90,000')

# ---- Theorem 2: copies on a chain
rows = {}
for label, eta in (('perfect gates', 1.0), ('this device (consistent)', eta_dev),
                   ('this device (as used)', eta_dev_used), ('at the threshold', eta_star)):
    C = C_chain(eta)
    rows[label] = dict(eta=eta, C=C, copies=C / EPS ** 2, copies_saved=(9 - C) / EPS ** 2,
                       fraction_saved=1 - C / 9)
out['theorem2'] = rows
check(abs(rows['perfect gates']['copies'] - 70000) < 1e-6, 'chain eta=1 copies != 70,000')
check(abs(rows['at the threshold']['fraction_saved']) < 1e-9, 'no saving expected at threshold')
d_units = abs(rows['this device (consistent)']['C'] - rows['this device (as used)']['C'])
out['units_discrepancy_in_C'] = d_units
check(d_units / rows['this device (consistent)']['C'] < 1e-3,
      'the eta_dev units slip moves C by more than 0.1% -- it would change a reported number')

# ---- the hardware transfer tolerance, in copies
C_best, N_budget = 7.19, 1e4
tau = np.sqrt(0.1 * C_best / N_budget)
out['transfer_tolerance'] = dict(tau=tau, copies_budget=N_budget,
                                 meaning='bias allowed before it adds >10% to the best mixture MSE')
check(abs(tau - 0.0085) < 5e-5, 'tau does not reproduce 0.0085')

# ---- the model-based projection (frozen c115)
p = os.path.join(os.path.dirname(OUT), 'sealed', 'c115_risk_curves.json')
rc = json.load(open(p))
adv = [1 - rc['target']['ratio'][0], 1 - rc['target']['ratio'][-1]]
out['risk_projection'] = dict(advantage_at_1e2=adv[0], advantage_at_1e5=adv[1],
                              simultaneous_bound=rc['target']['sim_hi'])
check(rc['target']['sim_hi'] < 1, 'risk projection no longer resolved simultaneously')

os.makedirs(OUT, exist_ok=True)
json.dump(out, open(os.path.join(OUT, 'c116_tangible.json'), 'w'), indent=2, default=float)

print('=' * 92)
print(f'NC-2 TANGIBLE QUANTITIES  (precision eps = {EPS} on every Pauli expectation)')
print('=' * 92)
t = out['threshold']
print(f"  threshold eta* = {t['eta']:.6f}: average gate infidelity {100*t['avg_gate_infidelity']:.4f}%"
      f"  (entanglement infidelity {100*t['entanglement_infidelity']:.4f}%)")
dv = out['device']
print(f"  this device: vendor 2q error {100*VENDOR_2Q_ERR:.3f}% -> eta = {dv['eta_consistent']:.6f} "
      f"(consistent) vs {dv['eta_used_in_hardware_work']:.6f} (as used)")
print(f"               {dv['margin_below_threshold']:.0f}x below the threshold infidelity")
print(f"  Theorem 1: {N_prod:,.0f} copies on any graph with a degree-3 vertex; no single-copy scheme does better")
for k, v in rows.items():
    print(f"  Theorem 2, {k:<26s} C = {v['C']:.4f}  copies {v['copies']:>9,.0f}  "
          f"saved {v['copies_saved']:>7,.0f}  ({100*v['fraction_saved']:.2f}%)")
print(f"  units slip: C moves by {d_units:.5f} ({100*d_units/rows['this device (consistent)']['C']:.3f}%)")
print(f"  transfer tolerance tau = {tau:.4f} at a {N_budget:,.0f}-copy budget")
print(f"  risk projection advantage {100*adv[0]:.1f}% (1e2 copies) -> {100*adv[1]:.1f}% (1e5 copies)")
print('=' * 92)
if FAILS:
    print('FAILED:'); [print('  - ' + f) for f in FAILS]; sys.exit(1)
print('all assertions hold')
