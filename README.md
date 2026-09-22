# Exact limits on entangled readout of local Pauli observables with noisy entangling gates

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22885706.svg)](https://doi.org/10.5281/zenodo.22885706)

Code and data for the article by **Jiachen Shen** (University of Houston) and **Hui Zhong** (Miami University,
corresponding author, zhongh7@miamioh.edu).

The repository contains the raw measurement counts of every device job used in the paper, the staged analysis of the
prospective test with its committed predictions, every script that produces a number, table or figure,
and the three executable test suites.

## Contents

| path | content |
|---|---|
| `analysis/data/raw/` | raw counts of the three earlier device jobs on IBM Quantum `ibm_cleveland`, with `MANIFEST.json` (sha256 of all four job files) |
| `analysis/data/sealed/` | the prospective test: raw counts (`c114_raw.json`), committed predictions with bootstrap draws (`c114_predictions.json`), and the second-moment projection of `c115` |
| `analysis/data/selfcheck/` | derived outputs of the analysis and check scripts (JSON / text) |
| `analysis/figs/` | Figs. 1–5 and Supplementary Figs. 1–5 (PDF and PNG) |
| `analysis/source_data/SourceData.xlsx` | Source Data, one sheet per figure panel |
| `analysis/c*.py` | analysis, figure and check scripts (below) |
| `analysis/regression_tests.py`, `production_tests.py`, `coverage_tests.py` | executable test suites |

### Device jobs

| job | file | sha256 | used for |
|---|---|---|---|
| `dam6td02fm4c73f2gl1g` | `data/raw/dam6td02fm4c73f2gl1g.json` | `eb60672c1cea119a9382757ddb5c957ef43be90260f9df80075acee5f633999e` | engineered-noise sweep (Supplementary Note 6) |
| `daoa3j8pqrnc7399miv0` | `data/raw/daoa3j8pqrnc7399miv0.json` | `6826c183db59b2c282a7ae3082b266b0ef3bb4d64539ea84a6a603070de37520` | held-out run, paired regression (Fig. 3) |
| `daobh65r85ps73ff233g` | `data/raw/daobh65r85ps73ff233g.json` | `8186bc63139a86ea4307d7ffdf0c3e91f92425c30d735035098e4a91e4d2c654` | target-resolved calibration (Fig. 3b) |
| `daoduuo2fm4c73f5acpg` | `data/sealed/c114_raw.json` | `0fd54ec8eaf58b95b0a7d45217b45b499a769d9fd10633955b31adc41c47213d` | prospective test (Figs. 4, 5) |

Paths are relative to `analysis/`. The same sha256 values are listed in `analysis/data/raw/MANIFEST.json`.

### Scripts

| script | produces |
|---|---|
| `c99_run_device.py` | circuit construction shared by all device runs (state preparation, bases, Bell decoder) |
| `c101_threshold_measurement.py`, `c106_recalibration.py` | engineered-noise sweep and its calibration analysis |
| `c123_rerun_c106_frozen.py` | runs `c106` on the frozen counts → `data/selfcheck/c106_frozen.txt` (Supplementary Note 6) |
| `c126_k_joint_bootstrap.py` | statistical error of the engineered-noise constant *k* |
| `c108_measurement_choice.py`, `c109_realised_reference.py`, `c110_paired_damping.py` | held-out run and paired regression |
| `c112_targeted_calibration.py` | target-resolved, interleaved calibration run |
| `c114_sealed.py` | the staged prospective test (`build`, `acquire`, `predict`, `evaluate`) |
| `c115_risk_curves.py` | coordinates, design coefficients and grid used by the Fig. 5 projection |
| `c130_exact_mse_projection.py` | Fig. 5 and Supplementary Table 8: exact model-internal mean squared error |
| `c116_tangible.py` | Supplementary Note 5 (results in numbers of copies) |
| `c117_theorem_checks.py`, `c125_lemma_checks.py` | numerical checks of the theorem ingredients and the budget lemmas |
| `c118_sensitivity.py` | post-hoc sensitivity of the prospective test (Supplementary Note 7) |
| `c128_endpoint_checks.py`, `c129_endpoint_coverage.py` | Supplementary Note 19 |
| `c119_freeze_raw.py` | retrieves finished jobs into `data/raw` (needs an IBM Quantum account) |
| `c120_figure_data.py`, `c121_figures.py`, `c124_si_figures.py` | figure data and Figs. 1–5, Supplementary Figs. 1–4 |
| `c122_source_data.py` | `source_data/SourceData.xlsx` |
| `c127_si_tables.py` | LaTeX rows of two Supplementary Tables |

## Reproducing the results

Python 3.11 with the packages in `requirements.txt`. Everything below runs from the frozen data; no device access is
needed.

```bash
pip install -r requirements.txt
cd analysis

python c123_rerun_c106_frozen.py > data/selfcheck/c106_frozen.txt   # engineered-noise analysis
python c126_k_joint_bootstrap.py
python c114_sealed.py evaluate                                     # pre-registered verdict of the prospective test
python c118_sensitivity.py
python c128_endpoint_checks.py
python c130_exact_mse_projection.py                                # Fig. 5 projection
python c116_tangible.py
python c117_theorem_checks.py
python c125_lemma_checks.py
python c129_endpoint_coverage.py                                   # slow: parametric replication
python c120_figure_data.py && python c121_figures.py && python c124_si_figures.py
python c122_source_data.py

python regression_tests.py && python production_tests.py && python coverage_tests.py
```

The check scripts exit with a non-zero status if any assertion fails.

## The prospective test

The design, hypotheses, tolerance and decision rule of the prospective test were fixed before the job was submitted
(Methods of the paper):

| | |
|---|---|
| device | `ibm_cleveland`, physical qubits `[50, 51, 58, 71, 70, 69]` |
| job | one job, every circuit in one shuffled, interleaved order, seed `20260922`; 20,240 circuits × 60 shots |
| calibration | directed grid over every qubit × axis, edge × axis pair and Bell covering × dimer × setting × readout, one dataset feeding both models; 14,240 circuits |
| evaluation | disjoint directed circuits at the five targets, 1,200 each |
| targets | `Z` on qubit 2; `ZZ` and `YY` on edge (2,3) in the product arm; the same two through the Bell decoder, read by the one-qubit (`w1`) and parity (`w2`) readouts |
| models | pooled: mean of per-combination `(a, c)` over the mode's class; target-resolved: `(a, c)` of the target combination |
| endpoint | `B = max` over the five targets of `|β|`, `β = (a_eval/â − 1)·θ_ref + (c_eval − ĉ)/â`, `θ_ref = 0.5` |
| decision | a model is adequate iff the one-sided 95% bootstrap upper bound of `B` (1,000 resamples) is below `τ = 0.0085` |
| hypotheses | H1: the target-resolved model is adequate. H2: the pooled model is not adequate |
| exclusions | none; a job returning fewer circuits or shots than submitted aborts the analysis |

`c114_sealed.py` runs the test in three stages. `acquire` stores the raw counts with a hash of the circuit schedule
and performs no analysis; `predict` reads calibration circuits only and writes both models' predictions with their
bootstrap draws; `evaluate` refuses to run unless the predictions file is committed and unmodified, the regression
tests pass, the rebuilt circuit schedule matches the one that ran, and the raw counts are unchanged since the
predictions were made. On the data in this repository it returns target-resolved `B = 0.00284`, upper bound
`0.00550` (adequate) and pooled `B = 0.00788`, upper bound `0.00997` (not adequate).

The scripts are the analysis code used for the paper. For this release their comments, docstrings and printed labels
were edited, the key-file path was made local, and `c121_figures.py` no longer writes draft captions; the executable
logic is otherwise unchanged.

## Running on a device

Re-acquiring data needs an IBM Quantum account. `c99_run_device.py` reads the API token at run time from
`analysis/apikey.json` (a JSON object with an `apikey` field; ignored by git). No token is stored in this repository.

## Licence

Code: MIT (`LICENSE`). Data, figures and Source Data: CC BY 4.0 (`LICENSE-DATA`).

## Citation

If you use this code or data, please cite the article (reference to be added on publication) and the archived
release: Shen, J. & Zhong, H. Code and data for: Exact limits on entangled readout of local Pauli observables with
noisy entangling gates. Zenodo https://doi.org/10.5281/zenodo.22885706 (2026).
