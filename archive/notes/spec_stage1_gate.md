# Spec: Stage 1 gate, 2PCF baseline reproduction

Written 2026-08-17, before any model was trained. Acceptance criteria in §7 were
fixed in advance and are not to be revised after seeing a result.

Plan context: [plans.md](plans.md) §11. Track order is point clouds first, then
merger trees.

---

## 1. Goal

Reproduce the 2PCF baseline of CosmoBench Table 2 on CAMELS-SAM and CAMELS.

The purpose is not a good model. It is to prove that the scoring, splitting and
normalisation in this repository are trustworthy before anything is built on
top of them. Until a published number reproduces, every number produced later is
uninterpretable.

## 2. Non-goals

Banned by name, so they cannot creep in:

- No architecture search, no supernet, no Optuna over anything except the
  hyperparameters this baseline already has.
- No graph neural network.
- No Quijote. Its positions are not downloaded and it is not needed here.
- No leak experiment. That is Stage 4.
- No SARA, no agent layer.
- No merger trees.
- **No attempt to beat the published number.** Landing above the band is as much
  a failure of reproduction as landing below it, and must be diagnosed rather
  than celebrated.

## 3. Environment

Conda env `KAAI`, Python 3.12.13. Corrfunc 2.5.3, torch 2.12.1, optuna 4.9.0,
numpy, h5py. Apple Silicon.

**Device: CPU, not MPS.** Measured 2026-08-17: the same 4 model trainings took
28.6 s on CPU and 88.5 s on MPS. These tensors are far too small for GPU
transfer overhead to pay for itself.

## 4. Interface contract

**Input.** The position files `data/CAMELS/ALL_galaxies_{split}.hdf5` and
`data/CAMELS-SAM/top5000_galaxies_{split}.hdf5`.

The shipped `tpcf_*.hdf5` files are **not** used. Measured 2026-08-17, their
binning does not match the binning that produced Table 2:

| Suite | Shipped file | Paper, Table 2 |
|---|---|---|
| CAMELS | 19 bins, 0.1 to 12 | 25 bins, 0.0125 to 12 |
| CAMELS-SAM | 19 bins, 1.0 to 40 | 25 bins, 0.0125 to 12 |
| Quijote | 24 bins, 2.0 to 80 | 25 bins, 0.5 to 480 |

A correlation function measured over different distance ranges is a different
input, so those files cannot reproduce the published numbers.

**Output.** `point_clouds/results/step1_gate_2pcf.json`, holding per-target test
R², the spread across seeds, bootstrap error bars, the winning hyperparameters,
the seeds, the resolved device, and library versions.

## 5. Behaviour

1. Recompute ξ(r) with Corrfunc from the positions, 25 logarithmic bins per
   CosmoBench Sec. B.1 and Table 7: CAMELS-SAM 0.0125 to 12 (Base × Rmin/4),
   CAMELS 0.0125 to 12 (Base × 4Rmax). Periodic box.
2. Take absolute values, since shot noise makes ξ slightly negative at large
   radii, then take log10. Sec. B.1.
3. Standardise features and targets using **train statistics only**.
4. Four-layer MLP: input → h1 → h2 → h3 → (Ω_m, σ_8). Hidden sizes searched in
   [64,128], [64,128], [16,64].
5. Optuna TPE, 100 trials, 300 epochs each, learning rate log-uniform in
   [1e-5, 1e-2], dropout in [0, 0.5], batch size in {4, 16, 64}. Selected on
   validation.
6. Retrain the winner across three seeds. Report test R² as mean and standard
   deviation across seeds, plus bootstrap error bars on the test set.

## 6. Constraints

- Train-only normalisation statistics. Measuring across splits leaks test facts
  into training and inflates scores silently.
- Labels joined by position in `sim_names()` order, which `load.py` establishes.
  Never by row index against the shipped tpcf files, which disagree with the
  position files on CAMELS-SAM val (201 rows against 204 clouds).
- Everything seeded: `random`, `numpy`, `torch`, and the shuffling permutation.
- At least three seeds. A single run cannot support a comparative claim.
- One seed reports `(single run)`, never `+/- 0.0000`. Absent uncertainty is not
  a tight uncertainty.

## 7. Acceptance criteria

Fixed before the first run. Published values are CosmoBench Table 2, test split,
with their reported bootstrap standard deviation.

| Suite | Target | Published | Accept if mean across seeds falls in |
|---|---|---|---|
| CAMELS-SAM | Ω_m | 0.73 ± 0.03 | **[0.67, 0.79]** |
| CAMELS-SAM | σ_8 | 0.82 ± 0.02 | **[0.78, 0.86]** |
| CAMELS | Ω_m | 0.84 ± 0.02 | **[0.80, 0.88]** |
| CAMELS | σ_8 | 0.30 ± 0.06 | **[0.18, 0.42]** |

The band is the published value plus or minus **two** standard deviations,
widened from one because our hyperparameter search is not identical to theirs.

**Failing a band means stop and debug. It does not mean widen the band.**

## 8. Open questions

**Estimator, resolved and recorded.** The paper uses Landy-Szalay with 100 times
as many random points as data. For a periodic cube the random-random term has a
closed form, so this implementation uses ξ = DD/RR − 1 with
RR = N(N−1)·V_shell/V_box, which is exact rather than sampled and much cheaper.

Corrfunc's counting convention was **derived, not assumed**: against uniform
random points in a periodic box, its `npairs` matched the analytic ordered-pair
expectation to within 2% (ratios 0.980, 0.994, 1.000 across three bins, the
deviation consistent with Poisson noise in the smallest bin). The guard is
`tpcf.calibrate()`, which asserts ξ ≈ 0 for uniform random points and fails the
run otherwise. Measured: largest |ξ| = 0.0076 against a tolerance of 0.05.

**Bin count, unresolved.** The paper says "25 bins". The shipped Quijote file has
25 bin *edges*, meaning 24 bins. This implementation uses 25 bins (26 edges),
matching the paper's plain words. Table 6 shows Quijote R² moving only from 0.84
to 0.83 between 25 and 250 bins, so the choice is very unlikely to matter. It is
recorded in the output rather than left implicit.

**Innermost bins are shot-noise dominated, as the paper describes.** In
CAMELS-SAM, 85% of clouds have zero pairs in the innermost shell (0.0125 to
0.0164 cMpc/h), giving ξ = −1 exactly. The absolute-value-then-log step maps
those to 0. This is the behaviour Sec. B.1 refers to when it mentions
"occasional unphysical (small) negative values" and values that "are
significantly high" at low bins.
