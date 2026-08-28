# Results so far: the point cloud track, stages 0 to 3

What has been measured, what it means, and what is still open. Written
2026-08-21, covering work from 2026-08-17 onward.

Three companion documents, and the difference matters:

- [plans.md](plans.md) is the plan. Parts of it are now stale, see the end of
  this file.
- [runLog.md](../runLog.md) is the chronological record, including every bug,
  correction and retraction as it happened.
- This file is the durable summary of what was measured.

**Every number below was generated from the result JSON files, not retyped.**
Regenerate with the script recorded at the end of this document.

Confidence is tagged. MEASURED is a number we produced. PUBLISHED is somebody
else's number we have not reproduced. INTERPRETED is a conclusion drawn from a
measurement, which could be wrong.

---

## Status at a glance

| Stage | What it was | State |
|---|---|---|
| 0 | Verify the data is what we believe | Done, no code artifact |
| 1 | Reproduce a published number before building anything | **PASS**, all four targets |
| 2 | Three diagnostics to decide the search space | Done, all three predictions falsified |
| 3 | Build the field of competitors | Done, three models |
| 4 | The architecture search | Not started |

Track order is **point clouds first, merger trees second**, reversing what
plans.md sections 5 and 7 still say. The reason is that the paper's spine is the
counting shortcut, and that shortcut lives in the CAMELS point clouds.

---

## Stage 0: verifying the ingredients

- MEASURED In CAMELS, the number of galaxies in a box correlates with Omega_m at
  0.709, 0.758 and 0.712 across train, validation and test.
- VERIFIED This reproduces the 0.758 recorded in plans.md section 8. That figure
  is the highest of the three splits. **The defensible number is about 0.73.**
- MEASURED The same count correlates only 0.110 to 0.152 with sigma_8, so the
  shortcut is specific to Omega_m.
- MEASURED CAMELS galaxy counts span 588 to 4511 with 857 distinct values across
  1000 boxes. CAMELS-SAM and Quijote are fixed at exactly 5000 everywhere.
- VERIFIED The shortcut-open and shortcut-closed pair the paper needs does exist.
- HONEST CAVEAT The pair is not matched. CAMELS-SAM differs from CAMELS in box
  size, physics and mass resolution as well as in the shortcut.
- MEASURED The shipped `tpcf_*.hdf5` files use different distance bins from the
  paper. CAMELS 19 bins from 0.1 to 12 against 25 bins from 0.0125 to 12.
  CAMELS-SAM 19 bins from 1.0 to 40 against 25 bins from 0.0125 to 12. Quijote
  24 bins from 2.0 to 80 against 25 bins from 0.5 to 480.
- RETRACTION An earlier claim that the flagship Quijote number was reproducible
  from files already on disk was wrong. Exact Quijote reproduction still needs
  the 4.1 GB position download.
- MEASURED Corrfunc returns ordered pairs, derived by comparing against the
  analytic expectation for uniform random points. Ratios 0.980, 0.994, 1.000.
- VERIFIED A permanent guard asserts xi is near zero for random points and aborts
  otherwise. Largest measured deviation 0.0076 against a tolerance of 0.05.

---

### Stage 1 gate: 2PCF plus MLP

| Suite | Target | Ours (3 seeds) | Published | Band | Verdict |
|---|---|---|---|---|---|
| CAMELS-SAM | Omega_m | 0.7784 +/- 0.0024 | 0.73 +/- 0.03 | [0.67, 0.79] | PASS |
| CAMELS-SAM | sigma_8 | 0.8231 +/- 0.0057 | 0.82 +/- 0.02 | [0.78, 0.86] | PASS |
| CAMELS | Omega_m | 0.8597 +/- 0.0011 | 0.84 +/- 0.02 | [0.80, 0.88] | PASS |
| CAMELS | sigma_8 | 0.3772 +/- 0.0074 | 0.30 +/- 0.06 | [0.18, 0.42] | PASS |

### Stage 3 model 1: LLS, 49 parameters

- VERIFIED The gate passes, so scoring, splitting and normalisation are
  trustworthy.
- MEASURED All four landed **above** the published value, by +1.61, +0.15, +0.98
  and +1.29 published standard deviations. Four of four in the same direction.
- FLAG That is a systematic offset, not scatter, and it remains unexplained.

---

## Stage 2: three diagnostics


| Suite | Comparison | Omega_m | sigma_8 |
|---|---|---|---|
| CAMELS-SAM | count added as a feature | +0.0000 | +0.0000 |
| CAMELS | count added as a feature | -0.0114 | -0.0109 |
| CAMELS-SAM | trimmed to 588 galaxies | -0.0593 | -0.6135 |
| CAMELS | trimmed to 588 galaxies | -0.2588 | -0.1587 |
| CAMELS-SAM | spread over 5 search seeds | 0.0026 | 0.0026 |
| CAMELS | spread over 5 search seeds | 0.0037 | 0.0090 |

- MEASURED The spread across five independent hyperparameter searches is 0.0026
  to 0.0090, far too tight to explain the Stage 1 offset.
- INTERPRETED The offset is real, not search noise. Prediction that it was noise:
  **falsified**.
- MEASURED Handing the model the galaxy count as an extra input made CAMELS
  *worse* by 0.0114, and changed CAMELS-SAM by exactly 0.0000.
- INTERPRETED The correlation function is a ratio normalised by the pair count,
  so the count is already divided out. Prediction that it would help:
  **falsified**.
- INTERPRETED, IMPORTANT The shortcut is therefore not carried by the raw count.
  It is carried by **which galaxies made it into the file**, which changes the
  shape of the clustering measurement itself. That is far harder to screen for
  than removing a feature.
- MEASURED Trimming every cloud to 588 galaxies cost CAMELS 0.2588 on Omega_m but
  cost the already-closed CAMELS-SAM only 0.0593.
- HONEST CAVEAT The two suites do not lose the same fraction of galaxies, 88
  percent against about 70, so the excess is an estimate rather than a clean
  number.

---

## Stage 3: the field of competitors

### Stage 3 model 1: LLS, 49 parameters

| Suite | Target | Ours | Published | Band | Verdict |
|---|---|---|---|---|---|
| CAMELS-SAM | Omega_m | 0.7517 | 0.77 +/- 0.03 | [0.71, 0.83] | PASS |
| CAMELS-SAM | sigma_8 | 0.8291 | 0.82 +/- 0.02 | [0.78, 0.86] | PASS |
| CAMELS | Omega_m | 0.8034 | 0.78 +/- 0.03 | [0.72, 0.84] | PASS |
| CAMELS | sigma_8 | 0.2786 | 0.28 +/- 0.06 | [0.16, 0.40] | PASS |

### Stage 3 model 2: DeepSets, pooling comparison

| Suite | Pooling | Omega_m | sigma_8 |
|---|---|---|---|
| CAMELS | *count only, no model* | *+0.5058* | *+0.0145* |
| CAMELS | sum | +0.5233 +/- 0.0049 | +0.0045 +/- 0.0038 |
| CAMELS | mean | -0.0006 +/- 0.0074 | +0.0005 +/- 0.0117 |
| CAMELS | max | +0.2463 +/- 0.0523 | -0.0080 +/- 0.0040 |
| CAMELS-SAM | *count only, no model* | *-0.0089* | *-0.0466* |
| CAMELS-SAM | sum | +0.0792 +/- 0.0387 | -0.0485 +/- 0.0044 |
| CAMELS-SAM | mean | +0.0792 +/- 0.0387 | -0.0485 +/- 0.0044 |
| CAMELS-SAM | max | +0.0100 +/- 0.0080 | -0.0506 +/- 0.0023 |

### Stage 3 model 3: graph network, pooling comparison

| Suite | Pooling | Omega_m | sigma_8 |
|---|---|---|---|
| CAMELS | mean | +0.6600 +/- 0.0057 | +0.1931 +/- 0.0139 |
| CAMELS | sum | +0.8020 +/- 0.0088 | +0.3572 +/- 0.0368 |
| CAMELS | **sum minus mean** | **+0.1420** | **+0.1642** |
| CAMELS-SAM | mean | +0.5196 +/- 0.0077 | +0.2845 +/- 0.0125 |
| CAMELS-SAM | sum | +0.5170 +/- 0.0039 | +0.2861 +/- 0.0134 |
| CAMELS-SAM | **sum minus mean** | **-0.0026** | **+0.0016** |


### What the models say when placed side by side

CAMELS, Omega_m, which is where the shortcut lives:

| Model | Score | Reading |
|---|---|---|
| Counting galaxies, no model at all | 0.5058 | the pure shortcut |
| DeepSets, mean pooling | -0.0006 | a set model sees nothing without the count |
| DeepSets, sum pooling | 0.5233 | **its entire score was counting** |
| Graph network, mean pooling | 0.6600 | **real structure, no shortcut** |
| Graph network, sum pooling | 0.8020 | structure plus shortcut |
| Published graph network | 0.78 | PUBLISHED, for reference |

---

## The headline result

One word in the architecture, sum against mean pooling, decides whether a model
has access to the shortcut. Neither word looks like a shortcut.

| | mean pooling | sum pooling | difference |
|---|---|---|---|
| CAMELS, count varies | 0.6600 +/- 0.0057 | 0.8020 +/- 0.0088 | **+0.1420** |
| CAMELS-SAM, count fixed | 0.5196 +/- 0.0077 | 0.5170 +/- 0.0039 | **-0.0026** |

- VERIFIED The bottom row is a **measured null**, not an assumption. Same
  architecture, same code path, same seeds. Where the count cannot carry
  information, the pooling choice is worth less than the seed spread.
- MEASURED The CAMELS gap is 16 to 25 times the seed spread.
- INTERPRETED Message passing extracts genuine structure. A set model without
  count access scores zero on the same data where a graph model without count
  access scores 0.66.
- INTERPRETED The shortcut's value is not fixed. For DeepSets it was the entire
  score. For a graph network it adds about 0.14 on top of real signal.
- MEASURED Independent check that was not planned: on CAMELS-SAM, DeepSets sum
  and mean returned identical numbers to four decimal places including their
  seed spread. With every cloud at exactly 5000 points the two are mathematically
  the same model, so identical output confirms the implementation.

---

## Open questions

- **The Stage 1 offset.** All four 2PCF targets sit above published; the LLS
  model is scattered two above and two below. Both share the splits, labels, R2
  function and bootstrap, so the shared harness is cleared. The cause lies in the
  correlation-function computation or in the MLP training. Prime remaining
  suspects: 25 bins against the shipped files' 24, and the analytic estimator
  substituted for Landy-Szalay.
- **sigma_8 gains too much from sum pooling.** On CAMELS it rises 0.164, from
  0.1931 to 0.3572. Counting correlates only 0.11 to 0.15 with sigma_8, which
  squares to roughly 0.02. Pure counting does not explain the gain. Unexplained.
- **Is the published graph network itself partly counting?** Our sum-pooled model
  reaches 0.8020 against their 0.78. Not evidence, since our architecture differs
  in node features, size and tuning, but it is the question the screening work
  exists to settle.

---

## What is deliberately not done

- **Quijote.** Positions are not downloaded, 4.1 GB. Its shortcut is closed
  anyway, so it does not serve the paper's spine.
- **Merger trees.** Nothing built beyond the pre-existing dataloader.
- **Strict reproduction of the published graph network.** Ours uses constant node
  features and separation as the only edge feature, 67k parameters against their
  1.0 to 1.2M, one config against their 100-config search. Absolute positions
  inside a periodic box are meaningless, so feeding them was a deliberate
  deviation. Their numbers are a reference point, not a pass or fail gate.

---

## Where the code lives

| File | Purpose |
|---|---|
| `common/metrics.py` | R2, bootstrap error bars, seeding, device resolution |
| `point_clouds/tpcf.py` | correlation function from positions, with the random-point guard |
| `point_clouds/lls.py` | the 49-parameter linear model |
| `point_clouds/pointnet.py` | DeepSets and the three pooling operations |
| `point_clouds/gnn.py` | radius graphs and message passing |
| `point_clouds/training/*.py` | one script per experiment |
| `point_clouds/results/*.json` | every number, with seeds, versions and configuration |

---

## Known debt

- `plans.md` sections 5, 7 and 11 still describe merger trees as the first track,
  and section 11.3's diagnostics are the tree ones that were replaced.
- `features.py` and `mlp.py` were written during a rename that was not finished.
  Nothing imports them, and the MLP class exists twice.
- Two experiment scripts are still named `step1_` and `step2_`.
- Stage 0 produced no code, so its measurements cannot be reproduced by running
  anything.
- Two stray smoke-test logs sit in `point_clouds/results/`.

---

## Regenerating the tables

Every table above came from the result JSON files rather than being typed. The
generating script is recorded in `runLog.md` under the 2026-08-21 entry.
