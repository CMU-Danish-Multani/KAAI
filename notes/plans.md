# LtU-ILI with Shared Agentic Hyperpriors

Rewritten 2026-08-25. Replaces the previous 876 line plan, archived at
`archive/notes/plansPreLtuIli.md`. That plan aimed at beating CosmoBench with a weight
sharing supernet. It was aimed at the wrong target.

## 1. The task

Two deliverables, from the assigned brief:

- **A curated model zoo** of neural architectures for astrophysical machine learning
  tasks, each with standardised evaluation results and documented failure modes.
- **A Claude skill** that searches the zoo given data modality, parameter
  dimensionality and compute budget, and returns ranked architecture recommendations.

Stated success criteria:

1. A taxonomy stable enough that community contributions do not require constant
   re-curation.
2. Three external groups contribute entries.
3. Recommended architectures reach MCMC-equivalent posterior quality in under two GPU
   hours.

Supervisor: Matt Ho, first author of LtU-ILI (arXiv 2402.05137).

## 2. The correction this plan encodes

The word in the task title is **with**. LtU-ILI is the substrate, not a reference. The
previous work built a parallel framework: a bespoke schema, a bespoke recommender, four
inference heads calling `sbi` directly, and a bespoke coverage function. Every one of
those has a more complete counterpart inside LtU-ILI. As of 2026-08-25 the package
`ili` had never been installed on this machine.

Building a worse copy of the supervisor's own framework is not a contribution, and its
novelty is zero by construction. Everything below runs on his framework.

## 3. Where the novelty actually is

LtU-ILI Section 7 states its own future work:

> "We also intend to integrate LtU-ILI with state-of-the-art hyperparameter tuning
> (e.g. Akiba et al. 2019) to provide an automatic, yet principled way for doing model
> selection... This includes developing and implementing consolidated metrics for
> calibrating posterior coverage... to guarantee accurate posterior coverage through
> hyperparameter search."

Optuna, plus coverage as a gate on model selection. The assigned task adds one word:
**shared**. The prior over configurations accumulates across tasks instead of restarting
each time.

Our contribution is the admission gate on that shared prior, and the argument for why it
must exist:

- In a per task search, an architecture that scores well by exploiting a dataset
  artefact costs one wrong answer.
- In a shared prior, that architecture gains prior weight and is then recommended to
  every future task. The artefact propagates.

So screening is a precondition for sharing, not polish. LtU-ILI Section 5.3, the point
cloud example closest to our data, uses the 10,000 heaviest halos, a fixed count, and
reports no coverage test. The failure mode we measured could not have surfaced there.

## 4. Evidence status of everything this plan rests on

| Claim | Status |
|---|---|
| CAMELS galaxy count correlates with Omega_m at 0.73 | MEASURED BY US, `point_clouds/results/step2_diagnostics.json` |
| Sum and max pooling recover log N from their output; mean pooling does not | MEASURED BY US, calibrated in `blocks/count_screen.py` (+0.9138, +0.8968, -0.6616) |
| Node degree recovers N at R2 0.71 to 0.96 depending on graph cutoff | MEASURED BY US, `notes/related_papers.md` line 309 |
| Trimming CAMELS-SAM to fixed N costs 0.0593 on Omega_m, with the count channel already closed | MEASURED BY US, pure information loss control |
| Single NDEs are prone to overconfidence; ensembling is the fix | PUBLISHED, Hermans et al. 2022, endorsed by LtU-ILI Sections 3.2 and 6 |
| TARP is necessary and sufficient for posterior coverage given enough samples | PUBLISHED, Lemos et al. 2023, LtU-ILI Section 2.5 |
| NPE suits large dim(x); NLE suits large dim(theta); NRE struggles at large dim(theta) | PUBLISHED AS EXPERIENCE, LtU-ILI Section 2.3, no numbers attached |
| `lampe` gives tighter and better calibrated posteriors than `sbi` | ASSERTED, LtU-ILI Section 3.4, "in our experience", unquantified |
| A shared hyperprior reduces trials to reach a target score on a held out task | UNTESTED HYPOTHESIS. This is the headline experiment. |
| Recommended architectures reach MCMC-equivalent quality in under two GPU hours | UNTESTED. No MCMC reference posterior exists for CAMELS. See Stage D3. |

Anything moving from the bottom rows to the top rows must move by measurement, in the
run log, with predictions registered before the run.

## 5. Conventions this plan must obey

- Predictions written into `runLog.md` before the run that tests them, never after.
- Three seeds minimum for any comparative claim. Single run numbers are labelled
  `(single run)` and their uncertainty is written as `null`, never `0`.
- Every number in a deliverable is derived from a JSON by path. No number is retyped.
- A fix that changes a random stream invalidates every number downstream. Rerun them.
- Retractions go in a new run log entry, never by editing the old one.
- Environment: LtU-ILI work runs in a separate conda env, because `ltu-ili` pins
  `sbi<=0.22.0` and the working env has `sbi 0.27.0`. The data pipeline stays where it
  is and hands over arrays.

## 6. Non-goals, banned by name

A helpful assistant would plausibly add these. They are out of scope.

- **`pydelfi` and the TensorFlow backend.** Its dependency requires `python_version<3.7`
  and this machine runs 3.12. Documented as a limitation, not worked around.
- **Sequential inference (SNPE, SNLE, SNRE).** Requires an on the fly simulator. We have
  fixed catalogues. Out of scope, stated explicitly.
- **Merger trees.** Not in the brief. `merger_trees/` stays as committed Phase 0 work and
  is not extended.
- **Beating CosmoBench.** Not the task. Published numbers are used as a wiring check.
- **New aggregation blocks.** Four were built and archived. The zoo needs breadth across
  engines and backends, not more pooling variants.
- **A new NAS search space.** The search runs over LtU-ILI configurations.

## 7. What gets built

```
env/                          conda env spec for ltuili (sbi<=0.22)
ili_kaai/
  loaders.py                  CAMELS and CAMELS-SAM as ili dataloaders
  embeddings.py               our GNN and fishnets wrapped as lampe embedding nets
  configs/                    one yaml per zoo entry, the format ili consumes
  sweep.py                    Stage C driver, writes results/sweep.json
  validate.py                 TARP, PIT, C2ST via ili.validation
  screen.py                   model level count screen, extends count_screen.py
  hyperprior.py               fit a prior over configs, warm start optuna
skill/
  SKILL.md                    the deliverable skill
  query.py                    ranked recommendation plus an emitted ili yaml
  heldout.json                problem descriptions taken from published papers
  CONTRIBUTING.md             what a submitter must supply for admission
```

Kept from before and reused: `point_clouds/load.py`, `point_clouds/tpcf.py`,
`point_clouds/gnn.py`, `point_clouds/blocks/count_screen.py`,
`point_clouds/blocks/fishnets.py`, `common/metrics.py`, and all of
`point_clouds/results/`.

## 7.5 Where the build order actually stands, 2026-08-28

    Stage A  framework, proven to work              DONE   14/16 tests, toy problem
                                                           reproduces both degeneracies
    Stage B  our data inside their framework        DONE   3 CAMELS tasks
    Stage C  the sweep that fills the zoo           DONE   72/72 cells, 16.6 h, 0 errors
    Stage D  validation, done their way             PART   TARP and PIT done and
                                                           validated. C2ST blocked, no
                                                           reference posterior exists.
    Stage E  the admission gate                     DONE   threshold measured not
                                                           chosen; zoo.json built
    Stage F  the shared hyperprior                  NOT STARTED
    Stage G  the skill                              NOT STARTED
    Stage H  write up                               PART   guide, email, compute request

Three things changed the plan on 2026-08-28 and are not reflected in the stages below.

**Quijote was found on disk, unused.** 19,651 training simulations, 5 parameters,
1000 Mpc/h box, already in correlation function form. Wired into `tasks.py` as
`quijoteAll` and `quijoteJoint`, not yet run. This is 33 times the training data the
zoo has seen and it settles the open question about whether the measured overconfidence
is a property of the architectures or of an 800-simulation budget. Running it comes
before anything in Stage F or G.

**The zoo covers one modality of four.** A literature sweep, written up in
`notes/zooCandidates.md`, found that the field uses convolutional networks for fields
and images, set networks and graph networks for catalogues, and 1D convolutions or
transformers for spectra and light curves. We cover only compressed summary vectors,
which is the easy case. 21 architecture and engine combinations across two backends are
verified to build in our stack; the zoo uses 8, all on one backend.

**Balanced NRE is missing from the framework.** The method designed to produce
conservative rather than overconfident posteriors exists in sbi 0.22 and is not exposed
by ltu-ili. It targets our headline finding directly, and exposing it is a small pull
request to the supervisor's package.

## 8. Build order

### Stage A. Framework, proven to work

- A1. Create env `ltuili`, install `ltu-ili[pytorch]` plus `tarp`.
- A2. Run their own test suite.
- A3. Reproduce their Section 4.1 toy problem (Equation 14, 10-d data, 3 parameters).
  This is analytic, so correctness is checkable without our data.

Verification:
```
conda run -n ltuili pytest ltu-ili/tests -q
conda run -n ltuili python ili_kaai/checks/toyModel.py     # 68% interval contains truth
```

### Stage B. Our data inside their framework

- B1. `ili` dataloader for the 2PCF summary vector, 25 log bins, from `tpcf.py`.
- B2. `ili` dataloader for point clouds, with `gnn.py` wrapped as a lampe embedding.
- B3. Port check against our own prior measurement.

Verification: a single NPE run on CAMELS 2PCF must land near our existing measured
Omega_m R2. If it does not, the wiring is wrong, not the method. Our old numbers earn
their keep here as the reference that proves the port.

### Stage C. The sweep that fills the zoo

Axes, taken from LtU-ILI Sections 2.3 and 3.4:

| axis | values |
|---|---|
| engine | NPE, NLE, NRE |
| backend | sbi, lampe |
| NDE | MAF, NSF, MDN |
| embedding | none, MLP, GNN, fishnets |
| ensemble | 1, 4 |
| task | CAMELS Omega_m, CAMELS sigma_8, CAMELS joint, CAMELS-SAM joint |

Three seeds per cell, matched compute budget. Recorded per cell: R2, test log
posterior, TARP coverage, marginal PIT, wall time, parameter count.

Predictions to register before running, derived from Section 2.3:
- For the 2PCF vector, dim(x)=25 and dim(theta)=2, so neither NPE nor NLE is clearly
  favoured. A large gap either way would contradict the paper's own reasoning.
- For point clouds, dim(x) is large, so NPE should beat NLE.
- Ensembles of 4 should widen error bars and improve coverage relative to single models,
  at unchanged or slightly worse R2.

### Stage D. Validation, done their way

- D1. TARP multivariate coverage. Our previous coverage function was the marginal PIT of
  their Equation 13, which can pass while the joint is wrong for correlated
  Omega_m and sigma_8.
- D2. Marginal PIT and P-P plots.
- D3. C2ST. It needs a reference posterior, which CAMELS does not have. Plan: validate
  our C2ST implementation on their toy model, where a long run HMC reference is cheap,
  then state plainly that CAMELS has no reference and the brief's MCMC-equivalence
  criterion is therefore untested. Do not quietly drop the criterion.

### Stage E. The admission gate

- E1. Extend `count_screen.py` from component level to model level: probe the trained
  posterior's embedding for the galaxy count.
- E2. Run it across every Stage C cell.
- E3. Flag any entry whose TARP coverage is overconfident. Overconfidence is the
  dangerous direction and is not silently ranked beside calibrated entries.
- E4. Ablation: show the gate changes the recommendation. The component level version
  changed advice on 3 of 8 queries. Redo at framework level.

### Stage F. The shared hyperprior. This is the paper.

- F1. Fit a prior over the configuration space from Stage C, weighted by score,
  calibration pass and screen status.
- F2. Warm start Optuna with it on a task held out of the prior fit.
- F3. Measure trials to reach a target score, warm start against cold, three seeds each.

Prediction to register: warm start reaches the cold search's 20 trial score in fewer
than 20 trials. If it does not, that is a negative result and gets reported as one.

### Stage G. The skill

- G1. `SKILL.md` plus `query.py`. Inputs must cover the decision variables the framework
  paper actually uses: modality, dim(theta), compute budget, and two we currently
  cannot express, namely how many observations will be inferred on (amortisation) and
  whether the posterior feeds a downstream hierarchical model.
- G2. Held out evaluation on problem descriptions lifted from published astrophysics
  machine learning papers, using `notes/related_papers.md`. The previous held out set
  was self written, which is its weakest point.
- G3. `CONTRIBUTING.md`: exactly what a submitter supplies for an entry to be admitted.
  This is success criterion 1 and 2, and it is currently unaddressed.

### Stage H. Write up

- H1. Report and dashboard, every number derived by path.
- H2. Email to Matt.

## 9. What I cannot predict

- Whether `ltu-ili` 0.1.5 runs on this machine at all. Its pinned `sbi<=0.22.0` against
  `torch 2.12` is untested here, and Apple Silicon MPS support inside `sbi` 0.22 is
  unknown to me. Stage A exists to find out early and cheaply. If it fails, the fallback
  is an older torch inside the same isolated env.
- Whether the sweep in Stage C fits the compute budget. It is roughly 3 engines x 2
  backends x 3 NDEs x 4 embeddings x 2 ensemble sizes x 4 tasks x 3 seeds, which is far
  too many cells to run in full. It will be pruned to a fractional design, and what was
  dropped will be logged rather than left to read as full coverage.
- Whether the hyperprior helps. Stage F is a real experiment with a real chance of a
  negative result.
- How long any of this takes. Six previous time estimates in this project were wrong,
  every one by extrapolating from a timing measured while other work shared the machine.
  Stage ordering is committed; durations are not.

## 10. Presentable earlier than complete

If the presentation cannot wait for Stage F, the shortest honest story is
A, B, C pruned, D, E: the zoo exists inside the supervisor's framework, it is validated
with his own metrics, and it has an admission gate that his framework does not have,
with an ablation showing the gate changes the answer. Stage F is what turns that from a
useful artifact into a paper.
