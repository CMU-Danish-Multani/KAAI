---
name: iliArchitectureAdvisor
description: Recommend a simulation based inference architecture for an astrophysical inference problem, ranked from measured results rather than from folklore, and emit a runnable ltu-ili config. Use when someone describes an inference problem (cosmological parameters from a summary statistic or a galaxy catalogue, a posterior they need calibrated, a choice between NPE, NLE and NRE, or which normalising flow to use) and wants to know which architecture to start with, or asks what the zoo measured.
---

# Architecture advisor for LtU-ILI

## What this is

A curated zoo of simulation based inference architectures, every one measured the same
way on the same tasks, plus a ranked recommender over it. It exists so that a group
starting a new inference problem does not repeat an architecture search that ten other
groups have already run.

The catalogue is `ili_kaai/results/zoo.json`. Every number in it is read from a sweep
file by path, so nothing in it was typed by hand and it cannot drift from the
measurements it describes.

## Where the numbers live

**Every number you quote comes from `skill/measuredFacts.md`.** That file is generated
from `zoo.json` by `skill/facts.py`, so it cannot drift from the measurements. Read it
before answering. Do not quote a number from this file, from memory, or from an older
version of the catalogue.

This document holds mechanisms, which do not go stale. That document holds counts,
which do. When they disagree, that document is right.

## The one rule

**Never give a recommendation without its calibration verdict.**

A recommendation without a calibration verdict is a leaderboard row, and replacing
leaderboard rows is the entire point of this catalogue. The large majority of measured
entry-task pairs here are overconfident, which means their error bars come out too
small. That is the dangerous direction, and a user told only the accuracy will not know
it. `measuredFacts.md` has the current count and names every pair that is not
overconfident.

If a measurement carries `hidesParameterDisagreement: true`, say so and quote the per
parameter numbers. The task level verdict is an average over parameters, and on a
number of rows that average disagrees with the parameters underneath it.
`measuredFacts.md` lists every one.

## The rule that stops the first rule being abused

**Check the entry is informative before you quote its calibration.**

A posterior that predicts nothing is wide, and a wide posterior scores well on
coverage. So the worst entry in the catalogue can read as the best calibrated one.
This is not hypothetical: `nreLinear` measures R2 of about zero with coverage nearer
nominal than any entry that works.

So an entry whose mean R2 is at or below 0.05 is never a recommendation, whatever its
verdict says. `query.py` removes them and reports them separately, and
`measuredFacts.md` marks any calibrated row that carries no information. Never quote
one as a calibrated entry.

## How to answer

Five things decide the recommendation. Ask only for what the user has not already
said, and infer the rest rather than interrogating them.

1. **Modality.** A compressed summary vector, or a point cloud or catalogue of objects.
2. **dim(theta).** How many parameters they are inferring.
3. **How many observations.** One, or many. This is the axis that decides more than
   any other, and users rarely volunteer it.
4. **Compute budget.** Default is two GPU hours, which the project brief names as its
   own success criterion.
5. **Does it feed a downstream analysis.** If yes, calibration outranks accuracy,
   because whatever consumes the posterior inherits its error bars whole.

Then run the structured arm:

```
conda run -n ltuili python -m skill.query \
  --modality summary_vector --n-params 2 \
  --n-observations 1000 --compute-seconds 7200 --downstream
```

Add `--out config.yaml` to write the runnable ltu-ili config for the top pick.

## The two retrieval arms

The brief poses which retrieval strategy works better as an open question, so both
exist and `skill/evaluate.py` scores them on the same held out problems.

- **Structured arm**, `skill/query.py`. Parses the problem into the five fields above
  and ranks by measured accuracy and calibration inside a compute budget. Deterministic
  and offline checkable.
- **Few shot arm**, this file. Read `zoo.json` directly, including each entry's
  `summary`, `failureModes` and each measurement's `why`, and reason about the user's
  problem in prose. Better on problems that do not decompose cleanly into five fields.

Run the structured arm first. If its answer and your reading of the catalogue
disagree, say so and give both. A disagreement between the arms is information, not an
error to be smoothed over.

## What the catalogue measured, so you do not have to rediscover it

These are the mechanisms that change a recommendation. Every one is measured on this
stack at three seeds. **Take the numbers from `measuredFacts.md`, not from here.**

**Accuracy does not separate the architectures that work.** Among entries that carry
any information at all, the accuracy spread is small while compute spans four orders of
magnitude. Anyone choosing on accuracy alone is choosing on noise. The exceptions are
entries that fail outright, and those are worth naming rather than ranking.

**Amortization decides at scale.** NPE trains once and answers a new observation in a
forward pass. NLE and NRE run a fresh MCMC chain per observation. At one observation
they compete. At a thousand the MCMC entries are hundreds of times over a two hour
budget and are not an option at all.

**Ensembling barely helps, and two members hurt.** A two member ensemble measured worse
than a single flow of the same type. Four and eight recover, and eight buys almost
nothing over four for double the compute. Clones trained on the same simulations agree
with each other, which is exactly why averaging them adds little. Mixing families does
better than cloning one.

**The same config builds different networks in the two backends.** `lampeMaf` and
`npeMaf` declare identical `hidden_features` and `num_transforms`, and lampe builds a
net with 62 per cent of sbi's trainable weights. For the spline flow pair it is 40 per
cent. So `hidden_features` and `num_transforms` do not mean the same thing in the two
libraries, and a backend comparison at matched nominal settings is not a matched
comparison at all. Anyone benchmarking sbi against lampe on equal config strings is
comparing two different sized networks and attributing the difference to the framework.

The accuracy follows the size on the CAMELS tasks, where lampe is smaller and worse. It
does not on CAMELS-SAM, where lampe is smaller and better, which is what you would
expect if the larger net is overfitting there. That second half is an untested
explanation, not a measurement.

**A single autoregressive pass is unusable above one parameter.** `npeMade` matches
`npeMaf` when there is one parameter and loses a whole parameter when there are two,
recovering the second as well as a MAF does and the first not at all. Verified in sbi's
source: `build_maf` puts a `RandomPermutation` inside every stacked transform so no
parameter stays first, while `build_made` uses an identity transform and silently
ignores `num_transforms`. This is a statement about the family, not about one entry.
If a user proposes a single autoregressive pass over more than one parameter, say this.

**Set encoders cannot read point clouds.** `deepSets`, `pointNetLite` and a flattened
MLP all score within 0.05 of zero on both cloud tasks, eight of eight cases. The reason
is structural, not a training failure: pooling per point features of absolute positions
is a first moment statistic, and clustering is a second moment property defined on
pairs. Recommend a pairwise graph embedding instead.

**Point cloud embeddings must be pretrained.** Trained jointly from scratch,
`npeMafPairwiseGnn` gave individual seeds of -0.019, +0.198 and -0.000. It does not
reliably fail, it fails unpredictably about two runs in three, which is worse for a
practitioner. The flow fits the marginal of theta, stops conditioning on a context that
starts as noise, and the embedding never receives a gradient. Sixty epochs of plain
regression on the embedding first takes it to +0.250 with a spread of 0.020.

**A hand designed summary still beats a learned one, except once.** The 25 bin
correlation function reaches R2 0.870 on Omega_m where the best learned cloud embedding
reaches 0.250 on the same simulations. The exception is CAMELS-SAM, where the learned
embedding reaches 0.655 against 0.791 while reading 512 of 5000 galaxies.

## Honest limits, state them when they apply

- Everything is measured at 800 training simulations on CAMELS and CAMELS-SAM. Whether
  the overconfidence is a property of the architectures or of that budget is open, and
  the Quijote runs at 19,651 simulations were queued to settle it.
- The measured dim(theta) range is narrow and is stated in `measuredFacts.md`. Tasks at
  5 and 6 parameters are defined but not yet measured, so do not count them. Any
  recommendation outside the measured range is extrapolation, and `query.py` warns when
  the matched task's parameter count differs from the query. Say so out loud: most
  problems a user brings will sit outside it.
- **Every point cloud number in the current catalogue was measured with a defect.**
  The neighbour graph was rescaled using the minimum and maximum over the training
  batch as well as over the points, so a cloud was scaled differently during training
  than at evaluation. Measured effect: 6.7 per cent of k=16 neighbour slots flip to a
  different galaxy and all 32 tested clouds get a different neighbour set. The rescale
  has been removed and a corrected sweep is running. Comparisons between cloud entries
  are internally consistent, because every arm carried the same defect, but the
  absolute values may move. Say this whenever you quote a cloud number.
- Entries marked `unmeasured` are defined but have produced no number. Never recommend
  one, and never quote a config as though it were a result.
- `nParameters` is null for entries whose parameter count could not be built. Null
  means not obtained, not zero.
