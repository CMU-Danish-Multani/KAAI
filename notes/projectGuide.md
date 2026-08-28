# KAAI Project Guide

One file. Read this before a meeting, or to restart after time away.

Written for someone with no astrophysics background and some coding background.
Every technical term is defined the first time it appears, and again in the
glossary at the end.

Last updated 2026-08-28. Update the Results and Status sections whenever a run
finishes. Do not edit past entries, add new ones.

---

# 1. The problem, from scratch

## 1.1 What we are measuring

The universe began about 13.8 billion years ago, hot and almost perfectly
smooth. Almost. Some regions held a tiny fraction of a percent more matter than
others.

Gravity amplified that. A denser region pulls in more matter, which makes it
denser still. Over billions of years those tiny ripples became everything we
see: galaxies in clumps, clumps joined by filaments, and enormous near empty
gaps between them. The whole pattern is called the **cosmic web**.

The key point: how fast that clumping happened depends on the physics. Change
the physics and you get a differently textured web. So the arrangement of
galaxies today is a fingerprint of the laws that made it.

## 1.2 The two numbers

Two quantities do most of the work.

**Omega_m** ("omega matter"). The fraction of the universe's energy that is
matter, both ordinary and dark. About 0.3 in our universe. More matter means
more gravity means faster clumping.

**sigma_8** ("sigma eight"). How lumpy the universe is, measured on spheres
about 8 megaparsecs across. A megaparsec is roughly 3.26 million light years, so
these are huge spheres holding many galaxies. About 0.8 in our universe. Higher
means lumpier.

These two are partly interchangeable. More matter clumping slowly can look like
less matter clumping quickly. That trade off is called a **degeneracy**, and it
is why we infer both together rather than separately.

## 1.3 Why the problem is hard

Forwards is easy. Pick Omega_m and sigma_8, run a simulation, get galaxies.

Backwards is the science. We have one universe, we cannot rerun it, and we want
to know which values produced it.

The normal statistical route is closed. Normally you write a **likelihood**, a
formula for how probable this data is given these parameter values, combine it
with a **prior** (what you believed before seeing data), and Bayes' theorem
gives the **posterior** (what you should believe now). The posterior is the goal,
and it is not a single number. It is a full probability distribution, so it
carries an error bar.

Nobody can write that formula for a 3D map of galaxies. The physics is
nonlinear, and real galaxies involve gas cooling, stars igniting, supernovae
exploding and black holes blowing gas out. There is no clean equation.

## 1.4 The workaround

If you cannot write the likelihood, learn it.

1. Run thousands of simulations, each with different Omega_m and sigma_8.
2. You now have thousands of pairs: these parameters produced this galaxy map.
3. Train a network on those pairs to run backwards. Data in, probability
   distribution over parameters out.

This field has three names for the same thing: **simulation based inference
(SBI)**, **implicit likelihood inference (ILI)**, and likelihood free inference.

## 1.5 Three ways to do it

This distinction is the spine of the project.

**NPE, neural posterior estimation.** Learn P(theta|x) directly. Data in,
posterior out. Train once, and any new observation gets an answer instantly.
That property is called **amortized**: the cost is paid up front.

**NLE, neural likelihood estimation.** Learn P(x|theta) instead, the thing we
could not write down. Then you still have to run MCMC to turn it into a
posterior, and that costs real time for every single observation. The payoff is
that you can change the prior afterwards without retraining.

**NRE, neural ratio estimation.** Train a classifier to answer "do this data and
these parameters belong together?", then convert its confidence into a
likelihood ratio. No assumption about the shape of the distribution.

**MCMC** (Markov chain Monte Carlo) is the classical way to draw samples from a
distribution you can evaluate but cannot sample directly. It is slow.

## 1.6 The networks that output distributions

A normal network outputs a number. These must output a whole distribution. Such
a network is a **neural density estimator (NDE)**. Two designs matter here.

**Normalizing flow.** Start with a plain bell curve and apply a chain of learned,
reversible warps until it matches the shape you want. Two variants appear
constantly: **MAF** (masked autoregressive flow) and **NSF** (neural spline
flow). NSF is more flexible per warp and costs more.

**MDN, mixture density network.** Output the centres and widths of a few
Gaussians and add them together. Cheap and stable, and cannot represent a sharply
non Gaussian shape.

## 1.7 Calibration, which is the thing that actually matters

A posterior makes two claims: a best guess, and an error bar. Both must be right.

The test. Take 100 test simulations where the true answer is known. For each,
ask whether the truth fell inside the range the model said held 68 per cent of
its belief. An honest model should be right about 68 times.

- 68 of 100. Honest.
- 55 of 100. Error bars too small. The model is **overconfident**. This is the
  dangerous failure, because you publish a result claiming more precision than
  you have, and you can be confidently wrong.
- 90 of 100. Error bars too wide. Wasteful but safe.

This check is called **coverage**. Refined versions have names that appear in the
papers: **PIT** for one parameter at a time, and **TARP** for the multi parameter
case. TARP is the one Ho et al. adopt as necessary and sufficient.

Hold on to this section. It is where the results land.

---

# 2. The data

**CAMELS** and **CAMELS-SAM** are libraries of cosmological simulations. Each is
a box of simulated universe, run with a chosen Omega_m and sigma_8, producing a
catalogue of galaxy positions.

| | CAMELS | CAMELS-SAM |
|---|---|---|
| Box size | 25 Mpc/h | 100 Mpc/h |
| Galaxies per box | varies, 588 to 4511 | fixed, exactly 5000 |
| Train / validate / test | 600 / 200 / 200 | 600 / 204 / 196 |

## 2.1 Compressing a galaxy map into 25 numbers

Feeding raw 3D positions to a network is possible but hard. The standard
compression is the **two point correlation function**, written xi(r).

Take every pair of galaxies, measure how far apart they are, and histogram those
distances. Compare against what you would get from the same number of galaxies
scattered at random. xi(r) is the excess. xi = 0 means no different from random.
xi = 1 means twice as many pairs at that separation as random would give.

A clumpy universe has a large excess of close pairs. That is the signal.

We compute it in 25 logarithmically spaced distance bins, so **every network in
this project takes a vector of 25 numbers as input**.

We recompute this from raw positions rather than using the files shipped with the
dataset, because those files used different distance bins than the published
results did. The code carries a self check: scatter uniform random points and
xi must come out zero in every bin. MEASURED largest deviation 0.0076 against a
tolerance of 0.05.

---

# 3. The assignment

## 3.1 LtU-ILI

Matt Ho, the supervisor, wrote **LtU-ILI** (arXiv 2402.05137). Before it, three
separate Python packages each did simulation based inference with their own
conventions, so nobody could compare them fairly. LtU-ILI puts one interface over
all of them, adds ways to feed in images and graphs, and adds the calibration
checks. It is infrastructure, and that is the point of it.

## 3.2 Proposal 2.6, "LtU-ILI with shared agentic hyperpriors"

The problem it names: every new group repeats the same work. Search the
literature, pick an architecture, implement it, tune it, check calibration. What
actually works lives in people's heads and gets rediscovered from scratch.

Two deliverables:

1. **A model zoo.** A public, curated collection of architectures for common
   astrophysics tasks, each with standardised evaluation results and documented
   failure modes.
2. **A Claude skill.** Describe your data modality, parameter dimensionality and
   compute budget, and get back ranked architecture recommendations.

"Shared agentic hyperpriors" means the accumulated results become a starting
belief about which settings work, shared across tasks, so a new problem does not
begin tuning from zero.

## 3.3 The stated success criteria

1. The skill returns the best architecture for at least 4 of 5 held out problem
   descriptions.
2. A new user following a recommendation reaches MCMC equivalent posterior
   quality in under 2 GPU hours on a held out CAMELS task.
3. At least three external groups contribute a model within three months.

## 3.4 The 1 to 2 week proof of concept, verbatim from the brief

- Populate the zoo with architectures already evaluated in ltu-ili: normalising
  flows, neural ratio estimators, mixture density networks, benchmarked on three
  CAMELS inference tasks at matched compute.
- Implement the Claude skill as a RAG wrapper over zoo metadata and test whether
  it returns correct recommendations on five held out problem descriptions from
  recent astrophysics ML papers.

**RAG** means retrieval augmented generation: the model looks things up in a
store of documents before answering, instead of relying on memory.

---

# 4. What has been built

## 4.1 The framework, verified before use

LtU-ILI installed in its own environment. Its own test suite passes 14 of 16.

Then his Section 4.1 toy problem was reproduced as a wiring check. It is analytic,
so a mistake shows up as a violated prediction rather than a plausible wrong
number. Four predictions were written down before running.

The decisive one: in that problem one parameter enters only as theta_2 squared,
so positive and negative theta_2 are mathematically indistinguishable and the
model must fail to recover it.

- MEASURED theta_2 recovery R2 = -0.008, indistinguishable from zero.
- MEASURED theta_0 and theta_1 anticorrelated at -0.76, as the equations require.
- MEASURED coverage 0.713 and 0.958 against nominal 0.68 and 0.95.
- INTERPRETED the pipeline reproduces a symmetry nobody told it about, so the
  wiring is right.

## 4.2 The three tasks

| task | dim(theta) | data regime |
|---|---|---|
| camelsJoint | 2 (Omega_m, sigma_8) | counts vary |
| camelsOmega | 1 (Omega_m) | counts vary |
| camelsSamJoint | 2 | counts fixed at 5000 |

Chosen so a recommendation has something to discriminate on. Parameter
dimensionality is one of the three inputs the skill takes.

## 4.3 The eight architectures

Every entry is an LtU-ILI configuration, not bespoke code, so a recommendation is
something a user can actually run.

| family | entries |
|---|---|
| normalising flows | npeMaf, npeNsf, nleMaf, npeMafEnsemble4 |
| mixture density networks | npeMdn, nleMdn |
| ratio estimators | nreMlp, nreResnet |

An **ensemble** means training several copies with different random starts and
averaging them. It is the standard remedy for overconfidence.

## 4.4 Matched compute

Identical batch size, learning rate, epoch cap and early stopping rule for every
entry, with wall clock recorded so compute is measured rather than assumed equal.

A **seed** is the starting number for the random number generator. Running three
seeds and reporting the spread is how you tell a real difference from luck.

Eight architectures, three tasks, three seeds. 72 runs.

## 4.5 The metric validation, which turned out to matter most

Before trusting any calibration number, both metrics were tested against
posteriors whose true coverage is known analytically, using a conjugate Gaussian
so the exact answer is available.

Two things came out.

- BUG CAUGHT the TARP output was being read off the wrong axis. Every TARP number
  produced before this was wrong. Fixed.
- MEASURED at 100 evaluation points, an exactly calibrated posterior can read as
  low as 0.605. So 100 points cannot resolve an effect of the size being claimed.
- DECISION the whole sweep was rerun on all 200 test simulations.
- MEASURED after the fix, on an exactly calibrated posterior, marginal coverage
  reads 0.678 and TARP reads 0.678, against a true 0.680. Both order
  overconfident, calibrated and underconfident correctly.

LESSON a metric that has never been run against a case with a known answer is not
a measurement, it is a number.

---

# 5. Results

Final. Sweep complete at 72 of 72 cells, 0 failures, 200 evaluation points, 1000
posterior draws, 3 seeds, 16.6 hours of compute. **R2** is the accuracy score: 1.0
perfect, 0.0 no better than guessing the average.

## 5.1 Not one architecture is calibrated, and both tests agree

| | measured | nominal | how many pass |
|---|---|---|---|
| coverage at 68 per cent | mean 0.603, range 0.520 to 0.672 | 0.680 | **0 of 24** |
| TARP at 68 per cent | mean 0.617, range 0.574 to 0.666 | 0.680 | **0 of 16** |
| coverage at 95 per cent | mean 0.887, range 0.817 to 0.942 | 0.950 | **0 of 24** |

- MEASURED no architecture-task pair reaches nominal coverage on any test.
- INTERPRETED every entry reports error bars that are too small.
- HONEST CAVEAT this is at 800 training simulations, and cannot yet be separated from
  "800 simulations is not many". Quijote can settle it and has not been run.

## 5.2 Accuracy does not separate the architectures

- MEASURED on Omega_m in camelsJoint all eight land between 0.806 and 0.870, a spread
  of 0.064.
- INTERPRETED an accuracy leaderboard calls them interchangeable while the coverage
  column says they are all wrong about their own precision. This is the clearest
  result we have and the central argument for the zoo.

## 5.3 NPE beats NLE, confirming the framework paper's own rule

- MEASURED on sigma_8 in camelsJoint, best NPE 0.371, best NLE 0.193, gap +0.179.
- PUBLISHED LtU-ILI Section 2.3 says high dimensional input to low dimensional output
  favours NPE. Here dim(x) is 25 and dim(theta) is 2.
- CORRECTION our registered prediction hedged that neither would be favoured. The rule
  was right and the prediction was wrong.

## 5.4 Compute varies 4,797 times for the same answer

| architecture | total per cell | train | inference | parameters |
|---|---|---|---|---|
| npeMdn | 0.7 s | 0.7 | 0.1 | 7,930 |
| npeMaf | 3.3 s | 2.2 | 1.1 | 33,770 |
| npeNsf | 6.6 s | 5.1 | 1.6 | 78,175 |
| npeMafEnsemble4 | 11.0 s | 8.8 | 2.3 | 135,080 |
| nreMlp | 400.2 s | 1.2 | 399.0 | 4,201 |
| nreResnet | 752.5 s | 1.0 | 751.4 | 16,751 |
| nleMaf | 1,889.7 s | 4.2 | 1,885.5 | 45,500 |
| nleMdn | 3,570.7 s | 4.5 | 3,566.2 | 94,755 |

- MEASURED training is a few seconds for every entry. The whole spread is MCMC.
- INTERPRETED this is the amortization argument of Section 2.3 made concrete. NPE
  trains once and samples instantly; NLE and NRE need a fresh MCMC run per observation.

## 5.5 Ensembling barely helps

- MEASURED four MAFs moved coverage by +0.010, +0.013 and +0.003 across the three
  tasks, against a gap to nominal of about 0.08.
- PUBLISHED LtU-ILI Sections 3.2 and 6 recommend ensembling as the fix.
- INTERPRETED at 800 simulations it closes under a fifth of the gap at four times the
  compute. Not a claim about other scales.

## 5.6 The zoo, and the one entry that passes

`ili_kaai/results/zoo.json`, 8 entries. The admission rule admits on calibration being
MEASURED, never on it passing, because a gate that rejects everything leaves an empty
zoo. The verdict travels with the entry.

| entry | verdict | sigma from nominal |
|---|---|---|
| nreMlp | **calibrated** | -1.1 |
| npeNsf | overconfident | -3.2 |
| npeMdn | overconfident | -6.7 |
| npeMafEnsemble4 | overconfident | -7.0 |
| npeMaf | overconfident | -7.8 |
| nreResnet | overconfident | -9.1 |
| nleMaf | overconfident | -9.5 |
| nleMdn | overconfident | -11.6 |

- MEASURED nreMlp accuracy is indistinguishable from the cheapest entry: 0.865 against
  0.866 on Omega_m. It is the only entry with trustworthy error bars and it costs 399 s
  of inference against 0.1 s.
- BUG CAUGHT the first admission rule used a tolerance of 0.05 chosen by eye. That is
  an arbitrary rule dressed as a measurement.
- METHOD measured it instead. An exactly calibrated posterior at 200 points reads
  0.676 against nominal 0.680, bias -0.004, and a 3-seed mean has standard deviation
  0.011, so 2 sigma is 0.022. `ili_kaai/results/calibrationNoiseBand.json`.
- CORRECTION under the measured threshold npeNsf flips from calibrated to
  overconfident. Its verdict was wrong for as long as the tolerance was invented.

## 5.7 Where the miscalibration lives

A prior is a box, and its walls sit exactly where the CAMELS design stops. About a
tenth of test points sit within 10 per cent of a wall.

| | our deficit | a provably correct posterior | excess |
|---|---|---|---|
| Omega_m | -0.178 | -0.158 | **-0.020** |
| sigma_8 | -0.581 | -0.279 | **-0.303** |

- MEASURED coverage collapses near the prior walls, worst for the parameter that is
  least constrained.
- METHOD the baseline is a posterior that is exactly correct by construction under the
  same prior box, matched to our measured R2, so the difference is the part the
  architectures are responsible for.
- INTERPRETED at the Omega_m boundary the models behave correctly and the deficit is
  entirely the prior. At the sigma_8 boundary they lose about twice what the prior
  accounts for, and that excess is a genuine failure.
- MEASURED nreMlp, the one entry that passes overall, carries the same boundary
  failure: -0.179 and -0.512. So this is a property of the setup, not of any
  architecture.
- INTERPRETED an aggregate calibration number hides local failure exactly as an
  accuracy number hides aggregate failure.
- PUBLISHED the literature already knew both halves. Thiele 2026 Figure 5 says a
  uniform prior with sharp edges "would still introduce errors" because neural
  functions are regular. Section 3.2 says miscalibration that "averages out" needs
  local coverage tests such as local C2ST. We rediscovered the need for both.
- HONEST CAVEAT the correct-posterior baseline returned 0.655 overall rather than
  0.680, about 2 sigma low. The comparison is a difference so most cancels, but the
  baseline is not perfectly centred and I have not chased down why.

## 5.8 One earlier finding, kept for reference

Before the project was re-grounded on LtU-ILI a different failure mode was measured. In
CAMELS the number of galaxies correlates with Omega_m at 0.73, so a network can score
well by counting rather than learning structure.

- MEASURED sum pooling 0.809 against mean pooling 0.660, a gap of +0.149.
- MEASURED hold the count fixed and the gap collapses to +0.0003.
- NOTE this belongs in the zoo as one documented failure mode. It should not lead any
  conversation about the project.

# 6. What is not done

| | Status |
|---|---|
| The zoo assembled from LtU-ILI results | **Done**, 8 entries, `ili_kaai/results/zoo.json` |
| The Claude skill | Not built. The brief's named primary contribution. |
| Five held-out problem descriptions from real papers | Not done |
| Quijote (19,651 training sims, 5 params, 1 Gpc/h) | **Wired into `tasks.py`, never run** |
| Point cloud, field, image, spectra modalities | Not started |
| MCMC equivalence (success criterion 2) | Blocked on CAMELS, no reference posterior exists |
| Contribution path for external groups (criterion 3) | Not written |

Coverage of the data actually available:

    modalities covered        1 of 4   (summary vectors only)
    data suites covered       2 of 3   (not Quijote)
    training simulations used 800      out of 20,451 available

The zoo is complete and internally consistent on what it has seen. It has seen the
smallest and easiest slice of the data on disk.

# 7. Action items

## 7.1 Ours, in order

1. **Run the existing 8 entries on Quijote.** 33 times the training data, 5 parameters
   instead of 2, 6,550 test simulations instead of 200. Wired and never run. This
   settles open question 2 by itself.
2. **Build the skill.** A RAG wrapper over the zoo metadata. The brief poses the
   retrieval method as an open question, so build both dense retrieval over structured
   metadata and few-shot with evaluation summaries, then measure.
3. **Five held-out problem descriptions** from published astrophysics ML papers, not
   self written. Material in `notes/related_papers.md` and `notes/zooCandidates.md`.
4. **Widen the zoo.** 21 architecture and engine combinations are verified to build
   across two backends. See `notes/zooCandidates.md` for the ranked list.
5. **Write the contribution path**, what a submitter must supply for admission.

## 7.2 Questions for the supervisor

1. How many test simulations are enough for a coverage claim? CAMELS gives 200 and our
   own measurement says the noise band there is 0.022 at 2 sigma. Every entry is
   admitted on a coverage measurement, so this constrains the whole zoo.
2. Should an entry be admitted at 2 sigma, or is that too loose? It changes which
   entries the skill recommends.
3. Dense retrieval or few-shot with evaluation summaries for the skill, or both?
4. Breadth first, or the skill first on the 8 entries that exist?

# 8. How to explain this in sixty seconds

Applying machine learning to a new astrophysics problem means guessing at
architecture choices that other groups already tested and never wrote down. We
are building the record, measured the same way for every entry, plus a tool that
reads it for you.

The first real result from that record: **every architecture we tested is
overconfident about its own error bars, and the accuracy number everyone reports
would never tell you.** All eight score within 0.064 of each other on accuracy,
so an accuracy table calls them interchangeable. None of them reaches nominal
coverage on any test.

The second: **compute varies about three thousand times for the same answer**,
because likelihood and ratio methods need a fresh MCMC run per observation while
posterior methods do not. That is why "what is your compute budget" is a real
question and not a form field.

---

# 9. Glossary

**Amortized.** Train once, then every new observation is answered instantly. NPE
is amortized. NLE and NRE are not.

**CAMELS, CAMELS-SAM.** Libraries of simulated universes with known parameters.

**Calibration.** Whether the stated error bars are the right size.

**Coverage.** The test for calibration. Out of many test cases, how often does the
truth land inside the stated interval.

**Degeneracy.** Two parameters that trade off, so different combinations produce
similar data.

**Embedding network.** A network that compresses raw data (an image, a graph, a
point cloud) into a short vector before the density estimator sees it.

**Ensemble.** Several copies of a model trained with different random starts,
averaged together.

**ILI, implicit likelihood inference.** Same as SBI.

**Likelihood.** The probability of the data given the parameters.

**MAF, masked autoregressive flow.** A normalizing flow variant.

**MCMC.** Markov chain Monte Carlo. A classical, slow way to draw samples from a
distribution you can evaluate but not sample directly.

**MDN, mixture density network.** Outputs a few Gaussians and mixes them.

**NDE, neural density estimator.** A network whose output is a probability
distribution rather than a number.

**NLE, neural likelihood estimation.** Learns P(x|theta). Needs MCMC afterwards.

**NPE, neural posterior estimation.** Learns P(theta|x) directly. Amortized.

**NRE, neural ratio estimation.** Learns a classifier and converts it to a
likelihood ratio.

**NSF, neural spline flow.** A more flexible, more expensive normalizing flow.

**Omega_m.** Fraction of the universe's energy that is matter. About 0.3.

**Overconfident.** Error bars too small. The dangerous failure direction.

**PIT.** A calibration check on one parameter at a time.

**Posterior.** What you should believe about the parameters after seeing the data.
A full distribution, not a number.

**Prior.** What you believed before seeing the data.

**R2.** Accuracy score. 1.0 perfect, 0.0 no better than guessing the average.
Negative means worse than guessing the average.

**RAG, retrieval augmented generation.** Look things up in a document store before
answering.

**SBI, simulation based inference.** Learn the relationship between parameters and
data from simulations, because the likelihood cannot be written down.

**Seed.** The starting number for the random number generator. Multiple seeds show
whether a difference is real or luck.

**sigma_8.** How lumpy the universe is on 8 Mpc/h scales. About 0.8.

**TARP.** A calibration check across all parameters at once. The one Ho et al.
treat as necessary and sufficient.

**Two point correlation function, xi(r).** How much more often galaxy pairs occur
at a given separation than random scattering would give.

---

# 10. Where things live

| | |
|---|---|
| This guide | notes/projectGuide.md |
| Append only work log, every correction and retraction | runLog.md |
| Current plan, stages and non goals | notes/plans.md |
| Literature notes | notes/related_papers.md, notes/literature.md |
| Sweep results | ili_kaai/results/sweep.json |
| The zoo catalogue | ili_kaai/results/zoo.json |
| Candidate architectures from the literature | notes/zooCandidates.md |
| Compute request for the SCS cluster | notes/comms/computeRequest.md |
| Measured admission threshold | ili_kaai/results/calibrationNoiseBand.json |
| Edge coverage and its baseline | ili_kaai/results/edgeCoverage.json, edgeBaseline.json |
| Metric validation | ili_kaai/results/tarpCalibration.json |
| Toy problem check | ili_kaai/results/toyModel.json |
| Parameter counts | ili_kaai/results/paramCount.json |
| Superseded pre-LtU-ILI work | archive/ with a README explaining each piece |
| Draft email to the supervisor | notes/comms/mattUpdate1.md |

---

# 11. Update log

Append a line here each time this file changes. Do not edit past lines.

- 2026-08-28. Created. Sweep at 65 of 72 with corrected TARP. Both calibration
  metrics agree that zero of 22 architecture-task pairs reach nominal coverage.
- 2026-08-28, later. Sweep finished 72 of 72. Zoo assembled, 8 entries. Admission
  threshold changed from an invented 0.05 to a measured 0.022, which flipped npeNsf
  from calibrated to overconfident. Edge coverage measured and compared against a
  provably correct posterior: the Omega_m boundary deficit is entirely the prior, the
  sigma_8 one is not. Literature sweep written to notes/zooCandidates.md. Quijote
  found on disk with 19,651 training simulations and wired into tasks.py, not yet run.
  Compute request for the SCS GPU cluster drafted.
