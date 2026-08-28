# Optimal Neural Architectures for CosmoBench

Apply BioArc's NAS methodology to CosmoBench's cosmological parameter
prediction, for both point clouds and merger trees, with an agentic layer that
proposes architectures for new tasks.

---

## 1. The target: what we have to beat

CosmoBench Table 2 — **cosmological parameters from point clouds** (R², higher better):

| Method | Quijote Ω\_m / σ\_8 | Params | Time | CAMELS-SAM | CAMELS |
|---|---|---|---|---|---|
| 2PCF (Corrfunc + MLP) | **0.85 / 0.84** | 11K | 2 min | 0.73 / 0.82 | **0.84** / **0.30** |
| LLS (pairwise stats) | 0.83 / 0.80 | **49** | 24 sec | **0.77** / 0.82 | 0.78 / 0.28 |
| GNN | 0.80 / 0.77 | 671K | 1 day | 0.75 / **0.83** | 0.78 / 0.24 |
| GNN w/o edge-MP | 0.80 / 0.79 | 128K | 1 day | 0.72 / **0.84** | 0.80 / 0.27 |

**Read this table carefully — it defines the whole project.** On Quijote a
**49-parameter least-squares fit beats a 671,000-parameter GNN that trains for a
day**. The GNN is not slightly behind; it loses on both parameters. Deep learning
only pulls ahead on σ\_8 at the smaller scales (CAMELS-SAM). This is the gap NAS
is supposed to close, and it is also a warning: the winning architecture may not
be a big GNN.

CosmoBench Table 5 — **parameters from merger trees** (node features M, c, v\_max, a):

| Method | Ω\_m | σ\_8 | Params | Time |
|---|---|---|---|---|
| 1NN (KS statistic) | 0.64 | 0.31 | — | 4h49m |
| DeepSets (topology discarded) | 0.993 | 0.80 | 0.65k | 10 min |
| GNN (MPNN, 4-layer) | **0.996** | **0.82** | 2.8k | 13 min |

Trees are a different regime: models are **tiny** (thousands of parameters, not
hundreds of thousands) and Ω\_m is essentially solved at 0.996. **The headroom is
σ\_8 (0.82), not Ω\_m.** Note DeepSets gets 0.993 while ignoring all edges — so
topology is worth ~0.003 on Ω\_m and 0.02 on σ\_8. Any architecture search here is
fighting for small margins.

## 2. The method: how BioArc actually works

Four stages.

### 2.1 Search space

Each candidate is a **path** `a = (h, m)` of depth `d`:

- **Depth** `d ∈ D = {3, 4, 5, 6}`
- **Block type** `m_i ∈ M = {CNN, LSTM, Transformer, Mamba, Hyena}`
- **Hidden dim** `h_i ∈ H = {64, 128, 256, 512}`

Total space `A = ⋃_d (C_dim^(d) × C_type^(d))` — millions of candidates, pruned
to **360** by three mechanisms:

1. **Monotonic width constraint** `h_i ≥ h_{i-1}`, max dim fixed at the last
   layer. This is not just regularisation — it deliberately biases the space so
   parameter-heavy blocks appear in more valid paths and therefore get sampled
   more during supernet training, matching training intensity to block capacity.
2. **Distance-based filtering** on log-transformed dimensions, to drop
   topologically redundant paths.
3. **K-Means** on one-hot encoded architecture vectors; keep the centroids.

### 2.2 Supernet

Weight sharing. Collect every unique block across all paths,
`L = {l | ∃a ∈ A, l ∈ a}`, and let `W = ⋃_{l∈L} W_l`. Any path's weights are a
subset, `w(a) = {W_{l_i} | l_i ∈ a} ⊆ W`. One network contains every candidate.

### 2.3 Supernet pretraining — Single Path One-Shot

Activate **exactly one path per forward pass**, sampled uniformly:

```
min_W  E_{a~A} [ L(A(X; w(a))) ]
```

Single-path updates act as regularisation: they decouple block interactions and
prevent co-adaptation, so shared blocks learn features that transfer rather than
over-specialising to their neighbours. Pretraining is self-supervised — masked
modeling, contrastive, or next-token prediction.

### 2.4 Evaluation and ranking

**Each path is trained independently, not fine-tuned inside the supernet.**
BioArc is explicit about why: shared weights are a *compromise* optimised for
expected performance across all paths, never optimal for any single one. Ranking
off supernet weights directly would be measuring the compromise. Paths are either
fine-tuned from inherited supernet weights or retrained from scratch.

### 2.5 The agentic layer

Three tiers of architecture predictor, in increasing sophistication:

| Approach | Mechanism |
|---|---|
| **NN Predictor** | `h_a = GNN(I, X_a)` where `X_a = (onehot(m_i) ‖ Z(h_{i-1}) ‖ Z(h_i))`; task embedded by a PLM as `h_t`; MSE against true performance |
| **LLM + RAG** | Embed task → retrieve top-*n* similar historical tasks + their top-*k* architectures → prompt LLM to reason and emit top-*m* |
| **BioArcAgent** | Four decoupled roles (below) |

BioArcAgent pipeline:

| Role | Input | Output |
|---|---|---|
| Analyst | raw task description | structured metadata (modality, objective) |
| Task Retriever | metadata | semantically aligned tasks in knowledge base |
| Arch. Retriever | aligned tasks | proven architectures + empirical performance |
| Predictor | architectures + metrics | predicted optimal architecture |

Scored by Precision@k, Recall@k, Hit Rate@k against the empirically verified
top-architecture set. BioArcAgent reached Hit Rate 0.500 / Precision 0.500 @1,
versus 0.167 / 0.033 @5 for the NN predictor and 0.000 @1 for plain LLM+RAG.
GPT-4o was the default backbone; Qwen3-4B failed outright (0.00), so the search
space exceeds small-model reasoning capacity.

**Honest caveat:** these agent numbers come from a handful of tasks — Hit Rate
0.167 is literally 1-of-6. The ranking of the three methods is plausible, but the
absolute values are noisy and should not be treated as precise.

## 3. The transfer problem — the part that needs real work

**BioArc's block palette does not port to CosmoBench.** CNN, LSTM, Transformer,
Mamba, and Hyena are all *sequence* models operating on an ordered 1-D token
stream. CosmoBench data is point clouds and directed trees: permutation-invariant
sets and graphs carrying E(3) symmetry. There is no canonical ordering of 5,000
halos to feed a Mamba.

So the framework transfers, but the search space must be re-derived. The honest
mapping:

| BioArc axis | CosmoBench equivalent |
|---|---|
| Block types (CNN/LSTM/…) | message-passing variants, DeepSets/PointNet, E(3)-equivariant layers (EGNN), point-attention, **and a linear invariant-feature pathway** |
| Depth `d` | number of message-passing hops (baseline GNN uses 4) |
| Hidden dim `h` | embedding width (tree baselines find **16** best — larger overfits sparse trees) |
| **Tokenization** (k-mer / BPE) | **graph construction**: radius cutoff `R_c`, kNN vs radius graph, Delaunay tetrahedra, higher-order edges |

That last row is the important one. BioArc found architecture and tokenization
**deeply entangled** — architecture cannot be evaluated in isolation from how the
input is discretised. The exact analogue here is that a GNN's performance is
inseparable from `R_c` and the graph-construction rule. Any search that fixes the
graph and varies only the network is searching the wrong space.

The invariant-feature linear pathway must be *in* the search space, not a
baseline outside it. Otherwise the search cannot rediscover the thing that
currently wins.

## 4. Compute reality check

CosmoBench's own GNN took **1 GPU-day** for Quijote. BioArc trains **360**
architectures independently. 360 GPU-days on an M-series Mac is not happening.

Feasible scaling, cheapest first:

| Track | Cost per model | 40-arch search |
|---|---|---|
| **CS-Trees** (GNN, 2.8k params) | 13 min | ~9 hours |
| CAMELS / CAMELS-SAM point clouds | 2–3 hr | ~4 days |
| Quijote point clouds | 1 day | infeasible |

The K-Means reduction is the lever: BioArc chose 360; nothing forces that number.

## 5. Recommended starting point

**Start with CS-Trees.** Reasons: models are 2.8k parameters and 13 minutes, the
full search fits in one overnight run, and there is a clean, well-defined target
— **σ\_8 at 0.82** with Ω\_m already saturated at 0.996. A single headline number
to move, on the only track this hardware can actually search.

Then port whatever transfers to CAMELS point clouds, and treat Quijote as
inference-only using an architecture found elsewhere.

## 6. Decisions (locked)

| Axis | Decision |
|---|---|
| Compute | **This Mac (MPS) for now** — no cluster access yet. Constrains us to the CS-Trees track. |
| Search strategy | **Both** — Optuna sequential *and* weight-sharing supernet, compared |
| Success criteria | **Both** — reproduce BioArc's method in a new domain *and* beat CosmoBench baselines |
| Agent | **Both, staged** — BioArc's predictor first, then close the loop |

**Amended 2026-08-17**, after reading the third paper in `resources/`. Two rows
added rather than editing the four above, so the original decisions stay
readable:

| Axis | Decision |
|---|---|
| Hyperparameter tuning | **SARA (agentic Bayesian optimisation).** Training knobs only, not architecture. Full reasoning and boundaries in §10. Its role in the architecture search itself is deferred, not decided. |
| Headline claim | **All three at once**: beat the CosmoBench baselines, show the NAS method transfers to a new scientific domain, and measure how faithfully a weight-sharing supernet ranks architectures. |

This is the maximal scope on every axis. It only works if strictly sequenced,
because each phase produces the input the next one needs.

The headline-claim decision deserves one caution, because it is the only row
where the three options are not equally likely to land. Beating the baselines
depends on there being headroom above σ\_8 = 0.82 that is architectural rather
than noise, and §8 says part of that gap is irreducible. The other two claims
depend only on doing the work carefully, so they are close to guaranteed if the
project runs to completion. Pursue all three, but do not let the one that might
not arrive decide whether the work gets written up.

**Amended again 2026-08-17**, after the arXiv sweep recorded in
[literature.md](literature.md) and [findings.md](findings.md). This block
partly supersedes the one above, and says so rather than editing it:

| Axis | Decision |
|---|---|
| Venue | **An ML venue for the sciences, not astronomy.** ML4PS at NeurIPS first, then an ML main-track submission. Astronomy has no archival conference, it publishes in journals, so an astro-facing systematics version would be a separate paper off the same experiments. |
| Spine | **Automated architecture search amplifies shortcut learning.** One claim carries the paper. |
| Supporting sections | Representation is a larger lever than architecture. Supernet ranking fidelity in a regime where the linear model wins. |
| Phases 3 and 4 | **Novelty superseded** by AgentNAS (arXiv 2607.07984, July 2026). "An LLM proposes architectures and NAS searches" is published. Two openings survive: weight sharing inside a slotted space, and searching the input representation. |

**What this supersedes.** The "all three at once" row above is revised. Beating
the CosmoBench baselines drops from headline to supporting result. The reason is
not loss of nerve, it is scale: a claim about a *mechanism* shows up at small
scale, whereas a claim of *beating a baseline* requires full scale and therefore
hardware we do not have. Making the mechanism the spine is what makes the whole
project runnable on this machine.

## 7. Roadmap

> **Superseded 2026-08-17 by §11.** This section is left unedited because it
> records what was planned before the literature sweep, and deleting it would
> make the project look like it went straight there. Its Phase 0 gate is still
> correct and is still cited from `findings.md`. What changed: Phases 3 and 4
> lost their novelty to prior art (§6, second amendment), and the work
> reorganised around diagnostics that must run before any search space can be
> designed. Read §11 for what is actually being executed.

### Phase 0 — Data and harness *(in progress)*

Data downloaded (1.5 GB of the 324.5 GB available) and both datasets explored —
findings in `understanding_data.md`. Dataloader built and verified.

Still outstanding, and it is the gate for everything after: **reproduce one
published baseline number before building anything else.** The CS-Trees GNN at
Ω\_m R² = 0.996 is the cheapest check at ~13 min. If we cannot reproduce a
published number, every result downstream is uninterpretable. This remains the
single highest-value hour in the project.

### Phase 1 — Search space design

The real intellectual work, since BioArc's block palette does not transfer (§3).

- **Block palette:** MPNN variants, DeepSets/PointNet, EGNN (E(3)-equivariant),
  point-attention, and the linear invariant-feature pathway.
- **Graph construction axis** — the analogue of BioArc's tokenization: cutoff
  `R_c`, kNN vs radius graph, Delaunay tetrahedra, higher-order edges. Searched
  *jointly* with architecture, never fixed.
- **Pruning** to N candidates via monotonic width + K-Means. N is set by the
  cluster budget, not copied from BioArc's 360.

Because "beat the baselines" is now a success criterion, the linear pathway goes
**inside** the search space. Otherwise the search cannot rediscover the 49-parameter
model that currently wins on Quijote.

### Phase 2 — Both searches, compared

**Run Optuna first, then the supernet.** Not arbitrary: independently-trained
Optuna results are the *ground-truth ranking* needed to test whether the
supernet's shared weights rank architectures faithfully. Weight-sharing NAS is
known to suffer ranking disorder, and BioArc's own appendix addresses it. Running
the supernet first leaves nothing to validate it against.

Deliverable: a ranking-correlation number (supernet vs ground truth). That
comparison is a contribution in its own right and de-risks everything after it.

### Phase 3 — Knowledge base and predictor

Phases 2a/2b emit (architecture, task, R²) tuples — the knowledge base that could
not exist earlier. Then build BioArc's four roles: Analyst → Task Retriever →
Arch. Retriever → Predictor, scored by Precision@k / Recall@k / Hit Rate@k.

Use a frontier model. BioArc found Qwen3-4B scored 0.00 — the search space
exceeds small-model reasoning.

### Phase 4 — Closed loop

Agent proposes → train → results back into the knowledge base → propose again.
This extends past BioArc's paper, so it needs its own success metric: **does
best-found-so-far improve faster than random search and faster than Optuna's TPE
at equal GPU-hours?** Without that comparison a self-improving loop cannot be
distinguished from an expensive random search.

### Track order

CS-Trees (13 min/model) → CAMELS / CAMELS-SAM point clouds (2–3 hr) → Quijote
(1 day/model, likely inference-only with an architecture found elsewhere).

## 8. Which inputs the model is allowed to see

CosmoBench feeds the point-cloud task **positions only** — no mass, no
velocity. The paper is explicit: *"We use the point positions as the node
features X ∈ ℝⁿˣ³."* Merger trees are the opposite: all four node features
(mass, concentration, v_max, scale factor) are used.

That asymmetry is deliberate, and it is not about performance.

### The leak, measured

Footnote 1 of the paper warns that a model can *"cheat by focusing on halos of
the smallest mass"* — a halo is only tracked once it holds ~20 particles, and
particle mass is a direct function of Ω\_m. We tested this on the val splits:

| Single number, used alone — no structure, no model | CAMELS r(Ω\_m) | CAMELS-SAM r(Ω\_m) |
|---|---|---|
| number of galaxies | **0.758** | constant |
| smallest galaxy mass | −0.496 | 0.404 |
| average galaxy mass | 0.010 | 0.166 |

**In CAMELS, simply counting the galaxies gets 0.758.** The published GNN, using
real structure, gets 0.78. A shortcut requiring no learning nearly matches a
model that trains for hours.

CAMELS-SAM's fixed top-5000 selection is the fix: the count becomes constant and
the smallest-mass signal weakens. This is also why the merger trees were pruned
at 3×10¹⁰ M☉ — the mass cliff in the data is the anti-cheat measure, and it is
why mass correlates ~0.00 with σ\_8 in the trees. That signal was destroyed on
purpose.

### Hard constraint before Phase 1

**Screen every candidate input feature for leakage before it enters the search
space.** The test is cheap and the rule is simple:

> If a single number predicts the target without using any structure, it is a
> shortcut, not a feature.

This must be a constraint in code, not a convention. A NAS system optimising a
score will find these loopholes faster than a human reviewer will — and it will
report the result as a triumph.

### Two arms, kept separate

Because "beat the baselines" is a locked success criterion (§6), deviating from
the benchmark's inputs costs comparability. Scoring 0.92 against their 0.85 with
extra features means an easier problem was solved, not a better method found.

So run two clearly-labelled arms:

| Arm | Inputs | Purpose |
|---|---|---|
| **A — positions only** | 3D positions, as the benchmark defines | The headline. Directly comparable to published numbers. Where "we beat the baseline" can be claimed. |
| **B — features allowed** | positions + screened galaxy properties | A separate question: *how much is the benchmark leaving on the table by restricting inputs?* |

Arm B has real precedent. The paper cites Villaescusa-Navarro et al. [12]: the
internal properties of a **single galaxy** (stellar mass, stellar metallicity,
max circular velocity) predict Ω\_m to ~10% precision. Those are quantities
telescopes actually measure, so using them is legitimate science rather than
cheating — provided each one passes the screen above.

Arm B may well be the more novel contribution. Arm A is the safer one. Do not
merge their results into a single table.

### Astrophysical parameters: dropped as targets *(decided)*

We predict **only Ω\_m and σ\_8**. The astrophysical parameters (`A_SN1/2`,
`A_AGN1/2` in CAMELS; `A_sn1/2`, `Aagn1` in CAMELS-SAM) are extra labels stored
alongside the answers, and we ignore them. This matches the benchmark.

They were never inputs, so nothing is removed from what the model sees. But
they do not vanish either: **the feedback still happened inside the
simulation.** Two runs with identical Ω\_m and σ\_8 but different feedback
produce differently-arranged galaxies, so the model sees different universes
carrying the same correct answer.

That is irreducible noise and it **caps the achievable R²** — no architecture
can recover information that is not present. The proper term for what we are
doing is *marginalising over* the astrophysical parameters, and it is a
deliberate design choice by the CAMELS authors: they want models that work
regardless of feedback, since the real universe's values are unknown too.

**Consequence for the search:** part of the gap between a good architecture and
a great one may be noise. Do not over-interpret improvements of ~0.01 without
checking they exceed the bootstrap spread the paper reports.

Quijote is a different case — it is dark-matter-only, so there is no feedback at
all. Its `params/` holds five *cosmological* values (Ω\_m, σ\_8, Ω\_b, h, n\_s), so
the nuisance there is Ω\_b, h and n\_s varying instead. The paper reports those
three are harder to constrain than the two we target.

### Velocity is a separate case

Velocity does not leak in the way mass does, but predicting velocity *from*
position is a different task in the same benchmark. Our exploration found
`mean_speed` to be the single strongest predictor available (r = 0.76 for Ω\_m)
— the most useful number we are not allowed to touch. Keep it out of both arms.

Partial exception worth noting: the paper's redshift-space variant encodes
line-of-sight velocity into the positions themselves, and reports that GNNs
exploit it. If Arm B ever moves toward survey realism, that is the principled
route rather than feeding raw velocities.

## 9. How to pretrain the supernet

BioArc §3.2 trains its supernet with **self-supervised learning** — masked
modelling, next-token prediction, or contrastive — on a large corpus of
*unlabelled* biological sequence. None of those three transfers unchanged, and
before inventing replacements it is worth asking whether we need the step at all.

*Everything in this section is inference. Neither paper prescribes it.*

### First question: do we even need self-supervision?

BioArc pretrains because **labels are scarce and unlabelled data is abundant**.
That is not obviously our situation — every tree and every cloud arrives with
its Ω\_m and σ\_8 attached, free, because a simulation generated it.

So the default should be the simpler thing: **train the supernet supervised, on
the target task directly.** One task, labels everywhere, no invented objective to
get wrong. Single Path One-Shot does not require the loss to be self-supervised —
it only requires *a* loss.

**But there is a real counter-argument, and it is about effective sample size.**
CS-Trees train split has 14,997 trees — and only **600 distinct labels**, because
25 trees share each simulation's answer. For learning the *label mapping* the
effective dataset is 600, which is small. Self-supervision can exploit the
structure of all 14,997 trees without needing any label at all.

**Plan: supervised supernet as the default. Self-supervised pretraining as a
second arm, run only if the supervised supernet overfits** — which the 600-label
count makes plausible, and which the paper itself hints at when it reports that
*smaller* tree models beat larger ones.

### If we do pretrain — merger trees

Trees are the easier case, because unlike a point cloud **a tree has a time
axis**. Nodes are ordered by scale factor. So BioArc's sequence objectives have
genuine analogues here:

| Objective | Tree version |
|---|---|
| **Masked modelling** | Hide a node's four features, predict them from its neighbours. Direct analogue, and structure makes it well-posed. |
| **Next-token prediction** | Given the tree up to scale factor *a*, predict the next node. Works *because* trees are time-ordered — this is the one place BioArc's framing carries over literally. |
| **Contrastive** | See below — the data hands us positive pairs for free. |

**The free-positive-pairs observation.** Each simulation contributes **25 trees**
that share one universe and one answer. Any two of them are a natural positive
pair; two trees from different simulations are a natural negative. Contrastive
learning normally needs hand-built augmentations to manufacture positive pairs —
here the dataset already contains them, 25 deep, at no cost. This is the most
promising SSL route on the tree side and is not something either paper does.

**Already-downloaded auxiliary task.** `infilling_trees_25k_200_*.pt` (30 MB, on
disk) is CosmoBench's own merger-node reconstruction task, with real labels.
Pretraining on it would teach the model tree structure using supervision that
already exists — cheaper and less speculative than inventing an objective.

### If we do pretrain — point clouds

Harder, because a cloud has no order and no canonical starting point.

| Objective | Cloud version | Notes |
|---|---|---|
| **Contrastive on symmetries** | Two augmented views of one cloud should embed close together | **Preferred.** The augmentations are exact physical symmetries, not guesses: any translation (the box is periodic), 90° rotations and reflections (the cube maps onto itself), and random subsampling of galaxies |
| **Sub-box pairs** | Two disjoint sub-cubes from the same simulation share a cosmology → positive pair | Physically meaningful positives with no augmentation needed |
| **Masked density** | Hide a region, predict its *density* rather than exact positions | Predicting exact positions is ill-posed — the field is stochastic, so there is no single right answer. Density is recoverable |

Note the contrastive augmentations double as a **symmetry test**: if the encoder
does not already produce the same embedding for a shifted cloud, it has not
learned translation invariance, and that is a bug worth catching early.

### Ordering

1. Supervised supernet. Establish it works and check whether it overfits.
2. Only if it does: contrastive on the 25-trees-per-simulation pairs — the
   cheapest option, since the positives already exist.
3. Point-cloud SSL last. It is the most speculative part of the transfer and the
   cloud track is blocked on compute anyway.

## 10. Hyperparameter tuning with SARA

`resources/Agentic Bayesian Optimization through SARA.pdf` (Meta, arXiv
2608.00316). Like BioArc this is **method, not data**, and it enters the project
at exactly one point: **tuning the training knobs of an architecture we have
already chosen.** It is not, for now, the architecture search itself. Drawing
that boundary is most of what this section is for.

### 10.1 What Bayesian optimisation is

Suppose every setting you want to try costs 13 minutes to test. You cannot try
thousands. Bayesian optimisation is the standard way to spend a small number of
expensive tries well:

1. Fit a **surrogate**: a cheap statistical model (usually a Gaussian process)
   that guesses the score of settings you have not tried, and reports how unsure
   it is about each guess.
2. Use an **acquisition function** to choose what to try next, trading "this
   looks good" against "we know nothing about that region".
3. Run it, add the result, refit, repeat.

The uncertainty estimate is the whole trick. A model that only guessed scores
would keep re-testing near its current best and never look anywhere else.

Standard Bayesian optimisation fixes its own strategy before the first run:
which surrogate, which acquisition function, what bounds to search, which
measured outcomes count as objectives and which as constraints. That entire set
is chosen in advance and never revised in response to what is learned.

### 10.2 What SARA changes

SARA puts a language-model agent in the middle of that loop. The maths still
goes to a Gaussian process backend (`lenz`, built on BoTorch), but before
committing each expensive run the agent may:

- **probe**: ask the surrogate for predictions, diagnostics (cross-validated R²,
  per-dimension sensitivity), the current best, the whole trial log,
- **propose**: request candidate settings, optionally inside tightened bounds or
  near the current best, as many times as it likes at no evaluation cost,
- **reconfigure**: change the search bounds, swap the acquisition function, or
  promote a constraint into a second objective, mid-run, without invalidating a
  single past result,
- **override**: discard the surrogate's suggestion and run its own idea instead.

The second thing it adds is a **natural-language prior**. You describe the
problem in words, and the agent aims its early runs where a knowledgeable person
would start, rather than spending evaluations discovering that region.

### 10.3 The measured results, and the caveat that matters most here

From the paper, median over 10 seeds. Lower simple regret is better.

| Benchmark | Ax (classical BO) | SARA (Opus 4.8) |
|---|---|---|
| Hartmann 6-D, no prior | 0.0116 | 0.0114 |
| Ackley 10-D, no prior | 4.33 | 4.00 |
| Suzuki reaction yield, with prior | 1.31 | 0.46 |
| LCBench Dionis (**hyperparameter tuning**) | 3.01 | 2.60 |
| LCBench Covertype (**hyperparameter tuning**) | 3.63 | 3.50 |

Read the last two rows carefully, because they are the ones this project is
buying. **On the hyperparameter benchmark specifically, the paper's own text
says the final performance of SARA, LLAMBO and Ax "is very similar".** SARA's
separation on LCBench is in the *early* evaluations. The wide gaps appear on the
chemistry tasks, where the distance between a merely decent operating region and
the optimum is much larger.

That is not a reason to skip it, and it is close to the reason to use it. One
CS-Trees model takes 13 minutes, so a 40-trial budget is one overnight run, and
"reaches a good region in fewer evaluations" is precisely the property worth
paying for under that budget. But the honest expectation is **a faster route to
the same answer, not a better answer.** Any claim beyond that needs its own
measurement.

Two further findings constrain how we run it:

- **Model capability matters, for tool use specifically.** Haiku 4.5
  underperformed on the benchmark that leaned hardest on driving the backend;
  the gap was much smaller on the hyperparameter and yield tasks. Use a frontier
  model. This echoes BioArc's finding that Qwen3-4B scored 0.00 as an
  architecture predictor (§2.5).
- **Reasoning effort barely mattered**, with one exception where `off` produced a
  better warm start than `high`. Do not assume more reasoning is better.

### 10.4 The line between "hyperparameter" and "architecture"

This has to be fixed in advance, or the two searches overlap and neither result
means anything. Hidden dimension and depth are **architecture** in BioArc's
formulation (§2.1), so they belong to the NAS and not to SARA.

| Knob | Owned by | Why |
|---|---|---|
| block type per layer | NAS | this is the search space (§3) |
| depth, hidden width | NAS | BioArc's `d` and `h` axes |
| graph construction rule | NAS | BioArc's tokenization analogue (§3) |
| learning rate, weight decay | **SARA** | training, not design |
| batch size, epochs, schedule | **SARA** | training, not design |
| normalisation choices | **SARA** | preprocessing, not design |

One consequence is worth stating plainly, because it is easy to get wrong and
expensive to discover late: **an architecture comparison is only fair if every
candidate receives the same tuning treatment.** Tuning one architecture with
SARA while the others run on defaults measures tuning effort, not architecture.
Either tune every candidate identically, or tune once on the reproduced baseline
and then hold the training knobs fixed across the whole search. The second costs
far less and is the default until there is a specific reason to change it.

### 10.5 The hazard that is specific to this project

§8 documents shortcuts that produce excellent scores while using no structure at
all: counting galaxies reaches r = 0.758 for Ω\_m in CAMELS, and mean speed
reaches 0.76. §8 already warns that a NAS optimising a score will find those
loopholes faster than a human reviewer will, and will report the result as a
triumph.

**An agent that can rewrite its own objective is strictly worse in this
respect.** SARA is built to revise the problem specification mid-run, including
promoting and demoting constraints. Whatever we let it change, it may change in
whichever direction raises the number.

So §8's leakage screen cannot be a convention here. It has to be a property of
the code the agent calls, sitting outside anything the agent can reach: the
feature set is frozen before the agent starts, and the agent receives a score
back and nothing else. **SARA tunes training knobs. SARA does not choose
inputs.**

### 10.6 What this needs, and what it must not become

**Needs.** `botorch` and `gpytorch` are **not installed** in the `KAAI` env
(checked 2026-08-17, both raise `ModuleNotFoundError`); `ax` is absent too.
Adding them deserves the same care as everything else in
`KAAI_requirements.txt`, since the OpenMP conflict documented there is a hard
interpreter abort rather than a warning.

**Non-goals, named so they cannot creep in:**

- SARA does **not** drive the architecture search. That is deferred, not decided.
- SARA does **not** choose which input features the model sees (§10.5).
- We do **not** reimplement `lenz`. If the published implementation is not
  available, the fallback is Optuna, which is already installed, and the agentic
  layer waits until it is.
- We do **not** report a SARA-versus-Optuna comparison from single runs. SARA is
  non-deterministic by construction, and the paper reports **tool-use inertia**,
  where whichever calling pattern the agent settles into early tends to persist
  for the entire run. Ten seeds is the paper's own protocol and should be ours.

### 10.7 Where it lands in the roadmap

Nowhere before Phase 0 closes. The gate has not moved: **reproduce the CS-Trees
GNN at Ω\_m R² = 0.996 first.** Tuning the knobs of a training loop that has
never matched a published number would be optimising something unvalidated.

After that, the first use should be small and checkable: tune the reproduced
baseline's training knobs, change nothing about the architecture, and see
whether σ\_8 moves above 0.82. That is a real result in either direction.

- If it moves, part of the published headroom was tuning rather than
  architecture, which is worth knowing *before* spending nine hours searching
  architectures.
- If it does not move, 0.82 is a genuine architectural ceiling and the NAS has a
  clean, well-defined target.

Interpretation, not measurement: this also doubles as a cheap test of whether
the agentic layer is doing anything at all on our data, on a problem where we
already know the right answer.

## 11. The execution plan

**Supersedes the phase structure in §7**, as of 2026-08-17. Five stages, each
producing the input the next one needs.

### 11.0 The evidence status of everything below

This subsection exists because of a specific, named failure mode.

**The failure mode.** Three of the ideas driving this plan are *untested
hypotheses about our data*, not results. They are written up persuasively in
`findings.md`, and persuasive writing degrades predictably: after a few
retellings, "oversquashing is a candidate explanation for the sigma_8 gap"
becomes "we found that oversquashing causes the sigma_8 gap". Nobody decides to
do this. It happens by attrition, in handoffs and summaries, and by the time it
is in a draft the hedge is unrecoverable because nobody remembers it was there.

**The guard.** Every claim this plan rests on carries a status. Re-check this
table before writing any of it into a paper, a slide, or a handoff.

| Claim | Status |
|---|---|
| Concentration predicts Ω_m, time predicts σ_8, mass is nearly useless | **MEASURED BY US**, `explore.py` §5 on val |
| Splits are clean, no simulation straddles them | **MEASURED BY US**, `explore.py` §1 |
| The mass cliff at 10^10.477 is pruning, not physics | **MEASURED BY US**, `explore.py` §4 |
| Counting galaxies scores 0.758 for Ω_m in CAMELS | **MEASURED BY US**, `explore.py` §5 |
| CosmoBench baseline numbers (0.996, 0.82, 0.758, 0.85) | **PUBLISHED**, not reproduced by us. This is what Stage 1 exists to check |
| BioArc ranking correlations (0.32 and 0.73) | **PUBLISHED**, taken from the paper, not re-verified |
| SARA benchmark numbers | **PUBLISHED**, taken from the paper, not re-verified |
| Topology alone is sufficient to read the answer (2511.05367) | **UNTESTED HYPOTHESIS.** Abstract-level claim by its authors, on a different target, suite and tree construction. Stage 2.1 tests it |
| Oversquashing explains the σ_8 gap | **UNTESTED HYPOTHESIS.** Nobody has measured this, here or elsewhere. Stage 2.3 tests it, and see the caution there |
| NAS will find and exploit the leak | **UNTESTED HYPOTHESIS.** The leak is measured; that a search process amplifies it is not. Stage 4 tests it, and it is the spine of the paper |

The last three are the whole intellectual case for this project, and all three
are currently guesses. That is a normal position to be in before running
experiments. It stops being normal the moment one of them is written down
without its status.

### 11.1 Stage 0. Before any code runs

Nothing here trains anything.

**0.1 Read the reference implementation.** `nhuang37/cosmology_benchmark`, the
merger tree model. Two questions, not one:

- What layer structure gives 2.8k parameters? Deriving from the paper's own
  equations gives about 2,290, so roughly 500 parameters are unaccounted for and
  something is being read wrong. DeepSets by contrast derives exactly (610 for
  one input feature, 658 for four, against a reported 0.61k and 0.65k), so that
  model is unambiguous and the graph network is not.
- **Does the model pool globally?** One line of code, and it decides whether
  diagnostic 2.3 means anything at all.

**0.2 Check whether the trees leak.** Does `n_nodes` correlate with Ω_m in
CS-Trees the way `n_galaxies` does at 0.758 in CAMELS? `summarise()` already
computes `n_nodes` and `explore.py` §5 already correlates every column against
both targets, so this number may already exist in that output and simply was not
carried into `merger_trees/notes.md`. Tree sizes span 121 to 37,865, which is
ample room for a leak to hide in.

This **routes** Stage 4 rather than gating it, see 11.5.4.

**0.3 Write the spec.** Eight sections. The acceptance threshold for the gate is
written before anything runs. The benchmark reports 0.996 plus or minus 0.001,
so the tolerance is already chosen.

### 11.2 Stage 1. The gate

Reproduce the CS-Trees graph network at Ω_m R² = 0.996. Nothing else counts
until it passes. If it fails, stop and debug. Do not move on.

**1.1 Build the four missing harness pieces.** The repo currently has a
dataloader and nothing else:

- an R² function matching the benchmark's equation 1,
- the test split, since `get_loaders` builds only train and val while every
  published number is a test number,
- seeding across `random`, `numpy` and `torch`, plus an explicit seeded
  generator on the shuffling dataloader,
- bootstrap error bars on the test set, because a bare number cannot be compared
  against one that carries a spread.

**1.2 Train DeepSets first, not the graph network.** 658 parameters, a count
verified against the paper twice, and it discards every edge so there is far
less to get wrong. Target 0.993.

This does two jobs. If DeepSets lands, then data loading, batching,
normalisation and scoring are all proven, so a later graph network failure has
exactly one possible cause instead of five. **It is also diagnostic 2.2,
obtained for free.**

**1.3 Time one epoch and write it down.** The benchmark's 13 minutes is one-GPU
time on a research institute machine, not MPS on a Mac. Every budget on this
page inherits from that number. If it is really 40 minutes, the ranking harness
goes from roughly 6 hours to roughly 20 and this plan needs resizing. Find out
in the first hour, not the first week.

**1.4 Then the graph network.** Target 0.996 and 0.82.

### 11.3 Stage 2. Three diagnostics

About one hour of compute, if 1.3 confirms the timing. **Write the expected
result for each before running it.** The predictions below exist to be
disagreed with.

**2.1 Zero the four node features.** Does topology alone work?

> Prediction: Ω_m somewhere around 0.2 to 0.5. Clearly above zero, far below
> 0.9. Reasoning: given only concentration the graph network reaches 0.84; given
> nothing at all it has branching structure and no more. `Prediction. The
> direction is defensible, the range is a guess.`

**2.2 Drop all edges.** Our own DeepSets floor instead of the published 0.993.
**Produced by step 1.2.**

**2.3 Add one virtual global node.** Does σ_8 move?

> Prediction: it moves by less than 0.01. Reasoning: if the model already pools
> globally, every node already reaches the readout in one step, so a virtual
> node adds far less than "two hops instead of seven hundred" suggests. The
> strongest evidence is that DeepSets does no message passing at all and still
> reaches 0.80 against the graph network's 0.82, so a model with no whispering
> cannot be suffering from garbled whispers. Answer 0.1 first. If σ_8 jumps,
> this reasoning is wrong, and that is worth more than the null result.

Context for all three, computed from CosmoBench Table 5. Topology's measured
contribution is the gap between the graph network and shape-blind DeepSets:

| Node features given | Topology worth, Ω_m | Topology worth, σ_8 |
|---|---|---|
| concentration only | +0.16 | +0.14 |
| scale factor only | +0.10 | +0.05 |
| all four | +0.003 | +0.02 |

INTERPRETED, from published numbers: topology carries real information that is
largely a duplicate of what the node features already say. Starve the model of
features and it leans on shape. This is the prior these diagnostics test, and it
also dissolves most of the apparent conflict with 2511.05367 flagged in
`findings.md` §3, since that paper removed the features entirely.

These decide the search space. Run them before designing anything.

### 11.4 Stage 3. Build once, use three times

**3.1 The small split, cut on the correct axis.** Keep all 600 simulations and
use 5 trees each instead of 25. Gives 3,000 trees and a 5x speedup.
**Never subsample simulations.** 600 distinct labels is the scarce resource, not
14,997 trees.

Guard: assert the label count is still 600 after building it. Then deliberately
break it once and confirm the assertion fires. A guard that has never been
tripped is a guard we are hoping works.

**3.2 The ranking harness.** Spearman rank correlation between two orderings of
architectures. Three uses: small scale against full scale, supernet against
independently trained, any cheap proxy against expensive truth.

Method: about eight deliberately different architectures, trained at both
scales, reporting ρ with a spread across seeds. Depends on 3.1, which supplies
the small scale. Rough cost at eight architectures, two scales and three seeds
is about 6 hours if the 13 minutes holds.

### 11.5 Stage 4. The paper experiment

Two search spaces, `S_open` and `S_screened`, on identical budgets. Evaluate
both winners on the leaky suite and on the count-fixed control. Headline metric
is the leak-attributable fraction, not raw R², because R² is what the cheater
maximises.

Write the spec first, with a stated falsification criterion, or the experiment
cannot succeed either.

**4.1 Define the leak-attributable fraction before anything else in this
stage.** It is the paper's central number and it currently has no definition.
Until it is written down as arithmetic over named quantities, there is no way to
say whether the experiment worked.

**4.2 Build the matched control. Do not use CAMELS-SAM as the control for
CAMELS.** The count is fixed in CAMELS-SAM and Quijote, so the leak is closed
there, but they are not *matched* to CAMELS: box size (100 and 1000 against 25
cMpc/h), physics (N-body and semi-analytic against full hydrodynamics) and mass
resolution all differ too. Four things change at once, so a gap between them
cannot be attributed to the leak.

Fix: subsample every CAMELS cloud to a fixed count, the suite minimum of 588
galaxies, keeping the most massive. Same simulations, same physics, same box,
leak open in one version and closed in the other, so the difference is the leak
by construction rather than by assumption. Keep CAMELS-SAM in as independent
corroboration. This also turns "the benchmark ships a control" into "we built
the control", which is the stronger sentence.

**4.3 The screen must be code, not convention.** `S_screened` has to be a
mechanical, testable property sitting outside anything the search can reach
(§10.5). Then trip it: hand it a feature known to leak and confirm rejection.

**4.4 Scale is a deliberate choice, not a compromise.** Run this small. The
claim is about a mechanism, and a mechanism shows at any scale, whereas only a
beat-the-baseline claim needs full scale. This is what makes the leak experiment
affordable on CAMELS point clouds despite the compute table in §4, and it should
appear in the paper as a design decision rather than an apology. It is also why
0.2 routes rather than gates: if the trees leak, the whole paper fits on one
track; if they do not, the leak experiment runs on small CAMELS while the
architecture work stays on trees.

### 11.6 Running throughout

- Start `runLog.md`. Append at the moment each thing happens, never as an
  end-of-session summary.
- Look up the ML4PS at NeurIPS deadline. It sets the schedule. Not verified here.

### 11.7 Not yet

- No full search space design. Stage 2 changes it.
- No agent layer. It needs a knowledge base that does not exist, and its novelty
  is superseded (§6).
- No Quijote download. 4.1 GB that cannot be trained on.
- No more papers. The bottleneck is running code, not reading.
- No SARA install. It would tune an unvalidated model, and `botorch` and
  `gpytorch` are absent from the env anyway (§10.6).

### 11.8 What survives from §7

Phase 0's gate, unchanged, and still the single highest-value hour. Phase 1's
search space design, deferred until Stage 2 reports. Phase 2's ranking study,
now Stage 3.2 and demoted from headline to supporting section. Phases 3 and 4
are on hold with their novelty superseded.

### 11.9 If you only do one thing

Reproduce 0.996.

## 12. Still open

- Quijote positions are not downloaded (4.1 GB). It is the flagship suite and
  the one where the 49-parameter linear model beats the 671k GNN. Only its
  precomputed clustering curves are on disk.
- Combining the two views (§ multiview): CS-Trees and CAMELS-SAM clouds are the
  *same 1,000 simulations* with identical splits — verified. But trees carry no
  positions and clouds carry no halo IDs, so only simulation-level fusion is
  possible, not object-level. Deferred until the tree search is working.
