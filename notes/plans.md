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

This is the maximal scope on every axis. It only works if strictly sequenced,
because each phase produces the input the next one needs.

## 7. Roadmap

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

## 10. Still open

- Quijote positions are not downloaded (4.1 GB). It is the flagship suite and
  the one where the 49-parameter linear model beats the 671k GNN. Only its
  precomputed clustering curves are on disk.
- Combining the two views (§ multiview): CS-Trees and CAMELS-SAM clouds are the
  *same 1,000 simulations* with identical splits — verified. But trees carry no
  positions and clouds carry no halo IDs, so only simulation-level fusion is
  possible, not object-level. Deferred until the tree search is working.
