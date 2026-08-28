# Literature

Papers that bear on the plan in [plans.md](plans.md). Every entry says what it
changes for us, not just what it is about.

Confidence tags: `FULL` means the paper was read end to end. `REPORT` means a
structured summary was read. `ABSTRACT` means only the abstract was read and the
claims below are the authors' own, unverified.

Source: alphaXiv semantic search over arXiv, 2026-08-17. Six searches, 62 unique
papers surfaced, 41 of them cited in the sections below. arXiv only, so non-arXiv
venues are not covered. The full 62, including the ones judged off-target, are
listed in the appendix at the end of this file.

**Depth of engagement, stated plainly.** Exactly one paper was read properly
(2607.07984, as a structured report). One other had its full text searched for
specific claims (2511.05367). Everything else in this file is a title plus a
truncated abstract from the search index. These are leads, not knowledge.

---

## 1. The thing that changes the plan

### Agentic Neural Architecture Search (AgentNAS)

arXiv 2607.07984, July 2026. `REPORT`

An LLM designs a seed architecture for a task, then decomposes that seed into a
"slotted architecture", a scaffold with named interchangeable module slots. That
slotted scaffold **is** the search space. Conventional NAS (regularized
evolution) then searches inside it. Reported: best result on 11 of 17 tasks
across NAS-Bench-360 and Unseen NAS.

**Why this matters to us.** Phases 3 and 4 of `plans.md` describe an LLM agent
that proposes architectures and learns from past searches. AgentNAS is a
published version of roughly that idea, one month old, in the general-purpose
setting. "We put an LLM in the NAS loop" is no longer a contribution on its own.

**Two openings it leaves.**

- MEASURED: the paper's own future-work section names "exploring other efficient
  NAS algorithms, particularly weight-sharing methods, within the slotted
  framework" as unaddressed. That is exactly our Phase 2.
- MEASURED: their search space is code-level slots in a sequential scaffold. They
  never search over the input representation. Our graph-construction axis
  (`plans.md` §3) has no counterpart in their work.
- INTERPRETED: our defensible framing shifts from "LLM plus NAS" to "NAS on
  geometric scientific data where the trivial baseline currently wins, and where
  the score can be gamed by a leak." That is a different paper.

Related and worth a skim: **GraphIR** (2608.01633, `ABSTRACT`) argues code-level
LLM flexibility lacks the architecture-level state needed for good evolution.
**AgenticSciML** (2511.07262, `ABSTRACT`) is multi-agent design of scientific ML
architectures and losses, from Brown.

---

## 2. Closest prior work on our exact task

### Linking Warm Dark Matter to Merger Tree Histories via Deep Learning

arXiv 2511.05367, November 2025. `ABSTRACT` plus targeted text search.

Trains a GNN (adapted from CosmoGraphNet) on SubLink merger trees to predict the
warm dark matter particle mass.

- MEASURED (their claim, from the abstract): the GNN infers the WDM mass from
  merger tree histories **with no node features at all**, using only the tree
  structure.
- INTERPRETED: this cuts against what we wrote in
  [../merger_trees/notes.md](../merger_trees/notes.md), that architectures
  leaning on node features should beat ones leaning on topology, which we
  inferred from DeepSets scoring 0.993 against the GNN's 0.996 on Omega_m.
- OPEN: the two are not in direct conflict. Different target (WDM mass, not
  Omega_m / sigma_8), different simulation suite, different tree construction.
  But it is a cheap and high-value experiment for us: **train a topology-only
  model on CS-Trees with node features zeroed.** If topology alone carries real
  signal for sigma_8, the whole search space weighting changes.

### Mangrove: Learning Galaxy Properties from Merger Trees

arXiv 2210.13473, October 2022. Carnegie Mellon, Princeton, Flatiron. `ABSTRACT`

The GNN-on-merger-trees design precedent, cited by the WDM paper above. Read it
for the concrete engineering: how they featurize nodes, how they handle trees of
wildly varying size, what pooling they use. Our size range is 121 to 37,865
nodes, and their handling of that is directly reusable.

### Supporting tree work

- **FLORAH-Tree** (2507.10652, `ABSTRACT`), graph generative models that emulate
  merger trees. Relevant as a source of augmentation or synthetic pretraining
  data if the 600-distinct-labels problem in `plans.md` §9 bites.
- **Graph Path Likelihood for Galaxy Formation on Layered Halo Graphs**
  (2603.15128, `ABSTRACT`), trajectory-level likelihoods on halo graphs.
- **A graph-based NN surrogate for semi-analytic galaxy formation** (2604.23147,
  `ABSTRACT`), CAMELS-SAM adjacent.

---

## 3. Phase 2: does weight sharing rank architectures faithfully

This is the block that supports our stated Phase 2 deliverable, a
ranking-correlation number between the supernet and independently trained
ground truth. All `ABSTRACT`.

| Paper | arXiv | What it gives us |
|---|---|---|
| Single Path One-Shot NAS with Uniform Sampling | 1904.00420 | The method BioArc uses. Read the original, not the summary of it. |
| Weight-Sharing NAS: A Battle to Shrink the Optimization Gap | 2008.01475 | Survey framing of exactly the failure mode we plan to measure. Best single entry point. |
| Can weight sharing outperform random architecture search? (TuNAS) | 2008.06120 | The honest negative-result paper. This is the comparison our Phase 2 must beat, and the reason random search belongs in our baselines. |
| Prior-Guided One-shot NAS | 2206.13329 | Directly on poor ranking consistency between supernet and standalone training. |
| RD-NAS: ranking distillation from zero-cost proxies | 2301.09850 | A concrete fix if our supernet ranking turns out to be bad. |
| Few-shot Graph NAS via Partitioning Gradient Contribution | 2506.01231 | Weight coupling specifically in **graph** NAS. Nearest existing work to our supernet. |

**Cheap-evaluation escape hatch**, if 40 architectures at 13 minutes each is
still too slow. All `ABSTRACT`.

- **Zero-Cost Proxies for Lightweight NAS** (2101.08134). The standard reference.
- **AZ-NAS** (2403.19232), assembling multiple proxies.
- **TG-NAS** (2404.00271), operator-description embeddings plus graph learning,
  so a proxy that generalizes across search spaces.
- **Variation Matters** (2502.19657), on the variance of zero-shot ranking
  functions. Read this before trusting any single proxy number.

---

## 4. NAS outside image classification

- **NAS-Bench-360** (2110.05668, `ABSTRACT`). Benchmarking NAS on diverse tasks.
  Take the evaluation protocol from here. It is the standard AgentNAS is
  measured against, so matching its reporting style makes us comparable.
- **Neural Architecture Codesign for Fast Physics Applications** (2501.05515,
  `ABSTRACT`). NAS for physics, from UCSD, Northwestern, Fermilab. The nearest
  existing "NAS in a physical science" precedent.
- **GraphPNAS** (2211.15155, `ABSTRACT`), architectures as computational graphs,
  learned with a graph generative model.
- **Global optimization of graph acquisition functions for NAS** (2505.23640,
  `ABSTRACT`), Bayesian optimization over architecture graphs. Compare against
  Optuna's TPE, which is what we currently plan to use.

---

## 5. The methodological mirror of CosmoBench's headline

CosmoBench's finding is that a 49-parameter least-squares fit beats a 671,000
parameter GNN on Quijote. That is an instance of a pattern with its own
literature, and citing it makes our result look like a contribution to a
conversation rather than a surprise.

- **Classic GNNs are Strong Baselines** (2406.08993, `ABSTRACT`). Graph
  transformers do not beat properly tuned message-passing GNNs on node
  classification. The same shape of claim, in mainstream ML.
- **Graph Contrastive Learning versus Untrained Baselines: The Role of Dataset
  Size** (2509.01541, `ABSTRACT`). Directly relevant to `plans.md` §9. Our
  contrastive-pretraining arm needs to clear an untrained baseline, and this
  paper says the crossover depends on dataset size, which we have little of in
  the effective-label sense.
- **Loss Landscape Topology Reveals Why Simple Baselines Are Competitive at 3D
  Point Cloud Segmentation** (2607.21089, `ABSTRACT`). A mechanistic account of
  why the simple thing wins on point clouds.

---

## 6. Astrophysics context we actually need

### How much data is enough, and does the improvement mean anything

- **How many simulations do we need for SBI in cosmology?** (2503.13755,
  `ABSTRACT`). Johns Hopkins, Princeton, Flatiron, IAP. 29 votes on alphaXiv, so
  well read. This speaks directly to our 600-distinct-labels concern and should
  inform whether the supervised supernet is expected to overfit before we run it.
- **Robust marginalization of baryonic effects for cosmological inference at the
  field level** (2109.10360, `ABSTRACT`). The formal treatment of what
  `plans.md` §8 calls the irreducible-noise cap on achievable R squared.
- **Mitigating Simulator Dependence in AI Parameter Inference** (2601.05229,
  `ABSTRACT`). Epoch of Reionization rather than our task, but the lesson
  transfers: models trained on one simulator do not transfer, and simulation
  diversity is the fix. Relevant if we ever claim generality across CAMELS and
  CAMELS-SAM.
- **Interpreting Cosmological Information from Neural Networks in the
  Hydrodynamic Universe** (2504.17839, `ABSTRACT`). What CNNs actually latch
  onto when reading cosmology off a hydro simulation. Read alongside our own
  leakage screen.

### Learned summary statistics, the thing our search is implicitly competing with

- **Learning Optimal and Interpretable Summary Statistics of Galaxy Catalogs
  with SBI** (2411.08957, `ABSTRACT`).
- **How to evaluate the sufficiency and complementarity of summary statistics for
  cosmic fields** (2511.08716, `ABSTRACT`). Information-theoretic framing. Gives
  a principled way to say whether our architecture found anything the 2PCF did
  not, which is a better claim than a raw R squared delta.
- **TopoFisher** (2605.07720, `ABSTRACT`), topological summaries by maximizing
  Fisher information. A block worth putting in the search palette.
- **Interpretable Neural Marked Statistics for Cosmological Inference**
  (2606.11295, `ABSTRACT`).

### Arm B, the features-allowed arm

- **Cosmology with one galaxy: an analytic formula relating Omega_m with galaxy
  properties** (2602.07651, `ABSTRACT`). The 2026 follow-up to the
  Villaescusa-Navarro result our plan cites. It reduces the relation to a closed
  form, which means Arm B now has an analytic baseline to beat, not just a
  neural one. This raises the bar for Arm B and should be read before starting it.

---

## 7. Architecture ingredients for the search palette

- **PointNet** (1612.00593, `ABSTRACT`). The origin of the permutation-invariant
  point pathway.
- **Foundations of Equivariant Deep Learning: Unifying Graph and Sheaf Neural
  Networks** (2607.03798, `ABSTRACT`). Recent unifying treatment. Use for the
  E(3)-equivariant block definitions.
- **Point Group Equivariant GNNs for Materials** (2607.16871, `ABSTRACT`).
  Interesting for a specific reason: it deliberately respects a **subgroup**
  rather than full E(3). Our periodic cubic box has exactly that structure. The
  cube's symmetry group, not the full rotation group, is the honest symmetry of
  our data.
- Oversquashing on deep thin graphs. Our trees are long and narrow, median 769
  nodes and up to 37,865, so information from the early universe has to squeeze
  through a narrow path to reach the root. This is textbook oversquashing and it
  may be **the** reason sigma_8 is stuck at 0.82.
  - Complexity of optimal graph rewiring (2603.26140, `ABSTRACT`)
  - Local-global insights via entropic curvature (2607.22381, `ABSTRACT`)
  - Hierarchical multi-scale GNNs (2605.10975, `ABSTRACT`)

  INTERPRETED, not measured: if oversquashing is the bottleneck, then graph
  rewiring belongs in the search space next to graph construction, and a virtual
  global node is the cheapest thing to try first.

---

## 8. Where the novelty actually is

Honest reading of the above. All INTERPRETED.

Not novel any more:
- LLM proposes architectures, NAS searches. AgentNAS, July 2026.
- Weight-sharing supernet on a new domain. Method transfer alone.
- GNN reads cosmology off merger trees. Mangrove, the WDM paper, CosmoBench.

Plausibly novel, in rough order of strength:

1. **NAS finds the leak.** `plans.md` §8 measured that counting galaxies alone
   scores 0.758 for Omega_m in CAMELS, against the published GNN's 0.78. A search
   process maximizing R squared will find that shortcut faster than a human
   reviewer will, and will report it as a win. Demonstrating that on a real
   benchmark, and then showing a leakage-screened search space that closes it, is
   a result about the safety of automated model design, not about cosmology. That
   travels further than a delta on sigma_8. No paper found addresses it.
2. **Joint search over graph construction and architecture.** No paper in this
   sweep searches the input representation and the network together for geometric
   scientific data. If graph construction dominates architecture choice, that is
   a clean, quotable, and slightly embarrassing result for the field.
3. **Ranking fidelity where the trivial baseline wins.** Every weight-sharing
   ranking study above is on image classification, where deep models win by a
   wide margin. CosmoBench is a regime where a 49-parameter linear fit beats a
   671k GNN. Whether supernet ranking survives a search space containing a linear
   pathway is genuinely unknown, and AgentNAS names weight-sharing as its own
   open problem.

Venue reality check, INTERPRETED: item 1 or 2 with a solid experimental section
is plausible for a NeurIPS or ICML main track. Method transfer alone is a
workshop paper. ML4PS at NeurIPS and the ML4Astro workshops are the natural
stepping stones, and CosmoBench itself is NeurIPS 2025, so the community is
already there.

---

## 9. Not yet checked

- BioArc (arXiv 2512.00283) was not re-verified in this sweep. Claims about it
  here are taken from `plans.md`.
- arXiv only. Non-arXiv astronomy venues (MNRAS, ApJ direct) were not searched.
  NASA ADS would cover that and needs an `ADS_TOKEN`.
- No paper in this sweep was read in full except the AgentNAS structured report.
  Every `ABSTRACT` claim above is the authors' own and unverified.

---

## Appendix: everything the sweep surfaced

All 62 unique results across six searches. A tick means the paper is cited
somewhere in the sections above. A blank means it was surfaced, read at title and
abstract level, and judged off-target for this project. Blanks are kept on the
record so a later search does not rediscover them and assume they are new.

### A. Weight-sharing NAS and ranking fidelity

| arXiv | Title | Used |
|---|---|---|
| 1904.00420 | Single Path One-Shot NAS with Uniform Sampling | yes |
| 2008.01475 | Weight-Sharing NAS: A Battle to Shrink the Optimization Gap | yes |
| 2008.06120 | Can weight sharing outperform random architecture search? (TuNAS) | yes |
| 2206.13329 | Prior-Guided One-shot Neural Architecture Search | yes |
| 2301.09850 | RD-NAS: ranking distillation from zero-cost proxies | yes |
| 2506.01231 | Efficient Few-shot Graph NAS via Partitioning Gradient Contribution | yes |

### B. Zero-cost proxies

| arXiv | Title | Used |
|---|---|---|
| 2101.08134 | Zero-Cost Proxies for Lightweight NAS | yes |
| 2403.19232 | AZ-NAS: Assembling Zero-Cost Proxies | yes |
| 2404.00271 | TG-NAS: Generalizable Zero-Cost Proxies | yes |
| 2502.19657 | Variation Matters: Zero-Shot NAS Ranking Function Variation | yes |

### C. LLM-driven and agentic NAS

| arXiv | Title | Used |
|---|---|---|
| 2607.07984 | **Agentic Neural Architecture Search** (the one read properly) | yes |
| 2608.01633 | GraphIR: Architecture-Level Search States for LLM-Guided Evolution | yes |
| 2511.07262 | AgenticSciML: Multi-Agent Systems for Scientific ML | yes |
| 2606.10294 | LLM-Guided NAS for Robust Co-Design of Physical Neural Networks | |
| 2602.15039 | GRACE: Agentic AI for Particle Physics Experiment Design | |

### D. NAS beyond image classification

| arXiv | Title | Used |
|---|---|---|
| 2110.05668 | NAS-Bench-360: Benchmarking NAS on Diverse Tasks | yes |
| 2501.05515 | Neural Architecture Codesign for Fast Physics Applications | yes |
| 2211.15155 | GraphPNAS: Learning Distribution of Good Architectures | yes |
| 2505.23640 | Global optimization of graph acquisition functions for NAS | yes |
| 2602.17700 | MIDAS: Mosaic Input-Specific Differentiable Architecture Search | |
| 2510.05888 | BioAutoML-NAS: multimodal insect classification | |

### E. Merger trees

| arXiv | Title | Used |
|---|---|---|
| 2511.05367 | **Linking Warm Dark Matter to Merger Tree Histories** (text searched) | yes |
| 2210.13473 | Mangrove: Learning Galaxy Properties from Merger Trees | yes |
| 2507.10652 | FLORAH-Tree: Emulating Merger Trees with Graph Generative Models | yes |
| 2603.15128 | Graph Path Likelihood for Galaxy Formation on Layered Halo Graphs | yes |
| 2604.23147 | Graph-based NN surrogate for semi-analytic galaxy formation | yes |

### F. Cosmological inference, generalization, robustness

| arXiv | Title | Used |
|---|---|---|
| 2507.03707 | CosmoBench (our own benchmark) | yes |
| 2503.13755 | How many simulations do we need for SBI in cosmology? | yes |
| 2504.17839 | Interpreting Cosmological Information from NNs in the Hydrodynamic Universe | yes |
| 2109.10360 | Robust marginalization of baryonic effects at the field level | yes |
| 2601.05229 | Mitigating Simulator Dependence in AI Parameter Inference (EoR) | yes |
| 2602.07651 | Cosmology with one galaxy: an analytic formula for Omega_m | yes |
| 2606.10038 | CAMELS 2nd generation, 35 varied parameters | |
| 2601.06258 | Cosmological back-reaction of baryons on dark matter in CAMELS | |
| 2402.10997 | Cosmological multifield emulator | |
| 2510.19224 | BaryonBridge: stochastic interpolant for fast hydro simulations | |
| 2508.05744 | Detecting Model Misspecification with Scale-Dependent Normalizing Flows | |
| 2606.11309 | DES Y3 wCDM simulation-based inference, weak lensing | |
| 2606.12938 | Cluster Mass Inference from Galaxy Kinematics | |
| 2606.27439 | SBI for Cluster Cosmology with Set-Based Architectures | |
| 2605.21483 | Velocityformer: equivariant graph transformers for velocity reconstruction | |
| 2603.20855 | Field-Level Inference of Primordial Non-Gaussianity with Quijote | |

### G. Learned and interpretable summary statistics

| arXiv | Title | Used |
|---|---|---|
| 2411.08957 | Learning Optimal and Interpretable Summary Statistics with SBI | yes |
| 2511.08716 | Sufficiency and complementarity of summary statistics for cosmic fields | yes |
| 2605.07720 | TopoFisher: topological summaries by maximizing Fisher information | yes |
| 2606.11295 | Interpretable Neural Marked Statistics for Cosmological Inference | yes |
| 2512.09852 | Primordial non-Gaussianity, fast simulations and persistent statistics | |
| 2602.21307 | SymTorch: Symbolic Distillation of Neural Networks | |
| 2602.24022 | Symbolic regression for star/galaxy/quasar separation | |

### H. Architecture ingredients and GNN theory

| arXiv | Title | Used |
|---|---|---|
| 1612.00593 | PointNet | yes |
| 2607.03798 | Foundations of Equivariant Deep Learning: graph and sheaf networks | yes |
| 2607.16871 | Point Group Equivariant GNNs for Materials | yes |
| 2603.26140 | Complexity of Optimal Graph Rewiring for Oversmoothing and Oversquashing | yes |
| 2607.22381 | Local-Global Geometric Insights via Entropic Curvature | yes |
| 2605.10975 | Hierarchical Multi-Scale Graph Neural Networks | yes |
| 2406.08993 | Classic GNNs are Strong Baselines | yes |
| 2509.01541 | Graph Contrastive Learning versus Untrained Baselines | yes |
| 2607.21089 | Loss Landscape Topology: why simple baselines win at point clouds | yes |
| 2510.17457 | Deeper with Riemannian Geometry for Graph Foundation Models | |
| 2404.07194 | VN-EGNN: E(3)-equivariant GNNs with virtual nodes | see below |

### I. Surfaced, off-target

| arXiv | Title | Used |
|---|---|---|
| 2603.15736 | Halo assembly bias in the early Universe, Little Red Dots | |
| 2511.23452 | From metallicity distributions to mutual information, stellar halos | |

### One blank worth revisiting

**VN-EGNN**, arXiv 2404.07194. Filed under protein binding site prediction and
skipped on application grounds. The mechanism, however, is virtual nodes added to
an E(3)-equivariant graph network, which is exactly the oversquashing fix
proposed in §7 above. The science is irrelevant to us. The implementation detail
may be directly reusable. INTERPRETED: worth opening before writing our own
virtual-node code.

### Not from this sweep

**BioArc**, arXiv 2512.00283. Sits in `resources/` and is cited throughout
[plans.md](plans.md). It was never opened or re-verified during this sweep. Every
statement about BioArc in this repository traces back to `plans.md`, not to the
paper.
