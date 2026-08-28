# What to stock the zoo with, from the literature

Written 2026-08-28 after a literature sweep on what astrophysicists actually reach
for, rather than what is convenient for us to run.

The framing that produced this list: the zoo exists so that a new group does not
repeat an architecture search. So it should stock what people actually use, and it
should branch on the decisions people actually face.

## 1. The decision tree the zoo should serve

Thiele 2026 (arXiv 2605.10719), "Machine Learning Techniques for Astrophysics and
Cosmology: Simulation-Based Inference", Section 2.7, gives the field's practical
guidance in four rules. This is the closest thing to a consensus decision tree, and
our recommender should implement it rather than invent one.

- NPE is the default, for simplicity, speed, and high dimensional data vectors.
- Avoid NPE when inferring from multiple i.i.d. measurements, because the prior is
  baked into the posterior and does not factorise. Use NLE.
- NLE is also preferable when the parameter vector is comparatively high dimensional.
- NRE is the natural choice when a flow architecture is inconvenient or too expensive.
- Compression is necessary in most realistic scenarios. For tabular data start with
  simple linear compression (MOPED, score compression). For fields and images use a
  compressor that respects the symmetry of the data.

Two lines in that review bear directly on results we measured this week.

**On our edge finding.** Section 2.7, Figure 5: "the special case of uniform prior
with sharp edges may seem like an exception, but due to the regularity properties of
neural functions it would still introduce errors". We measured exactly this. Our
sigma_8 boundary deficit is 0.30 worse than a provably correct posterior gives, and
the published explanation is that a smooth neural function cannot represent a hard
prior wall.

**On our nreMlp finding.** Section 3.2: "There can be cases in which the posteriors
are mis-calibrated in such a way that the mis-calibration averages out when the entire
distribution is considered. Such problems can be diagnosed with local coverage tests,
such as local C2ST". We found precisely that: nreMlp passes on aggregate coverage at
-1.1 sigma and still fails badly near the sigma_8 boundary. We arrived at the need for
local coverage independently. The named tool for it is local C2ST (Linhart et al.,
arXiv 2306.03580), which we should adopt rather than reinvent.

**On our headline.** Section 5, "Current frontier": "with limited simulation budgets
the inferred posteriors can easily be quite wrong", and the review calls training with
limited simulation budgets "the critical problem for applications to cosmology and
astrophysics". Our 800-simulation result sits exactly in the regime the field names as
its hardest open problem.

## 2. What is actually used, by modality

Collected from the applications the review lists plus the discovery sweep. This is
what a zoo has to cover to be useful to a real astrophysicist.

| modality | what people use | representative work |
|---|---|---|
| Summary vectors (power spectra, correlation functions) | MLP or no embedding, into a flow | most SBI papers; our current zoo sits here |
| Density fields, weak lensing maps | CNN embedding, increasingly vision transformers | SimBIG field-level (Lemos et al. 2024), DES Y3 (arXiv 2511.04681), ViT for lensing (arXiv 2512.07125), HEALFormer (arXiv 2603.25471) |
| Galaxy and halo point clouds, catalogues | permutation invariant set networks, GNNs | Wang and Thiele set-based cluster mass (arXiv 2507.20378), cluster cosmology with set networks (arXiv 2606.27439), Quijote point clouds (arXiv 2405.13119) |
| Galaxy cluster X-ray images | CNN | Ho et al. 2023, LtU-ILI Section 5.1 |
| Sets of galaxy properties | Fishnets, precision weighted aggregation | Makinen et al. (arXiv 2310.03812), LtU-ILI Section 5.6 |
| Gravitational wave strain | 1D CNN, transformers, flow matching | DINGO (Dax et al. 2021), LtU-ILI Section 5.4, PTA transformers (arXiv 2607.03904) |
| Stellar spectra, light curves | 1D CNN, transformers | nbi (Zhang et al. 2023), IceCube transformer flows (arXiv 2604.19846) |
| Photometry, SEDs | MLP, VAE, diffusion | SBI++ (Wang et al. 2023), probabilistic autoencoder (arXiv 2603.24668) |
| Strong lensing images | CNN, diffusion, recurrent inference machines | arXiv 2607.19459 |

The zoo currently covers exactly one row of that table, the first.

## 3. Candidates, ranked

Reachability was checked against our installed stack, not assumed.

### Tier 1. Reachable today, config-only, cheap

| candidate | why it belongs | status |
|---|---|---|
| lampe NSF, NCSF, GF, CNF, NAF, SOSPF, UNAF, NICE | A whole second backend we have never touched. LtU-ILI Section 3.4 claims lampe gives "tighter and better-calibrated posteriors" from experience, with no number. Our harness measures calibration, so we can be the first to check it. | ltu-ili accepts maf, mdn, nsf, ncsf, nice, naf, unaf, sospf, cnf, gf through the lampe backend. NPE only. |
| `made` density estimator | Completes the sbi set. We have 3 of the 4. | reachable |
| `linear` ratio estimator | The simplest NRE, a useful floor. Our best calibrated entry is already the simplest ratio estimator, so the floor matters. | reachable |
| Mixed architecture ensemble (MAF + NSF + MDN) | The review names ensembling as a primary fix for limited budget overconfidence. We measured that an ensemble of four clones closes under a fifth of the gap. Mixed ensembles are the standard next step and we have not tried one. | reachable, needs a small change to allow heterogeneous nets |
| Ensembles at sizes 2 and 8 | We tested exactly one size. Whether the tiny gain scales is unknown. | reachable |

### Tier 2. Embedding networks, an axis currently at zero

Every entry in the zoo reads the raw 25 number vector with no embedding. The review
says compression is necessary in most realistic scenarios, and this is the axis that
decides whether the zoo generalises past summary vectors.

| candidate | why | status |
|---|---|---|
| FCN embedding | ltu-ili ships one at `ili/embedding/fcn.py`. Zero work. | reachable |
| CNN embedding | The standard for fields and maps, and what every weak lensing SBI analysis uses. Needs a field-valued task to be meaningful. | needs a new task |
| DeepSets / set embedding | The standard for catalogues, and the subject of two 2026 papers. | needs wiring |
| GNN embedding | We already have `point_clouds/gnn.py` written and screened. | needs wiring through lampe |
| Fishnets | We already have `point_clouds/blocks/fishnets.py`, and LtU-ILI Section 5.6 uses it. | needs wiring |

### Tier 3. Methods aimed at our exact failure

Our headline is that seven of eight entries are overconfident at 800 simulations. The
review names this as the field's critical open problem and lists the countermeasures.
These are the entries that would make the zoo say something useful rather than only
diagnostic.

| candidate | why | status |
|---|---|---|
| **Balanced NRE** (Delaunoy et al., arXiv 2208.13624) | Explicitly designed to produce conservative rather than overconfident posteriors. Directly targets our headline finding. | **`sbi/inference/snre/bnre.py` exists in sbi 0.22, but ltu-ili does not expose it. `engine="BNRE"` is rejected by `load_nde_sbi`. Adding it is a small, concrete pull request to the supervisor's package.** |
| Neural Quantile Estimation (Jia, arXiv 2405.xxxx) | No flow required, and argued to simplify calibration. | not in ltu-ili, would need implementing |
| Flow matching / continuous flows | The review calls continuous time flows "the standard for many applications", though it notes astrophysical posteriors are usually simple enough not to need them. | partially reachable, lampe `cnf` |
| Multi fidelity training | Train on many cheap simulations, correct with few expensive ones. The review lists this as a main direction for the limited budget problem. Directly relevant to the "would 100x coarser simulations help" question. | needs a second simulation fidelity, out of scope for now |

## 4. What I would add first, and why

Six entries, all Tier 1, all NPE family so no 3000 second MCMC cells:

1. lampe NSF
2. lampe NCSF
3. lampe GF
4. lampe CNF
5. sbi `made`
6. sbi `linear` NRE

That takes the zoo from 8 to 14, adds the second backend, and produces a measured
answer to an unquantified claim in the supervisor's own paper. At the measured NPE
rates the whole sweep is well under an hour.

Then Tier 3's Balanced NRE, because it is the only thing on this list that directly
attacks the problem the zoo has actually found.

## 5. Honest limits of this sweep

- Discovery was through the alphaXiv semantic index over arXiv. It does not cover
  non arXiv venues, and I did not run the NASA ADS index, which is astronomy native.
- I read one paper in full, the Thiele review. Everything else is from abstracts and
  the review's own reference list.
- "Commonly used" here means "appears repeatedly in the applications a 2026 review
  chose to list". I did not count citations or usage systematically.
