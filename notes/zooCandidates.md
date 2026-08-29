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

## 5. What the field actually uses, counted

The first version of this file said "commonly used" meant "appears repeatedly in the
applications one 2026 review chose to list", and admitted that was not a measurement.
This section replaces that with a count.

**Method.** Five search queries that name **no architecture at all**, so the counts are
what papers volunteer rather than what the query planted. Sources were arXiv, OpenAlex
and Crossref, which covers non arXiv venues. Semantic Scholar rate limited and was
dropped. 256 unique papers since 2021, of which 194 carry an abstract and are countable.

An earlier attempt used queries that did name architectures. Those counts are discarded:
searching for "convolutional neural network" and then reporting that convolutional
networks are common measures the query, not the field.

    architecture or embedding      papers    share      in our zoo
    normalizing flow                   14     7.2%      14 entries
    Bayesian neural network             8     4.1%      none
    ratio estimation                    6     3.1%      3 entries
    ensemble                            5     2.6%      4 entries
    CNN or convolutional                5     2.6%      none
    transformer or attention            4     2.1%      none
    flow matching or diffusion          4     2.1%      one, lampeCnf, adjacent
    set or permutation invariant        3     1.5%      none
    Gaussian process                    3     1.5%      out of ltu-ili scope
    autoencoder or VAE                  2     1.0%      out of ltu-ili scope
    mixture density network             1     0.5%      3 entries
    graph neural network                1     0.5%      none

    inference engine               papers    share      in our zoo
    NPE                                14     7.2%      17 entries
    NRE                                 5     2.6%      3 entries
    NLE                                 3     1.5%      3 entries

Shares are low in absolute terms because most abstracts never name an architecture. The
ratios are the signal, not the percentages.

**What this says about the zoo.**

- Engine balance is roughly right. The field runs about 5 NPE to 1.7 NRE to 1 NLE. We
  run 5.7 to 1 to 1, so NRE is slightly under weighted.
- Normalizing flows dominate and we cover them heavily. Correct call.
- **Mixture density networks are the most over covered thing we have**, at 3 entries
  against 0.5 per cent of the literature. That is worth keeping rather than cutting: our
  measurements put npeMdn at the same accuracy as everything else for 0.7 seconds, so
  the field may simply be under using the cheapest option.
- **Bayesian neural networks are the largest genuine gap**, second most mentioned and
  absent from the zoo. Mitigating fact: LtU-ILI Section 3.2 calls ensembling the
  practical alternative to expensive Bayesian networks, and we now carry four ensembles.
- CNN, transformer and set architectures are all absent, and all three need a data
  modality we have not wired in. That is one gap, not three.

## 6. The field's own decision table

Deistler et al., "Simulation-Based Inference: A Practical Guide" (arXiv 2508.12939),
from the group that maintains `sbi`. Read in full. Its Table 1 compares the three
engines on inference speed, handling of i.i.d. observations, data dimensionality,
training cost, and robustness to invalid simulations. **That is the decision table the
skill should implement**, alongside Thiele's Section 2.7 rules.

Their architecture guidance by data type, which is what a zoo has to stock:

    images                    CNN embedding
    time series               RNN or transformer embedding
    i.i.d. observations       permutation invariant, they name Set Transformers
    NPE inference net         normalizing flows, or diffusion models
    NLE inference net         normalizing flows
    NRE classifier            ResNets

Two things in it bear directly on our results.

**Their ensembling result complements ours rather than contradicting it.** On a 31
parameter neuroscience model trained on three million simulations, an ensemble of five
NPE models turned slight individual overconfidence into good global and local
calibration. We measured four members closing under a fifth of the gap at 800
simulations. Both are consistent with the same explanation: averaging needs members that
disagree, and clones trained on a small shared training set do not disagree. That is
also the argument for `npeMixedEnsemble3`.

**They name local diagnostics we do not run.** Global: expected coverage, SBC, TARP. We
have coverage and TARP, and no SBC. Local: LCT and L-C2ST. We have neither, and our
nreMlp finding, calibrated on aggregate and broken near the sigma_8 boundary, is exactly
what local tests exist to catch.

**They maintain a curated database of over 100 published SBI applications** (their
Appendix A4). That is the natural source for the five held out problem descriptions the
brief requires, and it removes the objection that we wrote our own test cases.

## 7. Revised priorities, now evidence backed

1. `lampeNsf` and `lampeMaf` against `npeNsf` and `npeMaf`. Measures an unquantified
   claim in the supervisor's own Section 3.4. Costs nothing, and flows are the most used
   family in the count above.
2. SBC and L-C2ST diagnostics. Named by both reviews, and L-C2ST is the named tool for
   the failure we already found by hand.
3. One data modality with an embedding network. CNN, transformer and set architectures
   are 6.2 per cent of the literature between them and 0 per cent of the zoo, and all
   three are blocked on the same thing.
4. Balanced NRE, still not exposed by ltu-ili, still the only method on this list aimed
   at the overconfidence the zoo actually measured.
5. Held out problem descriptions from the Deistler application database.

## 8. Honest limits, after closing the first three

Closed since the first version: non arXiv venues are now covered through OpenAlex and
Crossref; two papers are read in full rather than one; and usage is counted from
architecture neutral queries rather than asserted.

What remains:

- NASA ADS, the astronomy native index, still not queried. No token is configured.
  Astronomy specific venues that neither OpenAlex nor Crossref index well could shift
  the counts.
- Semantic Scholar rate limited during the neutral sweep, so citation weighted ranking
  was not applied. Counts are paper counts, not impact weighted.
- 194 of 256 papers had abstracts. The 62 without are uncounted, and they are not a
  random sample: older and paywalled records are likelier to lack one.
- Counting a term in an abstract is a proxy for using the method. A paper can use a
  normalizing flow and never write the phrase.
- Two papers read in full out of 256. The rest is abstracts plus two reviews'
  reference lists.
