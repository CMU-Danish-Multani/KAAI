# What the literature told us

Plain-language version of the paper sweep done on 2026-08-17. The technical
index, with arXiv numbers and one line per paper, is in
[literature.md](literature.md). This page is the reasoning.

Read [plans.md](plans.md) first. This document assumes it.

Confidence is tagged throughout. MEASURED means a number someone reported.
INTERPRETED means a conclusion drawn from it, which could be wrong.

---

## 0. The one paragraph version of what we are building

A thousand simulated universes were made by setting two dials, then letting
gravity run for 13 billion years. We look at the finished universe and guess the
dials. To do that we need a neural network, and there are millions of possible
designs. **Neural architecture search**, or NAS, is a machine that tries many
designs and reports which works best, instead of a human guessing.

The important limitation: NAS can only find designs that are on its list. That
list is the **search space**. If pizza is not on the ingredient list, no amount
of searching finds pizza. Designing that list is most of the real work, and it
is where our contribution has to live.

---

## 1. A paper from July 2026 already does our Phase 3 and 4

**Agentic Neural Architecture Search**, arXiv 2607.07984.

How it works. Imagine you want the best sandwich.

1. A language model writes one good recipe from scratch, having read every
   cookbook ever written.
2. It then turns that recipe into a fill-in-the-blanks template. "Bread: ____.
   Sauce: ____. Filling: ____", with a few options in each blank.

Step 2 is the clever part. **The template is the search space.** So instead of a
human writing the ingredient list, the language model writes one tailored to the
task. Ordinary NAS then fills in the blanks.

- MEASURED: best result on 11 of 17 tasks across two benchmark suites.
- INTERPRETED: "we used a language model to help search for architectures" is no
  longer a contribution. It is published, it works, and it is one month old.

**Two doors they left open.**

- MEASURED: their own future-work section says they did not try weight-sharing
  methods inside their template. That is exactly our Phase 2.
- MEASURED: they only ever swap out pieces of the network. They never change how
  the input data is prepared.
- INTERPRETED: our graph-construction axis has no counterpart in their work, and
  it is the more defensible of the two openings.

---

## 2. Weight sharing, and why our version of the question is unusual

Training 360 networks one at a time is like building 360 houses from scratch.
Too slow for the hardware we have.

**Weight sharing** is the shortcut. Build one giant house containing every
possible floor plan at once. Each candidate design is a path through the rooms.
Train the giant house once and you have trained every design at the same time.

The catch is real. Each room gets furnished as a compromise that works
acceptably for every path and perfectly for none. So when we ask "which path is
best?", the answer may be wrong. We are measuring which path suits the
compromise furniture, not which path is genuinely best once properly furnished.

That is the **ranking problem**. It is what Phase 2 sets out to measure.

Another way to see it. A five minute quiz claims to predict who tops the final
exam. It might get the class average right and still rank student A above
student B when the exam says the reverse. For picking a winner that quiz is
useless. The only way to find out is to give both tests to the same students and
compare the orderings.

**Why our version is unusual.** Every study of this problem we found is on image
recognition, where big deep networks win comfortably.

- MEASURED (CosmoBench Table 2): on Quijote a 49-parameter least-squares fit
  beats a 671,000-parameter graph network.
- INTERPRETED: nobody has checked whether the giant-house shortcut still ranks
  correctly in a regime where the tiny model wins. That is a genuine unknown, and
  the AgentNAS authors name it as their own open problem.

---

## 3. A paper claims the tree's shape alone is enough

There are two ways to read a family tree.

| Way | What you look at | In our data |
|---|---|---|
| **Node features** | Each individual: height, weight, age | The four numbers per blob: mass, concentration, v_max, time |
| **Topology** | Only the shape: who descends from whom, how many branches | The `edge_index` arrows |

Our own prediction, in [../merger_trees/notes.md](../merger_trees/notes.md), was
that individual features matter and shape barely does.

- MEASURED: DeepSets throws away shape entirely and scores 0.993 on Omega_m. The
  shape-aware graph network scores 0.996. So shape appeared worth about 0.003.

**Linking Warm Dark Matter to Merger Tree Histories** (arXiv 2511.05367,
November 2025) reports the opposite.

- MEASURED (their claim, abstract only, unverified): their network reads the
  answer from merger trees **with the node features removed entirely**, using
  nothing but shape.
- These are not strictly contradictory. Different target quantity, different
  simulation suite, different tree construction.
- INTERPRETED: still cheap to check on our data. Zero the four node features,
  train, see what survives. If shape alone carries real signal for sigma_8, the
  weighting between topology blocks and feature blocks in our search space
  changes.

---

## 4. Oversquashing may explain why sigma_8 is stuck

This is the most underrated lead from the sweep.

Our trees are long and thin. Typical tree is about 769 blobs, largest is 37,865,
and they form long chains stretching back in time.

Graph networks work by **message passing**. Each blob whispers a short message to
its immediate neighbours. Then everyone whispers again. One round of whispering
per layer of the network.

So to get information from the oldest ancestor down to the final blob, that
information has to survive hundreds of whispers in a row. That is a game of
telephone down a line of 700 people. What arrives is mush.

Worse: hundreds of different ancestors all send their messages through the same
narrow chain, so everything gets crushed into one small message. That crushing is
what the term **oversquashing** means.

Why this should bother us specifically:

- MEASURED (our own exploration): Omega_m is read from concentration, a property
  measured directly on each blob. sigma_8 is read from *when* things happened,
  meaning how much of the tree sits at early times.
- MEASURED (CosmoBench Table 5): Omega_m reaches 0.996. sigma_8 is stuck at 0.82.
- INTERPRETED, untested: the early-time information is exactly the information
  that has to travel furthest through the chain. So the signal we most need for
  sigma_8 is the signal the architecture is most likely to destroy.

**Cheapest test.** Add one **virtual global node**, a single extra node connected
to every other node, so any blob reaches any other blob in two hops instead of
seven hundred. One line of code. It either moves sigma_8 or it does not, and
either answer is informative.

Relevant papers are listed in [literature.md](literature.md) §7.

---

## 5. NAS will find the cheat, and that is the best paper here

We already measured something quietly alarming.

- MEASURED: in CAMELS, counting how many galaxies are in the box correlates
  0.758 with Omega_m. No structure. No model. Just counting.
- MEASURED: the published graph network, training for hours on real structure,
  reaches 0.78.

The cause is a technical artifact, not physics. A blob only gets recorded once it
contains about 20 simulation particles, and how heavy one particle is depends on
Omega_m. So the count leaks the answer.

This is a student who notices the answer to every textbook question is C, and
scores 90 percent without reading anything.

**Why this becomes a paper.** A human researcher who notices the trick feels
uneasy and stops. NAS has no conscience. It is a machine whose only job is to
push the score up. Point it at this data and it will find the counting trick
faster than any human reviewer, then hand the result over as a triumph.

- INTERPRETED: demonstrating that on a real benchmark, then designing a search
  space where the cheat is unavailable and showing what the score does, is a
  result about the safety of automated model design. Nothing found in the sweep
  addresses it.
- INTERPRETED: this is stronger than a small improvement in sigma_8, because a
  small improvement is a number and this is a lesson.

---

## 6. Where this could actually be published

**The premise needs correcting first.** Astrophysics does not publish at
conferences the way computer science does. In machine learning, NeurIPS and ICML
*are* the archival venue and a paper there is the publication. In astronomy,
conferences such as the AAS meetings and IAU symposia are talks and posters.
Proceedings exist but carry little weight, and nobody lists them as their main
output.

**Astronomy publishes in journals.** There is no A-star astrophysics conference
to target, because the category does not exist. The equivalents are:

| Venue | Type | What it wants |
|---|---|---|
| MNRAS, ApJ, A&A | astronomy journals | an astrophysics result |
| JCAP, PRD | cosmology and particle journals | a cosmology or statistics result |
| NeurIPS, ICML, ICLR | ML conferences, archival | a methods result |
| ML4PS workshop at NeurIPS | ML crossover workshop | either, at lower stakes |
| Machine Learning: Science and Technology | crossover journal | either |

**This matters because the two audiences want different papers.**

An ML venue wants a *methods* claim. "Automated architecture search exploits
resolution artifacts, and here is a search space design that prevents it." The
cosmology is the setting, not the point.

An astronomy journal wants a *physics* claim. "Here is how much information
merger trees carry about sigma_8, and here is what currently limits our access to
it." A referee at MNRAS reading a pure NAS paper will ask what the astrophysics
is, and the honest answer today would be "not much."

**Two of our three candidate contributions can be turned astro-facing.**

- **The leak, as a systematics paper.** Astronomers take systematics extremely
  seriously, because a wrong measurement of the universe is worse than no
  measurement. "Machine learning results on these benchmarks are partly reading a
  resolution artifact rather than structure, here is how much, here is how to
  test for it" is a real astronomy contribution. It is also useful to the CAMELS
  and CosmoBench community directly. INTERPRETED: this is the most journal-viable
  version.
- **The sigma_8 bottleneck.** "The missing sigma_8 information is not absent from
  merger trees, it is destroyed by the architecture" is a physics claim about
  information content, and it is testable. INTERPRETED: astro-viable if the
  measurement holds up.
- **Ranking fidelity of weight-sharing NAS.** INTERPRETED: this is an ML paper
  and will not interest an astronomy referee. Keep it for the ML track.

**Realistic sequencing**, all INTERPRETED:

1. ML4PS workshop at NeurIPS. Short, low risk, gets the work in front of exactly
   the right people, and does not burn the result.
2. Then either a full ML conference submission for the methods claim, or an
   MNRAS or JCAP submission for the systematics claim. Possibly both, since they
   are genuinely different papers off the same experiments.

**The blunt version.** Method transfer alone, meaning "we applied an existing NAS
technique to cosmology data", is a workshop paper anywhere and a rejection at a
journal. One of the three findings above has to be the spine, with real
measurements behind it.

---

## 7. What has to happen before any of this is a paper

Stated plainly, because the gap is large.

- **No model has been trained yet.** [plans.md](plans.md) §7 names the gate:
  reproduce one published baseline number before building anything else. The
  CS-Trees graph network at Omega_m R-squared 0.996 is the cheapest check at
  about 13 minutes. Until that reproduces, every number downstream is
  uninterpretable. This is still the single highest-value hour in the project.
- **Compute.** A Mac with MPS constrains us to the trees track. Every claim about
  point clouds, and anything on Quijote, needs hardware we do not have.
- **Repetition.** No comparative claim from a single run. Anything that goes in a
  paper needs a mean and a spread across seeds, and improvements around 0.01 have
  to be shown to exceed the bootstrap spread the benchmark itself reports.

---

## 8. What we have not checked

- Only one paper was read properly, the structured report for AgentNAS.
  Everything else in this document sourced from a paper is an abstract-level
  claim by its authors, unverified by us.
- The search covered arXiv only. MNRAS and ApJ papers that never post to arXiv
  were not searched. NASA ADS would cover them and needs an `ADS_TOKEN`.
- BioArc itself was not re-verified. Claims about it come from
  [plans.md](plans.md).
- Nothing in §3, §4 or §5 has been tested on our data. All three are experiments
  we could run, not results we have.

---

Style note: this file follows the global no-dash and no-contraction rules, which
differ from the older files in this directory. Existing files were left alone.
