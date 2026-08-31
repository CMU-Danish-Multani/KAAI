# Cheat sheet

Every term in this project, in plain language. Add to it as you go.

Definitions marked ▷ are quoted or paraphrased from the CosmoBench paper's own
glossary (Appendix A.1), those are the authors' words, not mine.

---

## The two things we predict

| Term | Plain meaning | Range | Paper's definition |
|---|---|---|---|
| **Ω\_m** (omega-m) | How much **stuff** the universe contains | 0.1 to 0.5 | ▷ The fraction of the total energy density of the universe made up of matter (dark + normal). If Ω\_m is 1, the universe is matter-dominated and flat. |
| **σ\_8** (sigma-8) | How **lumpy** it started out | 0.6 to 1.0 | ▷ A measure of how much matter has clumped together at a specific scale (8 Mpc/h). Quantifies the amplitude of matter fluctuations. |

Both are **cosmological parameters**, ▷ *key numerical values that define the
properties of the universe.*

### Other cosmological parameters (Quijote varies five)

| Term | Meaning |
|---|---|
| **Ω\_b** | Fraction made of *normal* matter, protons and neutrons. A subset of Ω\_m. |
| **n\_s** | Spectral index. Whether small or large fluctuations dominated the early universe. |
| **h** | Expansion rate of the universe, divided by 100 km/s/Mpc. Shows up in distance units as `/h`. |

---

## The nuisance parameters (the labels you didn't recognise)

These sit in `params/` alongside the answers, but **we are not predicting them.**
They are **astrophysical parameters**, ▷ *values characterising the physical
processes governing galaxies: supernova feedback, active galactic nuclei, gas
cooling, star formation.*

They were varied between simulations too, so the same cosmology can look
different depending on them. That makes them **noise for our task.**

| Field | Appears in | What it controls |
|---|---|---|
| `A_SN1`, `A_SN2` | CAMELS | **Supernova feedback**, how hard exploding stars blow gas out of a galaxy |
| `A_AGN1`, `A_AGN2` | CAMELS | **AGN feedback**, how hard the central black hole blows gas out |
| `A_sn1`, `A_sn2` | CAMELS-SAM | Same idea, different naming *(note the lowercase, the suites are inconsistent)* |
| `Aagn1` | CAMELS-SAM | Same idea |
| `seed` | CAMELS | Random seed / simulation ID. Not physics. |
| `LH` | CAMELS-SAM | Simulation ID. **LH = Latin Hypercube**, the scheme used to spread parameter choices evenly across the range. |

**AGN** = Active Galactic Nucleus, a supermassive black hole actively eating
gas and blasting energy out. **Feedback** = the general term for a galaxy
pushing its own gas around, which changes how it grows.

---

## Objects in the universe

| Term | Plain meaning |
|---|---|
| **Dark matter** | ▷ Matter that emits no light, detectable only by gravity. Most of the universe's mass. |
| **Halo** | ▷ A massive invisible blob of dark matter surrounding galaxies. **The basic building block of cosmic structure** — galaxies nearly always live inside one. |
| **Galaxy** | The visible thing sitting at the bottom of a halo's gravity well. |
| **Large-scale structure** | ▷ How matter is spread on scales of millions of light years — filaments, walls, voids, clusters, forming a **cosmic web**. |
| **Filament / void** | The strands galaxies string along, and the empty gaps between them. |
| **Merger** | Two halos colliding and becoming one. |

---

## Data we actually have

### Per galaxy (point clouds)

| Field | Meaning | Notes |
|---|---|---|
| `X`, `Y`, `Z` | Position in the cube | **The only input the benchmark allows** |
| `VX`, `VY`, `VZ` | Velocity, km/s | Excluded, it's a *different* task's answer |
| `Mstar` | Stellar mass, weight of all its stars | Stored **raw**, not logged |
| `Mgas` | Gas mass | CAMELS only (it simulates gas) |
| `Metal_star` | Stellar metallicity, how much heavy-element pollution | "Metal" in astronomy = anything heavier than helium |
| `Vmax` | ▷ Maximum circular velocity — fastest orbit anything reaches | CAMELS only |
| `mHI` | Mass of neutral hydrogen | CAMELS-SAM only |

### Per blob (merger trees)

| Field | Meaning | Notes |
|---|---|---|
| mass **M** | How heavy the blob is | Stored as **log10**, "12.5" means 10^12.5 suns |
| **concentration c** | ▷ How centrally dense a halo is — ratio of its outer radius to its dense core (NFW profile) | Secretly records *when* it formed. **Best single predictor of Ω\_m (0.70)** |
| **v_max** | Max circular velocity | Also log10 |
| **scale factor a** | *When* this blob existed | Only 172 distinct values, it's discrete |

### Per tree (bookkeeping)

| Field | Meaning |
|---|---|
| `lh_id` | Which simulation the tree came from. **Split by this, never randomly**, 25 trees share one answer |
| `node_halo_id` | Each blob's global ID |
| `mask_main` | IDs of the tree's main trunk |
| `edge_attr` | How many time-steps an arrow skips (2 to 70, because blobs were deleted) |

---

## Time and distance

| Term | Meaning |
|---|---|
| **scale factor a** | ▷ The relative size of the universe. `a = 1` is today, `a = 0.5` is when it was half its current width. |
| **redshift z** | ▷ How much light has been stretched by expansion. Also a clock: `z = 0` is now, higher z is further back. Related by **a = 1/(1+z)** |
| **Mpc** | ▷ Megaparsec ≈ **3.26 million light years** |
| **cMpc** | ▷ *Comoving* megaparsec — a distance that accounts for the universe's expansion, so objects drifting apart with it stay a fixed distance apart |
| **cMpc/h** | The same, divided by the expansion rate. Astronomers use this so results don't depend on knowing `h` exactly |
| **dex** | One unit in log10, "0.3 dex" means a factor of 2 |
| **M☉** | One solar mass (the Sun's weight) |

---

## Kinds of simulation

| Term | Meaning |
|---|---|
| **N-body** | ▷ Models only gravity between particles. Fast, dark matter only. *(Quijote, CAMELS-SAM)* |
| **Hydrodynamical** | ▷ Models gravity **and** gas, stars, black holes, feedback. Far more expensive. *(CAMELS)* |
| **SAM** (Semi-Analytical Model) | ▷ Fast equations for galaxy formation layered on top of N-body merger trees. Cheap way to get galaxies without full hydro. *(the "SAM" in CAMELS-SAM)* |
| **Snapshot** | One frozen moment of a simulation, saved to disk |
| **Latin Hypercube (LH)** | A way of choosing parameter combinations so they cover the range evenly rather than clustering |

## The four datasets

| Name | Type | Box | Objects | Count |
|---|---|---|---|---|
| **Quijote** | point clouds | 1000 cMpc/h | halos | 32,752 |
| **CAMELS-SAM** | point clouds | 100 cMpc/h | galaxies | 1,000 |
| **CAMELS** | point clouds | 25 cMpc/h | galaxies | 1,000 |
| **CS-Trees** | merger trees | 100 cMpc/h | halos | 24,996 |

`top5000` in a filename = only the 5,000 heaviest objects kept.
`ALL` = every object kept, so cloud sizes vary.

---

## Measurements and statistics

| Term | Meaning |
|---|---|
| **R²** | Score from 0 to 1. **1 = perfect prediction, 0 = no better than guessing the average.** Can go negative if worse than guessing |
| **correlation r** | −1 to +1. **0 = no relationship.** +1 = perfectly move together, −1 = perfectly opposite |
| **2PCF** / ξ(r) | Two-point correlation function, counts how many object pairs sit at each separation, compared to random. The classic clustering measure |
| **Power spectrum P(k)** | The same information as 2PCF, expressed in terms of wave-sizes instead of distances |
| **NFW profile** | The standard formula describing how density falls off from a halo's centre. `concentration` is defined against it |
| **periodic boundaries** | The cube wraps around like Pac-Man, a galaxy at 99.9 neighbours one at 0.1 |

---

## Machine learning terms in this project

| Term | Meaning |
|---|---|
| **GNN** | Graph Neural Network, works on things connected by edges rather than fixed grids |
| **MPNN** | Message-Passing Neural Network. Each node repeatedly updates itself from its neighbours. "4-layer" = 4 rounds of that |
| **DeepSets** | A model that treats input as an unordered *set*, ignores all connections |
| **pooling** | Squashing many node vectors into one summary for the whole graph |
| **NAS** | Neural Architecture Search, searching for the best network design automatically instead of guessing |
| **supernet** | One big network containing every candidate design as a path through it, so they share weights and train together |
| **SPOS** | Single Path One-Shot, train the supernet by activating one random path per step |
| **normalisation** | Rescaling features to mean 0, spread 1, so a numerically bigger feature doesn't dominate |
| **leakage** | When information that shouldn't be available sneaks into training. Inflates scores silently |

---

## Traps found in this data

| Trap | What happens |
|---|---|
| **Counting galaxies** | In CAMELS, `n_galaxies` alone predicts Ω\_m at **0.758**, no structure needed. Halos are only counted once they hold ~20 particles, and particle mass depends on Ω\_m |
| **The mass cliff** | Tree masses stop dead at 10^10.477, a 2,000× jump in one bin. That's the anti-cheat pruning, not physics |
| **Amputated history** | Because small blobs were deleted, `a_50` (the standard formation-time measure) scores **0.05**. Useless here |
| **Velocity** | Strongest predictor available (0.76) and **we may not use it**, it's a different task's answer |
| **Clustering runs backwards** | Higher Ω\_m gives *fewer* close pairs (−0.71), because the top-5000 cut selects rarer, more clustered objects in matter-poor universes |

---

## Things I still need to look up

<!-- Add as you hit them -->
- 
