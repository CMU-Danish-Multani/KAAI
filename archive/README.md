# Archive

Nothing here is deleted, and nothing here is on the live path. These files were built
before the project was re-grounded on LtU-ILI (Ho et al. 2024, arXiv 2402.05137), the
framework named in the assigned task. They are kept because measurements we still quote
were produced by them, and a quoted number whose producing code is gone is not evidence.

Every number these produced lives in `point_clouds/results/*.json` and `archive/zooV1Results/*.json`,
which are kept as the evidence for those numbers.

## blocks/
Aggregation blocks. Each contributed one screened entry to the pre-LtU-ILI catalogue.
- `pna.py` PNA with degree scalers. Measured to leak the galaxy count at R2 0.70 while
  its own `depends_on_point_count()` claimed it was blind. Bug found and fixed; the
  episode is the reason the screen is measured rather than declared.
- `attention_readout.py`, `quasi_arithmetic.py` screened clean, one measurement each.
- `edge_features.py` 845 lines. Angular edge features. Retracted as a finding: they
  appeared in every search leader, but the means were flat (+0.4367 with, +0.4387
  without). That was selection, not evidence.

Kept live: `count_screen.py` (the admission criterion) and `fishnets.py` (LtU-ILI
Section 5.6 uses fishnets as a set embedding, so it maps onto a real framework entry).

## nas/
Bespoke architecture search: a hand written search space, a searchable model, and an
Optuna driver. Superseded. The search now runs over LtU-ILI configurations instead of
over our own layer vocabulary, which is what makes a shared hyperprior meaningful.

## training/
The drivers that produced the results JSONs. Their settings are recorded inside each
JSON (seeds, device, library versions), so the JSON remains readable without them.

## zooV1/
The pre-LtU-ILI zoo: schema, registry, recommender, evaluation harness, and four
inference heads calling `sbi` directly. The ranking logic is worth re-reading when the
LtU-ILI version is written. The four heads are replaced by `ili.inference`, which
covers NPE, NLE and NRE across two backends rather than NPE and NRE in one.

## notes/
- `plansPreLtuIli.md` the 876 line plan aimed at beating CosmoBench with a weight
  sharing supernet. Superseded in full by the rewritten `notes/plans.md`.
- `spec_stage1_gate.md` the spec for the reproduction gate, which passed.

## notes/ (added 2026-08-28)

Four documents superseded when the project was re-grounded on LtU-ILI. Archived rather
than deleted because they record what was believed at the time.

- `resultsPreLtuIli.md` results of the point cloud track, stages 0 to 3. Superseded by
  `notes/projectGuide.md` Section 5 and by `ili_kaai/results/*.json`, which are derived
  from data rather than typed.
- `findingsNasEra.md` and `literatureNasEra.md` the literature sweep from 2026-08-17,
  written when the goal was neural architecture search over CosmoBench point clouds.
  That is not the goal. The current sweep is `notes/zooCandidates.md`, and the technical
  paper index that survives is `notes/related_papers.md`.
- `cheatsheet.md` MOVED OUT 2026-08-30 to `notes/glossary.md`. This entry used to call
  it "a glossary of five terms, subsumed by notes/projectGuide.md Section 9". That was
  wrong on both counts: it holds 94 rows, and it covers physics vocabulary (Omega_m,
  halo, Mpc/h, NFW, the correlation function) where Section 9 covers inference
  vocabulary (NPE, TARP, PIT, amortized). They barely overlap, three current notes link
  to it, and it was never superseded.

## pointClouds/ (added 2026-08-28)

- `lls.py` linear least squares on pairwise distance statistics, the CosmoBench baseline
  we beat in Stage 1. Its driver was archived earlier, nothing imports it, and it has no
  entry point. Its measurements survive in `point_clouds/results/lls_baseline.json`.
