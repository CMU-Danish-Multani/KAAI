# KAAI

**Project 2.6: a curated model zoo for astrophysical simulation-based inference, plus a
Claude skill that searches it.** Built on LtU-ILI (Ho et al. 2024, arXiv 2402.05137).

The problem: every group starting a new astrophysical inference problem repeats the same
architecture search. What works is not written down, so it gets rediscovered each time.
The zoo records it, measured the same way for every entry, and the skill reads it.

## Start here

1. **[notes/projectGuide.md](notes/projectGuide.md)** explains the whole project end to
   end for someone with no astrophysics background. Physics, statistics, what was built,
   what was measured, what is left, and a glossary. If you read one file, read that one.
2. **[skill/measuredFacts.md](skill/measuredFacts.md)** holds every current number.
3. **[runLog.md](runLog.md)** is the append-only record of every run, correction and
   retraction, newest at the bottom.

## Where numbers live, and why none are in this file

Every number in a deliverable is derived from a JSON by path. Nothing is retyped. This
file deliberately quotes no results, because an earlier version of it carried a headline
that went stale the moment the catalogue was rebuilt.

    ili_kaai/results/zoo.json     the catalogue. Built by ili_kaai/zoo.py from the
                                  sweep files, never hand edited.
    skill/measuredFacts.md        generated from zoo.json by skill/facts.py. This is
                                  what the skill quotes and what you should quote.

If those two disagree with any prose in the repository, they are right and the prose is
stale. Regenerate with `ili_kaai/rebuild.py` rather than editing by hand.

## The two deliverables

**The zoo.** `ili_kaai/`. Architectures defined as LtU-ILI configs, swept across tasks at
three seeds each, scored for accuracy and for calibration, and admitted to the catalogue
per (entry, task) pair rather than per entry.

**The skill.** `skill/`. A Claude Skill, symlinked into `.claude/skills/` so any Claude
session opened in this repository picks it up. It has two retrieval arms, a structured
ranker and a few-shot reader, because the brief poses which works better as an open
question. `skill/evaluate.py` scores them on held-out problems taken from published work.

## Map of the code

    ili_kaai/                  THE LIVE PROJECT
      tasks.py                   the inference tasks and how their data is loaded
      architectures.py           the zoo entries, as LtU-ILI configs
      embeddings.py              point cloud encoders (DeepSets, PointNet, pairwise GNN)
      sweep.py                   trains and scores every entry on every task
      zoo.py                     merges the sweeps into the catalogue
      paramCount.py              trainable weight counts, measured separately
      rebuild.py                 runs all of the above in dependency order
      checks/                    each check tests a case with a known answer
      results/                   every measurement, see results/README.md

    skill/                     THE CLAUDE SKILL
      SKILL.md                   the instructions Claude reads
      query.py                   the structured ranker, and the ltu-ili config emitter
      facts.py                   regenerates measuredFacts.md from zoo.json
      measuredFacts.md           GENERATED. Do not edit by hand.
      evaluate.py                scores the skill on held-out problems
      heldOut.json               development set, tuned against
      heldOutTwo.json            clean set, never tuned against
      CONTRIBUTING.md            what an external submitter must supply

    common/metrics.py          seeding, R2, credible coverage. Shared.
    common/viz.py              plotting helpers, used only by the explore scripts.

    point_clouds/              DATA PRODUCTION
      load.py                    opens the HDF5 suites
      tpcf.py                    builds data/tpcf_cache/*.npz, which tasks.py reads
      cloudCache.py              builds the point cloud cache, which tasks.py reads
      gnn.py, pointnet.py,       PRE-LTU-ILI. Not on the live path. Kept because
      blocks/, explore.py        measurements in point_clouds/results/ came from them.

    merger_trees/              Committed Phase 0. Not in the brief, not extended.
    notes/                     All prose. See notes/ below.
    archive/                   Superseded work, with a README saying what each produced.
    env/                       Conda environment build and patch scripts.
    resources/                 Assignment PDFs.
    data/                      1.5 GB, gitignored, NOT present in a fresh clone.
                               See notes/dataRecovery.md for URLs, layout and rebuild.

## Notes

All prose lives in `notes/`. Nothing is kept beside the code it describes.

| file | what it is |
|---|---|
| [notes/projectGuide.md](notes/projectGuide.md) | the handover document, read this first |
| [notes/plans.md](notes/plans.md) | build order, non-goals, and where each stage stands |
| [notes/zooCandidates.md](notes/zooCandidates.md) | what the field uses, and what the zoo is missing |
| [notes/related_papers.md](notes/related_papers.md) | paper index with arXiv numbers |
| [notes/understanding_data.md](notes/understanding_data.md) | the datasets, what is in them |
| [notes/dataRecovery.md](notes/dataRecovery.md) | how to re-fetch data/ and rebuild its caches |
| [notes/glossary.md](notes/glossary.md) | every term, field name and unit, in plain language |
| [notes/pointCloudsData.md](notes/pointCloudsData.md) | point cloud data notes |
| [notes/mergerTreesData.md](notes/mergerTreesData.md) | merger tree data notes |
| [notes/comms/](notes/comms/) | drafts: supervisor update, compute request |

## Running things

Two conda environments, deliberately. `ltu-ili` pins `sbi<=0.22.0` and the data pipeline
needs newer packages, so they do not share.

    env/buildLtuIli.sh          creates the ltuili environment
    env/patchLtuIli.sh          one-line NumPy 2 fix that ltu-ili 0.1.5 needs

Rebuild everything derived from the sweeps, in dependency order:

    conda run -n ltuili python -m ili_kaai.rebuild
    conda run -n ltuili python -m ili_kaai.rebuild --with-param-count

Run a benchmark sweep, or ask the skill a question:

    conda run -n ltuili python -m ili_kaai.sweep
    conda run -n ltuili python -m skill.query --modality summary_vector --n-params 2 \
        --n-observations 1000 --compute-seconds 7200 --downstream

Checks, each against a case whose answer is known independently:

    conda run -n ltuili python -m ili_kaai.checks.toyModel          # is the pipeline right
    conda run -n ltuili python -m ili_kaai.checks.tarpCalibration   # are the metrics right
    conda run -n ltuili python -m ili_kaai.checks.emittedConfig     # do emitted configs load

Rebuild the input caches, in the other environment:

    python -m point_clouds.tpcf

## Conventions

Recorded in full in the global `CLAUDE.md`. The ones that bite:

- Predictions go in `runLog.md` **before** the run that tests them.
- Three seeds minimum for any comparative claim. Single-run uncertainty is `null`,
  never `0`.
- Every number in a document is derived from a JSON by path. Nothing is retyped.
- Retractions go in a new log entry, never by editing the old one.
- A fix that changes a random stream invalidates every number downstream. Rerun them.

## Known gaps, so nobody rediscovers them

Stated in full in [notes/plans.md](notes/plans.md) section 7.5. The short version:

- **Stage F, the shared hyperprior, was never started.** There is no `hyperprior.py`.
  The admission gate that guards it exists and is measured. The prior it guards does not.
- **Two of four modalities.** Summary vectors and point clouds only. No fields or images,
  no spectra or light curves. The skill declines these rather than guessing.
- **C2ST is blocked, not skipped.** No reference posterior exists for CAMELS. TARP and
  PIT both run.
- **The few-shot arm of the skill is built and gradeable but unmeasured.** Scoring it
  needs a session that has not already read the held-out answer keys.
- **`archive/` does not import.** Its modules reference paths that moved when the project
  was re-grounded on LtU-ILI. It is kept as a record of what produced which measurement,
  not as runnable code.
