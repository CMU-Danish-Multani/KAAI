# KAAI

**Project 2.6: a model zoo for astrophysical simulation-based inference, plus a Claude
skill that searches it.** Built on LtU-ILI (Ho et al. 2024, arXiv 2402.05137).

The problem: every group starting a new astrophysical inference problem repeats the same
architecture search. What works is not written down, so it gets rediscovered each time.
The zoo records it, measured the same way for every entry, and the skill reads it.

## Start here

**[projectGuide.md](projectGuide.md)** explains the whole project end to end for someone
with no astrophysics background. Physics, statistics, what was built, what was measured,
what is left, and a glossary. If you read one file, read that one.

## The one-line result

Eight architectures scored within 0.064 of each other on accuracy, and seven of the
eight report error bars that are too small. An accuracy leaderboard calls them
interchangeable and would never show you that.

## Map

| file | what it is |
|---|---|
| [projectGuide.md](projectGuide.md) | the handover document, read this first |
| [plans.md](plans.md) | the build order, non-goals, and where each stage stands |
| [../runLog.md](../runLog.md) | append-only log of every run, correction and retraction |
| [zooCandidates.md](zooCandidates.md) | what the field actually uses, and what the zoo is missing |
| [related_papers.md](related_papers.md) | technical paper index with arXiv numbers |
| [understanding_data.md](understanding_data.md) | the datasets |
| [comms/](comms/) | drafts: supervisor update, compute request |

## Code

    ili_kaai/          the live project
      tasks.py           the inference tasks and their data
      architectures.py   the zoo entries, as LtU-ILI configs
      sweep.py           trains and scores every entry on every task
      zoo.py             turns results into the catalogue
      paramCount.py      parameter counts, measured separately
      checks/            four checks, each against a case with a known answer
    point_clouds/      data production: loading, correlation function
      load.py, tpcf.py   produce data/tpcf_cache/*.npz, which ili_kaai reads
      gnn.py, pointnet.py, blocks/   for the point cloud phase, not yet wired in
    common/metrics.py  seeding, R2, coverage. Shared by both.
    merger_trees/      committed Phase 0. Not in the brief, not extended.
    archive/           superseded work, with a README saying what each piece produced

## Running things

Two environments, deliberately. `ltu-ili` pins `sbi<=0.22.0` and the data pipeline needs
newer packages, so they do not share.

    conda run -n ltuili python -m ili_kaai.sweep          # the benchmark
    conda run -n ltuili python -m ili_kaai.zoo            # build the catalogue
    conda run -n ltuili python -m ili_kaai.checks.toyModel        # is the pipeline right
    conda run -n ltuili python -m ili_kaai.checks.tarpCalibration # are the metrics right
    python -m point_clouds.tpcf                           # rebuild the correlation functions

`env/buildLtuIli.sh` creates the `ltuili` environment. `env/patchLtuIli.sh` applies a
one-line NumPy 2 fix that ltu-ili 0.1.5 needs.

## Conventions

Recorded in full in the global CLAUDE.md. The ones that bite:

- Predictions go in `runLog.md` **before** the run that tests them.
- Three seeds minimum for any comparative claim. Single-run uncertainty is `null`,
  never `0`.
- Every number in a document is derived from a JSON by path. Nothing is retyped.
- Retractions go in a new log entry, never by editing the old one.
