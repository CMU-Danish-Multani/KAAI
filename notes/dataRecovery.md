# Recovering data/

`data/` is gitignored and holds 1.5 GB of downloaded simulation files plus two derived
caches. Nothing in it is unique to this project: the raw files come from a public
archive, and both caches rebuild from them.

Written 2026-08-30 after `data/` was deleted by an over-broad `git clean -fdX` during a
repository tidy. The `.gitignore` claimed the download URLs were recorded here, and they
were not. They are now.

## Source

    users.flatironinstitute.org/~fvillaescusa/CosmoBench/

The full archive is 324.5 GB. This project uses 1.5 GB of it. The velocity-task files
(137 GB) and the pre-built graphs (78 GB) are deliberately skipped, because everything
they contain is rebuildable from the files below.

## Layout the code expects

Paths are hard-coded in `point_clouds/load.py` (`PATHS`) and `ili_kaai/tasks.py`
(`QUIJOTE`). Each suite ships its own train, validation and test split as separate files,
so the split is the dataset provider's, not ours.

    data/
      CAMELS/       ALL_galaxies_{train,val,test}.hdf5        96 MB
      CAMELS-SAM/   top5000_galaxies_{train,val,test}.hdf5   175 MB
      Quijote/      tpcf_top5000_{train,val,test}.hdf5

Inside each file, one HDF5 group per simulation holding `X`, `Y`, `Z` as three separate
lists, plus a `params/` group holding one value per simulation. `point_clouds/load.py`
documents the structure in full.

## Rebuilding the caches

Both are derived, and both regenerate without touching the network once the raw files
are in place.

    python -m point_clouds.tpcf     -> data/tpcf_cache/*.npz    the 25 bin correlation
                                                                functions every summary
                                                                vector task reads
                                       data/cloud_cache/*.npz   built on demand by
                                                                point_clouds/cloudCache.py
                                                                the first time a point
                                                                cloud task runs

## Verifying a recovery

Do not trust a download because it returned HTTP 200. Two checks, in order.

**1. Are the source files the ones the project measured?** The galaxy count per
simulation is a fingerprint, and `ili_kaai/tasks.py` documents the expected range.

    CAMELS       1000 clouds   N from 588 to 4511      matches tasks.py
    CAMELS-SAM   1000 clouds   N fixed at 5000         matches tasks.py

Split sizes are 600/200/200 for CAMELS and **600/204/196** for CAMELS-SAM. The uneven
CAMELS-SAM split is correct and is recorded in `notes/projectGuide.md`. A separate known
quirk: CAMELS-SAM val has 201 rows in the shipped tpcf file but 204 clouds in the
position file, noted in `runLog.md`.

**2. Does the pipeline reproduce a recorded number?** This is the real test, and the
pipeline is bit-reproducible, so an exact match is achievable and anything less means
something is wrong.

    conda run -n ltuili python -m ili_kaai.sweep --architectures npeMaf \
        --tasks camelsJoint --seeds 0 --n-eval 200

Verified 2026-08-30, all three to every recorded digit:

    npeMaf                       camelsJoint  seed 0   R2 [0.8412, 0.3582]   exact
    lampeMaf                     camelsJoint  seed 0   R2 [0.8141, 0.3520]   exact
    npeMafPairwiseGnnPretrained  camelsCloud  seed 0   R2 [0.2803, 0.1650]   exact

**`--n-eval 200` is required.** The CLI default is 100, and every recorded sweep used
200. Reproducing at the default gives values that are close, plausible, and wrong: the
first attempt at this check read +0.0287 high on npeMaf and sent the investigation
looking for a data fault that did not exist. Each cell records the value it used in
`nEvalPoints`; read it before reproducing.

## What does not need data/

Everything already measured. The catalogue, the generated facts, the emitted configs and
the skill all read JSON, not simulations. Verified while `data/` was absent:

    conda run -n ltuili python -m ili_kaai.rebuild --skip-evaluation   3/3 stages ok
    conda run -n ltuili python -m skill.evaluate                       ran, scored
    conda run -n ltuili python -m skill.query ...                      ranked, emitted

`data/` is needed only to run a new sweep or rebuild a cache.
