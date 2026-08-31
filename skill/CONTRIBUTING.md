# Contributing an entry to the zoo

The brief's second success criterion is that three external groups contribute entries.
That only works if admission is a rule rather than a judgement, so this page is the
rule.

## The one thing that gets you rejected

An entry is admitted when its calibration has been **measured**. Not when it passes.

A recommendation without a calibration verdict is a leaderboard row, and replacing
leaderboard rows is why this catalogue exists. So a submission with excellent accuracy
and no coverage measurement is rejected, and a submission that measures its coverage
and comes out badly calibrated is admitted with that verdict attached to it forever.

Fourteen of the fifteen currently admitted entries are overconfident. Being bad is not
disqualifying. Being unmeasured is.

## What you must supply

**An `Architecture` in `ili_kaai/architectures.py`.** Engine, model, family, backend,
`model_args`, and `repeats` for an ensemble. If your entry reads a point cloud it also
needs `embedding` naming a class in `ili_kaai/embeddings.py`, and `pretrainEpochs` if
it needs the embedding trained alone first.

**A `summary`**, one or two sentences on what the entry is and when a person would
reach for it.

**`known_failure_modes`**, written before you run it. This is the part people skip and
it is the part the catalogue is for. Cite a paper where one exists. If you do not know
a failure mode, say what you expect to go wrong and label it as an expectation.

**Cells from `ili_kaai/sweep.py` at matched settings.** Every existing entry was
measured at `--n-eval 200 --n-draws 1000 --seeds 0 1 2` on `--device cpu`, with the
training budget in `TRAIN_ARGS`. Change any of those and your numbers are not
comparable with anyone else's.

`--n-eval 200` is not a style choice. `calibrationNoiseBand.json` measured the
overconfidence threshold at 200 evaluation points, so coverage estimated from a
different number is judged against a noise band built for the wrong sample size.

**Three seeds minimum.** A single seed has no spread, the recommender ranks on mean R2
minus one standard deviation, and an entry with no spread cannot be ranked honestly.
Single seed submissions are recorded with `r2Std: null`, never zero.

## What happens to your submission

`ili_kaai/zoo.py` merges every completed sweep file and rebuilds the catalogue. Your
entry appears with:

- a calibration verdict per task, in sigma from nominal, against a threshold that was
  measured rather than chosen
- coverage per parameter as well as averaged, because on 3 of 30 measured rows the
  average disagrees with the parameters underneath it
- measured train and inference seconds, which is the axis that actually separates these
  architectures: accuracy spans 0.073 across the catalogue and cost spans a factor
  of 3700
- your failure modes, plus any the rebuild measures and adds

An incomplete sweep file is refused by name and reported, never blended in. A partial
sweep is not a measurement.

## What the catalogue will not do for you

It will not compare your entry against a task it was never run on. Admission is per
`(entry, task)` pair, and `unmeasuredTasks` lists what is missing. A recommendation is
made for one task, so the verdict that travels with it is that task's.

It will not recommend an entry whose mean R2 minus one seed standard deviation is at
or below 0.05, however well calibrated it is. A posterior that predicts nothing is
wide, and a wide posterior scores well on coverage. Calibration is only meaningful
conditional on the posterior being informative.

## Modalities currently stocked

`summary_vector` and `point_cloud`. A literature sweep found the field also uses
convolutional networks for fields and images, and 1D convolutions or transformers for
spectra and light curves. Those are not stocked, and `skill/query.py` declines rather
than guessing when it is asked about them. An entry that opens a new modality is the
most valuable kind of contribution available right now.
