# results/

Every measurement the project has produced. Nothing here is edited by hand, and nothing
here is deleted when it turns out to be wrong. A superseded file stays, renamed or noted,
because a retracted number whose evidence is gone reads as though the work went smoothly.

Regenerate everything derived from these with `python -m ili_kaai.rebuild`.

## How to tell which file is authoritative

`zoo.json` records its own provenance. Open it and read `source`:

    source.sweepsUsed         the sweep files merged into the catalogue, oldest first
    source.sweepsSkipped      files deliberately not merged, each with a reason
    source.supersededPairs    every (entry, task) pair a later sweep replaced

Later files win per (entry, task) pair. That ordering is explicit in `SWEEP_FILES` in
`ili_kaai/zoo.py`, because when it was implicit a lookup took the first match and the
older defective numbers silently won.

## The catalogue

| file | status | what it is |
|---|---|---|
| `zoo.json` | LIVE | the catalogue. Built by `ili_kaai/zoo.py`. Every downstream document reads this. |
| `paramCount.json` | LIVE | trainable weight counts per entry. Built separately by `ili_kaai/paramCount.py` because it rebuilds every net. `null` means could not be built, never zero. |

## Sweeps

| file | status | cells | what it measured |
|---|---|---|---|
| `sweep.json` | MERGED | 72 | the first full benchmark, summary vector entries on the three CAMELS tasks |
| `sweepPosterior.json` | MERGED | 117 | the broad sweep after posterior handling was corrected |
| `sweepMcmc.json` | MERGED | 18 | the MCMC engines, NLE and NRE, which need a fresh chain per observation |
| `sweepCloud.json` | SUPERSEDED | 42 | point cloud entries, measured with the batch-dependent rescale defect |
| `sweepCloudFixedScaling.json` | MERGED | 42 | the same cells after the rescale was removed. Supersedes 14 pairs above. |
| `sweepQuijoteJoint.json` | MERGED | 48 | Quijote at 26,202 simulations, 2 parameters. Overturned the overconfidence finding. |
| `sweepQuijoteAll.json` | MERGED | 45 | Quijote at 26,202 simulations, 5 parameters |
| `sweepQuijote.json` | REFUSED | 2 | incomplete. `zoo.py` refuses to merge a partial sweep, and records the refusal. |
| `sweepQuijoteJoint800.json` | HELD OUT | 12 | Quijote deliberately cut to 800 simulations. The control that isolates training-set size from simulation suite. **Not merged on purpose**: it was measured at a different training-set size from every other entry, so merging it would break the comparability the catalogue rests on. |
| `sweepSupersededTarpBug.json` | SUPERSEDED | 72 | the original sweep, kept because a TARP bug invalidated its coverage numbers |
| `probeQuijoteAll.json` | PROBE | 1 | a single cell run to time the sweep before committing to it |

## Checks

Each check tests a case whose answer is known independently of the code being checked.

| file | what it establishes |
|---|---|
| `toyModel.json` | the LtU-ILI pipeline reproduces the paper's own analytic toy problem |
| `tarpCalibration.json` | the coverage metric reads nominal on a posterior known to be calibrated |
| `calibrationNoiseBand.json` | how much coverage moves between seeds when nothing changes. This is where the calibration tolerance comes from, so the threshold is measured rather than chosen. |
| `edgeCoverage.json` | whether coverage differs near the edge of the prior box |
| `edgeBaseline.json` | the same, against a matched-accuracy baseline |
| `emittedConfigCheck.json` | every config the skill emits actually loads in ltu-ili |
| `scalingComparison.json` | whether removing the rescale defect changed any result beyond seed spread |
| `pointCloudDiagnostics.json` | the point cloud collapse investigation |

## Skill evaluation

| file | what it is |
|---|---|
| `skillEvaluation.json` | development set. Tuned against, so treat it as optimistic. |
| `skillEvaluationTwo.json` | clean set, never tuned against. This is the honest score. |

## runs/

Saved posteriors and per-epoch summaries that ltu-ili writes to `out_dir`. Gitignored.
Nothing reads them. Every number quoted anywhere in this project lives in the JSON files
above, not in these.
