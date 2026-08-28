# Request for access to the CMU SCS GPU cluster

Project 2.6, LtU-ILI model zoo. Prepared 2026-08-28.

Every rate below is measured on work already completed on a laptop, not estimated.
Where a number is scaled rather than measured, it says so.

## Summary of the request

    500 GPU-hours on the SCS cluster
     16 CPU cores and 64 GB RAM per job
    100 GB scratch storage
    single GPU per job, no multi-GPU or multi-node needed
    typical job under 2 hours wall clock, so the queue is short-job friendly

Any GPU generation is sufficient. The models in this project range from 4,201 to
135,080 parameters. An A6000 or older is ample and an A100 or H100 would be wasted.
The binding resource is job throughput, not per-GPU capability.

## What has been completed already, and on what

A full benchmark of 8 architectures across 3 CAMELS inference tasks, 3 seeds each.
72 runs, no failures, on one laptop CPU core.

    total wall clock                    16.6 hours
    of which MCMC sampling              16.5 hours   (99.7 per cent)

Measured cost per run, training plus inference over 200 observations:

    npeMdn                 0.7 s        (0.7 train +    0.1 inference)
    npeMaf                 3.3 s        (2.2 train +    1.1 inference)
    npeNsf                 6.6 s        (5.1 train +    1.6 inference)
    npeMafEnsemble4       11.0 s        (8.8 train +    2.3 inference)
    nreMlp               400.2 s        (1.2 train +  399.0 inference)
    nreResnet            752.5 s        (1.0 train +  751.4 inference)
    nleMaf              1889.7 s        (4.2 train + 1885.5 inference)
    nleMdn              3570.7 s        (4.5 train + 3566.2 inference)

Separately measured: a message passing graph network over galaxy point clouds runs at
1.55 s per epoch on laptop GPU, so 7.1 minutes for a 200 epoch fit.

## Why a GPU cluster rather than more laptop time

The completed work covers one data modality, the compressed summary vector. That is
the cheapest modality and the least representative of what the zoo has to serve.

A literature sweep of what astrophysicists actually use for simulation-based inference
gives the following, and the zoo currently covers only the first row.

    modality                          what the field uses         status
    summary vectors                   MLP or none, into a flow    covered
    density fields, lensing maps      CNN, vision transformers    not started
    galaxy and halo point clouds      set networks, GNNs          not started
    cluster and lensing images        CNN                         not started
    spectra, light curves             1D CNN, transformers        not started

Every uncovered row is convolutional or graph based, trained on high dimensional
inputs. Those are GPU workloads, and they are the reason for this request. The
summary vector work fits on a laptop precisely because it is the easy case.

Three further pressures:

**Quijote.** The zoo is trained on 600 simulations. The Quijote suite provides 19,651
training and 6,550 test simulations, five parameters instead of two, and a 1000 Mpc/h
box instead of 25. It is already downloaded. That is 33 times the training data and
33 times the test set. Scaled from the rates above, a single likelihood estimation run
over the full Quijote test set is between 3.7 and 32.7 hours on one core.

**Architecture coverage.** The zoo holds 8 entries. The framework exposes 21 verified
architecture and engine combinations across two backends, plus embedding networks and
ensembles, giving roughly 26.

**Reruns.** Four defects have been found and fixed so far, two of them inside the
evaluation metrics themselves. A fix that changes any input to a measurement
invalidates every number downstream of it, so the expensive sweeps must be repeatable.
That allowance is budgeted explicitly below rather than hoped for.

## Itemised

Cell counts are architectures x tasks x 3 seeds.

    phase                                              cells    GPU-hours
    A  summary vector zoo, CAMELS and Quijote            336          150
    B  point cloud modality, graph and set embeddings     45           25
    C  field and image modality, CNN embeddings           36           50
                                                        ----         ----
       subtotal                                          417          225
       one full rerun of phases A and B                                175
       contingency, 30 per cent                                        100
                                                                     ----
       TOTAL                                                          500

Phase A is dominated by MCMC in the likelihood and ratio estimators, which runs
independently per observation and therefore uses the node's CPU cores rather than the
GPU. This is why the request asks for 16 cores per job. On those cores phase A is
roughly 10 hours of wall clock rather than 150.

Phases B and C are genuinely GPU bound and are the reason for the allocation.

## What is not being asked for

No multi-GPU jobs, no multi-node jobs, no long-running reservations, and no
high-memory GPUs. Jobs are short, independent, and individually small, which suits a
shared scheduler. Input data is 1.5 GB and already downloaded. Results are JSON and
currently under 1 MB; the storage line is for model checkpoints.

## Scale check

The work completed so far consumed 16.6 core-hours. This request is roughly thirty
times that, and it covers a five-fold increase in architectures, a thirty-three-fold
increase in training data, three additional data modalities, and a full rerun
allowance.
