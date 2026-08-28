Subject: 2.6 update

Hi Matt,

Sorry for going quiet, and for missing last week's discussion. Classes started
and orientation landed in the same week and I lost the schedule completely. I
did keep working, so this is late rather than empty. I'll be there tomorrow.

What I did

Got ltu-ili running and checked it before trusting it. I rebuilt the Section 4.1
toy problem and wrote down what I expected first, since it has a known answer.
theta_2 only enters squared, so its sign is unrecoverable, and the model duly
failed on it at R2 -0.008. theta_0 and theta_1 came out anticorrelated at -0.76,
coverage 0.713 and 0.958 against nominal 0.68 and 0.95. Enough to trust the
wiring.

Then the main sweep. Eight architectures (MAF, NSF, MDN, a 4-MAF ensemble,
NLE-MAF, NLE-MDN, NRE-MLP, NRE-ResNet) across three CAMELS tasks, three seeds
each, at matched compute. Input is the correlation function in 25 log bins, 800
training simulations. 72 runs, none failed.

Then I checked whether my calibration metrics work at all, which I should have
done first. I built posteriors whose coverage I know analytically and asked each
metric to recover it. Two things fell out, neither flattering. I had been
misreading one of the two metrics, so those numbers were junk and are now fixed.
And at 100 evaluation points a perfectly calibrated posterior can read as low as
0.605, which means 100 points could not resolve the effect I was about to claim.
So I reran the whole sweep on all 200 test simulations. Everything below is from
that rerun.

Results

1. Accuracy doesn't separate the architectures at all. All eight land between
   0.806 and 0.870 R2 on Omega_m in CAMELS, a spread of 0.064. On an accuracy leaderboard they're
   interchangeable. That is most of the argument for the zoo carrying more than
   accuracy, and it is the clearest result I have.

2. NPE beats NLE on sigma_8 by 0.179 (0.371 +/- 0.008 against 0.193 +/- 0.052),
   at dim(x) 25 and dim(theta) 2. That's the direction Section 2.3 predicts. I'd
   expected no clear winner at that ratio, so the rule was right and I wasn't.

3. Compute varies far more than accuracy does. Counting inference as well as
   training, the MDN finishes in 0.7 seconds with 7,930 parameters while the
   likelihood estimator takes 3,571 seconds. That is 4,797 times the compute for
   an answer that differs by 0.064. Training is a few seconds for every entry;
   the whole spread is MCMC, which posterior estimators avoid and likelihood and
   ratio estimators need once per observation.

4. Not one architecture is calibrated. Coverage at the 68 per cent level averages
   0.603 against nominal 0.680, and TARP agrees at 0.617. Zero of 24
   architecture-task pairs reach nominal on either test, and zero of 24 reach it
   at the 95 per cent level either. Ensembling four MAFs moved coverage by only
   +0.010, +0.013 and +0.003 across the three tasks, against a gap of about 0.08,
   so it closes well under a fifth of it at four times the compute. I am not
   reading that as a general result, only as what happens at 800 simulations on
   these tasks.

Separately, I ran into a few small compatibility issues in ltu-ili with newer
numpy and sbi. Nothing that blocked me, and I have fixes or workarounds for all
of them. Happy to write them up if that is useful to you.

Next, and where I'm stuck

Mine to do: build the skill and run the five held-out problem descriptions. I
also found Quijote sitting unused on disk, 19,651 training simulations against
the 800 I have been using, with five parameters instead of two. It is wired in
and running it is next, since it settles question 2 below by itself.

Four things I'd like your view on:

1. How many test simulations do you consider enough for a coverage claim? CAMELS
   gives me 200 and my own check says 100 isn't enough. Since every entry has to
   be admitted on a coverage measurement, this constrains the whole zoo, not
   just my sweep.

2. Everything is at one training set size, so I can't separate "these models are
   overconfident" from "800 simulations isn't many". Worth a learning curve at
   200, 400 and 800?

3. For the skill, dense retrieval over the metadata, or few-shot with the
   evaluation summaries as context? Happy to build both and measure rather than
   guess.

4. Eight architectures is thin for a zoo. Breadth first, or get the skill
   working on what I have?

One thing I can't do yet: MCMC equivalence needs a reference posterior and
CAMELS doesn't have one. I can check the machinery on your toy problem, but I
can't make the claim on CAMELS, and I didn't want to quietly drop it from the
success criteria.

Thanks,
Danish
