#!/bin/bash
# Version-rot fixes for ltu-ili 0.1.5 under numpy>=2. Re-runnable and idempotent.
# Reported cause: scipy gaussian_kde.logpdf returns shape (1,); numpy 2 refuses to
# assign that into a scalar slot, where numpy 1 accepted it silently.
E=/Users/danishmultani/miniconda3/envs/ltuili
M=$E/lib/python3.11/site-packages/ili/validation/metrics.py
grep -q "kde.logpdf(trues\[i, :\])\[0\]" "$M" && { echo "already patched"; exit 0; }
cp -n "$M" "$M.orig"
perl -pi -e 's/logprobs\[i\] = kde\.logpdf\(trues\[i, :\]\)$/logprobs[i] = kde.logpdf(trues[i, :])[0]/' "$M"
grep -n "kde.logpdf" "$M"
